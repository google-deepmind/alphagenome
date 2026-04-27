import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed
import warnings
import argparse
warnings.filterwarnings('ignore')

# Variant frequency thresholds
# Rare variants have frequency below RARE_VARIANT_THRESHOLD
# Common variants have frequency above COMMON_VARIANT_THRESHOLD
RARE_VARIANT_THRESHOLD = 0.10  # 10%
COMMON_VARIANT_THRESHOLD = 0.80  # 80%

def load_and_clean_snp_data(filepath):
    """Load SNP genotype matrix and convert to numeric format - Optimized"""
    print("Loading SNP genotype data...")

    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        print("UTF-8 decode failed, attempting to read as Excel file...")
        df = pd.read_excel(filepath)
    except Exception:
        print("CSV read failed, attempting to read as Excel file...")
        df = pd.read_excel(filepath)

    # Remove empty/unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all')

    # Set SNP identifier column as index
    snp_id_col = next((col for col in ['rs_ID', 'rsid', 'rsID', 'Name'] if col in df.columns), None)
    if snp_id_col is None:
        raise KeyError("Could not find SNP identifier column (tried 'rs_ID', 'rsid', 'rsID', 'Name')")
    df.set_index(snp_id_col, inplace=True)

    if 'index' in df.columns:
        df = df.drop('index', axis=1)

    print(f"Loaded {len(df)} SNPs across {len(df.columns)} patients")

    # Replace missing data markers with np.nan
    df = df.replace(['--', '#N/A', 'N/A'], np.nan)

    # Optimized genotype to numeric conversion using vectorized operations
    def genotype_to_numeric(series):
        def map_genotype(g):
            if pd.isna(g) or not isinstance(g, str) or len(g) != 2:
                return np.nan
            alleles = sorted(g)
            if alleles[0] == alleles[1]:  # Homozygous
                return 0 if g in ['AA', 'TT', 'GG', 'CC'] else 2
            return 1  # Heterozygous
        return series.apply(map_genotype)

    print("Converting genotypes to numeric format...")
    numeric_df = df.apply(genotype_to_numeric, axis=0).astype('float32')  # Use float32 for memory efficiency

    # Calculate missing data stats
    missing_stats = {
        'total_calls': df.size,
        'missing_calls': df.isna().sum().sum(),
        'missing_percentage': (df.isna().sum().sum() / df.size) * 100
    }
    print(f"Missing data: {missing_stats['missing_calls']}/{missing_stats['total_calls']} ({missing_stats['missing_percentage']:.2f}%)")

    return df, numeric_df, missing_stats

def analyze_patient_similarity(numeric_df):
    """Calculate patient-patient similarity matrix - Optimized with parallel processing"""
    print("\nCalculating patient similarity matrix...")

    patient_matrix = numeric_df.T.astype('float32')  # Transpose and reduce memory footprint

    # Pearson correlation using numpy for speed
    similarity_matrix = np.corrcoef(patient_matrix.fillna(0).values)

    # Convert to DataFrame
    similarity_matrix = pd.DataFrame(similarity_matrix, index=patient_matrix.index, columns=patient_matrix.index)

    # Optimized Jaccard similarity with parallel processing
    def jaccard_similarity_pair(i, j, patient_matrix_values):
        s1, s2 = patient_matrix_values[i], patient_matrix_values[j]
        valid_mask = ~(np.isnan(s1) | np.isnan(s2))
        if not valid_mask.sum():
            return 0
        s1_valid, s2_valid = s1[valid_mask], s2[valid_mask]
        intersection = np.sum((s1_valid == s2_valid) & (s1_valid > 0))
        union = np.sum((s1_valid > 0) | (s2_valid > 0))
        return intersection / union if union > 0 else 0

    patients = patient_matrix.index
    n_patients = len(patients)
    patient_matrix_values = patient_matrix.values
    jaccard_matrix = np.zeros((n_patients, n_patients))

    # Parallel computation of Jaccard similarity
    results = Parallel(n_jobs=-1)(
        delayed(jaccard_similarity_pair)(i, j, patient_matrix_values)
        for i in range(n_patients) for j in range(i, n_patients)
    )

    k = 0
    for i in range(n_patients):
        for j in range(i, n_patients):
            jaccard_matrix[i, j] = results[k]
            jaccard_matrix[j, i] = results[k]
            k += 1

    jaccard_matrix = pd.DataFrame(jaccard_matrix, index=patients, columns=patients)
    return similarity_matrix, jaccard_matrix

def analyze_variant_frequencies(numeric_df):
    """Analyze variant frequencies across patients - Optimized with numpy"""
    print("\nAnalyzing variant frequencies...")

    data_array = numeric_df.values
    total_calls = np.sum(~np.isnan(data_array), axis=1)
    ref_homo = np.sum(data_array == 0, axis=1)
    het = np.sum(data_array == 1, axis=1)
    alt_homo = np.sum(data_array == 2, axis=1)

    freq_stats = {
        'total_calls': total_calls,
        'ref_homozygous': ref_homo,
        'heterozygous': het,
        'alt_homozygous': alt_homo,
        'ref_freq': np.divide(ref_homo, total_calls, out=np.zeros_like(ref_homo, dtype=float), where=total_calls != 0),
        'het_freq': np.divide(het, total_calls, out=np.zeros_like(het, dtype=float), where=total_calls != 0),
        'alt_freq': np.divide(alt_homo, total_calls, out=np.zeros_like(alt_homo, dtype=float), where=total_calls != 0),
        'variant_freq': np.divide(het + alt_homo, total_calls, out=np.zeros_like(het, dtype=float), where=total_calls != 0)
    }

    freq_df = pd.DataFrame(freq_stats, index=numeric_df.index)
    rare_variants = freq_df[freq_df['variant_freq'] < RARE_VARIANT_THRESHOLD].sort_values('variant_freq')
    common_variants = freq_df[freq_df['variant_freq'] > COMMON_VARIANT_THRESHOLD]

    print(
        f"Found {len(rare_variants)} rare variants (variant freq < {RARE_VARIANT_THRESHOLD * 100:02.0f}%)"
    )
    print(
        f"Found {len(common_variants)} common variants (variant freq > {COMMON_VARIANT_THRESHOLD * 100:02.0f}%)"
    )
    return freq_df, rare_variants, common_variants

def patient_stratification_analysis(numeric_df, similarity_matrix):
    """Perform patient stratification based on genotype patterns - Optimized"""
    print("\nPerforming patient stratification...")

    distance_matrix = 1 - similarity_matrix.fillna(0).values
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, None)

    condensed_distances = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_distances, method='ward')
    cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')

    patients = list(similarity_matrix.index)
    patient_clusters = dict(zip(patients, cluster_labels))

    print("Patient cluster assignments:")
    for cluster in sorted(set(cluster_labels)):
        cluster_patients = [p for p, c in patient_clusters.items() if c == cluster]
        print(f"Cluster {cluster}: {cluster_patients}")

    return patient_clusters, linkage_matrix


def compute_patient_pca(numeric_df, n_components=2):
    """Compute PCA on patient genotype matrix."""
    print("\nComputing patient PCA...")

    # patients x SNPs
    X = numeric_df.T.fillna(0).values.astype("float32")
    # Center SNPs
    X = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :n_components] * S[:n_components]

    # Explained variance ratio for each component
    n_samples = X.shape[0]
    eigenvalues = (S ** 2) / max(n_samples - 1, 1)
    explained_var_ratio = eigenvalues / eigenvalues.sum()

    patients = numeric_df.columns
    pca_df = pd.DataFrame(
        coords,
        index=patients,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return pca_df, explained_var_ratio[:n_components]

def identify_patient_specific_variants(numeric_df, rare_variants):
    """Identify variants specific to individual patients or groups - Optimized"""
    print("\nIdentifying patient-specific variants...")

    patient_variants = {}
    data_array = numeric_df.values
    n_snps = data_array.shape[0]

    for idx, patient in enumerate(numeric_df.columns):
        patient_data = data_array[:, idx]
        variant_mask = (patient_data > 0) & (~np.isnan(patient_data))
        patient_variant_indices = np.where(variant_mask)[0]

        other_counts = np.sum((data_array > 0) & (~np.isnan(data_array)), axis=1) - (variant_mask.astype(int))
        unique_mask = other_counts[patient_variant_indices] == 0
        rare_shared_mask = (other_counts[patient_variant_indices] > 0) & (other_counts[patient_variant_indices] <= 2)

        unique_variants = numeric_df.index[patient_variant_indices[unique_mask]].tolist()
        rare_shared_variants = numeric_df.index[patient_variant_indices[rare_shared_mask]].tolist()

        patient_variants[patient] = {
            'total_variants': len(patient_variant_indices),
            'unique_variants': unique_variants,
            'rare_shared_variants': rare_shared_variants,
            'unique_count': len(unique_variants),
            'rare_shared_count': len(rare_shared_variants)
        }
        print(f"{patient}: {len(patient_variant_indices)} total variants, {len(unique_variants)} unique, {len(rare_shared_variants)} rare shared")

    return patient_variants


def load_snp_annotations(filepath):
    """Load SNP-to-gene annotation table, indexed by rs_ID."""
    print(f"\nLoading SNP annotation file from '{filepath}'...")
    annot = pd.read_csv(filepath, sep=None, engine='python')

    snp_candidates = ['rs_ID', 'rsid', 'rsID', 'snp_id', 'variant_id', 'SNP']
    snp_col = next((col for col in snp_candidates if col in annot.columns), None)
    if snp_col is None:
        raise KeyError(
            "Could not find SNP ID column in annotation file (tried: "
            + ", ".join(snp_candidates)
            + ")."
        )

    gene_col = next(
        (col for col in ['gene', 'Gene', 'gene_symbol', 'GENE_SYMBOL', 'symbol'] if col in annot.columns),
        None,
    )
    if gene_col is None:
        raise KeyError(
            "Could not find gene column in annotation file (tried 'gene', 'Gene', 'gene_symbol', 'GENE_SYMBOL', 'symbol')."
        )

    annot = annot[[snp_col, gene_col]].dropna()
    annot = annot.rename(columns={snp_col: 'rs_ID', gene_col: 'gene'})
    annot = annot.drop_duplicates(subset=['rs_ID'])
    return annot.set_index('rs_ID')


def export_patient_specific_variants(
    patient_variants,
    filepath="snp2genotype_patient_variants.tsv",
    snp_annotations=None,
):
    """Export per-patient unique and rare-shared variants to a TSV."""
    rows = []
    for patient, data in patient_variants.items():
        for snp in data.get('unique_variants', []):
            rows.append({'patient': patient, 'snp_id': snp, 'category': 'unique'})
        for snp in data.get('rare_shared_variants', []):
            rows.append({'patient': patient, 'snp_id': snp, 'category': 'rare_shared'})

    if not rows:
        print("No patient-specific variants to export.")
        return

    df = pd.DataFrame(rows)
    if snp_annotations is not None:
        df = df.merge(
            snp_annotations.reset_index(),
            how='left',
            left_on='snp_id',
            right_on='rs_ID',
        )
        df = df.drop(columns=['rs_ID'])

    df.to_csv(filepath, sep='\t', index=False)
    print(f"Patient-specific variants saved to '{filepath}'")


def clinical_interpretation(patient_variants, rare_variants, patient_clusters):
    """Generate clinical interpretation and recommendations - Optimized"""
    print("\nGenerating clinical interpretation...")

    patient_priorities = [
        (patient, data['unique_count'] * 3 + data['rare_shared_count'], data['unique_count'], data['rare_shared_count'])
        for patient, data in patient_variants.items()
    ]
    patient_priorities.sort(key=lambda x: x[1], reverse=True)

    recommendations = {
        'high_priority_patients': patient_priorities[:5],
        'cluster_analysis': patient_clusters,
        'rare_variant_count': len(rare_variants),
        'clinical_actions': []
    }

    for patient, score, unique, rare_shared in patient_priorities[:3]:
        if unique > 10:
            recommendations['clinical_actions'].append(
                f"{patient}: High unique variant burden ({unique} unique variants) - recommend whole genome sequencing and copy number variant analysis"
            )
        elif unique > 5:
            recommendations['clinical_actions'].append(
                f"{patient}: Moderate unique variant burden ({unique} unique variants) - recommend targeted gene panel expansion"
            )
        elif rare_shared > 10:
            recommendations['clinical_actions'].append(
                f"{patient}: High rare shared variant burden ({rare_shared} rare variants) - investigate family history and population stratification"
            )

    return recommendations

def create_visualizations(similarity_matrix, jaccard_matrix, freq_df, patient_clusters, pca_df, pca_var_ratio):
    """Create comprehensive visualizations - Optimized"""
    print("\nCreating visualizations...")
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SNP2Genotype Analysis', fontsize=14)

    # Patient similarity heatmap (Pearson correlation)
    sns.heatmap(
        similarity_matrix,
        annot=False,
        cmap='RdYlBu_r',
        ax=axes[0, 0],
        cbar_kws={'label': 'Correlation'},
    )
    axes[0, 0].set_title('Patient Similarity (Correlation)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Patient similarity heatmap (Jaccard similarity)
    sns.heatmap(
        jaccard_matrix,
        annot=False,
        cmap='viridis',
        ax=axes[0, 1],
        cbar_kws={'label': 'Jaccard Similarity'},
    )
    axes[0, 1].set_title('Patient Similarity (Jaccard)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Variant frequency distribution
    axes[0, 2].hist(
        freq_df['variant_freq'],
        bins=20,
        alpha=0.6,
        color='skyblue',
        edgecolor='black',
    )
    axes[0, 2].axvline(
        RARE_VARIANT_THRESHOLD,
        color='red',
        linestyle='--',
        label=f"Rare ({RARE_VARIANT_THRESHOLD * 100:02.0f}%)",
    )
    axes[0, 2].axvline(
        COMMON_VARIANT_THRESHOLD,
        color='green',
        linestyle='--',
        label=f"Common ({COMMON_VARIANT_THRESHOLD * 100:02.0f}%)",
    )
    axes[0, 2].set_title('Variant Frequency')
    axes[0, 2].legend()

    # Genotype distribution per patient (simplified)
    patients = list(similarity_matrix.index)
    ref_counts = freq_df['ref_homozygous'].mean()
    het_counts = freq_df['heterozygous'].mean()
    alt_counts = freq_df['alt_homozygous'].mean()

    x_pos = np.arange(len(patients))
    axes[1, 0].bar(x_pos, ref_counts, label='Ref', alpha=0.6)
    axes[1, 0].bar(x_pos, het_counts, bottom=ref_counts, label='Het', alpha=0.6)
    axes[1, 0].bar(
        x_pos,
        alt_counts,
        bottom=ref_counts + het_counts,
        label='Alt',
        alpha=0.6,
    )
    axes[1, 0].set_title('Avg Genotype/Patient')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(patients, rotation=45)
    axes[1, 0].legend()

    # Patient clustering visualization (PCA)
    cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']
    for cluster in sorted(set(patient_clusters.values())):
        cluster_patients = [
            p for p, c in patient_clusters.items() if c == cluster
        ]
        color = cluster_colors[(cluster - 1) % len(cluster_colors)]
        axes[1, 1].scatter(
            pca_df.loc[cluster_patients, "PC1"],
            pca_df.loc[cluster_patients, "PC2"],
            c=color,
            s=50,
            alpha=0.7,
            label=f"Cluster {cluster}",
        )

    pc1_var = pca_var_ratio[0] * 100 if len(pca_var_ratio) > 0 else 0
    pc2_var = pca_var_ratio[1] * 100 if len(pca_var_ratio) > 1 else 0
    axes[1, 1].set_title('Patient Clusters (PCA)')
    axes[1, 1].set_xlabel(f'PC1 ({pc1_var:.1f}%)')
    axes[1, 1].set_ylabel(f'PC2 ({pc2_var:.1f}%)')
    axes[1, 1].legend()

    # Rare variant analysis
    rare_snps = freq_df[freq_df['variant_freq'] < RARE_VARIANT_THRESHOLD]
    if len(rare_snps) > 0:
        axes[1, 2].hist(
            rare_snps['variant_freq'],
            bins=15,
            alpha=0.6,
            color='coral',
            edgecolor='black',
        )
        axes[1, 2].set_title(f'Rare Variants (n={len(rare_snps)})')
    else:
        axes[1, 2].text(
            0.5,
            0.5,
            'No rare variants',
            ha='center',
            va='center',
            transform=axes[1, 2].transAxes,
        )

    plt.tight_layout()
    plt.savefig('snp2genotype_analysis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Visualizations saved as 'snp2genotype_analysis.png'")
    return fig

def generate_comprehensive_report(missing_stats, freq_df, rare_variants, common_variants, 
                                patient_variants, recommendations, patient_clusters):
    """Generate comprehensive clinical report - Optimized for brevity"""
    report = f"""
SNP2GENOTYPE ANALYSIS REPORT
============================
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
Total SNPs: {len(freq_df)}
Total Patients: {len(list(patient_variants.keys()))}
Missing Data: {missing_stats['missing_percentage']:.2f}%

VARIANT ANALYSIS
Rare Variants (<{RARE_VARIANT_THRESHOLD * 100:02.0f}%): {len(rare_variants)}
Common Variants (>{COMMON_VARIANT_THRESHOLD * 100:02.0f}%): {len(common_variants)}

CLINICAL RECOMMENDATIONS
High Priority Patients:
"""
    for patient, score, unique, rare_shared in recommendations['high_priority_patients']:
        report += f"{patient}: Score {score} ({unique} unique, {rare_shared} rare shared)\n"

    if len(common_variants) > 0:
        max_snps_to_show = 20
        common_ids = list(common_variants.index[:max_snps_to_show])
        report += (
            f"\nCommon Variant SNP IDs (top {len(common_ids)}):\n"
            + ", ".join(common_ids)
            + "\n"
        )

    with open('snp2genotype_report.txt', 'w') as f:
        f.write(report)
    print("Report saved as 'snp2genotype_report.txt'")
    return report

def main():
    """Main analysis pipeline - Optimized"""

    global RARE_VARIANT_THRESHOLD, COMMON_VARIANT_THRESHOLD

    parser = argparse.ArgumentParser(
        description="SNP2Genotype comprehensive analysis."
    )
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=RARE_VARIANT_THRESHOLD,
        help=(
            "Rare variant frequency cutoff "
            f"(default: {RARE_VARIANT_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--common-threshold",
        type=float,
        default=COMMON_VARIANT_THRESHOLD,
        help=(
            "Common variant frequency cutoff "
            f"(default: {COMMON_VARIANT_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--snp-annotation",
        type=str,
        default=None,
        help=(
            "Optional path to SNP annotation file (CSV/TSV) with rs_ID and gene columns "
            "to annotate exported TSVs."
        ),
    )
    args = parser.parse_args()

    RARE_VARIANT_THRESHOLD = args.rare_threshold
    COMMON_VARIANT_THRESHOLD = args.common_threshold

    snp_annotations = None
    if args.snp_annotation:
        try:
            snp_annotations = load_snp_annotations(args.snp_annotation)
        except Exception as exc:
            print(
                f"Warning: could not load SNP annotation file '{args.snp_annotation}': {exc}"
            )
            snp_annotations = None

    print("=== SNP2GENOTYPE ANALYSIS PIPELINE ===")
    try:
        original_df, numeric_df, missing_stats = load_and_clean_snp_data(
            "snp2genotype_matrix_fixed.csv"
        )
        similarity_matrix, jaccard_matrix = analyze_patient_similarity(
            numeric_df
        )
        freq_df, rare_variants, common_variants = analyze_variant_frequencies(
            numeric_df
        )

        if not common_variants.empty:
            common_out = common_variants
            if snp_annotations is not None:
                common_out = common_out.join(snp_annotations, how='left')
            common_out.to_csv(
                "snp2genotype_common_variants.tsv", sep="\t"
            )
        if not rare_variants.empty:
            rare_out = rare_variants
            if snp_annotations is not None:
                rare_out = rare_out.join(snp_annotations, how='left')
            rare_out.to_csv(
                "snp2genotype_rare_variants.tsv", sep="\t"
            )

        patient_clusters, linkage_matrix = patient_stratification_analysis(
            numeric_df, similarity_matrix
        )
        pca_df, pca_var_ratio = compute_patient_pca(numeric_df)
        pca_export = pca_df.copy()
        pca_export['cluster'] = [
            patient_clusters.get(p) for p in pca_export.index
        ]
        pca_export.to_csv(
            "snp2genotype_patient_pca.tsv", sep="\t"
        )
        print(
            "Patient PCA coordinates saved to 'snp2genotype_patient_pca.tsv'"
        )

        patient_variants = identify_patient_specific_variants(
            numeric_df, rare_variants
        )
        export_patient_specific_variants(
            patient_variants,
            snp_annotations=snp_annotations,
        )
        recommendations = clinical_interpretation(
            patient_variants, rare_variants, patient_clusters
        )
        create_visualizations(
            similarity_matrix,
            jaccard_matrix,
            freq_df,
            patient_clusters,
            pca_df,
            pca_var_ratio,
        )
        generate_comprehensive_report(
            missing_stats,
            freq_df,
            rare_variants,
            common_variants,
            patient_variants,
            recommendations,
            patient_clusters,
        )
        print("\n=== ANALYSIS COMPLETED ===")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
