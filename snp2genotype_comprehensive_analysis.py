
"""
Comprehensive SNP2Genotype Analysis Pipeline
Analyzing SNP genotype matrix for rare disease genetic insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_snp_data(filepath):
    """Load SNP genotype matrix and convert to numeric format"""
    print("Loading SNP genotype data...")
    
    # Load data - handle both CSV and Excel formats
    try:
        # First try as CSV
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        # If UTF-8 decode fails, try as Excel file
        print("UTF-8 decode failed, attempting to read as Excel file...")
        df = pd.read_excel(filepath)
    except Exception:
        # If CSV fails for other reasons, try Excel
        print("CSV read failed, attempting to read as Excel file...")
        df = pd.read_excel(filepath)
    
    # Remove empty/unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all')  # Drop columns that are entirely empty
    
    # Set SNP identifier column as index (handle different possible column names)
    snp_id_col = None
    for candidate in ['rs_ID', 'rsid', 'rsID', 'Name']:
        if candidate in df.columns:
            snp_id_col = candidate
            break
    if snp_id_col is None:
        raise KeyError("Could not find SNP identifier column (tried 'rs_ID', 'rsid', 'rsID', 'Name')")
    df.set_index(snp_id_col, inplace=True)
    
    # Remove the 'index' column if it exists
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    
    print(f"Loaded {len(df)} SNPs across {len(df.columns)} patients")
    
    # Clean missing data markers
    df = df.replace(['--', '#N/A', 'N/A'], np.nan)
    
    # Convert genotypes to numeric scoring
    # AA/TT/GG/CC (homozygous reference) = 0
    # Het variants (AG, AT, etc.) = 1  
    # Alternative homozygous = 2
    def genotype_to_numeric(genotype):
        if pd.isna(genotype):
            return np.nan
        
        # If already numeric, return as is
        if isinstance(genotype, (int, float)):
            return genotype
        
        # Convert to string to handle any non-string types
        genotype_str = str(genotype)
        
        # Check if it's a valid 2-character genotype
        if len(genotype_str) != 2:
            return np.nan
        
        alleles = sorted(genotype_str)
        if alleles[0] == alleles[1]:  # Homozygous
            if genotype_str in ['AA', 'TT', 'GG', 'CC']:
                return 0  # Assume reference
            else:
                return 2  # Alternative homozygous
        else:  # Heterozygous
            return 1
    
    print("Converting genotypes to numeric format...")
    numeric_df = df.applymap(genotype_to_numeric)
    
    # Calculate missing data statistics
    missing_stats = {
        'total_calls': df.size,
        'missing_calls': df.isna().sum().sum(),
        'missing_percentage': (df.isna().sum().sum() / df.size) * 100
    }
    
    print(f"Missing data: {missing_stats['missing_calls']}/{missing_stats['total_calls']} ({missing_stats['missing_percentage']:.2f}%)")
    
    return df, numeric_df, missing_stats

def analyze_patient_similarity(numeric_df):
    """Calculate patient-patient similarity matrix"""
    print("\nCalculating patient similarity matrix...")
    
    # Transpose to have patients as rows
    patient_matrix = numeric_df.T
    
    # Calculate correlation matrix (Pearson correlation)
    similarity_matrix = patient_matrix.corr()
    
    # Calculate Jaccard similarity for genotype presence
    def jaccard_similarity(s1, s2):
        # Remove NaN values
        valid_mask = ~(pd.isna(s1) | pd.isna(s2))
        if valid_mask.sum() == 0:
            return 0
        
        s1_valid = s1[valid_mask]
        s2_valid = s2[valid_mask]
        
        intersection = ((s1_valid == s2_valid) & (s1_valid > 0)).sum()
        union = ((s1_valid > 0) | (s2_valid > 0)).sum()
        
        return intersection / union if union > 0 else 0
    
    # Calculate Jaccard matrix
    patients = list(patient_matrix.index)
    jaccard_matrix = pd.DataFrame(index=patients, columns=patients, dtype=float)
    
    for i, p1 in enumerate(patients):
        for j, p2 in enumerate(patients):
            if i <= j:
                jaccard_val = jaccard_similarity(patient_matrix.loc[p1], patient_matrix.loc[p2])
                jaccard_matrix.loc[p1, p2] = jaccard_val
                jaccard_matrix.loc[p2, p1] = jaccard_val
    
    return similarity_matrix, jaccard_matrix

def analyze_variant_frequencies(numeric_df):
    """Analyze variant frequencies across patients"""
    print("\nAnalyzing variant frequencies...")
    
    # Count genotype frequencies per SNP
    freq_stats = {}
    
    for snp_id in numeric_df.index:
        snp_data = numeric_df.loc[snp_id].dropna()
        if len(snp_data) == 0:
            continue
            
        counts = Counter(snp_data)
        total = len(snp_data)
        
        freq_stats[snp_id] = {
            'total_calls': total,
            'ref_homozygous': counts.get(0, 0),
            'heterozygous': counts.get(1, 0),
            'alt_homozygous': counts.get(2, 0),
            'ref_freq': counts.get(0, 0) / total,
            'het_freq': counts.get(1, 0) / total,
            'alt_freq': counts.get(2, 0) / total,
            'variant_freq': (counts.get(1, 0) + counts.get(2, 0)) / total
        }
    
    freq_df = pd.DataFrame(freq_stats).T
    
    # Identify rare variants (present in <20% of patients)
    rare_variants = freq_df[freq_df['variant_freq'] < 0.2].copy()
    rare_variants = rare_variants.sort_values('variant_freq')
    
    # Identify common variants (present in >80% of patients) 
    common_variants = freq_df[freq_df['variant_freq'] > 0.8].copy()
    
    print(f"Found {len(rare_variants)} rare variants (variant freq < 20%)")
    print(f"Found {len(common_variants)} common variants (variant freq > 80%)")
    
    return freq_df, rare_variants, common_variants

def patient_stratification_analysis(numeric_df, similarity_matrix):
    """Perform patient stratification based on genotype patterns"""
    print("\nPerforming patient stratification...")
    
    # Hierarchical clustering of patients
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    
    # Convert similarity to distance matrix
    distance_matrix = 1 - similarity_matrix.fillna(0)

    # Ensure a valid, non-negative distance matrix
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # enforce symmetry
    np.fill_diagonal(distance_matrix.values, 0)  # zero self-distance
    distance_matrix = distance_matrix.clip(lower=0)  # clip tiny negative values from numerical noise
    
    # Perform clustering
    condensed_distances = squareform(distance_matrix.values, checks=False)
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Get clusters (3 clusters as example)
    cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
    
    # Assign clusters to patients
    patients = list(similarity_matrix.index)
    patient_clusters = dict(zip(patients, cluster_labels))
    
    print("Patient cluster assignments:")
    for cluster in sorted(set(cluster_labels)):
        cluster_patients = [p for p, c in patient_clusters.items() if c == cluster]
        print(f"Cluster {cluster}: {cluster_patients}")
    
    return patient_clusters, linkage_matrix

def identify_patient_specific_variants(numeric_df, rare_variants):
    """Identify variants specific to individual patients or patient groups"""
    print("\nIdentifying patient-specific variants...")
    
    patient_variants = {}
    
    for patient in numeric_df.columns:
        patient_data = numeric_df[patient].dropna()
        
        # Find variants present in this patient
        patient_variant_snps = patient_data[patient_data > 0].index.tolist()
        
        # Count how many other patients have each variant
        variant_counts = {}
        for snp in patient_variant_snps:
            other_patients_with_variant = (numeric_df.loc[snp] > 0).sum() - (numeric_df.loc[snp, patient] > 0)
            variant_counts[snp] = other_patients_with_variant
        
        # Find unique or rare variants for this patient
        unique_variants = [snp for snp, count in variant_counts.items() if count == 0]
        rare_shared_variants = [snp for snp, count in variant_counts.items() if 0 < count <= 2]
        
        patient_variants[patient] = {
            'total_variants': len(patient_variant_snps),
            'unique_variants': unique_variants,
            'rare_shared_variants': rare_shared_variants,
            'unique_count': len(unique_variants),
            'rare_shared_count': len(rare_shared_variants)
        }
        
        print(f"{patient}: {len(patient_variant_snps)} total variants, {len(unique_variants)} unique, {len(rare_shared_variants)} rare shared")
    
    return patient_variants

def clinical_interpretation(patient_variants, rare_variants, patient_clusters):
    """Generate clinical interpretation and recommendations"""
    print("\nGenerating clinical interpretation...")
    
    # Prioritize patients based on unique variant burden
    patient_priorities = []
    for patient, data in patient_variants.items():
        priority_score = data['unique_count'] * 3 + data['rare_shared_count']
        patient_priorities.append((patient, priority_score, data['unique_count'], data['rare_shared_count']))
    
    patient_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Generate clinical recommendations
    recommendations = {
        'high_priority_patients': patient_priorities[:5],
        'cluster_analysis': patient_clusters,
        'rare_variant_count': len(rare_variants),
        'clinical_actions': []
    }
    
    # Add specific recommendations
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

def create_visualizations(similarity_matrix, jaccard_matrix, freq_df, patient_clusters):
    """Create comprehensive visualizations"""
    print("\nCreating visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('SNP2Genotype Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Patient similarity heatmap (Pearson correlation)
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                ax=axes[0,0], cbar_kws={'label': 'Correlation'})
    axes[0,0].set_title('Patient Similarity (Pearson Correlation)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].tick_params(axis='y', rotation=0)
    
    # 2. Patient similarity heatmap (Jaccard similarity)
    sns.heatmap(jaccard_matrix.astype(float), annot=True, fmt='.2f', cmap='viridis', 
                ax=axes[0,1], cbar_kws={'label': 'Jaccard Similarity'})
    axes[0,1].set_title('Patient Similarity (Jaccard Index)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].tick_params(axis='y', rotation=0)
    
    # 3. Variant frequency distribution
    axes[0,2].hist(freq_df['variant_freq'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,2].axvline(0.2, color='red', linestyle='--', label='Rare threshold (20%)')
    axes[0,2].axvline(0.8, color='green', linestyle='--', label='Common threshold (80%)')
    axes[0,2].set_xlabel('Variant Frequency')
    axes[0,2].set_ylabel('Number of SNPs')
    axes[0,2].set_title('Variant Frequency Distribution')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Genotype distribution per patient
    patients = list(similarity_matrix.index)
    genotype_counts = {patient: {'Ref': 0, 'Het': 0, 'Alt': 0} for patient in patients}
    
    for patient in patients:
        patient_data = freq_df.apply(lambda row: {
            'Ref': row[f'ref_homozygous'] if patient in similarity_matrix.index else 0,
            'Het': row[f'heterozygous'] if patient in similarity_matrix.index else 0,
            'Alt': row[f'alt_homozygous'] if patient in similarity_matrix.index else 0
        }, axis=1)
    
    # Simplified genotype counts per patient
    from collections import defaultdict
    patient_genotype_counts = defaultdict(lambda: {'Ref': 0, 'Het': 0, 'Alt': 0})
    
    for patient in patients:
        for snp in freq_df.index:
            if snp in similarity_matrix.columns:  # Ensure SNP exists
                ref_count = freq_df.loc[snp, 'ref_homozygous']
                het_count = freq_df.loc[snp, 'heterozygous'] 
                alt_count = freq_df.loc[snp, 'alt_homozygous']
                
                # Approximate distribution per patient
                patient_genotype_counts[patient]['Ref'] += ref_count / len(patients)
                patient_genotype_counts[patient]['Het'] += het_count / len(patients)
                patient_genotype_counts[patient]['Alt'] += alt_count / len(patients)
    
    ref_counts = [patient_genotype_counts[p]['Ref'] for p in patients]
    het_counts = [patient_genotype_counts[p]['Het'] for p in patients]
    alt_counts = [patient_genotype_counts[p]['Alt'] for p in patients]
    
    x_pos = np.arange(len(patients))
    axes[1,0].bar(x_pos, ref_counts, label='Reference Homozygous', alpha=0.7)
    axes[1,0].bar(x_pos, het_counts, bottom=ref_counts, label='Heterozygous', alpha=0.7)
    axes[1,0].bar(x_pos, alt_counts, bottom=np.array(ref_counts) + np.array(het_counts), 
                  label='Alternative Homozygous', alpha=0.7)
    axes[1,0].set_xlabel('Patients')
    axes[1,0].set_ylabel('Average Genotype Counts')
    axes[1,0].set_title('Genotype Distribution per Patient')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(patients, rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Patient clustering visualization
    cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, patient in enumerate(patients):
        cluster = patient_clusters.get(patient, 1)
        color = cluster_colors[(cluster-1) % len(cluster_colors)]
        axes[1,1].scatter(i, cluster, c=color, s=100, alpha=0.7, label=f'Cluster {cluster}' if i == 0 or patient_clusters.get(patients[i-1], 1) != cluster else '')
        axes[1,1].text(i, cluster + 0.1, patient, rotation=45, ha='left', va='bottom', fontsize=8)
    
    axes[1,1].set_xlabel('Patient Index')
    axes[1,1].set_ylabel('Cluster Assignment')
    axes[1,1].set_title('Patient Cluster Assignments')
    axes[1,1].set_xticks(range(len(patients)))
    axes[1,1].set_xticklabels([f'P{i+1}' for i in range(len(patients))], rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = axes[1,1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1,1].legend(by_label.values(), by_label.keys())
    
    # 6. Rare variant analysis
    rare_snps = freq_df[freq_df['variant_freq'] < 0.2].copy()
    if len(rare_snps) > 0:
        axes[1,2].hist(rare_snps['variant_freq'], bins=20, alpha=0.7, color='coral', edgecolor='black')
        axes[1,2].set_xlabel('Variant Frequency')
        axes[1,2].set_ylabel('Number of Rare SNPs')
        axes[1,2].set_title(f'Rare Variant Distribution (n={len(rare_snps)})')
        axes[1,2].grid(True, alpha=0.3)
    else:
        axes[1,2].text(0.5, 0.5, 'No rare variants found\n(frequency < 20%)', 
                      ha='center', va='center', transform=axes[1,2].transAxes,
                      fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1,2].set_title('Rare Variant Analysis')
    
    plt.tight_layout()
    plt.savefig('snp2genotype_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'snp2genotype_comprehensive_analysis.png'")
    plt.close(fig)
    
    return fig

def generate_comprehensive_report(missing_stats, freq_df, rare_variants, common_variants, 
                                patient_variants, recommendations, patient_clusters):
    """Generate comprehensive clinical report"""
    
    report = f"""
SNP2GENOTYPE COMPREHENSIVE ANALYSIS REPORT
==========================================
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
================
Total SNPs analyzed: {len(freq_df)}
Total patients: {len(list(patient_variants.keys()))}
Missing data: {missing_stats['missing_calls']:,} / {missing_stats['total_calls']:,} ({missing_stats['missing_percentage']:.2f}%)

VARIANT FREQUENCY ANALYSIS
==========================
Rare variants (freq < 20%): {len(rare_variants)} ({len(rare_variants)/len(freq_df)*100:.1f}%)
Common variants (freq > 80%): {len(common_variants)} ({len(common_variants)/len(freq_df)*100:.1f}%)
Intermediate frequency variants: {len(freq_df) - len(rare_variants) - len(common_variants)}

TOP 10 RAREST VARIANTS
======================
"""
    
    if len(rare_variants) > 0:
        top_rare = rare_variants.head(10)
        for snp_id, row in top_rare.iterrows():
            report += f"{snp_id}: {row['variant_freq']:.1%} frequency ({row['heterozygous']} het, {row['alt_homozygous']} hom alt)\n"
    else:
        report += "No rare variants found with frequency < 20%\n"
    
    report += f"""

PATIENT STRATIFICATION
======================
Cluster assignments:
"""
    
    for cluster in sorted(set(patient_clusters.values())):
        cluster_patients = [p for p, c in patient_clusters.items() if c == cluster]
        report += f"Cluster {cluster}: {', '.join(cluster_patients)}\n"
    
    report += f"""

PATIENT-SPECIFIC VARIANT ANALYSIS
==================================
"""
    
    for patient, data in patient_variants.items():
        report += f"{patient}:\n"
        report += f"  - Total variants: {data['total_variants']}\n"
        report += f"  - Unique variants: {data['unique_count']}\n"
        report += f"  - Rare shared variants: {data['rare_shared_count']}\n"
        if data['unique_count'] > 0:
            report += f"  - Top unique variants: {', '.join(data['unique_variants'][:5])}\n"
        report += "\n"
    
    report += f"""
CLINICAL RECOMMENDATIONS
========================
High Priority Patients for Follow-up:
"""
    
    for patient, score, unique, rare_shared in recommendations['high_priority_patients']:
        report += f"{patient}: Priority Score {score} ({unique} unique, {rare_shared} rare shared variants)\n"
    
    report += f"""
Specific Clinical Actions:
"""
    
    if recommendations['clinical_actions']:
        for action in recommendations['clinical_actions']:
            report += f"- {action}\n"
    else:
        report += "- Standard genetic counseling recommended for all patients\n"
        report += "- Consider population-specific variant interpretation\n"
        report += "- Follow-up with functional validation of high-frequency variants\n"
    
    report += f"""

SUMMARY AND NEXT STEPS
======================
1. Patient stratification identified {len(set(patient_clusters.values()))} distinct genetic clusters
2. {len([p for p, data in patient_variants.items() if data['unique_count'] > 5])} patients have >5 unique variants requiring investigation  
3. {len(rare_variants)} rare variants identified for functional prioritization
4. Recommend integration with clinical phenotype data for genotype-phenotype correlation
5. Consider copy number variant analysis for patients with high unique variant burden

Analysis completed successfully.
"""
    
    # Save report
    with open('snp2genotype_comprehensive_report.txt', 'w') as f:
        f.write(report)
    
    print("Comprehensive report saved as 'snp2genotype_comprehensive_report.txt'")
    
    return report

def main():
    """Main analysis pipeline"""
    print("=== SNP2GENOTYPE COMPREHENSIVE ANALYSIS PIPELINE ===")
    print("Analyzing gold-standard SNP genotype matrix...")
    
    try:
        # 1. Load and process data
        original_df, numeric_df, missing_stats = load_and_clean_snp_data('gene_patient_matrix2.xlsx')
        
        # 2. Patient similarity analysis
        similarity_matrix, jaccard_matrix = analyze_patient_similarity(numeric_df)
        
        # 3. Variant frequency analysis
        freq_df, rare_variants, common_variants = analyze_variant_frequencies(numeric_df)
        
        # 4. Patient stratification
        patient_clusters, linkage_matrix = patient_stratification_analysis(numeric_df, similarity_matrix)
        
        # 5. Patient-specific variant identification
        patient_variants = identify_patient_specific_variants(numeric_df, rare_variants)
        
        # 6. Clinical interpretation
        recommendations = clinical_interpretation(patient_variants, rare_variants, patient_clusters)
        
        # 7. Create visualizations
        fig = create_visualizations(similarity_matrix, jaccard_matrix, freq_df, patient_clusters)
        
        # 8. Generate comprehensive report
        report = generate_comprehensive_report(missing_stats, freq_df, rare_variants, common_variants,
                                             patient_variants, recommendations, patient_clusters)
        
        print("\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        print("Generated files:")
        print("- snp2genotype_comprehensive_analysis.png (visualizations)")
        print("- snp2genotype_comprehensive_report.txt (detailed report)")
        print("\nKey findings:")
        print(f"- {len(freq_df)} SNPs analyzed across {len(list(patient_variants.keys()))} patients")
        print(f"- {len(rare_variants)} rare variants identified")
        print(f"- {len(set(patient_clusters.values()))} patient clusters detected")
        print(f"- {len([p for p, data in patient_variants.items() if data['unique_count'] > 5])} high-priority patients identified")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()