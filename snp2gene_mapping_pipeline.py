#!/usr/bin/env python3
"""
SNP to Gene Mapping Pipeline
Maps SNPs to their associated genes using multiple data sources
Integrates with gpt_snp2gene_pipeline.py for comprehensive genetic analysis
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import concurrent.futures
from threading import Lock
import xml.etree.ElementTree as ET

warnings.filterwarnings('ignore')

# Thread-safe print lock for concurrent operations
print_lock = Lock()

def safe_print(message):
    """Thread-safe print function"""
    with print_lock:
        print(message)

class SNPToGeneMapper:
    """Comprehensive SNP to Gene mapping with multiple data sources"""
    
    def __init__(self):
        self.snp_gene_cache = {}
        self.failed_snps = set()
        self.rate_limit_delay = 0.5  # seconds between API calls
        
    def query_ncbi_eutils(self, snp_id: str) -> Optional[Dict]:
        """Query NCBI eUtils API for SNP information"""
        try:
            # Remove 'rs' prefix for API call
            snp_numeric = snp_id.replace('rs', '') if snp_id.startswith('rs') else snp_id
            
            # First get SNP summary
            summary_url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
            params = {
                'db': 'snp',
                'id': snp_numeric,
                'retmode': 'json',
                'rettype': 'docsum'
            }
            
            response = requests.get(summary_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract gene information
            if 'result' in data and snp_numeric in data['result']:
                snp_data = data['result'][snp_numeric]
                
                gene_info = {
                    'snp_id': snp_id,
                    'chromosome': snp_data.get('chrpos', '').split(':')[0] if ':' in str(snp_data.get('chrpos', '')) else '',
                    'position': snp_data.get('chrpos', '').split(':')[1] if ':' in str(snp_data.get('chrpos', '')) else '',
                    'genes': [],
                    'functional_class': snp_data.get('fxn_class', ''),
                    'alleles': snp_data.get('alleles', ''),
                    'maf': snp_data.get('gmaf', ''),
                    'source': 'NCBI_eUtils'
                }
                
                # Extract gene names from docsum
                if 'genes' in snp_data and snp_data['genes']:
                    for gene in snp_data['genes']:
                        if isinstance(gene, dict) and 'name' in gene:
                            gene_info['genes'].append({
                                'symbol': gene['name'],
                                'gene_id': gene.get('geneid', ''),
                                'location': gene.get('locus', '')
                            })
                
                return gene_info
                
        except Exception as e:
            safe_print(f"NCBI eUtils error for {snp_id}: {str(e)}")
            return None
            
        return None
    
    def query_ensembl_rest(self, snp_id: str) -> Optional[Dict]:
        """Query Ensembl REST API for SNP information"""
        try:
            # Ensembl REST API endpoint
            url = f'https://rest.ensembl.org/variation/human/{snp_id}'
            headers = {'Content-Type': 'application/json'}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            gene_info = {
                'snp_id': snp_id,
                'chromosome': data.get('mappings', [{}])[0].get('seq_region_name', '') if data.get('mappings') else '',
                'position': str(data.get('mappings', [{}])[0].get('start', '')) if data.get('mappings') else '',
                'genes': [],
                'consequence_types': [],
                'source': 'Ensembl_REST'
            }
            
            # Extract consequence type information which includes gene data
            if 'most_severe_consequence' in data:
                gene_info['most_severe_consequence'] = data['most_severe_consequence']
            
            # Extract transcript consequences for gene mapping
            if 'transcript_consequences' in data:
                gene_symbols = set()
                for tc in data['transcript_consequences']:
                    if 'gene_symbol' in tc:
                        gene_symbols.add(tc['gene_symbol'])
                        gene_info['genes'].append({
                            'symbol': tc['gene_symbol'],
                            'gene_id': tc.get('gene_id', ''),
                            'transcript_id': tc.get('transcript_id', ''),
                            'consequence_terms': tc.get('consequence_terms', [])
                        })
            
            return gene_info if gene_info['genes'] else None
            
        except Exception as e:
            safe_print(f"Ensembl REST error for {snp_id}: {str(e)}")
            return None
    
    def create_fallback_mapping(self, snp_id: str) -> Dict:
        """Create fallback mapping for SNPs without gene associations"""
        return {
            'snp_id': snp_id,
            'chromosome': 'Unknown',
            'position': 'Unknown',
            'genes': [{'symbol': 'INTERGENIC', 'gene_id': '', 'location': 'intergenic'}],
            'source': 'Fallback'
        }
    
    def map_snp_to_genes(self, snp_id: str) -> Dict:
        """Map a single SNP to its associated genes using multiple sources"""
        if snp_id in self.snp_gene_cache:
            return self.snp_gene_cache[snp_id]
            
        if snp_id in self.failed_snps:
            return self.create_fallback_mapping(snp_id)
        
        # Try NCBI first
        gene_info = self.query_ncbi_eutils(snp_id)
        if gene_info and gene_info['genes']:
            self.snp_gene_cache[snp_id] = gene_info
            time.sleep(self.rate_limit_delay)
            return gene_info
        
        # Try Ensembl as backup
        gene_info = self.query_ensembl_rest(snp_id)
        if gene_info and gene_info['genes']:
            self.snp_gene_cache[snp_id] = gene_info
            time.sleep(self.rate_limit_delay)
            return gene_info
        
        # Fallback mapping
        self.failed_snps.add(snp_id)
        fallback = self.create_fallback_mapping(snp_id)
        self.snp_gene_cache[snp_id] = fallback
        return fallback
    
    def batch_map_snps(self, snp_list: List[str], max_workers: int = 5) -> List[Dict]:
        """Map multiple SNPs to genes with parallel processing"""
        safe_print(f"Mapping {len(snp_list)} SNPs to genes...")
        safe_print("This may take several minutes due to API rate limits...")
        
        results = []
        
        # Process in smaller batches to respect rate limits
        batch_size = 50
        total_batches = len(snp_list) // batch_size + (1 if len(snp_list) % batch_size > 0 else 0)
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(snp_list))
            batch_snps = snp_list[start_idx:end_idx]
            
            safe_print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_snps)} SNPs)")
            
            batch_results = []
            for snp_id in batch_snps:
                result = self.map_snp_to_genes(snp_id)
                batch_results.append(result)
                
                # Progress indication
                if len(batch_results) % 10 == 0:
                    safe_print(f"  Processed {len(batch_results)}/{len(batch_snps)} SNPs in current batch")
            
            results.extend(batch_results)
            
            # Longer delay between batches
            if batch_idx < total_batches - 1:
                safe_print(f"Batch complete. Pausing 2 seconds before next batch...")
                time.sleep(2)
        
        safe_print(f"Completed mapping {len(results)} SNPs")
        safe_print(f"Successful mappings: {len([r for r in results if r['source'] != 'Fallback'])}")
        safe_print(f"Fallback mappings: {len([r for r in results if r['source'] == 'Fallback'])}")
        
        return results

def load_snp_list(filepath: str) -> List[str]:
    """Load SNP IDs from genotype matrix"""
    print("Loading SNP IDs from genotype matrix...")
    try:
        df = pd.read_csv(filepath)
        snp_ids = df.iloc[:, 0].astype(str).tolist()  # First column contains SNP IDs
        print(f"Loaded {len(snp_ids)} SNP IDs")
        return snp_ids
    except Exception as e:
        print(f"Error loading SNP list: {str(e)}")
        return []

def create_snp2gene_matrix(mapping_results: List[Dict]) -> pd.DataFrame:
    """Create comprehensive SNP to Gene matrix"""
    print("Creating SNP to Gene matrix...")
    
    # Prepare data for DataFrame
    matrix_data = []
    
    for result in mapping_results:
        snp_id = result['snp_id']
        chromosome = result.get('chromosome', 'Unknown')
        position = result.get('position', 'Unknown')
        source = result.get('source', 'Unknown')
        
        # Handle multiple genes per SNP
        if result['genes']:
            for gene in result['genes']:
                row = {
                    'SNP_ID': snp_id,
                    'Gene_Symbol': gene.get('symbol', 'Unknown'),
                    'Gene_ID': gene.get('gene_id', ''),
                    'Chromosome': chromosome,
                    'Position': position,
                    'Location_Type': gene.get('location', ''),
                    'Consequence': gene.get('consequence_terms', []),
                    'Source': source
                }
                matrix_data.append(row)
        else:
            # SNP with no gene associations
            row = {
                'SNP_ID': snp_id,
                'Gene_Symbol': 'INTERGENIC',
                'Gene_ID': '',
                'Chromosome': chromosome,
                'Position': position,
                'Location_Type': 'intergenic',
                'Consequence': [],
                'Source': source
            }
            matrix_data.append(row)
    
    # Create DataFrame
    snp2gene_df = pd.DataFrame(matrix_data)
    
    # Add summary statistics
    print(f"\nSNP2Gene Matrix Summary:")
    print(f"Total SNP-Gene associations: {len(snp2gene_df)}")
    print(f"Unique SNPs: {snp2gene_df['SNP_ID'].nunique()}")
    print(f"Unique Genes: {snp2gene_df['Gene_Symbol'].nunique()}")
    print(f"Chromosomes represented: {sorted(snp2gene_df['Chromosome'].unique())}")
    
    # Gene frequency analysis
    gene_counts = snp2gene_df['Gene_Symbol'].value_counts()
    print(f"\nTop 10 most frequent genes:")
    print(gene_counts.head(10))
    
    return snp2gene_df

def create_gene_patient_matrix(genotype_df: pd.DataFrame, snp2gene_df: pd.DataFrame) -> pd.DataFrame:
    """Create gene-level patient matrix by aggregating SNP data"""
    print("Creating gene-level patient matrix...")
    
    # Merge genotype data with SNP-gene mapping
    genotype_df_reset = genotype_df.reset_index()
    index_col_name = genotype_df_reset.columns[0]  # Get the actual index column name
    
    genotype_melted = genotype_df_reset.melt(
        id_vars=[index_col_name], 
        var_name='Patient', 
        value_name='Genotype'
    )
    genotype_melted.rename(columns={index_col_name: 'SNP_ID'}, inplace=True)
    
    # Merge with SNP to gene mapping
    gene_genotype = pd.merge(
        genotype_melted, 
        snp2gene_df[['SNP_ID', 'Gene_Symbol']], 
        on='SNP_ID', 
        how='left'
    )
    
    # Convert genotypes to numeric (same logic as original pipeline)
    def genotype_to_numeric(genotype):
        if pd.isna(genotype) or genotype in ['--', '#N/A', 'N/A']:
            return np.nan
        
        if isinstance(genotype, (int, float)):
            return genotype
        
        genotype_str = str(genotype)
        if len(genotype_str) != 2:
            return np.nan
        
        alleles = sorted(genotype_str)
        if alleles[0] == alleles[1]:  # Homozygous
            if genotype_str in ['AA', 'TT', 'GG', 'CC']:
                return 0  # Reference
            else:
                return 2  # Alternative homozygous
        else:  # Heterozygous
            return 1
    
    gene_genotype['Numeric_Genotype'] = gene_genotype['Genotype'].apply(genotype_to_numeric)
    
    # Aggregate by gene and patient (take mean of variants within gene)
    gene_patient_matrix = gene_genotype.groupby(['Gene_Symbol', 'Patient'])['Numeric_Genotype'].mean().unstack(fill_value=np.nan)
    
    print(f"Gene-Patient matrix created: {len(gene_patient_matrix)} genes x {len(gene_patient_matrix.columns)} patients")
    
    return gene_patient_matrix

def generate_gene_analysis_report(snp2gene_df: pd.DataFrame, gene_patient_matrix: pd.DataFrame) -> str:
    """Generate comprehensive gene-level analysis report"""
    report = f"""
SNP2GENE MAPPING ANALYSIS REPORT
================================
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

MAPPING SUMMARY
===============
Total SNP-Gene associations: {len(snp2gene_df)}
Unique SNPs mapped: {snp2gene_df['SNP_ID'].nunique()}
Unique genes identified: {snp2gene_df['Gene_Symbol'].nunique()}
Data sources used: {', '.join(snp2gene_df['Source'].unique())}

CHROMOSOME DISTRIBUTION
=======================
"""
    
    # Chromosome distribution
    chrom_dist = snp2gene_df['Chromosome'].value_counts().sort_index()
    for chrom, count in chrom_dist.head(10).items():
        report += f"Chromosome {chrom}: {count} SNP-gene associations\n"
    
    report += f"""

GENE FREQUENCY ANALYSIS
=======================
Top 20 genes with most SNP associations:
"""
    
    gene_counts = snp2gene_df['Gene_Symbol'].value_counts()
    for gene, count in gene_counts.head(20).items():
        report += f"{gene}: {count} SNPs\n"
    
    report += f"""

GENE-PATIENT MATRIX STATISTICS
==============================
Genes in patient matrix: {len(gene_patient_matrix)}
Patients analyzed: {len(gene_patient_matrix.columns)}
Missing data percentage: {(gene_patient_matrix.isna().sum().sum() / gene_patient_matrix.size * 100):.2f}%

HIGH-IMPACT GENES (>10 SNPs)
=============================
"""
    
    high_impact_genes = gene_counts[gene_counts > 10]
    for gene, count in high_impact_genes.items():
        if gene != 'INTERGENIC':
            report += f"{gene}: {count} SNPs - High priority for functional analysis\n"
    
    report += f"""

RECOMMENDATIONS
===============
1. Focus functional analysis on high-impact genes ({len(high_impact_genes)} genes with >10 SNPs)
2. Investigate intergenic regions ({gene_counts.get('INTERGENIC', 0)} SNPs) for regulatory elements
3. Consider gene-based association testing for improved statistical power
4. Integrate with pathway databases for biological interpretation
5. Validate gene assignments using functional annotation tools

Analysis completed successfully.
"""
    
    return report

def save_results(snp2gene_df: pd.DataFrame, gene_patient_matrix: pd.DataFrame, report: str):
    """Save all results to files"""
    print("Saving results...")
    
    # Save SNP to Gene mapping matrix
    snp2gene_df.to_csv('snp2gene_comprehensive_matrix.csv', index=False)
    print("Saved: snp2gene_comprehensive_matrix.csv")
    
    # Save gene-patient matrix
    gene_patient_matrix.to_csv('gene_patient_matrix.csv')
    print("Saved: gene_patient_matrix.csv")
    
    # Save report
    with open('snp2gene_analysis_report.txt', 'w') as f:
        f.write(report)
    print("Saved: snp2gene_analysis_report.txt")
    
    # Save mapping cache for future use
    print("All results saved successfully!")

def main():
    """Main pipeline execution"""
    print("=== SNP TO GENE MAPPING PIPELINE ===")
    
    try:
        # 1. Load SNP IDs from genotype matrix
        snp_ids = load_snp_list('snp2genotype_matrix_fixed.csv')
        if not snp_ids:
            print("Error: Could not load SNP IDs. Please check file path.")
            return
        
        # Process all SNPs (full analysis)
        test_snps = snp_ids  # Full analysis
        print(f"Processing {len(test_snps)} SNPs for full analysis...")
        
        # 2. Map SNPs to genes
        mapper = SNPToGeneMapper()
        mapping_results = mapper.batch_map_snps(test_snps)
        
        # 3. Create SNP to Gene matrix
        snp2gene_df = create_snp2gene_matrix(mapping_results)
        
        # 4. Load genotype data and create gene-patient matrix
        print("Loading genotype data for gene-level analysis...")
        genotype_df = pd.read_csv('snp2genotype_matrix_fixed.csv', index_col=0)
        
        # Filter genotype data to match processed SNPs
        genotype_df_filtered = genotype_df.loc[genotype_df.index.isin(test_snps)]
        
        # Create gene-patient matrix
        gene_patient_matrix = create_gene_patient_matrix(genotype_df_filtered, snp2gene_df)
        
        # 5. Generate comprehensive report
        report = generate_gene_analysis_report(snp2gene_df, gene_patient_matrix)
        
        # 6. Save all results
        save_results(snp2gene_df, gene_patient_matrix, report)
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Generated files:")
        print("- snp2gene_comprehensive_matrix.csv (SNP to gene mappings)")
        print("- gene_patient_matrix.csv (gene-level genotype matrix)")  
        print("- snp2gene_analysis_report.txt (detailed analysis report)")
        
        print(f"\nKey Results:")
        print(f"- {len(test_snps)} SNPs processed")
        print(f"- {snp2gene_df['Gene_Symbol'].nunique()} unique genes identified")
        print(f"- {len(gene_patient_matrix)} genes in patient matrix")
        print(f"- Ready for integration with gpt_snp2gene_pipeline.py")
        
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()