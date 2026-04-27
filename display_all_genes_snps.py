#!/usr/bin/env python3
"""
Display All Genes and SNPs from SNP2Gene Mapping Results
Reads the comprehensive matrix and shows detailed information
"""

import pandas as pd
import numpy as np
from collections import Counter

def display_comprehensive_results():
    """Display all genes and SNPs from the mapping results"""
    
    try:
        # Load the comprehensive SNP to Gene matrix
        print("Loading SNP2Gene comprehensive matrix...")
        snp2gene_df = pd.read_csv('snp2gene_comprehensive_matrix.csv')
        
        print(f"\n=== COMPLETE SNP2GENE ANALYSIS RESULTS ===")
        print(f"Total SNP-Gene associations: {len(snp2gene_df)}")
        print(f"Unique SNPs: {snp2gene_df['SNP_ID'].nunique()}")
        print(f"Unique Genes: {snp2gene_df['Gene_Symbol'].nunique()}")
        
        # Display all unique genes
        print(f"\n=== ALL GENES IDENTIFIED ({snp2gene_df['Gene_Symbol'].nunique()} total) ===")
        gene_counts = snp2gene_df['Gene_Symbol'].value_counts()
        for i, (gene, count) in enumerate(gene_counts.items(), 1):
            print(f"{i:2d}. {gene:<20} ({count:3d} SNPs)")
        
        # Display chromosomes
        print(f"\n=== CHROMOSOME DISTRIBUTION ===")
        chrom_counts = snp2gene_df['Chromosome'].value_counts().sort_index()
        for chrom, count in chrom_counts.items():
            print(f"Chromosome {chrom}: {count} associations")
        
        # Display all SNPs by gene
        print(f"\n=== SNPs BY GENE (detailed breakdown) ===")
        for gene in gene_counts.index:
            snps_for_gene = snp2gene_df[snp2gene_df['Gene_Symbol'] == gene]['SNP_ID'].tolist()
            print(f"\n{gene} ({len(snps_for_gene)} SNPs):")
            # Show SNPs in rows of 5 for better readability
            for i in range(0, len(snps_for_gene), 5):
                snp_row = snps_for_gene[i:i+5]
                print(f"  {', '.join(snp_row)}")
        
        # Display SNP details with positions
        print(f"\n=== SNP DETAILS WITH POSITIONS (first 50) ===")
        print(f"{'SNP_ID':<15} {'Gene':<15} {'Chromosome':<4} {'Position':<12} {'Source'}")
        print("-" * 70)
        for _, row in snp2gene_df.head(50).iterrows():
            print(f"{row['SNP_ID']:<15} {row['Gene_Symbol']:<15} {str(row['Chromosome']):<4} {str(row['Position']):<12} {row['Source']}")
        
        if len(snp2gene_df) > 50:
            print(f"... and {len(snp2gene_df) - 50} more SNP-gene associations")
        
        # Gene-patient matrix if available
        try:
            gene_patient_df = pd.read_csv('gene_patient_matrix.csv', index_col=0)
            print(f"\n=== GENE-PATIENT MATRIX SUMMARY ===")
            print(f"Genes in matrix: {len(gene_patient_df)}")
            print(f"Patients: {len(gene_patient_df.columns)}")
            print(f"Patient IDs: {list(gene_patient_df.columns)}")
            
            print(f"\n=== GENE EXPRESSION/GENOTYPE SUMMARY ===")
            for gene in gene_patient_df.index:
                non_na_count = gene_patient_df.loc[gene].notna().sum()
                mean_value = gene_patient_df.loc[gene].mean()
                print(f"{gene:<20}: {non_na_count:2d}/{len(gene_patient_df.columns)} patients, mean={mean_value:.3f}")
                
        except FileNotFoundError:
            print("Gene-patient matrix not found. Run the full pipeline to generate it.")
        
        # Export detailed lists to files
        print(f"\n=== CREATING DETAILED OUTPUT FILES ===")
        
        # All genes list
        with open('all_genes_list.txt', 'w') as f:
            f.write("ALL GENES IDENTIFIED\n")
            f.write("===================\n\n")
            for i, (gene, count) in enumerate(gene_counts.items(), 1):
                f.write(f"{i:2d}. {gene} ({count} SNPs)\n")
        print("Created: all_genes_list.txt")
        
        # All SNPs list
        with open('all_snps_list.txt', 'w') as f:
            f.write("ALL SNPs MAPPED\n")
            f.write("===============\n\n")
            all_snps = sorted(snp2gene_df['SNP_ID'].unique())
            for i, snp in enumerate(all_snps, 1):
                genes = snp2gene_df[snp2gene_df['SNP_ID'] == snp]['Gene_Symbol'].tolist()
                f.write(f"{i:4d}. {snp} -> {', '.join(genes)}\n")
        print("Created: all_snps_list.txt")
        
        # SNPs by gene detailed file
        with open('snps_by_gene_detailed.txt', 'w') as f:
            f.write("SNPs BY GENE - DETAILED BREAKDOWN\n")
            f.write("==================================\n\n")
            for gene in gene_counts.index:
                gene_data = snp2gene_df[snp2gene_df['Gene_Symbol'] == gene]
                f.write(f"\n{gene} ({len(gene_data)} SNPs):\n")
                f.write("-" * (len(gene) + 20) + "\n")
                for _, row in gene_data.iterrows():
                    f.write(f"  {row['SNP_ID']} - Chr{row['Chromosome']}:{row['Position']} ({row['Source']})\n")
        print("Created: snps_by_gene_detailed.txt")
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print("All genes and SNPs have been displayed and exported to detailed files.")
        
    except FileNotFoundError:
        print("Error: snp2gene_comprehensive_matrix.csv not found.")
        print("Please run the SNP2Gene mapping pipeline first.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    display_comprehensive_results()