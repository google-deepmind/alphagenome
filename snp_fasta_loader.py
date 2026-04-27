#!/usr/bin/env python3
"""
SMS SNP Analysis Script

This script:
1. Loads SMS SNPs from a TSV file with columns: chr, pos, ref, alt
2. Downloads hg19 chr17 FASTA sequence for region 14.89-22 Mb using pyfaidx
3. Provides basic analysis of the SNPs in relation to the downloaded sequence

Requirements:
- pandas
- pyfaidx
"""

import pandas as pd
import os
import sys
from pyfaidx import Fasta
import urllib.request
import gzip
import shutil


def download_hg19_chr17_fasta(output_dir="./", force_download=False):
    """
    Download hg19 chr17 FASTA file using pyfaidx.
    
    Args:
        output_dir (str): Directory to save the FASTA file
        force_download (bool): Force re-download even if file exists
        
    Returns:
        str: Path to the downloaded FASTA file
    """
    fasta_file = os.path.join(output_dir, "chr17.fa")
    
    if os.path.exists(fasta_file) and not force_download:
        print(f"FASTA file already exists: {fasta_file}")
        return fasta_file
    
    print("Downloading hg19 chr17 FASTA from UCSC...")
    
    # UCSC hg19 chr17 URL
    url = "https://hgdownload.cse.ucsc.edu/goldenpath/hg19/chromosomes/chr17.fa.gz"
    gz_file = fasta_file + ".gz"
    
    try:
        # Download compressed file
        print("Downloading compressed FASTA file...")
        urllib.request.urlretrieve(url, gz_file)
        
        # Decompress
        print("Decompressing FASTA file...")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(fasta_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove compressed file
        os.remove(gz_file)
        print(f"Downloaded and extracted: {fasta_file}")
        
    except Exception as e:
        print(f"Error downloading FASTA: {e}")
        if os.path.exists(gz_file):
            os.remove(gz_file)
        raise
    
    return fasta_file


def load_snps_from_tsv(tsv_file):
    """
    Load SNPs from TSV file.
    
    Args:
        tsv_file (str): Path to TSV file with columns: chr, pos, ref, alt
        
    Returns:
        pandas.DataFrame: DataFrame containing SNP data
    """
    try:
        # Load TSV file
        print(f"Loading SNPs from: {tsv_file}")
        snps_df = pd.read_csv(tsv_file, sep='\t')
        
        # Validate required columns
        required_cols = ['chr', 'pos', 'ref', 'alt']
        if not all(col in snps_df.columns for col in required_cols):
            raise ValueError(f"TSV file must contain columns: {required_cols}")
        
        print(f"Loaded {len(snps_df)} SNPs from TSV file")
        
        # Filter for chr17 SNPs if any exist
        chr17_snps = snps_df[snps_df['chr'].astype(str).str.contains('17', na=False)]
        if not chr17_snps.empty:
            print(f"Found {len(chr17_snps)} SNPs on chromosome 17")
        
        return snps_df
        
    except Exception as e:
        print(f"Error loading TSV file: {e}")
        raise


def extract_region_sequence(fasta_file, chromosome, start_pos, end_pos):
    """
    Extract sequence from specific genomic region.
    
    Args:
        fasta_file (str): Path to FASTA file
        chromosome (str): Chromosome identifier
        start_pos (int): Start position (1-based)
        end_pos (int): End position (1-based)
        
    Returns:
        str: DNA sequence for the specified region
    """
    try:
        print(f"Loading FASTA file: {fasta_file}")
        fasta = Fasta(fasta_file)
        
        # Handle different chromosome naming conventions
        chr_name = chromosome
        if chr_name not in fasta.keys():
            chr_name = f"chr{chromosome}" if not chromosome.startswith('chr') else chromosome.replace('chr', '')
            if chr_name not in fasta.keys():
                available_chrs = list(fasta.keys())[:5]  # Show first 5 available
                raise KeyError(f"Chromosome '{chromosome}' not found. Available: {available_chrs}...")
        
        print(f"Extracting region {chr_name}:{start_pos:,}-{end_pos:,}")
        
        # Extract sequence (pyfaidx uses 0-based indexing internally but accepts 1-based)
        sequence = fasta[chr_name][start_pos-1:end_pos].seq
        
        print(f"Extracted {len(sequence):,} bp from {chr_name}")
        return sequence
        
    except Exception as e:
        print(f"Error extracting sequence: {e}")
        raise


def analyze_snps_in_region(snps_df, sequence, chr_id, start_pos, end_pos):
    """
    Analyze SNPs within the downloaded genomic region.
    
    Args:
        snps_df (pandas.DataFrame): SNP data
        sequence (str): DNA sequence
        chr_id (str): Chromosome identifier
        start_pos (int): Region start position
        end_pos (int): Region end position
    """
    print(f"\nAnalyzing SNPs in region chr{chr_id}:{start_pos:,}-{end_pos:,}")
    
    # Filter SNPs in the region
    region_snps = snps_df[
        (snps_df['chr'].astype(str).str.contains(str(chr_id), na=False)) &
        (snps_df['pos'] >= start_pos) &
        (snps_df['pos'] <= end_pos)
    ].copy()
    
    if region_snps.empty:
        print("No SNPs found in the specified region")
        return
    
    print(f"Found {len(region_snps)} SNPs in the region:")
    
    # Analyze each SNP
    for idx, snp in region_snps.iterrows():
        pos_in_seq = int(snp['pos']) - start_pos
        if 0 <= pos_in_seq < len(sequence):
            ref_in_seq = sequence[pos_in_seq].upper()
            ref_expected = str(snp['ref']).upper()
            
            match_status = "✓" if ref_in_seq == ref_expected else "✗"
            
            print(f"  {snp['chr']}:{snp['pos']} {snp['ref']}→{snp['alt']} "
                  f"(seq: {ref_in_seq}) {match_status}")
        else:
            print(f"  {snp['chr']}:{snp['pos']} {snp['ref']}→{snp['alt']} (position out of range)")


def main():
    """Main function to orchestrate the SNP analysis workflow."""
    
    # Configuration
    TSV_FILE = "sms_snps.tsv"  # Default filename - change as needed
    REGION_CHR = "17"
    REGION_START = 14890000  # 14.89 Mb
    REGION_END = 22000000    # 22 Mb
    
    print("SMS SNP Analysis Script")
    print("=" * 50)
    
    try:
        # Check if TSV file exists
        if len(sys.argv) > 1:
            TSV_FILE = sys.argv[1]
        
        if not os.path.exists(TSV_FILE):
            print(f"Error: TSV file '{TSV_FILE}' not found")
            print(f"Usage: python {sys.argv[0]} [tsv_file]")
            print(f"Expected TSV format: chr\\tpos\\tref\\talt")
            return 1
        
        # Step 1: Load SNPs from TSV
        snps_df = load_snps_from_tsv(TSV_FILE)
        
        # Step 2: Download hg19 chr17 FASTA
        fasta_file = download_hg19_chr17_fasta()
        
        # Step 3: Extract the specific region (14.89-22 Mb)
        sequence = extract_region_sequence(fasta_file, REGION_CHR, REGION_START, REGION_END)
        
        # Step 4: Analyze SNPs in the region
        analyze_snps_in_region(snps_df, sequence, REGION_CHR, REGION_START, REGION_END)
        
        # Summary statistics
        print(f"\n" + "=" * 50)
        print("Summary:")
        print(f"  Total SNPs loaded: {len(snps_df):,}")
        print(f"  Region analyzed: chr{REGION_CHR}:{REGION_START:,}-{REGION_END:,}")
        print(f"  Region size: {len(sequence):,} bp")
        print(f"  FASTA file: {fasta_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())