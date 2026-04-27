"""Analyze patient SNPs using AlphaGenome.

This script takes SNP IDs (rs numbers), retrieves their genomic coordinates
and alleles, and then uses AlphaGenome to predict the functional effects
across different tissues and molecular outputs.
"""

import json
import os
import time
import requests
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components


def fetch_snp_info(snp_id: str) -> Optional[Dict]:
  """Fetch SNP genomic information from dbSNP via MyVariant.info API.
  
  Args:
    snp_id: SNP identifier (e.g., 'rs2230317')
    
  Returns:
    Dictionary with chromosome, position, ref, alt information, or None if not found
  """
  # MyVariant.info provides a cleaner API than raw NCBI
  url = f'https://myvariant.info/v1/variant/{snp_id}'
  
  try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    # Check if variant was found
    if 'error' in data or '_id' not in data:
      print(f'  No variant data found for {snp_id}')
      return None
    
    # Try to get hg38 coordinates (preferred)
    position = None
    chromosome = None
    
    if 'hg38' in data:
      position = data['hg38'].get('start')
      chromosome = data['hg38'].get('chrom', data.get('chrom'))
    elif 'chrom' in data and 'hg19' in data:
      # Fall back to hg19 if hg38 not available
      print(f'  WARNING: Using hg19 coordinates for {snp_id} - may need manual hg38 conversion')
      position = data['hg19'].get('start')
      chromosome = data.get('chrom')
    elif 'chrom' in data and any(coord in data for coord in ['start', 'pos']):
      # Use whatever coordinates are available
      position = data.get('start') or data.get('pos')
      chromosome = data.get('chrom')
    else:
      print(f'  No genomic coordinates found for {snp_id}')
      return None
    
    if not position or not chromosome:
      print(f'  Incomplete coordinate information for {snp_id}')
      return None
    
    # Get alleles from VCF fields or dbsnp
    ref = data.get('ref')
    alt = data.get('alt')
    
    # If not in main data, try dbsnp section
    if not ref or not alt:
      if 'dbsnp' in data:
        dbsnp = data['dbsnp']
        ref = ref or dbsnp.get('ref')
        alt = alt or dbsnp.get('alt')
    
    if not ref or not alt:
      print(f'  No allele information found for {snp_id}')
      return None
    
    # Get gene information
    gene_name = 'Unknown'
    if 'dbsnp' in data and 'gene' in data['dbsnp']:
      gene_info = data['dbsnp']['gene']
      if isinstance(gene_info, list) and gene_info:
        # Take the first gene if multiple
        gene_name = gene_info[0].get('symbol', 'Unknown')
      elif isinstance(gene_info, dict):
        gene_name = gene_info.get('symbol', 'Unknown')
    
    # Ensure chromosome has 'chr' prefix
    if not str(chromosome).startswith('chr'):
      chromosome = f'chr{chromosome}'
    
    return {
        'snp_id': snp_id,
        'chromosome': chromosome,
        'position': int(position),  # Should be 1-based already
        'reference': str(ref).upper(),
        'alternate': str(alt).upper(),
        'gene': gene_name,
    }
    
  except Exception as e:
    print(f'  Error fetching {snp_id}: {str(e)}')
    return None


def create_variant_from_snp(snp_info: Dict) -> Optional[genome.Variant]:
  """Create an AlphaGenome Variant object from SNP info.
  
  Args:
    snp_info: Dictionary with SNP information
    
  Returns:
    genome.Variant object or None if invalid
  """
  try:
    # Handle multi-allelic sites - take first alternate allele
    alt = snp_info['alternate']
    if isinstance(alt, list):
      alt = alt[0]
    
    variant = genome.Variant(
        chromosome=snp_info['chromosome'],
        position=snp_info['position'],
        reference_bases=snp_info['reference'].upper(),
        alternate_bases=alt.upper(),
        name=snp_info['snp_id']
    )
    return variant
  except Exception as e:
    print(f"  Error creating variant for {snp_info['snp_id']}: {str(e)}")
    return None


def analyze_snp_with_alphagenome(
    variant: genome.Variant,
    api_key: str,
    context_window: int = 16384,
    scorer_window: int = 10001,
    tissues: Optional[List[str]] = None,
    output_types: Optional[List[str]] = None,
) -> Dict:
  """Analyze a SNP using AlphaGenome predictions.
  
  Args:
    variant: genome.Variant object
    api_key: AlphaGenome API key
    context_window: Size of genomic window to analyze (must be power of 2: 16384, 131072, 524288, 1048576)
    scorer_window: Size of window for scoring variant effects (typically 10001 for 10kb)
    tissues: List of UBERON tissue ontology terms. Defaults to brain/lung/liver.
    output_types: List of output types to analyze. Defaults to RNA_SEQ, ATAC, DNASE, CAGE.
    
  Returns:
    Dictionary with prediction results
  """
  if tissues is None:
    # Default to some interesting tissues
    tissues = [
        'UBERON:0001157',  # brain
        'UBERON:0002048',  # lung
        'UBERON:0002107',  # liver
        'UBERON:0000948',  # heart
    ]
  
  if output_types is None:
    # Default to key functional genomics assays
    output_types = [
        'RNA_SEQ',      # Gene expression
        'ATAC',         # Chromatin accessibility
        'DNASE',        # DNase hypersensitivity
        'CAGE',         # Transcription start sites
    ]
  
  print(f'  Analyzing {variant.name} ({variant})')
  print(f'    Tissues: {len(tissues)}')
  print(f'    Output types: {len(output_types)}')
  
  # Create client
  model = dna_client.create(api_key)
  
  # Create genomic interval centered on variant
  # AlphaGenome uses 0-based coordinates
  variant_center = variant.position - 1  # Convert to 0-based
  interval = genome.Interval(
      chromosome=variant.chromosome,
      start=max(0, variant_center - context_window // 2),
      end=variant_center + context_window // 2,
  )
  
  print(f'    Interval: {interval}')
  
  # Define variant scorers for different output types
  scorers = []
  for output_type in output_types:
    try:
      scorer = variant_scorers.CenterMaskScorer(
          requested_output=getattr(dna_client.OutputType, output_type),
          width=scorer_window,
          aggregation_type=variant_scorers.AggregationType.DIFF_MEAN
      )
      scorers.append(scorer)
    except AttributeError:
      print(f'    Warning: Unknown output type {output_type}, skipping')
  
  # Score variant
  try:
    scores = model.score_variants(
        intervals=[interval],
        variants=[variant],
        variant_scorers=scorers,
        organism=dna_client.Organism.HOMO_SAPIENS,
    )
    
    # Convert AnnData results to DataFrame for easier handling
    import pandas as pd
    
    # scores is a list of lists of AnnData objects
    # For simplicity, let's just return the raw scores for now
    return {
        'variant': str(variant),
        'interval': str(interval),
        'scores': scores,  # This will be a list of AnnData objects
        'tissues': tissues,
    }
  except Exception as e:
    print(f'  Error scoring variant: {str(e)}')
    return None


def get_manual_snp_coordinates():
  """Manual SNP coordinate lookup for SNPs not found in MyVariant.info."""
  # If you have known coordinates for missing SNPs, add them here
  # Format: {snp_id: {'chromosome': 'chr1', 'position': 12345, 'reference': 'A', 'alternate': 'T', 'gene': 'GENE1'}}
  manual_coords = {
      # Add any known SNP coordinates here if needed
      # 'rs2214565': {'chromosome': 'chr1', 'position': 12345, 'reference': 'A', 'alternate': 'T', 'gene': 'Unknown'}
  }
  return manual_coords


def main():
  """Main analysis pipeline for patient SNPs."""
  
  # ========== CONFIGURATION ==========
  
  # Your patient SNPs
  patient_snps = [
      'rs2230317',
      'rs10273927',
      'rs7398691',
      'rs10082916',
      'rs2214565'
  ]
  
  # Genomic context window (must be power of 2)
  # Options: 16384 (16kb), 131072 (100kb), 524288 (500kb), 1048576 (1mb)
  context_window = 16384
  
  # Scoring window around variant (typically 10kb)
  scorer_window = 10001
  
  # Tissues to analyze (UBERON ontology terms)
  # See: https://www.alphagenomedocs.com/outputs for full list
  tissues = [
      'UBERON:0001157',  # brain
      'UBERON:0002048',  # lung
      'UBERON:0002107',  # liver
      'UBERON:0000948',  # heart
      'UBERON:0002113',  # kidney
      'UBERON:0000970',  # eye
      'UBERON:0002106',  # spleen
  ]
  
  # Output types to analyze
  # Available: RNA_SEQ, ATAC, DNASE, CAGE, PROCAP, CHIP_HISTONE, CHIP_TF
  output_types = [
      'RNA_SEQ',   # Gene expression (RNA-seq)
      'ATAC',      # Chromatin accessibility (ATAC-seq)
      'DNASE',     # DNase hypersensitivity
      'CAGE',      # TSS activity (CAGE)
      'PROCAP',    # TSS activity (PRO-cap)
  ]
  
  # ===================================
  
  # Get API key (use your working key directly for now)
  api_key = "AlzaSyCy8DOGotXZ93omnBh8hGf8jlJd7eLaLO4"
  print(f'Using API key: {api_key[:10]}... (length: {len(api_key)})')
  
  print('='*80)
  print('AlphaGenome Patient SNP Analysis')
  print('='*80)
  print(f'SNPs to analyze: {len(patient_snps)}')
  print(f'Context window: {context_window:,} bp ({context_window/1024:.0f}kb)')
  print(f'Scoring window: {scorer_window:,} bp ({scorer_window/1024:.1f}kb)')
  print(f'Tissues: {len(tissues)}')
  print(f'Output types: {len(output_types)}')
  print('='*80 + '\n')
  
  # Step 1: Fetch SNP information
  print('Step 1: Fetching SNP genomic coordinates...')
  snp_info_list = []
  manual_coords = get_manual_snp_coordinates()
  
  for snp_id in patient_snps:
    print(f'  Fetching {snp_id}...')
    info = fetch_snp_info(snp_id)
    
    # If not found in API, check manual coordinates
    if not info and snp_id in manual_coords:
      manual_info = manual_coords[snp_id].copy()
      manual_info['snp_id'] = snp_id
      info = manual_info
      print(f'  Using manual coordinates for {snp_id}')
    
    if info:
      snp_info_list.append(info)
      print(f'    → {info["chromosome"]}:{info["position"]} {info["reference"]}>{info["alternate"]} (Gene: {info["gene"]})')
    time.sleep(0.5)  # Be nice to the API
  
  print(f'\nSuccessfully retrieved {len(snp_info_list)}/{len(patient_snps)} SNPs\n')
  
  # Step 2: Create Variant objects
  print('Step 2: Creating Variant objects...')
  variants = []
  for snp_info in snp_info_list:
    variant = create_variant_from_snp(snp_info)
    if variant:
      variants.append((snp_info, variant))
      print(f'  ✓ Created variant for {snp_info["snp_id"]}')
  
  print(f'\nCreated {len(variants)} valid variants\n')
  
  # Step 3: Analyze with AlphaGenome
  print('Step 3: Analyzing variants with AlphaGenome...')
  print(f'This will make {len(variants)} API calls...\n')
  results = []
  for i, (snp_info, variant) in enumerate(variants, 1):
    print(f'[{i}/{len(variants)}] Analyzing {snp_info["snp_id"]} (Gene: {snp_info["gene"]})...')
    result = analyze_snp_with_alphagenome(
        variant, 
        api_key,
        context_window=context_window,
        scorer_window=scorer_window,
        tissues=tissues,
        output_types=output_types,
    )
    if result:
      results.append({
          'snp_info': snp_info,
          'variant': variant,
          'alphagenome_result': result,
      })
    # Small delay between API calls
    if i < len(variants):
      time.sleep(1)
  
  # Step 4: Summarize results
  print('\n' + '='*80)
  print('RESULTS SUMMARY')
  print('='*80 + '\n')
  
  for result in results:
    snp_info = result['snp_info']
    scores = result['alphagenome_result']['scores']
    
    print(f"\n{snp_info['snp_id']} - {snp_info['gene']} ({snp_info['chromosome']}:{snp_info['position']})")
    print(f"  Reference: {snp_info['reference']} → Alternate: {snp_info['alternate']}")
    print(f'\n  AlphaGenome Scores (AnnData objects): {len(scores[0])} scorers')
    
    # Print summary of each scorer result
    for i, anndata_obj in enumerate(scores[0]):
      print(f"    Scorer {i+1}: shape {anndata_obj.X.shape}, {anndata_obj.uns.get('variant_scorer', 'Unknown scorer')}")
    print('\n' + '-'*80)
  
  # Save results to file
  output_file = 'patient_snp_alphagenome_results.json'
  print(f'\nSaving detailed results to {output_file}...')
  
  # Convert to serializable format
  serializable_results = []
  for result in results:
    # Convert AnnData objects to summary info
    scores_summary = []
    for anndata_obj in result['alphagenome_result']['scores'][0]:
      scores_summary.append({
          'scorer': str(anndata_obj.uns.get('variant_scorer', 'Unknown')),
          'shape': anndata_obj.X.shape,
          'mean_score': float(anndata_obj.X.mean()) if anndata_obj.X.size > 0 else 0.0,
          'tissue_tracks': anndata_obj.var.get('name', []).tolist() if hasattr(anndata_obj.var, 'name') else [],
      })
    
    serializable_results.append({
        'snp_id': result['snp_info']['snp_id'],
        'gene': result['snp_info']['gene'],
        'chromosome': result['snp_info']['chromosome'],
        'position': result['snp_info']['position'],
        'reference': result['snp_info']['reference'],
        'alternate': result['snp_info']['alternate'],
        'scores': scores_summary,
    })
  
  with open(output_file, 'w') as f:
    json.dump(serializable_results, f, indent=2)
  
  print(f'✓ Results saved to {output_file}')
  print('\nAnalysis complete!')


if __name__ == '__main__':
  main()
