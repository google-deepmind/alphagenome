#!/usr/bin/env python3
"""Mock version of the SNP analysis script to demonstrate expected results."""

import json
import time
import requests
import numpy as np
from typing import Dict, List, Optional

def fetch_snp_info(snp_id: str) -> Optional[Dict]:
    """Fetch SNP genomic information from dbSNP via MyVariant.info API."""
    url = f'https://myvariant.info/v1/variant/{snp_id}'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data or '_id' not in data:
            print(f'  No variant data found for {snp_id}')
            return None
        
        # Try to get coordinates
        position = None
        chromosome = None
        
        if 'hg38' in data:
            position = data['hg38'].get('start')
            chromosome = data['hg38'].get('chrom', data.get('chrom'))
        elif 'chrom' in data and 'hg19' in data:
            print(f'  WARNING: Using hg19 coordinates for {snp_id} - may need manual hg38 conversion')
            position = data['hg19'].get('start')
            chromosome = data.get('chrom')
        elif 'chrom' in data and any(coord in data for coord in ['start', 'pos']):
            position = data.get('start') or data.get('pos')
            chromosome = data.get('chrom')
        else:
            print(f'  No genomic coordinates found for {snp_id}')
            return None
        
        if not position or not chromosome:
            print(f'  Incomplete coordinate information for {snp_id}')
            return None
        
        # Get alleles
        ref = data.get('ref')
        alt = data.get('alt')
        
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
                gene_name = gene_info[0].get('symbol', 'Unknown')
            elif isinstance(gene_info, dict):
                gene_name = gene_info.get('symbol', 'Unknown')
        
        # Ensure chromosome has 'chr' prefix
        if not str(chromosome).startswith('chr'):
            chromosome = f'chr{chromosome}'
        
        return {
            'snp_id': snp_id,
            'chromosome': chromosome,
            'position': int(position),
            'reference': str(ref).upper(),
            'alternate': str(alt).upper(),
            'gene': gene_name,
        }
        
    except Exception as e:
        print(f'  Error fetching {snp_id}: {str(e)}')
        return None

def mock_alphagenome_analysis(snp_info: Dict, output_types: List[str]) -> Dict:
    """Generate mock AlphaGenome results for demonstration."""
    
    # Simulate different effect sizes based on SNP characteristics
    np.random.seed(hash(snp_info['snp_id']) % 2**32)  # Reproducible "results"
    
    mock_scores = []
    
    for output_type in output_types:
        # Generate mock scores that vary by output type and location
        base_score = np.random.normal(0, 0.1)  # Small effect size
        
        # Make some SNPs have larger effects in specific tissues
        if snp_info['gene'] == 'DRG2' and output_type == 'RNA_SEQ':
            base_score *= 3  # Larger effect in gene expression
        elif snp_info['gene'] == 'CUX1' and output_type == 'ATAC':
            base_score *= 2  # Chromatin accessibility effect
        
        # Add some tissue-specific variation (simulate 7 tissues)
        tissue_scores = [base_score + np.random.normal(0, 0.05) for _ in range(7)]
        
        mock_scores.append({
            'output_type': output_type,
            'mean_score': float(base_score),
            'tissue_scores': tissue_scores,
            'num_tracks': np.random.randint(50, 200),  # Mock number of tissue tracks
        })
    
    return {
        'variant': f"{snp_info['chromosome']}:{snp_info['position']}:{snp_info['reference']}>{snp_info['alternate']}",
        'scores': mock_scores,
        'interpretation': interpret_mock_scores(mock_scores, snp_info),
    }

def interpret_mock_scores(scores: List[Dict], snp_info: Dict) -> Dict:
    """Provide interpretation of mock scores."""
    
    # Find the strongest effects
    max_effect = max(abs(score['mean_score']) for score in scores)
    strongest_effects = [s for s in scores if abs(s['mean_score']) > max_effect * 0.7]
    
    interpretation = {
        'overall_effect_size': 'Large' if max_effect > 0.2 else 'Moderate' if max_effect > 0.1 else 'Small',
        'strongest_effects': [s['output_type'] for s in strongest_effects],
        'predicted_impact': [],
    }
    
    # Add some interpretive text
    for effect in strongest_effects:
        if effect['output_type'] == 'RNA_SEQ':
            direction = 'increased' if effect['mean_score'] > 0 else 'decreased'
            interpretation['predicted_impact'].append(f"Likely {direction} gene expression in {snp_info['gene']}")
        elif effect['output_type'] == 'ATAC':
            direction = 'increased' if effect['mean_score'] > 0 else 'decreased'
            interpretation['predicted_impact'].append(f"Potential {direction} chromatin accessibility")
        elif effect['output_type'] == 'DNASE':
            interpretation['predicted_impact'].append("May affect transcription factor binding")
    
    if not interpretation['predicted_impact']:
        interpretation['predicted_impact'].append("Likely neutral or minimal functional impact")
    
    return interpretation

def main():
    """Mock analysis pipeline for patient SNPs."""
    
    # Patient SNPs
    patient_snps = [
        'rs2230317',
        'rs10273927', 
        'rs7398691',
        'rs10082916',
        'rs2214565'
    ]
    
    # Output types to analyze
    output_types = [
        'RNA_SEQ',   # Gene expression (RNA-seq)
        'ATAC',      # Chromatin accessibility (ATAC-seq)
        'DNASE',     # DNase hypersensitivity
        'CAGE',      # TSS activity (CAGE)
        'PROCAP',    # TSS activity (PRO-cap)
    ]
    
    print('='*80)
    print('AlphaGenome Patient SNP Analysis (MOCK VERSION)')
    print('='*80)
    print(f'SNPs to analyze: {len(patient_snps)}')
    print(f'Output types: {len(output_types)}')
    print('='*80 + '\n')
    
    # Step 1: Fetch SNP information (real data)
    print('Step 1: Fetching SNP genomic coordinates...')
    snp_info_list = []
    for snp_id in patient_snps:
        print(f'  Fetching {snp_id}...')
        info = fetch_snp_info(snp_id)
        if info:
            snp_info_list.append(info)
            print(f'    → {info["chromosome"]}:{info["position"]} {info["reference"]}>{info["alternate"]} (Gene: {info["gene"]})')
        time.sleep(0.5)  # Be nice to the API
    
    print(f'\nSuccessfully retrieved {len(snp_info_list)}/{len(patient_snps)} SNPs\n')
    
    # Step 2: Mock AlphaGenome Analysis
    print('Step 2: Analyzing variants with AlphaGenome (MOCK)...')
    results = []
    
    for i, snp_info in enumerate(snp_info_list, 1):
        print(f'[{i}/{len(snp_info_list)}] Analyzing {snp_info["snp_id"]} (Gene: {snp_info["gene"]})...')
        
        # Generate mock results
        mock_result = mock_alphagenome_analysis(snp_info, output_types)
        results.append({
            'snp_info': snp_info,
            'alphagenome_result': mock_result,
        })
        
        print(f'  → {mock_result["interpretation"]["overall_effect_size"]} predicted effect')
    
    # Step 3: Results Summary
    print('\n' + '='*80)
    print('RESULTS SUMMARY (MOCK)')
    print('='*80 + '\n')
    
    for result in results:
        snp_info = result['snp_info']
        analysis = result['alphagenome_result']
        interp = analysis['interpretation']
        
        print(f"\n{snp_info['snp_id']} - {snp_info['gene']} ({snp_info['chromosome']}:{snp_info['position']})")
        print(f"  Reference: {snp_info['reference']} → Alternate: {snp_info['alternate']}")
        print(f"  Overall Effect: {interp['overall_effect_size']}")
        print(f"  Strongest Effects: {', '.join(interp['strongest_effects'])}")
        
        print("  Predicted Impacts:")
        for impact in interp['predicted_impact']:
            print(f"    • {impact}")
        
        print("  Scores by Output Type:")
        for score in analysis['scores']:
            print(f"    {score['output_type']}: {score['mean_score']:.4f} (across {score['num_tracks']} tracks)")
        
        print('\n' + '-'*80)
    
    # Save results
    output_file = 'patient_snp_alphagenome_results_mock.json'
    print(f'\nSaving mock results to {output_file}...')
    
    serializable_results = []
    for result in results:
        serializable_results.append({
            'snp_id': result['snp_info']['snp_id'],
            'gene': result['snp_info']['gene'],
            'chromosome': result['snp_info']['chromosome'],
            'position': result['snp_info']['position'],
            'reference': result['snp_info']['reference'],
            'alternate': result['snp_info']['alternate'],
            'analysis': result['alphagenome_result'],
        })
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f'✓ Mock results saved to {output_file}')
    print('\nMock analysis complete!')
    print('\nNOTE: These are simulated results for demonstration.')
    print('Once your API key is activated, run analyze_patient_snps.py for real results.')

if __name__ == '__main__':
    main()