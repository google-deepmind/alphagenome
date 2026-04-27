"""Script to acquire predictions for the 17p11.2 chromosomal locus.

The 17p11.2 region is clinically significant - associated with Smith-Magenis
syndrome (deletion) and Potocki-Lupski syndrome (duplication).

The region spans approximately chr17:16,700,000-24,100,000 (hg38).
This corresponds to the hg19 coordinates chr17:14,890,000-22,230,032.
"""

import os
import pickle
import numpy as np
from alphagenome.data import genome
from alphagenome.models import dna_client

# Define the 17p11.2 locus coordinates (hg38 assembly)
# This is the common deletion/duplication region (~7.4 Mb)
# Converted from hg19: chr17:14,890,000-22,230,032
LOCUS_17P11_2_START = 16_700_000  # hg38 coordinates
LOCUS_17P11_2_END = 24_100_000    # hg38 coordinates

# For AlphaGenome, we need to work with supported sequence lengths
# The locus is ~7.4 Mb, so we'll need to break it into chunks
# Available lengths: 16KB, 100KB, 500KB, 1MB
CHUNK_SIZE = dna_client.SEQUENCE_LENGTH_500KB  # 524,288 bp


def main():
  # Get API key from environment
  api_key = os.getenv('ALPHAGENOME_API_KEY')
  if not api_key:
    raise ValueError(
        'ALPHAGENOME_API_KEY environment variable not set. '
        'Please set it with your API key.'
    )

  # Create the model client
  print('Connecting to AlphaGenome API...')
  model = dna_client.create(api_key)

  # Define the full locus
  full_locus = genome.Interval(
      chromosome='chr17',
      start=LOCUS_17P11_2_START,
      end=LOCUS_17P11_2_END,
  )

  print(f'Target locus (hg38): {full_locus}')
  print(f'Locus width: {full_locus.width:,} bp')
  print(f'Chunk size: {CHUNK_SIZE:,} bp')

  # Break the locus into chunks
  chunks = []
  current_pos = full_locus.start
  # Supported sequence lengths
  supported_lengths = [16384, 131072, 524288, 1048576]
  
  while current_pos < full_locus.end:
    chunk_end = min(current_pos + CHUNK_SIZE, full_locus.end)
    chunk = genome.Interval(
        chromosome='chr17',
        start=current_pos,
        end=chunk_end,
    )
    # Only add if it matches a supported length
    if chunk.width in supported_lengths:
      chunks.append(chunk)
    current_pos = chunk_end

  print(f'\nBreaking locus into {len(chunks)} chunk(s)')

  # Define what predictions we want
  # For genomic characterization, common outputs include:
  requested_outputs = [
      dna_client.OutputType.RNA_SEQ,  # Gene expression
      dna_client.OutputType.ATAC,  # Chromatin accessibility
      dna_client.OutputType.DNASE,  # DNase hypersensitivity
  ]

  # Specify tissue/cell type context - using multiple relevant tissues
  # 17p11.2 deletions affect neurodevelopment and multiple systems
  # Note: Using only confirmed supported ontology terms
  ontology_terms = [
      'UBERON:0000955',  # Brain (confirmed supported)
      'UBERON:0002048',  # Lung (respiratory system)
      'UBERON:0002107',  # Liver (metabolic effects)
      'UBERON:0002113',  # Kidney (renal system)
  ]

  # Process each chunk
  outputs = []
  for i, chunk in enumerate(chunks):
    print(f'\nProcessing chunk {i+1}/{len(chunks)}: {chunk}')

    try:
      output = model.predict_interval(
          interval=chunk,
          requested_outputs=requested_outputs,
          ontology_terms=ontology_terms,
      )
      outputs.append(output)
      print(f'  ✓ Successfully retrieved predictions')

      # Show some basic info about the output
      if output.rna_seq is not None:
        print(f'  RNA-seq tracks: {output.rna_seq.values.shape}')
      if output.atac is not None:
        print(f'  ATAC tracks: {output.atac.values.shape}')
      if output.dnase is not None:
        print(f'  DNase tracks: {output.dnase.values.shape}')

    except Exception as e:
      print(f'  ✗ Error processing chunk: {e}')
      continue

  print(f'\n\nSuccessfully processed {len(outputs)}/{len(chunks)} chunks')

  # Save outputs to disk
  output_dir = 'chr17_17p11_2_outputs_hg38'
  os.makedirs(output_dir, exist_ok=True)
  
  print(f'\nSaving outputs to {output_dir}/ directory...')
  
  # Save the complete outputs object
  with open(f'{output_dir}/complete_outputs.pkl', 'wb') as f:
    pickle.dump(outputs, f)
  print(f'  ✓ Saved complete outputs: {output_dir}/complete_outputs.pkl')
  
  # Save individual track data as numpy arrays
  for i, (chunk, output) in enumerate(zip(chunks[:len(outputs)], outputs)):
    chunk_name = f'chunk_{i+1:02d}_chr17_{chunk.start}_{chunk.end}'
    
    if output.rna_seq is not None:
      np.save(f'{output_dir}/{chunk_name}_rna_seq.npy', output.rna_seq.values)
      print(f'  ✓ Saved RNA-seq: {output_dir}/{chunk_name}_rna_seq.npy')
    
    if output.atac is not None and output.atac.values.shape[1] > 0:
      np.save(f'{output_dir}/{chunk_name}_atac.npy', output.atac.values)
      print(f'  ✓ Saved ATAC: {output_dir}/{chunk_name}_atac.npy')
    
    if output.dnase is not None:
      np.save(f'{output_dir}/{chunk_name}_dnase.npy', output.dnase.values)
      print(f'  ✓ Saved DNase: {output_dir}/{chunk_name}_dnase.npy')
  
  # Create a summary file
  with open(f'{output_dir}/README.txt', 'w') as f:
    f.write(f'17p11.2 Locus Analysis Results (hg38)\n')
    f.write(f'=====================================\n\n')
    f.write(f'Target locus (hg38): {full_locus}\n')
    f.write(f'Equivalent hg19: chr17:14,890,000-22,230,032\n')
    f.write(f'Total chunks processed: {len(outputs)}\n')
    f.write(f'Chunk size: {CHUNK_SIZE:,} bp\n\n')
    f.write(f'Files generated:\n')
    f.write(f'- complete_outputs.pkl: Full AlphaGenome output objects\n')
    f.write(f'- *_rna_seq.npy: RNA-seq prediction arrays\n')
    f.write(f'- *_dnase.npy: DNase hypersensitivity arrays\n')
    f.write(f'- *_atac.npy: ATAC-seq arrays (if available)\n\n')
    f.write(f'Load numpy arrays with: np.load("filename.npy")\n')
    f.write(f'Load complete outputs with: pickle.load(open("complete_outputs.pkl", "rb"))\n')
  
  print(f'  ✓ Created summary: {output_dir}/README.txt')
  print(f'\nAll outputs saved! Check the {output_dir}/ directory.')

  return outputs


if __name__ == '__main__':
  outputs = main()