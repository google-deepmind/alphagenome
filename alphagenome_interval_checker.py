import os
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.visualization import plot_components

# Get API key from environment variable
API_KEY = os.getenv('ALPHAGENOME_API_KEY')
if not API_KEY:
    raise ValueError("Please set the ALPHAGENOME_API_KEY environment variable")
    
model = dna_client.create(API_KEY)

# Define the genomic interval - using 1MB (1,048,576 bp) which is supported
# Chr17p11.2 region around the SMS/PTLS region
start_pos = 17500000
interval = genome.Interval(chromosome='chr17', start=start_pos, end=start_pos + 1048576)

# Verify the interval size
print(f"Interval: {interval}")
print(f"Interval width: {interval.width} bp")
print(f"Supported lengths: [16384, 131072, 524288, 1048576]")
print(f"Using supported length: 1048576 (1MB)")
print()

# Call predict_interval with required arguments
outputs = model.predict_interval(
    interval=interval,
    requested_outputs=[dna_client.OutputType.RNA_SEQ],
    ontology_terms=['UBERON:0001157']  # Brain tissue
)

print("Prediction completed successfully!")