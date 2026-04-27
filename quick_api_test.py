#!/usr/bin/env python3
"""Quick test to verify the API key is working."""

from alphagenome.models import dna_client

# Your working API key
my_api_key = "AlzaSyCy8DOGotXZ93omnBh8hGf8jlJd7eLaLO4"
alphagenome_model = dna_client.create(my_api_key)

print("✓ Client created successfully!")

try:
    # Test a simple API call
    print("Testing API call...")
    metadata = alphagenome_model.output_metadata()
    print(f"✓ API is working! Found {len(metadata.output_types)} output types:")
    for output_type in metadata.output_types[:3]:  # Show first 3
        print(f"  - {output_type.name}")
    print("✓ Ready to run full SNP analysis!")
    
except Exception as e:
    print(f"✗ API call failed: {e}")