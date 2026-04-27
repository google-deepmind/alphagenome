#!/usr/bin/env python3
"""Simple test script to validate the AlphaGenome API key."""

import os
from alphagenome.models import dna_client

def test_api_key():
    """Test if the API key is valid by attempting to create a client."""
    
    # Get API key from environment
    api_key = os.getenv('ALPHA_GENOME_API_KEY')
    if not api_key:
        print('ERROR: ALPHA_GENOME_API_KEY environment variable not set')
        return False
    
    print(f'Testing API key (length: {len(api_key)})...')
    print(f'First 8 characters: {api_key[:8]}...')
    
    try:
        # Try to create client
        model = dna_client.create(api_key)
        print('✓ API key appears to be valid - client created successfully')
        
        # Try to get metadata (simple API call)
        print('Testing basic API call (get metadata)...')
        metadata = model.output_metadata()
        print(f'✓ API call successful - got {len(metadata.output_types)} output types')
        return True
        
    except Exception as e:
        print(f'✗ API key test failed: {e}')
        return False

if __name__ == '__main__':
    test_api_key()