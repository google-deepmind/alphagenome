#!/usr/bin/env python3
"""Test network connectivity to AlphaGenome service."""

import grpc
import os

def test_connection():
    """Test basic connectivity to the AlphaGenome service."""
    
    api_key = os.getenv('ALPHA_GENOME_API_KEY')
    print(f'Using API key: {api_key[:10]}... (length: {len(api_key)})')
    
    # Test basic gRPC connection
    address = 'dns:///gdmscience.googleapis.com:443'
    options = [
        ('grpc.max_send_message_length', -1),
        ('grpc.max_receive_message_length', -1),
    ]
    
    try:
        print(f'Connecting to {address}...')
        channel = grpc.secure_channel(address, grpc.ssl_channel_credentials(), options=options)
        
        print('Testing channel readiness...')
        grpc.channel_ready_future(channel).result(timeout=10)
        print('✓ Channel connection successful')
        
        # Test basic metadata
        metadata = [('x-goog-api-key', api_key)]
        print(f'Metadata: {[("x-goog-api-key", api_key[:10] + "...")]}')
        
        channel.close()
        print('✓ Connection test completed')
        return True
        
    except Exception as e:
        print(f'✗ Connection failed: {e}')
        return False

if __name__ == '__main__':
    test_connection()