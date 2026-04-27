import os
from alphagenome.data import genome
from alphagenome.models import dna_client

def test_api_connection():
    # Get API key from environment
    api_key = os.getenv('ALPHAGENOME_API_KEY')
    
    if not api_key:
        print("❌ ALPHAGENOME_API_KEY environment variable not set")
        return False
        
    print(f"✅ API key found: {api_key[:10]}...{api_key[-5:]}")
    
    # Test creating a client
    try:
        model = dna_client.create(api_key)
        print("✅ DNA client created successfully")
    except Exception as e:
        print(f"❌ Failed to create DNA client: {e}")
        return False
    
    # Test a very small interval first (16KB)
    try:
        interval = genome.Interval(chromosome='chr22', start=35677410, end=35677410 + 16384)
        print(f"✅ Test interval created: {interval} (width: {interval.width} bp)")
        
        # Try to make a prediction
        print("🔄 Testing prediction with small interval...")
        outputs = model.predict_interval(
            interval=interval,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=['UBERON:0001157']  # Brain tissue
        )
        print("✅ Prediction successful!")
        print(f"Output type: {type(outputs)}")
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing AlphaGenome API Connection...")
    print("=" * 50)
    success = test_api_connection()
    print("=" * 50)
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed. Check your API key and network connection.")