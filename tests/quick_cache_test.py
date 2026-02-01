"""Quick test script to verify the enhanced DiskCache implementation."""

import time
import tempfile
import shutil
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alphagenome.cache_utils import DiskCache, CacheConfig


def test_basic_operations():
    """Test basic cache operations."""
    print("=" * 70)
    print("TEST 1: Basic Operations")
    print("=" * 70)
    
    temp_dir = tempfile.mkdtemp()
    try:
        cache = DiskCache(cache_dir=temp_dir, log_level='INFO')
        
        # Set and get
        cache.set("key1", "value1")
        result = cache.get("key1")
        
        assert result == "value1", f"Expected 'value1', got {result}"
        print("✓ Basic set/get works")
        
        # Stats
        stats = cache.cache_stats()
        print(f"✓ Stats: {stats['total_entries']} entries, {stats['hits']} hits")
        
        assert stats['total_entries'] == 1
        assert stats['hits'] == 1
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ TEST 1 PASSED\n")


def test_ttl_expiration():
    """Test TTL expiration."""
    print("=" * 70)
    print("TEST 2: TTL Expiration")
    print("=" * 70)
    
    temp_dir = tempfile.mkdtemp()
    try:
        cache = DiskCache(
            cache_dir=temp_dir,
            config=CacheConfig(ttl_seconds=1.0),
            log_level='INFO'
        )
        
        cache.set("key1", "value1")
        print("✓ Entry added with 1 second TTL")
        
        # Should be available immediately
        result = cache.get("key1")
        assert result == "value1"
        print("✓ Entry available immediately")
        
        # Wait for expiration
        print("  Waiting 1.5 seconds for expiration...")
        time.sleep(1.5)
        
        # Should be expired
        result = cache.get("key1")
        assert result is None, f"Expected None (expired), got {result}"
        print("✓ Entry expired after TTL")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ TEST 2 PASSED\n")


def test_size_limit_and_lru():
    """Test size limit enforcement with LRU eviction."""
    print("=" * 70)
    print("TEST 3: Size Limit & LRU Eviction")
    print("=" * 70)
    
    temp_dir = tempfile.mkdtemp()
    try:
        cache = DiskCache(
            cache_dir=temp_dir,
            config=CacheConfig(max_size_bytes=100_000),  # 100 KB
            log_level='INFO'
        )
        
        # Add multiple large entries
        large_data = "x" * 50_000  # ~50 KB each
        
        for i in range(5):  # 250 KB total, exceeds limit
            cache.set(f"key{i}", large_data)
            print(f"  Added entry {i}")
        
        stats = cache.cache_stats()
        print(f"✓ Final size: {stats['total_size_bytes'] / 1024:.2f} KB")
        print(f"✓ Total entries: {stats['total_entries']}")
        print(f"✓ Evictions: {stats['evictions']}")
        
        # Size should not exceed limit significantly
        assert stats['total_size_bytes'] <= 120_000, "Size limit not enforced"
        
        # Some evictions should have occurred
        assert stats['evictions'] > 0, "No evictions occurred"
        
        print("✓ Size limit enforced via LRU eviction")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ TEST 3 PASSED\n")


def test_metadata_persistence():
    """Test metadata persistence across restarts."""
    print("=" * 70)
    print("TEST 4: Metadata Persistence")
    print("=" * 70)
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create cache and add data
        cache1 = DiskCache(cache_dir=temp_dir, log_level='INFO')
        cache1.set("key1", "value1")
        cache1.set("key2", "value2")
        
        stats1 = cache1.cache_stats()
        print(f"✓ Cache 1: {stats1['total_entries']} entries")
        
        # Force sync
        cache1._metadata.force_sync()
        print("✓ Metadata synced to disk")
        
        # Create new cache instance (simulating restart)
        cache2 = DiskCache(cache_dir=temp_dir, log_level='INFO')
        
        # Data should still be available
        assert cache2.get("key1") == "value1"
        assert cache2.get("key2") == "value2"
        
        stats2 = cache2.cache_stats()
        print(f"✓ Cache 2: {stats2['total_entries']} entries")
        
        assert stats2['total_entries'] == stats1['total_entries']
        print("✓ Metadata persisted across restart")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ TEST 4 PASSED\n")


def test_auto_migration():
    """Test auto-migration of existing cache files."""
    print("=" * 70)
    print("TEST 5: Auto-Migration")
    print("=" * 70)
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Manually create a .pkl file (simulating old cache)
        import pickle
        old_file = temp_dir / "abc123.pkl"
        
        with open(old_file, 'wb') as f:
            pickle.dump("old_value", f)
        
        print("✓ Created old cache file without metadata")
        
        # Create DiskCache (should auto-migrate)
        cache = DiskCache(
            cache_dir=str(temp_dir),
            config=CacheConfig(auto_migrate=True),
            log_level='INFO'
        )
        
        stats = cache.cache_stats()
        print(f"✓ Migrated: {stats['total_entries']} entries")
        
        assert stats['total_entries'] == 1
        assert stats['total_size_bytes'] > 0
        
        print("✓ Auto-migration successful")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("✅ TEST 5 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Enhanced DiskCache - Quick Validation Tests")
    print("=" * 70 + "\n")
    
    tests = [
        test_basic_operations,
        test_ttl_expiration,
        test_size_limit_and_lru,
        test_metadata_persistence,
        test_auto_migration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {e}\n")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
