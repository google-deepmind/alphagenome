"""Comprehensive unit tests for cache_utils module.

Tests cover:
- Basic cache operations (get/set)
- LRU eviction logic
- TTL expiration
- Size limit enforcement
- Metadata persistence
- Auto-migration
- Statistics tracking
- Edge cases and error handling
"""

import time
import pytest
from pathlib import Path
from alphagenome.cache_utils import (
    DiskCache,
    CacheConfig,
    CacheEntry,
    CacheStatistics,
    NoCache,
    make_key,
)


# ============================================================================
# Basic Cache Operations
# ============================================================================

def test_nocache_always_returns_none():
    """NoCache should always return None on get()."""
    cache = NoCache()
    cache.set("key", "value")
    assert cache.get("key") is None


def test_basic_set_get(basic_cache):
    """Test basic cache set and get operations."""
    basic_cache.set("test_key", "test_value")
    assert basic_cache.get("test_key") == "test_value"


def test_get_nonexistent_returns_none(basic_cache):
    """Getting a non-existent key should return None."""
    assert basic_cache.get("nonexistent") is None


def test_cache_complex_objects(basic_cache):
    """Test caching complex Python objects."""
    data = {
        'list': [1, 2, 3],
        'dict': {'a': 'b'},
        'tuple': (4, 5, 6),
        'nested': {'x': [1, 2, {'y': 3}]}
    }
    
    basic_cache.set("complex", data)
    retrieved = basic_cache.get("complex")
    
    assert retrieved == data
    assert isinstance(retrieved['list'], list)
    assert isinstance(retrieved['dict'], dict)


def test_overwrite_existing_key(basic_cache):
    """Overwriting an existing key should work."""
    basic_cache.set("key", "value1")
    basic_cache.set("key", "value2")
    assert basic_cache.get("key") == "value2"


# ============================================================================
# TTL (Time-To-Live) Tests
# ============================================================================

def test_ttl_expiration(limited_cache):
    """Entries should expire after TTL."""
    limited_cache.set("key1", "value1")
    
    # Should be available immediately
    assert limited_cache.get("key1") == "value1"
    
    # Wait for TTL to expire (2 seconds + buffer)
    time.sleep(2.5)
    
    # Should be expired now
    assert limited_cache.get("key1") is None


def test_ttl_not_expired_within_window(limited_cache):
    """Entries should NOT expire before TTL."""
    limited_cache.set("key1", "value1")
    
    # Wait less than TTL (1 second < 2 seconds TTL)
    time.sleep(1.0)
    
    # Should still be available
    assert limited_cache.get("key1") == "value1"


def test_ttl_cleanup_multiple_entries(temp_cache_dir):
    """Cleanup should remove all expired entries."""
    cache = DiskCache(
        cache_dir=str(temp_cache_dir),
        config=CacheConfig(ttl_seconds=1.0),
        log_level='DEBUG'
    )
    
    # Add multiple entries
    for i in range(5):
        cache.set(f"key{i}", f"value{i}")
    
    # All should be available
    assert cache.cache_stats()['total_entries'] == 5
    
    # Wait for expiration
    time.sleep(1.5)
    
    # Trigger cleanup
    cache.cleanup()
    
    # All should be removed
    stats = cache.cache_stats()
    assert stats['total_entries'] == 0


# ============================================================================
# Size Limit & LRU Eviction Tests
# ============================================================================

def test_size_limit_enforcement(limited_cache, sample_data):
    """Cache should enforce size limits via LRU eviction."""
    # limited_cache has 1 MB limit
    
    # Add entries totaling more than 1 MB
    # Each large entry is ~100 KB, so 15 entries = ~1.5 MB
    for i in range(15):
        limited_cache.set(f"key{i}", sample_data['large'])
    
    # Total size should not exceed limit significantly
    stats = limited_cache.cache_stats()
    assert stats['total_size_bytes'] <= 1024 * 1024 * 1.2  # Allow 20% overhead
    
    # Some evictions should have occurred
    assert stats['evictions'] > 0


def test_lru_eviction_order(temp_cache_dir, sample_data):
    """LRU should evict least recently accessed entries first."""
    cache = DiskCache(
        cache_dir=str(temp_cache_dir),
        config=CacheConfig(max_size_bytes=500_000),  # 500 KB
        log_level='DEBUG'
    )
    
    # Add 5 entries (~100 KB each = 500 KB total)
    for i in range(5):
        cache.set(f"key{i}", sample_data['large'])
        time.sleep(0.1)  # Ensure different timestamps
    
    # Access key0 and key1 to make them "recently used"
    cache.get("key0")
    cache.get("key1")
    time.sleep(0.1)
    
    # Add a new entry, forcing eviction
    cache.set("new_key", sample_data['large'])
    
    # key0 and key1 should still exist (recently used)
    assert cache.get("key0") == sample_data['large']
    assert cache.get("key1") == sample_data['large']
    
    # But some of key2, key3, key4 should be evicted
    evicted_count = sum([
        cache.get("key2") is None,
        cache.get("key3") is None,
        cache.get("key4") is None,
    ])
    assert evicted_count > 0


def test_no_eviction_without_size_limit(basic_cache, sample_data):
    """Without size limit, no evictions should occur."""
    # Add many large entries
    for i in range(20):
        basic_cache.set(f"key{i}", sample_data['large'])
    
    stats = basic_cache.cache_stats()
    assert stats['evictions'] == 0
    assert stats['total_entries'] == 20


# ============================================================================
# Statistics & Monitoring
# ============================================================================

def test_cache_statistics_tracking(basic_cache):
    """Cache should accurately track statistics."""
    # Initial stats
    stats = basic_cache.cache_stats()
    assert stats['total_entries'] == 0
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    
    # Add entries
    basic_cache.set("key1", "value1")
    basic_cache.set("key2", "value2")
    
    stats = basic_cache.cache_stats()
    assert stats['total_entries'] == 2
    
    # Cache hits
    basic_cache.get("key1")
    basic_cache.get("key1")
    
    stats = basic_cache.cache_stats()
    assert stats['hits'] == 2
    
    # Cache miss
    basic_cache.get("nonexistent")
    
    stats = basic_cache.cache_stats()
    assert stats['misses'] == 1
    assert stats['hit_rate'] == pytest.approx(66.67, rel=1e-1)  # 2/3 * 100


def test_cache_size_calculation(basic_cache, sample_data):
    """Cache should accurately calculate total size."""
    basic_cache.set("small", sample_data['small'])
    basic_cache.set("medium", sample_data['medium'])
    
    stats = basic_cache.cache_stats()
    assert stats['total_size_bytes'] > 0
    assert stats['total_size_mb'] > 0


def test_get_total_size(basic_cache, sample_data):
    """get_total_size() should return accurate size."""
    basic_cache.set("key1", sample_data['large'])
    
    total_size = basic_cache.get_total_size()
    assert total_size > 90_000  # ~100 KB, accounting for pickle overhead


# ============================================================================
# Cleanup Operations
# ============================================================================

def test_manual_cleanup(temp_cache_dir):
    """Manual cleanup should remove expired entries."""
    cache = DiskCache(
        cache_dir=str(temp_cache_dir),
        config=CacheConfig(ttl_seconds=1.0),
        log_level='DEBUG'
    )
    
    # Add entries
    for i in range(5):
        cache.set(f"key{i}", f"value{i}")
    
    # Wait for expiration
    time.sleep(1.5)
    
    # Manual cleanup
    result = cache.cleanup()
    
    assert result['expired_removed'] == 5
    assert result['total_cleaned'] == 5


def test_aggressive_cleanup(temp_cache_dir, sample_data):
    """Aggressive cleanup should enforce size limits."""
    cache = DiskCache(
        cache_dir=str(temp_cache_dir),
        config=CacheConfig(max_size_bytes=500_000),  # 500 KB
        log_level='DEBUG'
    )
    
    # Fill beyond threshold
    for i in range(10):
        cache.set(f"key{i}", sample_data['large'])
    
    # Aggressive cleanup
    result = cache.cleanup(aggressive=True)
    
    # Should have freed space
    final_size = cache.get_total_size()
    assert final_size < 500_000 * 0.9  # Below threshold


# ============================================================================
# Metadata Persistence
# ============================================================================

def test_metadata_persists_across_restarts(temp_cache_dir):
    """Metadata should persist when cache is recreated."""
    # Create cache and add entries
    cache1 = DiskCache(cache_dir=str(temp_cache_dir), log_level='DEBUG')
    cache1.set("key1", "value1")
    cache1.set("key2", "value2")
    
    stats1 = cache1.cache_stats()
    
    # Force metadata sync
    cache1._metadata.force_sync()
    
    # Create new cache instance (simulating restart)
    cache2 = DiskCache(cache_dir=str(temp_cache_dir), log_level='DEBUG')
    
    # Data should still be available
    assert cache2.get("key1") == "value1"
    assert cache2.get("key2") == "value2"
    
    # Stats should be preserved
    stats2 = cache2.cache_stats()
    assert stats2['total_entries'] == stats1['total_entries']


def test_auto_migration_of_existing_cache(temp_cache_dir):
    """Cache should auto-migrate existing .pkl files without metadata."""
    # Manually create a .pkl file (simulating old cache)
    import pickle
    cache_dir = Path(temp_cache_dir)
    old_file = cache_dir / "abc123.pkl"
    
    with open(old_file, 'wb') as f:
        pickle.dump("old_value", f)
    
    # Create DiskCache (should auto-migrate)
    cache = DiskCache(
        cache_dir=str(temp_cache_dir),
        config=CacheConfig(auto_migrate=True),
        log_level='DEBUG'
    )
    
    # Should have migrated the file
    stats = cache.cache_stats()
    assert stats['total_entries'] == 1
    assert stats['total_size_bytes'] > 0


# ============================================================================
# Error Handling & Edge Cases
# ============================================================================

def test_corrupt_cache_file_handling(basic_cache, temp_cache_dir):
    """Corrupt cache files should be handled gracefully."""
    # Set a valid entry
    basic_cache.set("key1", "value1")
    
    # Corrupt the file
    cache_dir = Path(temp_cache_dir)
    pkl_file = list(cache_dir.glob("*.pkl"))[0]
    
    with open(pkl_file, 'wb') as f:
        f.write(b"corrupt data!!!")
    
    # Should return None instead of crashing
    assert basic_cache.get("key1") is None


def test_concurrent_access_safety(basic_cache):
    """Cache should handle concurrent access safely."""
    import threading
    
    results = []
    errors = []
    
    def writer(i):
        try:
            for j in range(10):
                basic_cache.set(f"key_{i}_{j}", f"value_{i}_{j}")
        except Exception as e:
            errors.append(e)
    
    def reader(i):
        try:
            for j in range(10):
                basic_cache.get(f"key_{i}_{j}")
        except Exception as e:
            errors.append(e)
    
    # Create threads
    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=writer, args=(i,)))
        threads.append(threading.Thread(target=reader, args=(i,)))
    
    # Run all threads
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Should have no errors
    assert len(errors) == 0


def test_cache_with_none_values(basic_cache):
    """Cache should handle None values correctly."""
    basic_cache.set("null_key", None)
    
    # Should return None (cached value)
    result = basic_cache.get("null_key")
    assert result is None
    
    # But stats should show a hit, not miss
    stats = basic_cache.cache_stats()
    assert stats['hits'] == 1


# ============================================================================
# make_key() Function Tests
# ============================================================================

def test_make_key_deterministic():
    """make_key() should produce deterministic hashes."""
    key1 = make_key("func", (1, 2), {"a": "b"})
    key2 = make_key("func", (1, 2), {"a": "b"})
    
    assert key1 == key2


def test_make_key_different_args():
    """make_key() should produce different keys for different args."""
    key1 = make_key("func", (1, 2), {"a": "b"})
    key2 = make_key("func", (1, 3), {"a": "b"})
    
    assert key1 != key2


def test_make_key_kwargs_order_independent():
    """make_key() should be independent of kwargs order."""
    key1 = make_key("func", (), {"a": 1, "b": 2})
    key2 = make_key("func", (), {"b": 2, "a": 1})
    
    assert key1 == key2


# ============================================================================
# Performance Tests
# ============================================================================

def test_large_number_of_entries(temp_cache_dir):
    """Cache should handle large numbers of entries efficiently."""
    cache = DiskCache(
        cache_dir=str(temp_cache_dir),
        log_level='WARNING'  # Reduce log noise
    )
    
    # Add 1000 small entries
    import time
    start = time.time()
    
    for i in range(1000):
        cache.set(f"key{i}", f"value{i}")
    
    elapsed = time.time() - start
    
    # Should complete in reasonable time (<5 seconds)
    assert elapsed < 5.0
    
    # All entries should be retrievable
    assert cache.get("key0") == "value0"
    assert cache.get("key999") == "value999"


def test_metadata_sync_interval(temp_cache_dir):
    """Metadata should sync periodically, not on every operation."""
    cache = DiskCache(
        cache_dir=str(temp_cache_dir),
        config=CacheConfig(metadata_sync_interval=10),
        log_level='DEBUG'
    )
    
    # Add 5 entries (less than sync interval)
    for i in range(5):
        cache.set(f"key{i}", f"value{i}")
    
    # Metadata should not have synced yet (operations_since_sync < 10)
    assert cache._metadata._operations_since_sync < 10
    
    # Add 5 more (total 10, should trigger sync)
    for i in range(5, 10):
        cache.set(f"key{i}", f"value{i}")
    
    # Should have synced now
    assert cache._metadata._operations_since_sync < 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
