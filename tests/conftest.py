"""Pytest configuration and shared fixtures for AlphaGenome tests."""

import tempfile
import shutil
from pathlib import Path
import pytest
from alphagenome.cache_utils import DiskCache, CacheConfig


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def basic_cache(temp_cache_dir):
    """Create a basic DiskCache instance for testing."""
    return DiskCache(
        cache_dir=str(temp_cache_dir),
        log_level='DEBUG'
    )


@pytest.fixture
def limited_cache(temp_cache_dir):
    """Create a DiskCache with size and TTL limits."""
    return DiskCache(
        cache_dir=str(temp_cache_dir),
        config=CacheConfig(
            max_size_bytes=1024 * 1024,  # 1 MB
            ttl_seconds=2.0,  # 2 seconds
        ),
        log_level='DEBUG'
    )


@pytest.fixture
def sample_data():
    """Generate sample data for caching tests."""
    return {
        'small': 'x' * 100,  # 100 bytes
        'medium': 'y' * 10_000,  # ~10 KB
        'large': 'z' * 100_000,  # ~100 KB  
    }
