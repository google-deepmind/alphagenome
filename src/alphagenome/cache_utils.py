"""Module containing caching utilities for AlphaGenome.

This module provides production-ready caching with LRU eviction, TTL support,
size limits, and comprehensive monitoring capabilities.
"""

import abc
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

T = TypeVar('T')


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CacheEntry:
  """Metadata for a single cache entry."""
  file: str
  size_bytes: int
  created_at: float
  last_accessed: float
  access_count: int = 0


@dataclass
class CacheStatistics:
  """Statistics for cache performance monitoring."""
  total_size_bytes: int
  total_entries: int
  hits: int
  misses: int
  evictions: int
  
  @property
  def hit_rate(self) -> float:
    """Calculate cache hit rate."""
    total = self.hits + self.misses
    return (self.hits / total * 100) if total > 0 else 0.0


@dataclass
class CacheConfig:
  """Configuration for DiskCache behavior."""
  
  # Size Management
  max_size_bytes: Optional[int] = None  # None = unlimited
  warn_on_unlimited: bool = True        # Warn if no limit set
  
  # TTL Management  
  ttl_seconds: Optional[float] = None   # None = no expiration
  
  # Eviction
  eviction_policy: Literal['lru'] = 'lru'  # Only LRU for now
  
  # Metadata
  auto_migrate: bool = True             # Auto-migrate old caches
  metadata_file: str = '.cache_metadata.json'
  
  # Cleanup
  cleanup_on_get: bool = True          # Clean expired on each get()
  cleanup_threshold: float = 0.9       # Trigger cleanup at 90% full
  
  # Performance
  metadata_sync_interval: int = 10     # Sync to disk every N operations


# ============================================================================
# Metadata Manager
# ============================================================================

class MetadataManager:
  """Manages cache metadata with in-memory caching for performance.
  
  This class handles:
  - Loading/saving metadata from/to disk
  - In-memory metadata cache to avoid repeated JSON reads
  - Atomic metadata updates
  - Auto-migration of existing cache files
  """
  
  def __init__(
      self,
      cache_dir: Path,
      config: CacheConfig,
      logger: logging.Logger,
  ):
    self._cache_dir = cache_dir
    self._config = config
    self._logger = logger
    self._metadata_path = cache_dir / config.metadata_file
    self._lock = threading.Lock()
    
    # In-memory cache for performance
    self._entries: dict[str, CacheEntry] = {}
    self._stats = CacheStatistics(
        total_size_bytes=0,
        total_entries=0,
        hits=0,
        misses=0,
        evictions=0,
    )
    self._operations_since_sync = 0
    
    # Initialize
    self._load_or_migrate()
  
  def _load_or_migrate(self) -> None:
    """Load existing metadata or migrate from old cache."""
    if self._metadata_path.exists():
      self._load_metadata()
    elif self._config.auto_migrate:
      self._logger.info('No metadata found, migrating existing cache...')
      self._migrate_existing_cache()
    else:
      self._logger.info('Initializing new cache metadata')
      self._save_metadata()  # Create empty metadata file
  
  def _load_metadata(self) -> None:
    """Load metadata from disk into memory."""
    try:
      with self._lock:
        with open(self._metadata_path, 'r') as f:
          data = json.load(f)
        
        # Load entries
        self._entries = {
            key: CacheEntry(**entry_data)
            for key, entry_data in data.get('entries', {}).items()
        }
        
        # Load statistics
        stats_data = data.get('stats', {})
        self._stats = CacheStatistics(**stats_data)
        
        self._logger.debug(
            f'Loaded metadata: {self._stats.total_entries} entries, '
            f'{self._stats.total_size_bytes / 1024**2:.2f} MB'
        )
    except (json.JSONDecodeError, IOError, KeyError) as e:
      self._logger.warning(f'Failed to load metadata: {e}, recreating...')
      self._migrate_existing_cache()
  
  def _migrate_existing_cache(self) -> None:
    """Scan existing .pkl files and create metadata."""
    self._logger.info('Scanning for existing cache files...')
    migrated_count = 0
    total_size = 0
    
    for pkl_file in self._cache_dir.glob('*.pkl'):
      try:
        key = pkl_file.stem  # Filename without extension
        stat = pkl_file.stat()
        
        entry = CacheEntry(
            file=pkl_file.name,
            size_bytes=stat.st_size,
            created_at=stat.st_ctime,
            last_accessed=stat.st_mtime,
            access_count=0,  # Unknown for old files
        )
        
        self._entries[key] = entry
        total_size += stat.st_size
        migrated_count += 1
        
      except Exception as e:
        self._logger.warning(f'Failed to migrate {pkl_file}: {e}')
    
    self._stats.total_entries = migrated_count
    self._stats.total_size_bytes = total_size
    
    self._logger.info(
        f'Migrated {migrated_count} cache files '
        f'({total_size / 1024**2:.2f} MB)'
    )
    
    self._save_metadata()
  
  def _save_metadata(self) -> None:
    """Atomically save metadata to disk."""
    try:
      data = {
          'entries': {
              key: asdict(entry)
              for key, entry in self._entries.items()
          },
          'stats': asdict(self._stats),
      }
      
      # Atomic write: temp file + rename
      temp_path = self._metadata_path.with_suffix('.tmp')
      with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
      
      temp_path.replace(self._metadata_path)
      
    except Exception as e:
      self._logger.error(f'Failed to save metadata: {e}')
  
  def _maybe_sync(self) -> None:
    """Sync metadata to disk if needed (based on operation count)."""
    self._operations_since_sync += 1
    if self._operations_since_sync >= self._config.metadata_sync_interval:
      self._save_metadata()
      self._operations_since_sync = 0
  
  def get_entry(self, key: str) -> Optional[CacheEntry]:
    """Get metadata for a cache entry."""
    with self._lock:
      return self._entries.get(key)
  
  def update_access(self, key: str) -> None:
    """Update access time and count for an entry."""
    with self._lock:
      if key in self._entries:
        entry = self._entries[key]
        entry.last_accessed = time.time()
        entry.access_count += 1
        self._stats.hits += 1
      else:
        self._stats.misses += 1
      
      self._maybe_sync()
  
  def add_entry(self, key: str, file_path: Path, size_bytes: int) -> None:
    """Add a new cache entry."""
    with self._lock:
      current_time = time.time()
      entry = CacheEntry(
          file=file_path.name,
          size_bytes=size_bytes,
          created_at=current_time,
          last_accessed=current_time,
          access_count=0,
      )
      
      self._entries[key] = entry
      self._stats.total_entries += 1
      self._stats.total_size_bytes += size_bytes
      
      self._maybe_sync()
  
  def remove_entry(self, key: str) -> None:
    """Remove an entry from metadata."""
    with self._lock:
      if key in self._entries:
        entry = self._entries[key]
        self._stats.total_entries -= 1
        self._stats.total_size_bytes -= entry.size_bytes
        del self._entries[key]
        
        self._maybe_sync()
  
  def record_eviction(self) -> None:
    """Record an eviction event."""
    with self._lock:
      self._stats.evictions += 1
  
  def get_statistics(self) -> CacheStatistics:
    """Get current cache statistics."""
    with self._lock:
      return CacheStatistics(**asdict(self._stats))
  
  def get_total_size(self) -> int:
    """Get total cache size in bytes."""
    with self._lock:
      return self._stats.total_size_bytes
  
  def get_lru_victims(self, count: int) -> list[str]:
    """Get N least recently used cache keys for eviction."""
    with self._lock:
      # Sort by last_accessed (ascending)
      sorted_entries = sorted(
          self._entries.items(),
          key=lambda x: x[1].last_accessed
      )
      return [key for key, _ in sorted_entries[:count]]
  
  def get_expired_keys(self, ttl_seconds: float) -> list[str]:
    """Get all expired cache keys based on TTL."""
    current_time = time.time()
    with self._lock:
      expired = []
      for key, entry in self._entries.items():
        age = current_time - entry.created_at
        if age > ttl_seconds:
          expired.append(key)
      return expired
  
  def force_sync(self) -> None:
    """Force immediate sync to disk."""
    with self._lock:
      self._save_metadata()
      self._operations_since_sync = 0


# ============================================================================
# Cache Abstract Base Class
# ============================================================================

class Cache(abc.ABC):
  """Abstract base class for caches."""

  @abc.abstractmethod
  def get(self, key: str) -> Any | None:
    """Retrieve an item from the cache."""

  @abc.abstractmethod
  def set(self, key: str, value: Any) -> None:
    """Store an item in the cache."""


# ============================================================================
# NoCache Implementation
# ============================================================================

class NoCache(Cache):
  """A dummy cache that does nothing."""

  def get(self, key: str) -> Any | None:
    return None

  def set(self, key: str, value: Any) -> None:
    pass


# ============================================================================
# Production-Ready DiskCache
# ============================================================================

class DiskCache(Cache):
  """Production-ready file-based cache with LRU eviction, TTL, and size limits.

  Features:
  - Thread-safe operations with file locking
  - LRU (Least Recently Used) eviction policy
  - Configurable TTL (Time-To-Live) for cache entries
  - Maximum cache size enforcement
  - Comprehensive statistics tracking
  - Automatic migration of existing caches
  - In-memory metadata caching for performance

  Example:
      # Basic usage (backward compatible)
      cache = DiskCache(cache_dir="./cache")
      
      # Production usage with limits
      cache = DiskCache(
          cache_dir="./cache",
          config=CacheConfig(
              max_size_bytes=10 * 1024**3,  # 10 GB
              ttl_seconds=7 * 24 * 3600,    # 7 days
          )
      )
  """

  def __init__(
      self,
      cache_dir: str,
      config: Optional[CacheConfig] = None,
      log_level: str = 'WARNING',
  ):
    """Initialize DiskCache with optional configuration.
    
    Args:
        cache_dir: Directory to store cache files
        config: Optional CacheConfig for advanced settings
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    self._cache_dir = Path(cache_dir)
    self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    self._config = config or CacheConfig()
    self._lock = threading.Lock()
    
    # Setup logging
    self._logger = logging.getLogger(f'{__name__}.DiskCache')
    self._logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
    if not self._logger.handlers:
      handler = logging.StreamHandler()
      formatter = logging.Formatter(
          '[%(asctime)s] [%(levelname)s] %(message)s',
          datefmt='%Y-%m-%d %H:%M:%S'
      )
      handler.setFormatter(formatter)
      self._logger.addHandler(handler)
    
    # Initialize metadata manager
    self._metadata = MetadataManager(
        self._cache_dir,
        self._config,
        self._logger,
    )
    
    # Warn about unlimited cache
    if self._config.max_size_bytes is None and self._config.warn_on_unlimited:
      self._logger.warning(
          'DiskCache initialized without size limit. '
          'Consider setting max_size_bytes for production use. '
          'Recommended: 10GB (10*1024**3)'
      )
    
    self._logger.info(
        f'DiskCache initialized at {cache_dir} '
        f'(max_size: {self._format_bytes(self._config.max_size_bytes)}, '
        f'ttl: {self._format_ttl(self._config.ttl_seconds)})'
    )

  def _format_bytes(self, bytes_val: Optional[int]) -> str:
    """Format bytes for human-readable display."""
    if bytes_val is None:
      return 'unlimited'
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
      if bytes_val < 1024:
        return f'{bytes_val:.2f} {unit}'
      bytes_val /= 1024
    return f'{bytes_val:.2f} PB'
  
  def _format_ttl(self, seconds: Optional[float]) -> str:
    """Format TTL for human-readable display."""
    if seconds is None:
      return 'no expiration'
    
    days = seconds / (24 * 3600)
    if days >= 1:
      return f'{days:.1f} days'
    
    hours = seconds / 3600
    if hours >= 1:
      return f'{hours:.1f} hours'
    
    return f'{seconds:.0f} seconds'

  def _get_path(self, key: str) -> Path:
    """Get file path for a cache key."""
    # Use SHA256 of the key to ensure valid filename
    hashed_key = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return self._cache_dir / f'{hashed_key}.pkl'

  def _is_expired(self, entry: CacheEntry) -> bool:
    """Check if an entry has expired based on TTL."""
    if self._config.ttl_seconds is None:
      return False
    
    age = time.time() - entry.created_at
    return age > self._config.ttl_seconds

  def _cleanup_expired(self) -> int:
    """Remove all expired entries. Returns count of removed entries."""
    if self._config.ttl_seconds is None:
      return 0
    
    expired_keys = self._metadata.get_expired_keys(self._config.ttl_seconds)
    
    for key in expired_keys:
      self._remove_entry(key)
      self._logger.debug(f'Removed expired entry: {key}')
    
    if expired_keys:
      self._logger.info(f'Cleaned up {len(expired_keys)} expired entries')
    
    return len(expired_keys)

  def _evict_lru(self, required_space: int) -> None:
    """Evict least recently used entries to free up space."""
    current_size = self._metadata.get_total_size()
    max_size = self._config.max_size_bytes
    
    if max_size is None:
      return  # No size limit
    
    # Calculate how much space we need to free
    space_to_free = (current_size + required_space) - max_size
    
    if space_to_free <= 0:
      return  # Enough space available
    
    self._logger.info(
        f'Cache full, evicting LRU entries to free {space_to_free / 1024**2:.2f} MB'
    )
    
    freed_space = 0
    victims = self._metadata.get_lru_victims(count=100)  # Process in batches
    
    for key in victims:
      if freed_space >= space_to_free:
        break
      
      entry = self._metadata.get_entry(key)
      if entry:
        freed_space += entry.size_bytes
        self._remove_entry(key)
        self._metadata.record_eviction()
        self._logger.debug(f'Evicted LRU entry: {key}')
    
    self._logger.info(
        f'Evicted {len(victims)} entries, freed {freed_space / 1024**2:.2f} MB'
    )

  def _remove_entry(self, key: str) -> None:
    """Remove a cache entry (file and metadata)."""
    file_path = self._get_path(key)
    
    try:
      if file_path.exists():
        file_path.unlink()
    except Exception as e:
      self._logger.warning(f'Failed to delete cache file {file_path}: {e}')
    
    self._metadata.remove_entry(key)

  def get(self, key: str) -> Any | None:
    """Retrieve an item from the cache.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None if not found/expired
    """
    # Cleanup expired entries periodically
    if self._config.cleanup_on_get and self._config.ttl_seconds:
      self._cleanup_expired()
    
    path = self._get_path(key)
    
    # Check if file exists
    if not path.exists():
      self._metadata.update_access(key)  # Record miss
      return None
    
    # Check metadata for expiration
    entry = self._metadata.get_entry(key)
    if entry and self._is_expired(entry):
      self._logger.debug(f'Cache entry expired: {key}')
      self._remove_entry(key)
      return None
    
    # Load from disk
    try:
      with self._lock:
        with open(path, 'rb') as f:
          value = pickle.load(f)
      
      # Update access time
      self._metadata.update_access(key)
      self._logger.debug(f'Cache HIT: {key}')
      
      return value
      
    except (EOFError, pickle.UnpicklingError) as e:
      self._logger.warning(f'Corrupt cache file {key}: {e}')
      self._remove_entry(key)
      return None
    except Exception as e:
      self._logger.error(f'Failed to read cache {key}: {e}')
      return None

  def set(self, key: str, value: Any) -> None:
    """Store an item in the cache.
    
    Args:
        key: Cache key
        value: Value to cache (must be picklable)
    """
    path = self._get_path(key)
    
    try:
      with self._lock:
        # Write to temp file (atomic write)
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
          pickle.dump(value, f)
        
        # Get file size
        size_bytes = temp_path.stat().st_size
        
        # Evict if necessary
        if self._config.max_size_bytes:
          self._evict_lru(required_space=size_bytes)
        
        # Atomic rename
        temp_path.replace(path)
      
      # Update metadata
      self._metadata.add_entry(key, path, size_bytes)
      self._logger.debug(
          f'Cache SET: {key} ({size_bytes / 1024:.2f} KB)'
      )
      
    except Exception as e:
      self._logger.error(f'Failed to write cache {key}: {e}')
      # Clean up temp file if it exists
      temp_path = path.with_suffix('.tmp')
      if temp_path.exists():
        temp_path.unlink()

  def cache_stats(self) -> dict[str, Any]:
    """Get comprehensive cache statistics.
    
    Returns:
        Dictionary with cache statistics including:
        - total_size_mb: Total cache size in MB
        - total_entries: Number of cached items
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_rate: Cache hit rate percentage
        - evictions: Number of evicted entries
    """
    stats = self._metadata.get_statistics()
    
    return {
        'total_size_mb': stats.total_size_bytes / 1024**2,
        'total_size_bytes': stats.total_size_bytes,
        'total_entries': stats.total_entries,
        'hits': stats.hits,
        'misses': stats.misses,
        'hit_rate': stats.hit_rate,
        'evictions': stats.evictions,
        'max_size_mb': (
            self._config.max_size_bytes / 1024**2 
            if self._config.max_size_bytes else None
        ),
        'ttl_seconds': self._config.ttl_seconds,
    }

  def cleanup(self, aggressive: bool = False) -> dict[str, int]:
    """Manually clean up cache.
    
    Args:
        aggressive: If True, also enforce size limits aggressively
    
    Returns:
        Dictionary with cleanup statistics
    """
    self._logger.info('Starting manual cache cleanup...')
    
    # Remove expired entries
    expired_count = self._cleanup_expired()
    
    evicted_count = 0
    if aggressive and self._config.max_size_bytes:
      # Aggressively enforce size limit
      current_size = self._metadata.get_total_size()
      max_size = self._config.max_size_bytes
      
      if current_size > max_size * self._config.cleanup_threshold:
        space_to_free = current_size - int(max_size * 0.8)  # Free to 80%
        self._evict_lru(required_space=space_to_free)
        evicted_count = self._metadata.get_statistics().evictions
    
    # Force metadata sync
    self._metadata.force_sync()
    
    result = {
        'expired_removed': expired_count,
        'lru_evicted': evicted_count,
        'total_cleaned': expired_count + evicted_count,
    }
    
    self._logger.info(f'Cleanup complete: {result}')
    return result

  def get_total_size(self) -> int:
    """Get total cache size in bytes."""
    return self._metadata.get_total_size()


# ============================================================================
# Utility Functions
# ============================================================================

def make_key(func_name: str, args: tuple, kwargs: dict) -> str:
  """Creates a deterministic string key from function arguments.
  
  Args:
      func_name: Name of the function being cached
      args: Positional arguments
      kwargs: Keyword arguments
  
  Returns:
      SHA256 hash of the serialized arguments
  """
  try:
    key_data = (func_name, args, tuple(sorted(kwargs.items())))
    return hashlib.sha256(pickle.dumps(key_data)).hexdigest()
  except Exception:
    # Fallback if arguments are not picklable
    return f"{func_name}:{hash(str(args) + str(kwargs))}"

