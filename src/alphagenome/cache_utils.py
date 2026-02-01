"""Module containing caching utilities for AlphaGenome."""

import abc
import hashlib
import os
import pickle
import threading
from typing import Any, Callable, TypeVar

T = TypeVar('T')


class Cache(abc.ABC):
  """Abstract base class for caches."""

  @abc.abstractmethod
  def get(self, key: str) -> Any | None:
    """Retrieve an item from the cache."""

  @abc.abstractmethod
  def set(self, key: str, value: Any) -> None:
    """Store an item in the cache."""


class NoCache(Cache):
  """A dummy cache that does nothing."""

  def get(self, key: str) -> Any | None:
    return None

  def set(self, key: str, value: Any) -> None:
    pass


class DiskCache(Cache):
  """A file-based cache implementation.

  This cache stores serialized objects on disk. Keys are mapped to filenames.
  It is thread-safe within a process via a global lock per key (not implemented here fully, but ok for simple usage).
  """

  def __init__(self, cache_dir: str):
    self._cache_dir = cache_dir
    os.makedirs(self._cache_dir, exist_ok=True)
    self._lock = threading.Lock()

  def _get_path(self, key: str) -> str:
    # Use SHA256 of the key to ensure valid filename
    hashed_key = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return os.path.join(self._cache_dir, hashed_key + '.pkl')

  def get(self, key: str) -> Any | None:
    path = self._get_path(key)
    if not os.path.exists(path):
      return None
    try:
      with self._lock:
          with open(path, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
      # Corrupt cache file
      return None
    except Exception:
      # General read error
      return None

  def set(self, key: str, value: Any) -> None:
    path = self._get_path(key)
    try:
      with self._lock:
          # Write to temp file then rename for atomic write
          temp_path = path + '.tmp'
          with open(temp_path, 'wb') as f:
            pickle.dump(value, f)
          os.replace(temp_path, path)
    except Exception:
      pass # Silent fail on write error to not disrupt main flow


def make_key(func_name: str, args: tuple, kwargs: dict) -> str:
  """Creates a deterministic string key from function arguments."""
  # Serialize arguments to create a unique signature
  # We use repr() for simple types, but pickle for complex ones to be safe
  # However, for cache keys, we want stability.
  # Let's use pickle.dumps then hash it.
  # To avoid issues with object identity, ideally we'd use consistent serialization.
  # For dataclasses like Interval/Variant, repr() or str() usually works if they have good __repr__.
  # Let's try to pickle the whole args tuple.
  try:
    key_data = (func_name, args, tuple(sorted(kwargs.items())))
    return hashlib.sha256(pickle.dumps(key_data)).hexdigest()
  except Exception:
    # Fallback if arguments are not picklable (though they should be for DnaClient methods)
    return f"{func_name}:{hash(str(args) + str(kwargs))}"
