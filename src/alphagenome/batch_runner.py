"""Resilient batch processing for AlphaGenome.

This module provides a production-ready batch processing system with:
- Automatic checkpointing and resume
- Configurable error handling strategies
- Signal handling for graceful shutdown
- Optional JSONL streaming for memory efficiency
- Progress tracking with tqdm
"""

import json
import logging
import os
import random
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Callable, Literal
from pathlib import Path

import tqdm.auto


@dataclass
class CheckpointEntry:
  """Represents a single item's status in the checkpoint."""
  status: Literal['success', 'failed', 'pending']
  error: str | None = None
  attempts: int = 0
  result_file: str | None = None  # For JSONL streaming


class CheckpointManager:
  """Manages checkpoint files for batch processing."""
  
  def __init__(self, checkpoint_file: str):
    self.checkpoint_file = Path(checkpoint_file)
    self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    self._lock = threading.Lock()
    self._data = self._load_or_create()
  
  def _load_or_create(self) -> dict:
    """Load existing checkpoint or create new one."""
    if self.checkpoint_file.exists():
      try:
        with open(self.checkpoint_file, 'r') as f:
          return json.load(f)
      except (json.JSONDecodeError, IOError):
        # Corrupted checkpoint, start fresh
        return self._create_empty()
    return self._create_empty()
  
  def _create_empty(self) -> dict:
    """Create an empty checkpoint structure."""
    return {
      'total': 0,
      'completed': 0,
      'failed': 0,
      'entries': {}
    }
  
  def initialize(self, total: int):
    """Initialize checkpoint for a new batch."""
    with self._lock:
      if self._data['total'] == 0:
        self._data['total'] = total
        self._data['entries'] = {
            str(i): asdict(CheckpointEntry(status='pending'))
            for i in range(total)
        }
        self._save()
  
  def mark_success(self, index: int, result_file: str | None = None):
    """Mark an item as successfully processed."""
    with self._lock:
      entry = CheckpointEntry(
          status='success',
          result_file=result_file
      )
      self._data['entries'][str(index)] = asdict(entry)
      self._data['completed'] += 1
      self._save()
  
  def mark_failed(self, index: int, error: str, attempts: int):
    """Mark an item as failed."""
    with self._lock:
      entry = CheckpointEntry(
          status='failed',
          error=error,
          attempts=attempts
      )
      self._data['entries'][str(index)] = asdict(entry)
      self._data['failed'] += 1
      self._save()
  
  def increment_attempt(self, index: int):
    """Increment retry attempt counter."""
    with self._lock:
      entry = self._data['entries'][str(index)]
      entry['attempts'] = entry.get('attempts', 0) + 1
      self._save()
  
  def get_pending_indices(self) -> list[int]:
    """Get list of pending item indices."""
    with self._lock:
      return [
          int(idx)
          for idx, entry in self._data['entries'].items()
          if entry['status'] == 'pending'
      ]
  
  def get_stats(self) -> dict[str, int]:
    """Get current checkpoint statistics."""
    with self._lock:
      return {
          'total': self._data['total'],
          'completed': self._data['completed'],
          'failed': self._data['failed'],
          'pending': self._data['total'] - self._data['completed'] - self._data['failed']
      }
  
  def _save(self):
    """Save checkpoint with atomic write."""
    temp_file = self.checkpoint_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
      json.dump(self._data, f, indent=2)
    # Atomic rename (POSIX-compliant)
    temp_file.replace(self.checkpoint_file)
  
  def cleanup(self):
    """Remove checkpoint file."""
    if self.checkpoint_file.exists():
      self.checkpoint_file.unlink()


class BatchRunner:
  """Resilient batch processor for AlphaGenome operations."""
  
  def __init__(
      self,
      client,
      checkpoint_file: str = 'batch_checkpoint.json',
      on_error: Literal['skip', 'abort'] = 'skip',
      max_retries: int = 3,
      checkpoint_interval: int = 10,
      stream_results: bool = False,
      results_file: str | None = None,
  ):
    """Initialize BatchRunner.
    
    Args:
      client: DnaClient instance
      checkpoint_file: Path to checkpoint file
      on_error: Error handling strategy ('skip' or 'abort')
      max_retries: Maximum retry attempts per item
      checkpoint_interval: Save checkpoint every N items
      stream_results: Stream results to JSONL file instead of keeping in memory
      results_file: Path to results JSONL file (required if stream_results=True)
    """
    self.client = client
    self.checkpoint_manager = CheckpointManager(checkpoint_file)
    self.on_error = on_error
    self.max_retries = max_retries
    self.checkpoint_interval = checkpoint_interval
    self.stream_results = stream_results
    self.results_file = Path(results_file) if results_file else None
    
    if stream_results and not results_file:
      raise ValueError("results_file must be specified when stream_results=True")
    
    self._logger = logging.getLogger(f'{__name__}.BatchRunner')
    self._results = []
    self._shutdown_requested = False
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, self._handle_shutdown)
    signal.signal(signal.SIGTERM, self._handle_shutdown)
  
  def _handle_shutdown(self, signum, frame):
    """Handle shutdown signals gracefully."""
    self._logger.warning(f'Received signal {signum}, shutting down gracefully...')
    self._shutdown_requested = True
  
  def _exponential_backoff_with_jitter(self, attempt: int, base_delay: float = 1.0) -> float:
    """Calculate exponential backoff with jitter."""
    max_delay = base_delay * (2 ** attempt)
    # Add jitter: random value between 0 and max_delay
    jitter = random.uniform(0, max_delay * 0.3)  # 30% jitter
    return max_delay + jitter
  
  def _process_item(
      self, 
      index: int, 
      process_func: Callable, 
      *args, 
      **kwargs
  ) -> tuple[int, Any | None, str | None]:
    """Process a single item with retry logic.
    
    Returns:
      Tuple of (index, result, error_message)
    """
    for attempt in range(self.max_retries + 1):
      try:
        if self._shutdown_requested:
          return index, None, 'Shutdown requested'
        
        self.checkpoint_manager.increment_attempt(index)
        result = process_func(*args, **kwargs)
        return index, result, None
        
      except Exception as e:
        error_msg = str(e)
        self._logger.warning(
            f'Item {index} failed (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}'
        )
        
        if attempt < self.max_retries:
          backoff = self._exponential_backoff_with_jitter(attempt)
          self._logger.info(f'Retrying item {index} after {backoff:.2f}s...')
          time.sleep(backoff)
        else:
          return index, None, error_msg
    
    return index, None, 'Max retries exceeded'
  
  def _save_result_to_jsonl(self, index: int, result: Any):
    """Save a single result to JSONL file."""
    if not self.results_file:
      return
    
    entry = {
        'index': index,
        'result': result,
        'timestamp': time.time()
    }
    
    # Append to JSONL file
    with open(self.results_file, 'a') as f:
      f.write(json.dumps(entry) + '\n')
  
  def batch_predict_variants(
      self,
      intervals: list,
      variants: list,
      max_workers: int = 5,
      progress_bar: bool = True,
      **predict_kwargs
  ) -> list[Any | None]:
    """Batch predict variants with checkpointing.
    
    Args:
      intervals: List of genome.Interval objects
      variants: List of genome.Variant objects
      max_workers: Number of parallel workers
      progress_bar: Show tqdm progress bar
      **predict_kwargs: Additional arguments for predict_variant()
    
    Returns:
      List of results (None for failed items)
    """
    if len(intervals) != len(variants):
      raise ValueError("intervals and variants must have the same length")
    
    total = len(variants)
    self.checkpoint_manager.initialize(total)
    
    # Get pending indices (skip already completed)
    pending_indices = self.checkpoint_manager.get_pending_indices()
    self._logger.info(
        f'Starting batch: {total} total, {len(pending_indices)} pending'
    )
    
    # Initialize results list or file
    if self.stream_results:
      self.results_file.parent.mkdir(parents=True, exist_ok=True)
      if not self.results_file.exists():
        self.results_file.touch()
    else:
      self._results = [None] * total
    
    # Process items
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {}
      
      for idx in pending_indices:
        if self._shutdown_requested:
          break
        
        future = executor.submit(
            self._process_item,
            idx,
            self.client.predict_variant,
            interval=intervals[idx],
            variant=variants[idx],
            **predict_kwargs
        )
        futures[future] = idx
      
      # Progress bar
      pbar = tqdm.auto.tqdm(
          total=len(pending_indices),
          desc='Processing variants',
          disable=not progress_bar
      )
      
      items_since_checkpoint = 0
      
      for future in as_completed(futures):
        if self._shutdown_requested:
          self._logger.warning('Shutdown requested, cancelling remaining tasks...')
          executor.shutdown(wait=False, cancel_futures=True)
          break
        
        idx, result, error = future.result()
        
        if error is None:
          # Success
          if self.stream_results:
            self._save_result_to_jsonl(idx, result)
            self.checkpoint_manager.mark_success(idx, str(self.results_file))
          else:
            self._results[idx] = result
            self.checkpoint_manager.mark_success(idx)
          
          items_since_checkpoint += 1
        else:
          # Failed
          self.checkpoint_manager.mark_failed(idx, error, self.max_retries)
          
          if self.on_error == 'abort':
            self._logger.error(f'Aborting batch due to error on item {idx}')
            executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(f'Batch aborted: {error}')
        
        pbar.update(1)
        
        # Periodic checkpoint save
        if items_since_checkpoint >= self.checkpoint_interval:
          items_since_checkpoint = 0
      
      pbar.close()
    
    # Final statistics
    stats = self.checkpoint_manager.get_stats()
    self._logger.info(
        f"Batch complete: {stats['completed']} succeeded, "
        f"{stats['failed']} failed, {stats['pending']} pending"
    )
    
    if self.stream_results:
      self._logger.info(f'Results saved to {self.results_file}')
      return None  # Results are in file
    else:
      return self._results
  
  def cleanup(self):
    """Clean up checkpoint and result files."""
    self.checkpoint_manager.cleanup()
    if self.results_file and self.results_file.exists():
      self.results_file.unlink()
