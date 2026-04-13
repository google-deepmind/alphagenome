# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for tensor_utils."""

import functools
import itertools
import tracemalloc

import google_benchmark
import numpy as np
from absl import flags

from alphagenome import tensor_utils
from alphagenome.protos import tensor_pb2

_MB_TO_BYTES = 2**20

_TENSOR_SIZE_IN_MB = flags.DEFINE_integer(
    'tensor_size_in_mb', 128, 'Size of the tensor to benchmark.'
)

_COMPRESSION_TYPE = flags.DEFINE_enum(
    'compression_type',
    'zstd',
    ['none', 'zstd'],
    'Compression type to benchmark.',
)


@google_benchmark.register
@google_benchmark.option.use_real_time()
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.range_multiplier(2)
@google_benchmark.option.range(1, limit=10)
def pack_tensor_benchmark_chunks(state):
  """Benchmark for packing tensors."""
  rng = np.random.default_rng(seed=42)
  num_elements = (_TENSOR_SIZE_IN_MB.value * _MB_TO_BYTES) // np.dtype(
      np.float32
  ).itemsize
  values = rng.random(num_elements, dtype=np.float32)
  compression_type = tensor_pb2.CompressionType.Value(
      f'COMPRESSION_TYPE_{_COMPRESSION_TYPE.value.upper()}'
  )

  uncompressed_bytes = 0
  compressed_bytes = 0
  tracemalloc.start()

  while state:
    packed, chunks = tensor_utils.pack_tensor(
        values,
        bytes_per_chunk=state.range(0) * _MB_TO_BYTES,
        compression_type=compression_type,
    )
    uncompressed_bytes += values.nbytes
    compressed_bytes += functools.reduce(
        lambda x, y: x + len(y.data), itertools.chain(chunks, [packed.array]), 0
    )

  _, peak_mem = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  state.counters['compressed_bytes'] = google_benchmark.Counter(
      compressed_bytes
  )
  state.counters['uncompressed_bytes'] = google_benchmark.Counter(
      uncompressed_bytes
  )
  state.counters['peak_mem'] = google_benchmark.Counter(peak_mem)


@google_benchmark.register
@google_benchmark.option.use_real_time()
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def unpack_tensor_benchmark_chunks(state):
  """Benchmark for unpacking tensors."""
  rng = np.random.default_rng(seed=42)
  num_elements = (_TENSOR_SIZE_IN_MB.value * _MB_TO_BYTES) // np.dtype(
      np.float32
  ).itemsize
  compression_type = tensor_pb2.CompressionType.Value(
      f'COMPRESSION_TYPE_{_COMPRESSION_TYPE.value.upper()}'
  )
  packed, chunks = tensor_utils.pack_tensor(
      rng.random(num_elements, dtype=np.float32),
      bytes_per_chunk=1 * _MB_TO_BYTES,
      compression_type=compression_type,
  )

  tracemalloc.start()

  while state:
    tensor_utils.unpack_proto(packed, chunks)

  _, peak_mem = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  state.counters['peak_mem'] = google_benchmark.Counter(peak_mem)


if __name__ == '__main__':
  google_benchmark.main()
