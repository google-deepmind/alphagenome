# Copyright 2024 Google LLC.
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

"""Utility functions for converting NumPy arrays to Tensor protocol buffers."""

from collections.abc import Iterable, Sequence

import immutabledict
import ml_dtypes
import numpy as np
import zstandard

from alphagenome.protos import tensor_pb2

_TENSOR_DTYPE_TO_NUMPY_DTYPE = immutabledict.immutabledict({
    tensor_pb2.DataType.DATA_TYPE_BFLOAT16: np.dtype(ml_dtypes.bfloat16),
    tensor_pb2.DataType.DATA_TYPE_FLOAT16: np.dtype(np.float16),
    tensor_pb2.DataType.DATA_TYPE_FLOAT32: np.dtype(np.float32),
    tensor_pb2.DataType.DATA_TYPE_FLOAT64: np.dtype(np.float64),
    tensor_pb2.DataType.DATA_TYPE_INT8: np.dtype(np.int8),
    tensor_pb2.DataType.DATA_TYPE_INT32: np.dtype(np.int32),
    tensor_pb2.DataType.DATA_TYPE_INT64: np.dtype(np.int64),
    tensor_pb2.DataType.DATA_TYPE_UINT8: np.dtype(np.uint8),
    tensor_pb2.DataType.DATA_TYPE_UINT32: np.dtype(np.uint32),
    tensor_pb2.DataType.DATA_TYPE_UINT64: np.dtype(np.uint64),
    tensor_pb2.DataType.DATA_TYPE_BOOL: np.dtype(bool),
})

_NUMPY_DTYPE_TO_TENSOR_DTYPE = immutabledict.immutabledict(
    {value: key for key, value in _TENSOR_DTYPE_TO_NUMPY_DTYPE.items()}
)


def _compress_bytes(
    array: np.ndarray, compression_type: tensor_pb2.CompressionType
):
  """Compresses a c-contiguous array to the specified compression type."""
  assert array.flags.c_contiguous
  array = array.view(np.uint8)
  match compression_type:
    case tensor_pb2.CompressionType.COMPRESSION_TYPE_ZSTD:
      return zstandard.compress(array.data)
    case tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE:
      return bytes(array.data)


def _decompress_bytes(
    data: bytes, compression_type: tensor_pb2.CompressionType
):
  """Decompress bytes using the specified compression type."""
  match compression_type:
    case tensor_pb2.CompressionType.COMPRESSION_TYPE_ZSTD:
      return zstandard.decompress(data)
    case tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE:
      return data


def pack_tensor(
    value: ...,
    *,
    bytes_per_chunk: int = 0,
    compression_type: tensor_pb2.CompressionType = (
        tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE
    ),
) -> tuple[tensor_pb2.Tensor, Sequence[tensor_pb2.TensorChunk]]:
  """Encodes the value as a Tensor and optional sequence of chunks.

  Args:
    value: An array-like object to pack. For example, scalar (float, int, bool,
      etc.), NumPy array, or nested lists of scalars.
    bytes_per_chunk: The number of bytes to include in each chunk. If 0, the
      entire value will be packed into the Tensor proto, otherwise the value
      will be split into chunks of this size.
    compression_type: The type of compression to apply to the data. This is
      applied to each chunk separately.

  Returns:
    Tuple of Tensor protocol buffer and, if items_per_chunk is greater than 0, a
    sequence of TensorChunk protos.
  """
  packed = tensor_pb2.Tensor()
  value = np.ascontiguousarray(value)

  packed.shape[:] = value.shape
  packed.data_type = _NUMPY_DTYPE_TO_TENSOR_DTYPE[value.dtype]

  chunks = []
  if bytes_per_chunk > 0:
    items_per_chunk = bytes_per_chunk // value.itemsize
    if bytes_per_chunk < value.itemsize:
      raise ValueError(f'{bytes_per_chunk=} must be >= {value.itemsize=}.')
    for chunk in np.split(
        value.ravel(), range(items_per_chunk, value.size, items_per_chunk)
    ):
      chunks.append(
          tensor_pb2.TensorChunk(
              data=_compress_bytes(chunk, compression_type),
              compression_type=compression_type,
          )
      )
    packed.chunk_count = len(chunks)
  else:
    packed.array.data = _compress_bytes(value, compression_type)
    packed.array.compression_type = compression_type

  return packed, chunks


def unpack_proto(
    proto: tensor_pb2.Tensor,
    chunks: Iterable[tensor_pb2.TensorChunk] = (),
) -> np.ndarray:
  """Converts a Tensor proto and any chunks into a NumPy array.

  Args:
    proto: Tensor proto to unpack.
    chunks: Optional sequence of TensorChunk protos to unpack.

  Returns:
    NumPy array of the unpacked data.
  """
  dtype = _TENSOR_DTYPE_TO_NUMPY_DTYPE[proto.data_type]
  match proto.WhichOneof('payload'):
    case 'array':
      data = _decompress_bytes(proto.array.data, proto.array.compression_type)
      array = np.frombuffer(data, dtype=dtype).reshape(proto.shape)
    case 'chunk_count':
      array = np.empty(np.prod(proto.shape) * dtype.itemsize, dtype=np.uint8)
      bytes_received = 0
      for chunk in chunks:
        chunk_data = np.frombuffer(
            _decompress_bytes(chunk.data, chunk.compression_type),
            dtype=np.uint8,
        )
        array[bytes_received : bytes_received + chunk_data.nbytes] = chunk_data
        bytes_received += chunk_data.nbytes

      if bytes_received != array.nbytes:
        raise ValueError(
            f'Expected {array.nbytes} bytes but only received {bytes_received} '
            'bytes.'
        )
      array = array.view(dtype).reshape(proto.shape)
    case _:
      raise ValueError(
          f'Unsupported payload type: {proto.WhichOneof("payload")}'
      )

  return array


def upcast_floating(x: np.ndarray) -> np.ndarray:
  """Helper to upcast low-precision floating point arrays to float32."""
  dtype = np.result_type(x)
  if (
      np.issubdtype(dtype, np.floating) or dtype == ml_dtypes.bfloat16
  ) and dtype.itemsize < 4:
    return x.astype(np.float32)
  else:
    return x
