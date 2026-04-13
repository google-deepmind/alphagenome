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

import math

import ml_dtypes
import numpy as np
import zstandard
from absl.testing import absltest, parameterized

from alphagenome import tensor_utils
from alphagenome.protos import tensor_pb2


class TensorUtilsTest(parameterized.TestCase):

  @parameterized.product(
      (
          dict(
              array=np.array([[1, 2], [3, 4]], dtype=ml_dtypes.bfloat16),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_BFLOAT16,
          ),
          dict(
              array=np.array([[1, 2], [3, 4]], dtype=np.float16),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_FLOAT16,
          ),
          dict(
              array=np.array([[1, 2, 3, 4]], dtype=np.float32),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_FLOAT32,
          ),
          dict(
              array=np.array([[1], [2], [3], [4]], dtype=np.float64),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_FLOAT64,
          ),
          dict(
              array=np.array([[1, 2], [3, 4]], dtype=np.int8),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_INT8,
          ),
          dict(
              array=np.array([1, 2, 3, 4], dtype=np.int32),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_INT32,
          ),
          dict(
              array=np.array([1, 2, 3, 4], dtype=np.int64),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_INT64,
          ),
          dict(
              array=np.array([1, 2, 3, 4], dtype=np.uint8),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_UINT8,
          ),
          dict(
              array=np.array([1, 2, 3, 4], dtype=np.uint32),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_UINT32,
          ),
          dict(
              array=np.array([1, 2, 3, 4], dtype=np.uint64),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_UINT64,
          ),
          dict(
              array=np.array([True, False, True, False], dtype=bool),
              expected_dtype=tensor_pb2.DataType.DATA_TYPE_BOOL,
          ),
      ),
      compression_type=[
          tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE,
          tensor_pb2.CompressionType.COMPRESSION_TYPE_ZSTD,
      ],
  )
  def test_pack_tensor(self, array, expected_dtype, compression_type):
    packed, chunks = tensor_utils.pack_tensor(
        array, compression_type=compression_type
    )

    self.assertEmpty(chunks)
    self.assertSequenceEqual(packed.shape, array.shape)
    self.assertEqual(packed.data_type, expected_dtype)

    expected = (
        zstandard.compress(array.tobytes())
        if compression_type == tensor_pb2.CompressionType.COMPRESSION_TYPE_ZSTD
        else array.tobytes()
    )
    self.assertEqual(expected, packed.array.data)

  @parameterized.product(
      dtype=(
          ml_dtypes.bfloat16,
          np.float16,
          np.float32,
          np.float64,
          np.int8,
          np.int32,
          np.int64,
          np.uint8,
          np.uint32,
          np.uint64,
          bool,
      ),
      bytes_per_chunk=[10, 13, 200],
      compression_type=[
          tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE,
          tensor_pb2.CompressionType.COMPRESSION_TYPE_ZSTD,
      ],
  )
  def test_pack_tensor_chunks(self, dtype, bytes_per_chunk, compression_type):
    if dtype == bool:
      data = np.ones(1000, dtype=bool)
    else:
      data = np.arange(1000, dtype=dtype)
    data = data.reshape((10, -1))
    packed, chunks = tensor_utils.pack_tensor(
        data, bytes_per_chunk=bytes_per_chunk, compression_type=compression_type
    )
    self.assertEmpty(packed.array.data)
    self.assertLen(chunks, packed.chunk_count)

    items_per_chunk = bytes_per_chunk // data.itemsize
    self.assertEqual(packed.chunk_count, math.ceil(data.size / items_per_chunk))
    first_chunk = data.flatten()[:items_per_chunk].tobytes()
    expected = (
        zstandard.compress(first_chunk)
        if compression_type == tensor_pb2.CompressionType.COMPRESSION_TYPE_ZSTD
        else first_chunk
    )
    self.assertEqual(expected, chunks[0].data)

  @parameterized.product(
      array=(
          np.array([[1, 2], [3, 4]], dtype=ml_dtypes.bfloat16),
          np.array([[1, 2], [3, 4]], dtype=np.float16),
          np.array([[1, 2, 3, 4]], dtype=np.float32),
          np.array([[1], [2], [3], [4]], dtype=np.float64),
          np.array([[1, 2], [3, 4]], dtype=np.int8),
          np.array([1, 2, 3, 4], dtype=np.int32),
          np.array([1, 2, 3, 4], dtype=np.int64),
          np.array([1, 2, 3, 4], dtype=np.uint8),
          np.array([1, 2, 3, 4], dtype=np.uint32),
          np.array([1, 2, 3, 4], dtype=np.uint64),
          np.array([True, False, True, False], dtype=bool),
          np.arange(100)[::2],
      ),
      bytes_per_chunk=[8, 12, 15],
      compression_type=[
          tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE,
          tensor_pb2.CompressionType.COMPRESSION_TYPE_ZSTD,
      ],
  )
  def test_unpack_proto(self, array, bytes_per_chunk, compression_type):
    tensor, chunks = tensor_utils.pack_tensor(
        array,
        bytes_per_chunk=bytes_per_chunk,
        compression_type=compression_type,
    )
    self.assertLen(chunks, tensor.chunk_count)
    round_trip = tensor_utils.unpack_proto(tensor, chunks)
    np.testing.assert_array_equal(round_trip, array)

  def test_pack_proto_invalid_chunk_size_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'bytes_per_chunk=1 must be >= value.itemsize=8'
    ):
      tensor_utils.pack_tensor(
          np.array([[1, 2], [3, 4]], dtype=np.int64), bytes_per_chunk=1
      )

  def test_missing_chunk_raises(self):
    tensor, chunks = tensor_utils.pack_tensor(
        np.arange(128, dtype=np.float32), bytes_per_chunk=32
    )
    chunks = chunks[::2]
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Expected 512 bytes but only received 256 bytes.'
    ):
      tensor_utils.unpack_proto(tensor, chunks)

  @parameterized.named_parameters(
      (
          'float16',
          np.array([1, 2, 3], dtype=np.float16),
          np.array([1, 2, 3], dtype=np.float32),
      ),
      (
          'bfloat16',
          np.array([1, 2, 3], dtype=ml_dtypes.bfloat16),
          np.array([1, 2, 3], dtype=np.float32),
      ),
      (
          'float32',
          np.array([1, 2, 3], dtype=np.float32),
          np.array([1, 2, 3], dtype=np.float32),
      ),
      (
          'float64',
          np.array([1, 2, 3], dtype=np.float64),
          np.array([1, 2, 3], dtype=np.float64),
      ),
      (
          'int',
          np.array([1, 2, 3], dtype=np.int32),
          np.array([1, 2, 3], dtype=np.int32),
      ),
  )
  def test_upcast_floating(self, array, expected):
    np.testing.assert_array_equal(tensor_utils.upcast_floating(array), expected)


if __name__ == '__main__':
  absltest.main()
