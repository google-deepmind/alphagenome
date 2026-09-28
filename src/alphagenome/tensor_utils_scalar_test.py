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

"""Scalar tensor serialization preserves zero-dimensional shapes."""

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome import tensor_utils
from alphagenome.protos import tensor_pb2
import ml_dtypes
import numpy as np
import zstandard

_DTYPES = (
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
)
_COMPRESSION = (
    tensor_pb2.COMPRESSION_TYPE_NONE,
    tensor_pb2.COMPRESSION_TYPE_ZSTD,
)


class ScalarTensorTest(parameterized.TestCase):

  @parameterized.product(
      dtype=_DTYPES,
      numpy_scalar=(False, True),
      compression=_COMPRESSION,
      chunked=(False, True),
  )
  def test_numpy_scalar_shape_dtype_and_bytes(
      self, dtype, numpy_scalar, compression, chunked
  ):
    value = np.asarray(1 if dtype is bool else 3, dtype=dtype)
    original = value.copy()
    tensor, chunks = tensor_utils.pack_tensor(
        value[()] if numpy_scalar else value,
        bytes_per_chunk=value.itemsize if chunked else 0,
        compression_type=compression,
    )
    self.assertEmpty(tensor.shape)
    self.assertEqual(
        tensor.WhichOneof('payload'), 'chunk_count' if chunked else 'array'
    )
    restored = tensor_pb2.Tensor.FromString(tensor.SerializeToString())
    restored_chunks = [
        tensor_pb2.TensorChunk.FromString(c.SerializeToString()) for c in chunks
    ]
    actual = tensor_utils.unpack_proto(restored, iter(restored_chunks))
    self.assertEqual(actual.shape, ())
    self.assertEqual(actual.dtype, value.dtype)
    self.assertEqual(actual.tobytes(), value.tobytes())
    np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(value, original)
    if chunked:
      self.assertLen(chunks, 1)
      payload = chunks[0].data
    else:
      self.assertEmpty(chunks)
      payload = tensor.array.data
    expected = (
        zstandard.compress(value.tobytes())
        if compression == tensor_pb2.COMPRESSION_TYPE_ZSTD
        else value.tobytes()
    )
    self.assertEqual(payload, expected)

  @parameterized.product(
      value=(True, False, -7, 2**55 + 1, 1.25),
      compression=_COMPRESSION,
      chunked=(False, True),
  )
  def test_python_scalar_roundtrip(self, value, compression, chunked):
    tensor, chunks = tensor_utils.pack_tensor(
        value,
        bytes_per_chunk=32 if chunked else 0,
        compression_type=compression,
    )
    actual = tensor_utils.unpack_proto(tensor, chunks)
    expected = np.asarray(value)
    self.assertEmpty(tensor.shape)
    self.assertEqual(actual.shape, expected.shape)
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertEqual(actual.tobytes(), expected.tobytes())

  @parameterized.product(compression=_COMPRESSION, chunked=(False, True))
  def test_external_scalar_wire_format(self, compression, chunked):
    expected = np.asarray(-2.5, dtype=np.float64)
    payload = (
        zstandard.compress(expected.tobytes())
        if compression == tensor_pb2.COMPRESSION_TYPE_ZSTD
        else expected.tobytes()
    )
    tensor = tensor_pb2.Tensor(data_type=tensor_pb2.DATA_TYPE_FLOAT64)
    if chunked:
      tensor.chunk_count = 1
      chunks = [
          tensor_pb2.TensorChunk(data=payload, compression_type=compression)
      ]
    else:
      tensor.array.data = payload
      tensor.array.compression_type = compression
      chunks = []
    actual = tensor_utils.unpack_proto(
        tensor_pb2.Tensor.FromString(tensor.SerializeToString()), chunks
    )
    self.assertEqual(actual.shape, ())
    np.testing.assert_array_equal(actual, expected)

  @parameterized.product(compression=_COMPRESSION, chunked=(False, True))
  def test_existing_array_shapes_and_layouts(self, compression, chunked):
    arrays = (
        np.array([3]),
        np.zeros((2, 0, 3)),
        np.arange(24).reshape(4, 6)[:, ::-2],
        np.asfortranarray(np.ones((2, 3))),
        np.broadcast_to(np.array(2.5), (2, 3)),
    )
    for value in arrays:
      with self.subTest(shape=value.shape, strides=value.strides):
        before = value.copy()
        tensor, chunks = tensor_utils.pack_tensor(
            value,
            bytes_per_chunk=32 if chunked else 0,
            compression_type=compression,
        )
        self.assertEqual(tuple(tensor.shape), value.shape)
        actual = tensor_utils.unpack_proto(tensor, chunks)
        self.assertEqual(actual.shape, value.shape)
        self.assertEqual(actual.dtype, value.dtype)
        np.testing.assert_array_equal(actual, before)
        np.testing.assert_array_equal(value, before)

  def test_invalid_or_missing_chunks_still_raise(self):
    value = np.asarray(1.25, dtype=np.float64)
    tensor, _ = tensor_utils.pack_tensor(value, bytes_per_chunk=8)
    with self.assertRaisesRegex(ValueError, 'Expected 8 bytes'):
      tensor_utils.unpack_proto(tensor, [])
    with self.assertRaisesRegex(ValueError, 'must be >='):
      tensor_utils.pack_tensor(value, bytes_per_chunk=1)


if __name__ == '__main__':
  absltest.main()
