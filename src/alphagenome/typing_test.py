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

import importlib.metadata
from unittest import mock

import jaxtyping
import numpy as np
from absl.testing import absltest, parameterized
from jaxtyping import (
    Float32,  # pylint: disable=g-multiple-import, g-importing-member
)

from alphagenome import typing


class TypingTest(parameterized.TestCase):

  @parameterized.parameters((2,), (3,))
  def test_wrapped(self, typeguard_version: int):
    with mock.patch.object(
        importlib.metadata, 'version'
    ) as mock_typeguard_version:
      mock_typeguard_version.return_value = f'{typeguard_version}.0.0'

      @typing.jaxtyped
      def foo(x: Float32[np.ndarray, 'B']):
        return x

      expected = np.zeros((1,), dtype=np.float32)
      self.assertEqual(expected, foo(expected))

      if typeguard_version < 3:
        with self.assertRaises(jaxtyping.TypeCheckError):
          _ = foo(np.zeros((6, 6, 6), dtype=bool))


if __name__ == '__main__':
  absltest.main()
