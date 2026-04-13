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

import os
import sys
from unittest import mock

from absl.testing import absltest

from alphagenome import colab_utils

_TEST_SECRET_KEY = '_TEST_ALPHAGENOME_API_KEY'


class ColabUtilsTest(absltest.TestCase):

  def test_get_api_key_from_environment(self):
    with mock.patch.dict(os.environ, {_TEST_SECRET_KEY: 'foo'}):
      self.assertEqual(colab_utils.get_api_key(_TEST_SECRET_KEY), 'foo')

  def test_get_api_key_from_environment_not_found_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        f"Cannot find API key with secret='{_TEST_SECRET_KEY}'.",
    ):
      _ = colab_utils.get_api_key(_TEST_SECRET_KEY)

  def test_get_api_key_from_colab_secrets(self):
    mock_colab = mock.MagicMock()
    mock_colab.userdata.get.return_value = 'bar'

    with mock.patch.dict(
        sys.modules, {'google': mock.MagicMock(), 'google.colab': mock_colab}
    ):
      self.assertEqual(colab_utils.get_api_key(), 'bar')

  def test_get_api_key_from_colab_secrets_not_found_raises_error(self):
    mock_colab = mock.MagicMock()
    mock_colab.userdata.NotebookAccessError = Exception
    mock_colab.userdata.SecretNotFoundError = Exception
    mock_colab.userdata.TimeoutException = Exception

    mock_colab.userdata.get.side_effect = Exception()

    with mock.patch.dict(
        sys.modules, {'google': mock.MagicMock(), 'google.colab': mock_colab}
    ):
      secret = 'my_secret'
      with self.assertRaisesRegex(
          ValueError,
          f'Cannot find or access API key in Colab secrets with {secret=}.',
      ):
        _ = colab_utils.get_api_key(secret)


if __name__ == '__main__':
  absltest.main()
