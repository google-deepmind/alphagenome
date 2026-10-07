# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for Google Colab."""

import os


def get_api_key(secret: str = 'ALPHA_GENOME_API_KEY'):
  """Returns API key from environment variable or Colab secrets.

  Tries to retrieve the API key from the environment first. If not found,
  attempts to retrieve it from Colab secrets (if running in Colab).

  Args:
    secret: The name of the environment variable or Colab secret key to
      retrieve.

  Raises:
    ValueError: If the API key cannot be found in the environment or Colab
      secrets.
  """

  if api_key := os.environ.get(secret):
    return api_key

  try:
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    from google.colab import userdata  # pyrefly: ignore[missing-import]
    # pylint: enable=g-import-not-at-top, import-outside-toplevel

    try:
      api_key = userdata.get(secret)
      return api_key
    except (
        userdata.NotebookAccessError,
        userdata.SecretNotFoundError,
        userdata.TimeoutException,
    ) as e:
      raise ValueError(
          f'Cannot find or access API key in Colab secrets with {secret=}. Make'
          ' sure you have added the API key to Colab secrets and enabled'
          ' access. See'
          ' https://www.alphagenomedocs.com/installation.html#add-api-key-to-secrets'
          ' for more details.'
      ) from e
  except ImportError:
    # Not running in Colab.
    pass

  raise ValueError(f'Cannot find API key with {secret=}.')
