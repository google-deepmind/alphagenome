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

"""Utility functions for type annotations."""

import importlib.metadata
from typing import TypeVar

import jaxtyping
import typeguard

_T = TypeVar('_T')


def jaxtyped(fn: _T) -> _T:
  """Wrapper around jaxtyping.jaxtyped that uses typeguard if typeguard < 3."""
  try:
    major, *_ = importlib.metadata.version('typeguard').split('.')
  except importlib.metadata.PackageNotFoundError:
    major = -1

  # Only use jaxtyping if typeguard is < 3. See
  # https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#runtime-type-checking
  # for more details.
  if int(major) < 3:
    return jaxtyping.jaxtyped(fn, typechecker=typeguard.typechecked)
  else:
    return fn
