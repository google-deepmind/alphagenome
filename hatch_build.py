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

"""Hatch build hook to generate Python bindings for protos."""

import os
from typing import Any

from grpc_tools import protoc
from hatchling.builders.hooks.plugin.interface import (
  BuildHookInterface,  # pylint: disable=g-importing-member
)

_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

# Tuple of proto message definitions to build Python bindings for. Paths must
# be relative to root directory.
_ALPHAGENOME_PROTOS = (
    'alphagenome/protos/dna_model.proto',
    'alphagenome/protos/dna_model_service.proto',
    'alphagenome/protos/tensor.proto',
)


class GenerateProtos(BuildHookInterface):
  """Generates Python protobuf bindings for alphagenome.protos."""

  def initialize(self, version: str, build_data: dict[str, Any]) -> None:
    del version, build_data  # Unused.

    for proto_path in _ALPHAGENOME_PROTOS:
      proto_args = [
          'grpc_tools.protoc',
          f'--proto_path={_ROOT_DIR}',
          f'--python_out={_ROOT_DIR}',
          f'--grpc_python_out={_ROOT_DIR}',
          os.path.join(_ROOT_DIR, proto_path),
      ]
      if protoc.main(proto_args) != 0:
        raise RuntimeError(f'ERROR: {proto_args}')
