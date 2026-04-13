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

"""Handling of biological ontologies."""

import dataclasses
import enum
from collections.abc import Sequence

import immutabledict

from alphagenome.protos import dna_model_pb2


class OntologyType(enum.IntEnum):
  """Supported ontology types.

  CLO: Cell Line Ontology.
  UBERON: Uber-anatomy ontology.
  CL: Cell Ontology.
  EFO: Experimental Factor Ontology.
  NTR: New Term Requested.
  """

  CLO = 1
  UBERON = 2
  CL = 3
  EFO = 4
  NTR = 5


_ONTOLOGY_TYPE_TO_PROTO_ENUM = immutabledict.immutabledict({
    OntologyType.CLO: dna_model_pb2.OntologyType.ONTOLOGY_TYPE_CLO,
    OntologyType.UBERON: dna_model_pb2.OntologyType.ONTOLOGY_TYPE_UBERON,
    OntologyType.CL: dna_model_pb2.OntologyType.ONTOLOGY_TYPE_CL,
    OntologyType.EFO: dna_model_pb2.OntologyType.ONTOLOGY_TYPE_EFO,
    OntologyType.NTR: dna_model_pb2.OntologyType.ONTOLOGY_TYPE_NTR,
})


@dataclasses.dataclass(frozen=True)
class OntologyTerm:
  """A single biological ontology term.

  Attributes:
    type: The ontology type.
    id: The ID of the term within the ontology.
  """

  type: OntologyType
  id: int

  @property
  def ontology_curie(self) -> str:
    """Returns the CURIE (Compact Uniform Resource Identifier) for the term."""
    return f'{self.type.name}:{self.id:07d}'

  def to_proto(self) -> dna_model_pb2.OntologyTerm:
    """Converts the ontology term to a protobuf message."""
    return dna_model_pb2.OntologyTerm(
        ontology_type=_ONTOLOGY_TYPE_TO_PROTO_ENUM[self.type], id=self.id
    )


def from_curie(ontology_curie: str) -> OntologyTerm:
  """Creates an `OntologyTerm` from a CURIE.

  Args:
    ontology_curie: The CURIE (Compact Uniform Resource Identifier) for the
      ontology term.

  Returns:
    An `OntologyTerm` object.
  """
  try:
    ontology_type, local_id = ontology_curie.split(':')
  except ValueError as e:
    raise ValueError(
        f'Invalid {ontology_curie=}. CURIEs must be of the form <type>:<id>.'
    ) from e
  try:
    ontology_type = OntologyType[ontology_type]
  except KeyError as e:
    raise ValueError(f'Cannot parse {ontology_type=}') from e
  return OntologyTerm(ontology_type, int(local_id))


def from_curies(ontology_curies: Sequence[str]) -> Sequence[OntologyTerm]:
  """Creates a list of `OntologyTerm` objects from a list of CURIEs.

  Args:
    ontology_curies: A list of CURIEs (Compact Uniform Resource Identifiers).

  Returns:
    A list of `OntologyTerm` objects.
  """
  return [from_curie(curie) for curie in ontology_curies]


def from_proto(proto: dna_model_pb2.OntologyTerm) -> OntologyTerm:
  """Creates an `OntologyTerm` from a protobuf message.

  Args:
    proto: An `OntologyTerm` protobuf message.

  Returns:
    An `OntologyTerm` object.
  """
  return OntologyTerm(
      OntologyType(proto.ontology_type),
      proto.id,
  )
