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


"""Descriptors of a model condition."""

from collections.abc import Sequence
import dataclasses

from alphagenome.data import ontology

from alphagenome.protos.v2 import dna_model_pb2


@dataclasses.dataclass(frozen=True)
class ModelCondition:
  """Metadata describing the type of a model condition.

  Attributes:
    clo: Optional Cell Line Ontology (CLO) term.
    uberon: Optional Uber-anatomy Ontology (UBERON) term.
    cl: Optional Cell Ontology (CL) term.
    efo: Optional Experimental Factor Ontology (EFO) term.
    ntr: Optional New Term Request (NTR) ontology term.
    ontology_terms: Tuple of ontology terms set on this condition.
    ontology_curies: Tuple of ontology CURIE strings set on this condition.
  """

  clo: ontology.OntologyTerm | None = None
  uberon: ontology.OntologyTerm | None = None
  cl: ontology.OntologyTerm | None = None
  efo: ontology.OntologyTerm | None = None
  ntr: ontology.OntologyTerm | None = None

  def __str__(self):
    return '_'.join(self.ontology_curies)

  @property
  def ontology_terms(self) -> tuple[ontology.OntologyTerm, ...]:
    """Returns ontology terms that have been set for this ModelCondition."""
    return tuple(
        filter(None, [self.clo, self.uberon, self.cl, self.efo, self.ntr])
    )

  @property
  def ontology_curies(self) -> tuple[str, ...]:
    """Returns ontology curies that have been set for this ModelCondition."""
    return tuple(term.ontology_curie for term in self.ontology_terms)

  def to_proto(self) -> dna_model_pb2.Condition:
    """Converts to a dna_model_pb2.Condition proto."""

    return dna_model_pb2.Condition(
        clo=self.clo.id if self.clo else None,
        uberon=self.uberon.id if self.uberon else None,
        cl=self.cl.id if self.cl else None,
        efo=self.efo.id if self.efo else None,
        ntr=self.ntr.id if self.ntr else None,
    )


def from_curies(
    ontology_curies: Sequence[str],
) -> ModelCondition:
  """Get a ModelCondition from a list of ontology curies.

  Args:
    ontology_curies: A list of ontology curies.

  Returns:
      A ModelCondition with the ontology terms corresponding to the curies.

  Raises:
    ValueError: If there are duplicate ontology types provided.
  """
  ontology_terms = ontology.from_curies(ontology_curies)
  ontology_types = [term.type for term in ontology_terms]
  if len(set(ontology_types)) != len(ontology_types):
    raise ValueError(f'Duplicate ontology types: {ontology_types}')
  ontology_dict = {}
  for term in ontology_terms:
    ontology_dict[term.type.name.lower()] = term
  return ModelCondition(**ontology_dict)


def _term_from_proto(
    proto: dna_model_pb2.Condition,
    field_name: str,
    term_type: ontology.OntologyType,
) -> ontology.OntologyTerm | None:
  """Helper to create an OntologyTerm from a proto field."""
  if proto.HasField(field_name):
    return ontology.OntologyTerm(term_type, getattr(proto, field_name))
  return None


def from_proto(condition_proto: dna_model_pb2.Condition) -> ModelCondition:
  """Get a ModelCondition from a dna_model_pb2.Condition proto."""
  return ModelCondition(
      clo=_term_from_proto(condition_proto, 'clo', ontology.OntologyType.CLO),
      uberon=_term_from_proto(
          condition_proto, 'uberon', ontology.OntologyType.UBERON
      ),
      cl=_term_from_proto(condition_proto, 'cl', ontology.OntologyType.CL),
      efo=_term_from_proto(condition_proto, 'efo', ontology.OntologyType.EFO),
      ntr=_term_from_proto(condition_proto, 'ntr', ontology.OntologyType.NTR),
  )
