# Copyright 2026 Google LLC.
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

"""Utility functions for the Atlas client."""

from collections.abc import Iterable

from alphagenome.data import ontology
from alphagenome.protos import dna_model_pb2


def build_filter(
    *,
    requested_scorers: Iterable[str] | None = None,
    ontology_terms: Iterable[ontology.OntologyTerm | str] | None = None,
    gene_ids: Iterable[str] | None = None,
    gene_names: Iterable[str] | None = None,
) -> str:
  """Builds a filter string for the DenseVariantScores response."""

  scorer_filter = ' OR '.join(
      f'scores.variant_scorer.name = "{scorer}"'
      for scorer in requested_scorers or []
  )

  gene_filter = ' OR '.join(
      f'scores.metadata.gene_scorers.metadata.gene_id = "{gene_id}"'
      for gene_id in gene_ids or []
  )

  gene_names_filter = ' OR '.join(
      f'scores.metadata.gene_scorers.metadata.name = "{gene_name}"'
      for gene_name in gene_names or []
  )
  gene_filter = ' OR '.join(filter(None, [gene_filter, gene_names_filter]))

  ontology_protos = []
  for ontology_term in ontology_terms or []:
    if isinstance(ontology_term, str):
      ontology_protos.append(ontology.from_curie(ontology_term).to_proto())
    else:
      ontology_protos.append(ontology_term.to_proto())

  ontology_filter = ' OR '.join(
      '(scores.metadata.tracks.metadata.ontology_term.ontology_type ='
      f' {dna_model_pb2.OntologyType.Name(o.ontology_type)} AND'
      ' scores.metadata.tracks.metadata.ontology_term.id ='
      f' {o.id})'
      for o in ontology_protos
  )
  if ontology_filter:
    # Ignore ontology filter for scorers without ontology metadata.
    ontology_filter = (
        '-scores.metadata.tracks.metadata.ontology_term:* OR'
        f' ({ontology_filter})'
    )

  return ' AND '.join(
      f'({f})'
      for f in filter(None, [scorer_filter, ontology_filter, gene_filter])
  )
