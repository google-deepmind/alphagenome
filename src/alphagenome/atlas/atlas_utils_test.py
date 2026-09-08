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

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.atlas import atlas_utils
from alphagenome.data import ontology


class AtlasUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Empty',
          requested_scorers=[],
          ontology_terms=None,
          gene_ids=None,
          gene_names=None,
          expected='',
      ),
      dict(
          testcase_name='ScorerOnly',
          requested_scorers=['scorer1', 'scorer2'],
          ontology_terms=None,
          gene_ids=None,
          gene_names=None,
          expected=(
              '(scores.variant_scorer.name = "scorer1" OR'
              ' scores.variant_scorer.name = "scorer2")'
          ),
      ),
      dict(
          testcase_name='OntologyOnly',
          requested_scorers=None,
          ontology_terms=['CL:0000001', ontology.from_curie('UBERON:0000002')],
          gene_ids=None,
          gene_names=None,
          expected=(
              '(-scores.metadata.tracks.metadata.ontology_term:* OR'
              ' ((scores.metadata.tracks.metadata.ontology_term.ontology_type ='
              ' ONTOLOGY_TYPE_CL AND'
              ' scores.metadata.tracks.metadata.ontology_term.id = 1) OR'
              ' (scores.metadata.tracks.metadata.ontology_term.ontology_type ='
              ' ONTOLOGY_TYPE_UBERON AND'
              ' scores.metadata.tracks.metadata.ontology_term.id = 2)))'
          ),
      ),
      dict(
          testcase_name='GeneOnly',
          requested_scorers=None,
          ontology_terms=None,
          gene_ids=['gene_1', 'gene_2'],
          gene_names=None,
          expected=(
              '(scores.metadata.gene_scorers.metadata.gene_id = "gene_1" OR'
              ' scores.metadata.gene_scorers.metadata.gene_id = "gene_2")'
          ),
      ),
      dict(
          testcase_name='GeneNamesOnly',
          requested_scorers=None,
          ontology_terms=None,
          gene_ids=None,
          gene_names=['gene_1', 'gene_2'],
          expected=(
              '(scores.metadata.gene_scorers.metadata.name = "gene_1" OR'
              ' scores.metadata.gene_scorers.metadata.name = "gene_2")'
          ),
      ),
      dict(
          testcase_name='ScorerAndOntology',
          requested_scorers=['scorer1'],
          ontology_terms=['CL:0000001'],
          gene_ids=None,
          gene_names=None,
          expected=(
              '(scores.variant_scorer.name = "scorer1") AND'
              ' (-scores.metadata.tracks.metadata.ontology_term:* OR'
              ' ((scores.metadata.tracks.metadata.ontology_term.ontology_type ='
              ' ONTOLOGY_TYPE_CL AND'
              ' scores.metadata.tracks.metadata.ontology_term.id = 1)))'
          ),
      ),
      dict(
          testcase_name='ScorerAndGene',
          requested_scorers=['scorer1'],
          ontology_terms=None,
          gene_ids=['gene_1'],
          gene_names=None,
          expected=(
              '(scores.variant_scorer.name = "scorer1") AND'
              ' (scores.metadata.gene_scorers.metadata.gene_id = "gene_1")'
          ),
      ),
      dict(
          testcase_name='All',
          requested_scorers=['scorer1'],
          ontology_terms=['CL:0000001'],
          gene_ids=['gene_1'],
          gene_names=['gene_2'],
          expected=(
              '(scores.variant_scorer.name = "scorer1") AND'
              ' (-scores.metadata.tracks.metadata.ontology_term:* OR'
              ' ((scores.metadata.tracks.metadata.ontology_term.ontology_type ='
              ' ONTOLOGY_TYPE_CL AND'
              ' scores.metadata.tracks.metadata.ontology_term.id = 1))) AND'
              ' (scores.metadata.gene_scorers.metadata.gene_id = "gene_1" OR'
              ' scores.metadata.gene_scorers.metadata.name = "gene_2")'
          ),
      ),
  )
  def test_build_filter(
      self, requested_scorers, ontology_terms, gene_ids, gene_names, expected
  ):
    result = atlas_utils.build_filter(
        requested_scorers=requested_scorers,
        ontology_terms=ontology_terms,
        gene_ids=gene_ids,
        gene_names=gene_names,
    )
    self.assertEqual(result, expected)


if __name__ == '__main__':
  absltest.main()
