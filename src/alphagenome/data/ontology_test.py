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

from absl.testing import absltest, parameterized

from alphagenome.data import ontology


class OntologyTest(parameterized.TestCase):

  def test_ontology_term_by_curie(self):
    term = ontology.from_curie('UBERON:0000005')
    self.assertEqual(term.ontology_curie, 'UBERON:0000005')
    self.assertEqual(term.type, ontology.OntologyType.UBERON)
    self.assertEqual(term.id, 5)

  @parameterized.parameters(
      ('UBERON:0000005',),
      ('CL:0000005',),
      ('EFO:0000005',),
      ('NTR:0000512',),
  )
  def test_proto_roundtrip(self, curie: str):
    term = ontology.from_curie(curie)
    proto = term.to_proto()
    round_trip = ontology.from_proto(proto)
    self.assertEqual(term, round_trip)

  def test_ontology_terms_from_curies(self):
    terms = ontology.from_curies(['UBERON:0000005', 'CL:0000005'])
    self.assertEqual(
        terms,
        [
            ontology.OntologyTerm(ontology.OntologyType.UBERON, 5),
            ontology.OntologyTerm(ontology.OntologyType.CL, 5),
        ],
    )

  @parameterized.named_parameters(
      ('invalid_curie', 'invalid:0000005', 'Cannot parse ontology_type'),
      ('lowercase_curie', 'uberon:0000005', 'Cannot parse ontology_type'),
      ('too_many_colons', 'UBERON:0000005:0000005', 'Invalid ontology_curie'),
      ('missing_colon', 'foo_bar', "Invalid ontology_curie='foo_bar'"),
      (
          'invalid_id',
          'UBERON:foo',
          r"invalid literal for int\(\) with base 10: 'foo'",
      ),
  )
  def test_invalid_curie_raises_error(self, curie: str, expected_error: str):
    with self.assertRaisesRegex(ValueError, expected_error):
      ontology.from_curie(curie)


if __name__ == '__main__':
  absltest.main()
