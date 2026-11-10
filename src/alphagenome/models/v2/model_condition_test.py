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

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import ontology
from alphagenome.models.v2 import model_condition

from alphagenome.protos.v2 import dna_model_pb2


class ConditionTest(parameterized.TestCase):

  def test_from_curies(self):
    condition_metadata = model_condition.from_curies(
        ['UBERON:0002371', 'CL:0000037']
    )
    self.assertEqual(
        condition_metadata,
        model_condition.ModelCondition(
            clo=None,
            uberon=ontology.from_curie('UBERON:0002371'),
            cl=ontology.from_curie('CL:0000037'),
            efo=None,
            ntr=None,
        ),
    )

  def test_from_curies_duplicate_raises(self):
    with self.assertRaisesRegex(ValueError, 'Duplicate ontology types*'):
      model_condition.from_curies(['UBERON:0002371', 'UBERON:0002190'])

  def test_ontology_terms(self):
    condition_metadata = model_condition.ModelCondition(
        clo=ontology.from_curie('CLO:0021752'),
        uberon=ontology.from_curie('UBERON:0002371'),
        cl=ontology.from_curie('CL:0000037'),
        efo=None,
        ntr=None,
    )
    self.assertSequenceEqual(
        condition_metadata.ontology_terms,
        (
            ontology.from_curie('CLO:0021752'),
            ontology.from_curie('UBERON:0002371'),
            ontology.from_curie('CL:0000037'),
        ),
    )

  def test_ontology_curies(self):
    condition_metadata = model_condition.ModelCondition(
        clo=ontology.from_curie('CLO:0021752'),
        uberon=ontology.from_curie('UBERON:0002371'),
        cl=ontology.from_curie('CL:0000037'),
        efo=None,
        ntr=None,
    )
    self.assertSequenceEqual(
        condition_metadata.ontology_curies,
        ('CLO:0021752', 'UBERON:0002371', 'CL:0000037'),
    )

  def test_str(self):
    condition_metadata = model_condition.ModelCondition(
        clo=ontology.from_curie('CLO:0021752'),
        uberon=ontology.from_curie('UBERON:0002371'),
        cl=ontology.from_curie('CL:0000037'),
        efo=None,
        ntr=None,
    )
    self.assertEqual(
        str(condition_metadata), 'CLO:0021752_UBERON:0002371_CL:0000037'
    )

  def test_from_proto(self):
    proto = dna_model_pb2.Condition(
        clo=21752, uberon=2371, cl=37, efo=None, ntr=452
    )
    condition_metadata = model_condition.from_proto(proto)
    expected_condition_metadata = model_condition.ModelCondition(
        clo=ontology.from_curie('CLO:0021752'),
        uberon=ontology.from_curie('UBERON:0002371'),
        cl=ontology.from_curie('CL:0000037'),
        efo=None,
        ntr=ontology.from_curie('NTR:0000452'),
    )
    self.assertEqual(condition_metadata, expected_condition_metadata)

  def test_proto_roundtrip(self):
    condition_metadata = model_condition.ModelCondition(
        clo=ontology.from_curie('CLO:0021752'),
        uberon=None,
        cl=ontology.from_curie('CL:0000037'),
        efo=None,
        ntr=None,
    )
    proto = condition_metadata.to_proto()
    new_condition_metadata = model_condition.from_proto(proto)
    self.assertEqual(condition_metadata, new_condition_metadata)

  def test_from_proto_empty(self):
    proto = dna_model_pb2.Condition()
    condition_metadata = model_condition.from_proto(proto)
    expected_condition_metadata = model_condition.ModelCondition()
    self.assertEqual(condition_metadata, expected_condition_metadata)


if __name__ == '__main__':
  absltest.main()
