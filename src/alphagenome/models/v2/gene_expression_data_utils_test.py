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
from alphagenome.data import gene_expression_data
from alphagenome.data import genome
from alphagenome.models.v2 import gene_expression_data_utils
import numpy as np
import pandas as pd


class GeneExpressionDataTest(parameterized.TestCase):

  def _assert_gene_expression_data_equal(
      self,
      actual: gene_expression_data.GeneExpressionData,
      expected: gene_expression_data.GeneExpressionData,
      msg,
  ) -> None:
    pd.testing.assert_frame_equal(
        actual.track_metadata, expected.track_metadata
    )
    pd.testing.assert_frame_equal(actual.gene_metadata, expected.gene_metadata)
    np.testing.assert_array_equal(actual.values, expected.values)
    self.assertEqual(actual.interval, expected.interval, msg)

  def setUp(self):
    super().setUp()
    self.addTypeEqualityFunc(
        gene_expression_data.GeneExpressionData,
        self._assert_gene_expression_data_equal,
    )

  @parameterized.product(
      include_gene_name=[True, False],
      include_gene_strand=[True, False],
      include_gene_type=[True, False],
  )
  def test_proto_roundtrip(
      self,
      include_gene_name: bool,
      include_gene_strand: bool,
      include_gene_type: bool,
  ):
    interval = genome.Interval('chr1', 0, 2048)
    gene_metadata = pd.DataFrame({
        'gene_id': ['1', '2'],
        'Chromosome': ['chr1', 'chr2'],
        'Start': [20, 40],
        'End': [200, 400],
    })
    if include_gene_strand:
      gene_metadata['strand'] = ['+', '-']
    if include_gene_name:
      gene_metadata['gene_name'] = ['foo', 'bar']
    if include_gene_type:
      gene_metadata['gene_type'] = ['protein_coding', 'protein_coding']

    track_metadata = pd.DataFrame({
        'name': [f'track_{i}' for i in range(7)],
        'strand': ['.', '+', '-', '.', '+', '-', '.'],
        'biosample_curie': ['UBERON:0000005'] * 7,
        'imputation_mode': ['observed'] * 7,
        'biosample_type': ['tissue'] * 7,
        'biosample_name': ['biosample'] * 7,
        'Assay title': [
            'foo',
            'bar',
            'baz',
            'qux',
            'quux',
            'wuz',
            'buz',
        ],
    })
    values = np.arange(14).reshape((2, 7)).astype(np.float32)
    data = gene_expression_data.GeneExpressionData(
        values=values,
        track_metadata=track_metadata,
        gene_metadata=gene_metadata,
        interval=interval,
    )
    proto = gene_expression_data_utils.to_protos(data)
    round_trip = gene_expression_data_utils.from_protos(proto)
    self.assertEqual(round_trip, data)


if __name__ == '__main__':
  absltest.main()
