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
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.models.v2 import junction_data_utils
import numpy as np
import pandas as pd


class JunctionDataUtilsTest(parameterized.TestCase):

  def _assert_junction_data_equal(
      self,
      actual: junction_data.JunctionData,
      expected: junction_data.JunctionData,
      msg,
  ) -> None:
    pd.testing.assert_frame_equal(actual.metadata, expected.metadata)
    np.testing.assert_array_equal(actual.junctions, expected.junctions)
    np.testing.assert_array_equal(actual.values, expected.values)
    self.assertEqual(actual.interval, expected.interval, msg)
    self.assertEqual(actual.uns, expected.uns, msg)

  def setUp(self):
    super().setUp()
    self.addTypeEqualityFunc(
        junction_data.JunctionData, self._assert_junction_data_equal
    )

  @parameterized.parameters([None, genome.Interval('chr1', 0, 100)])
  def test_proto_roundtrip(self, interval: genome.Interval | None):
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
        genome.Interval('chr1', 10, 11, '-'),
    ])
    metadata = pd.DataFrame({
        'name': ['foo', 'bar'],
        'biosample_curie': ['UBERON:0000005', 'UBERON:0000006'],
        'imputation_mode': ['imputed', 'observed'],
        'biosample_type': ['tissue', 'cell_line'],
        'biosample_name': ['Liver', 'Kidney'],
        'Assay title': ['polyA plus RNA-seq', 'polyA plus RNA-seq'],
        'data_source': ['encode', 'encode'],
    })
    values = np.arange(8).reshape((4, 2)).astype(np.float32)
    junctions = junction_data.JunctionData(
        junctions, values, metadata, interval
    )
    proto = junction_data_utils.to_protos(junctions)
    round_trip = junction_data_utils.from_protos(proto)
    self.assertEqual(junctions, round_trip)

  def test_from_protos_with_interval(self):
    metadata = pd.DataFrame({
        'name': ['foo', 'bar'],
        'biosample_curie': ['UBERON:0000005', 'UBERON:0000006'],
        'imputation_mode': ['imputed', 'observed'],
        'biosample_type': ['tissue', 'cell_line'],
        'biosample_name': ['Liver', 'Kidney'],
        'Assay title': ['polyA plus RNA-seq', 'polyA plus RNA-seq'],
        'data_source': ['encode', 'encode'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
    ])
    values = np.arange(4).reshape((2, 2)).astype(np.float32)
    proto = junction_data_utils.to_protos(
        junction_data.JunctionData(junctions, values, metadata)
    )
    interval = genome.Interval('chr1', 0, 100)
    round_trip = junction_data_utils.from_protos(proto, interval=interval)
    expected = junction_data.JunctionData(junctions, values, metadata, interval)
    self.assertEqual(round_trip, expected)


if __name__ == '__main__':
  absltest.main()
