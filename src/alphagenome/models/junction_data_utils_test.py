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

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from alphagenome.data import genome, junction_data
from alphagenome.models import junction_data_utils


class JunctionDataUtilsTest(parameterized.TestCase):

  def _assert_junction_data_equal(
      self,
      actual: junction_data.JunctionData,
      expected: junction_data.JunctionData,
      msg: ...,
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
        'ontology_curie': ['UBERON:0000005', 'UBERON:0000006'],
        'biosample_name': ['Liver', 'Kidney'],
        'biosample_type': ['tissue', 'cell_line'],
        'biosample_life_stage': ['adult', 'adult'],
        'gtex_tissue': ['liver', 'kidney'],
        'data_source': ['encode', 'encode'],
        'Assay title': ['polyA plus RNA-seq', 'polyA plus RNA-seq'],
    })
    values = np.arange(8).reshape((4, 2)).astype(np.float32)
    junctions = junction_data.JunctionData(
        junctions, values, metadata, interval
    )
    proto, chunks = junction_data_utils.to_protos(junctions)
    round_trip = junction_data_utils.from_protos(proto, chunks)
    self.assertEqual(junctions, round_trip)

  def test_from_protos_with_interval(self):
    metadata = pd.DataFrame({
        'name': ['foo', 'bar'],
        'ontology_curie': ['UBERON:0000005', 'UBERON:0000006'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
    ])
    values = np.arange(4).reshape((2, 2)).astype(np.float32)
    proto, chunks = junction_data_utils.to_protos(
        junction_data.JunctionData(junctions, values, metadata)
    )
    interval = genome.Interval('chr1', 0, 100)
    round_trip = junction_data_utils.from_protos(
        proto, chunks, interval=interval
    )
    expected = junction_data.JunctionData(junctions, values, metadata, interval)
    self.assertEqual(round_trip, expected)


if __name__ == '__main__':
  absltest.main()
