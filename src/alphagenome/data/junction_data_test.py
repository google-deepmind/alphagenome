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

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome
from alphagenome.data import junction_data
import numpy as np
import pandas as pd


class JunctionDataTest(parameterized.TestCase):

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

  def test_invalid_metadata_raises(self):
    metadata = pd.DataFrame({'name': ['foo', 'bar', 'baz']})
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
        genome.Interval('chr1', 10, 11, '-'),
    ])
    values = np.zeros((4, 2), dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, 'Number of tracks .* do not match.'
    ):
      junction_data.JunctionData(junctions, values, metadata)

  def test_invalid_junctions_raises(self):
    metadata = pd.DataFrame({'name': ['foo', 'bar', 'baz']})
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
    ])
    values = np.zeros((4, 3), dtype=np.float32)

    with self.assertRaises(TypeError):
      junction_data.JunctionData(junctions, values, metadata)

  def test_duplicate_name_raises(self):
    metadata = pd.DataFrame({'name': ['foo', 'foo']})
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '+'),
    ])
    values = np.zeros((2, 2), dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, 'Metadata contain duplicated names'
    ):
      junction_data.JunctionData(junctions, values, metadata)

  def test_filter_by_ontology(self):
    metadata = pd.DataFrame({
        'name': ['foo', 'bar'],
        'ontology_curie': ['UBERON:0000005', 'UBERON:0000006'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
    ])
    values = np.arange(4).reshape((2, 2)).astype(np.float32)
    filtered = junction_data.JunctionData(
        junctions, values, metadata
    ).filter_by_ontology('UBERON:0000005')

    expected = junction_data.JunctionData(
        junctions=np.array([
            genome.Interval('chr1', 10, 11, '+'),
            genome.Interval('chr1', 10, 11, '-'),
        ]),
        metadata=pd.DataFrame(
            {'name': ['foo'], 'ontology_curie': ['UBERON:0000005']}
        ),
        values=np.array([[0], [2]], dtype=np.float32),
    )
    self.assertEqual(filtered, expected)

  def test_filter_to_strand(self):
    metadata = pd.DataFrame({
        'name': ['foo', 'bar'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
        genome.Interval('chr2', 10, 11, '-'),
        genome.Interval('chr2', 1, 2, '+'),
    ])
    values = np.arange(8).reshape((4, 2)).astype(np.float32)

    with self.subTest('positive_strand'):
      filtered = junction_data.JunctionData(
          junctions, values, metadata
      ).filter_to_positive_strand()

      expected = junction_data.JunctionData(
          junctions=np.array([
              genome.Interval('chr1', 10, 11, '+'),
              genome.Interval('chr2', 1, 2, '+'),
          ]),
          metadata=pd.DataFrame({'name': ['foo', 'bar']}),
          values=np.array([[0, 1], [6, 7]], dtype=np.float32),
      )
      self.assertEqual(filtered, expected)

    with self.subTest('negative_strand'):
      filtered = junction_data.JunctionData(
          junctions, values, metadata
      ).filter_to_negative_strand()
      expected = junction_data.JunctionData(
          junctions=np.array([
              genome.Interval('chr1', 10, 11, '-'),
              genome.Interval('chr2', 10, 11, '-'),
          ]),
          metadata=pd.DataFrame({'name': ['foo', 'bar']}),
          values=np.array([[2, 3], [4, 5]], dtype=np.float32),
      )
      self.assertEqual(filtered, expected)

  def test_filter_by_tissue(self):
    metadata = pd.DataFrame({
        'name': ['foo', 'bar'],
        'gtex_tissue': ['Brain_Cerebellum', 'Adipose_Subcutaneous'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
    ])
    values = np.arange(4).reshape((2, 2)).astype(np.float32)
    filtered = junction_data.JunctionData(
        junctions, values, metadata
    ).filter_by_tissue('Brain_Cerebellum')

    expected = junction_data.JunctionData(
        junctions=np.array([
            genome.Interval('chr1', 10, 11, '+'),
            genome.Interval('chr1', 10, 11, '-'),
        ]),
        metadata=pd.DataFrame(
            {'name': ['foo'], 'gtex_tissue': ['Brain_Cerebellum']}
        ),
        values=np.array([[0], [2]], dtype=np.float32),
    )

    self.assertEqual(filtered, expected)

  def test_intersect_with_interval(self):
    metadata = pd.DataFrame({'name': ['foo', 'bar']})
    junctions = np.array([
        genome.Interval('chr1', 18, 20, '+'),
        genome.Interval('chr1', 15, 18, '+'),
        genome.Interval('chr1', 10, 11, '-'),
        genome.Interval('chr1', 1, 2, '+'),
    ])
    values = np.arange(8).reshape((4, 2)).astype(np.float32)
    interval = genome.Interval('chr1', 8, 17)
    junctions = junction_data.JunctionData(junctions, values, metadata)
    filtered = junctions.intersect_with_interval(interval)
    expected = junction_data.JunctionData(
        junctions=np.array([
            genome.Interval('chr1', 15, 18, '+'),
            genome.Interval('chr1', 10, 11, '-'),
        ]),
        metadata=pd.DataFrame({'name': ['foo', 'bar']}),
        values=np.array([[2, 3], [4, 5]], dtype=np.float32),
    )
    self.assertEqual(filtered, expected)

  @parameterized.parameters([
      dict(
          name='foo',
          k_threshold=0.0,
          strand='+',
          expected=[
              genome.Junction('chr1', 10, 11, '+', k=0),
              genome.Junction('chr2', 10, 11, '+', k=4),
          ],
      ),
      dict(
          name='foo',
          k_threshold=0.0,
          strand='-',
          expected=[
              genome.Junction('chr1', 10, 11, '-', k=2),
              genome.Junction('chr2', 10, 11, '-', k=6),
          ],
      ),
      dict(
          name='bar',
          k_threshold=4,
          strand='+',
          expected=[
              genome.Junction('chr2', 10, 11, '+', k=5),
          ],
      ),
      dict(
          name='unknown_track_name',
          k_threshold=None,
          strand='+',
          expected=[],
      ),
  ])
  def test_get_junctions_to_plot(self, name, k_threshold, strand, expected):
    metadata = pd.DataFrame({
        'name': ['foo', 'bar'],
        'ontology_curie': ['UBERON:0000005', 'UBERON:0000006'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
        genome.Interval('chr1', 10, 11, '-'),
        genome.Interval('chr2', 10, 11, '+'),
        genome.Interval('chr2', 10, 11, '-'),
    ])
    values = np.arange(8).reshape((4, 2)).astype(np.float32)
    predictions = junction_data.JunctionData(junctions, values, metadata)

    with self.subTest('test_junctions'):
      actual = junction_data.get_junctions_to_plot(
          predictions=predictions,
          name=name,
          strand=strand,
          k_threshold=k_threshold,
      )
      self.assertEqual(actual, expected)
    with self.subTest('test_normalize_values'):
      self.assertEqual(
          predictions.normalize_values(total_k=10.0).values.sum(), 10.0
      )

  def test_with_name_suffix(self):
    metadata = pd.DataFrame({
        'name': ['track_1', 'track_2'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
    ])
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    jdata = junction_data.JunctionData(junctions, values, metadata)
    suffixed = jdata.with_name_suffix('_suff')
    self.assertListEqual(list(suffixed.names), ['track_1_suff', 'track_2_suff'])
    self.assertListEqual(list(jdata.names), ['track_1', 'track_2'])

  def test_with_metadata_column(self):
    metadata = pd.DataFrame({
        'name': ['track_1', 'track_2'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
    ])
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    jdata = junction_data.JunctionData(junctions, values, metadata)
    updated = jdata.with_metadata_column('biosample_name', 'cell_0')
    self.assertIn('biosample_name', updated.metadata.columns)
    self.assertListEqual(
        list(updated.metadata['biosample_name']), ['cell_0', 'cell_0']
    )
    self.assertNotIn('biosample_name', jdata.metadata.columns)

  def test_with_metadata_column_series(self):
    metadata = pd.DataFrame({
        'name': ['track_1', 'track_2'],
    })
    junctions = np.array([
        genome.Interval('chr1', 10, 11, '+'),
    ])
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    jdata = junction_data.JunctionData(junctions, values, metadata)
    series_value = pd.Series(['a', 'b'])
    updated = jdata.with_metadata_column('new_col', series_value)
    self.assertIn('new_col', updated.metadata.columns)
    self.assertListEqual(list(updated.metadata['new_col']), ['a', 'b'])
    self.assertNotIn('new_col', jdata.metadata.columns)


if __name__ == '__main__':
  absltest.main()
