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
import numpy as np
import pandas as pd

_G1 = gene_expression_data.GeneExpressionData(
    values=np.arange(6, dtype=np.float32).reshape((2, 3)),
    track_metadata=pd.DataFrame(
        {'name': ['t1', 't2', 't3'], 'strand': ['+', '+', '+']}
    ),
    gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
    interval=genome.Interval('chr1', 0, 1000),
)
_G2 = gene_expression_data.GeneExpressionData(
    values=np.arange(8, dtype=np.float32).reshape((2, 4)) + 6,
    track_metadata=pd.DataFrame(
        {'name': ['t4', 't5', 't6', 't7'], 'strand': ['-', '-', '-', '-']}
    ),
    gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
    interval=genome.Interval('chr1', 0, 1000),
)
_G3_DIFFERENT_INTERVAL = gene_expression_data.GeneExpressionData(
    values=_G2.values,
    track_metadata=_G2.track_metadata,
    gene_metadata=_G2.gene_metadata,
    interval=genome.Interval('chr1', 0, 2000),
)
_G4_DIFFERENT_GENE_METADATA = gene_expression_data.GeneExpressionData(
    values=_G1.values,
    track_metadata=_G1.track_metadata,
    gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g3']}),
    interval=_G1.interval,
)


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

  def test_invalid_num_tracks(self):
    with self.assertRaisesRegex(
        ValueError,
        'Number of tracks 7 and track metadata 0 do not match.',
    ):
      gene_expression_data.GeneExpressionData(
          values=np.zeros((2, 7)),
          track_metadata=pd.DataFrame(),
          gene_metadata=pd.DataFrame({'gene_id': ['1', '2']}),
          interval=genome.Interval('chr1', 0, 2048),
      )

  def test_invalid_num_genes(self):
    with self.assertRaisesRegex(
        ValueError,
        'Number of genes 1 and gene metadata 2 do not match.',
    ):
      gene_expression_data.GeneExpressionData(
          values=np.zeros((1, 7)),
          track_metadata=pd.DataFrame(
              {'name': ['track'] * 7, 'strand': ['+'] * 7}
          ),
          gene_metadata=pd.DataFrame({'gene_id': ['1', '2']}),
          interval=genome.Interval('chr1', 0, 2048),
      )

  def test_duplicated_metadata(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Metadata contains duplicated values for \(name, strand\) tuples.',
    ):
      gene_expression_data.GeneExpressionData(
          values=np.zeros((2, 8)),
          track_metadata=pd.DataFrame({
              'name': ['track'] * 4 + ['track_0'] * 4,
              'strand': ['+', '-', '+', '-'] * 2,
          }),
          gene_metadata=pd.DataFrame({'gene_id': ['1', '2']}),
          interval=genome.Interval('chr1', 0, 2048),
      )

  def test_to_anndata(self):
    data = gene_expression_data.GeneExpressionData(
        values=np.zeros((2, 7)),
        track_metadata=pd.DataFrame(
            {'name': [f'track_{i}' for i in range(7)], 'strand': ['+'] * 7},
            index=[f'{i}' for i in range(7)],
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['1', '2']}, index=['1', '2']),
        interval=genome.Interval('chr1', 0, 2048),
    )
    adata = data.to_anndata()
    np.testing.assert_array_equal(adata.X, data.values)
    pd.testing.assert_frame_equal(adata.var, data.track_metadata)
    pd.testing.assert_frame_equal(adata.obs, data.gene_metadata)
    self.assertEqual(adata.uns, {'interval': data.interval})

  @parameterized.named_parameters(
      (
          'slice_genes',
          (['g1'], slice(None)),
          np.arange(7).reshape((1, 7)),
          pd.DataFrame({'gene_id': ['g1']}, index=[1]),
          None,  # Use original track_metadata
      ),
      (
          'slice_tracks',
          (slice(None), ['track_1', 'track_3']),
          np.array([[1, 3], [8, 10]]),
          None,  # Use original gene_metadata
          pd.DataFrame(
              {'name': ['track_1', 'track_3'], 'strand': ['+', '+']},
              index=[1, 3],
          ),
      ),
      (
          'slice_both',
          (['g2'], ['track_0', 'track_6']),
          np.array([[7, 13]]),
          pd.DataFrame({'gene_id': ['g2']}, index=[2]),
          pd.DataFrame(
              {'name': ['track_0', 'track_6'], 'strand': ['+', '+']},
              index=[0, 6],
          ),
      ),
      (
          'slice_one_gene_scalar',
          ('g1', slice(None)),
          np.arange(7).reshape((1, 7)),
          pd.DataFrame({'gene_id': ['g1']}, index=[1]),
          None,
      ),
      (
          'slice_one_track_scalar',
          (slice(None), 'track_1'),
          np.array([[1], [8]]),
          None,
          pd.DataFrame({'name': ['track_1'], 'strand': ['+']}, index=[1]),
      ),
      (
          'slice_one_value_scalar',
          ('g2', 'track_3'),
          np.array([[10]]),
          pd.DataFrame({'gene_id': ['g2']}, index=[2]),
          pd.DataFrame({'name': ['track_3'], 'strand': ['+']}, index=[3]),
      ),
      (
          'slice_one_gene_scalar_no_track_spec',
          'g1',
          np.arange(7).reshape((1, 7)),
          pd.DataFrame({'gene_id': ['g1']}, index=[1]),
          None,
      ),
      (
          'slice_genes_np_int',
          (np.array([0]), slice(None)),
          np.arange(7).reshape((1, 7)),
          pd.DataFrame({'gene_id': ['g1']}, index=[1]),
          None,
      ),
      (
          'slice_tracks_np_int',
          (slice(None), np.array([1, 3])),
          np.array([[1, 3], [8, 10]]),
          None,
          pd.DataFrame(
              {'name': ['track_1', 'track_3'], 'strand': ['+', '+']},
              index=[1, 3],
          ),
      ),
      (
          'slice_both_np_int',
          (np.array([1]), np.array([0, 6])),
          np.array([[7, 13]]),
          pd.DataFrame({'gene_id': ['g2']}, index=[2]),
          pd.DataFrame(
              {'name': ['track_0', 'track_6'], 'strand': ['+', '+']},
              index=[0, 6],
          ),
      ),
  )
  def test_getitem(
      self,
      slc: tuple[
          gene_expression_data.GeneIndex, gene_expression_data.TrackIndex
      ],
      expected_values: np.ndarray,
      expected_gene_metadata: pd.DataFrame | None,
      expected_track_metadata: pd.DataFrame | None,
  ):
    data = gene_expression_data.GeneExpressionData(
        values=np.arange(14, dtype=np.float32).reshape((2, 7)),
        track_metadata=pd.DataFrame(
            {'name': [f'track_{i}' for i in range(7)], 'strand': ['+'] * 7},
            index=list(range(7)),
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}, index=[1, 2]),
        interval=genome.Interval('chr1', 0, 2048),
    )
    if expected_gene_metadata is None:
      expected_gene_metadata = data.gene_metadata
    if expected_track_metadata is None:
      expected_track_metadata = data.track_metadata
    expected = gene_expression_data.GeneExpressionData(
        values=expected_values.astype(np.float32),
        gene_metadata=expected_gene_metadata,
        track_metadata=expected_track_metadata,
        interval=data.interval,
    )
    self.assertEqual(data[slc], expected)

  def test_get_genes_as_intervals(self):
    gene_metadata = pd.DataFrame({
        'gene_id': ['g1', 'g2'],
        'Chromosome': ['chr1', 'chr1'],
        'Start': [101, 501],
        'End': [200, 600],
    })

    data = gene_expression_data.GeneExpressionData(
        values=np.zeros((2, 1)),
        track_metadata=pd.DataFrame({'name': ['t1'], 'strand': ['+']}),
        gene_metadata=gene_metadata,
        interval=genome.Interval('chr1', 0, 1000),
    )
    intervals = data.get_genes_as_intervals()
    expected = [
        genome.Interval('chr1', 101, 200),
        genome.Interval('chr1', 501, 600),
    ]
    self.assertEqual(intervals, expected)

  def test_slice_by_interval(self):
    values = np.arange(6, dtype=np.float32).reshape((3, 2))
    gene_metadata = pd.DataFrame({
        'gene_id': ['g1', 'g2', 'g3'],
        'Chromosome': ['chr1', 'chr1', 'chr2'],
        'Start': [101, 501, 201],
        'End': [200, 600, 300],
    })

    data = gene_expression_data.GeneExpressionData(
        values=values,
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't2'], 'strand': ['+', '+']}
        ),
        gene_metadata=gene_metadata,
        interval=genome.Interval('chr1', 0, 2000),
    )
    sliced_data = data.slice_by_interval(genome.Interval('chr1', 0, 1000))
    self.assertEqual(sliced_data.interval, genome.Interval('chr1', 0, 1000))
    expected = data[[0, 1]]
    pd.testing.assert_frame_equal(
        sliced_data.track_metadata, expected.track_metadata
    )
    pd.testing.assert_frame_equal(
        sliced_data.gene_metadata, expected.gene_metadata
    )
    np.testing.assert_array_equal(sliced_data.values, expected.values)

  def test_slice_by_interval_raises_error(self):
    data = gene_expression_data.GeneExpressionData(
        values=np.zeros((1, 1)),
        track_metadata=pd.DataFrame({'name': ['t1'], 'strand': ['+']}),
        gene_metadata=pd.DataFrame({'gene_id': ['g1']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    with self.assertRaisesRegex(ValueError, 'Gene coordinates are missing'):
      data.slice_by_interval(genome.Interval('chr1', 0, 100))

  def test_filter_tracks(self):
    data = gene_expression_data.GeneExpressionData(
        values=np.arange(4, dtype=np.float32).reshape((1, 4)),
        track_metadata=pd.DataFrame({
            'name': ['t1', 't2', 't3', 't4'],
            'strand': ['+', '-', '+', '.'],
        }),
        gene_metadata=pd.DataFrame({'gene_id': ['g1']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    filtered = data.filter_tracks(np.array([True, False, True, False]))
    expected = gene_expression_data.GeneExpressionData(
        values=np.array([[0, 2]], dtype=np.float32),
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't3'], 'strand': ['+', '+']}, index=[0, 2]
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    self.assertEqual(filtered, expected)

  @parameterized.named_parameters(
      ('positive', 'filter_to_positive_strand', ['+', '+'], [0, 2]),
      ('negative', 'filter_to_negative_strand', ['-'], [1]),
      (
          'non-negative',
          'filter_to_nonnegative_strand',
          ['+', '+', '.'],
          [0, 2, 3],
      ),
      (
          'non-positive',
          'filter_to_nonpositive_strand',
          ['-', '.'],
          [1, 3],
      ),
      ('stranded', 'filter_to_stranded', ['+', '-', '+'], [0, 1, 2]),
      ('unstranded', 'filter_to_unstranded', ['.'], [3]),
  )
  def test_filter_tracks_by_strand(
      self, filter_method_name, expected_strands, track_indices
  ):
    data = gene_expression_data.GeneExpressionData(
        values=np.arange(4, dtype=np.float32).reshape((1, 4)),
        track_metadata=pd.DataFrame({
            'name': ['t1', 't2', 't3', 't4'],
            'strand': ['+', '-', '+', '.'],
        }),
        gene_metadata=pd.DataFrame({'gene_id': ['g1']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    filter_method = getattr(data, filter_method_name)
    filtered = filter_method()
    self.assertListEqual(filtered.strands.tolist(), expected_strands)
    self.assertListEqual(filtered.track_metadata.index.tolist(), track_indices)

  def test_filter_to_observed(self):
    data = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((2, 3)),
        track_metadata=pd.DataFrame({
            'name': ['t1', 't2', 't3'],
            'strand': ['+', '+', '+'],
            'imputation_mode': ['observed', 'imputed_supported', 'observed'],
        }),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    filtered = data.filter_to_observed()
    expected = gene_expression_data.GeneExpressionData(
        values=np.array([[0, 2], [3, 5]], dtype=np.float32),
        track_metadata=pd.DataFrame(
            {
                'name': ['t1', 't3'],
                'strand': ['+', '+'],
                'imputation_mode': ['observed', 'observed'],
            },
            index=[0, 2],
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    self.assertEqual(filtered, expected)

  def test_filter_to_protein_coding(self):
    data = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((3, 2)),
        track_metadata=pd.DataFrame({
            'name': ['t1', 't2'],
            'strand': ['+', '+'],
        }),
        gene_metadata=pd.DataFrame({
            'gene_id': ['g1', 'g2', 'g3'],
            'gene_type': ['protein_coding', 'lncRNA', 'protein_coding'],
        }),
        interval=genome.Interval('chr1', 0, 1000),
    )
    filtered = data.filter_to_protein_coding()
    expected = gene_expression_data.GeneExpressionData(
        values=np.array([[0, 1], [4, 5]], dtype=np.float32),
        track_metadata=pd.DataFrame({
            'name': ['t1', 't2'],
            'strand': ['+', '+'],
        }),
        gene_metadata=pd.DataFrame(
            {
                'gene_id': ['g1', 'g3'],
                'gene_type': ['protein_coding', 'protein_coding'],
            },
            index=[0, 2],
        ),
        interval=genome.Interval('chr1', 0, 1000),
    )
    self.assertEqual(filtered, expected)

  def test_filter_to_protein_coding_missing_column_raises(self):
    data = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((3, 2)),
        track_metadata=pd.DataFrame({
            'name': ['t1', 't2'],
            'strand': ['+', '+'],
        }),
        gene_metadata=pd.DataFrame({
            'gene_id': ['g1', 'g2', 'g3'],
        }),
        interval=genome.Interval('chr1', 0, 1000),
    )
    with self.assertRaisesRegex(
        ValueError, "Column 'gene_type' not found in gene_metadata"
    ):
      data.filter_to_protein_coding()

  def test_resize(self):
    gene_metadata = pd.DataFrame({
        'gene_id': ['g1', 'g2', 'g3'],
        'Chromosome': ['chr1', 'chr1', 'chr1'],
        'Start': [101, 501, 1201],
        'End': [200, 600, 1300],
    })
    data = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((3, 2)),
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't2'], 'strand': ['+', '+']}
        ),
        gene_metadata=gene_metadata,
        interval=genome.Interval('chr1', 0, 2000),
    )
    resized = data.resize(width=1000)
    self.assertEqual(resized.interval, data.interval.resize(1000))
    expected = data[[1, 2]]
    pd.testing.assert_frame_equal(
        resized.track_metadata, expected.track_metadata
    )
    pd.testing.assert_frame_equal(resized.gene_metadata, expected.gene_metadata)
    np.testing.assert_array_equal(resized.values, expected.values)

  def test_add(self):
    g1 = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((2, 3)),
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't2', 't3'], 'strand': ['+', '+', '+']}
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    g2 = gene_expression_data.GeneExpressionData(
        values=np.ones((2, 3), dtype=np.float32),
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't2', 't3'], 'strand': ['+', '+', '+']}
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    expected = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((2, 3)) + 1.0,
        track_metadata=g1.track_metadata,
        gene_metadata=g1.gene_metadata,
        interval=g1.interval,
    )
    self.assertEqual(g1 + g2, expected)

  def test_sub(self):
    g1 = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((2, 3)),
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't2', 't3'], 'strand': ['+', '+', '+']}
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    g2 = gene_expression_data.GeneExpressionData(
        values=np.ones((2, 3), dtype=np.float32),
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't2', 't3'], 'strand': ['+', '+', '+']}
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    expected = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((2, 3)) - 1.0,
        track_metadata=g1.track_metadata,
        gene_metadata=g1.gene_metadata,
        interval=g1.interval,
    )
    self.assertEqual(g1 - g2, expected)

  def test_add_raises_type_error(self):
    with self.assertRaisesRegex(TypeError, 'Unsupported type'):
      _ = _G1 + 1  # pyrefly: ignore[unsupported-operation]

  @parameterized.named_parameters(
      dict(
          testcase_name='different_interval',
          other=_G3_DIFFERENT_INTERVAL,
          expected_regex='Intervals must match',
      ),
      dict(
          testcase_name='different_shape',
          other=_G2,
          expected_regex='Shapes must match',
      ),
      dict(
          testcase_name='different_gene_metadata',
          other=_G4_DIFFERENT_GENE_METADATA,
          expected_regex='Gene Metadata must match',
      ),
      dict(
          testcase_name='different_track_metadata_shape',
          other=gene_expression_data.GeneExpressionData(
              values=_G1.values,
              track_metadata=pd.DataFrame({
                  'name': _G1.track_metadata.name,
                  'strand': _G1.track_metadata.strand,
                  'extra': [1, 2, 3],
              }),
              gene_metadata=_G1.gene_metadata,
              interval=_G1.interval,
          ),
          expected_regex='Track Metadata shapes must match',
      ),
  )
  def test_add_raises_value_error(self, other, expected_regex):
    with self.assertRaisesRegex(ValueError, expected_regex):
      _ = _G1 + other

  @parameterized.named_parameters(
      dict(
          testcase_name='no_extra_metadata',
          geds=[_G1, _G2],
          extra_metadata_name_and_keys=None,
          expected_track_metadata=pd.concat(
              [_G1.track_metadata, _G2.track_metadata], ignore_index=True
          ),
      ),
      dict(
          testcase_name='with_extra_metadata',
          geds=[_G1, _G2],
          extra_metadata_name_and_keys=('sample', ['s1', 's2']),
          expected_track_metadata=pd.DataFrame({
              'sample': ['s1'] * 3 + ['s2'] * 4,
              'name': ['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
              'strand': ['+'] * 3 + ['-'] * 4,
          }),
      ),
  )
  def test_concat(
      self, geds, extra_metadata_name_and_keys, expected_track_metadata
  ):
    concatenated = gene_expression_data.concat(
        geds, extra_metadata_name_and_keys=extra_metadata_name_and_keys
    )
    expected = gene_expression_data.GeneExpressionData(
        values=np.hstack([g.values for g in geds]),
        track_metadata=expected_track_metadata,
        gene_metadata=geds[0].gene_metadata.copy(),
        interval=geds[0].interval,
    )
    self._assert_gene_expression_data_equal(
        concatenated, expected, msg='Gene expression data concatenation failed.'
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_input',
          ged_list=[],
          extra_metadata_name_and_keys=None,
          expected_regex='gene_expression_datas cannot be empty.',
      ),
      dict(
          testcase_name='different_intervals',
          ged_list=[_G1, _G3_DIFFERENT_INTERVAL],
          extra_metadata_name_and_keys=None,
          expected_regex='Gene expression data contain multiple intervals',
      ),
      dict(
          testcase_name='different_gene_metadata',
          ged_list=[_G1, _G4_DIFFERENT_GENE_METADATA],
          extra_metadata_name_and_keys=None,
          expected_regex='Gene expression data contain different gene_metadata',
      ),
      dict(
          testcase_name='mismatch_extra_metadata',
          ged_list=[_G1, _G2],
          extra_metadata_name_and_keys=('sample', ['s1']),
          expected_regex=(
              'Second element of extra_metadata_name_and_keys must be the same'
              ' length as gene_expression_datas'
          ),
      ),
  )
  def test_concat_raises_error(
      self, ged_list, extra_metadata_name_and_keys, expected_regex
  ):
    with self.assertRaisesRegex(ValueError, expected_regex):
      gene_expression_data.concat(
          ged_list, extra_metadata_name_and_keys=extra_metadata_name_and_keys
      )

  def test_with_name_suffix(self):
    gdata = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((2, 3)),
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't2', 't3'], 'strand': ['+', '+', '+']}
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    suffixed = gdata.with_name_suffix('_suffix')
    self.assertListEqual(
        list(suffixed.names), ['t1_suffix', 't2_suffix', 't3_suffix']
    )
    # Check that original is not modified
    self.assertListEqual(list(gdata.names), ['t1', 't2', 't3'])

  def test_with_metadata_column(self):
    gdata = gene_expression_data.GeneExpressionData(
        values=np.arange(6, dtype=np.float32).reshape((2, 3)),
        track_metadata=pd.DataFrame(
            {'name': ['t1', 't2', 't3'], 'strand': ['+', '+', '+']}
        ),
        gene_metadata=pd.DataFrame({'gene_id': ['g1', 'g2']}),
        interval=genome.Interval('chr1', 0, 1000),
    )
    updated = gdata.with_metadata_column('biosample_name', 'cell_0')
    self.assertIn('biosample_name', updated.track_metadata.columns)
    self.assertListEqual(
        list(updated.track_metadata['biosample_name']),
        ['cell_0', 'cell_0', 'cell_0'],
    )
    self.assertNotIn('biosample_name', gdata.track_metadata.columns)


if __name__ == '__main__':
  absltest.main()
