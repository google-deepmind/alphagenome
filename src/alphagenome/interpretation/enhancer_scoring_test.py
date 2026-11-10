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

import math
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.interpretation import enhancer_scoring
from alphagenome.models.v2 import dna_output
import numpy as np
import pandas as pd

_INTERVAL = genome.Interval('chr1', 10_000, 20_000)
_GENE_ID = 'ENSG00000000001'
_GENE_ID_2 = 'ENSG00000000002'
_GENE_INTERVAL = genome.Interval('chr1', 15_000, 16_000, '+')


def _make_output(
    gradients: np.ndarray,
    interval: genome.Interval = _INTERVAL,
    *,
    gene_id: str = _GENE_ID,
    gene_name: str | None = 'GENE1',
    biosample_name: str = 'K562',
    output_type: dna_output.OutputType = (
        dna_output.OutputType.SC_GENE_EXPRESSION
    ),
) -> dna_output.InputGradientsOutput:
  return dna_output.InputGradientsOutput(
      gradients=gradients.astype(np.float32),
      interval=interval,
      scalar_score=0.0,
      target_gene=dna_output.TargetGeneMetadata(
          gene_id=gene_id, interval=_GENE_INTERVAL, gene_name=gene_name
      ),
      track_metadata=pd.DataFrame(
          {'name': ['track_0'], 'biosample_name': [biosample_name]}
      ),
      output_type=output_type,
  )


def _make_gtf() -> pd.DataFrame:
  return pd.DataFrame({
      'Chromosome': ['chr1', 'chr1', 'chr1'],
      'Start': [15_000, 15_000, 17_000],
      'End': [16_000, 15_200, 18_500],
      'Strand': ['+', '+', '-'],
      'Feature': ['transcript', 'exon', 'transcript'],
      'gene_id': [f'{_GENE_ID}.5', f'{_GENE_ID}.5', f'{_GENE_ID_2}.3'],
      'tag': ['MANE_Select', 'MANE_Select', 'MANE_Select'],
  })


def _make_acc_output(
    values: np.ndarray,
    interval: genome.Interval = _INTERVAL,
    *,
    biosample_name: str | None = 'K562',
    output_type: dna_output.OutputType = dna_output.OutputType.SC_ATAC,
    imputation_modes: list[str] | None = None,
    resolution: int = 1,
) -> dna_output.Output:
  arr = values[:, None] if values.ndim == 1 else values
  meta: dict[str, list[str]] = {
      'name': [f'ch_{i}' for i in range(arr.shape[1])],
      'strand': ['.'] * arr.shape[1],
  }
  if biosample_name is not None:
    meta['biosample_name'] = [biosample_name] * arr.shape[1]
  if imputation_modes is not None:
    meta['imputation_mode'] = imputation_modes
  td = track_data.TrackData(
      values=arr.astype(np.float32),
      metadata=pd.DataFrame(meta),
      resolution=resolution,
      interval=interval,
  )
  return (
      dna_output.Output(sc_atac=td)
      if output_type == dna_output.OutputType.SC_ATAC
      else dna_output.Output(atac=td)
  )


class ScoreEnhancersTest(parameterized.TestCase):

  def test_output_frame(self):
    # A uniform gradient makes every scored enhancer's Gaussian-weighted mean
    # constant, including windows clipped at the interval edge.
    gradients = np.zeros((_INTERVAL.width, 4), np.float32)
    gradients[:, 2] = -0.5
    enhancers = [
        genome.Interval('chr1', 15_000, 15_500, name='center'),
        genome.Interval('chr1', 10_000, 10_200, name='left_edge'),
        genome.Interval('chr2', 15_000, 15_500, name='other_chromosome'),
        genome.Interval('chr1', 30_000, 30_500, name='outside'),
    ]
    df = enhancer_scoring.score_enhancers(_make_output(gradients), enhancers)
    expected = pd.DataFrame({
        'chromosome': ['chr1', 'chr1', 'chr2', 'chr1'],
        'start': [15_000, 10_000, 15_000, 30_000],
        'end': [15_500, 10_200, 15_500, 30_500],
        'name': ['center', 'left_edge', 'other_chromosome', 'outside'],
        'biosample_name': ['K562'] * 4,
        'gene_id': [_GENE_ID] * 4,
        'gradient_score': [0.5, 0.5, np.nan, np.nan],
        'is_scored': [True, True, False, False],
        'gene_name': ['GENE1'] * 4,
    })
    pd.testing.assert_frame_equal(df, expected)

  def test_abs_applied_after_nucleotide_sum(self):
    gradients = np.zeros((_INTERVAL.width, 4), np.float32)
    gradients[:, 0] = 1.0
    gradients[:, 1] = -1.0
    df = enhancer_scoring.score_enhancers(
        _make_output(gradients), [genome.Interval('chr1', 15_000, 15_001)]
    )
    self.assertEqual(df['gradient_score'].iloc[0], 0.0)

  def test_single_impulse_analytical_score(self):
    # Closed-form Gaussian weight ratio for a single impulse at offset d=30
    # inside a clipped edge window [0, 151) around midpoint m=50 (half=100).
    small_interval = genome.Interval('chr1', 1_000, 1_300)
    gradients = np.zeros((small_interval.width, 4), np.float32)
    gradients[80, 0] = 2.0
    sigma = 40.0
    df = enhancer_scoring.score_enhancers(
        _make_output(gradients, interval=small_interval),
        [genome.Interval('chr1', 1_050, 1_051)],
        window_size=200,
        sigma=sigma,
    )
    denom = sum(
        math.exp(-0.5 * ((idx - 50) / sigma) ** 2) for idx in range(151)
    )
    expected = 2.0 * math.exp(-0.5 * (30.0 / sigma) ** 2) / denom
    self.assertAlmostEqual(
        float(df['gradient_score'].iloc[0]), expected, places=12
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='midpoint_before_interval',
          enhancer=genome.Interval('chr1', 9_000, 9_998),
      ),
      dict(
          testcase_name='midpoint_at_interval_end',
          enhancer=genome.Interval('chr1', 20_000, 20_001),
      ),
      dict(
          testcase_name='far_away',
          enhancer=genome.Interval('chr1', 100_000, 100_500),
      ),
      dict(
          testcase_name='other_chromosome',
          enhancer=genome.Interval('chr2', 15_000, 15_500),
      ),
  )
  def test_not_scored(self, enhancer):
    df = enhancer_scoring.score_enhancers(
        _make_output(np.ones((_INTERVAL.width, 4))), [enhancer]
    )
    self.assertFalse(df['is_scored'].iloc[0])
    self.assertTrue(np.isnan(df['gradient_score'].iloc[0]))

  def test_midpoint_boundaries_scored(self):
    df = enhancer_scoring.score_enhancers(
        _make_output(np.ones((_INTERVAL.width, 4))),
        [
            genome.Interval('chr1', 10_000, 10_001),  # midpoint index 0
            genome.Interval('chr1', 19_999, 20_000),  # midpoint index S - 1
        ],
    )
    self.assertTrue(df['is_scored'].all())

  def test_empty_enhancers_and_optional_metadata_columns(self):
    df_with_meta = enhancer_scoring.score_enhancers(
        _make_output(np.ones((_INTERVAL.width, 4))), []
    )
    self.assertEmpty(df_with_meta)
    self.assertEqual(
        list(df_with_meta.columns),
        [
            'chromosome',
            'start',
            'end',
            'name',
            'biosample_name',
            'gene_id',
            'gradient_score',
            'is_scored',
            'gene_name',
        ],
    )

    out_no_gene_name = _make_output(
        np.ones((_INTERVAL.width, 4)), gene_name=None
    )
    df_no_gene_name = enhancer_scoring.score_enhancers(
        out_no_gene_name, [genome.Interval('chr1', 15_000, 15_500)]
    )
    self.assertNotIn('gene_name', df_no_gene_name.columns)
    self.assertEqual(list(df_no_gene_name['biosample_name']), ['K562'])

  def test_validation_errors(self):
    valid_out = _make_output(np.ones((_INTERVAL.width, 4)))
    enh = [genome.Interval('chr1', 15_000, 15_500)]

    with self.assertRaisesRegex(ValueError, 'window_size must be non-negative'):
      enhancer_scoring.score_enhancers(valid_out, enh, window_size=-1)
    with self.assertRaisesRegex(ValueError, 'sigma must be positive'):
      enhancer_scoring.score_enhancers(valid_out, enh, sigma=0.0)

  def test_tile_interval(self):
    tiles = enhancer_scoring.tile_interval(
        genome.Interval('chr2', 1_000, 2_000), tile_size=500, step_size=250
    )
    self.assertEqual(
        tiles,
        [
            genome.Interval('chr2', 1_000, 1_500, name='chr2:1000-1500'),
            genome.Interval('chr2', 1_250, 1_750, name='chr2:1250-1750'),
            genome.Interval('chr2', 1_500, 2_000, name='chr2:1500-2000'),
        ],
    )
    self.assertEmpty(
        enhancer_scoring.tile_interval(
            genome.Interval('chr2', 1_000, 1_400), tile_size=500, step_size=250
        )
    )
    with self.assertRaisesRegex(ValueError, 'tile_size must be positive'):
      enhancer_scoring.tile_interval(_INTERVAL, tile_size=0)
    with self.assertRaisesRegex(ValueError, 'step_size must be positive'):
      enhancer_scoring.tile_interval(_INTERVAL, step_size=-10)


class SummitAndMergeHelpersTest(parameterized.TestCase):
  # pylint: disable=protected-access

  def test_summit_plateaus_boundaries_and_duplicate_intervals(self):
    df = pd.DataFrame({
        'biosample_name': ['K562'] * 7,
        'gene_id': [_GENE_ID] * 7,
        'chromosome': ['chr1'] * 7,
        'start': [1_000, 1_250, 1_500, 1_750, 2_000, 2_500, 5_000],
        'end': [1_500, 1_750, 2_000, 2_250, 2_500, 3_000, 5_500],
        'gradient_score': [0.02, 0.02, 0.005, 0.03, 0.01, 0.008, 0.015],
    })
    summits = enhancer_scoring._extract_summit_tiles(df)
    self.assertEqual(list(summits['start']), [1_000, 1_750, 2_500, 5_000])

    # Translation equivariance: shifting coordinates preserves summits + offset.
    shifted = enhancer_scoring._extract_summit_tiles(
        df.assign(start=df['start'] + 10_000, end=df['end'] + 10_000)
    )
    pd.testing.assert_frame_equal(
        shifted.assign(
            start=shifted['start'] - 10_000, end=shifted['end'] - 10_000
        ),
        summits,
    )

    # Duplicate `(start, end)` intervals keep the highest `gradient_score`.
    dup_df = pd.DataFrame({
        'biosample_name': ['K562'] * 3,
        'gene_id': [_GENE_ID] * 3,
        'chromosome': ['chr1'] * 3,
        'start': [1_000, 1_000, 1_000],
        'end': [1_500, 1_500, 1_500],
        'gradient_score': [0.01, 0.05, 0.04],
    })
    dup_summits = enhancer_scoring._extract_summit_tiles(dup_df)
    self.assertLen(dup_summits, 1)
    self.assertAlmostEqual(float(dup_summits['gradient_score'].iloc[0]), 0.05)

  def test_merge_summits_across_genes(self):
    starts = [1_000, 1_400, 1_800, 2_300, 5_000, 5_200, 6_000]
    ends = [1_500, 1_900, 2_300, 2_800, 7_000, 5_500, 6_400]
    summits_df = pd.DataFrame({
        'biosample_name': ['K562'] * 7,
        'chromosome': ['chr1'] * 7,
        'start': starts,
        'end': ends,
        'name': [f'chr1:{s}-{e}' for s, e in zip(starts, ends, strict=True)],
        'gene_id': [
            _GENE_ID,
            _GENE_ID_2,
            _GENE_ID,
            _GENE_ID,
            _GENE_ID,
            _GENE_ID_2,
            _GENE_ID,
        ],
        'gradient_score': [0.02, 0.05, 0.04, 0.01, 0.02, 0.06, 0.045],
        'atac_score': [2.0, 6.0, 4.0, 1.5, 3.0, 7.0, 5.0],
    })
    merged = enhancer_scoring._merge_summits_across_genes(summits_df)
    self.assertEqual(list(merged['start']), [1_000, 1_000, 2_300, 5_000, 5_000])
    self.assertEqual(list(merged['end']), [2_300, 2_300, 2_800, 7_000, 7_000])
    self.assertEqual(
        list(merged['name']),
        [
            'chr1:1000-2300',
            'chr1:1000-2300',
            'chr1:2300-2800',
            'chr1:5000-7000',
            'chr1:5000-7000',
        ],
    )
    g1_chained = merged[
        (merged['gene_id'] == _GENE_ID) & (merged['start'] == 1_000)
    ].iloc[0]
    self.assertAlmostEqual(float(g1_chained['gradient_score']), 0.04)
    self.assertAlmostEqual(float(g1_chained['atac_score']), 4.0)
    g1_nested = merged[
        (merged['gene_id'] == _GENE_ID) & (merged['start'] == 5_000)
    ].iloc[0]
    self.assertAlmostEqual(float(g1_nested['gradient_score']), 0.045)

  def test_nan_empty_and_categorical_inputs(self):
    df = pd.DataFrame({
        'biosample_name': ['K562'] * 4,
        'gene_id': [_GENE_ID] * 4,
        'chromosome': ['chr1'] * 4,
        'start': [1_000, 1_250, 1_500, 3_000],
        'end': [1_500, 1_750, 2_000, 3_500],
        'name': [
            'chr1:1000-1500',
            'chr1:1250-1750',
            'chr1:1500-2000',
            'chr1:3000-3500',
        ],
        'gradient_score': [0.05, np.nan, 0.04, np.nan],
    })
    summits = enhancer_scoring._extract_summit_tiles(df)
    self.assertEqual(list(summits['start']), [1_000, 1_500])

    all_nan = df.assign(gradient_score=np.nan)
    empty_summits = enhancer_scoring._extract_summit_tiles(all_nan)
    self.assertEmpty(empty_summits)
    self.assertEqual(list(empty_summits.columns), list(df.columns))

    empty_merged = enhancer_scoring._merge_summits_across_genes(empty_summits)
    self.assertEmpty(empty_merged)
    self.assertEqual(list(empty_merged.columns), list(df.columns))

    cat_df = pd.DataFrame({
        'biosample_name': pd.Categorical(
            ['K562', 'K562'], categories=['K562', 'GM12878']
        ),
        'gene_id': pd.Categorical(
            [_GENE_ID, _GENE_ID], categories=[_GENE_ID, _GENE_ID_2]
        ),
        'chromosome': pd.Categorical(
            ['chr1', 'chr1'], categories=['chr1', 'chr2']
        ),
        'start': [1_000, 1_250],
        'end': [1_500, 1_750],
        'gradient_score': [0.04, 0.02],
    })
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      cat_merged = enhancer_scoring._merge_summits_across_genes(
          enhancer_scoring._extract_summit_tiles(cat_df)
      )
    self.assertEmpty(caught)
    self.assertLen(cat_merged, 1)


class GetActiveEnhancersTest(parameterized.TestCase):
  # pylint: disable=protected-access

  def test_summit_extraction_and_cross_gene_merge(self):
    gtf = _make_gtf()
    grads_g1 = np.zeros((_INTERVAL.width, 4), np.float32)
    grads_g1[2_250, 0] = 2.0
    grads_g2 = np.zeros((_INTERVAL.width, 4), np.float32)
    grads_g2[2_500, 0] = 3.0

    atac = np.full(_INTERVAL.width, 5.0, dtype=np.float32)
    out1 = _make_output(grads_g1, gene_id=f'{_GENE_ID}.5', gene_name='G1')
    out2 = _make_output(grads_g2, gene_id=_GENE_ID_2, gene_name='G2')
    acc = _make_acc_output(atac)

    df = enhancer_scoring.get_active_enhancers(
        [out1, out2], gtf=gtf, accessibility_outputs=[acc, acc]
    )
    self.assertLen(df, 2)
    self.assertEqual(
        list(df.columns),
        [
            'chromosome',
            'start',
            'end',
            'name',
            'biosample_name',
            'gene_id',
            'gradient_score',
            'tss',
            'tss_distance',
            'gene_name',
            'atac_score',
        ],
    )
    self.assertEqual(list(df['start']), [12_000, 12_000])
    self.assertEqual(list(df['end']), [12_750, 12_750])
    self.assertEqual(list(df['name']), ['chr1:12000-12750'] * 2)
    self.assertEqual(list(df['biosample_name']), ['K562'] * 2)
    g1_row = df[df['gene_name'] == 'G1'].iloc[0]
    g2_row = df[df['gene_name'] == 'G2'].iloc[0]
    self.assertEqual(g1_row['tss'], 15_000)
    self.assertAlmostEqual(g1_row['tss_distance'], abs(12_375.0 - 15_000))
    self.assertEqual(g2_row['tss'], 18_500)
    self.assertAlmostEqual(g2_row['tss_distance'], abs(12_375.0 - 18_500))

  def test_auto_modality_thresholds_and_bulk_atac(self):
    self.assertEqual(
        dict(enhancer_scoring.DEFAULT_GRADIENT_SCORE_THRESHOLDS),
        {
            dna_output.OutputType.SC_GENE_EXPRESSION: 0.0012,
            dna_output.OutputType.RNA_SEQ: 0.0017,
        },
    )
    gtf = _make_gtf()
    grads = np.zeros((_INTERVAL.width, 4), np.float32)
    grads[:, 0] = 0.0015
    sc_out = _make_output(
        grads,
        biosample_name='sc_bio',
        output_type=dna_output.OutputType.SC_GENE_EXPRESSION,
    )
    bulk_out = _make_output(
        grads,
        biosample_name='bulk_bio',
        output_type=dna_output.OutputType.RNA_SEQ,
    )
    atac_vals = np.full(_INTERVAL.width, 3.0, dtype=np.float32)
    sc_atac = _make_acc_output(
        atac_vals,
        biosample_name='sc_bio',
        output_type=dna_output.OutputType.ATAC,
    )
    bulk_atac = _make_acc_output(
        atac_vals,
        biosample_name='bulk_bio',
        output_type=dna_output.OutputType.ATAC,
    )
    df = enhancer_scoring.get_active_enhancers(
        [sc_out, bulk_out],
        gtf=gtf,
        accessibility_outputs=[sc_atac, bulk_atac],
        accessibility_output_type=dna_output.OutputType.ATAC,
    )
    self.assertEqual(set(df['biosample_name']), {'sc_bio'})

  def test_prefer_observed_accessibility_and_without_atac(self):
    gtf = _make_gtf()
    grads = np.zeros((_INTERVAL.width, 4), np.float32)
    grads[2_250, 0] = 2.0
    out = _make_output(grads)

    two_ch = np.column_stack([
        np.full(_INTERVAL.width, 0.5, dtype=np.float32),
        np.full(_INTERVAL.width, 10.0, dtype=np.float32),
    ])
    acc_obs_imp = _make_acc_output(
        two_ch, imputation_modes=['observed', 'imputed_supported']
    )
    df_obs = enhancer_scoring.get_active_enhancers(
        [out], gtf=gtf, accessibility_outputs=[acc_obs_imp]
    )
    self.assertEmpty(df_obs)
    self.assertIn('atac_score', df_obs.columns)

    df_all = enhancer_scoring.get_active_enhancers(
        [out],
        gtf=gtf,
        accessibility_outputs=[acc_obs_imp],
        prefer_observed_accessibility=False,
    )
    self.assertLen(df_all, 1)
    self.assertAlmostEqual(float(df_all['atac_score'].iloc[0]), 5.25)

    acc_all_imp = _make_acc_output(
        two_ch[:, 1:], imputation_modes=['imputed_supported']
    )
    df_imp_fallback = enhancer_scoring.get_active_enhancers(
        [out], gtf=gtf, accessibility_outputs=[acc_all_imp]
    )
    self.assertLen(df_imp_fallback, 1)

    df_no_atac = enhancer_scoring.get_active_enhancers(
        [out], gtf=gtf, atac_score_threshold=None
    )
    self.assertLen(df_no_atac, 1)
    self.assertNotIn('atac_score', df_no_atac.columns)

    zero_ch_acc = dna_output.Output(
        sc_atac=track_data.TrackData(
            values=np.empty((_INTERVAL.width, 0), dtype=np.float32),
            metadata=pd.DataFrame(columns=['name', 'strand']),
            interval=_INTERVAL,
        )
    )
    self.assertEmpty(
        enhancer_scoring.get_active_enhancers(
            [out], gtf=gtf, accessibility_outputs=[zero_ch_acc]
        )
    )

  def test_par_y_disambiguation_and_empty_inputs(self):
    par_gene = 'ENSG00000182484'
    par_gtf = pd.DataFrame({
        'Chromosome': ['chrX', 'chrX', 'chrY', 'chrY'],
        'Start': [20_000, 99_000, 35_000, 88_000],
        'End': [22_000, 99_500, 37_000, 88_500],
        'Strand': ['+', '+', '+', '+'],
        'Feature': ['transcript', 'exon', 'transcript', 'gene'],
        'gene_id': [
            f'{par_gene}.15',
            f'{par_gene}.15',
            f'{par_gene}.15_PAR_Y',
            f'{par_gene}.15_PAR_Y',
        ],
        'tag': ['MANE_Select', 'MANE_Select', 'MANE_Select', 'MANE_Select'],
    })
    grads = np.zeros((_INTERVAL.width, 4), np.float32)
    grads[2_250, 0] = 2.0
    chrx_interval = genome.Interval('chrX', 15_000, 25_000)
    chry_interval = genome.Interval('chrY', 30_000, 40_000)
    out_x = _make_output(
        grads, interval=chrx_interval, gene_id=f'{par_gene}.15'
    )
    out_y = _make_output(
        grads, interval=chry_interval, gene_id=f'{par_gene}.15_PAR_Y'
    )
    df = enhancer_scoring.get_active_enhancers(
        [out_x, out_y], gtf=par_gtf, atac_score_threshold=None
    )
    self.assertEqual(
        dict(zip(df['chromosome'], df['tss'], strict=True)),
        {'chrX': 20_000, 'chrY': 35_000},
    )

    empty_with_atac = enhancer_scoring.get_active_enhancers(
        [], gtf=pd.DataFrame(), accessibility_outputs=[]
    )
    self.assertEmpty(empty_with_atac)
    self.assertEqual(
        list(empty_with_atac.columns),
        [*enhancer_scoring._ACTIVE_ENHANCER_BASE_DTYPES, 'atac_score'],
    )
    empty_no_atac = enhancer_scoring.get_active_enhancers(
        [], gtf=pd.DataFrame(), atac_score_threshold=None
    )
    self.assertEmpty(empty_no_atac)
    self.assertEqual(
        list(empty_no_atac.columns),
        list(enhancer_scoring._ACTIVE_ENHANCER_BASE_DTYPES),
    )

  def test_atac_filtering_precedes_summit_extraction(self):
    # Three overlapping 500-bp tiles at [11000, 11500), [11250, 11750), and
    # [11500, 12000) all pass `gradient_score_threshold`, with the middle tile
    # holding the highest gradient (0.05) but failing `atac_score_threshold`.
    # Filtering ATAC before summit extraction removes the bridge tile so both
    # outer tiles survive as independent summits.
    gtf = _make_gtf()
    grads = np.zeros((_INTERVAL.width, 4), np.float32)
    grads[1_250, 0] = 0.02
    grads[1_500, 0] = 0.05
    grads[1_750, 0] = 0.03
    atac = np.zeros(_INTERVAL.width, dtype=np.float32)
    atac[1_100] = 5.0
    atac[1_800] = 5.0
    df = enhancer_scoring.get_active_enhancers(
        [_make_output(grads)],
        gtf=gtf,
        accessibility_outputs=[_make_acc_output(atac)],
        window_size=0,
    )
    self.assertEqual(list(df['start']), [11_000, 11_500])
    self.assertEqual(list(df['end']), [11_500, 12_000])

  def test_validation_errors(self):
    gtf = _make_gtf()
    grads = np.zeros((_INTERVAL.width, 4), np.float32)
    out = _make_output(grads)
    acc = _make_acc_output(np.ones(_INTERVAL.width, dtype=np.float32))

    with self.assertRaisesRegex(ValueError, 'tile_size must be positive'):
      enhancer_scoring.get_active_enhancers(
          [out], gtf=gtf, accessibility_outputs=[acc], tile_size=0
      )
    with self.assertRaisesRegex(ValueError, 'step_size must be positive'):
      enhancer_scoring.get_active_enhancers(
          [out], gtf=gtf, accessibility_outputs=[acc], step_size=0
      )
    with self.assertRaisesRegex(ValueError, 'window_size must be non-negative'):
      enhancer_scoring.get_active_enhancers(
          [out], gtf=gtf, accessibility_outputs=[acc], window_size=-1
      )
    with self.assertRaisesRegex(ValueError, 'sigma must be positive'):
      enhancer_scoring.get_active_enhancers(
          [out], gtf=gtf, accessibility_outputs=[acc], sigma=0.0
      )
    no_interval_track = track_data.TrackData(
        values=np.ones((10, 1), dtype=np.float32),
        metadata=pd.DataFrame({'name': ['ch_0'], 'strand': ['.']}),
    )
    with self.assertRaisesRegex(
        ValueError, 'Accessibility track must have an interval'
    ):
      enhancer_scoring._compute_tile_atac_scores(no_interval_track, [])
    with self.assertRaisesRegex(ValueError, 'not calibrated'):
      denoised = _make_output(
          grads, output_type=dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED
      )
      enhancer_scoring.get_active_enhancers(
          [denoised], gtf=gtf, accessibility_outputs=[acc]
      )
    with self.assertRaisesRegex(
        ValueError, '`accessibility_outputs` must be provided'
    ):
      enhancer_scoring.get_active_enhancers([out], gtf=gtf)
    with self.assertRaisesRegex(ValueError, 'same length'):
      enhancer_scoring.get_active_enhancers(
          [out], gtf=gtf, accessibility_outputs=[acc, acc]
      )
    with self.assertRaisesRegex(ValueError, '1-bp ATAC track'):
      enhancer_scoring.get_active_enhancers(
          [out],
          gtf=gtf,
          accessibility_outputs=[acc],
          accessibility_output_type=dna_output.OutputType.ATAC,
      )
    short_acc = _make_acc_output(
        np.ones(1_000, dtype=np.float32),
        interval=genome.Interval('chr1', 10_000, 11_000),
    )
    with self.assertRaisesRegex(ValueError, 'must contain'):
      enhancer_scoring.get_active_enhancers(
          [out], gtf=gtf, accessibility_outputs=[short_acc]
      )
    for bad_acc in (
        _make_acc_output(
            np.ones(_INTERVAL.width, dtype=np.float32), biosample_name=None
        ),
        _make_acc_output(
            np.ones(_INTERVAL.width, dtype=np.float32), biosample_name='GM12878'
        ),
        dna_output.Output(
            sc_atac=track_data.TrackData(
                values=np.ones((_INTERVAL.width, 2), dtype=np.float32),
                metadata=pd.DataFrame({
                    'name': ['ch_0', 'ch_1'],
                    'strand': ['.', '.'],
                    'biosample_name': ['K562', 'GM12878'],
                }),
                interval=_INTERVAL,
            )
        ),
        dna_output.Output(
            sc_atac=track_data.TrackData(
                values=np.ones((_INTERVAL.width, 2), dtype=np.float32),
                metadata=pd.DataFrame({
                    'name': ['ch_0', 'ch_1'],
                    'strand': ['.', '.'],
                    'biosample_name': pd.Series(
                        ['K562', pd.NA], dtype='string'
                    ),
                }),
                interval=_INTERVAL,
            )
        ),
    ):
      with self.assertRaisesRegex(
          ValueError, 'Accessibility `biosample_name` must match'
      ):
        enhancer_scoring.get_active_enhancers(
            [out], gtf=gtf, accessibility_outputs=[bad_acc]
        )
    with self.assertRaisesRegex(ValueError, 'Duplicate'):
      enhancer_scoring.get_active_enhancers(
          [out, out], gtf=gtf, accessibility_outputs=[acc, acc]
      )
    missing_gene = _make_output(grads, gene_id='ENSG99999999999')
    with self.assertRaisesRegex(ValueError, 'MANE_Select TSS'):
      enhancer_scoring.get_active_enhancers(
          [missing_gene], gtf=gtf, accessibility_outputs=[acc]
      )
    wrong_chrom = genome.Interval('chr2', 10_000, 20_000)
    with self.assertRaisesRegex(ValueError, 'MANE_Select TSS'):
      enhancer_scoring.get_active_enhancers(
          [_make_output(grads, interval=wrong_chrom)],
          gtf=_make_gtf(),
          accessibility_outputs=[
              _make_acc_output(np.ones(10_000), wrong_chrom)
          ],
      )


class AssignConsensusEnhancersTest(parameterized.TestCase):

  def test_frequency_priority_tie_breaking_and_gene_independence(self):
    df = pd.DataFrame({
        'biosample_name': ['Erythroid', 'MEP', 'GMP', 'B', 'T'],
        'gene_id': [_GENE_ID] * 5,
        'chromosome': ['chr1'] * 5,
        'start': [11_000, 11_000, 11_000, 20_200, 20_000],
        'end': [11_500, 11_500, 11_750, 20_700, 20_500],
    })
    res = enhancer_scoring.assign_consensus_enhancers(df)
    self.assertEqual(
        list(res['consensus_peak_start']),
        [11_000, 11_000, 11_000, 20_000, 20_000],
    )
    self.assertEqual(
        list(res['consensus_peak_end']),
        [11_500, 11_500, 11_500, 20_500, 20_500],
    )

    # High-frequency calls at [20_200, 20_700) on a second gene must not alter
    # _GENE_ID's tie-broken consensus peak [20_000, 20_500).
    extra_g2 = pd.DataFrame({
        'biosample_name': ['M1', 'M2', 'M3'],
        'gene_id': [_GENE_ID_2] * 3,
        'chromosome': ['chr1'] * 3,
        'start': [20_200] * 3,
        'end': [20_700] * 3,
    })
    with_g2 = enhancer_scoring.assign_consensus_enhancers(
        pd.concat([df, extra_g2], ignore_index=True)
    )
    pd.testing.assert_frame_equal(
        with_g2.iloc[: len(df)], res, check_names=True
    )

  def test_mutual_vs_asymmetric_overlap_and_best_representative(self):
    # [1_000, 1_500) inside [1_000, 2_500) has mutual overlap 500/1500 = 1/3.
    asym_df = pd.DataFrame({
        'gene_id': [_GENE_ID] * 3,
        'chromosome': ['chr1'] * 3,
        'start': [1_000, 1_000, 1_000],
        'end': [2_500, 2_500, 1_500],
    })
    unmerged = enhancer_scoring.assign_consensus_enhancers(
        asym_df, min_reciprocal_overlap=0.5
    )
    self.assertEqual(
        list(unmerged['consensus_peak_end']), [2_500, 2_500, 1_500]
    )
    merged = enhancer_scoring.assign_consensus_enhancers(
        asym_df, min_reciprocal_overlap=0.3
    )
    self.assertEqual(list(merged['consensus_peak_end']), [2_500] * 3)

    # Candidate [1_260, 1_760) overlaps accepted [1_000, 1_500) by 0.48 and
    # accepted [1_500, 2_000) by 0.52; highest overlap wins even over higher
    # frequency, while abutting [1_000, 1_500) and [1_500, 2_000) stay separate.
    best_df = pd.DataFrame({
        'gene_id': [_GENE_ID] * 6,
        'chromosome': ['chr1'] * 6,
        'start': [1_000, 1_000, 1_000, 1_500, 1_500, 1_260],
        'end': [1_500, 1_500, 1_500, 2_000, 2_000, 1_760],
    })
    best_res = enhancer_scoring.assign_consensus_enhancers(
        best_df, min_reciprocal_overlap=0.4
    )
    self.assertEqual(
        list(best_res['consensus_peak_start']),
        [1_000, 1_000, 1_000, 1_500, 1_500, 1_500],
    )
    abutting = enhancer_scoring.assign_consensus_enhancers(
        best_df.iloc[:5], min_reciprocal_overlap=1e-12
    )
    self.assertEqual(
        list(abutting['consensus_peak_start']),
        [1_000, 1_000, 1_000, 1_500, 1_500],
    )

  def test_reclustering_index_preservation_and_empty(self):
    sliced = pd.DataFrame(
        {
            'gene_id': [_GENE_ID] * 3,
            'chromosome': ['chr2'] * 3,
            'start': [10_000, 10_000, 10_200],
            'end': [10_500, 10_500, 10_700],
            'size': [500, 500, 500],
        },
        index=pd.Index([10, 42, 7], name='call_id'),
    )
    first = enhancer_scoring.assign_consensus_enhancers(
        sliced, min_reciprocal_overlap=0.5
    )
    self.assertNotIn('consensus_peak_start', sliced.columns)
    pd.testing.assert_index_equal(first.index, sliced.index)
    self.assertEqual(list(first['size']), [500, 500, 500])
    self.assertEqual(list(first['consensus_peak_start']), [10_000] * 3)

    # Re-clustering with a stricter threshold replaces existing consensus bounds
    # cleanly without creating _x/_y columns.
    second = enhancer_scoring.assign_consensus_enhancers(
        first, min_reciprocal_overlap=0.9
    )
    pd.testing.assert_index_equal(second.index, sliced.index)
    self.assertEqual(list(second.columns), list(first.columns))
    self.assertEqual(
        list(second['consensus_peak_start']), [10_000, 10_000, 10_200]
    )

    empty_in = sliced.iloc[0:0]
    empty_res = enhancer_scoring.assign_consensus_enhancers(empty_in)
    self.assertEmpty(empty_res)
    pd.testing.assert_index_equal(empty_res.index, empty_in.index)
    self.assertEqual(empty_res['consensus_peak_start'].dtype, int)
    self.assertEqual(empty_res['consensus_peak_end'].dtype, int)

  @parameterized.named_parameters(
      dict(testcase_name='zero_overlap', overlap=0.0),
      dict(testcase_name='negative_overlap', overlap=-0.2),
      dict(testcase_name='above_one_overlap', overlap=1.1),
      dict(testcase_name='nan_overlap', overlap=float('nan')),
  )
  def test_invalid_min_reciprocal_overlap_raises(self, overlap):
    df = pd.DataFrame({
        'gene_id': [_GENE_ID],
        'chromosome': ['chr1'],
        'start': [1_000],
        'end': [1_500],
    })
    with self.assertRaisesRegex(ValueError, 'min_reciprocal_overlap'):
      enhancer_scoring.assign_consensus_enhancers(
          df, min_reciprocal_overlap=overlap
      )

  def test_invalid_dataframe_raises(self):
    with self.assertRaisesRegex(ValueError, 'missing required columns'):
      enhancer_scoring.assign_consensus_enhancers(
          pd.DataFrame({'gene_id': [_GENE_ID], 'start': [1_000]})
      )
    with self.assertRaisesRegex(ValueError, 'must not contain null'):
      enhancer_scoring.assign_consensus_enhancers(
          pd.DataFrame({
              'gene_id': [_GENE_ID],
              'chromosome': ['chr1'],
              'start': [np.nan],
              'end': [1_500],
          })
      )
    for bad_end in (1_000, 900):
      with self.assertRaisesRegex(ValueError, 'start < end'):
        enhancer_scoring.assign_consensus_enhancers(
            pd.DataFrame({
                'gene_id': [_GENE_ID],
                'chromosome': ['chr1'],
                'start': [1_000],
                'end': [bad_end],
            })
        )


if __name__ == '__main__':
  absltest.main()
