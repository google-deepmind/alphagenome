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

import warnings

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import gene_expression_data
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.data import track_data
from alphagenome.data import transcript as transcript_utils
from alphagenome.visualization import plot as plot_lib
from alphagenome.visualization import plot_components
import anndata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ATAC_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar', 'baz', 'buz', 'fux'],
        strand=['.'] * 5,
        padding=[False] * 5,
    )
)

_CONTACT_MAP_METADATA = pd.DataFrame(
    dict(
        name=['H1', 'HFF'],
        strand=['.'] * 2,
        padding=[False] * 2,
    )
)

_RNA_SEQ_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'boo', 'bar', 'far'],
        strand=['.'] * 4,
        padding=[False] * 4,
    )
)

_SPLICING_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar'],
        ontology_curie=['ontology_curie_1', 'ontology_curie_2'],
        padding=[False] * 2,
    )
)


def _create_feature_contributions(
    n_positions: int = 10, n_features: int = 3
) -> anndata.AnnData:
  scores = (
      np.random.default_rng(0)
      .standard_normal((n_positions, n_features))
      .astype(np.float32)
  )
  var = pd.DataFrame({'name': [f'feature_{i}' for i in range(n_features)]})
  variants = [
      genome.Variant('chr1', 101 + i, 'T', 'A') for i in range(n_positions)
  ]
  obs = pd.DataFrame({'variant': variants})
  return anndata.AnnData(X=scores, var=var, obs=obs)


class PlotComponentsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(subinterval=genome.Interval('chr1', 100, 4196), logo_ylim=[0, 1]),
      dict(subinterval=genome.Interval('chr1', 0, 2**20), logo_ylim=None),
  ])
  def test_track_component(self, subinterval, logo_ylim):
    interval = genome.Interval('chr1', 0, 2**20)

    # Mock 1D track data.
    atac_values = np.stack(
        [np.arange(interval.width) * 0.0001] * len(_ATAC_METADATA), axis=1
    ).astype(np.float32)
    atac_tdata = track_data.TrackData(
        values=atac_values,
        metadata=_ATAC_METADATA,
        interval=interval,
        resolution=1,
    )

    rna_seq_values_ref = np.stack(
        [np.arange(interval.width) * 0.0001] * len(_RNA_SEQ_METADATA), axis=1
    ).astype(np.float32)
    rna_seq_tdata_ref = track_data.TrackData(
        values=rna_seq_values_ref,
        metadata=_RNA_SEQ_METADATA,
        interval=interval,
        resolution=1,
    )
    # Mimic the case where the ALT allele increases expression by 50%.
    rna_seq_values_alt = rna_seq_values_ref * 1.5
    rna_seq_tdata_alt = track_data.TrackData(
        values=rna_seq_values_alt,
        metadata=_RNA_SEQ_METADATA,
        interval=interval,
        resolution=1,
    )

    # Mock contact maps.
    contact_maps_shape = (
        interval.width // 2048,
        interval.width // 2048,
        len(_CONTACT_MAP_METADATA),
    )

    contact_maps_1 = track_data.TrackData(
        values=np.random.normal(0, 1, contact_maps_shape).astype(np.float32),
        metadata=_CONTACT_MAP_METADATA,
        interval=interval,
        resolution=2048,
    )

    contact_maps_2 = track_data.TrackData(
        values=contact_maps_1.values * 0.5,
        metadata=_CONTACT_MAP_METADATA,
        interval=interval,
        resolution=2048,
    )
    contact_maps_diff = track_data.TrackData(
        values=contact_maps_1.values - contact_maps_2.values,
        metadata=_CONTACT_MAP_METADATA,
        interval=interval,
        resolution=2048,
    )

    # Mock contribution scores.
    contrib = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        * (interval.width // 4)
    ).astype(np.float32)

    # Get actual transcripts.
    transcripts = [
        transcript_utils.Transcript(
            exons=[genome.Interval('chr1', 0, 100)], info=dict(gene_name='gene')
        )
    ]

    # Mock junctions.
    junctions = np.array([
        genome.Junction('chr1', 102, 108, '+'),
        genome.Junction('chr1', 80, 100, '+'),
        genome.Junction('chr1', 104, 108, '+'),
        genome.Junction('chr1', 110, 120, '+'),
    ])
    junctions = junction_data.JunctionData(
        junctions,
        values=np.zeros((4, 2), dtype=np.float32),
        metadata=_SPLICING_METADATA,
    )

    # Mock variants within subinterval to annotate.
    annotated_variants = [
        genome.Variant(
            chromosome=subinterval.chromosome,
            position=pos,
            reference_bases='A',
            alternate_bases='G',
        )
        for pos in subinterval.start + subinterval.width / np.array(
            [2, 3, 10], dtype=np.float32
        )
    ]

    # Mock intervals  within subinterval to annotate.
    annotated_intervals = [
        interval.resize(int(interval.width // 10)),
        interval.resize(int(interval.width // 100)).shift(
            -int(interval.width // 10)
        ),
    ]
    annotated_intervals_labels = [
        f'my_interval_{i+1}' for i in range(len(annotated_intervals))
    ]

    fig = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(transcripts, fig_height=3),
            plot_components.TranscriptAnnotation(
                transcripts, adaptive_fig_height=False
            ),
            plot_components.TranscriptAnnotation(
                transcripts, adaptive_fig_height=True, fig_height=1
            ),
            plot_components.Sashimi(junctions),
            plot_components.Tracks(atac_tdata, filled=False, linestyle='--'),
            plot_components.Tracks(atac_tdata, filled=True),
            plot_components.Tracks(atac_tdata, shared_y_scale=True),
            plot_components.Tracks(atac_tdata, shared_y_scale=False),
            plot_components.Tracks(
                rna_seq_tdata_ref, cmap='viridis', truncate_cmap=True
            ),
            plot_components.Tracks(
                rna_seq_tdata_ref, cmap='Set2', truncate_cmap=False
            ),
            plot_components.OverlaidTracks(
                {'REF': rna_seq_tdata_ref, 'ALT': rna_seq_tdata_alt}
            ),
            plot_components.OverlaidTracks(
                {'REF': rna_seq_tdata_ref, 'ALT': rna_seq_tdata_alt},
                colors={'REF': 'blue', 'ALT': 'red'},
            ),
            plot_components.OverlaidTracks(
                {'REF': rna_seq_tdata_ref, 'ALT': rna_seq_tdata_alt},
                cmap='Set2',
                ylabel_template='{name}',
            ),
            plot_components.ContactMaps(
                tdata=contact_maps_1, cmap=matplotlib.pyplot.get_cmap('autumn')
            ),
            plot_components.ContactMapsDiff(
                tdata=contact_maps_diff,
            ),
            plot_components.SeqLogo(
                contrib,
                subinterval,
                ylabel='contribution scores',
                ylim=logo_ylim,
            ),
            plot_components.FeatureContribution(
                _create_feature_contributions()
            ),
            plot_components.ScoreBar(_create_feature_contributions()),
        ],
        interval=subinterval,
        annotations=[
            plot_components.IntervalAnnotation(
                annotated_intervals,
                labels=annotated_intervals_labels,
                colors='darkgray',
            ),
            plot_components.VariantAnnotation(
                annotated_variants, use_default_labels=False, colors='purple'
            ),
            plot_components.VariantAnnotation(
                annotated_variants, use_default_labels=True, colors='orange'
            ),
        ],
    )
    self.assertIsInstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

  def test_feature_contribution_axes_config(self):
    adata = _create_feature_contributions()
    comp = plot_components.FeatureContribution(adata, fig_height=4.0)
    self.assertEqual(comp.num_axes, 1)
    self.assertEqual(comp.get_ax_height(0), 4.0)

  def test_feature_contribution_missing_variant_raises(self):
    adata = anndata.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame({'other': [1, 2]}),
    )
    with self.assertRaisesRegex(ValueError, "must have 'variant' in obs"):
      plot_components.FeatureContribution(adata)

  def test_feature_contribution_duplicate_positions_takes_max_sum(self):
    variants = [
        genome.Variant('chr1', 100, 'A', 'C'),  # sum = 1.0 + 1.0 = 2.0
        genome.Variant('chr1', 100, 'A', 'G'),  # sum = 4.0 - 0.5 = 3.5 (max)
        genome.Variant('chr1', 100, 'A', 'T'),  # sum = 5.0 - 3.0 = 2.0
    ]
    adata = anndata.AnnData(
        X=np.array([[1.0, 1.0], [4.0, -0.5], [5.0, -3.0]], dtype=np.float32),
        obs=pd.DataFrame({'variant': variants}),
        var=pd.DataFrame({'name': ['f0', 'f1']}),
    )
    comp = plot_components.FeatureContribution(
        adata, feature_order=['f0', 'f1']
    )
    _, ax = plt.subplots()
    comp.plot_ax(ax, axis_index=0, interval=genome.Interval('chr1', 99, 102))

    self.assertLen(ax.collections, 2)
    self.assertLen(ax.collections[0].get_paths(), 1)
    self.assertLen(ax.collections[1].get_paths(), 1)
    np.testing.assert_allclose(
        ax.collections[0].get_paths()[0].vertices[:4],
        [[100.0, 0.0], [101.0, 0.0], [101.0, 4.0], [100.0, 4.0]],
    )
    np.testing.assert_allclose(
        ax.collections[1].get_paths()[0].vertices[:4],
        [[100.0, 0.0], [101.0, 0.0], [101.0, -0.5], [100.0, -0.5]],
    )
    plt.close()

  def test_feature_contribution_feature_ordering_and_stacking(self):
    variant = genome.Variant('chr1', 100, 'T', 'A')
    adata = anndata.AnnData(
        X=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        obs=pd.DataFrame({'variant': [variant]}),
        var=pd.DataFrame(
            {'name': ['unknown_b', 'unknown_a', 'merged_splicing']},
            index=['unknown_b', 'unknown_a', 'merged_splicing'],
        ),
    )
    comp = plot_components.FeatureContribution(adata)
    _, ax = plt.subplots()
    comp.plot_ax(ax, axis_index=0, interval=genome.Interval('chr1', 99, 102))

    self.assertLen(ax.collections, 3)
    self.assertEqual(ax.collections[0].get_label(), 'merged_splicing')
    self.assertEqual(ax.collections[1].get_label(), 'unknown_a')
    self.assertEqual(ax.collections[2].get_label(), 'unknown_b')

    # Verify cumulative positive stacking: [0, 3], [3, 5], [5, 6].
    np.testing.assert_allclose(
        ax.collections[0].get_paths()[0].vertices[:4],
        [[100.0, 0.0], [101.0, 0.0], [101.0, 3.0], [100.0, 3.0]],
    )
    np.testing.assert_allclose(
        ax.collections[1].get_paths()[0].vertices[:4],
        [[100.0, 3.0], [101.0, 3.0], [101.0, 5.0], [100.0, 5.0]],
    )
    np.testing.assert_allclose(
        ax.collections[2].get_paths()[0].vertices[:4],
        [[100.0, 5.0], [101.0, 5.0], [101.0, 6.0], [100.0, 6.0]],
    )
    plt.close()

  def test_feature_contribution_polycollections_and_legend(self):
    variant1 = genome.Variant('chr1', 100, 'T', 'A')
    variant2 = genome.Variant('chr1', 101, 'T', 'A')
    adata = anndata.AnnData(
        X=np.array([[2.0, 0.0, 0.0], [0.0, -1.5, 0.0]], dtype=np.float32),
        obs=pd.DataFrame({'variant': [variant1, variant2]}),
        var=pd.DataFrame({'name': ['f0', 'f1', 'f2']}),
    )
    comp = plot_components.FeatureContribution(
        adata,
        feature_order=['F0', 'F1', 'F2'],
        show_legend=True,
        max_y_ticks=2,
    )
    _, ax = plt.subplots()
    comp.plot_ax(ax, axis_index=0, interval=genome.Interval('chr1', 100, 102))

    self.assertEmpty(ax.patches)
    self.assertLen(ax.collections, 3)
    self.assertIsInstance(
        ax.collections[0], matplotlib.collections.PolyCollection
    )
    self.assertTrue(ax.collections[0].get_rasterized())
    self.assertLen(ax.collections[0].get_paths(), 1)
    self.assertLen(ax.collections[1].get_paths(), 1)
    self.assertEmpty(ax.collections[2].get_paths())

    locator = ax.yaxis.get_major_locator()
    self.assertIsInstance(locator, matplotlib.ticker.MaxNLocator)

    legend = ax.get_legend()
    self.assertIsNotNone(legend)
    self.assertLen(legend.get_texts(), 3)
    self.assertEqual(legend.get_texts()[0].get_text(), 'f0')
    self.assertEqual(legend.get_texts()[1].get_text(), 'f1')
    self.assertEqual(legend.get_texts()[2].get_text(), 'f2')
    plt.close()

  def test_score_bar_single_feature(self):
    variants = [
        genome.Variant('chr1', 100, 'A', 'C'),
        genome.Variant('chr1', 101, 'T', 'G'),
    ]
    adata = anndata.AnnData(
        X=np.array([[3.5], [1.2]], dtype=np.float32),
        obs=pd.DataFrame({'variant': variants}),
    )
    comp = plot_components.ScoreBar(
        adata, color='#0053d6', ylabel='AVI Phred', fig_height=1.5
    )
    self.assertEqual(comp.get_ax_height(0), 1.5)

    _, ax = plt.subplots()
    comp.plot_ax(ax, axis_index=0, interval=genome.Interval('chr1', 100, 102))

    self.assertLen(ax.collections, 1)
    self.assertIsNone(ax.get_legend())
    self.assertEqual(ax.get_ylabel(), 'AVI Phred')
    np.testing.assert_allclose(
        ax.collections[0].get_facecolor()[0],
        matplotlib.colors.to_rgba('#0053d6'),
    )
    np.testing.assert_allclose(
        ax.collections[0].get_paths()[0].vertices[:4],
        [[100.0, 0.0], [101.0, 0.0], [101.0, 3.5], [100.0, 3.5]],
    )
    np.testing.assert_allclose(
        ax.collections[0].get_paths()[1].vertices[:4],
        [[101.0, 0.0], [102.0, 0.0], [102.0, 1.2], [101.0, 1.2]],
    )
    plt.close()

  def test_score_bar_multiple_features_sums_with_nans(self):
    variants = [
        genome.Variant('chr1', 100, 'A', 'C'),  # nansum = 1.5 + 2.5 = 4.0
        genome.Variant('chr1', 100, 'A', 'G'),  # nansum = 5.0 + 0.0 = 5.0
    ]
    adata = anndata.AnnData(
        X=np.array([[1.5, 2.5], [5.0, np.nan]], dtype=np.float32),
        obs=pd.DataFrame({'variant': variants}),
        var=pd.DataFrame({'name': ['f0', 'f1']}),
    )
    comp = plot_components.ScoreBar(adata)
    _, ax = plt.subplots()
    comp.plot_ax(ax, axis_index=0, interval=genome.Interval('chr1', 99, 102))

    self.assertLen(ax.collections, 1)
    self.assertLen(ax.collections[0].get_paths(), 1)
    np.testing.assert_allclose(
        ax.collections[0].get_paths()[0].vertices[:4],
        [[100.0, 0.0], [101.0, 0.0], [101.0, 5.0], [100.0, 5.0]],
    )
    plt.close()


class EnhancerArcsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.interval = genome.Interval('chr2', 60_490_000, 60_560_000)
    self.links = pd.DataFrame({
        'chromosome': ['chr2', 'chr2', 'chr2', 'chr1', 'chr2'],
        'start': [
            60_495_000,
            60_498_000,
            60_500_000,
            60_495_000,
            60_600_000,
        ],
        'end': [
            60_495_500,
            60_498_500,
            60_500_500,
            60_495_500,
            60_600_500,
        ],
        'gene_id': [
            'ENSG00000119866.1',
            'ENSG00000119866.22',
            'ENSG00000999999.1',
            'ENSG00000119866.1',
            'ENSG00000119866.1',
        ],
        'tss': [60_553_651] * 5,
        'gradient_score': [0.005, 0.015, 0.012, 0.010, 0.010],
    })

  @parameterized.parameters([
      dict(
          upward=None,
          fig_height=None,
          expected_height=1.0,
          expected_ylim=(-0.05, 1.08),
          sign=1.0,
      ),
      dict(
          upward=False,
          fig_height=0.9,
          expected_height=0.9,
          expected_ylim=(-1.08, 0.05),
          sign=-1.0,
      ),
  ])
  def test_enhancer_arcs_geometry_and_styling(
      self, upward, fig_height, expected_height, expected_ylim, sign
  ):
    extra_kwargs = {}
    if upward is not None:
      extra_kwargs['upward'] = upward
    if fig_height is not None:
      extra_kwargs['fig_height'] = fig_height
    arcs_comp = plot_components.EnhancerArcs(
        self.links,
        gene_id='ENSG00000119866.5',
        cmap='Purples',
        vmin=0.002,
        vmax=0.020,
        alpha_range=(0.2, 0.8),
        linewidth_range=(1.0, 2.8),
        ylabel='E2G arcs',
        **extra_kwargs,
    )
    self.assertEqual(arcs_comp.num_axes, 1)
    self.assertEqual(arcs_comp.get_ax_height(0), expected_height)
    view = genome.Interval('chr2', 60_390_000, 60_590_000)
    _, ax = plt.subplots()
    arcs_comp.plot_ax(ax, axis_index=0, interval=view)
    self.assertLen(ax.lines, 1)
    self.assertLen(ax.patches, 2)
    self.assertEqual(ax.get_ylim(), expected_ylim)

    verts0 = ax.patches[0].get_path().vertices
    verts1 = ax.patches[1].get_path().vertices
    span_frac0 = min(1.0, (60_553_651.0 - 60_495_250.0) / 200_000.0 * 1.5)
    expected_apex0 = sign * (0.35 + 0.65 * span_frac0)
    bezier_mid0 = (
        0.125 * verts0[0]
        + 0.375 * verts0[1]
        + 0.375 * verts0[2]
        + 0.125 * verts0[3]
    )
    np.testing.assert_allclose(verts0[0], [60_553_651.0, 0.0])
    np.testing.assert_allclose(verts0[-1], [60_495_250.0, 0.0])
    np.testing.assert_allclose(
        bezier_mid0, [0.5 * (60_553_651.0 + 60_495_250.0), expected_apex0]
    )
    np.testing.assert_allclose(verts1[0], [60_553_651.0, 0.0])
    np.testing.assert_allclose(verts1[-1], [60_498_250.0, 0.0])

    # Explicit vmin=0.002, vmax=0.020 -> norm_scores = [1/6, 13/18].
    self.assertAlmostEqual(ax.patches[0].get_alpha(), 0.2 + 0.6 * (1.0 / 6.0))
    self.assertAlmostEqual(
        ax.patches[0].get_linewidth(), 1.0 + 1.8 * (1.0 / 6.0)
    )
    self.assertAlmostEqual(ax.patches[1].get_alpha(), 0.2 + 0.6 * (13.0 / 18.0))
    self.assertAlmostEqual(
        ax.patches[1].get_linewidth(), 1.0 + 1.8 * (13.0 / 18.0)
    )
    plt.close()

  @parameterized.parameters([
      dict(
          view=genome.Interval('chr2', 60_490_000, 60_560_000),
          tss=None,
          interval_contained=True,
          expected_patches=2,
      ),
      dict(
          view=genome.Interval('chr2', 60_490_000, 60_560_000),
          tss=None,
          interval_contained=False,
          expected_patches=3,
      ),
      dict(
          view=genome.Interval('chr2', 60_545_000, 60_560_000),
          tss=None,
          interval_contained=True,
          expected_patches=0,
      ),
      dict(
          view=genome.Interval('chr2', 60_545_000, 60_560_000),
          tss=None,
          interval_contained=False,
          expected_patches=3,
      ),
      dict(
          view=genome.Interval('chr2', 60_510_000, 60_530_000),
          tss=None,
          interval_contained=False,
          expected_patches=2,
      ),
      dict(
          view=genome.Interval('chr2', 60_490_000, 60_560_000),
          tss=60_580_000,
          interval_contained=True,
          expected_patches=0,
      ),
      dict(
          view=genome.Interval('chr2', 60_490_000, 60_560_000),
          tss=60_580_000,
          interval_contained=False,
          expected_patches=2,
      ),
      dict(
          view=genome.Interval('chr2', 60_495_250, 60_498_250),
          tss=60_498_250,
          interval_contained=True,
          expected_patches=2,
      ),
      dict(
          view=genome.Interval('chr2', 60_495_250, 60_498_250),
          tss=60_498_250,
          interval_contained=False,
          expected_patches=2,
      ),
      dict(
          view=genome.Interval('chr2', 60_495_250, 60_497_000),
          tss=60_495_250,
          interval_contained=False,
          expected_patches=3,
      ),
  ])
  def test_enhancer_arcs_interval_contained(
      self, view, tss, interval_contained, expected_patches
  ):
    arcs_comp = plot_components.EnhancerArcs(
        self.links,
        tss=tss,
        gene_id='ENSG00000119866.5',
        interval_contained=interval_contained,
    )
    _, ax = plt.subplots()
    arcs_comp.plot_ax(ax, axis_index=0, interval=view)
    self.assertLen(ax.patches, expected_patches)
    self.assertEqual(ax.get_xlim(), (float(view.start), float(view.end)))
    plt.close()

  @parameterized.parameters([
      dict(
          peak_col=None,
          expected_xs=[60_495_750.0, 60_500_500.0, 60_510_500.0],
      ),
      dict(
          peak_col='summit',
          expected_xs=[60_495_100.0, 60_500_500.0, 60_510_900.0],
      ),
  ])
  def test_enhancer_arcs_peak_col_and_nan_scores(self, peak_col, expected_xs):
    links = pd.DataFrame({
        'chromosome': ['chr2', 'chr2', 'chr2'],
        'start': [60_495_000, 60_500_000, 60_510_000],
        'end': [60_496_500, 60_501_000, 60_511_000],
        'summit': [60_495_100.0, np.nan, 60_510_900.0],
        'tss': [60_553_651, 60_553_651, 60_553_651],
        'gradient_score': [0.0012, 0.0013, np.nan],
    })
    with warnings.catch_warnings():
      warnings.simplefilter('error', RuntimeWarning)
      arcs = plot_components.EnhancerArcs(
          links,
          peak_col=peak_col,
          alpha_range=(0.3, 0.9),
          linewidth_range=(1.0, 2.0),
      )
      _, ax = plt.subplots()
      arcs.plot_ax(ax, axis_index=0, interval=self.interval)

    self.assertLen(ax.patches, 3)
    for patch, expected_x in zip(ax.patches, expected_xs, strict=True):
      np.testing.assert_allclose(patch.get_path().vertices[-1], [expected_x, 0])
    self.assertGreater(ax.patches[0].get_path().vertices[1][1], 0.0)
    self.assertAlmostEqual(ax.patches[0].get_linewidth(), 1.0 + 0.0012 / 0.0013)
    self.assertAlmostEqual(ax.patches[1].get_linewidth(), 2.0)
    self.assertAlmostEqual(ax.patches[2].get_linewidth(), 1.0)
    self.assertAlmostEqual(ax.patches[2].get_alpha(), 0.3)
    plt.close()

  def test_enhancer_arcs_nullable_dtypes_and_empty_links(self):
    nullable_links = pd.DataFrame({
        'start': pd.Series([60_495_000, 60_500_000], dtype='Int64'),
        'end': pd.Series([60_496_000, 60_501_000], dtype='Int64'),
        'summit': pd.Series([pd.NA, 60_500_250.0], dtype='Float64'),
        'tss': pd.Series([60_553_651, 60_553_651], dtype='Int64'),
        'gradient_score': pd.Series([0.01, pd.NA], dtype='Float64'),
    })
    _, ax_na = plt.subplots()
    plot_components.EnhancerArcs(nullable_links, peak_col='summit').plot_ax(
        ax_na, axis_index=0, interval=self.interval
    )
    self.assertLen(ax_na.patches, 2)
    np.testing.assert_allclose(
        ax_na.patches[0].get_path().vertices[-1], [60_495_500.0, 0.0]
    )
    np.testing.assert_allclose(
        ax_na.patches[1].get_path().vertices[-1], [60_500_250.0, 0.0]
    )
    plt.close()

    _, ax_empty = plt.subplots()
    plot_components.EnhancerArcs(
        self.links.iloc[0:0], tss=60_553_651, score_col=None
    ).plot_ax(ax_empty, axis_index=0, interval=self.interval)
    self.assertLen(ax_empty.lines, 1)
    self.assertEmpty(ax_empty.patches)
    self.assertEqual(ax_empty.get_ylim(), (-0.05, 1.08))
    plt.close()

  def test_enhancer_arcs_plot_standalone_and_validation(self):
    _, ax_standalone = plt.subplots()
    plot_lib.enhancer_arcs_plot(self.links.iloc[:1], ax=ax_standalone)
    self.assertLen(ax_standalone.lines, 1)
    self.assertLen(ax_standalone.patches, 1)
    self.assertEqual(ax_standalone.get_ylim(), (-0.05, 1.08))
    v_max = ax_standalone.patches[0].get_path().vertices
    mid_max = 0.125 * (v_max[0] + v_max[3]) + 0.375 * (v_max[1] + v_max[2])
    self.assertAlmostEqual(mid_max[1], 1.0)
    plt.close()

    _, ax_zero = plt.subplots()
    plot_lib.enhancer_arcs_plot(self.links.iloc[:1], ax=ax_zero, tss=60_495_250)
    v_zero = ax_zero.patches[0].get_path().vertices
    mid_zero = 0.125 * (v_zero[0] + v_zero[3]) + 0.375 * (v_zero[1] + v_zero[2])
    self.assertAlmostEqual(mid_zero[1], 0.35)
    plt.close()

    _, ax_err = plt.subplots()
    with self.assertRaisesRegex(ValueError, 'Either `tss` must be provided'):
      plot_components.EnhancerArcs(
          pd.DataFrame({'start': [1], 'end': [2]})
      ).plot_ax(ax_err, axis_index=0, interval=self.interval)
    plt.close()

    _, ax_key_err = plt.subplots()
    with self.assertRaises(KeyError):
      plot_components.EnhancerArcs(self.links, peak_col='summit_typo').plot_ax(
          ax_key_err, axis_index=0, interval=self.interval
      )
    plt.close()


class GeneExpressionPlotTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.interval = genome.Interval('chr1', 0, 2000, strand='+')
    self.gene_metadata = pd.DataFrame({
        'gene_id': ['gene_a', 'gene_b', 'gene_c'],
        'gene_name': ['GeneA', 'GeneB', 'GeneC'],
        'strand': ['+', '-', '+'],
        'Chromosome': ['chr1', 'chr1', 'chr1'],
        'Start': [100, 500, 1200],
        'End': [200, 800, 1500],
        'gene_type': ['protein_coding'] * 3,
    })
    self.track_metadata = pd.DataFrame({
        'name': ['track_1', 'track_2'],
        'strand': ['+', '-'],
    })
    self.gdata = gene_expression_data.GeneExpressionData(
        values=np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]]),
        gene_metadata=self.gene_metadata,
        track_metadata=self.track_metadata,
        interval=self.interval,
    )

  @parameterized.parameters(
      dict(filled=False, shared_y_scale=False, log_scale=False),
      dict(filled=True, shared_y_scale=False, log_scale=False),
      dict(filled=False, shared_y_scale=True, log_scale=False),
      dict(filled=False, shared_y_scale=False, log_scale=True),
  )
  def test_gene_expression(self, filled, shared_y_scale, log_scale):
    component = plot_components.GeneExpression(
        gdata=self.gdata,
        filled=filled,
        shared_y_scale=shared_y_scale,
        log_scale=log_scale,
    )
    fig = plot_components.plot([component], interval=self.interval)
    self.assertIsInstance(fig, matplotlib.figure.Figure)

  def test_fails_without_gene_coords(self):
    gdata = gene_expression_data.GeneExpressionData(
        values=np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]]),
        gene_metadata=self.gene_metadata.drop(
            columns=['Chromosome', 'Start', 'End']
        ),
        track_metadata=self.track_metadata,
        interval=self.interval,
    )
    component = plot_components.GeneExpression(
        gdata=gdata,
    )
    with self.assertRaisesRegex(ValueError, 'Gene coordinates are missing'):
      plot_components.plot([component], interval=self.interval)

  @parameterized.parameters(
      dict(colors=None, shared_y_scale=False, log_scale=False),
      dict(
          colors={'ref': 'blue', 'alt': 'red'},
          shared_y_scale=False,
          log_scale=False,
      ),
      dict(colors=None, shared_y_scale=True, log_scale=False),
      dict(colors=None, shared_y_scale=False, log_scale=True),
  )
  def test_overlaid_gene_expression(self, colors, shared_y_scale, log_scale):
    gdata_alt = gene_expression_data.GeneExpressionData(
        values=self.gdata.values * 1.5,
        gene_metadata=self.gdata.gene_metadata,
        track_metadata=self.gdata.track_metadata,
        interval=self.gdata.interval,
    )
    gdata_map = {'ref': self.gdata.copy(), 'alt': gdata_alt}
    component = plot_components.OverlaidGeneExpression(
        gdata=gdata_map,
        colors=colors,
        shared_y_scale=shared_y_scale,
        log_scale=log_scale,
    )
    fig = plot_components.plot([component], interval=self.interval)
    self.assertIsInstance(fig, matplotlib.figure.Figure)


if __name__ == '__main__':
  absltest.main()
