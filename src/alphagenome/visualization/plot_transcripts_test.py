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


from unittest import mock

import matplotlib
import matplotlib.axes
import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.transforms
from absl.testing import absltest, parameterized

from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.visualization import plot_transcripts


def _make_transcript(
    exons: list[genome.Interval],
    gene_name: str = 'TEST_GENE',
    transcript_id: str = 'ENST00000000001',
) -> transcript_utils.Transcript:
  """Creates a Transcript for testing."""
  return transcript_utils.Transcript(
      exons=exons,
      info={'gene_name': gene_name},
      transcript_id=transcript_id,
  )


def _positive_strand_transcript_with_introns() -> transcript_utils.Transcript:
  """Creates a positive-strand transcript with two introns spanning 100-1000."""

  return _make_transcript(
      exons=[
          genome.Interval('chr1', 100, 200, strand='+'),
          genome.Interval('chr1', 400, 500, strand='+'),
          genome.Interval('chr1', 800, 1000, strand='+'),
      ],
  )


def _negative_strand_transcript_with_introns() -> transcript_utils.Transcript:
  """Creates a negative-strand transcript with two introns spanning 100-1000."""
  return _make_transcript(
      exons=[
          genome.Interval('chr1', 100, 200, strand='-'),
          genome.Interval('chr1', 400, 500, strand='-'),
          genome.Interval('chr1', 800, 1000, strand='-'),
      ],
  )


def _single_exon_transcript() -> transcript_utils.Transcript:
  """Creates a single-exon transcript (no introns)."""
  return _make_transcript(
      exons=[genome.Interval('chr1', 100, 500, strand='+')],
  )


class DrawStrandArrowsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ax = mock.create_autospec(matplotlib.axes.Axes, instance=True)
    mock_fig = self.ax.get_figure.return_value
    mock_fig.get_size_inches.return_value = (10.0, 4.0)
    mock_position = mock.create_autospec(
        matplotlib.transforms.Bbox, instance=True
    )
    mock_position.height = 0.8
    self.ax.get_position.return_value = mock_position

  def test_positive_strand_draws_arrows(self):
    """Draw arrows for positive-strand transcript with introns."""
    transcript = _positive_strand_transcript_with_introns()
    interval = genome.Interval('chr1', 0, 1100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertNotEmpty(marker_calls)
    marker = marker_calls[0].kwargs['marker']
    self.assertIsInstance(marker, matplotlib.path.Path)
    self.assertLess(marker.vertices[0][0], 0)

  def test_negative_strand_draws_arrows(self):
    """Draw arrows for negative-strand with left-pointing chevron."""
    transcript = _negative_strand_transcript_with_introns()
    interval = genome.Interval('chr1', 0, 1100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertNotEmpty(marker_calls)
    marker = marker_calls[0].kwargs['marker']
    self.assertIsInstance(marker, matplotlib.path.Path)
    self.assertGreater(marker.vertices[0][0], 0)

  def test_no_introns_no_arrows(self):
    """Produce no arrow markers for a single-exon transcript."""
    transcript = _single_exon_transcript()
    interval = genome.Interval('chr1', 0, 600)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertEmpty(marker_calls)

  def test_small_intron_skipped(self):
    """Produce no arrows for introns < 1% of interval width."""
    transcript = _make_transcript(
        exons=[
            genome.Interval('chr1', 100, 200, strand='+'),
            genome.Interval('chr1', 205, 300, strand='+'),
        ],
    )
    interval = genome.Interval('chr1', 0, 10000)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertEmpty(marker_calls)

  def test_max_arrows_capped(self):
    """Test the max number of arrows per intron is capped."""

    transcript = _make_transcript(
        exons=[
            genome.Interval('chr1', 0, 100, strand='+'),
            genome.Interval('chr1', 9900, 10000, strand='+'),
        ],
    )
    interval = genome.Interval('chr1', 0, 10000)
    max_arrows = 3

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
        max_arrows_per_intron=max_arrows,
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertNotEmpty(marker_calls)
    arrow_positions = marker_calls[0].args[0]
    self.assertLessEqual(len(arrow_positions), max_arrows)

  def test_arrows_near_edges_filtered(self):
    """Filter out arrows within 10bp of interval edges."""
    transcript = _make_transcript(
        exons=[
            genome.Interval('chr1', 0, 5, strand='+'),
            genome.Interval('chr1', 95, 100, strand='+'),
        ],
    )
    interval = genome.Interval('chr1', 0, 100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
        max_arrows_per_intron=1,
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    if marker_calls:
      arrow_positions = marker_calls[0].args[0]
      start_bound = interval.start + 10
      end_bound = interval.end - 10
      for pos in arrow_positions:
        self.assertBetween(pos, start_bound, end_bound)

  def test_positive_chevron_exact_vertices(self):
    """Verify positive-strand chevron has correct > shape vertices."""
    transcript = _positive_strand_transcript_with_introns()
    interval = genome.Interval('chr1', 0, 1100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    chevron = marker_calls[0].kwargs['marker']
    self.assertSequenceAlmostEqual(chevron.vertices[0], [-0.5, 0.5])
    self.assertSequenceAlmostEqual(chevron.vertices[1], [0.5, 0.0])
    self.assertSequenceAlmostEqual(chevron.vertices[2], [-0.5, -0.5])
    self.assertEqual(
        list(chevron.codes),
        [
            matplotlib.path.Path.MOVETO,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.LINETO,
        ],
    )

  def test_negative_chevron_exact_vertices(self):
    """Verify negative-strand chevron has correct < shape vertices."""
    transcript = _negative_strand_transcript_with_introns()
    interval = genome.Interval('chr1', 0, 1100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    chevron = marker_calls[0].kwargs['marker']
    self.assertSequenceAlmostEqual(chevron.vertices[0], [0.5, 0.5])
    self.assertSequenceAlmostEqual(chevron.vertices[1], [-0.5, 0.0])
    self.assertSequenceAlmostEqual(chevron.vertices[2], [0.5, -0.5])

  def test_markersize_with_single_transcript(self):
    """Verify markersize is positive for a single transcript."""
    transcript = _positive_strand_transcript_with_introns()
    interval = genome.Interval('chr1', 0, 1100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
        num_transcripts=1,
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertNotEmpty(marker_calls)
    self.assertGreater(marker_calls[0].kwargs['markersize'], 0)

  def test_markersize_with_many_transcripts(self):
    """Verify markersize is positive for many transcripts."""

    transcript = _positive_strand_transcript_with_introns()
    interval = genome.Interval('chr1', 0, 1100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
        num_transcripts=10,
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertNotEmpty(marker_calls)
    self.assertGreater(marker_calls[0].kwargs['markersize'], 0)

  def test_arrow_positions_evenly_spaced(self):
    """Ensure arrow positions within a single intron are evenly spaced."""
    transcript = _make_transcript(
        exons=[
            genome.Interval('chr1', 100, 200, strand='+'),
            genome.Interval('chr1', 900, 1000, strand='+'),
        ],
    )
    interval = genome.Interval('chr1', 0, 1100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='black',
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertNotEmpty(marker_calls)
    positions = marker_calls[0].args[0]
    self.assertGreater(len(positions), 1)
    for pos in positions:
      self.assertBetween(pos, 200, 900)
    spacings = [
        positions[i + 1] - positions[i] for i in range(len(positions) - 1)
    ]
    self.assertNotEmpty(spacings)
    self.assertAlmostEqual(min(spacings), max(spacings), places=1)

  def test_plot_style_kwargs_passed(self):
    """Verify fillstyle, markeredgewidth, linestyle, and clip_on are set."""
    transcript = _positive_strand_transcript_with_introns()
    interval = genome.Interval('chr1', 0, 1100)

    plot_transcripts.draw_strand_arrows(
        ax=self.ax,
        transcript=transcript,
        interval=interval,
        y=0.0,
        color='red',
    )

    marker_calls = [
        c for c in self.ax.plot.call_args_list if 'marker' in c.kwargs
    ]
    self.assertNotEmpty(marker_calls)
    kwargs = marker_calls[0].kwargs
    self.assertEqual(kwargs['fillstyle'], 'none')
    self.assertEqual(kwargs['markeredgewidth'], 0.8)
    self.assertEqual(kwargs['linestyle'], 'none')
    self.assertTrue(kwargs['clip_on'])
    self.assertEqual(kwargs['color'], 'red')


class DrawTranscriptIntegrationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.fig, self.ax = plt.subplots(figsize=(10, 4))

  def tearDown(self):
    plt.close(self.fig)
    super().tearDown()

  def test_draw_transcript_passes_num_transcripts(self):
    """Forward num_transcripts from draw_transcript to draw_strand_arrows."""

    transcript = _positive_strand_transcript_with_introns()
    interval = genome.Interval('chr1', 0, 1100)

    with mock.patch.object(
        plot_transcripts, 'draw_strand_arrows', autospec=True
    ) as mock_arrows:
      plot_transcripts.draw_transcript(
          ax=self.ax,
          transcript=transcript,
          interval=interval,
          y=0.0,
          num_transcripts=5,
      )

    mock_arrows.assert_called_once()
    _, call_kwargs = mock_arrows.call_args
    self.assertEqual(call_kwargs['num_transcripts'], 5)

  def test_plot_transcripts_passes_num_transcripts(self):
    """Pass len(transcripts) as num_transcripts in plot_transcripts."""
    transcripts = [
        _make_transcript(
            exons=[genome.Interval('chr1', 100, 500, strand='+')],
            gene_name='GENE_A',
            transcript_id='ENST00000000001',
        ),
        _make_transcript(
            exons=[genome.Interval('chr1', 600, 1000, strand='+')],
            gene_name='GENE_B',
            transcript_id='ENST00000000002',
        ),
    ]
    interval = genome.Interval('chr1', 0, 1100)

    with mock.patch.object(
        plot_transcripts, 'draw_transcript', autospec=True
    ) as mock_draw:
      plot_transcripts.plot_transcripts(
          ax=self.ax,
          transcripts=transcripts,
          interval=interval,
          label_name='gene_name',
      )

    self.assertEqual(mock_draw.call_count, 2)
    for call in mock_draw.call_args_list:
      _, call_kwargs = call
      self.assertEqual(call_kwargs['num_transcripts'], 2)


if __name__ == '__main__':
  absltest.main()
