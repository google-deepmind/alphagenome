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

"""Visualize transcripts/gene annotation in matplotlib."""

from collections.abc import Sequence
import dataclasses
import enum
from typing import Any

from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
import matplotlib as mpl
import matplotlib.figure
import matplotlib.path
import matplotlib.pyplot as plt


@dataclasses.dataclass
class TranscriptStyle:
  """Style specification for a transcript plot.

  CDS = protein coding sequence.
  UTR = untranslated region.

  Attributes:
    cds_height: CDS height.
    utr_height: UTR height.
    cds_color: CDS color.
    utr5_color: 5' UTR color.
    utr3_color: 3' UTR color.
    first_noncoding_exon_color: Color of the first non-coding exon. This helps
      to indicate transcript directionality.
    label_color: Label color.
    xlim_pad: Controls amount of whitespace on the region flanks.
  """

  cds_height: float
  utr_height: float
  cds_color: str
  utr5_color: str
  utr3_color: str
  first_noncoding_exon_color: str
  label_color: str
  xlim_pad: float


class TranscriptStylePreset(enum.Enum):
  """Style enum for transcript plots.

  Attributes:
    STANDARD: Standard transcript style.
    MINIMAL: Minimal transcript style.
  """

  STANDARD = TranscriptStyle(
      cds_height=0.7,
      utr_height=0.35,
      cds_color='#7f7f7f',  # Grey from tab10 palette.
      utr5_color='#ff7f0e',  # Orange.
      utr3_color='#1f77b4',  # Blue.
      first_noncoding_exon_color='#2ca02c',  # Green.
      label_color='#7f7f7f',
      xlim_pad=0.01,
  )

  MINIMAL = TranscriptStyle(
      cds_height=0.4,
      utr_height=0.22,
      cds_color='black',
      utr5_color='black',
      utr3_color='black',
      # TODO(b/377291432): find a nice way of specifying strand orientation.
      first_noncoding_exon_color='black',
      label_color='black',
      xlim_pad=0.05,
  )


def plot_transcripts(
    ax: plt.Axes,
    transcripts: Sequence[transcript_utils.Transcript],
    interval: genome.Interval,
    zero_origin: bool = False,
    label_name: str | None = None,
    transcript_style: TranscriptStyle = TranscriptStylePreset.STANDARD.value,
    plot_labels_once: bool = False,
    **kwargs,
):
  """Plot transcripts.

  Loops over each transcript in `transcripts` and calls `draw_transcript`.

  Args:
    ax: Matplotlib axis onto which to plot transcript annotations.
    transcripts: Sequence of transcripts returned by a
      transcript.TranscriptExtractor.
    interval: Genomic interval at which to visualize the transcripts.
    zero_origin: If True, the beginning of the interval will start with 0.
    label_name: Which label in transcript.info to draw next to the transcript.
    transcript_style: specification of transcript styling details.
    plot_labels_once: If True, labels will only be plotted once.
    **kwargs: kwargs passed to draw_transcript.
  """
  if not transcripts:
    return

  # Slightly pad x limits for nicer spacing, and shift x axis limits if needed.
  shift = -interval.start if zero_origin else 0
  xlim_pad = interval.width * transcript_style.xlim_pad
  ax.set_xlim(
      [interval.start + shift - xlim_pad, interval.end + shift + xlim_pad]
  )

  # Get typical label width and transcript heights.
  if label_name is not None:
    text_width = _get_text_width(
        transcripts[0].info[label_name],
        ax=ax,
        fontsize=kwargs.get('fontsize', None),
    )
  else:
    text_width = 0
  heights = _get_placement_heights(
      transcripts, extend_fraction=1.0, front_padding=text_width
  )

  labels_already_drawn = []
  for transcript in transcripts:
    # Add transcript labels.
    if label_name is not None:
      label = transcript.info[label_name]
    else:
      label = None

    draw_transcript(
        ax=ax,
        transcript=transcript,
        interval=interval,
        y=heights[transcript.transcript_id],
        cds_height=transcript_style.cds_height,
        utr_height=transcript_style.utr_height,
        cds_color=transcript_style.cds_color,
        utr5_color=transcript_style.utr5_color,
        utr3_color=transcript_style.utr3_color,
        first_noncoding_exon_color=transcript_style.first_noncoding_exon_color,
        label_color=transcript_style.label_color,
        shift=shift,
        label=None
        if (label in labels_already_drawn and plot_labels_once)
        else label,
        num_transcripts=len(transcripts),
        **kwargs,
    )

    labels_already_drawn.append(label)

  ax.set_ylim([min(heights.values()) - 1, max(heights.values()) + 1])


def draw_transcript(
    ax: plt.Axes,
    transcript: transcript_utils.Transcript,
    interval: genome.Interval,
    y: float,
    cds_height: float = 0.7,
    utr_height: float = 0.35,
    cds_color: str = '#7f7f7f',  # Grey from tab10 palette.
    utr5_color: str = '#ff7f0e',  # Orange.
    utr3_color: str = '#1f77b4',  # Blue.
    first_noncoding_exon_color: str = '#2ca02c',  # Green.
    shift: int = 0,
    label: str | None = None,
    label_color: str = '#7f7f7f',
    num_transcripts: int = 1,
    **kwargs,
) -> None:
  """Draw an individual transcript as rectangular components on an axis.

  CDS = protein coding sequence.
  UTR = untranslated region.

  This function is used by `plot_transcripts`.

  Args:
    ax: Matplotlib axis onto which to draw the transcript.
    transcript: Transcript to draw.
    interval: Genomic interval at which to visualize the transcript.
    y: Vertical position at which to draw the transcript.
    cds_height: CDS height.
    utr_height: UTR height.
    cds_color: CDS color in hex string format.
    utr5_color: 5' UTR color in hex string format.
    utr3_color: 3' UTR color  in hex string format.
    first_noncoding_exon_color: Color of the first non-coding exon. This helps
      to indicate transcript directionality. Hex string format.
    shift: X-axis shift.
    label: Optional label to draw next to the transcript.
    label_color: Label color.
    num_transcripts: Total number of transcripts being drawn, used for dynamic
      arrow sizing.
    **kwargs: Additional keyword arguments passed to matplotlib plotting
      functions.
  """
  ax.set_yticklabels([])
  ax.set_yticks([])

  introns = transcript.introns
  if introns:
    intron_x = []
    intron_y = []
    for intron in introns:
      intron_x.extend([intron.start + shift, intron.end + shift, None])
      intron_y.extend([y, y, None])
    ax.plot(intron_x, intron_y, color=cds_color, linewidth=0.5)

  # TODO: b/377291432 - Skip drawing an exon if it will be drawn below
  # separately to avoid overlap if alpha<1 and overlaps in vector format.
  if transcript.cds is not None:
    for exon in transcript.utr5:
      draw_interval(
          ax=ax,
          interval=exon,
          y=y,
          shift=shift,
          height=utr_height,
          color=utr5_color,
          **kwargs,
      )
    for exon in transcript.cds:
      draw_interval(
          ax=ax,
          interval=exon,
          y=y,
          shift=shift,
          height=cds_height,
          color=cds_color,
          **kwargs,
      )
    for exon in transcript.utr3:
      draw_interval(
          ax=ax,
          interval=exon,
          y=y,
          shift=shift,
          height=utr_height,
          color=utr3_color,
          **kwargs,
      )
  else:
    for exon in transcript.exons:
      draw_interval(
          ax=ax,
          interval=exon,
          y=y,
          shift=shift,
          height=utr_height,
          color=cds_color,
          **kwargs,
      )

  # Draw the first non-coding exon with a special color if specified.
  if transcript.exons:
    first_exon_index = 0 if transcript.is_negative_strand else -1
    draw_interval(
        ax=ax,
        interval=transcript.exons[first_exon_index],
        y=y,
        height=utr_height,
        shift=shift,
        color=first_noncoding_exon_color,
        **kwargs,
    )

  if label is not None and transcript.exons:
    draw_interval(
        ax=ax,
        interval=transcript.exons[0],
        y=y,
        shift=shift,
        label=label,
        height=utr_height,
        color=cds_color,
        label_color=label_color,
        **kwargs,
    )

  # Draw strand arrows across the full transcript span.
  draw_strand_arrows(
      ax=ax,
      transcript=transcript,
      interval=interval,
      y=y,
      color=cds_color,
      cds_height=cds_height,
      num_transcripts=num_transcripts,
  )


def draw_strand_arrows(
    ax: plt.Axes,
    transcript: transcript_utils.Transcript,
    interval: genome.Interval,
    y: float,
    color: str,
    *,
    cds_height: float = 0.22,
    num_transcripts: int = 1,
    max_arrows_per_intron: int = 5,
) -> None:
  """Draw strand direction arrows on intron lines.

  Arrow count per intron is computed dynamically based on the intron's width
  relative to the visible interval. Marker size is derived from the UTR height
  so arrows are always visually smaller than UTR exons.

  Args:
    ax: Matplotlib axis.
    transcript: The transcript being drawn.
    interval: The visible genomic interval.
    y: Vertical position of the transcript.
    color: Arrow color.
    cds_height: CDS height in data coordinates, used to scale arrows.
    num_transcripts: Total number of transcripts being drawn.
    max_arrows_per_intron: Maximum number of arrows per intron.
  """
  introns = transcript.introns
  if not introns:
    return

  fig = ax.get_figure()
  if fig is not None:
    _, fig_height_inches = fig.get_size_inches()
    ax_height_inches = ax.get_position().height * fig_height_inches
    y_range = num_transcripts + 2
    if y_range > 0:
      pts_per_data = (ax_height_inches * 72) / y_range
      markersize = min(4.0, cds_height * pts_per_data * 2)
    else:
      markersize = 4.0
  else:
    markersize = 4.0

  # Custom chevron path: two line segments forming > or < shape.
  if transcript.is_negative_strand:
    chevron = matplotlib.path.Path(
        [(0.5, 0.5), (-0.5, 0.0), (0.5, -0.5)],
        [
            matplotlib.path.Path.MOVETO,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.LINETO,
        ],
    )
  else:
    chevron = matplotlib.path.Path(
        [(-0.5, 0.5), (0.5, 0.0), (-0.5, -0.5)],
        [
            matplotlib.path.Path.MOVETO,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.LINETO,
        ],
    )

  arrow_positions = []
  for intron in introns:
    intron_to_interval_fraction = intron.width / interval.width
    # Skip arrows for introns that are too small.
    if intron_to_interval_fraction < 0.01:
      continue
    # Use sqrt scaling so large introns don't get overwhelmed with arrows.
    num_arrows = min(
        max(1, round(intron_to_interval_fraction**0.5 * max_arrows_per_intron)),
        max_arrows_per_intron,
    )
    space = intron.width / (num_arrows + 1)
    for i in range(1, num_arrows + 1):
      arrow_pos = intron.start + i * space
      # Skip arrows too close to interval edges.
      if arrow_pos < interval.start + 10 or arrow_pos > interval.end - 10:
        continue
      arrow_positions.append(arrow_pos)

  if arrow_positions:
    ax.plot(
        arrow_positions,
        [y] * len(arrow_positions),
        marker=chevron,
        markersize=markersize,
        color=color,
        fillstyle='none',
        markeredgewidth=0.8,
        linestyle='none',
        clip_on=True,
    )


def draw_interval(
    ax: plt.Axes,
    interval: genome.Interval,
    y: float,
    label: str | None = None,
    height: float = 0.5,
    shift: int = 0,
    label_color: str = '#7f7f7f',
    **kwargs,
):
  """Draw rectangle patch on the axis given a genomic interval.

  Args:
    ax: Matplotlib axis onto which to draw the interval.
    interval: Genomic interval to draw.
    y: Vertical position at which to draw the interval.
    label: Optional label to draw next to the interval.
    height: Height of the interval.
    shift: X-axis shift.
    label_color: Label color in hex string format.
    **kwargs: Additional keyword arguments passed to matplotlib plotting
      functions.
  """
  xy = (interval.start + shift, y - height / 2)
  ax.add_patch(
      mpl.patches.Rectangle(
          xy=xy,
          width=interval.width,
          height=height,
          clip_on=True,
          linewidth=0,
          **kwargs,
      )
  )

  # Add center-aligned text label.
  if label is not None:
    ax.text(
        x=max(xy[0], ax.get_xlim()[0]),
        y=y,
        s=label,
        color=label_color,
        horizontalalignment='right',
        verticalalignment='center',
    )


def _get_placement_heights(
    transcripts: Sequence[transcript_utils.Transcript],
    extend_fraction: float = 1.0,
    front_padding: float = 0.0,
) -> dict[Any, int]:
  """Get heights at which to place the transcripts.

  Uses a greedy interval packing algorithm: transcripts are sorted by width in
  descending order and placed sequentially into the highest non-overlapping
  vertical row level.

  Args:
    transcripts: Sequence of transcripts to place.
    extend_fraction: Multiplier for extending transcript end bounds.
    front_padding: Extra padding to subtract from transcript start bounds.

  Returns:
    Mapping of transcript ID to vertical row level index.
  """
  sorted_transcripts = sorted(
      transcripts, key=lambda x: x.transcript_interval.width, reverse=True
  )
  level_intervals: list[list[tuple[float, float]]] = []
  transcript_levels = {}
  for transcript in sorted_transcripts:
    t_start = transcript.transcript_interval.start - front_padding
    t_end = int(transcript.transcript_interval.end * extend_fraction)
    placed = False
    level_idx = 0
    while not placed:
      if level_idx >= len(level_intervals):
        level_intervals.append([])
      overlaps = any(
          (t_start < end and t_end > start)
          for start, end in level_intervals[level_idx]
      )
      if overlaps:
        level_idx += 1
      else:
        level_intervals[level_idx].append((t_start, t_end))
        transcript_levels[transcript.transcript_id] = level_idx
        placed = True

  max_level = len(level_intervals) - 1 if level_intervals else 0
  return {
      transcript_id: max_level - level_idx
      for transcript_id, level_idx in transcript_levels.items()
  }


def _get_text_width(
    label: str, ax: plt.Axes, fontsize: float | None = None
) -> float:
  """Get text width in data coordinates."""
  if not label:
    return 0.0

  if fontsize is None:
    fontsize = float(mpl.rcParams['font.size'])

  xlim = ax.get_xlim()
  x_span = abs(xlim[1] - xlim[0])
  text_width_pts = len(label) * 0.6 * fontsize

  fig = ax.get_figure()
  if fig is not None:
    fig_width_inches = fig.get_size_inches()[0]
    ax_width_inches = ax.get_position().width * fig_width_inches
    ax_width_pts = ax_width_inches * 72.0
    if ax_width_pts > 0:
      return text_width_pts * (x_span / ax_width_pts)

  return len(label) * 0.008 * x_span
