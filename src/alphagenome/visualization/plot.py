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

"""Plotting functions."""

from collections.abc import Callable, Mapping, Sequence
import functools
import math
from typing import Any, Literal

from absl import logging
from alphagenome.data import genome
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@functools.lru_cache(maxsize=8)
def _get_text_path(
    letter: str, letter_colors_scheme: str
) -> tuple[mpl.text.TextPath, str]:
  """Returns a memoized TextPath and color for the given letter and scheme."""
  letters_width_colors = dict(
      # Consistent with `classic` scheme from
      # https://github.com/gecrooks/weblogo/blob/master/weblogo/logo.py#L136
      default={
          'A': (0.35, 'darkgreen'),
          'C': (0.366, 'blue'),
          'G': (0.384, 'orange'),
          'T': (0.305, 'red'),
      },
      # Consistent with factorbook scheme from https://www.factorbook.org/
      factorbook={
          'A': (0.305, 'red'),
          'C': (0.366, 'blue'),
          'G': (0.384, 'orange'),
          'T': (0.35, 'darkgreen'),
      },
  )
  if (
      letters_width_color := letters_width_colors.get(letter_colors_scheme)
  ) is None:
    raise ValueError(
        f'letter_colors_scheme must be one of {letters_width_colors.keys()}.'
        f'Got "{letter_colors_scheme}".'
    )
  else:
    if (letter_width_color := letters_width_color.get(letter)) is None:
      raise ValueError(
          f'letter must be one of {letters_width_color.keys()}. Got "{letter}".'
      )
    else:
      letter_width, letter_color = letter_width_color
      font = mpl.font_manager.FontProperties(weight='bold')
      return (
          mpl.text.TextPath((-letter_width, 0), letter, size=1, prop=font),
          letter_color,
      )


def seqlogo(
    letter_heights: np.ndarray,
    alphabet: str = 'ACGT',
    one_based: bool = True,
    start: int = 0,
    ax: mpl.axes.Axes | None = None,
    letter_colors: Literal['default', 'factorbook'] = 'default',
):
  """Sequence logo plot.

  Args:
    letter_heights: 2D array with shape (sequence length, len(alphabet)) shape.
      Positive values indicate that a logo element extends above the horizontal
      axis, and negative values indicate that the element extends below.
    alphabet: Alphabet corresponding to the channel axis of `letter_heights`.
    one_based: If True, plot letters at 1-based coordinates: first letter will
      be centered at 1. If False, the first letter will be between 0 and 1.
    start: Interval start.
    ax: matplotlib axis to add figures to.
    letter_colors: Name of the color scheme to use for coloring each letter.
      Must be one of 'default', a scheme consistent with weblogo
      (https://github.com/gecrooks/weblogo/blob/master/weblogo/logo.py#L136) or
      'factorbook', a scheme consistent with Factorbook
      (https://www.factorbook.org/).
  """
  ax = ax if ax is not None else plt.gca()

  if letter_heights.ndim != 2:
    raise ValueError(
        'Expecting a 2D matrix of shape (sequence_length, len(alphabed))'
    )
  if letter_heights.shape[1] != len(alphabet):
    raise ValueError('Last axis needs to match len(alphabet)')

  globscale = 1.35
  paths = []
  facecolors = []

  for x_offset, heights in enumerate(letter_heights):
    last_positive_y = 0.0
    last_negative_y = 0.0
    x = start + x_offset + 0.5 + 0.5 * one_based

    for height, letter in sorted(zip(heights, alphabet)):
      if height == 0:
        continue
      # Start with lowest height and keep track of the last y coordinate.
      if height > 0:
        y_position = last_positive_y
        last_positive_y += height
      else:
        y_position = last_negative_y
        last_negative_y += height

      base_path, letter_color = _get_text_path(letter, letter_colors)

      transform = (
          mpl.transforms.Affine2D()
          .scale(globscale, height * globscale)
          .translate(x, y_position)
      )
      paths.append(base_path.transformed(transform))
      facecolors.append(letter_color)

  if paths:
    collection = mpl.collections.PathCollection(
        paths,
        facecolors=facecolors,
        edgecolors='none',
        linewidths=0,
        transform=ax.transData,
    )
    ax.add_collection(collection)

  # Make sure all letters are displayed.
  ax.autoscale_view()


def plot_contact_map(
    contact_map: pd.DataFrame | np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    square: bool = True,
    cbar_shrink: float = 0.4,
    ax=None,
    **kwargs,
):
  """Visualize a contact map.

  A contact map is a 2D array of values, where each value represents the
  probability that two DNA bases are in contact.

  The array is symmetric, i.e., the (i, j) values and the (j, i) values of
  `contact_map` are equal.

  Args:
    contact_map: Contact map to visualize.
    vmin: Minimum value for the colorbar.
    vmax: Maximum value for the colorbar.
    square: If `True`, plots the contact map as a square. If `False`, plots the
      contact map as a rectangle with a shorter height than width.
    cbar_shrink: Fraction by which to multiply the size of the colorbar.
    ax: Matplotlib axis to add the heatmap to.
    **kwargs: Additional keyword arguments passed to `sns.heatmap`.
  """

  def tuple2string(interval_tuple):
    chromosome, start, end = interval_tuple
    return f'{chromosome}:{start:,}-{end:,}'

  if isinstance(contact_map, pd.DataFrame):
    interval_string = (
        tuple2string(contact_map.index[0])
        + ' - '
        + tuple2string(contact_map.index[-1])
    )
    contact_map: np.ndarray = contact_map.values
  else:
    interval_string = ''

  if vmin is None:
    vmin = np.nanmin(contact_map[contact_map > 0])
  if vmax is None:
    vmax = np.nanmax(contact_map)
  fall_cmap = mpl.colors.LinearSegmentedColormap.from_list(
      'fall',
      colors=[
          'white',
          (245 / 256, 166 / 256, 35 / 256),
          (208 / 256, 2 / 256, 27 / 256),
          'black',
      ],
  )
  log_norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
  cbar_min = math.floor(math.log10(log_norm.vmin))
  cbar_max = 1 + math.ceil(math.log10(log_norm.vmax))
  cbar_ticks = [math.pow(10, i) for i in range(cbar_min, cbar_max)]

  if ax is None:
    ax = plt.gca()
  sns.heatmap(
      contact_map + 1e-10,
      cmap=fall_cmap,
      norm=log_norm,
      cbar_kws={'ticks': cbar_ticks, 'shrink': cbar_shrink},
      vmin=log_norm.vmin,
      vmax=log_norm.vmax,
      xticklabels=False,
      yticklabels=False,
      square=square,
      ax=ax,
      **kwargs,
  )

  ax.set_xlabel(interval_string)


def plot_track(
    arr: np.ndarray,
    ax: plt.Axes,
    x: np.ndarray | None = None,
    legend: bool = False,
    ylim: float | None = None,
    color: str | Sequence[str] | None = None,
    filled: bool = False,
) -> None:
  """Plot a single track on the axis after inferring track type from array.

  This function infers the type of track plot depending on channel
  dimensionality of `arr`:

    - If `arr` is a single-dimensional array, plot a single track line.
    - If `arr` is a 2D array, and interpreting the second dimension as the
      channel dimension:

      - Plot a sequence logo if the number of channels is 4 (which suggests one
        channel per DNA base).
      - Plot two overlapping line plots if the number of channels is 2 (which
        suggests one channel per DNA strand).

  Args:
    arr: array of values to plot.
    ax: matplotlib axis to use for plotting the values on.
    x: optional array of values to use for setting the x axis values. If absent,
      then sequential values starting from 1 to the length of the array arr will
      be used.
    legend: whether to draw a legend or not on the double stranded DNA plot.
    ylim: y axis limit.
    color: color(s) used for line plots.
    filled: whether to fill the space between the x axis and the line (or leave
      it as whitespace).

  Returns:
    None. Adds a plot to the provided axis.
  """
  sequence_length = len(arr)

  if x is None:
    x = np.arange(1, sequence_length + 1)  # One-based indexing.

  if arr.ndim == 1 or arr.shape[1] == 1:
    if color is not None:
      if isinstance(color, (list, tuple)):
        logging.warning(
            'A sequence of colors was passed to plot_track but the second '
            'array dimension is 1 suggesting single stranded DNA. Using the '
            'first entry %s as the line color.',
            color[0],
        )
        color = color[0]

    # In the case of a boolean array, we fill between values and the x axis and
    # simplify the y axis components.
    if arr.dtype == bool:
      ax.fill_between(x, arr, step='mid', color=color)
      ax.yaxis.set_ticks_position('none')
      ax.set_yticklabels([])
      ax.spines['left'].set_visible(False)
      ax.set_ylim([0, 1.5])

    else:
      if filled:
        ax.fill_between(x, np.ravel(arr), color=color)
      ax.plot(x, np.ravel(arr), color=color)

  elif arr.shape[1] == 4:
    seqlogo(arr, ax=ax)

  elif arr.shape[1] == 2:
    if color is not None:
      if not isinstance(color, Sequence) or len(color) != 2:
        raise ValueError(
            'Must pass sequence of 2 colors if plotting 2 dimensional array.'
        )
      color_1, color_2 = color
    else:
      color_1, color_2 = None, None
    ax.plot(x, arr[:, 0], label='pos', color=color_1)
    ax.plot(x, arr[:, 1], label='neg', color=color_2)
    if legend:
      ax.legend()
  else:
    raise ValueError(
        f'Do not know how to plot array with shape[1] != {arr.shape[1]}. '
        'Valid values are: 1, 2, or 4.'
    )
  if ylim is not None:
    ax.set_ylim(ylim)


def plot_tracks(
    tracks: Mapping[str, Any],
    x: np.ndarray | None = None,
    title: str | None = None,
    legend: bool = False,
    fig_width: float = 20,
    fig_track_height: float | Mapping[str, float] = 1.5,
    ylim: str | Mapping[str, tuple[int, int]] | tuple[int, int] | None = 'auto',
    yticks_min_max_only: bool = False,
    ylab: bool = True,
    color: str | Mapping[str, str] | None = None,
    horizontal_ylab: bool = True,
    filled_tracks: Sequence[str] | None = None,
    despine: bool = True,
    despine_keep_bottom: bool = False,
    plot_track_fn: Callable[..., Any] = plot_track,
) -> mpl.figure.Figure:
  """Plot multiple tracks as subplots within one matplotlib figure.

  Custom plotting functions may be passed as `plot_track_fn`. Otherwise,
  `plot_track` will be used by default.

  Args:
    tracks: dictionary of tracks to plot. Example input: tracks = {'a': [1,2,3],
      'b': [4,5,4]}. The arrays in the dict must have the same length.
    x: optional array of x axis values. If absent, these will be inferred to be
      from 1 to the length of the arrays in tracks.
    title: Optional title to be added to the first subplot in the figure.
    legend: Whether to plot a legend or not.
    fig_width: Figure width.
    fig_track_height: Either a scalar specifying the total height of the figure,
      or a dictionary specifying the height of the subplot for each track.
    ylim: y axis limit. Must be either "same" (shared y limit across subplots)
      or "auto" (free scales, indicating separate limits per subplot).
    yticks_min_max_only: whether the only y axis ticks should be the min and max
      values.
    ylab: y axis label.
    color:  Either a string or a dict indicating a color per track.
    horizontal_ylab: Whether to display the y axis label horizontally (instead
      of the default vertical orientation).
    filled_tracks: List of track names that should have filled line plots drawn
      (instead of the default whitespace between the values and x axis).
    despine: Whether to remove top, right, and bottom spines.
    despine_keep_bottom: Whether to remove top and right spines, but keep the
      bottom spine.
    plot_track_fn: Optional custom function for plotting tracks. If absent,
      defaults to `plot_track`.

  Returns:
    Matplotlib figure.
  """
  # Set figure height (either total height, or per subplot if passed a dict).
  if isinstance(fig_track_height, dict):
    fig_height = sum(fig_track_height.values())
    gridspec_kw = {
        'height_ratios': [
            height / fig_height for height in fig_track_height.values()
        ]
    }
  else:
    fig_height = fig_track_height
    gridspec_kw = dict()

  # Generate figure axes.
  fig, axes = plt.subplots(
      nrows=len(tracks),
      ncols=1,
      figsize=(fig_width, fig_height),
      gridspec_kw=gridspec_kw,
      sharex=True,
  )
  if len(tracks) == 1:
    axes = [axes]

  # Set y axis limits.
  if ylim == 'same':
    ylim = (
        0,
        max(v.max() for v in tracks.values() if isinstance(v, np.ndarray)),
    )
  elif ylim == 'auto':
    ylim = None
  elif isinstance(ylim, str):
    raise ValueError('Only "same" and "auto" are valid strings for ylim.')

  # Iterate over tracks and plot one track per axis in the figure.
  for i, (ax, (track, arr)) in enumerate(zip(axes, tracks.items())):
    # Only set title in the first subplot.
    if i == 0 and title is not None:
      ax.set_title(title)

    # Only set x axis ticks in the final subplot.
    if i != len(tracks) - 1:
      ax.xaxis.set_ticks_position('none')

    track_ylim = ylim[track] if isinstance(ylim, dict) else ylim
    track_color = color[track] if isinstance(color, dict) else color

    # If the array is in fact a callable, then it is intended to be used as a
    # custom plotting function (most commonly, for plotting transcripts).
    if hasattr(arr, '__call__'):
      # TODO: b/377225766 - Allow custom plot functions as dictionary elements.
      arr(ax, ylim=ylim, color=color)
    else:
      filled_tracks = filled_tracks or []
      plot_track_fn(
          arr,
          ax=ax,
          x=x,
          legend=legend,
          ylim=track_ylim,
          color=track_color,
          filled=track in filled_tracks,
      )

    # Set y axis label.
    if ylab:
      if horizontal_ylab:
        ax.set_ylabel(
            track,
            rotation=0,
            multialignment='center',
            va='center',
            ha='right',
            labelpad=5,
        )
      else:
        ax.set_ylabel(track)

    # Only draw the y axis ticks on the min and max value locations.
    if yticks_min_max_only:
      if track_ylim is None:
        track_ylim = ax.get_yticks()[[0, -1]]
      # TODO: b/377226499 - Preventing negative values might be sub-optimal.
      assert isinstance(track_ylim, tuple[int, int])
      minimum = max(track_ylim[0], 0)
      maximum = track_ylim[1]
      ax.spines['left'].set_bounds(minimum, maximum)
      ax.set_yticks([minimum, maximum])

    if despine:
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      if i != len(tracks) - 1 and despine_keep_bottom:
        ax.spines['bottom'].set_visible(True)
      else:
        ax.spines['bottom'].set_visible(False)

  # Enable default tick locator for the final subplot.
  axes[-1].xaxis.set_major_locator(mpl.ticker.AutoLocator())

  # No height for whitespace between subplots.
  fig.subplots_adjust(hspace=0)

  return fig


def sashimi_plot(
    junctions: Sequence[genome.Junction],
    ax: plt.Axes,
    interval: genome.Interval | None = None,
    filter_threshold: float = 0.01,
    annotate_counts: bool = True,
    rng: np.random.Generator | None = None,
    color: str | None = None,
):
  """Plot splice junctions as a Sashimi plot.

  Sashimi plots (first described [here](https://arxiv.org/abs/1306.3466)
  visualize splice junctions from an RNA-seq experiment.

  According to the authors,
  * genomic reads are converted into read densities
  * junction reads are plotted as arcs whose width is determined by the number
  of reads aligned to the junction spanning the exons connected by the arc.

  Args:
    junctions: Splice junctions to plot.
    ax: Matplotlib axis to add the plot to.
    interval: Interval to plot, used to filter text annotations.
    filter_threshold: Junctions whose read count (`junction.k`) is None or below
      this threshold are filtered out.
    annotate_counts: Whether to annotate the junctions with read counts.
    rng: Optional random number generator to use for jittering junction paths.
      If unset will use NumPy's default random number generator.
    color: Color of the Sashimi arcs.
  """
  rng = rng or np.random.default_rng()
  total = np.sum([junction.k for junction in junctions if junction.k])
  # Random jitter position to avoid overlap.
  jitters = rng.uniform(low=0.05, high=0.15, size=len(junctions))
  for junction, jt in zip(junctions, jitters):
    if junction.k is None or junction.k < filter_threshold:
      continue
    k = junction.k / total
    verts = [
        (junction.start, 0.0),
        (junction.start, jt),
        (junction.end, jt),
        (junction.end, 0.0),
    ]

    path = mpl.path.Path(
        verts,
        [
            mpl.path.Path.MOVETO,
            mpl.path.Path.CURVE4,
            mpl.path.Path.CURVE4,
            mpl.path.Path.CURVE4,
        ],
    )
    patch = mpl.patches.PathPatch(
        path, facecolor='none', lw=min(k * 30, 5), edgecolor=color
    )
    ax.add_patch(patch)
    if annotate_counts:
      text_pos = junction.center()
      if interval is not None:
        if text_pos < interval.start or text_pos > interval.end:
          continue
      ax.text(text_pos, 0.8 * jt, junction.k, horizontalalignment='center')
  ax.set_ylim(0, 0.15)


def pad_track(track: np.ndarray, new_len: int, value: int = 0) -> np.ndarray:
  """Pad a track with `value` to the desired length.

  If new_len - len(track) is an even number, the same amount of padding is added
  to both sides.

  If new_len - len(track) is an odd number, the extra padded element will be
  added at the end of the array.

  Args:
    track: Track to pad.
    new_len: Desired length of the padded track.
    value: Value to use for padding.

  Returns:
    Padded track.
  """
  assert track.ndim == 2
  assert new_len >= len(track)

  out = np.empty((new_len, track.shape[1]))
  out[:] = value
  delta = new_len - len(track)
  i = delta // 2
  j = i + len(track)
  out[i:j] = track
  return out


_ColorType = (
    str | tuple[float, float, float] | tuple[float, float, float, float]
)
_ARC_MIN_HEIGHT = 0.35
_ARC_SPAN_SCALE = 1.5
_ARC_BEZIER_PEAK_SCALE = 4.0 / 3.0
_ARC_Y_LIM_LOW = -0.05
_ARC_Y_LIM_HIGH = 1.08


def _numeric_col(df: pd.DataFrame, col: str) -> np.ndarray:
  """Extracts `df[col]` as a float64 array, mapping missing values to `NaN`."""
  return df[col].to_numpy(dtype=np.float64, na_value=np.nan)


def _normalize_arc_scores(
    raw_scores: np.ndarray, vmin: float | None, vmax: float | None
) -> np.ndarray:
  """Normalizes enhancer scores to `[0, 1]`, mapping mixed NaN values to `0`."""
  finite_mask = np.isfinite(raw_scores)
  if not np.any(finite_mask):
    return np.ones(raw_scores.shape[0], dtype=np.float64)
  finite_vals = raw_scores[finite_mask]
  s_min = min(0.0, float(finite_vals.min())) if vmin is None else float(vmin)
  s_max = float(finite_vals.max()) if vmax is None else float(vmax)
  safe = np.where(finite_mask, raw_scores, s_min)
  if s_max > s_min:
    clipped = np.clip((safe - s_min) / (s_max - s_min), 0.0, 1.0)
    return np.where(finite_mask, clipped, 0.0)
  cond = safe > 0.0 if vmin is None and vmax is None else safe >= s_min
  return np.where(finite_mask & cond, 1.0, 0.0)


def enhancer_arcs_plot(
    links: pd.DataFrame,
    ax: mpl.axes.Axes | None = None,
    tss: int | None = None,
    *,
    interval: genome.Interval | None = None,
    interval_contained: bool = True,
    gene_id: str | None = None,
    start_col: str = 'start',
    end_col: str = 'end',
    peak_col: str | None = None,
    tss_col: str = 'tss',
    score_col: str | None = 'gradient_score',
    color: _ColorType = '#810f7c',
    cmap: str | mpl.colors.Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha_range: tuple[float, float] = (0.35, 0.9),
    linewidth_range: tuple[float, float] = (1.0, 2.6),
    upward: bool = True,
) -> None:
  """Plots enhancer-to-gene links as cubic Bezier arcs on `ax`.

  Args:
    links: DataFrame of enhancer-to-gene links.
    ax: Matplotlib axis to draw on (defaults to `plt.gca()`).
    tss: Target gene TSS coordinate; if None, uses `tss_col` from `links`.
    interval: Optional genomic interval filter and horizontal span reference.
    interval_contained: If True, only plot arcs whose endpoints both lie within
      `interval`; if False, plot any arc whose enhancer-to-TSS span overlaps
      `interval`. Ignored when `interval` is None.
    gene_id: Optional target gene ID filter (ignoring Ensembl version suffix).
    start_col: Column name for enhancer start coordinate.
    end_col: Column name for enhancer end coordinate.
    peak_col: Optional summit column name; if None, uses `0.5 * (start + end)`.
    tss_col: Column name for target gene TSS when `tss` is None.
    score_col: Column used to scale arc opacity, width, and colormap shading.
    color: Default arc color when `cmap` is None.
    cmap: Optional colormap for shading arcs by score.
    vmin: Optional minimum score for normalization.
    vmax: Optional maximum score for normalization.
    alpha_range: Minimum and maximum alpha transparency for arcs.
    linewidth_range: Minimum and maximum line width for arcs.
    upward: If True, arcs curve upward; otherwise downward.

  Raises:
    ValueError: If `tss` is None and `tss_col` is not in `links`.
    KeyError: If required coordinate, gene, peak, or score columns are missing
      from `links`.
  """
  if tss is None and tss_col not in links.columns:
    raise ValueError(
        f'Either `tss` must be provided or `{tss_col}` must be in `links`.'
    )

  ax = ax if ax is not None else plt.gca()
  ax.axhline(0.0, color='#bdbdbd', linewidth=0.6, zorder=1)
  y_scale = 1.0 if upward else -1.0

  if gene_id is not None:
    gene_ids = links['gene_id'].astype(str).str.split('.', n=1).str[0]
    gene_df = links[gene_ids == gene_id.split('.', 1)[0]]
  else:
    gene_df = links

  starts = _numeric_col(gene_df, start_col)
  midpoints = 0.5 * (starts + _numeric_col(gene_df, end_col))
  if peak_col is not None:
    peaks = _numeric_col(gene_df, peak_col)
    enh_vals = np.where(np.isfinite(peaks), peaks, midpoints)
  else:
    enh_vals = midpoints
  if tss is not None:
    tss_vals = np.full(len(gene_df), float(tss))
  else:
    tss_vals = _numeric_col(gene_df, tss_col)
  scores = None if score_col is None else _numeric_col(gene_df, score_col)

  finite = np.isfinite(enh_vals) & np.isfinite(tss_vals)
  if interval is not None:
    v0 = float(interval.start)
    v1 = float(interval.end)
    arc_lo = np.minimum(enh_vals, tss_vals)
    arc_hi = np.maximum(enh_vals, tss_vals)
    contained = (arc_lo >= v0) & (arc_hi <= v1)
    in_span = (
        contained
        if interval_contained
        else contained | ((arc_lo < v1) & (arc_hi > v0))
    )
    if 'chromosome' in gene_df.columns:
      same_chr = gene_df['chromosome'] == interval.chromosome
      in_chrom = same_chr.to_numpy(dtype=bool, na_value=False)
    else:
      in_chrom = True
    keep = finite & in_span & in_chrom
  else:
    keep = finite

  enh_positions = enh_vals[keep]
  tss_positions = tss_vals[keep]
  if enh_positions.size > 0:
    if scores is not None:
      norm_scores = _normalize_arc_scores(scores[keep], vmin, vmax)
    else:
      norm_scores = np.ones(enh_positions.size, dtype=np.float64)
    span = (
        float(interval.width)
        if interval is not None
        else float(np.ptp(np.concatenate([enh_positions, tss_positions])))
    )
    max_span = max(span, 1.0)

    colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    codes = [mpl.path.Path.MOVETO] + [mpl.path.Path.CURVE4] * 3
    min_alpha, max_alpha = alpha_range
    min_linewidth, max_linewidth = linewidth_range
    for enh_x, tss_x, score in zip(
        enh_positions.tolist(), tss_positions.tolist(), norm_scores.tolist()
    ):
      span_frac = min(1.0, abs(enh_x - tss_x) / max_span * _ARC_SPAN_SCALE)
      apex = _ARC_MIN_HEIGHT + (1.0 - _ARC_MIN_HEIGHT) * span_frac
      y_ctrl = y_scale * _ARC_BEZIER_PEAK_SCALE * apex
      verts = [
          (tss_x, 0.0),
          (tss_x + 0.25 * (enh_x - tss_x), y_ctrl),
          (tss_x + 0.75 * (enh_x - tss_x), y_ctrl),
          (enh_x, 0.0),
      ]
      ax.add_patch(
          mpl.patches.PathPatch(
              mpl.path.Path(verts, codes),
              facecolor='none',
              edgecolor=colormap(score) if colormap is not None else color,
              alpha=min_alpha + score * (max_alpha - min_alpha),
              linewidth=min_linewidth + score * (max_linewidth - min_linewidth),
              zorder=2 + score,
          )
      )

  if interval is not None:
    ax.set_xlim(interval.start, interval.end)
  elif enh_positions.size > 0:
    ax.autoscale_view(scalex=True, scaley=False)
  if upward:
    ax.set_ylim(_ARC_Y_LIM_LOW, _ARC_Y_LIM_HIGH)
  else:
    ax.set_ylim(-_ARC_Y_LIM_HIGH, -_ARC_Y_LIM_LOW)
