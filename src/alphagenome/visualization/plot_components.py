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


"""Module containing the main plotting code for AlphaGenome model outputs.

Three main elements are:

  * `plot` function: The primary function for visualizing model outputs.
  * Component classes: Implement the visualization components (e.g., tracks,
    contact maps).
  * Annotation classes: Implement the visualization annotations (e.g.,
    intervals, variants).
"""

import abc
from collections.abc import Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float32  # pylint: disable=g-importing-member
from matplotlib import colors as plt_colors

from alphagenome.data import genome, junction_data, track_data
from alphagenome.data import transcript as transcript_utils
from alphagenome.visualization import plot as plot_lib
from alphagenome.visualization import plot_transcripts

# String, RGB or RGBA color.
_ColorType = (
    str | tuple[float, float, float] | tuple[float, float, float, float]
)


def plot(
    components: Sequence['AbstractComponent'],
    interval: genome.Interval,
    fig_width: int = 20,
    fig_height_scale: float = 1.0,
    title: str | None = None,
    despine: bool = True,
    despine_keep_bottom: bool = False,
    annotations: Sequence['AbstractAnnotation'] | None = None,
    annotation_offset_range: tuple[float, float] = (0.1, 0.6),
    hspace: float = 0.3,
    xlabel: str | None = None,
) -> matplotlib.figure.Figure:
  """Plots AlphaGenome model outputs as individual panels of 'components'.

  This function generates a visualization of AlphaGenome model outputs
  using a combination of components (e.g., tracks, contact maps) and
  annotations (e.g., intervals, variants).

  Args:
    components: A sequence of components to visualize.
    interval: The genomic interval to focus on (similar to setting xlim in
      matplotlib).
    fig_width: The total figure width.
    fig_height_scale: Height of the individual track unit. Total plot height is
      determined as a sum of the individual component heights.
    title: An optional title for the overall plot.
    despine: Whether to remove top, right, and bottom spines from the axes.
    despine_keep_bottom: Whether to remove top and right spines, but keep the
      bottom spine. Does not apply to transcript components, which are always
      despined.
    annotations: Sequence of annotations to visualise across all components.
    annotation_offset_range: Relative positions in y-axis to place labels for
      annotations. Each set of labels in the 'annotations' list are spaced
      evenly within this range.
    hspace: Vertical whitespace between subplots to avoid tick label overlap
      (relative fraction).
    xlabel: If a non-empty string is provided, it is used as the x-axis label.
      If an empty string is provided, the x-axis label is removed. If None, the
      default x-axis label is used.

  Returns:
    A matplotlib figure.
  """

  # If we are adding text labels for any of the annotation components,
  # we need to add an extra empty component at the top.
  add_label_axis = False
  if annotations is not None:
    add_label_axis = any(annot.has_labels for annot in annotations)
    components = (
        [EmptyComponent()] + list(components)
        if add_label_axis
        else list(components)
    )

  num_axes = sum(component.num_axes for component in components)
  fig_height = sum(
      component.total_height * fig_height_scale for component in components
  )

  offset = 0
  gridspec_kw = {'height_ratios': []}
  for component in components:
    for i in range(component.num_axes):
      gridspec_kw['height_ratios'].append(
          component.get_ax_height(i) * fig_height_scale / fig_height
      )
      offset += 1

  fig, axes = plt.subplots(
      nrows=num_axes,
      ncols=1,
      figsize=(fig_width, fig_height),
      gridspec_kw=gridspec_kw,
      sharex=True,
  )
  # Handle the case of a single axis (e.g. single REF / ALT plot).
  if not isinstance(axes, np.ndarray):
    axes = [axes]

  offset = 0
  for component in components:
    for i in range(component.num_axes):
      ax = axes[offset]
      component.plot_ax(ax, axis_index=i, interval=interval)

      if offset == 0 and title is not None:
        ax.set_title(title)

      if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if (
            i != num_axes - 1
            and despine_keep_bottom
            and (not isinstance(component, TranscriptAnnotation))
        ):
          ax.spines['bottom'].set_visible(True)
        else:
          ax.spines['bottom'].set_visible(False)

      # Remove tick marks which become visible due to a subplots_adjust() call.
      if offset != num_axes - 1:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

      ax.set_xlim(interval.start, interval.end)

      offset += 1

  # Add annotations across all plotted components.
  if annotations is not None:
    # Add small vertical offset for each set of labels to avoid overlap.
    num_labelled_annotations = sum(
        annotation.has_labels for annotation in annotations
    )
    height_offsets = np.linspace(
        annotation_offset_range[0],
        annotation_offset_range[1],
        num_labelled_annotations,
    )
    # Identify which axes are transcripts and which we want to annotate.
    transcript_axes_idx = [
        i
        for i, comp in enumerate(components)
        if isinstance(comp, TranscriptAnnotation)
    ]
    axes_to_annotate_idx = (
        range(1, len(axes)) if add_label_axis else range(len(axes))
    )
    label_index = 0
    for annotation in annotations:
      for annotate_index in axes_to_annotate_idx:
        if not annotation.is_variant and annotate_index in transcript_axes_idx:
          # Do not add interval annotations to axes that involves transcripts.
          continue
        annotation.plot_ax(axes[annotate_index], interval, hspace)
      if annotation.has_labels:
        # Add labels to the empty axis at the top. All labels are added to
        # the top axis, in the order they are supplied in annotations list.
        annotation.plot_labels(axes[0], interval, height_offsets[label_index])
        label_index += 1

  # Enable default tick locator for the final subplot.
  axes[-1].xaxis.set_major_locator(matplotlib.ticker.AutoLocator())

  if xlabel is not None:
    axes[-1].set_xlabel(xlabel)
  else:
    axes[-1].set_xlabel(f'Chromosome position; interval={interval}')

  # Slight whitespace between subplots to avoid tick label overlap.
  fig.subplots_adjust(hspace=hspace)

  return fig


class AbstractComponent(abc.ABC):
  """Abstract base class for plot components."""

  @abc.abstractmethod
  def get_ax_height(self, axis_index: int) -> float:
    """Returns the plot height for the individual axis.

    Args:
      axis_index: The index of the axis.

    Returns:
      The height of the axis.
    """

  @property
  def total_height(self) -> float:
    """Returns the total figure height."""
    return sum(self.get_ax_height(i) for i in range(self.num_axes))

  @property
  @abc.abstractmethod
  def num_axes(self) -> int:
    """Returns the number of matplotlib axes required by the component."""

  @abc.abstractmethod
  def plot_ax(
      self,
      ax: matplotlib.axes.Axes,
      axis_index: int,
      interval: genome.Interval,
  ):
    """Plots the component on the given axis.

    Args:
      ax: The matplotlib axis to plot on.
      axis_index: The index of the axis.
      interval: The genomic interval to plot.
    """


class Tracks(AbstractComponent):
  """Component for visualizing tracks."""

  def __init__(
      self,
      tdata: track_data.TrackData,
      cmap: str = 'viridis',
      truncate_cmap: bool = True,
      track_height: float = 1.0,
      filled: bool = False,
      ylabel_template: str = '{name}:{strand}',
      ylabel_horizontal: bool = True,
      shared_y_scale: bool = False,
      global_ylims: tuple[float, float] | None = None,
      max_num_tracks: int = 50,
      track_colors: Sequence[_ColorType] | str | None = None,
      **kwargs,
  ):
    """Initializes the `Tracks` component.

    Args:
      tdata: The `TrackData` object to visualize.
      cmap: The colormap to use for the tracks.
      truncate_cmap: Whether to slightly truncate the colormap to avoid extreme
        values.
      track_height: The height of each track.
      filled: Whether to fill the area under the tracks.
      ylabel_template: A template for the y-axis labels.
      ylabel_horizontal: Whether to make the y-axis labels horizontal.
      shared_y_scale: Whether to use the same y-axis scale for all tracks.
      global_ylims: Optional global y-axis limits (min and max).
      max_num_tracks: The maximum number of tracks to plot.
      track_colors: An optional sequence of colors to use for the tracks. If a
        string is passed, it is used as a single color for all tracks.
      **kwargs: Additional keyword arguments to pass to the plotting function.

    Raises:
      ValueError: If the number of tracks exceeds `max_num_tracks` or if the
        track data has more than one positional axis or if the interval is not
        set within the track data.
    """
    if tdata.num_tracks > max_num_tracks:
      raise ValueError(
          f'Too many tracks to plot: {tdata.num_tracks} > {max_num_tracks}.'
      )
    self._tdata = tdata
    self._num_tracks = tdata.values.shape[-1]
    self._track_height = track_height
    self._filled = filled
    self._ylabel_horizontal = ylabel_horizontal
    self._ylabel_template = ylabel_template
    self._shared_y_scale = shared_y_scale
    self._global_ylims = global_ylims
    self._kwargs = kwargs

    if len(self._tdata.positional_axes) != 1:
      raise ValueError(
          'Only track_data with 1 positional axis is supported in Tracks.'
      )
    if self._tdata.interval is None:
      raise ValueError('.interval needs to be set in track_data.')

    # Set up color per set of interleaved tracks.
    if getattr(self._tdata, 'uns') is not None:
      self._num_tdata = self._tdata.uns['num_interleaved_trackdatas']
    else:
      self._num_tdata = 1

    cmap = plt.get_cmap(cmap)
    num_track_sets = self._num_tracks // self._num_tdata
    if truncate_cmap:
      # We do *1.2 to make the color change more gradual. This means that the
      # the upper range of the cmap is never displayed, which often tends to
      # achieve nicer results aesthetically.
      self._colors = cmap(np.linspace(0, 1, round(num_track_sets * 1.2)))
    else:
      self._colors = cmap(np.linspace(0, 1, num_track_sets))

    if track_colors is not None:
      if isinstance(track_colors, str):
        self._colors = [track_colors] * num_track_sets
      elif len(track_colors) == 1:
        self._colors = track_colors * num_track_sets
      elif len(track_colors) != num_track_sets:
        raise ValueError(
            f'track_colors argument (length: {len(track_colors)}) must be'
            ' either a single color, or the same number of track sets provided'
            f' in the tdata ({num_track_sets}).'
        )
      else:
        self._colors = track_colors

  def _get_ylimits(self, tdata: track_data.TrackData):
    """Computes y-axis limits for track sets."""
    return [
        (
            tdata.values[:, i : i + self._num_tdata].min(),
            tdata.values[:, i : i + self._num_tdata].max(),
        )
        for i in range(0, self._num_tracks, self._num_tdata)
    ]

  def get_ax_height(self, axis_index: int) -> float:
    """Returns the height of the axis."""
    return self._track_height

  @property
  def num_axes(self) -> int:
    """Returns the number of matplotlib axes required by the component."""
    return self._tdata.num_tracks

  def plot_ax(
      self,
      ax: matplotlib.axes.Axes,
      axis_index: int,
      interval: genome.Interval,
  ):
    """Plots the tracks on the given axis.

    Args:
      ax: The matplotlib axis to plot on.
      axis_index: The index of the axis.
      interval: The genomic interval to plot.
    """
    tdata = self._tdata
    assert tdata.interval is not None

    # If an interval is passed, zoom in on that specific sub-interval.
    if interval is not None:
      tdata = tdata.slice_by_interval(interval, match_resolution=True)
      del interval
    x = (
        np.arange(tdata.values.shape[0]) * tdata.resolution
        + tdata.interval.start
        + tdata.resolution / 2
    )
    arr = tdata.values[:, axis_index]

    # Same shared y-axis limits across all tracks.
    if self._shared_y_scale:
      if self._global_ylims is not None:
        ax.set_ylim(self._global_ylims)
      else:
        ax.set_ylim(tdata.values.min(), tdata.values.max())
    else:
      ax.set_ylim(arr.min(), arr.max())

    # Set the colour of this track.
    track_color = self._colors[axis_index // self._num_tdata]

    # Draw the line-plot, or filled line-plot if filled=True.
    if self._filled:
      ax.fill_between(x, np.ravel(arr), color=track_color, **self._kwargs)
    else:
      ax.plot(x, arr, c=track_color, **self._kwargs)

    if self._ylabel_template:
      _set_ylabel(ax, self._get_ylabel(axis_index), self._ylabel_horizontal)

  def _get_ylabel(self, axis_index: int) -> str:
    """Returns the y-axis label for the given axis index."""
    row = self._tdata.metadata.iloc[axis_index]
    return self._ylabel_template.format(**row.to_dict())


class OverlaidTracks(AbstractComponent):
  """Component for visualizing overlaid track pairs, such as REF/ALT tracks."""

  def __init__(
      self,
      tdata: Mapping[str, track_data.TrackData],
      colors: Mapping[str, str] | None = None,
      cmap: str | None = 'viridis',
      track_height: float = 1.0,
      ylabel_template: str = '{name}:{strand}',
      ylabel_horizontal: bool = True,
      shared_y_scale: bool = False,
      global_ylims: tuple[float, float] | None = None,
      yticks: Sequence[float] | None = None,
      yticklabels: Sequence[str] | None = None,
      alpha: float = 0.8,
      order_tdata_by_mean: bool = True,
      max_num_tracks: int = 50,
      legend_loc: str = 'upper right',
      **kwargs,
  ):
    """Initializes the `OverlaidTracks` component.

    Args:
      tdata: A dictionary mapping track names to `TrackData` objects.
      colors: An optional dictionary mapping track names to colors.
      cmap: The colormap to use if `colors` is not provided.
      track_height: The height of each track.
      ylabel_template: A template for the y-axis labels.
      ylabel_horizontal: Whether to make the y-axis labels horizontal.
      shared_y_scale: Whether to use the same y-axis scale for all tracks. This
        is inferred from the min/max data values across all tracks.
      global_ylims: Optional global y-axis limits (min and max).
      yticks: Optional set y-axis tick values manually. If not provided, the
        tick values will be automatically determined. If provided, the length of
        yticks must match the length of yticklabels.
      yticklabels: Optional set y-axis tick labels manually. If not provided,
        the tick values will be automatically determined. If provided, the
        length of yticklabels must match the length of yticks.
      alpha: The transparency of the tracks.
      order_tdata_by_mean: Whether to order the tracks by their mean value (in
        descending order).
      max_num_tracks: The maximum number of tracks to plot.
      legend_loc: The location of the legend (such as 'upper left' or 'best').
        See the matplotlib Axes legend documentation for more details and
        options. If None, no legend is shown.
      **kwargs: Additional keyword arguments to pass to the plotting function.

    Raises:
      ValueError: If the shapes or metadata of the track data do not match,
                  or if the number of tracks exceeds `max_num_tracks`, or if
                  the track data has more than one positional axis, or if the
                  interval is not set, or if colors are passed but do not match
                  the track data names.
    """
    self._tdata = tdata
    self._colors = colors
    self._cmap = cmap
    self._track_height = track_height
    self._ylabel_template = ylabel_template
    self._ylabel_horizontal = ylabel_horizontal
    self._shared_y_scale = shared_y_scale
    self._global_ylims = global_ylims
    self._yticks = yticks
    self._yticklabels = yticklabels
    self._alpha = alpha
    self._order_tdata_by_mean = order_tdata_by_mean
    self._kwargs = kwargs
    self._first_tdata = list(tdata.values())[0]
    self._legend_loc = legend_loc

    if (
        self._yticks is not None
        and self._yticklabels is not None
        and len(self._yticks) != len(self._yticklabels)
    ):
      raise ValueError(
          'If passing yticks and yticklabels, the length of yticks must match'
          ' the length of yticklabels.'
      )

    if len(set(data.values.shape for data in tdata.values())) != 1:
      raise ValueError('Shapes of track data values must be the same.')

    if not all(
        self._first_tdata.metadata.equals(data.metadata)
        for data in tdata.values()
    ):
      raise ValueError('Metadata of track data must be the same.')

    if self._first_tdata.num_tracks > max_num_tracks:
      raise ValueError(
          f'Too many tracks to plot: {self._first_tdata.num_tracks} >'
          f' {max_num_tracks}.'
      )

    # If a colors dict is passed, then there should be a matching color for each
    # track data name.
    if self._colors is not None:
      if self._tdata.keys() != self._colors.keys():
        raise ValueError(
            f'If passing colors, each tdata name {list(self._tdata.keys())} '
            'must have an associated color.'
        )
    # Otherwise, we define a color from a cmap.
    else:
      cmap = plt.get_cmap(self._cmap)
      colors = cmap(np.linspace(0, 1, len(tdata)))
      self._colors = dict(zip(self._tdata.keys(), colors))

    if len(self._first_tdata.positional_axes) != 1:
      raise ValueError(
          'Only track_data with 1 positional axis is supported in'
          ' OverlaidTracks.'
      )
    if self._first_tdata.interval is None:
      raise ValueError('.interval needs to be set in track_data.')

    if self._shared_y_scale:
      # We compute min and max over all the track data arrays in the tdata dict.
      all_values = np.stack([arr.values for arr in self._tdata.values()])
      self._vmin = np.min(all_values)
      self._vmax = np.max(all_values)

  def get_ax_height(self, axis_index: int) -> float:
    """Returns the height of the axis."""
    return self._track_height

  @property
  def num_axes(self) -> int:
    """Returns the number of matplotlib axes required by the component."""
    return self._first_tdata.num_tracks

  def plot_ax(
      self,
      ax: matplotlib.axes.Axes,
      axis_index: int,
      interval: genome.Interval,
  ):
    """Plots the overlaid tracks on the given axis.

    Args:
      ax: The matplotlib axis to plot on.
      axis_index: The index of the axis.
      interval: The genomic interval to plot.
    """

    def _maybe_slice_tdata(td):
      """Slices the track data to the interval if passed."""
      # If an interval is passed, zoom in on that specific sub-interval.
      if interval is not None:
        return td.slice_by_interval(interval, match_resolution=True)
      else:
        return td

    def _make_ordered_dict_by_mean(tdata):
      """Reorders track data dict by mean (descending) for better plotting."""
      mean_tuples = [
          (name, np.mean(td.values, dtype=np.float64))
          for name, td in tdata.items()
      ]
      sorted_mean_tuples = sorted(
          mean_tuples, key=lambda item: item[1], reverse=True
      )

      # Extract names in the sorted order and return ordered tdata.
      sorted_names = [name for name, _ in sorted_mean_tuples]
      ordered_tdata = {name: tdata[name] for name in sorted_names}
      return ordered_tdata

    # Maybe slice the track data to the interval, and reorder by mean.
    tdata_sliced = {
        name: _maybe_slice_tdata(td) for name, td in self._tdata.items()
    }
    self._tdata_ordered = (
        _make_ordered_dict_by_mean(tdata_sliced)
        if self._order_tdata_by_mean
        else tdata_sliced
    )

    for name, tdata in self._tdata_ordered.items():
      assert tdata.interval is not None
      x = (
          np.arange(tdata.values.shape[0]) * tdata.resolution
          + tdata.interval.start
          + tdata.resolution / 2
      )
      arr = tdata.values[:, axis_index]

      if self._global_ylims is not None:
        ax.set_ylim(self._global_ylims)
      elif self._shared_y_scale:
        ax.set_ylim(self._vmin, self._vmax)

      # Plot the two tracks on the same axis. We plot the larger values first so
      # that the overlap is more visible.
      ax.plot(
          x,
          arr,
          alpha=self._alpha,
          **self._kwargs,
          c=self._colors[name],
      )
      if axis_index == 0 and self._legend_loc is not None:
        ax.legend(self._tdata_ordered.keys(), loc=self._legend_loc)

      if self._yticks is not None:
        ax.set_yticks(self._yticks)
      if self._yticklabels is not None:
        ax.set_yticklabels(self._yticklabels)

    if self._ylabel_template:
      _set_ylabel(ax, self._get_ylabel(axis_index), self._ylabel_horizontal)

  def _get_ylabel(self, axis_index: int) -> str:
    """Returns the y-axis label for the given track."""
    # Metadata equality has already been checked so here we grab the first one.
    metadata = self._tdata[list(self._tdata.keys())[0]].metadata
    row = metadata.iloc[axis_index]
    return self._ylabel_template.format(**row.to_dict())


class ContactMaps(AbstractComponent):
  """Component for visualizing contact maps.

  The `vmin` and `vmax` parameters control the color scaling of the heatmap.
  Values outside this range will be clipped to `vmin` or `vmax`.
  """

  def __init__(
      self,
      tdata: track_data.TrackData,
      track_height: float = 10.0,
      vmin: float | None = -1.0,
      vmax: float | None = 2.0,
      norm: matplotlib.colors.TwoSlopeNorm | None = None,
      ylabel_horizontal: bool = True,
      ylabel_template: str = '{name}',
      cmap: matplotlib.colors.LinearSegmentedColormap | None = None,
      max_num_tracks: int = 10,
      **kwargs,
  ):
    """Initializes the `ContactMaps` component.

    Args:
      tdata: The `TrackData` object containing the contact maps.
      track_height: The height of each contact map.
      vmin: The minimum value for the color scale.
      vmax: The maximum value for the color scale.
      norm: An optional normalization for the color scale.
      ylabel_horizontal: Whether to make the y-axis labels horizontal.
      ylabel_template: A template for the y-axis labels.
      cmap: The colormap to use for the contact maps.
      max_num_tracks: The maximum number of tracks to plot.
      **kwargs: Additional keyword arguments to pass to the plotting function.

    Raises:
      ValueError: If the number of tracks exceeds `max_num_tracks`, or if the
        track data does not have 2 positional axes, or if the contact maps are
        not square, or if the interval is not set in the track data.
    """
    if tdata.num_tracks > max_num_tracks:
      raise ValueError(
          f'Too many tracks to plot: {tdata.num_tracks} > {max_num_tracks}.'
      )
    self._tdata = tdata
    self._resolution = tdata.resolution
    self._track_height = track_height
    self._vmin = vmin
    self._vmax = vmax
    self._norm = norm
    self._ylabel_horizontal = ylabel_horizontal
    self._ylabel_template = ylabel_template
    # TODO: b/377292012 - Add orca_color_map.
    self._cmap = (
        cmap if cmap is not None else matplotlib.pyplot.get_cmap('autumn_r')
    )
    self._kwargs = kwargs
    if len(self._tdata.positional_axes) != 2:
      raise ValueError(
          'Only track_data with 2 positional axes is supported in ContactMaps.'
      )
    if self._tdata.values.shape[0] != self._tdata.values.shape[1]:
      raise ValueError('Contact maps must be square.')

    if self._tdata.interval is None:
      raise ValueError('.interval needs to be set in track_data.')

  def get_ax_height(self, axis_index: int) -> float:
    """Returns the height of the axis."""
    return self._track_height

  @property
  def num_axes(self) -> int:
    """Returns the number of matplotlib axes required by the component."""
    return self._tdata.num_tracks

  def _get_bin_positions(
      self, interval: genome.Interval, resolution: int
  ) -> np.ndarray:
    """Gets the positions of contact map bins in chromosome coordinates."""
    bin_indices = np.arange(interval.width // resolution)
    return interval.start + (resolution * bin_indices)

  def _plot_pcolormesh(
      self,
      ax: matplotlib.axes.Axes,
      x: np.ndarray,
      y: np.ndarray,
      arr: np.ndarray,
      vmin: float | None = None,
      vmax: float | None = None,
      cmap: matplotlib.colors.Colormap | None = None,
      norm: matplotlib.colors.Normalize | None = None,
      **kwargs,
  ) -> matplotlib.collections.QuadMesh:
    """Plots the contact map heatmap using `pcolormesh`.

    Note that upsampling the contact maps to single base pair resolution and
    using something like .imshow() is infeasible since the upsampling blows up
    memory.

    Args:
      ax: The matplotlib axis to plot on.
      x: The x-axis coordinates.
      y: The y-axis coordinates.
      arr: The contact map data.
      vmin: The minimum value for the color scale.
      vmax: The maximum value for the color scale.
      cmap: The colormap to use.
      norm: An optional normalization for the color scale.
      **kwargs: Additional keyword arguments to pass to `pcolormesh`.

    Returns:
      The `matplotlib.collections.QuadMesh` object representing the plot.
    """
    if not norm:  # Cannot pass both norm and vmin/vmax simultaneously.
      return ax.pcolormesh(
          x,
          y,
          arr,
          vmin=vmin,
          vmax=vmax,
          cmap=cmap,
          **kwargs,
      )
    else:
      return ax.pcolormesh(
          x,
          y,
          arr,
          norm=norm,
          **kwargs,
      )

  def plot_ax(
      self,
      ax: matplotlib.axes.Axes,
      axis_index: int,
      interval: genome.Interval,
  ):
    """Plots the contact map on the given axis.

    Args:
      ax: The matplotlib axis to plot on.
      axis_index: The index of the axis.
      interval: The genomic interval to plot.
    """
    tdata = self._tdata
    assert tdata.interval is not None

    # If an interval is passed, zoom in on that specific sub-interval.
    if interval is not None:
      tdata = tdata.slice_by_interval(interval, match_resolution=True)
      del interval

    arr = tdata.values[:, :, axis_index]
    x = self._get_bin_positions(tdata.interval, self._resolution)

    # We shift the bin edges by half a step since pcolormesh will misalign
    # the x-axis by half a step, since the plot values are centered within each
    # bin, rather than at the bin edges.
    half_bin_width = self._resolution // 2
    x = x + half_bin_width

    # The -1 reverse ordering ensures that contact maps are plotted with
    # the diagonal going down from left to right.
    y = np.arange(arr.shape[0])[::-1]

    if not self._vmin:
      self._vmin = np.min(arr)

    if not self._vmax:
      self._vmax = np.max(arr)

    self._plot_pcolormesh(
        ax=ax,
        x=x,
        y=y,
        arr=arr,
        vmin=self._vmin,
        vmax=self._vmax,
        cmap=self._cmap,
        norm=self._norm,
        **self._kwargs,
    )

    if self._ylabel_template:
      _set_ylabel(ax, self._get_ylabel(axis_index), self._ylabel_horizontal)

  def _get_ylabel(self, axis_index: int) -> str:
    """Returns the y-axis label for the given contact map."""
    row = self._tdata.metadata.iloc[axis_index]
    return self._ylabel_template.format(**row.to_dict())


class ContactMapsDiff(ContactMaps):
  """Component for visualizing contact map differences.

  This component visualizes the difference between two contact maps. It uses
  a diverging red-blue color map with the center white color pinned to a
  value of zero, with negative values being blue and positive values being red.

  The `vmin` and `vmax` parameters control the color scaling of the heatmap.
  Values outside this range will be clipped to `vmin` or `vmax`.
  """

  def __init__(
      self,
      tdata: track_data.TrackData,
      track_height: float = 10.0,
      vmin: float | None = -1.0,
      vmax: float | None = 1.0,
      ylabel_horizontal: bool = True,
      ylabel_template: str = '{name}',
      cmap: matplotlib.colors.LinearSegmentedColormap | str | None = 'RdBu_r',
      max_num_tracks: int = 10,
      **kwargs,
  ):
    """Initializes the `ContactMapsDiff` component.

    Args:
      tdata: The `TrackData` object containing the contact map differences.
      track_height: The height of each contact map.
      vmin: The minimum value for the color scale.
      vmax: The maximum value for the color scale.
      ylabel_horizontal: Whether to make the y-axis labels horizontal.
      ylabel_template: A template for the y-axis labels.
      cmap: The colormap to use for the contact maps.
      max_num_tracks: The maximum number of tracks to plot.
      **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    self._norm = plt_colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    super().__init__(
        tdata=tdata,
        track_height=track_height,
        vmin=vmin,
        vmax=vmax,
        ylabel_horizontal=ylabel_horizontal,
        ylabel_template=ylabel_template,
        cmap=cmap,
        max_num_tracks=max_num_tracks,
        **kwargs,
    )


def _set_ylabel(ax: matplotlib.axes.Axes, ylabel: str, horizontal: bool):
  """Sets the y-axis label.

  Args:
    ax: The matplotlib axis to set the label on.
    ylabel: The label text.
    horizontal: Whether to make the label horizontal.
  """
  if ylabel:
    if horizontal:
      ax.set_ylabel(
          ylabel,
          rotation=0,
          multialignment='center',
          va='center',
          ha='right',
          labelpad=5,
      )
    else:
      ax.set_ylabel(ylabel)


class TranscriptAnnotation(AbstractComponent):
  """Visualizes transcript annotations."""

  def __init__(
      self,
      transcripts: Sequence[transcript_utils.Transcript],
      adaptive_fig_height: bool = True,
      fig_height: float = 1.0,
      transcript_style: plot_transcripts.TranscriptStyle = (
          plot_transcripts.TranscriptStylePreset.MINIMAL.value
      ),
      plot_labels_once: bool = True,
      label_name: str = 'gene_name',
      **kwargs,
  ):
    """Initializes the `TranscriptAnnotation` component.

    Args:
      transcripts: A sequence of `Transcript` objects to visualize.
      adaptive_fig_height: Whether to adjust the figure height based on the
        number of transcripts.
      fig_height: The base figure height.
      transcript_style: The style to use for plotting transcripts. The options
        are defined in `plot_transcripts.TranscriptStylePreset`.
      plot_labels_once: Whether to plot labels only once per transcript.
      label_name: The attribute of the transcript to use for labels.
      **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    self._transcripts = transcripts
    self._adaptive_fig_height = adaptive_fig_height
    self._fig_height = fig_height
    self._kwargs = kwargs
    self._kwargs['label_name'] = label_name
    self._kwargs['transcript_style'] = transcript_style
    self._kwargs['plot_labels_once'] = plot_labels_once

    self._num_transcripts = len(self._transcripts)
    # TODO(b/377291518): adaptive fig height should in theory scale with the
    # number of transcripts in the sub-interval, not the total number of
    # transcripts passed, but this is a bit tricky to implement in the code and
    # this approach seems to work well enough for now.
    if self._adaptive_fig_height:
      self._fig_height = max(0.05 * self._num_transcripts * self._fig_height, 1)

  def get_ax_height(self, axis_index: int) -> float:
    """Returns the height of the axis."""
    return self._fig_height

  @property
  def num_axes(self) -> int:
    """Returns the number of matplotlib axes required by the component."""
    return 1

  def plot_ax(
      self, ax: matplotlib.axes.Axes, axis_index: int, interval: genome.Interval
  ):
    """Plots the transcript annotations on the given axis.

    Args:
      ax: The matplotlib axis to plot on.
      axis_index: The index of the axis.
      interval: The genomic interval to plot.
    """
    # Update transcripts to only those overlapping interval.
    transcripts = [
        t for t in self._transcripts if t.transcript_interval.overlaps(interval)
    ]
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    plot_transcripts.plot_transcripts(ax, transcripts, interval, **self._kwargs)


class SeqLogo(AbstractComponent):
  """Visualizes a sequence logo."""

  def __init__(
      self,
      scores: Float32[np.ndarray, 'S A'],
      scores_interval: genome.Interval,
      fig_height: float = 1.0,
      alphabet: str = 'ACGT',
      max_width: int = 1000,
      ylabel: str = '',
      ylabel_horizontal: bool = True,
      ylim: tuple[float, float] | None = None,
      **kwargs,
  ):
    """Initializes the `SeqLogo` component.

    Args:
      scores: A numpy array of shape (sequence_length, alphabet_size) containing
        the sequence logo scores.
      scores_interval: The genomic interval corresponding to the scores.
      fig_height: The height of the figure.
      alphabet: The alphabet used in the sequence logo.
      max_width: The maximum width of the sequence logo to plot.
      ylabel: An optional label for the y-axis.
      ylabel_horizontal: Whether to make the y-axis label horizontal.
      ylim: An optional range to set for the y-axis.
      **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    self._scores = scores
    self._scores_interval = scores_interval
    self._fig_height = fig_height
    self._alphabet = alphabet
    self._max_width = max_width
    self._ylabel = ylabel
    self._ylabel_horizontal = ylabel_horizontal
    self._ylim = ylim
    self._kwargs = kwargs

  def get_ax_height(self, axis_index: int) -> float:
    """Returns the height of the axis."""
    return self._fig_height

  @property
  def num_axes(self) -> int:
    """Returns the number of matplotlib axes required by the component."""
    return 1

  def plot_ax(
      self, ax: matplotlib.axes.Axes, axis_index: int, interval: genome.Interval
  ):
    """Plots the sequence logo on the given axis.

    Args:
      ax: The matplotlib axis to plot on.
      axis_index: The index of the axis.
      interval: The genomic interval to plot.
    """
    intersection = self._scores_interval.intersect(interval)
    if intersection is None or intersection.width > self._max_width:
      return
    relative_start = intersection.start - self._scores_interval.start
    scores = self._scores[
        relative_start : (relative_start + intersection.width)
    ]

    plot_lib.seqlogo(
        scores,
        ax=ax,
        alphabet=self._alphabet,
        start=intersection.start,
        one_based=False,
        **self._kwargs,
    )

    _set_ylabel(ax, self._ylabel, self._ylabel_horizontal)
    if self._ylim is not None:
      ax.set_ylim(self._ylim)


class Sashimi(AbstractComponent):
  """Visualizes splice junctions as a Sashimi plot."""

  def __init__(
      self,
      junction_track: junction_data.JunctionData,
      fig_height: float = 1.0,
      filter_threshold: float | None = None,
      ylabel_template: str = '{name}',
      ylabel_horizontal: bool = True,
      annotate_counts: bool = True,
      normalize_values: bool = True,
      interval_contained: bool = True,
      rng: np.random.Generator | None = None,
  ):
    """Initializes the `Sashimi` component.

    Args:
      junction_track: A `JunctionData` object to visualize.
      fig_height: The height of the figure.
      filter_threshold: The minimum value for a junction to be included in the
        plot. This is typically based on the normalized read count. If None,
        filter out junction values below 5% of the maximum value.
      ylabel_template: A template for the y-axis labels.
      ylabel_horizontal: Whether to make the y-axis label horizontal.
      annotate_counts: Whether to annotate the junctions with read counts.
      normalize_values: Whether to normalize the values to a constant sum.
      interval_contained: Whether to only plot junctions contained in the
        interval.
      rng: Optional random number generator to use for jittering junction paths.
        If unset will use NumPy's default random number generator.
    """
    if normalize_values:
      self._junction_track = junction_track.normalize_values()
    else:
      self._junction_track = junction_track
    self._fig_height = fig_height
    self._filter_threshold = filter_threshold
    self._ylabel_template = ylabel_template
    self._ylabel_horizontal = ylabel_horizontal
    self._annotate_counts = annotate_counts
    self._interval_contained = interval_contained
    self._rng = rng or np.random.default_rng()

  def get_ax_height(self, axis_index: int) -> float:
    """Returns the height of the axis."""
    return self._fig_height

  @property
  def num_axes(self) -> int:
    """Returns the number of matplotlib axes required by the component."""
    # Metadata for JunctionData do not have strand information.
    # Strand information is in JunctionData.intervals.
    return self._junction_track.num_tracks * len(
        self._junction_track.possible_strands
    )

  def _get_strand_and_metadata_index(self, axis_index: int) -> tuple[str, int]:
    """Returns the strand and metadata index for the given axis index."""
    if len(self._junction_track.possible_strands) == 1:
      strand = self._junction_track.possible_strands[0]
      metadata_index = axis_index
    else:
      strand = '+' if axis_index % 2 == 0 else '-'
      metadata_index = axis_index // 2
    return strand, metadata_index

  def plot_ax(
      self, ax: matplotlib.axes.Axes, axis_index: int, interval: genome.Interval
  ):
    """Plots the Sashimi plot on the given axis.

    Args:
      ax: The matplotlib axis to plot on.
      axis_index: The index of the axis.
      interval: The genomic interval to plot.
    """
    strand, metadata_index = self._get_strand_and_metadata_index(axis_index)
    track_name = self._junction_track.metadata.iloc[metadata_index]['name']
    junction_track = self._junction_track.intersect_with_interval(interval)
    junctions = junction_data.get_junctions_to_plot(
        predictions=junction_track,
        strand=strand,
        name=track_name,
        k_threshold=self._filter_threshold,
    )
    if self._interval_contained:
      junctions = [j for j in junctions if interval.contains(j)]
    else:
      junctions = [j for j in junctions if j.overlaps(interval)]

    plot_lib.sashimi_plot(
        junctions,
        ax=ax,
        interval=interval,
        filter_threshold=0,
        annotate_counts=self._annotate_counts,
        rng=self._rng,
    )
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    if self._ylabel_template:
      _set_ylabel(ax, self._get_ylabel(axis_index), self._ylabel_horizontal)

  def _get_ylabel(self, axis_index: int) -> str:
    """Returns the y-axis label for the given axis index."""
    strand, metadata_index = self._get_strand_and_metadata_index(axis_index)
    row = self._junction_track.metadata.iloc[metadata_index]
    row = row.to_dict()
    row['strand'] = strand
    return self._ylabel_template.format(**row)


class EmptyComponent(AbstractComponent):
  """An empty plotting component."""

  def __init__(self, fig_height: float = 1.0):
    """Initializes the `EmptyComponent`.

    Args:
      fig_height: The height of the figure.
    """
    self._fig_height = fig_height

  def get_ax_height(self, axis_index: int) -> float:
    """Returns the height of the axis."""
    return self._fig_height

  @property
  def num_axes(self) -> int:
    """Returns the number of matplotlib axes required by the component."""
    return 1

  def plot_ax(
      self, ax: matplotlib.axes.Axes, axis_index: int, interval: genome.Interval
  ):
    """Plot an empty axis, removing all labels and spines from the axis.

    Args:
      ax: The matplotlib axis to plot on.
      axis_index: The index of the axis.
      interval: The genomic interval to plot.
    """
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)


class AbstractAnnotation(abc.ABC):
  """Abstract base class for plot annotations.

  Annotations are visual elements that can be added to plots to highlight
  specific features or regions. This class defines the common interface
  for all annotations.

  Attributes:
    annotations: A sequence of `Variant` or `Interval` objects representing the
      annotations.
    colors: An optional string or sequence of strings specifying the colors of
      the annotations.
    labels: An optional sequence of strings to use as labels for the
      annotations.
    use_default_labels: Whether to use default labels for the annotations if
      `labels` is not provided.
  """

  def __init__(
      self,
      annotations: Sequence[genome.Variant] | Sequence[genome.Interval],
      colors: str | Sequence[str] | None,
      labels: Sequence[str] | None,
      use_default_labels: bool,
  ):
    """Initializes the `AbstractAnnotation` class.

    Args:
      annotations: A sequence of `Variant` or `Interval` objects.
      colors: An optional string or sequence of strings specifying colors.
      labels: An optional sequence of strings to use as labels.
      use_default_labels: Whether to use default labels if `labels` is not
        provided.

    Raises:
      ValueError: If the length of `colors` or `labels` does not match the
        length of `annotations`.
    """
    self._annotations = annotations
    self._colors = colors
    self._labels = labels
    self._use_default_labels = use_default_labels

    # Pre-processing / validation of inputs.
    num_annotations = len(self._annotations)
    if (self._colors is not None) and (not isinstance(self._colors, str)):
      if len(self._colors) != num_annotations:
        raise ValueError(
            'Colors must have the same length as intervals/variants or just'
            ' a single color string.'
        )
    if self._labels is not None:
      if len(self._labels) != num_annotations:
        raise ValueError(
            'Labels must have the same length as intervals/variants.'
        )

  @abc.abstractmethod
  def plot_ax(
      self, ax: matplotlib.axes.Axes, interval: genome.Interval, hspace: float
  ):
    """Adds the annotation to an individual axis.

    Args:
      ax: The matplotlib axis to add the annotation to.
      interval: The genomic interval to plot.
      hspace: The vertical space between subplots.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def plot_labels(
      self,
      ax: matplotlib.axes.Axes,
      interval: genome.Interval,
      label_height_factor: float,
  ):
    """Adds labels for the annotation to an axis.

    Args:
      ax: The matplotlib axis to add the labels to.
      interval: The genomic interval to plot.
      label_height_factor: A scaling factor for the label height.
    """
    raise NotImplementedError

  @property
  def is_variant(self) -> bool:
    """Returns True if the annotation is a variant annotation."""
    return isinstance(self._annotations[0], genome.Variant)

  @property
  def has_labels(self) -> bool:
    """Returns True if the annotation has labels."""
    return (self._labels is not None) | (
        self.is_variant and self._use_default_labels
    )

  def add_label(
      self,
      ax: matplotlib.axes.Axes,
      label_x_position: float,
      label: str,
      angle: float,
      label_height_factor: float,
      label_position: str = 'left',
  ):
    """Adds a single angled label to an axis.

    Args:
      ax: The matplotlib axis to add the label to.
      label_x_position: The x position of the label.
      label: The label text.
      angle: The angle of the label.
      label_height_factor: A scaling factor for the label height.
      label_position: The (horizontal) placement of the label, relative to the x
        position. Can be any position string accepted by the horizontalalignment
        argument of matplotlib.axes.Axes.text.
    """
    ylims = ax.get_ylim()
    label_height = ylims[0] + label_height_factor * np.diff(ylims)[0]
    ax.text(
        label_x_position,
        label_height,
        label,
        color='black',
        fontsize=10,
        rotation=angle,
        ha=label_position,
        va='bottom',
    )
    ax.axvline(
        label_x_position, ymax=label_height * 0.95, color='black', alpha=0.1
    )


class IntervalAnnotation(AbstractAnnotation):
  """Visualizes intervals as rectangles across all plot components.

  A rectangle is drawn for each interval and overlaid on top of the final plot,
  spanning all plot components.
  """

  def __init__(
      self,
      intervals: Sequence[genome.Interval],
      colors: str | Sequence[str] = 'darkgray',
      alpha: float = 0.2,
      labels: Sequence[str] | None = None,
      use_default_labels: bool = True,
      label_angle: float = 15,
  ):
    """Initializes the `IntervalAnnotation` class.

    Args:
      intervals: A sequence of `Interval` objects to annotate.
      colors: An optional string or sequence of strings specifying the colors of
        the interval annotation.
      alpha: The transparency of the interval annotation.
      labels: An optional sequence of strings to use as labels for the
        intervals.
      use_default_labels: Whether to use default labels for the intervals if
        `labels` is not provided.
      label_angle: The angle of the interval labels.
    """
    super().__init__(intervals, colors, labels, use_default_labels)
    self._label_angle = label_angle
    self._alpha = alpha
    self._intervals = intervals

  def plot_ax(
      self,
      ax: matplotlib.axes.Axes,
      interval: genome.Interval,
      hspace: float = 0.0,
  ):
    """Adds the interval annotation to an individual axis.

    Args:
      ax: The matplotlib axis to add the annotation to.
      interval: The genomic interval to plot.
      hspace: The vertical space between subplots.
    """
    for i, interval_i in enumerate(self._intervals):
      if isinstance(self._colors, str):
        color = self._colors
      else:
        color = self._colors[i]
      # Only plot the piece of the annotation that intersects with the plotting
      # interval.
      intersection = interval_i.intersect(interval)
      if intersection is None:
        continue
      ax.axvspan(
          interval_i.start,
          interval_i.end,
          # Set y-maximum to be the top of the axis, including hspace.
          ymax=(1 + hspace) * 1.03,
          alpha=self._alpha,
          facecolor=color,
          # Set edgecolor to None for intervals to avoid horizongal lines
          # between axes, but keep it for variants. As variants are typically
          # a very thin rectangle, removing the edges results in the rectangle
          # not being visible.
          edgecolor=None,
          clip_on=False,
      )

  def plot_labels(
      self,
      ax: matplotlib.axes.Axes,
      interval: genome.Interval,
      label_height_factor: float,
  ):
    """Adds interval labels to an axis.

    Args:
      ax: The matplotlib axis to add the labels to.
      interval: The genomic interval to plot.
      label_height_factor: A scaling factor for the label height.
    """
    # Only add labels if they are provided.
    if self.has_labels:
      for i, interval_i in enumerate(self._intervals):
        label = self._labels[i]
        # Place label in the middle of the rectangle.
        # Only plot the piece of the annotation that intersects with the
        # plotting interval.
        intersection = interval_i.intersect(interval)
        if intersection is None:
          continue

        self.add_label(
            ax,
            label_x_position=np.mean((interval_i.start, interval_i.end)),
            label=label,
            angle=self._label_angle,
            label_height_factor=label_height_factor,
        )


class VariantAnnotation(AbstractAnnotation):
  """Visualizes variants as thin line-like rectangles across plot components."""

  def __init__(
      self,
      variants: Sequence[genome.Variant],
      colors: str | Sequence[str] = 'orange',
      alpha: float = 0.8,
      labels: Sequence[str] | None = None,
      use_default_labels: bool = True,
      label_angle: float = 15,
      label_position: str = 'left',
  ):
    """Initializes the `VariantAnnotation` class.

    Args:
      variants: A sequence of `Variant` objects to annotate.
      colors: An optional string or sequence of strings specifying the colors of
        the variant annotation.
      alpha: The transparency of the variant annotation.
      labels: An optional sequence of strings to use as labels for the variants.
      use_default_labels: Whether to use default labels for the variants if
        `labels` is not provided.
      label_angle: The angle of the variant labels.
      label_position: The (horizontal) placement of the variant label, relative
        to the variant position. Can be any position string accepted by the
        horizontalalignment argument of matplotlib.axes.Axes.text.
    """
    super().__init__(variants, colors, labels, use_default_labels)
    self._label_angle = label_angle
    self._alpha = alpha
    self._variants = variants
    self._label_position = label_position

  def plot_ax(
      self,
      ax: matplotlib.axes.Axes,
      interval: genome.Interval,
      hspace: float = 0.0,
  ):
    """Adds a variant annotation to an individual axis.

    Args:
      ax: The matplotlib axis to add the annotation to.
      interval: The genomic interval to plot.
      hspace: The vertical space between subplots.
    """
    for i, variant in enumerate(self._variants):
      if isinstance(self._colors, str):
        color = self._colors
      else:
        color = self._colors[i]
      interval_i = variant.reference_interval
      # Only plot the piece of the annotation that intersects with the plotting
      # interval.
      intersection = interval_i.intersect(interval)
      if intersection is None:
        continue
      ax.axvspan(
          interval_i.start,
          interval_i.end,
          # Set y-maximum to be the top of the axis, including hspace.
          ymax=(1 + hspace) * 1.03,
          alpha=self._alpha,
          facecolor=color,
          # Set edgecolor to None for intervals to avoid horizongal lines
          # between axes, but keep it for variants. As variants are typically
          # a very thin rectanle, removing the edges results in the rectangle
          # not being visible.
          edgecolor=color,
          clip_on=False,
      )

  def plot_labels(
      self,
      ax: matplotlib.axes.Axes,
      interval: genome.Interval,
      label_height_factor: float,
  ):
    """Adds variant labels to an axis.

    Args:
      ax: The matplotlib axis to add the labels to.
      interval: The genomic interval to plot.
      label_height_factor: A scaling factor for the label height.
    """
    # Only add labels if they are provided.
    if self.has_labels:
      for i, variant in enumerate(self._variants):
        interval_i = variant.reference_interval
        # Using truncated string method for genome.Variant class to get default
        # labels for variants.
        label = (
            variant.as_truncated_str(max_length=20)
            if self._use_default_labels
            else self._labels[i]
        )
        # Place label in the middle of the rectangle.
        # Only plot the piece of the annotation that intersects with the
        # plotting interval.
        intersection = interval_i.intersect(interval)
        if intersection is None:
          continue

        self.add_label(
            ax,
            label_x_position=np.mean((interval_i.start, interval_i.end)),
            label=label,
            angle=self._label_angle,
            label_height_factor=label_height_factor,
            label_position=self._label_position,
        )
