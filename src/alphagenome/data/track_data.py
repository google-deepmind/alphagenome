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


"""Track data container analogous to AnnData."""

from collections.abc import Sequence
import copy
import dataclasses
import enum
from typing import Any, Union

from alphagenome import typing
from alphagenome.data import genome
from alphagenome.data import ontology
from jaxtyping import Bool, Float32, Int32  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd

# Required columns: name, strand.
# Optional standardized columns: cell_type, assay, padding.
TrackMetadata = pd.DataFrame
PositionalIndex = slice | genome.Interval | int
TrackIndex = np.ndarray | Sequence[int] | Sequence[str] | slice | int | str
Index = PositionalIndex | tuple[PositionalIndex, TrackIndex] | TrackIndex


@enum.unique
class AggregationType(enum.Enum):
  """Aggregation types for downsampling/upsampling track resolutions.

  SUM: Sum pooling, where values within a bin are summed. This is recommended
    for continuous tracks where the total value within a bin is meaningful (e.g.
    read counts, coverage).
  MAX: Max pooling, where the maximum value within a bin is selected. This is
    recommended for binary tracks (e.g., gene masks, regions of interest).
  """

  SUM = 'sum'
  MAX = 'max'


@typing.jaxtyped
@dataclasses.dataclass(frozen=True)
class TrackData:
  """Container for storing track values and metadata.

  `TrackData` stores multiple genomic tracks at the same resolution, stacked
  into an ND matrix of shape (positional_bins, num_tracks). It also contains
  metadata information as a pandas DataFrame with `num_tracks` rows.

  Metadata DataFrame has two main required columns:

    * name: The name of the track.
    * strand: The strand of the track ('+', '-', or '.').

  Other columns are optional.

  Valid shapes of `TrackData.values` are:

    * [num_tracks]
    * [positional_bins, num_tracks]
    * [positional_bins, positional_bins, num_tracks]
    * ...

  `TrackData` can store both model predictions and raw data. It can
  optionally hold information about the `genome.Interval` from which the data
  were derived and `.uns` for storing additional unstructured data.

  In addition to being a container, `TrackData` provides functionality for
  common aggregation and slicing operations.

  Attributes:
    values: A numpy array of floats or integers representing the track values.
      Positional axes have the same length. Example valid shapes are:
      [num_tracks], [positional_bins, num_tracks], and [positional_bins,
      positional_bins, num_tracks].
    metadata: A pandas DataFrame containing metadata for each track. The
      DataFrame must have at least two columns: 'name' and 'strand'.
    resolution: The resolution of the track data in base pairs.
    interval: An optional `Interval` object representing the genomic region.
    uns: An optional dictionary to store additional unstructured data.

  Raises:
    ValueError: If the number of tracks in `values` does not match the  number
      of rows in `metadata`, or if `metadata` contains duplicate (name, strand)
      pairs, or if the positional axes have different lengths, or if the
      interval width does not match the expected width.
  """

  # Use Union due to https://github.com/patrick-kidger/jaxtyping/issues/73.
  values: Union[
      Float32[np.ndarray, '*positional_bins num_tracks'],
      Int32[np.ndarray, '*positional_bins num_tracks'],
      Bool[np.ndarray, '*positional_bins num_tracks'],
  ]
  metadata: TrackMetadata
  resolution: int = 1
  interval: genome.Interval | None = None
  uns: dict[str, Any] | None = (
      # Unstructured data dict, analagous to anndata.AnnData.uns.
      None
  )

  def __post_init__(self):
    """Validates the consistency of the data."""
    if self.values.shape[-1] != len(self.metadata):
      raise ValueError(
          f'value number of tracks {self.values.shape[-1]} and '
          f'metadata {len(self.metadata)} do not match.'
      )

    if self.positional_axes:
      if len(set(np.array(self.values.shape)[self.positional_axes])) != 1:
        raise ValueError('All positional axes must have the same length.')

      if self.interval and self.interval.width != self.width:
        raise ValueError(
            f'Interval width must match expected width. {self.interval.width=},'
            f' {self.width=}'
        )

    if not {'name', 'strand'}.issubset(self.metadata.columns):
      raise ValueError('Metadata must contain columns "name" and "strand".')

    if self.metadata[['name', 'strand']].duplicated().any():
      raise ValueError(
          'Metadata contain duplicated values for (name, strand) tuples.'
      )

  @property
  def positional_axes(self) -> list[int]:
    """Returns a list of the positional axes."""
    return list(range(self.values.ndim - 1))

  @property
  def num_tracks(self) -> int:
    """Returns the number of tracks."""
    return self.values.shape[-1]

  @property
  def width(self) -> int:
    """Returns the interval width covered by the tracks."""
    if self.positional_axes:
      return self.values.shape[0] * self.resolution
    else:
      return 0

  @property
  def names(self) -> np.ndarray:
    """Returns an array of track names (not necessarily unique)."""
    return self.metadata['name'].values

  @property
  def strands(self) -> np.ndarray:
    """Returns an array of track strands."""
    return self.metadata['strand'].values

  @property
  def ontology_terms(self) -> Sequence[ontology.OntologyTerm | None] | None:
    """Returns a list of ontology terms (if available)."""
    if 'ontology_curie' in self.metadata.columns:
      return [
          ontology.from_curie(curie) if pd.notna(curie) else None
          for curie in self.metadata['ontology_curie'].values
      ]
    else:
      return None

  def copy(self) -> 'TrackData':
    """Returns a deep copy of the `TrackData` object."""
    if self.interval:
      interval = self.interval.copy()
    else:
      interval = None
    return TrackData(
        self.values.copy(),
        resolution=self.resolution,
        metadata=self.metadata.copy(),
        interval=interval,
        uns=copy.deepcopy(self.uns),
    )

  def with_name_suffix(self, suffix: str) -> 'TrackData':
    """Returns a new `TrackData` object with suffixed track names."""
    new_metadata = self.metadata.copy()
    new_metadata['name'] = new_metadata['name'] + suffix
    return dataclasses.replace(self, metadata=new_metadata)

  def with_metadata_column(
      self,
      name: str,
      value: str | pd.Series,
  ) -> 'TrackData':
    """Returns a new `TrackData` with an added/updated metadata column."""
    new_metadata = self.metadata.copy()
    new_metadata[name] = value
    return dataclasses.replace(self, metadata=new_metadata)

  def bin_index(self, relative_position: int) -> int:
    """Returns the bin index for a relative position.

    Args:
      relative_position: The relative position within the interval.

    Returns:
      The corresponding bin index.
    """
    return relative_position // self.resolution

  def slice_by_positions(self, start: int, end: int) -> 'TrackData':
    """Slices the track data along the positional axes.

    The slicing follows Python slicing conventions (0 indexed, and includes
    elements up to end-1).

    Args:
      start: The 1-bp resolution start position for slicing.
      end: The 1-bp resolution end position for slicing.

    Returns:
      A new `TrackData` object with the sliced values.

    Raises:
      ValueError: If (end - start) is greater than the width, or if (end -
      start) is not divisible by the resolution.
    """
    if (end - start) > self.width:
      raise ValueError(
          'When slicing track data, (end - start) must be less than or '
          'equal to width.'
      )

    if start % self.resolution != 0 or end % self.resolution != 0:
      raise ValueError(
          f'start and end need to be divisible by {self.resolution=}'
      )

    sl = slice(self.bin_index(start), self.bin_index(end))
    slice_list = [slice(None)] * self.values.ndim
    for i in self.positional_axes:
      slice_list[i] = sl

    interval = self.interval
    if interval:
      interval = genome.Interval(
          interval.chromosome,
          interval.start + start,
          interval.start + end,
          strand=interval.strand,
          name=interval.name,
          info=interval.info,
      )

    return TrackData(
        self.values[tuple(slice_list)],
        resolution=self.resolution,
        metadata=self.metadata,
        interval=interval,
        uns=self.uns,
    )

  def slice_by_interval(
      self, interval: genome.Interval, match_resolution: bool = False
  ) -> 'TrackData':
    """Slices the track data using a `genome.Interval`.

    Args:
      interval: The interval to slice to.
      match_resolution: If True, the interval will first be extended to make
        sure the width is divisible by resolution.

    Returns:
      A new `TrackData` object sliced to the interval.

    Raises:
      ValueError: If `.interval` is not specified or if the specified interval
        is not fully contained within the current interval.
    """
    if self.interval is None:
      raise ValueError(
          '.interval is needs to be specified for slice_by_interval.'
      )
    if not self.interval.contains(interval):
      raise ValueError(
          f'Interval {self.interval=} does not fully contain {interval=}.'
      )
    start = interval.start - self.interval.start
    end = interval.end - self.interval.start

    if match_resolution and self.resolution != 1:
      start = int(np.floor(start / self.resolution) * self.resolution)
      end = int(np.ceil(end / self.resolution) * self.resolution)
    return self.slice_by_positions(start, end)

  def pad(self, start_pad: int, end_pad: int) -> 'TrackData':
    """Pads the track data along positional axes.

    Args:
      start_pad: The amount of padding to add at the beginning.
      end_pad: The amount of padding to add at the end.

    Returns:
      A new `TrackData` object with padded values.

    Raises:
      ValueError: If `start_pad` or `end_pad` is not divisible by the
      resolution.
    """
    if start_pad == 0 and end_pad == 0:
      return self
    if start_pad % self.resolution != 0:
      raise ValueError(f'start_pad needs to be divisible by {self.resolution}')
    if end_pad % self.resolution != 0:
      raise ValueError(f'end_pad needs to be divisible by {self.resolution}')

    pad = [(0, 0)] * self.values.ndim
    for axis in self.positional_axes:
      pad[axis] = (start_pad // self.resolution, end_pad // self.resolution)

    return TrackData(
        np.pad(self.values, tuple(pad)),
        resolution=self.resolution,
        metadata=self.metadata,
        interval=None,  # Padding invalidates the interval.
        uns=self.uns,
    )

  def resize(self, width: int) -> 'TrackData':
    """Resizes the track data by cropping or padding with a fixed center.

    Args:
      width: The desired width in base pairs.

    Returns:
      A new `TrackData` object with resized values.

    Raises:
      ValueError: If `width` is not divisible by the resolution.
    """
    if width == self.width:
      return self
    elif width > self.width:
      if width % self.resolution != 0:
        raise ValueError(f'width needs to be divisible by {self.resolution}')
      pad_amount = (width - self.width) // self.resolution
      pad_start = (pad_amount // 2 + pad_amount % 2) * self.resolution
      pad_end = (pad_amount // 2) * self.resolution
      return self.pad(pad_start, pad_end)
    else:
      crop_amount = (self.width - width) // self.resolution
      start = (crop_amount // 2 + crop_amount % 2) * self.resolution
      return self.slice_by_positions(start, start + width)

  def upsample(
      self,
      resolution: int,
      aggregation_type: AggregationType = AggregationType.SUM,
  ) -> 'TrackData':
    """Upsamples the track data to a higher resolution.

    Args:
      resolution: The desired resolution in base pairs.
      aggregation_type: The aggregation method to use for pooling the values.

    Returns:
      A new `TrackData` object with upsampled values.

    Raises:
      ValueError: If `resolution` is not lower than the current resolution
        or not divisible by the current resolution.
    """
    if resolution == self.resolution:
      return self
    if resolution > self.resolution:
      raise ValueError(f'Resolution must be lower than {self.resolution}')
    repeat = self.resolution // resolution
    if self.resolution % resolution != 0:
      raise ValueError(f'Resolution not divisible by {resolution}')

    values = self.values
    for axis in self.positional_axes:
      values = np.repeat(values, repeat, axis=axis)
      match aggregation_type:
        case AggregationType.SUM:
          values = values / repeat
        case AggregationType.MAX:
          pass
    return TrackData(
        values,
        resolution=resolution,
        metadata=self.metadata,
        interval=self.interval,
        uns=self.uns,
    )

  def downsample(
      self,
      resolution: int,
      aggregation_type: AggregationType = AggregationType.SUM,
  ) -> 'TrackData':
    """Downsamples the track data to a lower resolution.

    Args:
      resolution: The desired resolution in base pairs.
      aggregation_type: The aggregation method to use for pooling the values.

    Returns:
      A new `TrackData` object with downsampled values.

    Raises:
      ValueError: If `resolution` is not greater than the current resolution
        or not divisible by the current resolution.
    """
    if resolution == self.resolution:
      return self
    if resolution < self.resolution:
      raise ValueError(f'Resolution must be greater than {self.resolution}')
    if resolution % self.resolution != 0:
      raise ValueError(f'Resolution not divisible by {resolution}')
    pool_width = resolution // self.resolution

    values = self.values
    for axis in self.positional_axes:
      # Bring axis of interest to the front, reshape and aggregate, and reswap
      values = np.swapaxes(values, 0, axis)
      shape = list(values.shape)
      reshaped_values = values.reshape(
          [shape[0] // pool_width, pool_width] + shape[1:]
      )
      match aggregation_type:
        case AggregationType.SUM:
          values = reshaped_values.sum(axis=1)
        case AggregationType.MAX:
          values = reshaped_values.max(axis=1)
      values = np.swapaxes(values, 0, axis)

    return TrackData(
        values,
        resolution=resolution,
        metadata=self.metadata,
        interval=self.interval,
        uns=self.uns,
    )

  def change_resolution(
      self,
      resolution: int,
      aggregation_type: AggregationType = AggregationType.SUM,
  ) -> 'TrackData':
    """Changes the resolution of the track data.

    Args:
      resolution: The desired resolution in base pairs.
      aggregation_type: The aggregation method to use for pooling the values.

    Returns:
      A new `TrackData` object with the new resolution.
    """
    if resolution >= self.resolution:
      return self.downsample(resolution, aggregation_type)
    else:
      return self.upsample(resolution, aggregation_type)

  def filter_tracks(self, mask: np.ndarray | list[bool]) -> 'TrackData':
    """Filters tracks by a boolean mask.

    Args:
      mask: A boolean mask to select tracks.

    Returns:
      A new `TrackData` object with the filtered tracks.
    """
    return TrackData(
        self.values[..., mask],
        resolution=self.resolution,
        metadata=self.metadata.iloc[mask],
        interval=self.interval,
        uns=self.uns,
    )

  def filter_to_positive_strand(self) -> 'TrackData':
    """Filters tracks to the positive DNA strand."""
    return self.filter_tracks(self.strands == genome.STRAND_POSITIVE)

  def filter_to_negative_strand(self) -> 'TrackData':
    """Filters tracks to the negative DNA strand."""
    return self.filter_tracks(self.strands == genome.STRAND_NEGATIVE)

  def filter_to_nonnegative_strand(self) -> 'TrackData':
    """Filters tracks to the non-negative DNA strands (positive and unstranded)."""
    return self.filter_tracks(self.strands != genome.STRAND_NEGATIVE)

  def filter_to_nonpositive_strand(self) -> 'TrackData':
    """Filters tracks to the non-positive DNA strands (negative and unstranded)."""
    return self.filter_tracks(self.strands != genome.STRAND_POSITIVE)

  def filter_to_stranded(self) -> 'TrackData':
    """Filters tracks to stranded tracks (excluding unstranded)."""
    return self.filter_tracks(self.strands != genome.STRAND_UNSTRANDED)

  def filter_to_unstranded(self) -> 'TrackData':
    """Filters tracks to unstranded tracks."""
    return self.filter_tracks(self.strands == genome.STRAND_UNSTRANDED)

  def select_tracks_by_index(
      self, idx: np.ndarray | Sequence[int]
  ) -> 'TrackData':
    """Selects tracks by numerical index.

    Args:
      idx: A list or array of numerical indices to select tracks.

    Returns:
      A new `TrackData` object with the selected tracks.
    """
    return TrackData(
        self.values[..., idx],
        resolution=self.resolution,
        metadata=self.metadata.iloc[idx],
        interval=self.interval,
        uns=self.uns,
    )

  def select_tracks_by_name(
      self, names: np.ndarray | Sequence[str]
  ) -> 'TrackData':
    """Selects tracks by name.

    Args:
      names: A list or array of track names to select.

    Returns:
      A new `TrackData` object with the selected tracks.
    """
    track_idx = pd.Series(np.arange(self.num_tracks), index=self.names)
    return self.select_tracks_by_index(track_idx.loc[names].values)

  def __getitem__(self, index: Index) -> 'TrackData':
    """Retrieves a subset of TrackData using positional and/or track indices.

    This method allows slicing `TrackData` similar to numpy arrays or pandas
    DataFrames. The index can be a single value or a tuple.

    Args:
      index: A single index or a tuple of indices. If a single index, it's
        treated as a positional index if the `TrackData` has positional axes,
        otherwise as a track index. If a tuple, the first element specifies the
        positional slice, and the second element specifies the track index.
        Positional indices can be `int`, `slice`, or `genome.Interval`, and this
        slice is applied to all positional axes of the `values` array. Track
        indices can be `int`, `str` (track name), `slice`, `Sequence[int]`, or
        `Sequence[str]` (track names).

    Returns:
      A new `TrackData` object containing the selected subset.

    Raises:
      IndexError: If a slice step is not 1 for positional indexing.
      IndexError: If an unsupported index type is provided.
    """
    if isinstance(index, tuple):
      position_index, track_index = index
    elif self.positional_axes:
      position_index, track_index = index, None
    else:
      position_index, track_index = None, index
      if isinstance(track_index, genome.Interval):
        raise IndexError(
            'Track indexing by interval is supported only when there are'
            ' positional axes.'
        )

    tdata = self
    match position_index:
      case None:
        pass
      case int():
        tdata = tdata.slice_by_positions(position_index, position_index + 1)
      case slice():
        if position_index.step is not None and position_index.step != 1:
          raise IndexError('Slice step must be 1 for positional indexing.')
        if position_index != slice(None):
          tdata = tdata.slice_by_positions(
              0 if position_index.start is None else position_index.start,
              tdata.width
              if position_index.stop is None
              else position_index.stop,
          )
      case genome.Interval():
        tdata = tdata.slice_by_interval(position_index)
      case _:
        raise IndexError(
            f'Unsupported positional index type: {type(position_index)}'
        )

    match track_index:
      case None:
        pass
      case str():
        tdata = tdata.select_tracks_by_name([track_index])
      case int():
        tdata = tdata.select_tracks_by_index([track_index])
      case slice():
        if track_index != slice(None):
          indices = np.arange(tdata.num_tracks)[track_index]
          tdata = tdata.select_tracks_by_index(indices)
      case np.ndarray() if np.issubdtype(track_index.dtype, np.character):
        tdata = tdata.select_tracks_by_name(track_index)
      case np.ndarray():
        tdata = tdata.select_tracks_by_index(track_index)
      case Sequence():
        track_index_arr = np.asarray(track_index)
        if np.issubdtype(track_index_arr.dtype, np.character):
          tdata = tdata.select_tracks_by_name(track_index_arr)
        else:
          tdata = tdata.select_tracks_by_index(track_index_arr)
      case _:
        raise IndexError(f'Unsupported track index type: {type(track_index)}')
    return tdata

  def groupby(self, column: str) -> dict[str, 'TrackData']:
    """Splits tracks into groups based on a metadata column.

    This method splits the tracks in the `TrackData` object into separate
    `TrackData` objects based on the unique values in the specified metadata
    column. It returns a dictionary where the keys are the  unique values in
    the column, and the values are new `TrackData` objects containing the
    tracks corresponding to each key.

    Args:
      column: The name of the metadata column to split by.

    Returns:
      A dictionary mapping unique values in the column to `TrackData` objects
      containing the corresponding tracks.
    """
    output = {}
    for key in self.metadata[column].unique():
      mask = (self.metadata[column] == key).values
      output[key] = self.filter_tracks(mask)
    return output

  def _reverse_complement_idx(self) -> np.ndarray:
    """Gets indices for reverse complementing the tracks.

    Returns:
      An array of indices that reorders the tracks to achieve reverse
      complementation.

    Raises:
      ValueError: If not all stranded tracks have both '+' and '-' strands,
         or if the number of '+' and '-' stranded tracks is not equal.
    """
    df_strands = pd.DataFrame({
        'name': self.names,
        'strand': self.strands,
        'old_idx': np.arange(self.num_tracks),
    })
    df_strands = df_strands[df_strands.strand != genome.STRAND_UNSTRANDED]
    df_strands.sort_values(['strand', 'name'], inplace=True)
    if np.all(df_strands.groupby('name').size() != 2):
      raise ValueError('Not all stranded tracks have both + and - strand.')
    if (df_strands.strand == genome.STRAND_POSITIVE).sum() != (
        df_strands.strand == genome.STRAND_NEGATIVE
    ).sum():
      raise ValueError(
          'We need to have the exact same number of + and - stranded tracks'
      )
    new_idx = df_strands.old_idx.values.reshape((2, -1))[::-1].ravel()
    # Swap strands by idx.
    idx = np.arange(self.num_tracks)
    idx[df_strands.old_idx.values] = new_idx
    return idx

  def reverse_complement(self) -> 'TrackData':
    """Reverse complements the track data and interval if present.

    Returns:
      A new `TrackData` object with reverse complemented tracks.
    """
    if self.interval:
      # Note that the interval needs to be stranded in order to perform
      # this operation.
      interval = self.interval.swap_strand()
    else:
      interval = None

    idx = self._reverse_complement_idx()
    slices: list[slice | np.ndarray] = [slice(None)] * self.values.ndim
    slices[-1] = idx
    for axis in self.positional_axes:
      slices[axis] = slice(None, None, -1)

    return TrackData(
        self.values[tuple(slices)],
        resolution=self.resolution,
        metadata=self.metadata.iloc[idx],
        interval=interval,
        uns=self.uns,
    )

  def _check_track_data_compatibility(self, other: 'TrackData') -> None:
    """Checks if two `TrackData` objects are compatible for sum/diff.

    Args:
      other: The other `TrackData` object to compare.

    Raises:
      TypeError: If `other` is not a `TrackData` object.
      ValueError: If the intervals, resolutions, shapes, or metadata
        shapes don't match between the two objects.
    """
    if not isinstance(other, TrackData):
      raise TypeError(
          f'Unsupported type "{type(other)}". Must be a TrackData object'
      )
    if self.interval != other.interval:
      raise ValueError('Intervals must match for the two TrackData objects.')
    if self.resolution != other.resolution:
      raise ValueError('Resolutions must match for the two TrackData objects.')
    if self.values.shape != other.values.shape:
      raise ValueError('Shapes must match for the two TrackData objects.')
    if self.metadata.shape != other.metadata.shape:
      raise ValueError(
          'Metadata shapes must match for the two TrackData objects.'
      )

  def __add__(self, other: 'TrackData') -> 'TrackData':
    """Adds the values of two `TrackData` objects.

    Args:
      other: The `TrackData` object to add.

    Returns:
      A new `TrackData` object with the summed values.

    Raises:
      ValueError: If the objects are not compatible (see
        `_check_track_data_compatibility`).
      TypeError: If `other` is not a `TrackData` object.
    """
    self._check_track_data_compatibility(other)
    new_values = self.values + other.values
    return TrackData(
        values=new_values,
        metadata=self.metadata,
        resolution=self.resolution,
        interval=self.interval,
        uns=self.uns,
    )

  def __sub__(self, other: 'TrackData') -> 'TrackData':
    """Subtracts the values of two `TrackData` objects.

    Args:
      other: The `TrackData` object to subtract.

    Returns:
      A new `TrackData` object with the difference of the values.

    Raises:
      ValueError: If the objects are not compatible (see
        `_check_track_data_compatibility`).
      TypeError: If `other` is not a `TrackData` object.
    """
    self._check_track_data_compatibility(other)
    new_values = self.values - other.values
    return TrackData(
        values=new_values,
        metadata=self.metadata,
        resolution=self.resolution,
        interval=self.interval,
        uns=self.uns,
    )


def concat(
    track_datas: Sequence[TrackData],
    extra_metadata_name_and_keys: (
        tuple[str, Sequence[str | int | float]] | None
    ) = None,
) -> TrackData:
  """Concatenates multiple `TrackData` objects along the track dimension.

  This function combines multiple `TrackData` objects into a single object
  by concatenating their values and metadata. The resulting `TrackData`
  object will have the same resolution and interval as the input objects.

  Args:
    track_datas: A sequence of `TrackData` objects to concatenate. All objects
      must have the same resolution, interval, and width.
    extra_metadata_name_and_keys: An optional tuple specifying a new metadata
      column to add. The first element is the column name, and the second is a
      sequence of values to populate the column.

  Returns:
    A new `TrackData` object containing the concatenated data.

  Raises:
    ValueError: If the input `TrackData` objects have different resolutions,
      intervals, or widths, or if the length of
      `extra_metadata_name_and_keys[1]` does not match the length of
      `track_datas`.
  """
  if len(set(x.resolution for x in track_datas)) != 1:
    raise ValueError('Track data contain multiple resolutions')
  if len(set(str(x.interval) for x in track_datas)) != 1:
    raise ValueError('Track data contain multiple intervals')
  if len(set(x.width for x in track_datas)) != 1:
    raise ValueError('Track data are of different width')
  if extra_metadata_name_and_keys:
    if len(extra_metadata_name_and_keys[1]) != len(track_datas):
      raise ValueError(
          'Second element of new_metadata_name_and_keys must be the same'
          ' length as track_datas'
      )

  concatenated_metadata = (
      pd.concat(
          [x.metadata for x in track_datas],
          keys=extra_metadata_name_and_keys[1]
          if extra_metadata_name_and_keys
          else None,
          names=[extra_metadata_name_and_keys[0]]
          if extra_metadata_name_and_keys
          else None,
      )
      .reset_index(level=0, drop=extra_metadata_name_and_keys is None)
      .reset_index(drop=True)
  )
  return TrackData(
      np.concatenate(
          [x.values for x in track_datas], axis=track_datas[0].values.ndim - 1
      ),
      resolution=track_datas[0].resolution,
      metadata=concatenated_metadata,
      interval=track_datas[0].interval,
      uns=None,
  )


def interleave(
    track_datas: Sequence[TrackData], name_prefixes: Sequence[str]
) -> TrackData:
  """Interleaves multiple `TrackData` objects by alternating rows.

  This function combines multiple `TrackData` objects into a single object
  by interleaving their rows and metadata. This interleaves operation alternates
  between the trackdatas, like shuffling cards, i.e., 'abcd' interleaved with
  'efgh' would be "aebfcgdh". The resulting `TrackData` object will have the
  same resolution and interval as the input objects, but the number of tracks
  will be the sum of the tracks in the input objects.

  Args:
    track_datas: A sequence of `TrackData` objects to interleave. All objects
      must have the same shape, resolution, and interval. The order in this list
      will determine the interleaving order.
    name_prefixes: A sequence of name prefixes to add to the track names in the
      metadata to ensure uniqueness of (name, strand) pairs.

  Returns:
    A new `TrackData` object containing the interleaved data.

  Raises:
    ValueError: If the input `TrackData` objects have different shapes,
      resolutions, or intervals.
  """
  # Checks on the track data.
  shapes = set(data.values.shape for data in track_datas)
  if len(shapes) != 1:
    raise ValueError(
        'Cannot interleave track data which have different shapes. '
        f'Detected shapes: {shapes}'
    )

  if any(data.resolution != track_datas[0].resolution for data in track_datas):
    raise ValueError(
        'Cannot interleave track data which have different resolutions. '
        f'Detected shapes: {shapes}'
    )

  if any(data.interval != track_datas[0].interval for data in track_datas):
    raise ValueError(
        'Cannot interleave track data which have different intervals. '
    )

  # Interleave arrays.
  shape = list(track_datas[0].values.shape)
  shape[-1] = shape[-1] * len(track_datas)
  interleaved_data = np.empty(tuple(shape), dtype=track_datas[0].values.dtype)

  for i, data in enumerate(track_datas):
    interleaved_data[..., i :: len(track_datas)] = data.values

  # Interleave metadata.
  metadatas = []
  for prefix, data in zip(name_prefixes, track_datas):
    metadata = data.metadata.copy()
    metadata['name'] = prefix + metadata['name']
    metadatas.append(metadata)

  # Assign a new index idx that, when sorted, will produce an interleave.
  df = pd.concat([
      metadata.assign(
          idx=np.arange(
              0,
              stop=(len(metadata) * len(track_datas)),
              step=len(track_datas),
          )
          + i
      )
      for i, metadata in enumerate(metadatas)
  ])
  assert isinstance(df, pd.DataFrame)
  interleaved_metadata = (
      df.sort_values('idx').drop('idx', axis=1).reset_index(drop=True)
  )

  return TrackData(
      values=interleaved_data,
      metadata=interleaved_metadata,
      resolution=track_datas[0].resolution,
      interval=track_datas[0].interval,
      uns={'num_interleaved_trackdatas': len(track_datas)},
  )
