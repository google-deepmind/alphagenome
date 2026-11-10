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


"""Gene expression data container.

`Gene Expression` stores gene expression data for a given interval.
"""

from collections.abc import Sequence
import dataclasses

from alphagenome import typing
from alphagenome.data import genome
import anndata
from jaxtyping import Float  # pylint: disable=g-importing-member
import numpy as np
import pandas as pd

GeneMetadata = pd.DataFrame
TrackMetadata = pd.DataFrame
GeneIndex = TrackIndex = (
    pd.Series | np.ndarray | Sequence[int] | Sequence[str] | slice | int | str
)


@typing.jaxtyped
@dataclasses.dataclass(frozen=True)
class GeneExpressionData:
  """Container for storing gene expression data.

  Used to store gene-level predictions for denoised single-cell gene expression
  (`SC_GENE_EXPRESSION_DENOISED`; one channel per strand, in CPM) and
  protocol-specific single-cell gene expression (`SC_GENE_EXPRESSION`, in CPM).

  Attributes:
    values: A numpy array of floats representing the gene expression for each
      track.
    track_metadata: A pandas DataFrame containing metadata for each track.
    gene_metadata: A pandas DataFrame containing metadata for each gene.
    interval: An `Interval` object representing the genomic region containing
      the genes.
    names: Array of track names from `track_metadata['name']`.
    strands: Array of track strands from `track_metadata['strand']`.
    num_tracks: Total number of tracks along axis 1 of `values`.

  Raises:
    ValueError: If the number of tracks in `values` does not match the number
      of rows in `track_metadata`, or if the number of gene in `values` does not
      match the number of rows in `gene_metadata` or if `track_metadata`
      contains duplicate names.
  """

  values: Float[np.ndarray, 'num_genes num_tracks']
  track_metadata: TrackMetadata
  gene_metadata: GeneMetadata
  interval: genome.Interval

  def __post_init__(self):
    """Validates the consistency of the data."""
    if self.values.shape[1] != len(self.track_metadata):
      raise ValueError(
          f'Number of tracks {self.values.shape[1]} and '
          f'track metadata {len(self.track_metadata)} do not match.'
      )

    if self.values.shape[0] != len(self.gene_metadata):
      raise ValueError(
          f'Number of genes {self.values.shape[0]} and '
          f'gene metadata {len(self.gene_metadata)} do not match.'
      )
    if self.track_metadata[['name', 'strand']].duplicated().any():
      raise ValueError(
          'Metadata contains duplicated values for (name, strand) tuples.'
      )

  def __getitem__(
      self, index: GeneIndex | tuple[GeneIndex, TrackIndex]
  ) -> 'GeneExpressionData':
    """Subset/slice the `GeneExpressionData` by gene and track indices.

    This method allows slicing `GeneExpressionData` similar to numpy arrays or
    pandas DataFrames.

    Args:
      index: The index used for slicing. This can be an int, slice, list, numpy
        array, pandas Series, or a tuple thereof. If a single object is
        provided, only genes are sliced. String values in `gene_idx` are matched
        against `gene_metadata.gene_id`, and string values in `track_idx` are
        matched against `track_metadata.name`.

    Returns:
      A new `GeneExpressionData` object containing the subsetted data.
    """
    if isinstance(index, tuple):
      gene_idx, track_idx = index
    else:
      gene_idx = index
      track_idx = slice(None)

    adata = self.to_anndata()
    adata.obs['orig_idx'] = adata.obs_names.values.astype(int)
    adata.var['orig_idx'] = adata.var_names.values.astype(int)
    adata.obs_names = self.gene_metadata.gene_id
    adata.var_names = self.track_metadata.name
    new_adata = adata[gene_idx, track_idx]

    new_gene_metadata = new_adata.obs.set_index('orig_idx')
    new_gene_metadata.index.name = None
    new_track_metadata = new_adata.var.set_index('orig_idx')
    new_track_metadata.index.name = None

    return GeneExpressionData(
        values=np.asarray(new_adata.X),
        gene_metadata=new_gene_metadata,
        track_metadata=new_track_metadata,
        interval=self.interval,
    )

  def to_anndata(self) -> anndata.AnnData:
    """Converts the `GeneExpressionData` to an `AnnData` object."""
    # Convert to str to prevent implicit anndata conversion.
    var = self.track_metadata.copy()
    obs = self.gene_metadata.copy()
    var.index = var.index.astype(str)
    obs.index = obs.index.astype(str)
    return anndata.AnnData(
        self.values,
        var=var,
        obs=obs,
        uns={'interval': self.interval},
    )

  def slice_by_interval(
      self, interval: genome.Interval
  ) -> 'GeneExpressionData':
    """Slices the `GeneExpressionData` to include only genes within an interval.

    This method filters the `GeneExpressionData` to retain only those genes
    whose genomic intervals overlap with the provided `interval`.

    Args:
      interval: The `genome.Interval` object defining the genomic region.

    Returns:
      A new `GeneExpressionData` object containing only the genes that overlap
      with the specified `interval`.
    """

    gene_intervals = self.get_genes_as_intervals()
    gene_intervals = [g.overlaps(interval) for g in gene_intervals]
    new_gdata = self[gene_intervals]
    return dataclasses.replace(new_gdata, interval=interval)

  def get_genes_as_intervals(self) -> Sequence[genome.Interval]:
    """Returns the gene intervals as a sequence of `Interval` objects.

    Gene intervals are derived from 'Chr', 'Start', and 'End' columns in
    `self.gene_metadata`.

    Returns:
      A sequence of `genome.Interval` objects, each representing a gene.

    Raises:
      ValueError: If the gene metadata does not contain 'Chromosome', 'Start',
        or 'End' columns.
    """
    if not all(
        col in self.gene_metadata.columns
        for col in ['Chromosome', 'Start', 'End']
    ):
      raise ValueError('Gene coordinates are missing.')

    return [
        genome.Interval(row.Chromosome, row.Start, row.End)
        for row in self.gene_metadata.itertuples()
    ]

  def copy(self) -> 'GeneExpressionData':
    """Returns a deep copy of the `GeneExpressionData` object."""
    if self.interval:
      interval = self.interval.copy()
    else:
      interval = None
    return GeneExpressionData(
        values=self.values.copy(),
        track_metadata=self.track_metadata.copy(),
        gene_metadata=self.gene_metadata.copy(),
        interval=interval,  # pyrefly: ignore[bad-argument-type]
    )

  def with_name_suffix(self, suffix: str) -> 'GeneExpressionData':
    """Returns a new `GeneExpressionData` object with suffixed track names."""
    new_metadata = self.track_metadata.copy()
    new_metadata['name'] = new_metadata['name'] + suffix
    return dataclasses.replace(self, track_metadata=new_metadata)

  def with_metadata_column(
      self,
      name: str,
      value: str | pd.Series,
  ) -> 'GeneExpressionData':
    """Returns a new `GeneExpressionData` with updated track metadata."""
    new_metadata = self.track_metadata.copy()
    new_metadata[name] = value
    return dataclasses.replace(self, track_metadata=new_metadata)

  @property
  def names(self) -> np.ndarray:
    """Returns an array of track names (not necessarily unique)."""
    return self.track_metadata['name'].values

  @property
  def strands(self) -> np.ndarray:
    """Returns an array of track strands."""
    return self.track_metadata['strand'].values

  @property
  def num_tracks(self) -> int:
    """Returns the number of tracks."""
    return self.values.shape[-1]

  def select_tracks_by_index(
      self, idx: np.ndarray | Sequence[int]
  ) -> 'GeneExpressionData':
    """Selects tracks by numerical index.

    Args:
      idx: A list or array of numerical indices to select tracks.

    Returns:
      A new `GeneExpressionData` object with the selected tracks.
    """
    return self[:, idx]

  def filter_tracks(
      self, mask: np.ndarray | list[bool]
  ) -> 'GeneExpressionData':
    """Filters tracks by a boolean mask.

    Args:
      mask: A boolean mask to select tracks.

    Returns:
      A new `GeneExpressionData` object with the filtered tracks.
    """
    return self[:, mask]

  def filter_to_positive_strand(self) -> 'GeneExpressionData':
    """Filters tracks to the positive DNA strand."""
    return self.filter_tracks(self.strands == genome.STRAND_POSITIVE)

  def filter_to_negative_strand(self) -> 'GeneExpressionData':
    """Filters tracks to the negative DNA strand."""
    return self.filter_tracks(self.strands == genome.STRAND_NEGATIVE)

  def filter_to_nonnegative_strand(self) -> 'GeneExpressionData':
    """Filters tracks to the non-negative DNA strands (positive and unstranded)."""
    return self.filter_tracks(self.strands != genome.STRAND_NEGATIVE)

  def filter_to_nonpositive_strand(self) -> 'GeneExpressionData':
    """Filters tracks to the non-positive DNA strands (negative and unstranded)."""
    return self.filter_tracks(self.strands != genome.STRAND_POSITIVE)

  def filter_to_stranded(self) -> 'GeneExpressionData':
    """Filters tracks to stranded tracks (excluding unstranded)."""
    return self.filter_tracks(self.strands != genome.STRAND_UNSTRANDED)

  def filter_to_unstranded(self) -> 'GeneExpressionData':
    """Filters tracks to unstranded tracks."""
    return self.filter_tracks(self.strands == genome.STRAND_UNSTRANDED)

  def filter_to_observed(self) -> 'GeneExpressionData':
    """Filters tracks to observed tracks."""
    return self.filter_tracks(
        self.track_metadata['imputation_mode'] == 'observed'
    )

  def filter_to_protein_coding(self) -> 'GeneExpressionData':
    """Filters data to protein coding genes only.

    Returns:
      A new `GeneExpressionData` object containing only protein coding genes.

    Raises:
      ValueError: If the 'gene_type' column is missing from gene_metadata.
    """
    if 'gene_type' not in self.gene_metadata.columns:
      raise ValueError(
          "Column 'gene_type' not found in gene_metadata. Available columns:"
          f' {list(self.gene_metadata.columns)}'
      )
    return self[self.gene_metadata['gene_type'] == 'protein_coding']

  def resize(self, width: int) -> 'GeneExpressionData':
    """Resizes the interval, removing genes falling outside the new width.

    Args:
      width: The desired width in base pairs.

    Returns:
      A new `GeneExpressionData` object containing only genes overlapping the
      resized interval.
    """
    resized_interval = self.interval.resize(width)
    return self.slice_by_interval(resized_interval)

  def _check_compatibility(self, other: 'GeneExpressionData') -> None:
    """Checks if two `GeneExpressionData` objects are compatible for sum/diff.

    Args:
      other: The other `GeneExpressionData` object to compare.

    Raises:
      TypeError: If `other` is not a `GeneExpressionData` object.
      ValueError: If the intervals, shapes, gene_metadata information or
      track_metadata shapes don't match between the two objects.
    """
    if not isinstance(other, GeneExpressionData):
      raise TypeError(
          f'Unsupported type "{type(other)}". Must be a GeneExpressionData'
          ' object'
      )
    if self.interval != other.interval:
      raise ValueError(
          'Intervals must match for the two GeneExpressionData objects.'
      )
    if self.values.shape != other.values.shape:
      raise ValueError('Shapes must match for the two TrackData objects.')
    if not self.gene_metadata.equals(other.gene_metadata):
      raise ValueError(
          'Gene Metadata must match for the two GeneExpressionData objects.'
      )
    if self.track_metadata.shape != other.track_metadata.shape:
      raise ValueError(
          'Track Metadata shapes must match for the two TrackData objects.'
      )

  def __add__(self, other: 'GeneExpressionData') -> 'GeneExpressionData':
    """Adds the values of two `GeneExpressionData` objects element-wise."""
    self._check_compatibility(other)
    return GeneExpressionData(
        values=self.values + other.values,
        track_metadata=self.track_metadata,
        gene_metadata=self.gene_metadata,
        interval=self.interval,
    )

  def __sub__(self, other: 'GeneExpressionData') -> 'GeneExpressionData':
    """Subtracts the values of two `GeneExpressionData` objects element-wise."""
    self._check_compatibility(other)
    return GeneExpressionData(
        values=self.values - other.values,
        track_metadata=self.track_metadata,
        gene_metadata=self.gene_metadata,
        interval=self.interval,
    )


def concat(
    gene_expression_datas: Sequence[GeneExpressionData],
    extra_metadata_name_and_keys: (
        tuple[str, Sequence[str | int | float]] | None
    ) = None,
) -> GeneExpressionData:
  """Concatenates multiple `GeneExpressionData` objects along the track dimension.

  This function combines multiple `GeneExpressionData` objects into a single
  object by concatenating their values (along the track dimension) and
  track_metadata. The resulting `GeneExpressionData` object will have the same
  interval and gene_metadata as the input objects.

  Args:
    gene_expression_datas: A sequence of `GeneExpressionData` objects to
      concatenate. All objects must have the same interval and gene_metadata.
    extra_metadata_name_and_keys: An optional tuple specifying a new metadata
      column to add to track_metadata. The first element is the column name, and
      the second is a sequence of values to populate the column.

  Returns:
    A new `GeneExpressionData` object containing the concatenated data.

  Raises:
    ValueError: If `gene_expression_datas` is empty, or if input
      `GeneExpressionData` objects have different intervals or
      gene_metadatas, or if the length of `extra_metadata_name_and_keys[1]`
      does not match the length of `gene_expression_datas`.
  """
  if not gene_expression_datas:
    raise ValueError('gene_expression_datas cannot be empty.')
  first_interval = gene_expression_datas[0].interval
  if not all(x.interval == first_interval for x in gene_expression_datas):
    raise ValueError('Gene expression data contain multiple intervals')
  if not all(
      x.gene_metadata.equals(gene_expression_datas[0].gene_metadata)
      for x in gene_expression_datas
  ):
    raise ValueError('Gene expression data contain different gene_metadata')

  if extra_metadata_name_and_keys:
    if len(extra_metadata_name_and_keys[1]) != len(gene_expression_datas):
      raise ValueError(
          'Second element of extra_metadata_name_and_keys must be the same'
          ' length as gene_expression_datas'
      )
    concatenated_track_metadata = (
        pd.concat(
            [x.track_metadata for x in gene_expression_datas],
            keys=extra_metadata_name_and_keys[1],
            names=[extra_metadata_name_and_keys[0]],
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )
  else:
    concatenated_track_metadata = pd.concat(
        [x.track_metadata for x in gene_expression_datas], ignore_index=True
    )
  return GeneExpressionData(
      values=np.concatenate([x.values for x in gene_expression_datas], axis=1),
      track_metadata=concatenated_track_metadata,
      gene_metadata=gene_expression_datas[0].gene_metadata.copy(),
      interval=gene_expression_datas[0].interval,
  )
