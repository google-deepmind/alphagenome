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

"""Splice junction data container.

`JunctionData` stores splice junction data for a given transcript or interval.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from jaxtyping import (  # pylint: disable=g-multiple-import, g-importing-member
    Float,
    Shaped,
)

from alphagenome import typing
from alphagenome.data import genome, ontology

JunctionMetadata = pd.DataFrame


@typing.jaxtyped
@dataclasses.dataclass(frozen=True)
class JunctionData:
  """Container for storing splice junction data.

  Attributes:
    junctions: A numpy array representing the splice junctions.
    values: A numpy array of floats representing the values associated with each
      junction for each track.
    metadata: A pandas DataFrame containing metadata for each track.
    interval: An optional `Interval` object representing the genomic region
      containing the junctions.
    uns: An optional dictionary to store additional unstructured data.

  Raises:
    ValueError: If the number of tracks in `values` does not match the number
      of rows in `metadata`, or if `metadata` contains duplicate names.
  """

  junctions: Shaped[np.ndarray, 'num_junctions']
  values: Float[np.ndarray, 'num_junctions num_tracks']
  metadata: JunctionMetadata
  interval: genome.Interval | None = None
  uns: dict[str, Any] | None = None

  def __post_init__(self):
    """Validates the consistency of the data."""
    if self.values.shape[1] != len(self.metadata):
      raise ValueError(
          f'Number of tracks {self.values.shape[1]} and '
          f'metadata {len(self.metadata)} do not match.'
      )

    if self.metadata['name'].duplicated().any():
      raise ValueError('Metadata contain duplicated names.')

  def __len__(self):
    """Returns the number of junctions."""
    return len(self.junctions)

  @property
  def num_tracks(self) -> int:
    """Returns the number of tracks."""
    return len(self.metadata)

  @property
  def names(self) -> np.ndarray:
    """Returns an array of track names (not necessarily unique)."""
    return self.metadata['name'].values

  @property
  def strands(self) -> np.ndarray:
    """Returns an array of junction strands."""
    return np.array([j.strand for j in self.junctions])

  @property
  def possible_strands(self) -> np.ndarray:
    """All possible strands."""
    return np.unique(self.strands)

  @property
  def ontology_terms(self) -> Sequence[ontology.OntologyTerm | None] | None:
    """Returns a list of ontology terms (if available)."""
    if 'ontology_curie' in self.metadata.columns:
      return [
          ontology.from_curie(curie) if curie is not None else None
          for curie in self.metadata['ontology_curie'].values
      ]
    else:
      return None

  # Track order / subset wrangling:
  def filter_tracks(self, mask: np.ndarray | list[bool]) -> 'JunctionData':
    """Filters tracks by a boolean mask.

    Args:
      mask: A boolean mask to select tracks.

    Returns:
      A new `JunctionData` object with the filtered tracks.
    """
    return JunctionData(
        junctions=self.junctions,
        values=self.values[:, mask],
        metadata=self.metadata.loc[mask],
        interval=self.interval,
        uns=self.uns,
    )

  def filter_to_strand(self, strand: str) -> 'JunctionData':
    """Filters junctions to a specific DNA strand.

    Args:
      strand: The strand to filter by ('+' or '-').

    Returns:
      A new `JunctionData` object with junctions on the specified strand.
    """
    mask = self.strands == strand
    return JunctionData(
        junctions=self.junctions[mask],
        values=self.values[mask, :],
        metadata=self.metadata,
        interval=self.interval,
        uns=self.uns,
    )

  def normalize_values(self, total_k: float = 10.0) -> 'JunctionData':
    """Normalizes the values by the k value."""
    values = self.values * total_k / self.values.sum()
    return JunctionData(
        junctions=self.junctions,
        values=values,
        metadata=self.metadata,
        interval=self.interval,
        uns=self.uns,
    )

  def filter_to_positive_strand(self) -> 'JunctionData':
    """Filters junctions to the positive DNA strand."""
    return self.filter_to_strand(genome.STRAND_POSITIVE)

  def filter_to_negative_strand(self) -> 'JunctionData':
    """Filters junctions to the negative DNA strand."""
    return self.filter_to_strand(genome.STRAND_NEGATIVE)

  def filter_by_tissue(self, tissue: str) -> 'JunctionData':
    """Filters tracks by GTEx tissue type.

    Args:
      tissue: The GTEx tissue type to filter by.

    Returns:
      A new `JunctionData` object with tracks from the specified tissue.

    Raises:
      ValueError: If the metadata does not contain a 'gtex_tissue' column.
    """
    if 'gtex_tissue' not in self.metadata.columns:
      raise ValueError(
          'Metadata does not contain gtex_tissue column. '
          f'Got {set(self.metadata.columns)}.'
      )
    return self.filter_tracks(self.metadata['gtex_tissue'] == tissue)

  def filter_by_name(self, name: str) -> 'JunctionData':
    """Filters tracks by name."""
    return self.filter_tracks(self.metadata['name'] == name)

  def filter_by_ontology(self, ontology_curie: str) -> 'JunctionData':
    """Filters tracks by ontology term.

    Args:
      ontology_curie: The ontology term CURIE to filter by.

    Returns:
      A new `JunctionData` object with tracks associated with the specified
        ontology term.

    Raises:
      ValueError: If the metadata does not contain an 'ontology_curie' column.
    """
    if 'ontology_curie' not in self.metadata.columns:
      raise ValueError(
          'Metadata does not contain ontology_curie column. '
          f'Got {set(self.metadata.columns)}.'
      )
    return self.filter_tracks(self.metadata['ontology_curie'] == ontology_curie)

  def intersect_with_interval(
      self, interval: genome.Interval
  ) -> 'JunctionData':
    """Returns the intersection of the junctions and the interval."""
    mask = np.array([j.overlaps(interval) for j in self.junctions])
    return JunctionData(
        junctions=self.junctions[mask],
        values=self.values[mask, :],
        metadata=self.metadata,
        interval=self.interval,
        uns=self.uns,
    )


def get_junctions_to_plot(
    *,
    predictions: JunctionData,
    name: str,
    strand: str,
    k_threshold: float | None = 0.0,
) -> list[genome.Junction]:
  """Gets a list of junctions to plot.

  Filters the junctions in the `predictions` by name and strand, and
  applies a threshold on the `k` value (read count).

  Args:
    predictions: A `JunctionData` object containing junction predictions.
    name: The name to filter by.
    strand: The strand to filter by ('+' or '-').
    k_threshold: The minimum `k` value for a junction to be included. If None,
      use 5% of the maximum value.

  Returns:
    A list of `Junction` objects to plot.

  Raises:
    ValueError: If more than one track is found for the specified name.
  """
  filtered = predictions.filter_by_name(name)
  if filtered.num_tracks > 1:
    raise ValueError(
        f'Expected only one ontology term, got {filtered.num_tracks}.'
    )
  filtered_junctions = []
  if filtered.values.size == 0:
    return filtered_junctions

  if k_threshold is None:
    k_threshold = filtered.values.max() * 0.05

  # Round k for better visualization.
  for interval, k in zip(filtered.junctions, filtered.values, strict=True):
    # Filter by threshold and strand.
    if interval.strand != strand:
      continue
    k = k.item()
    if k >= k_threshold:
      k = round(k, 2)

      filtered_junctions.append(
          genome.Junction(
              interval.chromosome,
              interval.start,
              interval.end,
              interval.strand,
              k=k,
          )
      )
  return filtered_junctions
