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

"""Abstract base class for AlphaGenome v2 DNA models."""

import abc
from collections.abc import Iterable, Sequence
import concurrent.futures
import enum
from typing import Any, TypeVar

from alphagenome.protos.v1 import dna_model_pb2
from alphagenome.data import genome
from alphagenome.models.v2 import dna_output
from alphagenome.models.v2 import interval_scorers as interval_scorers_lib
from alphagenome.models.v2 import model_condition
from alphagenome.models.v2 import variant_scorers as variant_scorers_lib
import anndata
import numpy as np
import tqdm.auto

# Default maximum number of workers to use for parallel requests.
DEFAULT_MAX_WORKERS = 5


class Organism(enum.Enum):
  """Enumeration of all the available organisms.

  Attributes:
    HOMO_SAPIENS: Human reference genome (GRCh38 / hg38).
    MUS_MUSCULUS: Mouse reference genome (GRCm38 / mm10).
  """

  HOMO_SAPIENS = dna_model_pb2.ORGANISM_HOMO_SAPIENS
  MUS_MUSCULUS = dna_model_pb2.ORGANISM_MUS_MUSCULUS

  def to_proto(self) -> dna_model_pb2.Organism:
    return self.value


_T = TypeVar('_T')


def _validate_not_str(value: object, name: str) -> None:
  """Raises TypeError if `value` is a single `str` instead of a sequence."""
  if isinstance(value, str):
    raise TypeError(f'`{name}` must be a sequence, not a single str.')


def _validate_equal_lengths(
    expected_length: int,
    reference_name: str,
    **sequences: Sequence[Any] | None,
) -> None:
  """Raises ValueError if any non-None sequence length != `expected_length`."""
  for name, values in sequences.items():
    if values is not None and len(values) != expected_length:
      raise ValueError(
          f'{reference_name} and {name} must have the same length. Got'
          f' {expected_length} {reference_name.lower()} and {len(values)}'
          f' {name}.'
      )


def _pad_none(values: Sequence[_T] | None, length: int) -> Sequence[_T | None]:
  """Returns `values` if provided, otherwise `[None] * length`."""
  return [None] * length if values is None else values


class DnaModel(metaclass=abc.ABCMeta):
  """Abstract base class for AlphaGenome DNA models."""

  @abc.abstractmethod
  def predict_sequence(
      self,
      sequence: str,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      interval: genome.Interval | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> dna_output.Output:
    """Generate predictions for a given DNA sequence.

    Args:
      sequence: DNA sequence to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      condition: The condition or composite curie to generate the prediction
        for.
      interval: Optional interval from which the sequence was derived. This is
        used as the interval in the output TrackData.
      cell_conditioning: Optional human single-cell raw UMI counts (`AnnData`),
        unlocking denoised single-cell gene expression
        (`OutputType.SC_GENE_EXPRESSION_DENOISED`; one channel per strand, in
        CPM).
      cell_condition_name: Optional label for expression conditioning. When
        provided, output track names are suffixed with this name. Must be
        provided when `cell_conditioning` is used.

    Returns:
      Output for the provided DNA sequence.
    """

  def predict_sequences(
      self,
      sequences: Sequence[str],
      conditions: Sequence[model_condition.ModelCondition | str] | None = None,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      cell_conditionings: Sequence[anndata.AnnData] | None = None,
      cell_condition_names: Sequence[str] | None = None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
      intervals: Sequence[genome.Interval] | None = None,
  ) -> list[dna_output.Output]:
    """Generate predictions for a batch of DNA sequences.

    Args:
      sequences: DNA sequences to make predictions for.
      conditions: Optional sequence of conditions or composite curies to
        generate the predictions for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      cell_conditionings: Optional sequence of expression conditioning inputs
        (`AnnData`).
      cell_condition_names: Optional sequence of labels for expression
        conditioning. Must be provided when `cell_conditionings` is used.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.
      intervals: Optional intervals from which the sequences were derived. This
        is used as the interval in the output TrackData. Must be the same length
        as `sequences` if provided.

    Returns:
      Outputs for the provided DNA sequences.
    """
    _validate_not_str(sequences, 'sequences')
    _validate_not_str(conditions, 'conditions')
    _validate_not_str(cell_condition_names, 'cell_condition_names')
    num_requests = len(sequences)
    _validate_equal_lengths(
        num_requests,
        'Sequences',
        conditions=conditions,
        cell_conditionings=cell_conditionings,
        cell_condition_names=cell_condition_names,
        intervals=intervals,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.predict_sequence,
              sequence=sequence,
              organism=organism,
              condition=condition,
              requested_outputs=requested_outputs,
              interval=interval,
              cell_conditioning=cell_conditioning,
              cell_condition_name=cell_condition_name,
          )
          for (
              sequence,
              interval,
              condition,
              cell_conditioning,
              cell_condition_name,
          ) in zip(
              sequences,
              _pad_none(intervals, num_requests),
              _pad_none(conditions, num_requests),
              _pad_none(cell_conditionings, num_requests),
              _pad_none(cell_condition_names, num_requests),
              strict=True,
          )
      ]
      futures_as_completed = tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          disable=not progress_bar,
      )
      for future in futures_as_completed:
        if (exception := future.exception()) is not None:
          executor.shutdown(wait=False, cancel_futures=True)
          raise exception

      return [future.result() for future in futures]

  @abc.abstractmethod
  def predict_interval(
      self,
      interval: genome.Interval,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> dna_output.Output:
    """Generate predictions for a given DNA interval.

    Args:
      interval: DNA interval to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      condition: The condition or composite curie to generate the prediction
        for.
      cell_conditioning: Optional human single-cell raw UMI counts (`AnnData`),
        unlocking denoised single-cell gene expression
        (`OutputType.SC_GENE_EXPRESSION_DENOISED`; one channel per strand, in
        CPM).
      cell_condition_name: Optional label for expression conditioning. When
        provided, output track names are suffixed with this name. Must be
        provided when `cell_conditioning` is used.

    Returns:
      Output for the provided DNA interval.
    """

  def predict_intervals(
      self,
      intervals: Sequence[genome.Interval],
      conditions: Sequence[model_condition.ModelCondition | str] | None = None,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      cell_conditionings: Sequence[anndata.AnnData] | None = None,
      cell_condition_names: Sequence[str] | None = None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[dna_output.Output]:
    """Generate predictions for a sequence of DNA intervals.

    Args:
      intervals: DNA intervals to make predictions for.
      conditions: Optional sequence of conditions or composite curies to
        generate the predictions for.
      organism: Organism to use for the predictions.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      cell_conditionings: Optional sequence of expression conditioning inputs
        (`AnnData`).
      cell_condition_names: Optional sequence of labels for expression
        conditioning. Must be provided when `cell_conditionings` is used.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      Outputs for the provided DNA intervals.
    """
    _validate_not_str(conditions, 'conditions')
    _validate_not_str(cell_condition_names, 'cell_condition_names')
    num_requests = len(intervals)
    _validate_equal_lengths(
        num_requests,
        'Intervals',
        conditions=conditions,
        cell_conditionings=cell_conditionings,
        cell_condition_names=cell_condition_names,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.predict_interval,
              interval=interval,
              organism=organism,
              condition=condition,
              requested_outputs=requested_outputs,
              cell_conditioning=cell_conditioning,
              cell_condition_name=cell_condition_name,
          )
          for interval, condition, cell_conditioning, cell_condition_name in (
              zip(
                  intervals,
                  _pad_none(conditions, num_requests),
                  _pad_none(cell_conditionings, num_requests),
                  _pad_none(cell_condition_names, num_requests),
                  strict=True,
              )
          )
      ]
      futures_as_completed = tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          disable=not progress_bar,
      )
      for future in futures_as_completed:
        if (exception := future.exception()) is not None:
          executor.shutdown(wait=False, cancel_futures=True)
          raise exception

      return [future.result() for future in futures]

  @abc.abstractmethod
  def predict_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> dna_output.VariantOutput:
    """Generate predictions for a given DNA variant.

    Args:
      interval: DNA interval to make prediction for.
      variant: DNA variant to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      condition: The condition or composite curie to generate the prediction
        for.
      cell_conditioning: Optional human single-cell raw UMI counts (`AnnData`),
        unlocking denoised single-cell gene expression
        (`OutputType.SC_GENE_EXPRESSION_DENOISED`; one channel per strand, in
        CPM).
      cell_condition_name: Optional label for expression conditioning. When
        provided, output track names are suffixed with this name. Must be
        provided when `cell_conditioning` is used.

    Returns:
      Variant output for the provided DNA interval and variant.
    """

  def predict_variants(
      self,
      intervals: genome.Interval | Sequence[genome.Interval],
      variants: Sequence[genome.Variant],
      conditions: Sequence[model_condition.ModelCondition | str] | None = None,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      cell_conditionings: Sequence[anndata.AnnData] | None = None,
      cell_condition_names: Sequence[str] | None = None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[dna_output.VariantOutput]:
    """Generate predictions for a batch of DNA variants.

    Args:
      intervals: DNA interval(s) to make predictions for.
      variants: DNA variants to make prediction for.
      conditions: Optional sequence of conditions or composite curies to
        generate the predictions for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      cell_conditionings: Optional sequence of expression conditioning inputs
        (`AnnData`).
      cell_condition_names: Optional sequence of labels for expression
        conditioning. Must be provided when `cell_conditionings` is used.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      Variant outputs for each DNA interval and variant pair.
    """
    _validate_not_str(conditions, 'conditions')
    _validate_not_str(cell_condition_names, 'cell_condition_names')
    num_requests = len(variants)
    intervals_seq = (
        [intervals] * num_requests
        if not isinstance(intervals, Sequence)
        else intervals
    )
    _validate_equal_lengths(
        len(intervals_seq),
        'Intervals',
        variants=variants,
        conditions=conditions,
        cell_conditionings=cell_conditionings,
        cell_condition_names=cell_condition_names,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.predict_variant,
              interval=interval,
              variant=variant,
              organism=organism,
              condition=condition,
              requested_outputs=requested_outputs,
              cell_conditioning=cell_conditioning,
              cell_condition_name=cell_condition_name,
          )
          for (
              interval,
              variant,
              condition,
              cell_conditioning,
              cell_condition_name,
          ) in zip(
              intervals_seq,
              variants,
              _pad_none(conditions, num_requests),
              _pad_none(cell_conditionings, num_requests),
              _pad_none(cell_condition_names, num_requests),
              strict=True,
          )
      ]
      futures_as_completed = tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          disable=not progress_bar,
      )
      for future in futures_as_completed:
        if (exception := future.exception()) is not None:
          executor.shutdown(wait=False, cancel_futures=True)
          raise exception

      return [future.result() for future in futures]

  @abc.abstractmethod
  def score_interval(
      self,
      interval: genome.Interval,
      condition: model_condition.ModelCondition | str | None = None,
      interval_scorers: Sequence[interval_scorers_lib.IntervalScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> list[anndata.AnnData]:
    """Generate interval scores for a single given interval.

    Args:
      interval: Interval to make prediction for.
      condition: The condition or composite curie to generate the prediction
        for.
      interval_scorers: Sequence of interval scorers to use for scoring. If no
        interval scorers are provided, the recommended interval scorers for the
        organism will be used.
      organism: Organism to use for the prediction.
      cell_conditioning: Optional expression conditioning input (`AnnData`).
      cell_condition_name: Optional label for expression conditioning. Must be
        provided when `cell_conditioning` is used.

    Returns:
      List of `AnnData` interval scores.
    """

  def score_intervals(
      self,
      intervals: Sequence[genome.Interval],
      conditions: Sequence[model_condition.ModelCondition | str] | None = None,
      interval_scorers: Sequence[interval_scorers_lib.IntervalScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      cell_conditionings: Sequence[anndata.AnnData] | None = None,
      cell_condition_names: Sequence[str] | None = None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[list[anndata.AnnData]]:
    """Generate interval scores for a sequence of intervals.

    Args:
      intervals: Sequence of DNA intervals to make prediction for.
      conditions: Optional sequence of conditions or composite curies to
        generate the prediction for.
      interval_scorers: Sequence of interval scorers to use for scoring. If no
        interval scorers are provided, the recommended interval scorers for the
        organism will be used.
      organism: Organism to use for the prediction.
      cell_conditionings: Optional sequence of expression conditioning inputs
        (`AnnData`).
      cell_condition_names: Optional sequence of labels for expression
        conditioning. Must be provided when `cell_conditionings` is used.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      List of `AnnData` lists corresponding to score_interval
      outputs for each interval.
    """
    _validate_not_str(conditions, 'conditions')
    _validate_not_str(cell_condition_names, 'cell_condition_names')
    num_requests = len(intervals)
    _validate_equal_lengths(
        num_requests,
        'Intervals',
        conditions=conditions,
        cell_conditionings=cell_conditionings,
        cell_condition_names=cell_condition_names,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.score_interval,
              interval=interval,
              condition=condition,
              interval_scorers=interval_scorers,
              organism=organism,
              cell_conditioning=cell_conditioning,
              cell_condition_name=cell_condition_name,
          )
          for interval, condition, cell_conditioning, cell_condition_name in (
              zip(
                  intervals,
                  _pad_none(conditions, num_requests),
                  _pad_none(cell_conditionings, num_requests),
                  _pad_none(cell_condition_names, num_requests),
                  strict=True,
              )
          )
      ]
      futures_as_completed = tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          disable=not progress_bar,
      )
      for future in futures_as_completed:
        if (exception := future.exception()) is not None:
          executor.shutdown(wait=False, cancel_futures=True)
          raise exception

      results = [future.result() for future in futures]
    return results

  @abc.abstractmethod
  def score_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      condition: model_condition.ModelCondition | str | None = None,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> list[anndata.AnnData]:
    """Generate variant scores for a single given DNA variant.

    Args:
      interval: DNA interval to make prediction for.
      variant: DNA variant to make prediction for.
      condition: The condition or composite curie to generate the prediction
        for.
      variant_scorers: Sequence of variant scorers to use for scoring. If no
        variant scorers are provided, the recommended variant scorers for the
        organism will be used.
      organism: Organism to use for the prediction.
      cell_conditioning: Optional expression conditioning input (`AnnData`).
      cell_condition_name: Optional label for expression conditioning. Must be
        provided when `cell_conditioning` is used.

    Returns:
      List of `AnnData` variant scores.
    """

  def score_variants(
      self,
      intervals: genome.Interval | Sequence[genome.Interval],
      variants: Sequence[genome.Variant],
      conditions: Sequence[model_condition.ModelCondition | str] | None = None,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      cell_conditionings: Sequence[anndata.AnnData] | None = None,
      cell_condition_names: Sequence[str] | None = None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[list[anndata.AnnData]]:
    """Generate variant scores for a sequence of variants.

    Args:
      intervals: DNA interval(s) to make prediction for. If a single interval is
        provided, then the same interval will be used for all variants.
      variants: Sequence of DNA variants to make prediction for.
      conditions: Optional sequence of conditions or composite curies to
        generate the predictions for.
      variant_scorers: Sequence of variant scorers to use for scoring. If no
        variant scorers are provided, the recommended variant scorers for the
        organism will be used.
      organism: Organism to use for the prediction.
      cell_conditionings: Optional sequence of expression conditioning inputs
        (`AnnData`).
      cell_condition_names: Optional sequence of labels for expression
        conditioning. Must be provided when `cell_conditionings` is used.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      List of `AnnData` lists corresponding to score_variant outputs for each
      variant.
    """
    _validate_not_str(conditions, 'conditions')
    _validate_not_str(cell_condition_names, 'cell_condition_names')
    num_requests = len(variants)
    intervals_seq = (
        [intervals] * num_requests
        if not isinstance(intervals, Sequence)
        else intervals
    )
    _validate_equal_lengths(
        len(intervals_seq),
        'Intervals',
        variants=variants,
        conditions=conditions,
        cell_conditionings=cell_conditionings,
        cell_condition_names=cell_condition_names,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.score_variant,
              interval=interval,
              variant=variant,
              condition=condition,
              variant_scorers=variant_scorers,
              organism=organism,
              cell_conditioning=cell_conditioning,
              cell_condition_name=cell_condition_name,
          )
          for (
              interval,
              variant,
              condition,
              cell_conditioning,
              cell_condition_name,
          ) in zip(
              intervals_seq,
              variants,
              _pad_none(conditions, num_requests),
              _pad_none(cell_conditionings, num_requests),
              _pad_none(cell_condition_names, num_requests),
              strict=True,
          )
      ]
      futures_as_completed = tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          disable=not progress_bar,
      )
      for future in futures_as_completed:
        if (exception := future.exception()) is not None:
          executor.shutdown(wait=False, cancel_futures=True)
          raise exception

      return [future.result() for future in futures]

  @abc.abstractmethod
  def score_ism_variants(
      self,
      interval: genome.Interval,
      ism_interval: genome.Interval,
      condition: model_condition.ModelCondition | str | None = None,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      interval_variant: genome.Variant | None = None,
      organism: Organism = Organism.HOMO_SAPIENS,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> list[list[anndata.AnnData]]:
    """Generate in silico mutagenesis (ISM) variant scores for a given interval.

    Args:
      interval: DNA interval to make the prediction for.
      ism_interval: Interval to perform ISM.
      condition: The condition or composite curie to generate the prediction
        for.
      variant_scorers: Sequence of variant scorers to use for scoring each
        variant. If no variant scorers are provided, the recommended variant
        scorers for the organism will be used.
      interval_variant: Optional variant to apply to the sequence. If provided,
        the alternate allele is used for in silico mutagenesis, otherwise the
        unaltered reference sequence is used.
      organism: Organism to use for the prediction.
      cell_conditioning: Optional expression conditioning input (`AnnData`).
      cell_condition_name: Optional label for expression conditioning. Must be
        provided when `cell_conditioning` is used.

    Returns:
      List of variant scores for each variant in the ISM interval.
    """

  @abc.abstractmethod
  def encode_cells(
      self,
      cells: anndata.AnnData,
  ) -> np.ndarray:
    """Encode cells into latent embeddings.

    Args:
      cells: Cell input with gene counts (AnnData).

    Returns:
      np.ndarray of shape (num_cells, latent_dim).
    """

  @abc.abstractmethod
  def observed_metadata(
      self, organism: Organism = Organism.HOMO_SAPIENS
  ) -> dna_output.OutputMetadata:
    """Get the metadata for tracks observed by the model for a given organism.

    Args:
      organism: Organism to get metadata for.

    Returns:
      OutputMetadata for the provided organism.
    """

  @abc.abstractmethod
  def predict_interval_gradients(
      self,
      gene_id: str,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_output: dna_output.OutputType,
      sequence_length: int,
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> dna_output.InputGradientsOutput:
    """Generate input gradients of a gene's predicted expression.

    The input interval is centered on the gene's median TSS.

    Args:
      gene_id: Ensembl ID of the target gene. Version suffixes are ignored.
      organism: Organism to use for the prediction.
      requested_output: Output type to compute gradients for. One of RNA_SEQ,
        SC_GENE_EXPRESSION or SC_GENE_EXPRESSION_DENOISED.
      sequence_length: Width of the input interval. Must be
        `dna_client.SEQUENCE_LENGTH_100KB` or `dna_client.SEQUENCE_LENGTH_1MB`.
      condition: The condition or composite curie to generate the prediction
        for.
      cell_conditioning: Optional expression conditioning input (`AnnData`).
      cell_condition_name: Optional label for expression conditioning. When
        provided, the output track names are suffixed with this name. Must be
        provided when `cell_conditioning` is used.

    Returns:
      Input gradients for the interval centered on the gene.
    """

  def predict_interval_gradients_batch(
      self,
      gene_ids: str | Sequence[str],
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_output: dna_output.OutputType,
      sequence_length: int,
      conditions: Sequence[model_condition.ModelCondition | str] | None = None,
      cell_conditionings: Sequence[anndata.AnnData] | None = None,
      cell_condition_names: Sequence[str] | None = None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[dna_output.InputGradientsOutput]:
    """Generate input gradients for a batch of genes and/or conditions.

    Args:
      gene_ids: Single Ensembl gene ID (broadcast across conditions) or a
        sequence of Ensembl gene IDs.
      organism: Organism to use for the predictions.
      requested_output: Output type to compute gradients for.
      sequence_length: Width of the input interval. Must be
        `dna_client.SEQUENCE_LENGTH_100KB` or `dna_client.SEQUENCE_LENGTH_1MB`.
      conditions: Optional sequence of conditions or composite curies.
      cell_conditionings: Optional sequence of expression conditioning inputs
        (`AnnData`).
      cell_condition_names: Optional sequence of labels for expression
        conditioning. When provided, output track names are suffixed with each
        label. Must be provided when `cell_conditionings` is used.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      List of `InputGradientsOutput` objects in request order.

    Raises:
      TypeError: If `conditions` or `cell_condition_names` is passed as a bare
        `str` instead of a sequence of strings.
      ValueError: If any provided sequence length does not match the number of
        requests.
    """
    _validate_not_str(conditions, 'conditions')
    _validate_not_str(cell_condition_names, 'cell_condition_names')

    if isinstance(gene_ids, str):
      first_seq: Sequence[Any] | None = next(
          (
              seq
              for seq in (conditions, cell_conditionings, cell_condition_names)
              if seq is not None
          ),
          None,
      )
      num_requests = len(first_seq) if first_seq is not None else 1
      gene_id_seq = [gene_ids] * num_requests
    else:
      gene_id_seq = list(gene_ids)
      num_requests = len(gene_id_seq)

    _validate_equal_lengths(
        num_requests,
        'gene_ids',
        conditions=conditions,
        cell_conditionings=cell_conditionings,
        cell_condition_names=cell_condition_names,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.predict_interval_gradients,
              gene_id=gid,
              organism=organism,
              requested_output=requested_output,
              sequence_length=sequence_length,
              condition=cond,
              cell_conditioning=cell_cond,
              cell_condition_name=cell_cond_name,
          )
          for gid, cond, cell_cond, cell_cond_name in zip(
              gene_id_seq,
              _pad_none(conditions, num_requests),
              _pad_none(cell_conditionings, num_requests),
              _pad_none(cell_condition_names, num_requests),
              strict=True,
          )
      ]
      futures_as_completed = tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          disable=not progress_bar,
      )
      for future in futures_as_completed:
        if (exception := future.exception()) is not None:
          executor.shutdown(wait=False, cancel_futures=True)
          raise exception

      return [future.result() for future in futures]
