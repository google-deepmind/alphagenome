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

"""Abstract base class for AlphaGenome DNA models."""

import abc
import concurrent.futures
import enum
from collections.abc import Iterable, Sequence

import anndata
import tqdm.auto

from alphagenome.data import genome, ontology
from alphagenome.models import dna_output
from alphagenome.models import interval_scorers as interval_scorers_lib
from alphagenome.models import variant_scorers as variant_scorers_lib
from alphagenome.protos import dna_model_pb2

# Default maximum number of workers to use for parallel requests.
DEFAULT_MAX_WORKERS = 5


@enum.unique
class ModelVersion(enum.Enum):
  """Enumeration of all available model versions.

  A fold is a part of the genome that is held out from training, and is used for
  model validation.

  Folds are numbered from 0 to 3, and represent disjoint subsets of the genome.

  A model version that is designated FOLD_#, where # is an integer from 0 to 3,
  indicates that the model was trained with that particular fold held out.

  The ALL_FOLDS version refers to the distilled model that was trained on
  an ensemble of teacher models trained on all folds.
  """

  ALL_FOLDS = enum.auto()  # Default model version if None explicitly set.
  FOLD_0 = enum.auto()
  FOLD_1 = enum.auto()
  FOLD_2 = enum.auto()
  FOLD_3 = enum.auto()


class Organism(enum.Enum):
  """Enumeration of all the available organisms."""

  HOMO_SAPIENS = dna_model_pb2.ORGANISM_HOMO_SAPIENS
  MUS_MUSCULUS = dna_model_pb2.ORGANISM_MUS_MUSCULUS

  def to_proto(self) -> dna_model_pb2.Organism:
    return self.value

  def __lt__(self, other: 'Organism') -> bool:
    if not isinstance(other, Organism):
      return NotImplemented
    return self.value < other.value


class DnaModel(metaclass=abc.ABCMeta):
  """Abstract base class for AlphaGenome DNA models."""

  @abc.abstractmethod
  def predict_sequence(
      self,
      sequence: str,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
      interval: genome.Interval | None = None,
  ) -> dna_output.Output:
    """Generate predictions for a given DNA sequence.

    Args:
      sequence: DNA sequence to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      ontology_terms: Iterable of ontology terms or curies to generate
        predictions for. If None returns all ontologies.
      interval: Optional interval from which the sequence was derived. This is
        used as the interval in the output TrackData.

    Returns:
      Output for the provided DNA sequence.
    """

  def predict_sequences(
      self,
      sequences: Sequence[str],
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
      intervals: Sequence[genome.Interval] | None = None,
  ) -> list[dna_output.Output]:
    """Generate predictions for a given DNA sequence.

    Args:
      sequences: DNA sequences to make predictions for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      ontology_terms: Iterable of ontology terms or curies to generate
        predictions for. If None returns all ontologies.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.
      intervals: Optional intervals from which the sequences were derived. This
        is used as the interval in the output TrackData. Must be the same length
        as `sequences` if provided.

    Returns:
      Outputs for the provided DNA sequences.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.predict_sequence,
              sequence=sequence,
              organism=organism,
              ontology_terms=ontology_terms,
              requested_outputs=requested_outputs,
              interval=interval,
          )
          for sequence, interval in zip(
              sequences, intervals or [None] * len(sequences), strict=True
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
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
  ) -> dna_output.Output:
    """Generate predictions for a given DNA interval.

    Args:
      interval: DNA interval to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      ontology_terms: Iterable of ontology terms or curies to generate
        predictions for. If None returns all ontologies.

    Returns:
      Output for the provided DNA interval.
    """

  def predict_intervals(
      self,
      intervals: Sequence[genome.Interval],
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[dna_output.Output]:
    """Generate predictions for a sequence of DNA intervals.

    Args:
      intervals: DNA intervals to make predictions for.
      organism: Organism to use for the predictions.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      ontology_terms: Iterable of ontology terms or curies to generate
        predictions for. If None returns all ontologies.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      Outputs for the provided DNA intervals.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.predict_interval,
              interval=interval,
              organism=organism,
              ontology_terms=ontology_terms,
              requested_outputs=requested_outputs,
          )
          for interval in intervals
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
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
  ) -> dna_output.VariantOutput:
    """Generate predictions for a given DNA variant.

    Args:
      interval: DNA interval to make prediction for.
      variant: DNA variant to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      ontology_terms: Iterable of ontology terms or curies to generate
        predictions for. If None returns all ontologies.

    Returns:
      Variant output for the provided DNA interval and variant.
    """

  def predict_variants(
      self,
      intervals: genome.Interval | Sequence[genome.Interval],
      variants: Sequence[genome.Variant],
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[dna_output.VariantOutput]:
    """Generate predictions for a given DNA variant.

    Args:
      intervals: DNA interval(s) to make predictions for.
      variants: DNA variants to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      ontology_terms: Iterable of ontology terms or curies to generate
        predictions for. If None returns all ontologies.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      Variant outputs for each DNA interval and variant pair.
    """
    if not isinstance(intervals, Sequence):
      intervals = [intervals] * len(variants)
    elif len(intervals) != len(variants):
      raise ValueError(
          'Intervals and variants must have the same length.'
          f'Got {len(intervals)} intervals and {len(variants)} variants.'
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
              ontology_terms=ontology_terms,
              requested_outputs=requested_outputs,
          )
          for interval, variant in zip(intervals, variants, strict=True)
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
      interval_scorers: Sequence[interval_scorers_lib.IntervalScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
  ) -> list[anndata.AnnData]:
    """Generate interval scores for a single given interval.

    Args:
      interval: Interval to make prediction for.
      interval_scorers: Sequence of interval scorers to use for scoring. If no
        interval scorers are provided, the recommended interval scorers for the
        organism will be used.
      organism: Organism to use for the prediction.

    Returns:
      List of `AnnData` interval scores.
    """

  def score_intervals(
      self,
      intervals: Sequence[genome.Interval],
      interval_scorers: Sequence[interval_scorers_lib.IntervalScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[list[anndata.AnnData]]:
    """Generate interval scores for a sequence of intervals.

    Args:
      intervals: Sequence of DNA intervals to make prediction for.
      interval_scorers: Sequence of interval scorers to use for scoring. If no
        interval scorers are provided, the recommended interval scorers for the
        organism will be used.
      organism: Organism to use for the prediction.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      List of `AnnData` lists corresponding to score_interval
      outputs for each interval.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.score_interval,
              interval=interval,
              interval_scorers=interval_scorers,
              organism=organism,
          )
          for interval in intervals
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
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
  ) -> list[anndata.AnnData]:
    """Generate variant scores for a single given DNA variant.

    Args:
      interval: DNA interval to make prediction for.
      variant: DNA variant to make prediction for.
      variant_scorers: Sequence of variant scorers to use for scoring. If no
        variant scorers are provided, the recommended variant scorers for the
        organism will be used.
      organism: Organism to use for the prediction.

    Returns:
      List of `AnnData` variant scores.
    """

  def score_variants(
      self,
      intervals: genome.Interval | Sequence[genome.Interval],
      variants: Sequence[genome.Variant],
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> list[list[anndata.AnnData]]:
    """Generate variant scores for a sequence of variants.

    Args:
      intervals: DNA interval(s) to make prediction for. If a single interval is
        provided, then the same interval will be used for all variants.
      variants: Sequence of DNA variants to make prediction for.
      variant_scorers: Sequence of variant scorers to use for scoring. If no
        variant scorers are provided, the recommended variant scorers for the
        organism will be used.
      organism: Organism to use for the prediction.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      List of `AnnData` lists corresponding to score_variant outputs for each
      variant.
    """
    if not isinstance(intervals, Sequence):
      intervals = [intervals] * len(variants)
    if len(intervals) != len(variants):
      raise ValueError(
          'Intervals and variants must have the same length.'
          f'Got {len(intervals)} intervals and {len(variants)} variants.'
      )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(
              self.score_variant,
              interval=interval,
              variant=variant,
              variant_scorers=variant_scorers,
              organism=organism,
          )
          for interval, variant in zip(intervals, variants, strict=True)
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
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      interval_variant: genome.Variant | None = None,
      organism: Organism = Organism.HOMO_SAPIENS,
  ) -> list[list[anndata.AnnData]]:
    """Generate in-silico mutagenesis (ISM) variant scores for a given interval.

    Args:
      interval: DNA interval to make the prediction for.
      ism_interval: Interval to perform ISM.
      variant_scorers: Sequence of variant scorers to use for scoring each
        variant. If no variant scorers are provided, the recommended variant
        scorers for the organism will be used.
      interval_variant: Optional variant to apply to the sequence. If provided,
        the alternate allele is used for in-silico mutagenesis, otherwise the
        unaltered reference sequence is used.
      organism: Organism to use for the prediction.

    Returns:
      List of variant scores for each variant in the ISM interval.
    """

  @abc.abstractmethod
  def output_metadata(
      self, organism: Organism = Organism.HOMO_SAPIENS
  ) -> dna_output.OutputMetadata:
    """Get the metadata for a given organism.

    Args:
      organism: Organism to get metadata for.

    Returns:
      OutputMetadata for the provided organism.
    """
