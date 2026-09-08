# Copyright 2026 Google LLC.
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

"""Client library for querying intervals/variants from the Atlas API."""

from collections.abc import Iterable, Mapping, Sequence
import concurrent
import contextlib
import dataclasses
import importlib.resources
from typing import Any, Protocol

from alphagenome.atlas import atlas_utils
from alphagenome.data import genome
from alphagenome.data import ontology
from alphagenome.models import track_data_utils
from alphagenome.protos import dna_model_pb2
from alphagenome.protos import atlas_service_pb2
from alphagenome.protos import atlas_service_pb2_grpc
import anndata
import grpc
import numpy as np
import pandas as pd
import tqdm.auto

_LIST_DENSE_VARIANT_SCORES_FIELD_MASKS = (
    'interval',
    'next_page_token',
    'variant_scores.variant',
    'variant_scores.scores.variant_scorer',
    'variant_scores.scores.metadata.gene_scorers',
    'variant_scores.scores.shape',
    'variant_scores.scores.scores',
    'variant_scores.scores.calibrated_scores',
)

_INTERVAL_CHUNK_SIZE = 32

_GET_DENSE_VARIANT_SCORE_FIELD_MASKS = (
    'variant',
    'scores.variant_scorer',
    'scores.metadata.gene_scorers',
    'scores.shape',
    'scores.scores',
    'scores.calibrated_scores',
)

DEFAULT_MAX_WORKERS = 10


def _get_gene_metadata(
    scorer_metadata: Sequence[atlas_service_pb2.Metadata],
) -> Sequence[dict[str, str | int]] | None:
  """Extracts gene metadata from DenseVariantScore metadata."""
  for metadata in scorer_metadata:
    if metadata.HasField('gene_scorers'):
      result = []
      for gene_metadata in metadata.gene_scorers.metadata:
        row = {}
        if gene_metadata.gene_id:
          row['gene_id'] = gene_metadata.gene_id
        if gene_metadata.HasField('name'):
          row['gene_name'] = gene_metadata.name
        if gene_metadata.HasField('strand'):
          row['strand'] = str(genome.Strand.from_proto(gene_metadata.strand))
        if gene_metadata.HasField('junction_start'):
          row['junction_Start'] = gene_metadata.junction_start
        if gene_metadata.HasField('junction_end'):
          row['junction_End'] = gene_metadata.junction_end
        result.append(row)
      return result
  return None


def _get_track_metadata(
    scorer_metadata: Sequence[atlas_service_pb2.Metadata],
) -> pd.DataFrame | None:
  """Extracts track metadata from DenseVariantScore metadata."""
  for metadata in scorer_metadata:
    if metadata.HasField('tracks'):
      return track_data_utils.metadata_from_proto(metadata.tracks)
  return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScorerMetadata:
  """Metadata related to a variant scorer.

  Attributes:
    name: The name of the scorer.
    is_signed: Whether the scorer is signed (i.e., can produce negative scores).
    track_metadata: A DataFrame containing the track metadata for the scorer.
  """

  name: str
  is_signed: bool
  track_metadata: pd.DataFrame


def _filter_scorer_metadata(
    scorer_metadata: Mapping[str, ScorerMetadata],
    ontology_terms: Iterable[ontology.OntologyTerm | str] | None = None,
) -> Mapping[str, pd.DataFrame]:
  """Filters scorer metadata to specific ontology terms."""
  ontology_curies = None
  if ontology_terms is not None:
    ontology_curies = [
        ontology.from_curie(o).ontology_curie
        if isinstance(o, str)
        else o.ontology_curie
        for o in ontology_terms
    ]
  track_metadata_by_scorer = {}
  for name, metadata in scorer_metadata.items():
    track_metadata = metadata.track_metadata
    if ontology_curies is not None and 'ontology_curie' in track_metadata:
      track_metadata = track_metadata[
          track_metadata['ontology_curie'].isin(ontology_curies)
      ]
    track_metadata_by_scorer[name] = track_metadata
  return track_metadata_by_scorer


@contextlib.contextmanager
def handle_rpc_error():
  """Context manager for converting gRPC exceptions to Python exceptions."""
  try:
    yield
  except grpc.RpcError as error:
    assert isinstance(error, grpc.Call)
    match error.code():
      case grpc.StatusCode.INVALID_ARGUMENT | grpc.StatusCode.NOT_FOUND:
        raise ValueError(error.details()) from error
      case grpc.StatusCode.UNAUTHENTICATED:
        raise PermissionError(error.details()) from error
      case grpc.StatusCode.DEADLINE_EXCEEDED:
        raise TimeoutError(error.details()) from error
      case grpc.StatusCode.UNIMPLEMENTED:
        raise NotImplementedError(error.details()) from error
      case grpc.StatusCode.OUT_OF_RANGE:
        raise IndexError(error.details()) from error
      case grpc.StatusCode.PERMISSION_DENIED:
        raise PermissionError(error.details()) from error
      case _:
        raise error


def convert_variant_scores_to_anndata(
    variant_scores: Iterable[atlas_service_pb2.DenseVariantScores],
    scorer_track_metadata: Mapping[str, pd.DataFrame] | None,
) -> Mapping[str, anndata.AnnData]:
  """Converts a list of DenseVariantScores protos to an AnnData per scorer."""

  scores_by_scorer = {}
  calibrated_scores_by_scorer = {}
  obs_by_scorer = {}
  var_by_scorer = {}

  for scores in variant_scores:
    variant = genome.Variant.from_proto(scores.variant)

    for score in scores.scores:
      x = np.frombuffer(score.scores, dtype=np.float32).reshape(score.shape)

      scorer_name = score.variant_scorer.name
      scores_by_scorer.setdefault(scorer_name, []).append(x)

      if score.calibrated_scores:
        calibrated_scores = np.frombuffer(
            score.calibrated_scores, dtype=np.float32
        ).reshape(score.shape)
        calibrated_scores_by_scorer.setdefault(scorer_name, []).append(
            calibrated_scores
        )

      gene_metadata: Sequence[dict[str, Any]] | None = _get_gene_metadata(
          score.metadata
      )
      obs_by_scorer.setdefault(scorer_name, [])
      if gene_metadata is not None:
        for gene_row in gene_metadata:
          gene_row['variant'] = variant
        obs_by_scorer[scorer_name].extend(gene_metadata)
      elif x.shape[0] == 1:
        obs_by_scorer[scorer_name].append({'variant': variant})

      if scorer_name not in var_by_scorer:
        if scorer_track_metadata is not None:
          var_by_scorer[scorer_name] = scorer_track_metadata[scorer_name]
        elif (
            track_metadata := _get_track_metadata(score.metadata)
        ) is not None:
          var_by_scorer[scorer_name] = track_metadata

  results = {}
  for scorer_name, scorer_scores in scores_by_scorer.items():
    obs = None
    if scorer_obs := obs_by_scorer[scorer_name]:
      obs = pd.DataFrame(
          scorer_obs,
          index=[str(i) for i in range(len(scorer_obs))],
      )
    scores = np.concatenate(scorer_scores)
    layers = None
    if (
        calibrated_scores := calibrated_scores_by_scorer.get(scorer_name)
    ) is not None:
      layers = {'quantiles': np.concatenate(calibrated_scores)}
    var = var_by_scorer[scorer_name]
    var.index = var.index.map(str)
    results[scorer_name] = anndata.AnnData(
        X=scores, obs=obs, var=var, layers=layers
    )

  return results


class AtlasServiceStubType(Protocol):
  """Protocol for the AtlasService stub."""

  def ListDenseVariantScores(  # pylint: disable=invalid-name
      self,
      request: atlas_service_pb2.ListDenseVariantScoresRequest,
      metadata: Sequence[tuple[str, Any]] | None = None,
  ) -> atlas_service_pb2.ListDenseVariantScoresResponse:
    """Lists dense variant scores for an interval."""

  def GetDenseVariantScores(  # pylint: disable=invalid-name
      self,
      request: atlas_service_pb2.GetDenseVariantScoresRequest,
      metadata: Sequence[tuple[str, Any]] | None = None,
  ) -> atlas_service_pb2.DenseVariantScores:
    """Gets dense variant scores for a single variant."""

  def ListVariantScoresMetadata(  # pylint: disable=invalid-name
      self,
      request: atlas_service_pb2.ListVariantScoresMetadataRequest,
      metadata: Sequence[tuple[str, Any]] | None = None,
  ) -> atlas_service_pb2.ListVariantScoresMetadataResponse:
    """Lists variant scores metadata for all scorers."""


class AtlasClient:
  """Client for interacting with AlphaGenome Atlas scores."""

  def __init__(
      self,
      stub: AtlasServiceStubType,
      metadata: Sequence[tuple[str, str]] = (),
  ):
    self._stub = stub
    self._metadata = metadata

  def query_variant(
      self,
      variant: genome.Variant,
      *,
      requested_scorers: Iterable[str],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None = None,
      gene_ids: Iterable[str] | None = None,
      gene_names: Iterable[str] | None = None,
  ) -> Mapping[str, anndata.AnnData]:
    """Returns the scores for a single Variant within an Interval."""
    request = atlas_service_pb2.GetDenseVariantScoresRequest(
        variant=variant.to_proto(),
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        filter=atlas_utils.build_filter(
            requested_scorers=requested_scorers,
            ontology_terms=ontology_terms,
            gene_ids=gene_ids,
            gene_names=gene_names,
        ),
    )
    with handle_rpc_error():
      response = self._stub.GetDenseVariantScores(
          request, metadata=self._metadata
      )
    return convert_variant_scores_to_anndata([response], None)

  def query_variants(
      self,
      variants: Sequence[genome.Variant],
      *,
      requested_scorers: Iterable[str],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None = None,
      gene_ids: Iterable[str] | None = None,
      gene_names: Iterable[str] | None = None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> Mapping[str, anndata.AnnData]:
    """Returns the scores for a single Variant within an Interval."""

    filter_query = atlas_utils.build_filter(
        requested_scorers=requested_scorers,
        ontology_terms=ontology_terms,
        gene_ids=gene_ids,
        gene_names=gene_names,
    )

    field_mask = ','.join(_GET_DENSE_VARIANT_SCORE_FIELD_MASKS)
    metadata = [*self._metadata, ('x-goog-fieldmask', field_mask)]

    def _query_variant(
        variant: genome.Variant,
    ) -> atlas_service_pb2.DenseVariantScores:
      request = atlas_service_pb2.GetDenseVariantScoresRequest(
          variant=variant.to_proto(),
          organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
          filter=filter_query,
      )
      with handle_rpc_error():
        return self._stub.GetDenseVariantScores(request, metadata=metadata)

    scores = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(_query_variant, variant) for variant in variants
      ]

      scorer_metadata = _filter_scorer_metadata(
          self.scorer_metadata(), ontology_terms
      )

      for future in tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          unit='variants',
          disable=not progress_bar,
      ):
        scores.append(future.result())

    return convert_variant_scores_to_anndata(scores, scorer_metadata)

  def query_interval(
      self,
      interval: genome.Interval,
      *,
      requested_scorers: Iterable[str],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None = None,
      gene_ids: Iterable[str] | None = None,
      gene_names: Iterable[str] | None = None,
      progress_bar: bool = True,
      max_workers: int = DEFAULT_MAX_WORKERS,
  ) -> Mapping[str, anndata.AnnData]:
    """Returns the scores for all Variants within an Interval."""

    intervals = []
    for position in range(0, interval.width, _INTERVAL_CHUNK_SIZE):
      intervals.append(
          genome.Interval(
              interval.chromosome,
              interval.start + position,
              min(
                  interval.start + position + _INTERVAL_CHUNK_SIZE, interval.end
              ),
          )
      )

    filter_query = atlas_utils.build_filter(
        requested_scorers=requested_scorers,
        ontology_terms=ontology_terms,
        gene_ids=gene_ids,
        gene_names=gene_names,
    )

    field_mask = ','.join(_LIST_DENSE_VARIANT_SCORES_FIELD_MASKS)
    metadata = [*self._metadata, ('x-goog-fieldmask', field_mask)]

    def _query_sub_interval(
        sub_interval: genome.Interval,
    ) -> Sequence[atlas_service_pb2.DenseVariantScores]:
      request = atlas_service_pb2.ListDenseVariantScoresRequest(
          interval=sub_interval.to_proto(),
          organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
          filter=filter_query,
      )
      scores = []
      while True:
        with handle_rpc_error():
          response = self._stub.ListDenseVariantScores(
              request, metadata=metadata
          )
        scores.extend(response.variant_scores)
        if not response.next_page_token:
          return scores
        request.page_token = response.next_page_token

    scores = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(_query_sub_interval, sub_interval)
          for sub_interval in intervals
      ]

      scorer_metadata = _filter_scorer_metadata(
          self.scorer_metadata(), ontology_terms
      )
      progress = tqdm.auto.tqdm(
          total=interval.width * 3, unit='variants', disable=not progress_bar
      )
      for sub_interval, future in zip(
          intervals, concurrent.futures.as_completed(futures), strict=True
      ):
        scores.extend(future.result())
        progress.update(sub_interval.width * 3)

    return convert_variant_scores_to_anndata(scores, scorer_metadata)

  def scorer_metadata(self) -> Mapping[str, ScorerMetadata]:
    """Returns the metadata for all scorers."""
    with handle_rpc_error():
      response = self._stub.ListVariantScoresMetadata(
          atlas_service_pb2.ListVariantScoresMetadataRequest(
              organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS
          ),
          metadata=self._metadata,
      )
    metadata = {}
    for scorer_metadata in response.variant_scorer_metadata:
      track_metadata = _get_track_metadata(scorer_metadata.metadata)
      if track_metadata is not None:
        track_metadata.index = track_metadata.index.map(str)
      else:
        track_metadata = pd.DataFrame()
      metadata[scorer_metadata.variant_scorer.name] = ScorerMetadata(
          name=scorer_metadata.variant_scorer.name,
          is_signed=scorer_metadata.variant_scorer.is_signed,
          track_metadata=track_metadata,
      )
    return metadata


def create(
    api_key: str,
    *,
    timeout: float | None = None,
    address: str | None = None,
) -> AtlasClient:
  """Creates an AlphaGenome Atlas client for a given API key.

  Args:
    api_key: API key to use for authentication.
    timeout: Optional timeout to wait for the channel to be ready.
    address: Optional server address to connect to.

  Returns:
   `DnaClient` instance.
  """
  address = address or 'dns:///gdmscience.googleapis.com:443'
  service_config = (
      importlib.resources.files('alphagenome')
      / 'protos/grpc_service_config.json'
  ).read_text()

  channel = grpc.secure_channel(
      address,
      grpc.ssl_channel_credentials(),
      options=(('grpc.service_config', service_config),),
  )

  grpc.channel_ready_future(channel).result(timeout)

  stub = atlas_service_pb2_grpc.AtlasServiceStub(channel=channel)
  return AtlasClient(stub, metadata=[('x-goog-api-key', api_key)])
