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

"""Client implementation for interacting with a DNA model server."""

from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
import concurrent.futures
import functools
import logging
import random
import time
from typing import TypeVar

from alphagenome import cache_utils
from alphagenome import tensor_utils
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.data import ontology
from alphagenome.data import track_data
from alphagenome.models import dna_model
from alphagenome.models import dna_output
from alphagenome.models import interval_scorers as interval_scorers_lib
from alphagenome.models import junction_data_utils
from alphagenome.models import track_data_utils
from alphagenome.models import variant_scorers as variant_scorers_lib
from alphagenome.protos import dna_model_pb2
from alphagenome.protos import dna_model_service_pb2
from alphagenome.protos import dna_model_service_pb2_grpc
from alphagenome.protos import tensor_pb2
import anndata
import grpc
import numpy as np
import pandas as pd
import tqdm.auto


# Supported DNA sequence lengths.
SEQUENCE_LENGTH_16KB = 2**14  # 16_384
SEQUENCE_LENGTH_100KB = 2**17  # 131_072
SEQUENCE_LENGTH_500KB = 2**19  # 524_288
SEQUENCE_LENGTH_1MB = 2**20  # 1_048_576

SUPPORTED_SEQUENCE_LENGTHS = {
    'SEQUENCE_LENGTH_16KB': SEQUENCE_LENGTH_16KB,
    'SEQUENCE_LENGTH_100KB': SEQUENCE_LENGTH_100KB,
    'SEQUENCE_LENGTH_500KB': SEQUENCE_LENGTH_500KB,
    'SEQUENCE_LENGTH_1MB': SEQUENCE_LENGTH_1MB,
}

# Maximum width of an in silico mutagenesis (ISM) interval request.
# This controls the behavior of `score_ism_variants` in the client.
# When a user inputs an ISM interval that is wider than this value,
# the client will automatically split the interval into chunks of this width.
MAX_ISM_INTERVAL_WIDTH = 10

# Maximum number of variant scorers per request.
MAX_VARIANT_SCORERS_PER_REQUEST = 20

_VALID_SEQUENCE_CHARACTERS = frozenset('ACGTN')

RetryableFunction = TypeVar('RetryableFunction')


def retry_rpc(
    function: RetryableFunction,
    *,
    max_attempts: int = 5,
    initial_backoff: float = 1.25,
    backoff_multiplier: float = 1.5,
    retry_status_codes: Container[grpc.StatusCode] = frozenset(
        [grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.UNAVAILABLE]
    ),
    jitter: float = 0.2,
) -> RetryableFunction:
  """Decorator that retries when an RPC fails.

  gRPC currently doesn't support retries for streaming RPCs. This decorator is
  an implementation of the retry logic from https://grpc.io/docs/guides/retry.

  Arguments:
    function: Callable to retry.
    max_attempts: Maximum number of attempts to make.
    initial_backoff: Initial backoff time in seconds.
    backoff_multiplier: Backoff multiplier to apply on each retry.
    retry_status_codes: Set of status codes to retry on.
    jitter: Jitter to apply to the backoff time. For example, a jitter of 0.2
      and an initial backoff of 1.5 will result in a backoff time between 1.2
      and 1.8 seconds.

  Returns:
    A decorator that retries the function on retryable RPC errors.
  """
  if backoff_multiplier < 1.0:
    raise ValueError('Backoff multiplier must be >= 1.0.')

  @functools.wraps(function)
  def wrapper(*args, **kwargs):
    attempt = 0
    current_backoff = initial_backoff + (
        random.uniform(-jitter, jitter) * initial_backoff
    )
    while True:
      try:
        attempt += 1
        return function(*args, **kwargs)
      except grpc.RpcError as e:
        error_code = e.code()  # pytype: disable=attribute-error
        if error_code not in retry_status_codes or attempt >= max_attempts:
          raise e
        time.sleep(current_backoff)
        current_backoff *= backoff_multiplier

  return wrapper


ModelVersion = dna_model.ModelVersion
Output = dna_output.Output
OutputMetadata = dna_output.OutputMetadata
OutputType = dna_output.OutputType
VariantOutput = dna_output.VariantOutput
Organism = dna_model.Organism

PredictResponse = TypeVar(
    'PredictResponse',
    dna_model_service_pb2.PredictSequenceResponse,
    dna_model_service_pb2.PredictIntervalResponse,
)


def _read_tensor_chunks(
    responses: Iterator[
        PredictResponse
        | dna_model_service_pb2.PredictVariantResponse
        | dna_model_service_pb2.ScoreVariantResponse
        | dna_model_service_pb2.ScoreIsmVariantResponse
    ],
    chunk_count: int,
) -> Iterable[tensor_pb2.TensorChunk]:
  """Helper to read a sequence of tensor chunks from a response iterator."""
  for _ in range(chunk_count):
    try:
      response = next(responses)
    except StopIteration as e:
      raise ValueError('Expected tensor_chunk, got end of stream') from e
    if response.WhichOneof('payload') != 'tensor_chunk':
      raise ValueError(
          f'Expected tensor_chunk, got "{response.WhichOneof("payload")}"'
          ' payload'
      )
    yield response.tensor_chunk


def _make_output_data(
    output: dna_model_pb2.Output,
    responses: Iterator[
        PredictResponse | dna_model_service_pb2.PredictVariantResponse
    ],
    interval: genome.Interval | None = None,
) -> track_data.TrackData | np.ndarray | junction_data.JunctionData:
  """Helper to create an output data object from an output proto."""
  match output.WhichOneof('payload'):
    case 'track_data':
      return track_data_utils.from_protos(
          output.track_data,
          _read_tensor_chunks(responses, output.track_data.values.chunk_count),
          interval=interval,
      )
    case 'data':
      chunks = _read_tensor_chunks(responses, output.data.chunk_count)
      values = tensor_utils.unpack_proto(output.data, chunks)
      return tensor_utils.upcast_floating(values)
    case 'junction_data':
      return junction_data_utils.from_protos(
          output.junction_data,
          _read_tensor_chunks(
              responses, output.junction_data.values.chunk_count
          ),
          interval=interval,
      )
    case _:
      raise ValueError(
          f'Unsupported output type: {output.WhichOneof("payload")}'
      )


def construct_output_metadata(
    responses: Iterator[dna_model_service_pb2.MetadataResponse],
) -> OutputMetadata:
  """Constructs an OutputMetadata from a stream of responses."""
  metadata = {}
  for response in responses:
    for metadata_proto in response.output_metadata:
      match metadata_proto.WhichOneof('payload'):
        case 'tracks':
          metadata[metadata_proto.output_type] = (
              track_data_utils.metadata_from_proto(metadata_proto.tracks)
          )
        case 'junctions':
          metadata[metadata_proto.output_type] = (
              junction_data_utils.metadata_from_proto(metadata_proto.junctions)
          )
        case _:
          raise ValueError(
              'Unsupported metadata type:'
              f' {metadata_proto.WhichOneof("payload")}'
          )

  return OutputMetadata(
      atac=metadata.get(dna_model_pb2.OUTPUT_TYPE_ATAC),
      cage=metadata.get(dna_model_pb2.OUTPUT_TYPE_CAGE),
      dnase=metadata.get(dna_model_pb2.OUTPUT_TYPE_DNASE),
      rna_seq=metadata.get(dna_model_pb2.OUTPUT_TYPE_RNA_SEQ),
      chip_histone=metadata.get(dna_model_pb2.OUTPUT_TYPE_CHIP_HISTONE),
      chip_tf=metadata.get(dna_model_pb2.OUTPUT_TYPE_CHIP_TF),
      splice_sites=metadata.get(dna_model_pb2.OUTPUT_TYPE_SPLICE_SITES),
      splice_site_usage=metadata.get(
          dna_model_pb2.OUTPUT_TYPE_SPLICE_SITE_USAGE
      ),
      splice_junctions=metadata.get(dna_model_pb2.OUTPUT_TYPE_SPLICE_JUNCTIONS),
      contact_maps=metadata.get(dna_model_pb2.OUTPUT_TYPE_CONTACT_MAPS),
      procap=metadata.get(dna_model_pb2.OUTPUT_TYPE_PROCAP),
  )


def _construct_output(
    output_dict: Mapping[
        dna_model_pb2.OutputType,
        track_data.TrackData | np.ndarray | junction_data.JunctionData,
    ],
):
  """Helper to construct an Output dataclass from a mapping of output types."""
  output = Output(
      atac=output_dict.get(dna_model_pb2.OUTPUT_TYPE_ATAC),
      cage=output_dict.get(dna_model_pb2.OUTPUT_TYPE_CAGE),
      dnase=output_dict.get(dna_model_pb2.OUTPUT_TYPE_DNASE),
      rna_seq=output_dict.get(dna_model_pb2.OUTPUT_TYPE_RNA_SEQ),
      chip_histone=output_dict.get(dna_model_pb2.OUTPUT_TYPE_CHIP_HISTONE),
      chip_tf=output_dict.get(dna_model_pb2.OUTPUT_TYPE_CHIP_TF),
      splice_sites=output_dict.get(dna_model_pb2.OUTPUT_TYPE_SPLICE_SITES),
      splice_site_usage=output_dict.get(
          dna_model_pb2.OUTPUT_TYPE_SPLICE_SITE_USAGE
      ),
      splice_junctions=output_dict.get(
          dna_model_pb2.OUTPUT_TYPE_SPLICE_JUNCTIONS
      ),
      contact_maps=output_dict.get(dna_model_pb2.OUTPUT_TYPE_CONTACT_MAPS),
      procap=output_dict.get(dna_model_pb2.OUTPUT_TYPE_PROCAP),
  )
  return output


def _make_output(
    responses: Iterator[PredictResponse],
    *,
    interval: genome.Interval | None = None,
) -> Output:
  """Helper to load an Output dataclass from a stream of response protos."""
  outputs = {}

  for response in responses:
    match response.WhichOneof('payload'):
      case 'output':
        outputs[response.output.output_type] = _make_output_data(
            response.output, responses, interval=interval
        )
      case 'tensor_chunk':
        raise ValueError('Received tensor chunk before output proto.')
      case _:
        raise ValueError(
            f'Unsupported response type: {response.WhichOneof("payload")}'
        )

  return _construct_output(outputs)


def _make_variant_output(
    responses: Iterator[dna_model_service_pb2.PredictVariantResponse],
) -> VariantOutput:
  """Helper to load an Output dataclass from a stream of response protos."""
  ref_outputs = {}
  alt_outputs = {}

  for response in responses:
    match response.WhichOneof('payload'):
      case 'reference_output':
        ref_outputs[response.reference_output.output_type] = _make_output_data(
            response.reference_output, responses
        )
      case 'alternate_output':
        alt_outputs[response.alternate_output.output_type] = _make_output_data(
            response.alternate_output, responses
        )
      case 'tensor_chunk':
        raise ValueError('Received tensor chunk before output proto.')
      case _:
        raise ValueError(
            f'Unsupported response type: {response.WhichOneof("payload")}'
        )

  return VariantOutput(
      reference=_construct_output(ref_outputs),
      alternate=_construct_output(alt_outputs),
  )


def _construct_anndata_from_proto(
    scorer_metadata: Sequence[dna_model_pb2.GeneScorerMetadata] | None,
    track_metadata: Sequence[dna_model_pb2.TrackMetadata],
    values: np.ndarray,
    interval: genome.Interval,
    variant: dna_model_pb2.Variant | None = None,
) -> anndata.AnnData:
  """Helper to construct `AnnData` from a scoring proto."""
  obs = None
  if scorer_metadata:
    metadata = []
    for gene_proto in scorer_metadata:
      scorer_metadata = {
          'gene_id': gene_proto.gene_id,
      }
      if gene_proto.HasField('strand'):
        scorer_metadata['strand'] = str(
            genome.Strand.from_proto(gene_proto.strand)
        )

      if gene_proto.HasField('name'):
        scorer_metadata['gene_name'] = gene_proto.name
      if gene_proto.HasField('type'):
        scorer_metadata['gene_type'] = gene_proto.type
      if gene_proto.HasField('junction_start'):
        scorer_metadata['junction_Start'] = gene_proto.junction_start
      if gene_proto.HasField('junction_end'):
        scorer_metadata['junction_End'] = gene_proto.junction_end

      metadata.append(scorer_metadata)

    obs = pd.DataFrame(metadata)
    obs.index = obs.index.map(str)

  var = track_data_utils.metadata_from_proto(
      dna_model_pb2.TracksMetadata(metadata=track_metadata)
  )
  var.index = var.index.map(str)

  uns = {'interval': interval}
  if variant is not None:
    uns['variant'] = genome.Variant.from_proto(variant)
  layers = None
  if values.shape[0] > 1:
    layers = {'quantiles': values[1]}

  return anndata.AnnData(
      X=values[0],
      obs=obs,
      var=var,
      uns=uns,
      layers=layers,
  )


def _construct_score_variant(
    output: dna_model_pb2.ScoreVariantOutput,
    responses: Iterator[
        dna_model_service_pb2.ScoreVariantResponse
        | dna_model_service_pb2.ScoreIsmVariantResponse
    ],
    interval: genome.Interval,
) -> anndata.AnnData:
  """Returns `AnnData`  of variant scores from a ScoreVariantOutput proto."""
  # dna_model_pb2.ScoreVariantOutput currently has ONLY VariantData with values
  # and metadata.
  chunks = _read_tensor_chunks(
      responses, output.variant_data.values.chunk_count
  )
  values = tensor_utils.unpack_proto(output.variant_data.values, chunks)
  values = tensor_utils.upcast_floating(values)
  anndata_scores = _construct_anndata_from_proto(
      output.variant_data.metadata.gene_metadata,
      output.variant_data.metadata.track_metadata,
      values,
      interval=interval,
      variant=output.variant_data.metadata.variant,
  )
  return anndata_scores


def _make_score_variant_output(
    responses: Iterator[
        dna_model_service_pb2.ScoreVariantResponse
        | dna_model_service_pb2.ScoreIsmVariantResponse
    ],
    interval: genome.Interval,
) -> list[anndata.AnnData]:
  """Returns a list of `AnnData` variant scores from a stream of responses."""

  anndata_scores = []
  for response in responses:
    match response.WhichOneof('payload'):
      case 'output':
        anndata_scores.append(
            _construct_score_variant(
                response.output, responses, interval=interval
            )
        )
      case 'tensor_chunk':
        raise ValueError('Received tensor chunk before output proto.')
      case _:
        raise ValueError(
            f'Unsupported response type: {response.WhichOneof("payload")}'
        )
  return anndata_scores


def _construct_score_interval(
    output: dna_model_pb2.ScoreIntervalOutput,
    responses: Iterator[dna_model_service_pb2.ScoreIntervalResponse],
    interval: genome.Interval,
) -> anndata.AnnData:
  """Returns `AnnData` of interval scores from a ScoreIntervalOutput proto."""
  chunks = _read_tensor_chunks(
      responses, output.interval_data.values.chunk_count
  )
  values = tensor_utils.unpack_proto(output.interval_data.values, chunks)
  values = tensor_utils.upcast_floating(values)
  anndata_scores = _construct_anndata_from_proto(
      output.interval_data.metadata.gene_metadata,
      output.interval_data.metadata.track_metadata,
      values,
      interval=interval,
  )
  return anndata_scores


def _make_interval_output(
    responses: Iterator[dna_model_service_pb2.ScoreIntervalResponse],
    interval: genome.Interval,
) -> list[anndata.AnnData]:
  """Returns a list of `AnnData` interval scores from a stream of responses."""

  anndata_scores = []
  for response in responses:
    match response.WhichOneof('payload'):
      case 'output':
        anndata_scores.append(
            _construct_score_interval(
                response.output,
                responses,
                interval=interval,
            )
        )
      case 'tensor_chunk':
        raise ValueError('Received tensor chunk before output proto.')
      case _:
        raise ValueError(
            f'Unsupported response type: {response.WhichOneof("payload")}'
        )
  return anndata_scores


def _convert_ontologies_to_protos(
    ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
) -> Sequence[dna_model_pb2.OntologyTerm] | None:
  """Convert ontology terms or curies to unique list of OntologyTerm protos."""
  if ontology_terms is None:
    return None
  else:
    protos = []
    for term in dict.fromkeys(ontology_terms):
      if isinstance(term, ontology.OntologyTerm):
        protos.append(term.to_proto())
      else:
        protos.append(ontology.from_curie(term).to_proto())
    return protos


def validate_sequence_length(length: int):
  """Validate that the input sequence length is supported by the model."""
  if length not in SUPPORTED_SEQUENCE_LENGTHS.values():
    raise ValueError(
        f'Sequence length {length} not supported by the model.'
        f' Supported lengths: {[*SUPPORTED_SEQUENCE_LENGTHS.values()]}'
    )


class DnaClient(dna_model.DnaModel):
  """Client for interacting with a DNA model server.

  Attributes:
    channel: gRPC channel to the DNA model server.
    metadata: Metadata to send with each request.
    model_version: Optional model version to use for the DNA model server. If
      none provided, the default model version will be used.
  """

  def __init__(
      self,
      *,
      channel: grpc.Channel,
      model_version: ModelVersion | None = None,
      metadata: Sequence[tuple[str, str]] = (),
      cache_dir: str | None = None,
      log_level: str = 'WARNING',
  ):
    self._channel = channel
    self._metadata = metadata
    self._model_version = (
        model_version.name if model_version is not None else None
    )
    if cache_dir:
      self._cache = cache_utils.DiskCache(cache_dir)
    else:
      self._cache = cache_utils.NoCache()
    
    # Initialize logger
    self._logger = logging.getLogger(f'{__name__}.DnaClient')
    self._logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
    if not self._logger.handlers:
      handler = logging.StreamHandler()
      formatter = logging.Formatter(
          '[%(asctime)s] [%(levelname)s] %(message)s',
          datefmt='%Y-%m-%d %H:%M:%S'
      )
      handler.setFormatter(formatter)
      self._logger.addHandler(handler)
    
    # Cache statistics
    self._cache_stats = {
        'total_requests': 0,
        'cache_hits': 0,
        'api_calls': 0,
    }
    
    self._logger.info(
        f'DnaClient initialized with cache: {"enabled" if cache_dir else "disabled"}'
    )

  def _make_cache_key(self, method_name: str, *args, **kwargs) -> str:
    """Helper to create a cache key for a method call."""
    return cache_utils.make_key(method_name, args, kwargs)

  def print_cache_stats(self):
    """Print cache performance statistics."""
    total = self._cache_stats['total_requests']
    hits = self._cache_stats['cache_hits']
    calls = self._cache_stats['api_calls']
    
    if total == 0:
      print('[Cache Stats] No requests made yet.')
      return
    
    hit_rate = (hits / total * 100) if total > 0 else 0
    print(f'[Cache Stats] Total Requests: {total}')
    print(f'[Cache Stats] Cache Hits: {hits} ({hit_rate:.1f}%)')
    print(f'[Cache Stats] API Calls: {calls}')
    if calls > 0:
      print(f'[Cache Stats] Savings: {hits} requests avoided!')

  def close(self):
    """Closes the underlying gRPC channel."""
    self._logger.info('Closing DnaClient')
    self._channel.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  @retry_rpc
  def predict_sequence(
      self,
      sequence: str,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
      interval: genome.Interval | None = None,
  ) -> Output:
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
    # Check cache
    self._cache_stats['total_requests'] += 1
    cache_key = self._make_cache_key(
        'predict_sequence',
        sequence,
        organism=organism,
        requested_outputs=tuple(sorted(requested_outputs)), # sort for stability
        ontology_terms=tuple(sorted(ontology_terms)) if ontology_terms else None,
        interval=interval
    )
    if (cached := self._cache.get(cache_key)) is not None:
      self._cache_stats['cache_hits'] += 1
      self._logger.debug(f'Cache HIT: predict_sequence for sequence length {len(sequence)}')
      return cached
    
    self._cache_stats['api_calls'] += 1
    self._logger.info(f'API Call: predict_sequence for sequence length {len(sequence)}')
    start_time = time.time()

    if not (unique_values := set(sequence)).issubset(
        _VALID_SEQUENCE_CHARACTERS
    ):
      invalid_characters = ','.join(unique_values - _VALID_SEQUENCE_CHARACTERS)
      raise ValueError(
          'Invalid sequence, must only contain the characters "ACGTN". Found'
          f' invalid characters: "{invalid_characters}"'
      )
    validate_sequence_length(len(sequence))
    requested_outputs_protos = [o.to_proto() for o in dict.fromkeys(requested_outputs)]
    request = dna_model_service_pb2.PredictSequenceRequest(
        sequence=sequence,
        organism=organism.to_proto(),
        ontology_terms=_convert_ontologies_to_protos(ontology_terms),
        requested_outputs=requested_outputs_protos,
        model_version=self._model_version,
    )
    responses = dna_model_service_pb2_grpc.DnaModelServiceStub(
        self._channel
    ).PredictSequence(iter([request]), metadata=self._metadata)
    result = _make_output(responses, interval=interval)
    
    elapsed = time.time() - start_time
    self._logger.info(f'API Call completed: predict_sequence ({elapsed:.2f}s)')
    
    # Store in cache
    self._cache.set(cache_key, result)
    return result

  @retry_rpc
  def predict_interval(
      self,
      interval: genome.Interval,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
  ) -> Output:
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
    # Check cache
    self._cache_stats['total_requests'] += 1
    cache_key = self._make_cache_key(
        'predict_interval',
        interval,
        organism=organism,
        requested_outputs=tuple(sorted(requested_outputs)),
        ontology_terms=tuple(sorted(ontology_terms)) if ontology_terms else None
    )
    if (cached := self._cache.get(cache_key)) is not None:
      self._cache_stats['cache_hits'] += 1
      self._logger.debug(f'Cache HIT: predict_interval for {interval}')
      return cached
    
    self._cache_stats['api_calls'] += 1
    self._logger.info(f'API Call: predict_interval for {interval}')
    start_time = time.time()

    validate_sequence_length(interval.width)
    requested_outputs_protos = [o.to_proto() for o in dict.fromkeys(requested_outputs)]
    request = dna_model_service_pb2.PredictIntervalRequest(
        interval=interval.to_proto(),
        organism=organism.to_proto(),
        ontology_terms=_convert_ontologies_to_protos(ontology_terms),
        requested_outputs=requested_outputs_protos,
        model_version=self._model_version,
    )
    responses = dna_model_service_pb2_grpc.DnaModelServiceStub(
        self._channel
    ).PredictInterval(iter([request]), metadata=self._metadata)
    result = _make_output(responses)
    
    elapsed = time.time() - start_time
    self._logger.info(f'API Call completed: predict_interval ({elapsed:.2f}s)')

    # Store in cache
    self._cache.set(cache_key, result)
    return result

  @retry_rpc
  def predict_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
  ) -> VariantOutput:
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
    # Check cache
    self._cache_stats['total_requests'] += 1
    cache_key = self._make_cache_key(
        'predict_variant',
        interval,
        variant,
        organism=organism,
        requested_outputs=tuple(sorted(requested_outputs)),
        ontology_terms=tuple(sorted(ontology_terms)) if ontology_terms else None
    )
    if (cached := self._cache.get(cache_key)) is not None:
      self._cache_stats['cache_hits'] += 1
      self._logger.debug(f'Cache HIT: predict_variant for {variant}')
      return cached
    
    self._cache_stats['api_calls'] += 1
    self._logger.info(f'API Call: predict_variant for {variant}')
    start_time = time.time()

    validate_sequence_length(interval.width)
    requested_outputs_protos = [o.to_proto() for o in dict.fromkeys(requested_outputs)]
    request = dna_model_service_pb2.PredictVariantRequest(
        interval=interval.to_proto(),
        variant=variant.to_proto(),
        organism=organism.to_proto(),
        ontology_terms=_convert_ontologies_to_protos(ontology_terms),
        requested_outputs=requested_outputs_protos,
        model_version=self._model_version,
    )
    responses = dna_model_service_pb2_grpc.DnaModelServiceStub(
        self._channel
    ).PredictVariant(iter([request]), metadata=self._metadata)
    result = _make_variant_output(responses)
    
    elapsed = time.time() - start_time
    self._logger.info(f'API Call completed: predict_variant ({elapsed:.2f}s)')

    # Store in cache
    self._cache.set(cache_key, result)
    return result

  @retry_rpc
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
    if not interval_scorers:
      interval_scorers = list(
          interval_scorers_lib.RECOMMENDED_INTERVAL_SCORERS.values()
      )

    validate_sequence_length(interval.width)
    if len(interval_scorers) != len(set(interval_scorers)):
      raise ValueError(
          f'Duplicate interval scorers requested: {interval_scorers}.'
      )
    if len(interval_scorers) > MAX_VARIANT_SCORERS_PER_REQUEST:
      raise ValueError(
          'Too many interval scorers requested, '
          f'maximum allowed: {MAX_VARIANT_SCORERS_PER_REQUEST}.'
      )
    if any(
        scorer.width is not None and scorer.width > interval.width
        for scorer in interval_scorers
    ):
      raise ValueError('Interval scorers must have widths <= interval width.')
    request = dna_model_service_pb2.ScoreIntervalRequest(
        interval=interval.to_proto(),
        interval_scorers=[scorer.to_proto() for scorer in interval_scorers],
        organism=organism.to_proto(),
        model_version=self._model_version,
    )
    responses = dna_model_service_pb2_grpc.DnaModelServiceStub(
        self._channel
    ).ScoreInterval(iter([request]), metadata=self._metadata)
    interval_outputs = _make_interval_output(responses, interval)
    for ix in range(len(interval_scorers)):
      interval_outputs[ix].uns['interval_scorer'] = interval_scorers[ix]
    return interval_outputs

  @retry_rpc
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
    if not variant_scorers:
      variant_scorers = variant_scorers_lib.get_recommended_scorers(
          organism.to_proto()
      )

    validate_sequence_length(interval.width)
    for variant_scorer in variant_scorers:
      supported_organisms = variant_scorers_lib.SUPPORTED_ORGANISMS[
          variant_scorer.base_variant_scorer
      ]
      if organism.to_proto() not in supported_organisms:
        supported_organisms = [
            dna_model_pb2.Organism.Name(o).removeprefix('ORGANISM_')
            for o in supported_organisms
        ]
        raise ValueError(
            f'Unsupported organism: {organism.name} for scorer'
            f' {variant_scorer}. Supported organisms:'
            f' {supported_organisms}'
        )
    if len(variant_scorers) != len(set(variant_scorers)):
      raise ValueError(
          f'Duplicate variant scorers requested: {variant_scorers}.'
      )
    if len(variant_scorers) > MAX_VARIANT_SCORERS_PER_REQUEST:
      raise ValueError(
          'Too many variant scorers requested, '
          f'maximum allowed: {MAX_VARIANT_SCORERS_PER_REQUEST}.'
      )
    request = dna_model_service_pb2.ScoreVariantRequest(
        interval=interval.to_proto(),
        variant=variant.to_proto(),
        organism=organism.to_proto(),
        variant_scorers=[vs.to_proto() for vs in variant_scorers],
        model_version=self._model_version,
    )
    responses = dna_model_service_pb2_grpc.DnaModelServiceStub(
        self._channel
    ).ScoreVariant(iter([request]), metadata=self._metadata)
    variant_outputs = _make_score_variant_output(responses, interval)
    for ix in range(len(variant_scorers)):
      variant_outputs[ix].uns['variant_scorer'] = variant_scorers[ix]
    return variant_outputs

  def score_ism_variants(
      self,
      interval: genome.Interval,
      ism_interval: genome.Interval,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      interval_variant: genome.Variant | None = None,
      progress_bar: bool = True,
      max_workers: int = dna_model.DEFAULT_MAX_WORKERS,
  ) -> list[list[anndata.AnnData]]:
    """Generate in-silico mutagenesis (ISM) variant scores for a given interval.

    Args:
      interval: DNA interval to make the prediction for.
      ism_interval: Interval to perform ISM.
      variant_scorers: Sequence of variant scorers to use for scoring each
        variant. If no variant scorers are provided, the recommended variant
        scorers for the organism will be used.
      organism: Organism to use for the prediction.
      interval_variant: Optional variant to apply to the sequence. If provided,
        the alternate allele is used for in-silico mutagenesis, otherwise the
        unaltered reference sequence is used.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      List of variant scores for each variant in the ISM interval.
    """
    if not variant_scorers:
      variant_scorers = variant_scorers_lib.get_recommended_scorers(
          organism.to_proto()
      )

    validate_sequence_length(interval.width)
    if ism_interval.negative_strand:
      raise ValueError('ISM interval must be on the positive strand.')

    ism_intervals = []
    for position in range(0, ism_interval.width, MAX_ISM_INTERVAL_WIDTH):
      ism_intervals.append(
          genome.Interval(
              ism_interval.chromosome,
              ism_interval.start + position,
              min(
                  ism_interval.start + position + MAX_ISM_INTERVAL_WIDTH,
                  ism_interval.end,
              ),
          )
      )

    @retry_rpc
    def _score_ism_variant(
        ism_interval: genome.Interval,
    ) -> list[anndata.AnnData]:
      request = dna_model_service_pb2.ScoreIsmVariantRequest(
          interval=interval.to_proto(),
          ism_interval=ism_interval.to_proto(),
          organism=organism.to_proto(),
          variant_scorers=[vs.to_proto() for vs in variant_scorers],
          model_version=self._model_version,
          interval_variant=interval_variant.to_proto()
          if interval_variant is not None
          else None,
      )
      responses = dna_model_service_pb2_grpc.DnaModelServiceStub(
          self._channel
      ).ScoreIsmVariant(iter([request]), metadata=self._metadata)

      return _make_score_variant_output(responses, interval)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = [
          executor.submit(_score_ism_variant, ism_interval)
          for ism_interval in ism_intervals
      ]
      ism_scores = []
      for future in tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          disable=not progress_bar,
      ):
        ism_scores.extend(future.result())

    # Group scores by variant.
    ism_scores_by_variant: dict[str, list[anndata.AnnData]] = {}
    for variant_score in ism_scores:
      ism_scores_by_variant.setdefault(
          str(variant_score.uns['variant']), []
      ).append(variant_score)

    return list(ism_scores_by_variant.values())

  def output_metadata(
      self, organism: Organism = Organism.HOMO_SAPIENS
  ) -> OutputMetadata:
    """Get the metadata for a given organism.

    Args:
      organism: Organism to get metadata for.

    Returns:
      OutputMetadata for the provided organism.
    """
    request = dna_model_service_pb2.MetadataRequest(
        organism=organism.to_proto()
    )
    responses = dna_model_service_pb2_grpc.DnaModelServiceStub(
        self._channel
    ).GetMetadata(request, metadata=self._metadata)
    return construct_output_metadata(responses)


def create(
    api_key: str,
    *,
    model_version: ModelVersion | None = None,
    timeout: float | None = None,
    address: str | None = None,
    cache_dir: str | None = None,
    log_level: str = 'WARNING',
) -> DnaClient:
  """Creates a model client for a given API key.

  Args:
    api_key: API key to use for authentication.
    model_version: Optional model version to use for the DNA model server. If
      none is provided, the default model will be used.
    timeout: Optional timeout for waiting for the channel to be ready.
    address: Optional server address to connect to.
    cache_dir: Optional directory to store cached responses.
    log_level: Logging level (DEBUG, INFO, WARNING, ERROR). Default: WARNING.

  Returns:
    `DnaClient` instance.
  """
  options = [
      ('grpc.max_send_message_length', -1),
      ('grpc.max_receive_message_length', -1),
  ]
  address = address or 'dns:///gdmscience.googleapis.com:443'
  channel = grpc.secure_channel(
      address, grpc.ssl_channel_credentials(), options=options
  )
  grpc.channel_ready_future(channel).result(timeout)

  return DnaClient(
      channel=channel,
      model_version=model_version,
      metadata=[('x-goog-api-key', api_key)],
      cache_dir=cache_dir,
      log_level=log_level,
  )
