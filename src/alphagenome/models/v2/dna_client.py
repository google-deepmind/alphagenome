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

"""Client implementation for interacting with a DNA model server."""

from collections.abc import Iterable, Mapping, Sequence
import concurrent.futures
import enum
import importlib.resources
from typing import Any, TypeVar

from alphagenome import tensor_utils
from alphagenome.protos.v1 import dna_model_pb2 as dna_model_pb2_v1
from alphagenome.data import gene_expression_data
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.data import track_data
from alphagenome.models.v2 import dna_model
from alphagenome.models.v2 import dna_output
from alphagenome.models.v2 import gene_counts_condition
from alphagenome.models.v2 import gene_expression_data_utils
from alphagenome.models.v2 import interval_scorers as interval_scorers_lib
from alphagenome.models.v2 import junction_data_utils
from alphagenome.models.v2 import model_condition
from alphagenome.models.v2 import track_data_utils
from alphagenome.models.v2 import variant_scorers as variant_scorers_lib
from alphagenome.protos.v2 import dna_model_pb2 as dna_model_pb2_v2
from alphagenome.protos.v2 import dna_model_service_pb2
from alphagenome.protos.v2 import dna_model_service_pb2_grpc
import anndata
import grpc
import numpy as np
import pandas as pd
import tqdm.auto

# Supported DNA sequence lengths.
SEQUENCE_LENGTH_16KB = 2**14  #: 16,384 bp (2^14).
SEQUENCE_LENGTH_100KB = 2**17  #: 131,072 bp (2^17).
SEQUENCE_LENGTH_500KB = 2**19  #: 524,288 bp (2^19).
SEQUENCE_LENGTH_1MB = 2**20  #: 1,048,576 bp (2^20).

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
MAX_VARIANT_SCORERS_PER_REQUEST = 23

#: Maximum number of cells accepted per `cell_conditioning` call (256).
MAX_CONDITIONING_CELLS = 256

# Maximum number of cells per EncodeCells request.
# This controls the behavior of `encode_cells` in the client.
# When a user inputs more cells than this value, the client will automatically
# split the cells into chunks of this size and send them concurrently.
MAX_ENCODE_CELLS_PER_REQUEST = 1000

_VALID_SEQUENCE_CHARACTERS = frozenset('ACGTN')


@enum.unique
class ModelVersion(enum.Enum):
  """Enumeration of available model checkpoints.

  Cross-validation checkpoints use the eight-fold genome split (`fold0`–`fold7`)
  from Borzoi, where each checkpoint trains on six splits and holds out one
  validation split and one test split (see
  :func:`alphagenome.data.fold_intervals.get_fold_names`).

  Attributes:
    ALL_FOLDS: Model trained on all folds (default).
    FOLD_1: Cross-validation model holding out `fold3` (valid) and `fold4`
      (test).
  """

  ALL_FOLDS = enum.auto()  # Default model version if None explicitly set.
  FOLD_1 = enum.auto()


TargetGeneMetadata = dna_output.TargetGeneMetadata
Output = dna_output.Output
OutputMetadata = dna_output.OutputMetadata
OutputType = dna_output.OutputType
VariantOutput = dna_output.VariantOutput
InputGradientsOutput = dna_output.InputGradientsOutput
Organism = dna_model.Organism

PredictResponse = TypeVar(
    'PredictResponse',
    dna_model_service_pb2.PredictSequenceResponse,
    dna_model_service_pb2.PredictIntervalResponse,
)


def _make_output_data(
    output: dna_model_pb2_v2.Output,
    interval: genome.Interval | None = None,
) -> (
    track_data.TrackData
    | np.ndarray
    | junction_data.JunctionData
    | gene_expression_data.GeneExpressionData
):
  """Helper to create an output data object from an output proto."""
  match output.WhichOneof('payload'):
    case 'track_data':
      return track_data_utils.from_protos(
          output.track_data,
          interval=interval,
      )
    case 'data':
      values = tensor_utils.unpack_proto(output.data)
      return tensor_utils.upcast_floating(values)
    case 'junction_data':
      return junction_data_utils.from_protos(
          output.junction_data,
          interval=interval,
      )
    case 'gene_expression_data':
      return gene_expression_data_utils.from_protos(
          output.gene_expression_data,
      )
    case 'cell_embeddings':
      return tensor_utils.unpack_proto(output.cell_embeddings.values)
    case _:
      raise ValueError(
          f'Unsupported output type: {output.WhichOneof("payload")}'
      )


def construct_output_metadata(
    response: dna_model_service_pb2.ObservedMetadataResponse,
) -> OutputMetadata:
  """Constructs an OutputMetadata from a response."""
  metadata = {}
  for metadata_proto in response.observed_metadata:
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
            f'Unsupported metadata type: {metadata_proto.WhichOneof("payload")}'
        )

  return OutputMetadata(
      atac=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_ATAC),
      cage=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_CAGE),
      dnase=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_DNASE),
      rna_seq=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_RNA_SEQ),
      chip_histone=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_CHIP_HISTONE),
      chip_tf=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_CHIP_TF),
      splice_sites=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_SITES),
      splice_site_usage=metadata.get(
          dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_SITE_USAGE
      ),
      splice_junctions=metadata.get(
          dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_JUNCTIONS
      ),
      contact_maps=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_CONTACT_MAPS),
      sc_gene_expression=metadata.get(
          dna_model_pb2_v2.OUTPUT_TYPE_SC_GENE_EXPRESSION
      ),
      sc_atac=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_SC_ATAC),
      sc_gene_expression_denoised=metadata.get(
          dna_model_pb2_v2.OUTPUT_TYPE_SC_GENE_EXPRESSION_DENOISED
      ),
      methylation=metadata.get(dna_model_pb2_v2.OUTPUT_TYPE_METHYLATION),
  )


def _construct_output(
    output_dict: Mapping[
        dna_model_pb2_v2.OutputType,
        track_data.TrackData | np.ndarray | junction_data.JunctionData,
    ],
):
  """Helper to construct an Output dataclass from a mapping of output types."""
  output = Output(
      atac=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_ATAC
      ),
      cage=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_CAGE
      ),
      dnase=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_DNASE
      ),
      rna_seq=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_RNA_SEQ
      ),
      chip_histone=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_CHIP_HISTONE
      ),
      chip_tf=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_CHIP_TF
      ),
      splice_sites=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_SITES
      ),
      splice_site_usage=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_SITE_USAGE
      ),
      splice_junctions=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_JUNCTIONS
      ),
      contact_maps=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_CONTACT_MAPS
      ),
      sc_gene_expression=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_SC_GENE_EXPRESSION
      ),
      sc_atac=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_SC_ATAC
      ),
      sc_gene_expression_denoised=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_SC_GENE_EXPRESSION_DENOISED
      ),
      cell_embeddings=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_CELL_EMBEDDINGS
      ),
      methylation=output_dict.get(  # pyrefly: ignore[bad-argument-type]
          dna_model_pb2_v2.OUTPUT_TYPE_METHYLATION
      ),
  )
  return output


def _make_output(
    response: PredictResponse,
    *,
    interval: genome.Interval | None = None,
    cell_condition_name: str | None = None,
) -> Output:
  """Helper to load an Output dataclass from a stream of response protos."""
  outputs = {}

  for output in response.outputs:
    outputs[output.output_type] = _make_output_data(output, interval=interval)

  parsed_output = _construct_output(outputs)
  if cell_condition_name is not None:
    parsed_output = parsed_output.with_name_suffix(
        f'_{cell_condition_name}'
    ).with_metadata_column('biosample_name', cell_condition_name)
  return parsed_output


def _make_variant_output(
    response: dna_model_service_pb2.PredictVariantResponse,
    cell_condition_name: str | None = None,
) -> VariantOutput:
  """Helper to load an Output dataclass from a stream of response protos."""
  ref_outputs = {}
  alt_outputs = {}

  for output in response.reference_outputs:
    ref_outputs[output.output_type] = _make_output_data(output)

  for output in response.alternate_outputs:
    alt_outputs[output.output_type] = _make_output_data(output)

  reference = _construct_output(ref_outputs)
  alternate = _construct_output(alt_outputs)
  if cell_condition_name is not None:
    reference = reference.with_name_suffix(
        f'_{cell_condition_name}'
    ).with_metadata_column('biosample_name', cell_condition_name)
    alternate = alternate.with_name_suffix(
        f'_{cell_condition_name}'
    ).with_metadata_column('biosample_name', cell_condition_name)

  return VariantOutput(
      reference=reference,
      alternate=alternate,
  )


def _construct_anndata_from_proto(
    scorer_metadata: Sequence[dna_model_pb2_v1.GeneScorerMetadata] | None,
    track_metadata: Sequence[dna_model_pb2_v2.TrackMetadata],
    values: np.ndarray,
    interval: genome.Interval,
    variant: dna_model_pb2_v1.Variant | None = None,
    cell_condition_name: str | None = None,
    cpg_metadata: Sequence[dna_model_pb2_v2.CpgScorerMetadata] | None = None,
) -> anndata.AnnData:
  """Helper to construct `AnnData` from a scoring proto."""
  obs = None
  if scorer_metadata:
    metadata = []
    for gene_proto in scorer_metadata:
      entry: dict[str, Any] = {
          'gene_id': gene_proto.gene_id,
      }
      if gene_proto.HasField('strand'):
        entry['strand'] = str(genome.Strand.from_proto(gene_proto.strand))

      if gene_proto.HasField('name'):
        entry['gene_name'] = gene_proto.name
      if gene_proto.HasField('type'):
        entry['gene_type'] = gene_proto.type
      if gene_proto.HasField('junction_start'):
        entry['junction_Start'] = gene_proto.junction_start
      if gene_proto.HasField('junction_end'):
        entry['junction_End'] = gene_proto.junction_end

      metadata.append(entry)

    obs = pd.DataFrame(metadata)
    obs.index = obs.index.map(str)
  elif cpg_metadata:
    positions = [cpg_proto.position for cpg_proto in cpg_metadata]
    obs = pd.DataFrame({'cpg_position': positions})
    obs.index = [
        f'{interval.chromosome}:{position}-{position + 2}'
        for position in positions
    ]

  var = track_data_utils.metadata_from_proto(
      dna_model_pb2_v2.TracksMetadata(metadata=track_metadata)
  )
  var.index = var.index.map(str)
  if cell_condition_name is not None:
    var['name'] = var['name'] + f'_{cell_condition_name}'
    var['biosample_name'] = cell_condition_name

  uns: dict[str, Any] = {'interval': interval}
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
    output: dna_model_pb2_v2.ScoreVariantOutput,
    interval: genome.Interval,
    cell_condition_name: str | None = None,
) -> anndata.AnnData:
  """Returns `AnnData`  of variant scores from a ScoreVariantOutput proto."""
  # dna_model_pb2.ScoreVariantOutput currently has ONLY VariantData with values
  # and metadata.
  values = tensor_utils.unpack_proto(output.variant_data.values)
  values = tensor_utils.upcast_floating(values)
  anndata_scores = _construct_anndata_from_proto(
      output.variant_data.metadata.gene_metadata,
      output.variant_data.metadata.track_metadata,
      values,
      interval=interval,
      variant=output.variant_data.metadata.variant,
      cell_condition_name=cell_condition_name,
      cpg_metadata=output.variant_data.metadata.cpg_metadata,
  )
  return anndata_scores


def _make_score_variant_output(
    response: (
        dna_model_service_pb2.ScoreVariantResponse
        | dna_model_service_pb2.ScoreIsmVariantResponse
    ),
    interval: genome.Interval,
    cell_condition_name: str | None = None,
) -> list[anndata.AnnData]:
  """Returns a list of `AnnData` variant scores from a stream of responses."""

  anndata_scores = []
  for output in response.outputs:
    anndata_scores.append(
        _construct_score_variant(
            output, interval=interval, cell_condition_name=cell_condition_name
        )
    )
  return anndata_scores


def _construct_score_interval(
    output: dna_model_pb2_v2.ScoreIntervalOutput,
    interval: genome.Interval,
    cell_condition_name: str | None = None,
) -> anndata.AnnData:
  """Returns `AnnData` of interval scores from a ScoreIntervalOutput proto."""
  values = tensor_utils.unpack_proto(output.interval_data.values)
  values = tensor_utils.upcast_floating(values)
  anndata_scores = _construct_anndata_from_proto(
      output.interval_data.metadata.gene_metadata,
      output.interval_data.metadata.track_metadata,
      values,
      interval=interval,
      cell_condition_name=cell_condition_name,
  )
  return anndata_scores


def _make_interval_output(
    response: dna_model_service_pb2.ScoreIntervalResponse,
    interval: genome.Interval,
    cell_condition_name: str | None = None,
) -> list[anndata.AnnData]:
  """Returns a list of `AnnData` interval scores from a stream of responses."""

  anndata_scores = []
  for output in response.outputs:
    anndata_scores.append(
        _construct_score_interval(
            output, interval=interval, cell_condition_name=cell_condition_name
        )
    )
  return anndata_scores


def _make_input_gradients_output(
    output: dna_model_pb2_v2.InputGradientsOutput,
    *,
    cell_condition_name: str | None = None,
) -> InputGradientsOutput:
  """Helper to load an InputGradientsOutput dataclass from a response proto."""
  track_metadata = track_data_utils.metadata_from_proto(
      dna_model_pb2_v2.TracksMetadata(metadata=output.track_metadata)
  )
  if cell_condition_name is not None:
    track_metadata['name'] = track_metadata['name'] + f'_{cell_condition_name}'
    track_metadata['biosample_name'] = cell_condition_name
  return InputGradientsOutput(
      gradients=tensor_utils.upcast_floating(
          tensor_utils.unpack_proto(output.gradients)
      ),
      interval=genome.Interval.from_proto(output.interval),
      scalar_score=float(output.scalar_score),
      target_gene=TargetGeneMetadata.from_proto(output.gene_metadata),
      track_metadata=track_metadata,
      output_type=OutputType(output.output_type),
  )


def _set_condition_input(
    request: Any,
    condition: model_condition.ModelCondition | str | None,
    cell_conditioning: anndata.AnnData | None,
) -> None:
  """Sets the condition_input oneof on a request proto."""
  if condition is not None and cell_conditioning is not None:
    raise ValueError('Cannot specify both `condition` and `cell_conditioning`.')
  if condition is None and cell_conditioning is None:
    raise ValueError('Must specify either `condition` or `cell_conditioning`.')

  if cell_conditioning is not None:
    gene_counts_condition.from_anndata(
        cell_conditioning, target=request.cell_conditioning
    )
  elif condition is not None:
    request.condition.CopyFrom(_convert_condition_to_proto(condition))


def _validate_cell_conditioning(
    cell_conditioning: anndata.AnnData | None,
    cell_condition_name: str | None,
    organism: Organism = Organism.HOMO_SAPIENS,
) -> None:
  """Validates that cell_conditioning arguments are valid."""
  if cell_conditioning is not None and cell_condition_name is None:
    raise ValueError(
        'cell_condition_name must be provided when cell_conditioning is used.'
    )
  if cell_conditioning is not None and organism != Organism.HOMO_SAPIENS:
    raise ValueError(
        'Cell conditioning is only supported for Homo sapiens'
        f' (organism={organism.name}).'
    )
  if (
      cell_conditioning is not None
      and cell_conditioning.shape[0] > MAX_CONDITIONING_CELLS
  ):
    raise ValueError(
        f'Too many cells in cell_conditioning ({cell_conditioning.shape[0]}),'
        f' maximum allowed: {MAX_CONDITIONING_CELLS}.'
    )


def _convert_condition_to_proto(
    condition: model_condition.ModelCondition | str,
) -> dna_model_pb2_v2.Condition:
  """Convert ontology terms or curies to unique list of OntologyTerm protos."""
  if isinstance(condition, model_condition.ModelCondition):
    return condition.to_proto()
  return model_condition.from_curies(condition.split('_')).to_proto()


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
  ):
    self._channel = channel
    self._metadata = metadata
    self._model_version = (
        model_version.name if model_version is not None else None
    )

  def predict_sequence(
      self,
      sequence: str,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      interval: genome.Interval | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> Output:
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
      cell_conditioning: Optional human single-cell raw UMI counts (`AnnData`,
        up to `MAX_CONDITIONING_CELLS` cells). Unlocks
        `OutputType.SC_GENE_EXPRESSION_DENOISED` (denoised single-cell gene
        expression; one channel per strand, in CPM).
      cell_condition_name: Optional label for expression conditioning. When
        provided, the output track names are suffixed with this name. Must be
        provided when `cell_conditioning` is used.

    Returns:
      Output for the provided DNA sequence.
    """
    _validate_cell_conditioning(
        cell_conditioning, cell_condition_name, organism
    )
    if not (unique_values := set(sequence)).issubset(
        _VALID_SEQUENCE_CHARACTERS
    ):
      invalid_characters = ','.join(unique_values - _VALID_SEQUENCE_CHARACTERS)
      raise ValueError(
          'Invalid sequence, must only contain the characters "ACGTN". Found'
          f' invalid characters: "{invalid_characters}"'
      )
    validate_sequence_length(len(sequence))
    requested_output_protos = [
        o.to_proto() for o in dict.fromkeys(requested_outputs)
    ]
    request = dna_model_service_pb2.PredictSequenceRequest(
        sequence=sequence,
        organism=organism.to_proto(),
        requested_outputs=requested_output_protos,
        model_version=self._model_version,
    )
    _set_condition_input(request, condition, cell_conditioning)
    response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
        self._channel
    ).PredictSequence(request, metadata=self._metadata)
    return _make_output(
        response, interval=interval, cell_condition_name=cell_condition_name
    )

  def predict_interval(
      self,
      interval: genome.Interval,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> Output:
    """Generate predictions for a given DNA interval.

    Args:
      interval: DNA interval to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      condition: The condition or composite curie to generate the prediction
        for.
      cell_conditioning: Optional human single-cell raw UMI counts (`AnnData`,
        up to `MAX_CONDITIONING_CELLS` cells). Unlocks
        `OutputType.SC_GENE_EXPRESSION_DENOISED` (denoised single-cell gene
        expression; one channel per strand, in CPM).
      cell_condition_name: Optional label for expression conditioning. When
        provided, the output track names are suffixed with this name. Must be
        provided when `cell_conditioning` is used.

    Returns:
      Output for the provided DNA interval.
    """
    _validate_cell_conditioning(
        cell_conditioning, cell_condition_name, organism
    )
    validate_sequence_length(interval.width)
    requested_output_protos = [
        o.to_proto() for o in dict.fromkeys(requested_outputs)
    ]
    request = dna_model_service_pb2.PredictIntervalRequest(
        interval=interval.to_proto(),
        organism=organism.to_proto(),
        requested_outputs=requested_output_protos,
        model_version=self._model_version,
    )
    _set_condition_input(request, condition, cell_conditioning)
    response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
        self._channel
    ).PredictInterval(request, metadata=self._metadata)
    return _make_output(response, cell_condition_name=cell_condition_name)

  def predict_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> VariantOutput:
    """Generate predictions for a given DNA variant.

    Args:
      interval: DNA interval to make prediction for.
      variant: DNA variant to make prediction for.
      organism: Organism to use for the prediction.
      requested_outputs: Iterable of OutputTypes indicating which subsets of
        predictions to return.
      condition: The condition or composite curie to generate the prediction
        for.
      cell_conditioning: Optional human single-cell raw UMI counts (`AnnData`,
        up to `MAX_CONDITIONING_CELLS` cells). Unlocks
        `OutputType.SC_GENE_EXPRESSION_DENOISED` (denoised single-cell gene
        expression; one channel per strand, in CPM).
      cell_condition_name: Optional label for expression conditioning. When
        provided, the output track names are suffixed with this name. Must be
        provided when `cell_conditioning` is used.

    Returns:
      Variant output for the provided DNA interval and variant.
    """
    _validate_cell_conditioning(
        cell_conditioning, cell_condition_name, organism
    )
    validate_sequence_length(interval.width)
    requested_output_protos = [
        o.to_proto() for o in dict.fromkeys(requested_outputs)
    ]
    request = dna_model_service_pb2.PredictVariantRequest(
        interval=interval.to_proto(),
        variant=variant.to_proto(),
        organism=organism.to_proto(),
        requested_outputs=requested_output_protos,
        model_version=self._model_version,
    )
    _set_condition_input(request, condition, cell_conditioning)
    response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
        self._channel
    ).PredictVariant(request, metadata=self._metadata)
    return _make_variant_output(
        response, cell_condition_name=cell_condition_name
    )

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
      cell_conditioning: Optional human single-cell raw UMI counts (`AnnData`,
        up to `MAX_CONDITIONING_CELLS` cells). Unlocks
        `OutputType.SC_GENE_EXPRESSION_DENOISED` (denoised single-cell gene
        expression; one channel per strand, in CPM).
      cell_condition_name: Optional label for expression conditioning. When
        provided, the output track names are suffixed with this name and the
        biosample_name column is set in the var metadata. Must be provided when
        `cell_conditioning` is used.

    Returns:
      List of `AnnData` interval scores.
    """
    _validate_cell_conditioning(
        cell_conditioning, cell_condition_name, organism
    )
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
        organism=organism.to_proto(),
        interval_scorers=[scorer.to_proto() for scorer in interval_scorers],
        model_version=self._model_version,
    )
    _set_condition_input(request, condition, cell_conditioning)
    response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
        self._channel
    ).ScoreInterval(request, metadata=self._metadata)
    interval_outputs = _make_interval_output(
        response, interval, cell_condition_name=cell_condition_name
    )
    for ix in range(len(interval_scorers)):
      interval_outputs[ix].uns['interval_scorer'] = interval_scorers[ix]
    return interval_outputs

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
        organism and conditioning type will be used.
      organism: Organism to use for the prediction.
      cell_conditioning: Optional human single-cell raw UMI counts (`AnnData`,
        up to `MAX_CONDITIONING_CELLS` cells). Unlocks
        `OutputType.SC_GENE_EXPRESSION_DENOISED` (denoised single-cell gene
        expression; one channel per strand, in CPM).
      cell_condition_name: Optional label for expression conditioning. When
        provided, the output track names are suffixed with this name and the
        biosample_name column is set in the var metadata. Must be provided when
        `cell_conditioning` is used.

    Returns:
      List of `AnnData` variant scores.
    """
    _validate_cell_conditioning(
        cell_conditioning, cell_condition_name, organism
    )
    if not variant_scorers:
      variant_scorers = variant_scorers_lib.get_recommended_scorers(
          organism.to_proto(), cell_conditioning=cell_conditioning is not None
      )

    validate_sequence_length(interval.width)
    for variant_scorer in variant_scorers:
      supported_organisms = variant_scorers_lib.SUPPORTED_ORGANISMS[
          variant_scorer.base_variant_scorer
      ]
      if organism.to_proto() not in supported_organisms:
        supported_organisms = [
            dna_model_pb2_v1.Organism.Name(o).removeprefix('ORGANISM_')
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
    _set_condition_input(request, condition, cell_conditioning)
    response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
        self._channel
    ).ScoreVariant(request, metadata=self._metadata)
    variant_outputs = _make_score_variant_output(
        response, interval, cell_condition_name=cell_condition_name
    )
    for ix in range(len(variant_scorers)):
      variant_outputs[ix].uns['variant_scorer'] = variant_scorers[ix]
    return variant_outputs

  def score_ism_variants(
      self,
      interval: genome.Interval,
      ism_interval: genome.Interval,
      condition: model_condition.ModelCondition | str | None = None,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      interval_variant: genome.Variant | None = None,
      organism: Organism = Organism.HOMO_SAPIENS,
      progress_bar: bool = True,
      max_workers: int = dna_model.DEFAULT_MAX_WORKERS,
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
        scorers for the organism and conditioning type will be used.
      interval_variant: Optional variant to apply to the sequence. If provided,
        the alternate allele is used for in silico mutagenesis, otherwise the
        unaltered reference sequence is used.
      organism: Organism to use for the prediction.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.
      cell_conditioning: Optional expression conditioning input (`AnnData`).
      cell_condition_name: Optional label for expression conditioning. When
        provided, the output track names are suffixed with this name and the
        biosample_name column is set in the var metadata. Must be provided when
        `cell_conditioning` is used.

    Returns:
      List of variant scores for each variant in the ISM interval.
    """
    if not variant_scorers:
      variant_scorers = variant_scorers_lib.get_recommended_scorers(
          organism.to_proto(), cell_conditioning=cell_conditioning is not None
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

    def _score_ism_variant(
        ism_interval: genome.Interval,
    ) -> list[anndata.AnnData]:
      _validate_cell_conditioning(
          cell_conditioning, cell_condition_name, organism
      )
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
      _set_condition_input(request, condition, cell_conditioning)
      response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
          self._channel
      ).ScoreIsmVariant(request, metadata=self._metadata)

      return _make_score_variant_output(
          response, interval, cell_condition_name=cell_condition_name
      )

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

  def observed_metadata(
      self, organism: Organism = Organism.HOMO_SAPIENS
  ) -> OutputMetadata:
    """Get the metadata for tracks observed by the model for a given organism.

    Args:
      organism: Organism to get metadata for.

    Returns:
      OutputMetadata for the provided organism.
    """
    request = dna_model_service_pb2.ObservedMetadataRequest(
        organism=organism.to_proto()
    )
    response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
        self._channel
    ).GetObservedMetadata(request, metadata=self._metadata)
    return construct_output_metadata(response)

  def supported_conditions(
      self, organism: Organism = Organism.HOMO_SAPIENS
  ) -> pd.DataFrame:
    """List reference conditions with the imputation mode of each output type."""
    request = dna_model_service_pb2.SupportedConditionsRequest(
        organism=organism.to_proto()
    )
    response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
        self._channel
    ).GetSupportedConditions(request, metadata=self._metadata)
    return _construct_supported_conditions(response)

  def encode_cells(
      self,
      cells: anndata.AnnData,
      *,
      progress_bar: bool = True,
      max_workers: int = dna_model.DEFAULT_MAX_WORKERS,
  ) -> np.ndarray:
    """Encode single-cell count profiles into AlphaGenome 2 latent embeddings.

    Runs AlphaGenome 2's single-cell autoencoder (trained on CELLxGENE) without
    evaluating the DNA sequence model. Use this as a fast diagnostic check to
    verify that biological variation across cell states is preserved in
    embedding space before running sequence predictions or variant scoring.
    Because the encoder does not explicitly correct batch effects, stratify
    inputs by batch or donor when comparing conditions.

    Cells are split into chunks of at most `MAX_ENCODE_CELLS_PER_REQUEST` cells,
    which are encoded in parallel.

    Args:
      cells: Human AnnData object with raw integer UMI counts in `.X` and
        Ensembl gene IDs (`ENSG...`) in `.var_names`.
      progress_bar: If True, show a progress bar.
      max_workers: Number of parallel workers to use.

    Returns:
      NumPy array of per-cell embeddings with shape `(num_cells, latent_dim)`.
    """
    if cells.n_obs == 0:
      raise ValueError('must have at least one cell')

    cell_chunks = [
        cells[start : start + MAX_ENCODE_CELLS_PER_REQUEST]
        for start in range(0, cells.n_obs, MAX_ENCODE_CELLS_PER_REQUEST)
    ]

    def _encode_cells(cell_chunk: anndata.AnnData) -> np.ndarray:
      request = dna_model_service_pb2.EncodeCellsRequest()
      gene_counts_condition.from_anndata(
          cell_chunk, target=request.cell_conditioning
      )
      response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
          self._channel
      ).EncodeCells(request, metadata=self._metadata)
      return tensor_utils.unpack_proto(response.cell_embeddings.values)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      futures = {
          executor.submit(_encode_cells, cell_chunk): chunk_index
          for chunk_index, cell_chunk in enumerate(cell_chunks)
      }
      embeddings_by_chunk: dict[int, np.ndarray] = {}
      for future in tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          disable=not progress_bar,
      ):
        embeddings_by_chunk[futures[future]] = future.result()

    return np.concatenate(
        [embeddings_by_chunk[i] for i in range(len(cell_chunks))], axis=0
    )

  def predict_interval_gradients(
      self,
      gene_id: str,
      *,
      organism: Organism = Organism.HOMO_SAPIENS,
      requested_output: OutputType,
      sequence_length: int = SEQUENCE_LENGTH_1MB,
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> InputGradientsOutput:
    """Generate input gradients of a gene's predicted expression.

    The input interval is centered on the gene's median TSS.

    Args:
      gene_id: Ensembl ID of the target gene. Version suffixes are ignored.
      organism: Organism to use for the prediction.
      requested_output: Output type to compute gradients for. One of `RNA_SEQ`,
        `SC_GENE_EXPRESSION`, or `SC_GENE_EXPRESSION_DENOISED`.
      sequence_length: Width of the input interval. Must be
        `SEQUENCE_LENGTH_100KB` or `SEQUENCE_LENGTH_1MB` (defaults to
        `SEQUENCE_LENGTH_1MB`).
      condition: The condition or composite curie to generate the prediction
        for.
      cell_conditioning: Optional expression conditioning input (`AnnData`).
      cell_condition_name: Optional label for expression conditioning. When
        provided, the output track names are suffixed with this name. Must be
        provided when `cell_conditioning` is used.

    Returns:
      Input gradients for the interval centered on the gene.
    """
    _validate_cell_conditioning(
        cell_conditioning, cell_condition_name, organism
    )
    if sequence_length not in (SEQUENCE_LENGTH_100KB, SEQUENCE_LENGTH_1MB):
      raise ValueError(
          f'sequence_length must be {SEQUENCE_LENGTH_100KB} or'
          f' {SEQUENCE_LENGTH_1MB}, got {sequence_length}.'
      )
    request = dna_model_service_pb2.PredictIntervalGradientsRequest(
        sequence_length=sequence_length,
        organism=organism.to_proto(),
        target=dna_model_pb2_v2.InputGradientTarget(
            output_type=requested_output.to_proto(), gene_id=gene_id
        ),
        model_version=self._model_version,
    )
    _set_condition_input(request, condition, cell_conditioning)
    response = dna_model_service_pb2_grpc.DnaModelServiceV2Stub(
        self._channel
    ).PredictIntervalGradients(request, metadata=self._metadata)
    return _make_input_gradients_output(
        response.output, cell_condition_name=cell_condition_name
    )


def _construct_supported_conditions(
    response: dna_model_service_pb2.SupportedConditionsResponse,
) -> pd.DataFrame:
  """Construct a list of supported conditions from the response."""
  supported_conditions = []
  for supported_condition in response.supported_conditions:
    row = {
        'condition': str(
            model_condition.from_proto(supported_condition.condition)
        ),
        'biosample_type': (
            dna_model_pb2_v1.BiosampleType.Name(
                supported_condition.biosample.type
            )
            .removeprefix('BIOSAMPLE_TYPE_')
            .lower()
        ),
        'biosample_name': supported_condition.biosample.name,
        **{
            dna_model_pb2_v2.OutputType.Name(mode.output_type)
            .removeprefix('OUTPUT_TYPE_')
            .lower(): mode.imputation_mode
            for mode in supported_condition.output_type_imputation_modes
        },
    }
    supported_conditions.append(row)
  return pd.DataFrame(supported_conditions)


def create(
    api_key: str,
    *,
    model_version: ModelVersion | None = None,
    timeout: float | None = None,
    address: str | None = None,
) -> DnaClient:
  """Creates a model client for a given API key.

  Args:
    api_key: API key to use for authentication.
    model_version: Optional model version to use for the DNA model server. If
      none is provided, the default model will be used.
    timeout: Optional timeout for waiting for the channel to be ready.
    address: Optional server address to connect to.

  Returns:
   `DnaClient` instance.
  """
  service_config = (
      importlib.resources.files('alphagenome')
      / 'protos/v2/grpc_service_config.json'
  ).read_text()
  options = [
      ('grpc.max_send_message_length', -1),
      ('grpc.max_receive_message_length', -1),
      ('grpc.service_config', service_config),
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
  )
