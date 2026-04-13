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


"""Track data utils for converting to/from protos."""

from collections.abc import Iterable, Sequence

import pandas as pd

from alphagenome import tensor_utils
from alphagenome.data import genome, ontology, track_data
from alphagenome.protos import dna_model_pb2, tensor_pb2


def to_protos(
    data: track_data.TrackData,
    *,
    bytes_per_chunk: int = 0,
    compression_type: tensor_pb2.CompressionType = (
        tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE
    ),
) -> tuple[dna_model_pb2.TrackData, Sequence[tensor_pb2.TensorChunk]]:
  """Serializes `TrackData` to protobuf messages.

  Args:
    data: The `TrackData` object to serialize.
    bytes_per_chunk: The maximum number of bytes per tensor chunk.
    compression_type: The compression type to use for the tensor chunks.

  Returns:
    A tuple containing the `TrackData` protobuf message and a sequence of
    `TensorChunk` protobuf messages.
  """
  tensor, chunks = tensor_utils.pack_tensor(
      data.values,
      bytes_per_chunk=bytes_per_chunk,
      compression_type=compression_type,
  )
  return (
      dna_model_pb2.TrackData(
          values=tensor,
          metadata=metadata_to_proto(data.metadata).metadata,
          resolution=data.resolution,
          interval=data.interval.to_proto() if data.interval else None,
      ),
      chunks,
  )


def from_protos(
    proto: dna_model_pb2.TrackData,
    chunks: Iterable[tensor_pb2.TensorChunk] = (),
    *,
    interval: genome.Interval | None = None,
) -> track_data.TrackData:
  """Creates a `TrackData` object from protobuf messages.

  Args:
    proto: A `TrackData` protobuf message.
    chunks: A sequence of `TensorChunk` protobuf messages.
    interval: Optional `Interval` object representing the genomic region
      containing the tracks. Only used if the proto does not have an interval.

  Returns:
    A `TrackData` object.
  """
  metadata = metadata_from_proto(
      dna_model_pb2.TracksMetadata(metadata=proto.metadata)
  )

  values = tensor_utils.unpack_proto(proto.values, chunks)
  values = tensor_utils.upcast_floating(values)
  resolution = proto.resolution if proto.HasField('resolution') else 1
  if proto.HasField('interval'):
    interval = genome.Interval.from_proto(proto.interval)

  return track_data.TrackData(
      values, metadata, resolution=resolution, interval=interval
  )


def metadata_to_proto(
    metadata: track_data.TrackMetadata,
) -> dna_model_pb2.TracksMetadata:
  """Converts track metadata to a `TracksMetadata` protobuf message.

  Args:
    metadata: A pandas DataFrame containing track metadata, with the following
      required columns: name, strand.

  Returns:
    A `TracksMetadata` protobuf message.
  """
  names = metadata['name']
  default_values = [None] * len(names)

  columns = zip(
      metadata['name'],
      metadata['strand'],
      metadata.get('ontology_curie', default_values),
      metadata.get('biosample_type', default_values),
      metadata.get('biosample_name', default_values),
      metadata.get('biosample_life_stage', default_values),
      metadata.get('transcription_factor', default_values),
      metadata.get('histone_mark', default_values),
      metadata.get('gtex_tissue', default_values),
      metadata.get('Assay title', default_values),
      metadata.get('data_source', default_values),
      metadata.get('genetically_modified', default_values),
      metadata.get('endedness', default_values),
      metadata.get('nonzero_mean', default_values),
      strict=True,
  )

  metadata_protos = []
  for (
      name,
      strand,
      ontology_curie,
      biosample_type,
      biosample_name,
      biosample_life_stage,
      transcription_factor,
      histone_mark,
      gtex_tissue,
      assay,
      data_source,
      genetically_modified,
      endedness,
      nonzero_mean,
  ) in columns:
    if biosample_type:
      biosample = dna_model_pb2.Biosample(
          name=biosample_name,
          type=dna_model_pb2.BiosampleType.Value(
              f'BIOSAMPLE_TYPE_{biosample_type.upper()}'
          ),
          stage=biosample_life_stage,
      )
    else:
      biosample = None
    if endedness is not None:
      match endedness:
        case 'paired':
          endedness = dna_model_pb2.Endedness.ENDEDNESS_PAIRED
        case 'single':
          endedness = dna_model_pb2.Endedness.ENDEDNESS_SINGLE
        case _:
          raise ValueError(f'Unknown endedness: {endedness}')

    metadata_protos.append(
        dna_model_pb2.TrackMetadata(
            name=name,
            strand=genome.Strand.from_str(strand).to_proto()
            if strand
            else None,
            ontology_term=ontology.from_curie(ontology_curie).to_proto()
            if ontology_curie
            else None,
            biosample=biosample,
            transcription_factor_code=transcription_factor
            if isinstance(transcription_factor, str)
            else None,
            histone_mark_code=histone_mark
            if isinstance(histone_mark, str)
            else None,
            gtex_tissue=gtex_tissue,
            assay=assay,
            data_source=data_source,
            genetically_modified=genetically_modified,
            endedness=endedness,
            nonzero_mean=nonzero_mean,
        )
    )

  return dna_model_pb2.TracksMetadata(metadata=metadata_protos)


def metadata_from_proto(
    proto: dna_model_pb2.TracksMetadata,
) -> track_data.TrackMetadata:
  """Creates track metadata from a `TracksMetadata` protobuf message.

  Args:
    proto: A `TracksMetadata` protobuf message.

  Returns:
    A pandas DataFrame containing track metadata.
  """
  metadata = []
  for track_proto in proto.metadata:
    track_metadata = {
        'name': track_proto.name,
        'strand': str(genome.Strand.from_proto(track_proto.strand)),
    }

    if track_proto.HasField('assay'):
      track_metadata['Assay title'] = track_proto.assay

    if track_proto.HasField('ontology_term'):
      track_metadata['ontology_curie'] = ontology.from_proto(
          track_proto.ontology_term
      ).ontology_curie

    if track_proto.HasField('biosample'):
      track_metadata['biosample_name'] = track_proto.biosample.name
      track_metadata['biosample_type'] = (
          dna_model_pb2.BiosampleType.Name(track_proto.biosample.type)
          .removeprefix('BIOSAMPLE_TYPE_')
          .lower()
      )
      if track_proto.biosample.HasField('stage'):
        track_metadata['biosample_life_stage'] = track_proto.biosample.stage

    if track_proto.HasField('transcription_factor_code'):
      track_metadata['transcription_factor'] = (
          track_proto.transcription_factor_code
      )

    if track_proto.HasField('histone_mark_code'):
      track_metadata['histone_mark'] = track_proto.histone_mark_code

    if track_proto.HasField('gtex_tissue'):
      track_metadata['gtex_tissue'] = track_proto.gtex_tissue

    if track_proto.HasField('data_source'):
      track_metadata['data_source'] = track_proto.data_source

    if track_proto.HasField('endedness'):
      track_metadata['endedness'] = (
          dna_model_pb2.Endedness.Name(track_proto.endedness)
          .removeprefix('ENDEDNESS_')
          .lower()
      )

    if track_proto.HasField('genetically_modified'):
      track_metadata['genetically_modified'] = track_proto.genetically_modified

    if track_proto.HasField('nonzero_mean'):
      track_metadata['nonzero_mean'] = track_proto.nonzero_mean

    metadata.append(track_metadata)
  if metadata:
    return pd.DataFrame(metadata)
  else:
    return pd.DataFrame(columns=['name', 'strand'])
