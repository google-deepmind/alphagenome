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

"""Utils for converting JunctionData to/from protos."""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from alphagenome import tensor_utils
from alphagenome.data import genome, junction_data, ontology
from alphagenome.protos import dna_model_pb2, tensor_pb2


def to_protos(
    data: junction_data.JunctionData,
    *,
    bytes_per_chunk: int = 0,
    compression_type: tensor_pb2.CompressionType = (
        tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE
    ),
) -> tuple[dna_model_pb2.JunctionData, Sequence[tensor_pb2.TensorChunk]]:
  """Converts the `JunctionData` to protobuf messages.

  Args:
    data: The `JunctionData` object to convert to protos.
    bytes_per_chunk: The maximum number of bytes per tensor chunk.
    compression_type: The compression type to use for the tensor chunks.

  Returns:
    A tuple containing the `JunctionData` protobuf message and a sequence of
      `TensorChunk` protobuf messages.
  """
  tensor, chunks = tensor_utils.pack_tensor(
      data.values,
      bytes_per_chunk=bytes_per_chunk,
      compression_type=compression_type,
  )

  return (
      dna_model_pb2.JunctionData(
          junctions=[j.to_proto() for j in data.junctions],
          values=tensor,
          metadata=metadata_to_proto(data.metadata).metadata,
          interval=data.interval.to_proto() if data.interval else None,
      ),
      chunks,
  )


def from_protos(
    proto: dna_model_pb2.JunctionData,
    chunks: Iterable[tensor_pb2.TensorChunk] = (),
    *,
    interval: genome.Interval | None = None,
) -> junction_data.JunctionData:
  """Converts a `JunctionData` protobuf to a `JunctionData` object.

  Args:
    proto: A `JunctionData` protobuf message.
    chunks: A sequence of `TensorChunk` protobuf messages.
    interval: Optional `Interval` object representing the genomic region
      containing the junctions. Only used if the proto does not have an
      interval.

  Returns:
    A `JunctionData` object.
  """
  values = tensor_utils.unpack_proto(proto.values, chunks)
  values = tensor_utils.upcast_floating(values)

  metadata = metadata_from_proto(
      dna_model_pb2.JunctionsMetadata(metadata=proto.metadata)
  )

  if proto.HasField('interval'):
    interval = genome.Interval.from_proto(proto.interval)

  return junction_data.JunctionData(
      junctions=np.array(
          [genome.Interval.from_proto(j) for j in proto.junctions]
      ),
      values=values,
      metadata=metadata,
      interval=interval,
  )


def metadata_to_proto(
    metadata: junction_data.JunctionMetadata,
) -> dna_model_pb2.JunctionsMetadata:
  """Converts junction metadata to a JunctionsMetadata.

  Args:
    metadata: A pandas DataFrame containing junction metadata.

  Returns:
    A `JunctionsMetadata` protobuf message.
  """
  names = metadata['name']
  default_values = [None] * len(names)

  columns = zip(
      metadata['name'],
      metadata.get('ontology_curie', default_values),
      metadata.get('biosample_type', default_values),
      metadata.get('biosample_name', default_values),
      metadata.get('biosample_life_stage', default_values),
      metadata.get('gtex_tissue', default_values),
      metadata.get('data_source', default_values),
      metadata.get('Assay title', default_values),
      strict=True,
  )

  metadata_protos = []

  for (
      name,
      ontology_curie,
      biosample_type,
      biosample_name,
      biosample_life_stage,
      gtex_tissue,
      data_source,
      assay,
  ) in columns:
    if biosample_type is not None:
      biosample_proto = dna_model_pb2.Biosample(
          name=biosample_name,
          type=dna_model_pb2.BiosampleType.Value(
              f'BIOSAMPLE_TYPE_{biosample_type.upper()}'
          ),
          stage=biosample_life_stage,
      )
    else:
      biosample_proto = None

    metadata_protos.append(
        dna_model_pb2.JunctionMetadata(
            name=name,
            ontology_term=ontology.from_curie(ontology_curie).to_proto()
            if ontology_curie
            else None,
            biosample=biosample_proto,
            gtex_tissue=gtex_tissue,
            data_source=data_source,
            assay=assay,
        )
    )

  return dna_model_pb2.JunctionsMetadata(metadata=metadata_protos)


def metadata_from_proto(
    proto: dna_model_pb2.JunctionsMetadata,
) -> junction_data.JunctionMetadata:
  """Create JunctionMetadata from a dna_model_pb2.JunctionsMetadata.

  Args:
    proto: A `JunctionsMetadata` protobuf message.

  Returns:
    A pandas DataFrame containing junction metadata.
  """
  metadata = []
  for junction_proto in proto.metadata:
    junction_metadata = {
        'name': junction_proto.name,
    }

    if junction_proto.HasField('ontology_term'):
      junction_metadata['ontology_curie'] = ontology.from_proto(
          junction_proto.ontology_term
      ).ontology_curie

    if junction_proto.HasField('biosample'):
      junction_metadata['biosample_name'] = junction_proto.biosample.name
      junction_metadata['biosample_type'] = (
          dna_model_pb2.BiosampleType.Name(junction_proto.biosample.type)
          .removeprefix('BIOSAMPLE_TYPE_')
          .lower()
      )
      if junction_proto.biosample.HasField('stage'):
        junction_metadata['biosample_life_stage'] = (
            junction_proto.biosample.stage
        )

    if junction_proto.HasField('gtex_tissue'):
      junction_metadata['gtex_tissue'] = junction_proto.gtex_tissue

    if junction_proto.HasField('data_source'):
      junction_metadata['data_source'] = junction_proto.data_source

    if junction_proto.HasField('assay'):
      junction_metadata['Assay title'] = junction_proto.assay

    metadata.append(junction_metadata)

  if metadata:
    return pd.DataFrame(metadata)
  else:
    return pd.DataFrame(columns=['name'])
