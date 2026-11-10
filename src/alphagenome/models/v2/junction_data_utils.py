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

"""Functions for converting JunctionData to/from protos."""

from alphagenome import tensor_utils
from alphagenome.protos.v1 import dna_model_pb2 as dna_model_pb2_v1
from alphagenome.protos import tensor_pb2
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.models.v2 import model_condition
from alphagenome.protos.v2 import dna_model_pb2 as dna_model_pb2_v2
import numpy as np
import pandas as pd

JunctionMetadata = pd.DataFrame


def to_protos(
    data: junction_data.JunctionData,
    *,
    compression_type: tensor_pb2.CompressionType = (
        tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE
    ),
) -> dna_model_pb2_v2.JunctionData:
  """Converts the `JunctionData` to protobuf messages.

  Args:
    data: The `JunctionData` to convert.
    compression_type: The compression type to use for the tensor chunks.

  Returns:
    A `JunctionData` protobuf message.
  """
  tensor, _ = tensor_utils.pack_tensor(
      data.values,
      bytes_per_chunk=0,
      compression_type=compression_type,
  )

  return dna_model_pb2_v2.JunctionData(
      junctions=[j.to_proto() for j in data.junctions],
      values=tensor,
      metadata=metadata_to_proto(data.metadata).metadata,
      interval=data.interval.to_proto() if data.interval else None,
  )


def from_protos(
    proto: dna_model_pb2_v2.JunctionData,
    *,
    interval: genome.Interval | None = None,
) -> junction_data.JunctionData:
  """Converts a `JunctionData` protobuf to a `JunctionData` object.

  Args:
    proto: A `JunctionData` protobuf message.
    interval: Optional `Interval` object representing the genomic region
      containing the junctions. Only used if the proto does not have an
      interval.

  Returns:
    A `JunctionData` object.
  """
  values = tensor_utils.unpack_proto(proto.values)
  values = tensor_utils.upcast_floating(values)

  metadata = metadata_from_proto(
      dna_model_pb2_v2.JunctionsMetadata(metadata=proto.metadata)
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
) -> dna_model_pb2_v2.JunctionsMetadata:
  """Converts junction metadata to a JunctionsMetadata proto.

  Args:
    metadata: A pandas DataFrame containing junction metadata.

  Returns:
    A `JunctionsMetadata` protobuf message.
  """
  names = metadata['name']
  default_values = [None] * len(names)
  columns = zip(
      metadata['name'],
      metadata.get('biosample_curie', default_values),
      metadata.get('imputation_mode', default_values),
      metadata.get('biosample_type', default_values),
      metadata.get('biosample_name', default_values),
      metadata.get('Assay title', default_values),
      metadata.get('data_source', default_values),
      strict=True,
  )
  metadata_protos = []
  for (
      name,
      biosample_curie,
      imputation_mode,
      biosample_type,
      biosample_name,
      assay,
      data_source,
  ) in columns:
    if isinstance(biosample_type, str):
      biosample_proto = dna_model_pb2_v1.Biosample(
          name=biosample_name if isinstance(biosample_name, str) else '',
          type=dna_model_pb2_v1.BiosampleType.Value(
              f'BIOSAMPLE_TYPE_{biosample_type.upper()}'
          ),
      )
    else:
      biosample_proto = None

    if isinstance(biosample_curie, str):
      condition_proto = model_condition.from_curies(
          biosample_curie.split('_')
      ).to_proto()
    else:
      condition_proto = None

    metadata_protos.append(
        dna_model_pb2_v2.JunctionMetadata(
            name=name if isinstance(name, str) else '',
            biosample=biosample_proto,
            condition=condition_proto,
            imputation_mode=imputation_mode
            if isinstance(imputation_mode, str)
            else None,
            assay=assay if isinstance(assay, str) else None,
            data_source=data_source if isinstance(data_source, str) else None,
        )
    )
  return dna_model_pb2_v2.JunctionsMetadata(metadata=metadata_protos)


def metadata_from_proto(
    proto: dna_model_pb2_v2.JunctionsMetadata,
) -> JunctionMetadata:
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
        'biosample_curie': str(
            model_condition.from_proto(junction_proto.condition)
        ),
        'imputation_mode': junction_proto.imputation_mode,
        'biosample_type': (
            dna_model_pb2_v1.BiosampleType.Name(junction_proto.biosample.type)
            .removeprefix('BIOSAMPLE_TYPE_')
            .lower()
        ),
        'biosample_name': junction_proto.biosample.name,
    }
    if junction_proto.HasField('assay'):
      junction_metadata['Assay title'] = junction_proto.assay
    if junction_proto.HasField('data_source'):
      junction_metadata['data_source'] = junction_proto.data_source
    metadata.append(junction_metadata)
  if metadata:
    return pd.DataFrame(metadata)
  else:
    return pd.DataFrame(columns=['name'])
