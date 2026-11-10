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

from alphagenome import tensor_utils
from alphagenome.protos import tensor_pb2
from alphagenome.data import gene_expression_data
from alphagenome.data import genome
from alphagenome.models.v2 import track_data_utils
from alphagenome.protos.v2 import dna_model_pb2 as dna_model_pb2_v2
import pandas as pd


def gene_metadata_to_proto(
    metadata: gene_expression_data.GeneMetadata,
) -> dna_model_pb2_v2.GenesExpressionMetadata:
  """Converts the `GeneMetadata` to a `GenesExpressionMetadata` protobuf."""
  metadata_protos = []
  for _, row in metadata.iterrows():
    if (strand := row.get('strand')) is not None:
      strand = genome.Strand.from_str(strand).to_proto()

    metadata_protos.append(
        dna_model_pb2_v2.GeneExpressionMetadata(
            gene_id=row['gene_id'],
            strand=strand,
            name=row.get('gene_name'),
            type=row.get('gene_type'),
            chromosome=row['Chromosome'],
            start=row['Start'],
            end=row['End'],
        )
    )
  return dna_model_pb2_v2.GenesExpressionMetadata(metadata=metadata_protos)


def gene_metadata_from_proto(
    gene_metadata: dna_model_pb2_v2.GenesExpressionMetadata,
) -> gene_expression_data.GeneMetadata:
  """Converts `GenesExpressionMetadata` to a `GeneMetadata` dataframe."""
  metadata = []
  for gene_proto in gene_metadata.metadata:
    entry = {
        'gene_id': gene_proto.gene_id,
        'Chromosome': gene_proto.chromosome,
        'Start': gene_proto.start,
        'End': gene_proto.end,
    }
    if gene_proto.HasField('strand'):
      entry['strand'] = str(genome.Strand.from_proto(gene_proto.strand))

    if gene_proto.HasField('name'):
      entry['gene_name'] = gene_proto.name
    if gene_proto.HasField('type'):
      entry['gene_type'] = gene_proto.type
    metadata.append(entry)
  return pd.DataFrame(metadata)


def to_protos(
    data: gene_expression_data.GeneExpressionData,
    *,
    compression_type: tensor_pb2.CompressionType = (
        tensor_pb2.CompressionType.COMPRESSION_TYPE_NONE
    ),
) -> dna_model_pb2_v2.GeneExpressionData:
  """Serializes `GeneExpressionData` to a `GeneExpressionData` protobuf message.

  Args:
    data: The `GeneExpressionData` object to serialize.
    compression_type: The compression type to use for the tensor.

  Returns:
    The `GeneExpressionData` protobuf message.
  """
  tensor, _ = tensor_utils.pack_tensor(
      data.values,
      bytes_per_chunk=0,
      compression_type=compression_type,
  )
  return dna_model_pb2_v2.GeneExpressionData(
      values=tensor,
      track_metadata=track_data_utils.metadata_to_proto(
          data.track_metadata
      ).metadata,
      gene_metadata=gene_metadata_to_proto(data.gene_metadata).metadata,
      interval=data.interval.to_proto(),
  )


def from_protos(
    proto: dna_model_pb2_v2.GeneExpressionData,
) -> gene_expression_data.GeneExpressionData:
  """Creates a `GeneExpressionData` object from a `GeneExpressionData` protobuf.

  Args:
    proto: A `GeneExpressionData` protobuf message.

  Returns:
    A `GeneExpressionData` object.
  """
  track_metadata = track_data_utils.metadata_from_proto(
      dna_model_pb2_v2.TracksMetadata(metadata=proto.track_metadata)
  )
  gene_metadata = gene_metadata_from_proto(
      dna_model_pb2_v2.GenesExpressionMetadata(metadata=proto.gene_metadata)
  )

  values = tensor_utils.unpack_proto(proto.values)
  values = tensor_utils.upcast_floating(values)
  interval = genome.Interval.from_proto(proto.interval)

  return gene_expression_data.GeneExpressionData(
      values=values,
      track_metadata=track_metadata,
      gene_metadata=gene_metadata,
      interval=interval,
  )
