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
"""Module for AlphaGenome model outputs."""

from collections.abc import Callable, Iterable
import dataclasses
import enum
from typing import Literal

from alphagenome import typing
from alphagenome.data import gene_expression_data
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.data import track_data
from alphagenome.protos.v2 import dna_model_pb2
import numpy as np
import pandas as pd


class OutputType(enum.Enum):
  """Enumeration of all the available types of outputs.

  Attributes:
    ATAC: ATAC-seq tracks capturing chromatin accessibility.
    CAGE: CAGE (Cap Analysis of Gene Expression) tracks capturing gene
      expression.
    DNASE: DNase I hypersensitive site tracks capturing chromatin accessibility.
    RNA_SEQ: RNA sequencing tracks capturing gene expression.
    CHIP_HISTONE: ChIP-seq tracks capturing histone modifications.
    CHIP_TF: ChIP-seq tracks capturing transcription factor binding.
    SPLICE_SITES: Condition-specific splice site classification probabilities
      (donor, acceptor, and non-splice-site padding).
    SPLICE_SITE_USAGE: Splice site usage tracks capturing the fraction of the
      time that each splice site is used.
    SPLICE_JUNCTIONS: Splice junction tracks capturing split read RNA-seq counts
      for each junction.
    CONTACT_MAPS: Relative 3D chromatin contact frequencies
      (observed-over-expected).
    SC_GENE_EXPRESSION: Protocol-specific single-cell RNA-seq expression tracks
      (in CPM) across training sequencing technologies, supported with both
      reference conditions (`condition=`) and expression conditioning
      (`cell_conditioning=`).
    SC_ATAC: ATAC-seq tracks capturing single-cell chromatin accessibility.
    SC_GENE_EXPRESSION_DENOISED: Denoised single-cell gene expression tracks
      (one channel per strand, in CPM) predicted from human expression
      conditioning inputs (`cell_conditioning=`).
    CELL_EMBEDDINGS: Learned low-dimensional cell state representations derived
      from single-cell gene expression profiles.
    METHYLATION: DNA methylation tracks capturing CpG methylation probabilities.
  """

  ATAC = dna_model_pb2.OUTPUT_TYPE_ATAC
  CAGE = dna_model_pb2.OUTPUT_TYPE_CAGE
  DNASE = dna_model_pb2.OUTPUT_TYPE_DNASE
  RNA_SEQ = dna_model_pb2.OUTPUT_TYPE_RNA_SEQ
  CHIP_HISTONE = dna_model_pb2.OUTPUT_TYPE_CHIP_HISTONE
  CHIP_TF = dna_model_pb2.OUTPUT_TYPE_CHIP_TF
  SPLICE_SITES = dna_model_pb2.OUTPUT_TYPE_SPLICE_SITES
  SPLICE_SITE_USAGE = dna_model_pb2.OUTPUT_TYPE_SPLICE_SITE_USAGE
  SPLICE_JUNCTIONS = dna_model_pb2.OUTPUT_TYPE_SPLICE_JUNCTIONS
  CONTACT_MAPS = dna_model_pb2.OUTPUT_TYPE_CONTACT_MAPS
  SC_GENE_EXPRESSION = dna_model_pb2.OUTPUT_TYPE_SC_GENE_EXPRESSION
  SC_ATAC = dna_model_pb2.OUTPUT_TYPE_SC_ATAC
  SC_GENE_EXPRESSION_DENOISED = (
      dna_model_pb2.OUTPUT_TYPE_SC_GENE_EXPRESSION_DENOISED
  )
  CELL_EMBEDDINGS = dna_model_pb2.OUTPUT_TYPE_CELL_EMBEDDINGS
  METHYLATION = dna_model_pb2.OUTPUT_TYPE_METHYLATION

  def __lt__(self, other: 'OutputType'):
    """Compares if an other `OutputType` enum value is less than this one."""
    return self.value < other.value

  def to_proto(self) -> dna_model_pb2.OutputType:
    """Converts the `OutputType` enum to a protobuf enum."""
    return self.value

  def __repr__(self) -> str:
    """Returns name of the `OutputType` enum as the string representation."""
    return self.name


@typing.jaxtyped
@dataclasses.dataclass(frozen=True)
class Output:
  """Model outputs for a single prediction.

  Attributes:
    atac: TrackData of type OutputType.ATAC.
    cage: TrackData of type OutputType.CAGE.
    dnase: TrackData of type OutputType.DNASE.
    rna_seq: TrackData of type OutputType.RNA_SEQ.
    chip_histone: TrackData of type OutputType.CHIP_HISTONE.
    chip_tf: TrackData of type OutputType.CHIP_TF.
    splice_sites: TrackData of type OutputType.SPLICE_SITES.
    splice_site_usage: TrackData of type OutputType.SPLICE_SITE_USAGE.
    splice_junctions: JunctionData of type OutputType.SPLICE_JUNCTIONS.
    contact_maps: TrackData of type OutputType.CONTACT_MAPS.
    sc_gene_expression: GeneExpressionData of type
      OutputType.SC_GENE_EXPRESSION.
    sc_atac: TrackData of type OutputType.SC_ATAC.
    sc_gene_expression_denoised: GeneExpressionData of type
      OutputType.SC_GENE_EXPRESSION_DENOISED (returned when requesting denoised
      expression with `cell_conditioning=`).
    cell_embeddings: Numpy array of cell embeddings.
    methylation: TrackData of type OutputType.METHYLATION.
  """

  atac: track_data.TrackData | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.ATAC}
  )
  cage: track_data.TrackData | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.CAGE}
  )
  dnase: track_data.TrackData | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.DNASE}
  )
  rna_seq: track_data.TrackData | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.RNA_SEQ}
  )
  chip_histone: track_data.TrackData | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.CHIP_HISTONE},
  )
  chip_tf: track_data.TrackData | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.CHIP_TF}
  )

  splice_sites: track_data.TrackData | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.SPLICE_SITES},
  )
  splice_site_usage: track_data.TrackData | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.SPLICE_SITE_USAGE},
  )
  splice_junctions: junction_data.JunctionData | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.SPLICE_JUNCTIONS},
  )

  contact_maps: track_data.TrackData | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.CONTACT_MAPS},
  )
  sc_gene_expression: gene_expression_data.GeneExpressionData | None = (
      dataclasses.field(
          default=None,
          metadata={'output_type': OutputType.SC_GENE_EXPRESSION},
      )
  )
  sc_atac: track_data.TrackData | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.SC_ATAC},
  )
  sc_gene_expression_denoised: (
      gene_expression_data.GeneExpressionData | None
  ) = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.SC_GENE_EXPRESSION_DENOISED},
  )
  cell_embeddings: np.ndarray | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.CELL_EMBEDDINGS},
  )
  methylation: track_data.TrackData | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.METHYLATION},
  )

  def get(
      self, output_type: OutputType
  ) -> (
      track_data.TrackData
      | junction_data.JunctionData
      | gene_expression_data.GeneExpressionData
      | None
  ):
    """Gets the track data for the specified output type.

    Args:
      output_type: The type of output to retrieve.

    Returns:
      The track data for the specified output type, or None if no such data
      exists.
    """
    for field in dataclasses.fields(self):
      if field.metadata['output_type'] == output_type:
        return getattr(self, field.name)

  def map_over_outputs(
      self,
      fn: Callable[
          [
              track_data.TrackData | gene_expression_data.GeneExpressionData,
              OutputType,
          ],
          track_data.TrackData | gene_expression_data.GeneExpressionData | None,
      ],
  ) -> 'Output':
    """Applies a transformation function to each `TrackData` or `GeneExpressionData`.

    Note: Currently Junctions will not be transformed, but returned as-is.

    Args:
      fn: The function to apply to each `TrackData` or `GeneExpressionData`. It
        should take a `TrackData` or `GeneExpressionData` object and an
        `OutputType` enum as input and return a corresponding `TrackData` or
        `GeneExpressionData` object or None.

    Returns:
      A new `Output` object with the transformed track data.
    """
    output_dict = {}
    for field in dataclasses.fields(self):
      output_type = field.metadata['output_type']
      value = self.get(output_type)
      match value:
        case track_data.TrackData() | gene_expression_data.GeneExpressionData():
          output_dict[field.name] = fn(value, output_type)
        case _:
          output_dict[field.name] = value
    return Output(**output_dict)

  def filter_to_strand(self, strand: Literal['+', '-', '.']) -> 'Output':
    """Filters tracks by DNA strand.

    Args:
      strand: The strand to filter by ('+', '-', or '.').

    Returns:
      A new `Output` object with only the tracks on the specified strand.
    """

    def _filter_to_strand(
        tdata: track_data.TrackData | gene_expression_data.GeneExpressionData,
        output_type: OutputType,
    ) -> track_data.TrackData | gene_expression_data.GeneExpressionData | None:
      del output_type  # Unused.
      return tdata.filter_tracks(tdata.strands == strand)

    return self.map_over_outputs(_filter_to_strand)

  def filter_to_observed(self) -> 'Output':
    """Filters tracks to observed tracks across all populated outputs.

    Filters `TrackData`, `GeneExpressionData`, and `JunctionData`
    (`splice_junctions`) fields to tracks where `imputation_mode == 'observed'`.

    Returns:
      A new `Output` object with only observed tracks.
    """

    def _filter_to_observed(
        tdata: track_data.TrackData | gene_expression_data.GeneExpressionData,
        output_type: OutputType,
    ) -> track_data.TrackData | gene_expression_data.GeneExpressionData | None:
      del output_type  # Unused.
      return tdata.filter_to_observed()

    output = self.map_over_outputs(_filter_to_observed)
    # Handle splice junctions that are skipped by map_over_outputs.
    if output.splice_junctions is not None:
      output = dataclasses.replace(
          output,
          splice_junctions=output.splice_junctions.filter_to_observed(),
      )
    return output

  def filter_output_type(self, output_types: Iterable[OutputType]) -> 'Output':
    """Filters tracks to specific output type.

    Args:
      output_types: An iterable of `OutputType` enums to filter by.

    Returns:
      A new `Output` object with only the tracks of the specified output types.
    """
    output_dict = {}
    for field in dataclasses.fields(self):
      output_type = field.metadata['output_type']
      if output_type in output_types:
        output_dict[field.name] = self.get(output_type)
      else:
        output_dict[field.name] = None

    return Output(**output_dict)

  def resize(self, width: int) -> 'Output':
    """Resizes all tracks to a specified width.

    Args:
      width: The desired width in base pairs.

    Returns:
      A new `Output` object with resized tracks.
    """
    return self.map_over_outputs(lambda tdata, _: tdata.resize(width))

  def __add__(self, other: 'Output') -> 'Output':
    """Adds the values of two `Output` objects element-wise.

    Args:
      other: The `Output` object to add.

    Returns:
      A new `Output` object with the summed values.
    """

    def add_output_data(
        output_data1,
        output_type: OutputType,
    ):
      """Adds two `TrackData` (or `GeneExpressionData`) objects for a specific output type."""
      output_data2 = other.get(output_type)
      if output_data1 is None or output_data2 is None:
        return None
      return output_data1 + output_data2

    return self.map_over_outputs(add_output_data)

  def __sub__(self, other: 'Output') -> 'Output':
    """Subtracts the values of two `Output` objects element-wise.

    Args:
      other: The `Output` object to subtract.

    Returns:
      A new `Output` object with the difference of the values.
    """

    def sub_output_data(output_data1, output_type: OutputType):
      """Subtracts two `TrackData` (or `GeneExpressionData`) objects for a specific output type."""
      output_data2 = other.get(output_type)
      if output_data1 is None or output_data2 is None:
        return None
      return output_data1 - output_data2

    return self.map_over_outputs(sub_output_data)

  def with_name_suffix(self, suffix: str) -> 'Output':
    """Returns a new `Output` object with suffixed track names on all tracks."""

    def _suffix_track(tdata, _):
      if hasattr(tdata, 'track_metadata'):
        new_metadata = tdata.track_metadata.copy()
        new_metadata['name'] = new_metadata['name'] + suffix
        return dataclasses.replace(tdata, track_metadata=new_metadata)
      new_metadata = tdata.metadata.copy()
      new_metadata['name'] = new_metadata['name'] + suffix
      return dataclasses.replace(tdata, metadata=new_metadata)

    output = self.map_over_outputs(_suffix_track)
    # Handle splice junctions that are skipped by map_over_outputs.
    if self.splice_junctions is not None:
      new_metadata = self.splice_junctions.metadata.copy()
      new_metadata['name'] = new_metadata['name'] + suffix
      output = dataclasses.replace(
          output,
          splice_junctions=dataclasses.replace(
              self.splice_junctions, metadata=new_metadata
          ),
      )
    return output

  def with_metadata_column(
      self,
      name: str,
      value: str | pd.Series,
  ) -> 'Output':
    """Returns a new `Output` with updated metadata column on all tracks."""

    def _add_metadata(tdata, _):
      if hasattr(tdata, 'track_metadata'):
        new_metadata = tdata.track_metadata.copy()
        new_metadata[name] = value
        return dataclasses.replace(tdata, track_metadata=new_metadata)
      new_metadata = tdata.metadata.copy()
      new_metadata[name] = value
      return dataclasses.replace(tdata, metadata=new_metadata)

    output = self.map_over_outputs(_add_metadata)
    # Handle splice junctions that are skipped by map_over_outputs.
    if self.splice_junctions is not None:
      new_metadata = self.splice_junctions.metadata.copy()
      new_metadata[name] = value
      output = dataclasses.replace(
          output,
          splice_junctions=dataclasses.replace(
              self.splice_junctions, metadata=new_metadata
          ),
      )
    return output


@dataclasses.dataclass(frozen=True)
class OutputMetadata:
  """Metadata detailing the content of model output.

  Attributes:
    atac: Metadata for ATAC-seq tracks.
    cage: Metadata for CAGE tracks.
    dnase: Metadata for DNase I hypersensitive site tracks.
    rna_seq: Metadata for RNA sequencing tracks.
    chip_histone: Metadata for ChIP-seq tracks capturing histone modifications.
    chip_tf: Metadata for ChIP-seq tracks capturing transcription factor
      binding.
    splice_sites: Metadata for splice site tracks.
    splice_site_usage: Metadata for splice site usage tracks.
    splice_junctions: Metadata for splice junction tracks.
    contact_maps: Metadata for contact map tracks.
    sc_gene_expression: Metadata for single-cell gene expression.
    sc_atac: Metadata for single-cell ATAC-seq tracks.
    sc_gene_expression_denoised: Metadata for denoised single-cell gene
      expression.
    methylation: Metadata for DNA methylation tracks.
  """

  atac: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.ATAC}
  )
  cage: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.CAGE}
  )
  dnase: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.DNASE}
  )
  rna_seq: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.RNA_SEQ}
  )
  chip_histone: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.CHIP_HISTONE}
  )
  chip_tf: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.CHIP_TF}
  )
  splice_sites: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.SPLICE_SITES}
  )
  splice_site_usage: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.SPLICE_SITE_USAGE}
  )
  splice_junctions: junction_data.JunctionMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.SPLICE_JUNCTIONS}
  )
  contact_maps: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.CONTACT_MAPS}
  )
  sc_gene_expression: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.SC_GENE_EXPRESSION}
  )
  sc_atac: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.SC_ATAC}
  )
  sc_gene_expression_denoised: track_data.TrackMetadata | None = (
      dataclasses.field(
          default=None,
          metadata={'output_type': OutputType.SC_GENE_EXPRESSION_DENOISED},
      )
  )
  methylation: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.METHYLATION}
  )

  def get(self, output: OutputType) -> track_data.TrackMetadata | None:
    """Gets the track metadata for a given output type.

    Args:
      output: The `OutputType` enum value.

    Returns:
      The corresponding track metadata, or None if it doesn't exist.
    """
    match output:
      case OutputType.ATAC:
        return self.atac
      case OutputType.CAGE:
        return self.cage
      case OutputType.DNASE:
        return self.dnase
      case OutputType.RNA_SEQ:
        return self.rna_seq
      case OutputType.CHIP_HISTONE:
        return self.chip_histone
      case OutputType.CHIP_TF:
        return self.chip_tf
      case OutputType.SPLICE_SITES:
        return self.splice_sites
      case OutputType.SPLICE_JUNCTIONS:
        return self.splice_junctions
      case OutputType.SPLICE_SITE_USAGE:
        return self.splice_site_usage
      case OutputType.CONTACT_MAPS:
        return self.contact_maps
      case OutputType.SC_GENE_EXPRESSION:
        return self.sc_gene_expression
      case OutputType.SC_ATAC:
        return self.sc_atac
      case OutputType.SC_GENE_EXPRESSION_DENOISED:
        return self.sc_gene_expression_denoised
      case OutputType.METHYLATION:
        return self.methylation
      case OutputType.CELL_EMBEDDINGS:
        return None

  def concatenate(self) -> track_data.TrackMetadata:
    """Concatenates all metadata into a single DataFrame.

    Returns:
      A pandas DataFrame containing all the track metadata, with an additional
      column 'output_type' specifying the type of each track.
    """
    df_list = []
    for output_type in OutputType:
      if (df := self.get(output_type)) is not None:
        assert isinstance(df, pd.DataFrame)
        df_list.append(df.assign(output_type=output_type))
    return pd.concat(df_list)  # pyrefly: ignore[bad-return]


@typing.jaxtyped
@dataclasses.dataclass(frozen=True)
class VariantOutput:
  """Model outputs for a variant prediction.

  Attributes:
    reference: The model output for the reference sequence.
    alternate: The model output for the alternate sequence.
  """

  reference: Output
  alternate: Output


@typing.jaxtyped
@dataclasses.dataclass(frozen=True)
class TargetGeneMetadata:
  """Metadata for a single target gene.

  Attributes:
    gene_id: Ensembl gene identifier (e.g. `'ENSG00000119866'`).
    interval: Genomic interval of the gene (`chromosome`, `start`, `end`, and
      `strand`).
    gene_name: Gene symbol (e.g. `'BCL11A'`), if populated.
    gene_type: Biotype (e.g. `'protein_coding'`), if populated.
  """

  gene_id: str
  interval: genome.Interval
  gene_name: str | None = None
  gene_type: str | None = None

  def __post_init__(self):
    if not self.gene_id:
      raise ValueError('TargetGeneMetadata `gene_id` must be non-empty.')

  @classmethod
  def from_proto(
      cls, proto: dna_model_pb2.GeneExpressionMetadata
  ) -> 'TargetGeneMetadata':
    """Creates a `TargetGeneMetadata` from a `GeneExpressionMetadata` proto."""
    strand = (
        str(genome.Strand.from_proto(proto.strand))
        if proto.HasField('strand')
        else genome.STRAND_UNSTRANDED
    )
    return cls(
        gene_id=proto.gene_id,
        interval=genome.Interval(
            chromosome=proto.chromosome,
            start=proto.start,
            end=proto.end,
            strand=strand,
        ),
        gene_name=proto.name if proto.HasField('name') else None,
        gene_type=proto.type if proto.HasField('type') else None,
    )


@typing.jaxtyped
@dataclasses.dataclass(frozen=True)
class InputGradientsOutput:
  """Model outputs for an input sequence gradient prediction.

  Attributes:
    gradients: Mean-centered input gradients times the one-hot input sequence,
      of shape `[sequence_length, 4]` (`float32`). Only the reference base is
      non-zero, so summing over the last axis gives per-position scores.
    interval: Genomic interval corresponding to the sequence, centered on the
      gene's median TSS. Positions beyond the chromosome ends are N-padded.
    scalar_score: Objective the gradients are taken of: the log of the target
      gene's prediction averaged over the tracks in `track_metadata` (summed
      over the gene's exon positions for track outputs such as `RNA_SEQ`).
    target_gene: Metadata of the target gene.
    track_metadata: Metadata for the tracks aggregated in the gradient target.
    output_type: Output type for which gradients were computed.
    gene_id: Ensembl gene identifier from `target_gene`.
    gene_name: Optional gene symbol from `target_gene`.
    biosample_name: Single biosample name shared across `track_metadata`.
  """

  gradients: np.ndarray
  interval: genome.Interval
  scalar_score: float
  target_gene: TargetGeneMetadata
  track_metadata: track_data.TrackMetadata
  output_type: OutputType

  def __post_init__(self):
    if self.gradients.shape != (self.interval.width, 4):
      raise ValueError(
          f'Expected gradients of shape ({self.interval.width}, 4), got'
          f' {self.gradients.shape}.'
      )
    if 'biosample_name' not in self.track_metadata.columns:
      raise ValueError('Track metadata does not contain `biosample_name`.')
    col = self.track_metadata['biosample_name']
    if not all(isinstance(v, str) and v for v in col) or col.nunique() != 1:
      raise ValueError(
          'Expected a single non-empty `biosample_name` across'
          f' `track_metadata`, got {col.tolist()!r}.'
      )

  @property
  def gene_id(self) -> str:
    """Returns the target Ensembl gene identifier."""
    return self.target_gene.gene_id

  @property
  def gene_name(self) -> str | None:
    """Returns the target gene symbol, or None if unset."""
    return self.target_gene.gene_name or None

  @property
  def biosample_name(self) -> str:
    """Returns the single biosample name shared across `track_metadata`."""
    return str(self.track_metadata['biosample_name'].iloc[0])

  def to_track_data(
      self,
      *,
      signed: bool = False,
      name: str | None = None,
  ) -> track_data.TrackData:
    """Converts per-nucleotide input gradients into a 1-bp resolution TrackData.

    Args:
      signed: If False (default), returns absolute gradient-times-input scores
        `|sum_b grad_x_input[pos, b]|`. If True, preserves the sign.
      name: Optional track name. When omitted, formatted as `'{gene_name or
        gene_id} input gradients ({biosample_name})'`.

    Returns:
      A 1-bp resolution `TrackData` of shape `[sequence_length, 1]`.
    """
    raw_values = self.gradients.sum(axis=-1, keepdims=True).astype(np.float32)
    values = raw_values if signed else np.abs(raw_values)

    gene_id = self.gene_id
    gene_name = self.gene_name
    biosample_name = self.biosample_name
    track_name = (
        name or f'{gene_name or gene_id} input gradients ({biosample_name})'
    )

    meta_dict: dict[str, list[str]] = {
        'name': [track_name],
        'strand': ['.'],
        'gene_id': [gene_id],
        'biosample_name': [biosample_name],
    }
    if gene_name is not None:
      meta_dict['gene_name'] = [gene_name]

    return track_data.TrackData(
        values=values,
        metadata=pd.DataFrame(meta_dict),
        resolution=1,
        interval=self.interval,
    )
