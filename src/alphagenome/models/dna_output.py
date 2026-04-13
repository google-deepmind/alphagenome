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
"""Module for AlphaGenome model outputs."""

import dataclasses
import enum
from collections.abc import Callable, Iterable, Mapping
from typing import Literal

import pandas as pd

from alphagenome import typing
from alphagenome.data import junction_data, ontology, track_data
from alphagenome.protos import dna_model_pb2


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
    SPLICE_SITES: Splice site tracks capturing donor and acceptor splice sites.
    SPLICE_SITE_USAGE: Splice site usage tracks capturing the fraction of the
      time that each splice site is used.
    SPLICE_JUNCTIONS: Splice junction tracks capturing split read RNA-seq counts
      for each junction.
    CONTACT_MAPS: Contact map tracks capturing 3D DNA-DNA contact probabilities.
    PROCAP: Precision Run-On sequencing and capping, used to measure gene
      expression.
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
  PROCAP = dna_model_pb2.OUTPUT_TYPE_PROCAP

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
    splice_junctions: TrackData of type OutputType.SPLICE_JUNCTIONS.
    contact_maps: TrackData of type OutputType.CONTACT_MAPS.
    procap: TrackData of type OutputType.PROCAP.
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
  procap: track_data.TrackData | None = dataclasses.field(
      default=None,
      metadata={'output_type': OutputType.PROCAP},
  )

  def get(
      self, output_type: OutputType
  ) -> track_data.TrackData | junction_data.JunctionData | None:
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

  def map_track_data(
      self,
      fn: Callable[
          [track_data.TrackData, OutputType],
          track_data.TrackData | None,
      ],
  ) -> 'Output':
    """Applies a transformation function to each `TrackData`.

    Args:
      fn: The function to apply to each `TrackData`. It should take a
        `TrackData` object and an `OutputType` enum as input and return a
        `TrackData` object or None.

    Returns:
      A new `Output` object with the transformed track data.
    """
    output_dict = {}
    for field in dataclasses.fields(self):
      output_type = field.metadata['output_type']
      value = self.get(output_type)
      if isinstance(value, track_data.TrackData):
        output_dict[field.name] = fn(value, output_type)
      else:
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
        tdata: track_data.TrackData, output_type: OutputType
    ) -> track_data.TrackData | None:
      del output_type  # Unused.
      return tdata.filter_tracks(tdata.strands == strand)

    output = self.map_track_data(_filter_to_strand)
    if output.splice_junctions is not None:
      output = dataclasses.replace(
          output,
          splice_junctions=output.splice_junctions.filter_to_strand(strand),
      )
    return output

  def filter_ontology_terms(
      self, ontology_terms: Iterable[ontology.OntologyTerm]
  ) -> 'Output':
    """Filters tracks to specific ontology terms.

    Args:
      ontology_terms: An iterable of `OntologyTerm` objects to filter to.

    Returns:
      A new `Output` object with only the tracks associated with the specified
      ontology terms.
    """

    def _filter_ontology(
        tdata: track_data.TrackData, output_type: OutputType
    ) -> track_data.TrackData | None:
      del output_type  # Unused.
      if track_ontologies := tdata.ontology_terms:
        return tdata.filter_tracks(
            [o in ontology_terms for o in track_ontologies]
        )
      else:
        return tdata

    return self.map_track_data(_filter_ontology)

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
    """Resizes all track data to a specified width.

    Args:
      width: The desired width in base pairs.

    Returns:
      A new `Output` object with resized track data.
    """
    return self.map_track_data(lambda tdata, _: tdata.resize(width))

  def __add__(self, other: 'Output') -> 'Output':
    """Adds the values of two `Output` objects element-wise.

    Args:
      other: The `Output` object to add.

    Returns:
      A new `Output` object with the summed values.
    """

    def add_track_data(
        track_data1,
        output_type: OutputType,
    ):
      """Adds two `TrackData` objects for a specific output type."""
      track_data2 = other.get(output_type)
      if track_data1 is None or track_data2 is None:
        return None
      return track_data1 + track_data2

    return self.map_track_data(add_track_data)

  def __sub__(self, other: 'Output') -> 'Output':
    """Subtracts the values of two `Output` objects element-wise.

    Args:
      other: The `Output` object to subtract.

    Returns:
      A new `Output` object with the difference of the values.
    """

    def sub_track_data(track_data1, output_type: OutputType):
      """Subtracts two `TrackData` objects for a specific output type."""
      track_data2 = other.get(output_type)
      if track_data1 is None or track_data2 is None:
        return None
      return track_data1 - track_data2

    return self.map_track_data(sub_track_data)


@dataclasses.dataclass(frozen=True, kw_only=True)
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
    procap: Metadata for procap tracks.
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
  procap: track_data.TrackMetadata | None = dataclasses.field(
      default=None, metadata={'output_type': OutputType.PROCAP}
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
      case OutputType.PROCAP:
        return self.procap

  @classmethod
  def from_outputs(
      cls,
      outputs: Mapping[
          OutputType, track_data.TrackData | junction_data.JunctionData
      ],
  ) -> 'OutputMetadata':
    """Creates an `OutputMetadata` from a mapping of output types to data.

    Args:
      outputs: A mapping from `OutputType` to `TrackData` or `JunctionData`.

    Returns:
      An `OutputMetadata` object.
    """
    kwargs = {}
    for field in dataclasses.fields(cls):
      output_type = field.metadata['output_type']
      if data := outputs.get(output_type):
        kwargs[field.name] = data.metadata
    return cls(**kwargs)

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
    return pd.concat(df_list)


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
