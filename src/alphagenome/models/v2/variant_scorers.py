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

"""Module containing variant scorer dataclasses for variant scoring."""

from collections.abc import Sequence
import dataclasses
import enum
import itertools
from typing import TypeAlias

from alphagenome.protos.v1 import dna_model_pb2 as dna_model_pb2_v1
from alphagenome.models.v2 import dna_output
from alphagenome.protos.v2 import dna_model_pb2 as dna_model_pb2_v2
import anndata
import immutabledict
import pandas as pd


class AggregationType(enum.Enum):
  """Enum indicating the type of variant scorer aggregation.

  Attributes:
    DIFF_MEAN: Difference of means, i.e., mean(`ALT`) - mean(`REF`).
    DIFF_SUM: Difference of sums, i.e., sum(`ALT`) - sum(`REF`).
    DIFF_SUM_LOG2: Log scales predictions, then takes the sum and then the
      difference, i.e., sum(log2(`ALT`)) - sum(log2(`REF`)).
    DIFF_LOG2_SUM: Takes the sum of predictions, applies a log transform, then
      takes the difference between predictions, i.e., log2(sum(`ALT`)) -
      log2(sum(`REF`)).
    L2_DIFF: Takes the difference of `ALT` and `REF` predictions, then computes
      the L2 norm, i.e., l2_norm(`ALT` - `REF`).
    L2_DIFF_LOG1P: Log scales the predictions + 1, takes the difference of `ALT`
      and `REF` predictions, then computes the L2 norm, i.e.,
      l2_norm(log1p(`ALT`) - log1p(`REF`)).
    ACTIVE_MEAN: Maximum of means, i.e., max(mean(`ALT`), mean(`REF`)).
    ACTIVE_SUM: Maximum of sums, i.e., max(sum(`ALT`), sum(`REF`)).

  Methods:
    to_proto: Converts the aggregation type to its corresponding proto enum.
  """

  DIFF_MEAN = dna_model_pb2_v2.AggregationType.AGGREGATION_TYPE_DIFF_MEAN
  DIFF_SUM = dna_model_pb2_v2.AggregationType.AGGREGATION_TYPE_DIFF_SUM
  DIFF_SUM_LOG2 = (
      dna_model_pb2_v2.AggregationType.AGGREGATION_TYPE_DIFF_SUM_LOG2
  )
  DIFF_LOG2_SUM = (
      dna_model_pb2_v2.AggregationType.AGGREGATION_TYPE_DIFF_LOG2_SUM
  )
  L2_DIFF = dna_model_pb2_v2.AggregationType.AGGREGATION_TYPE_L2_DIFF
  L2_DIFF_LOG1P = (
      dna_model_pb2_v2.AggregationType.AGGREGATION_TYPE_L2_DIFF_LOG1P
  )
  ACTIVE_MEAN = dna_model_pb2_v2.AggregationType.AGGREGATION_TYPE_ACTIVE_MEAN
  ACTIVE_SUM = dna_model_pb2_v2.AggregationType.AGGREGATION_TYPE_ACTIVE_SUM

  def to_proto(self) -> dna_model_pb2_v2.AggregationType:
    return self.value

  def __repr__(self) -> str:
    return self.name


class BaseVariantScorer(enum.Enum):
  """Enum indicating the type of variant scoring.

  Attributes:
    CENTER_MASK: Center mask scorer.
    CONTACT_MAP: Contact map scorer, a center mask scorer that handles 2D
      tensors.
    GENE_MASK_LFC: Gene-mask scoring based on log fold change.
    GENE_MASK_ACTIVE: Gene-mask scoring based on active allele.
    GENE_MASK_SPLICING: Gene-mask scoring for splicing.
    METHYLATION: CpG methylation scoring.
    PA_QTL: Polyadenylation QTL scoring.
    SPLICE_JUNCTION: Splice junction scoring.
  """

  CENTER_MASK = enum.auto()
  CONTACT_MAP = enum.auto()
  GENE_MASK_LFC = enum.auto()
  GENE_MASK_ACTIVE = enum.auto()
  GENE_MASK_SPLICING = enum.auto()
  PA_QTL = enum.auto()
  SPLICE_JUNCTION = enum.auto()
  METHYLATION = enum.auto()


SUPPORTED_ORGANISMS = immutabledict.immutabledict({
    BaseVariantScorer.CENTER_MASK: dna_model_pb2_v1.Organism.values(),
    BaseVariantScorer.CONTACT_MAP: dna_model_pb2_v1.Organism.values(),
    BaseVariantScorer.GENE_MASK_LFC: dna_model_pb2_v1.Organism.values(),
    BaseVariantScorer.GENE_MASK_ACTIVE: dna_model_pb2_v1.Organism.values(),
    BaseVariantScorer.GENE_MASK_SPLICING: dna_model_pb2_v1.Organism.values(),
    BaseVariantScorer.METHYLATION: dna_model_pb2_v1.Organism.values(),
    BaseVariantScorer.PA_QTL: [dna_model_pb2_v1.Organism.ORGANISM_HOMO_SAPIENS],
    BaseVariantScorer.SPLICE_JUNCTION: dna_model_pb2_v1.Organism.values(),
})

SUPPORTED_OUTPUT_TYPES = immutabledict.immutabledict({
    BaseVariantScorer.CENTER_MASK: [
        dna_output.OutputType.ATAC,
        dna_output.OutputType.CAGE,
        dna_output.OutputType.DNASE,
        dna_output.OutputType.RNA_SEQ,
        dna_output.OutputType.CHIP_HISTONE,
        dna_output.OutputType.CHIP_TF,
        dna_output.OutputType.SPLICE_SITES,
        dna_output.OutputType.SPLICE_SITE_USAGE,
        dna_output.OutputType.SC_ATAC,
        dna_output.OutputType.METHYLATION,
    ],
    BaseVariantScorer.CONTACT_MAP: [dna_output.OutputType.CONTACT_MAPS],
    BaseVariantScorer.GENE_MASK_LFC: [
        dna_output.OutputType.ATAC,
        dna_output.OutputType.CAGE,
        dna_output.OutputType.DNASE,
        dna_output.OutputType.RNA_SEQ,
        dna_output.OutputType.SPLICE_SITES,
        dna_output.OutputType.SPLICE_SITE_USAGE,
        dna_output.OutputType.SC_ATAC,
        dna_output.OutputType.SC_GENE_EXPRESSION,
        dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED,
        dna_output.OutputType.METHYLATION,
    ],
    BaseVariantScorer.GENE_MASK_ACTIVE: [
        dna_output.OutputType.ATAC,
        dna_output.OutputType.CAGE,
        dna_output.OutputType.DNASE,
        dna_output.OutputType.RNA_SEQ,
        dna_output.OutputType.SPLICE_SITES,
        dna_output.OutputType.SPLICE_SITE_USAGE,
        dna_output.OutputType.SC_ATAC,
        dna_output.OutputType.SC_GENE_EXPRESSION,
        dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED,
        dna_output.OutputType.METHYLATION,
    ],
    BaseVariantScorer.GENE_MASK_SPLICING: [
        dna_output.OutputType.SPLICE_SITES,
        dna_output.OutputType.SPLICE_SITE_USAGE,
    ],
    BaseVariantScorer.METHYLATION: [dna_output.OutputType.METHYLATION],
})

SUPPORTED_WIDTHS = immutabledict.immutabledict({
    BaseVariantScorer.CENTER_MASK: [
        None,
        501,
        1201,
        2001,
        10_001,
        100_001,
        200_001,
    ],
    BaseVariantScorer.GENE_MASK_SPLICING: [None, 101, 1_001, 10_001],
    BaseVariantScorer.METHYLATION: [1_201],
})

SUPPORTED_AGGREGATIONS = immutabledict.immutabledict({
    BaseVariantScorer.CENTER_MASK: [
        AggregationType.DIFF_MEAN,
        AggregationType.DIFF_SUM,
        AggregationType.DIFF_SUM_LOG2,
        AggregationType.DIFF_LOG2_SUM,
        AggregationType.L2_DIFF,
        AggregationType.L2_DIFF_LOG1P,
        AggregationType.ACTIVE_MEAN,
        AggregationType.ACTIVE_SUM,
    ],
})


@dataclasses.dataclass(frozen=True)
class CenterMaskScorer:
  """Variant scorer for center mask scorers.

  Variant scorer that aggregates ALT and REF values using a spatial mask
  centered around the variant before computing the difference. This scorer
  aggregates over the spatial axis and returns one score per output track.

  Attributes:
    requested_output: The requested output type (e.g., ATAC, DNASE, etc.)
    width: The width of the mask around the variant. If None, the score is
      computed over the entire sequence.
    aggregation_type: The aggregation type.
    base_variant_scorer: The base variant scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).
    is_signed: Whether this variant scorer is directional (such that scores may
      be negative or positive) or non-directional (scores are always
      non-negative).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.

  Raises:
    ValueError: If the requested output, width, or aggregation type is not
      supported.
  """

  requested_output: dna_output.OutputType
  width: int | None
  aggregation_type: AggregationType

  @property
  def base_variant_scorer(self) -> BaseVariantScorer:
    """Returns the base variant scorer enum value."""
    return BaseVariantScorer.CENTER_MASK

  @property
  def name(self) -> str:
    """Returns the unique string name of this variant scorer."""
    return str(self)

  @property
  def is_signed(self) -> bool:
    """Returns True if scores are directional (signed) rather than unsigned."""
    # The 'active' scorers track the maximum of the ALT and REF, and are
    # non-directional. All other scorers track ALT - REF, with L2_DIFF
    # nondirectional and the remaining directional.
    return self.aggregation_type in [
        AggregationType.DIFF_MEAN,
        AggregationType.DIFF_SUM,
        AggregationType.DIFF_SUM_LOG2,
        AggregationType.DIFF_LOG2_SUM,
    ]

  def __post_init__(self):
    if (
        self.requested_output
        not in SUPPORTED_OUTPUT_TYPES[self.base_variant_scorer]
    ):
      raise ValueError(
          f'Unsupported requested output: {self.requested_output}. Supported'
          f' output types: {SUPPORTED_OUTPUT_TYPES[self.base_variant_scorer]}'
      )
    if self.width not in SUPPORTED_WIDTHS[self.base_variant_scorer]:
      raise ValueError(
          f'Unsupported width: {self.width}. Supported widths:'
          f' {SUPPORTED_WIDTHS[self.base_variant_scorer]}'
      )
    if (
        self.aggregation_type
        not in SUPPORTED_AGGREGATIONS[self.base_variant_scorer]
    ):
      raise ValueError(
          f'Unsupported aggregation type: {self.aggregation_type}. Supported'
          ' aggregation types:'
          f' {SUPPORTED_AGGREGATIONS[self.base_variant_scorer]}'
      )

  def to_proto(self) -> dna_model_pb2_v2.VariantScorer:
    return dna_model_pb2_v2.VariantScorer(
        center_mask=dna_model_pb2_v2.CenterMaskScorer(
            requested_output=self.requested_output.to_proto(),
            width=self.width,
            aggregation_type=self.aggregation_type.to_proto(),
        )
    )


@dataclasses.dataclass(frozen=True)
class ContactMapScorer:
  """Variant scorer for contact map scorers.

  Variant scorer that quantifies local contact disruption between ALT and REF
  alleles. Uses a 1 Mb window centered around the variant to calculate the
  mean absolute difference of contact frequencies, for all interactions
  involving the variant-containing bin.

  Attributes:
    requested_output: The requested output type (e.g., CONTACT_MAPS).
    base_variant_scorer: The base variant scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).
    is_signed: Whether this variant scorer is directional (such that scores may
      be negative or positive) or non-directional (scores are always
      non-negative).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.
  """

  @property
  def base_variant_scorer(self) -> BaseVariantScorer:
    """Returns the base variant scorer enum value."""
    return BaseVariantScorer.CONTACT_MAP

  @property
  def name(self) -> str:
    """Returns the unique string name of this variant scorer."""
    return str(self)

  @property
  def is_signed(self) -> bool:
    """Returns True if scores are directional (signed) rather than unsigned."""
    return False

  def to_proto(self) -> dna_model_pb2_v2.VariantScorer:
    return dna_model_pb2_v2.VariantScorer(
        contact_map=dna_model_pb2_v2.ContactMapScorer()
    )

  @property
  def requested_output(self) -> dna_output.OutputType:
    """Returns the `OutputType` scored by this variant scorer."""
    return dna_output.OutputType.CONTACT_MAPS


@dataclasses.dataclass(frozen=True)
class GeneMaskLFCScorer:
  """Variant scorer for gene-mask log fold change scoring.

  Variant scorer that quantifies impact on overall gene transcript abundance.
  Calculates the log fold change of gene expression level between ALT and REF
  alleles using a gene exon mask.

  Attributes:
    requested_output: The requested output type (e.g. ATAC, DNASE, etc.)
    aggregation_type: The aggregation type (DIFF_MEAN or DIFF_SUM).
    base_variant_scorer: The base variant scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).
    is_signed: Whether this variant scorer is directional (such that scores may
      be negative or positive) or non-directional (scores are always
      non-negative).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.

  Raises:
    ValueError: If the requested output is not supported.
  """

  requested_output: dna_output.OutputType
  aggregation_type: AggregationType

  @property
  def base_variant_scorer(self) -> BaseVariantScorer:
    """Returns the base variant scorer enum value."""
    return BaseVariantScorer.GENE_MASK_LFC

  @property
  def name(self) -> str:
    """Returns the unique string name of this variant scorer."""
    return str(self)

  @property
  def is_signed(self) -> bool:
    """Returns True if scores are directional (signed) rather than unsigned."""
    return True

  def __post_init__(self):
    if (
        self.requested_output
        not in SUPPORTED_OUTPUT_TYPES[self.base_variant_scorer]
    ):
      raise ValueError(
          f'Unsupported requested output: {self.requested_output}. Supported'
          f' output types: {SUPPORTED_OUTPUT_TYPES[self.base_variant_scorer]}'
      )

  def to_proto(self) -> dna_model_pb2_v2.VariantScorer:
    return dna_model_pb2_v2.VariantScorer(
        gene_mask=dna_model_pb2_v2.GeneMaskLFCScorer(
            requested_output=self.requested_output.to_proto(),
            aggregation_type=self.aggregation_type.to_proto(),
        )
    )


@dataclasses.dataclass(frozen=True)
class GeneMaskActiveScorer:
  """Variant scorer for gene-mask active allele scoring.

  Variant scorer that captures absolute activity level associated with one of
  the alleles, rather than quantifying the difference between the ALT and REF.
  Active allele gene scores are calculated by taking the maximum of the
  aggregated ALT and REF signals across exons for a gene of interest.

  Attributes:
    requested_output: The requested output type (e.g. ATAC, DNASE, etc.)
    base_variant_scorer: The base variant scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).
    is_signed: Whether this variant scorer is directional (such that scores may
      be negative or positive) or non-directional (scores are always
      non-negative).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.

  Raises:
    ValueError: If the requested output is not supported.
  """

  requested_output: dna_output.OutputType

  @property
  def base_variant_scorer(self) -> BaseVariantScorer:
    """Returns the base variant scorer enum value."""
    return BaseVariantScorer.GENE_MASK_ACTIVE

  @property
  def name(self) -> str:
    """Returns the unique string name of this variant scorer."""
    return str(self)

  @property
  def is_signed(self) -> bool:
    """Returns True if scores are directional (signed) rather than unsigned."""
    return False

  def __post_init__(self):
    if (
        self.requested_output
        not in SUPPORTED_OUTPUT_TYPES[self.base_variant_scorer]
    ):
      raise ValueError(
          f'Unsupported requested output: {self.requested_output}. Supported'
          f' output types: {SUPPORTED_OUTPUT_TYPES[self.base_variant_scorer]}'
      )

  def to_proto(self) -> dna_model_pb2_v2.VariantScorer:
    return dna_model_pb2_v2.VariantScorer(
        gene_mask_active=dna_model_pb2_v2.GeneMaskActiveScorer(
            requested_output=self.requested_output.to_proto(),
        )
    )


@dataclasses.dataclass(frozen=True)
class GeneMaskSplicingScorer:
  """Variant scorer for gene-mask scoring for splicing.

  Variant scorer that quantifies changes in class assignment probabilities
  (`dna_output.OutputType.SPLICE_SITES`) or changes in the usage of splice sites
  (`dna_output.OutputType.SPLICE_SITE_USAGE`) between `ALT` and `REF` alleles
  using a gene body mask (exons and introns).

  Attributes:
    requested_output: The requested output type (`SPLICE_SITES` or
      `SPLICE_SITE_USAGE`).
    width: The width of the mask around the variant, or `None` for the full gene
      body.
    base_variant_scorer: The base variant scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).
    is_signed: Whether this variant scorer is directional (such that scores may
      be negative or positive) or non-directional (scores are always
      non-negative).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.

  Raises:
    ValueError: If the requested output or width is not supported.
  """

  requested_output: dna_output.OutputType
  width: int | None

  @property
  def base_variant_scorer(self) -> BaseVariantScorer:
    """Returns the base variant scorer enum value."""
    return BaseVariantScorer.GENE_MASK_SPLICING

  @property
  def name(self) -> str:
    """Returns the unique string name of this variant scorer."""
    return str(self)

  @property
  def is_signed(self) -> bool:
    """Returns True if scores are directional (signed) rather than unsigned."""
    return False

  def __post_init__(self):
    if (
        self.requested_output
        not in SUPPORTED_OUTPUT_TYPES[self.base_variant_scorer]
    ):
      raise ValueError(
          f'Unsupported requested output: {self.requested_output}. Supported'
          f' output types: {SUPPORTED_OUTPUT_TYPES[self.base_variant_scorer]}'
      )
    if self.width not in SUPPORTED_WIDTHS[self.base_variant_scorer]:
      raise ValueError(
          f'Unsupported width: {self.width}. Supported widths:'
          f' {SUPPORTED_WIDTHS[self.base_variant_scorer]}'
      )

  def to_proto(self) -> dna_model_pb2_v2.VariantScorer:
    return dna_model_pb2_v2.VariantScorer(
        gene_mask_splicing=dna_model_pb2_v2.GeneMaskSplicingScorer(
            requested_output=self.requested_output.to_proto(),
            width=self.width,
        )
    )


@dataclasses.dataclass(frozen=True)
class MethylationScorer:
  """Variant scorer for CpG methylation.

  Variant scorer that quantifies a variant's impact on CpG methylation. Scores
  are reported per CpG site found in the reference sequence within a window
  centered on the variant. Rows of the returned data are CpG positions.

  Only CpG sites present in both the REF and ALT sequences are scored. Sites
  created or destroyed by the variant are omitted, since no ALT-versus-REF
  difference is defined for them.

  Attributes:
    width: The full width of the window around the variant within which CpG
      sites are scored, i.e. 2 * radius + 1.
    requested_output: The requested output type (always METHYLATION).
    base_variant_scorer: The base variant scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).
    is_signed: Whether this variant scorer is directional (such that scores may
      be negative or positive) or non-directional (scores are always
      non-negative).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.

  Raises:
    ValueError: If the width is not supported.
  """

  width: int

  @property
  def requested_output(self) -> dna_output.OutputType:
    """Returns the scorer's underlying output type."""
    return dna_output.OutputType.METHYLATION

  @property
  def base_variant_scorer(self) -> BaseVariantScorer:
    """Returns the base variant scorer enum value."""
    return BaseVariantScorer.METHYLATION

  @property
  def name(self) -> str:
    """Returns the unique string name of this variant scorer."""
    return str(self)

  @property
  def is_signed(self) -> bool:
    """Returns True if scores are directional (signed) rather than unsigned."""
    return True

  def __post_init__(self):
    if self.width not in SUPPORTED_WIDTHS[self.base_variant_scorer]:
      raise ValueError(
          f'Unsupported width: {self.width}. Supported widths:'
          f' {SUPPORTED_WIDTHS[self.base_variant_scorer]}'
      )

  def to_proto(self) -> dna_model_pb2_v2.VariantScorer:
    return dna_model_pb2_v2.VariantScorer(
        methylation=dna_model_pb2_v2.MethylationScorer(width=self.width)
    )


@dataclasses.dataclass(frozen=True)
class PolyadenylationScorer:
  """Variant scorer for polyadenylation QTLs (paQTLs).

  Variant scorer for polyadenylation quantitative trait loci (paQTLs) to capture
  a variant's impact on RNA isoform production.

  The scoring approach quantifies the maximum log fold change in expression
  between the set of proximal polyadenylation sites (PAS)s vs. the set of
  distal PASs, where the sets of proximal and distal PASs are the PASs upstream
  or downstream of any given 3' cleavage site respectively.

  Attributes:
    base_variant_scorer: The base variant scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).
    is_signed: Whether this variant scorer is directional (such that scores may
      be negative or positive) or non-directional (scores are always
      non-negative).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.
  """

  @property
  def requested_output(self) -> dna_output.OutputType:
    """Returns the scorer's underlying output type."""
    return dna_output.OutputType.RNA_SEQ

  @property
  def base_variant_scorer(self) -> BaseVariantScorer:
    """Returns the base variant scorer enum value."""
    return BaseVariantScorer.PA_QTL

  @property
  def name(self) -> str:
    """Returns the unique string name of this variant scorer."""
    return str(self)

  @property
  def is_signed(self) -> bool:
    """Returns True if scores are directional (signed) rather than unsigned."""
    return False

  def to_proto(self) -> dna_model_pb2_v2.VariantScorer:
    return dna_model_pb2_v2.VariantScorer(
        pa_qtl=dna_model_pb2_v2.PolyadenylationScorer()
    )


@dataclasses.dataclass(frozen=True)
class SpliceJunctionScorer:
  """Variant scorer for splice junction scoring.

  Attributes:
    base_variant_scorer: The base variant scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).
    is_signed: Whether this variant scorer is directional (such that scores may
      be negative or positive) or non-directional (scores are always
      non-negative).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.
  """

  @property
  def requested_output(self) -> dna_output.OutputType:
    """Returns the scorer's underlying output type."""
    return dna_output.OutputType.SPLICE_JUNCTIONS

  @property
  def base_variant_scorer(self) -> BaseVariantScorer:
    """Returns the base variant scorer enum value."""
    return BaseVariantScorer.SPLICE_JUNCTION

  @property
  def name(self) -> str:
    """Returns the unique string name of this variant scorer."""
    return str(self)

  @property
  def is_signed(self) -> bool:
    """Returns True if scores are directional (signed) rather than unsigned."""
    return False

  def to_proto(self) -> dna_model_pb2_v2.VariantScorer:
    return dna_model_pb2_v2.VariantScorer(
        splice_junction=dna_model_pb2_v2.SpliceJunctionScorer()
    )


# Type alias for all variant scorer types.
VariantScorerTypes: TypeAlias = (
    CenterMaskScorer
    | ContactMapScorer
    | GeneMaskLFCScorer
    | GeneMaskActiveScorer
    | GeneMaskSplicingScorer
    | MethylationScorer
    | PolyadenylationScorer
    | SpliceJunctionScorer
)
"""Union type alias of supported AlphaGenome 2 variant scorer configurations."""

# A dict of variant scorers, with our recommended settings for a wide range
# of different use cases and settings.
RECOMMENDED_VARIANT_SCORERS = immutabledict.immutabledict({
    # keep-sorted start
    'ATAC': CenterMaskScorer(
        requested_output=dna_output.OutputType.ATAC,
        width=501,
        aggregation_type=AggregationType.DIFF_LOG2_SUM,
    ),
    'ATAC_ACTIVE': CenterMaskScorer(
        requested_output=dna_output.OutputType.ATAC,
        width=501,
        aggregation_type=AggregationType.ACTIVE_SUM,
    ),
    'CAGE': CenterMaskScorer(
        requested_output=dna_output.OutputType.CAGE,
        width=501,
        aggregation_type=AggregationType.DIFF_LOG2_SUM,
    ),
    'CAGE_ACTIVE': CenterMaskScorer(
        requested_output=dna_output.OutputType.CAGE,
        width=501,
        aggregation_type=AggregationType.ACTIVE_SUM,
    ),
    'CHIP_HISTONE': CenterMaskScorer(
        requested_output=dna_output.OutputType.CHIP_HISTONE,
        width=2001,
        aggregation_type=AggregationType.DIFF_LOG2_SUM,
    ),
    'CHIP_HISTONE_ACTIVE': CenterMaskScorer(
        requested_output=dna_output.OutputType.CHIP_HISTONE,
        width=2001,
        aggregation_type=AggregationType.ACTIVE_SUM,
    ),
    'CHIP_TF': CenterMaskScorer(
        requested_output=dna_output.OutputType.CHIP_TF,
        width=501,
        aggregation_type=AggregationType.DIFF_LOG2_SUM,
    ),
    'CHIP_TF_ACTIVE': CenterMaskScorer(
        requested_output=dna_output.OutputType.CHIP_TF,
        width=501,
        aggregation_type=AggregationType.ACTIVE_SUM,
    ),
    'CONTACT_MAPS': ContactMapScorer(),
    'DNASE': CenterMaskScorer(
        requested_output=dna_output.OutputType.DNASE,
        width=501,
        aggregation_type=AggregationType.DIFF_LOG2_SUM,
    ),
    'DNASE_ACTIVE': CenterMaskScorer(
        requested_output=dna_output.OutputType.DNASE,
        width=501,
        aggregation_type=AggregationType.ACTIVE_SUM,
    ),
    'METHYLATION': MethylationScorer(width=1201),
    'POLYADENYLATION': PolyadenylationScorer(),
    # Expression log fold change.
    'RNA_SEQ': GeneMaskLFCScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
        aggregation_type=AggregationType.DIFF_MEAN,
    ),
    'RNA_SEQ_ACTIVE': GeneMaskActiveScorer(
        requested_output=dna_output.OutputType.RNA_SEQ
    ),
    'SC_ATAC': CenterMaskScorer(
        requested_output=dna_output.OutputType.SC_ATAC,
        width=1201,
        aggregation_type=AggregationType.DIFF_LOG2_SUM,
    ),
    'SC_ATAC_ACTIVE': CenterMaskScorer(
        requested_output=dna_output.OutputType.SC_ATAC,
        width=1201,
        aggregation_type=AggregationType.ACTIVE_SUM,
    ),
    'SC_GENE_EXPRESSION': GeneMaskLFCScorer(
        requested_output=dna_output.OutputType.SC_GENE_EXPRESSION,
        aggregation_type=AggregationType.DIFF_SUM,
    ),
    'SC_GENE_EXPRESSION_ACTIVE': GeneMaskActiveScorer(
        requested_output=dna_output.OutputType.SC_GENE_EXPRESSION,
    ),
    'SC_GENE_EXPRESSION_DENOISED': GeneMaskLFCScorer(
        requested_output=dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED,
        aggregation_type=AggregationType.DIFF_SUM,
    ),
    'SPLICE_JUNCTIONS': SpliceJunctionScorer(),
    'SPLICE_SITES': GeneMaskSplicingScorer(
        requested_output=dna_output.OutputType.SPLICE_SITES,
        width=None,
    ),
    'SPLICE_SITE_USAGE': GeneMaskSplicingScorer(
        requested_output=dna_output.OutputType.SPLICE_SITE_USAGE,
        width=None,
    ),
    # keep-sorted end
})
"""Immutable mapping from a short name (e.g. `'RNA_SEQ'`, `'ATAC'`,
`'SC_GENE_EXPRESSION_DENOISED'`, `'SC_GENE_EXPRESSION'`) to a variant scorer
configured with our recommended settings for that output type. Each output type
has a log-fold-change style scorer and, where applicable, an `_ACTIVE` variant
that scores absolute activity. Use
:func:`~alphagenome.models.v2.variant_scorers.get_recommended_scorers` to get
the subset supported for a given organism and request type.
"""


# Output types whose scorers the server only accepts with expression
# conditioning (`cell_conditioning=...`), not with a reference `condition=`.
CELL_CONDITIONING_ONLY_OUTPUT_TYPES = frozenset(
    {dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED}
)


def get_recommended_scorers(
    organism: dna_model_pb2_v1.Organism,
    cell_conditioning: bool = False,
) -> list[VariantScorerTypes]:
  """Returns the recommended variant scorers for a given organism.

  Args:
    organism: Organism to return recommended scorers for.
    cell_conditioning: Whether the scorers will be used with expression
      conditioning (`cell_conditioning=`). When True, includes
      `'SC_GENE_EXPRESSION_DENOISED'` (denoised single-cell gene expression; one
      channel per strand, in CPM); when False, scorers in
      `CELL_CONDITIONING_ONLY_OUTPUT_TYPES` are excluded.

  Returns:
    List of recommended variant scorers.
  """
  return [
      scorer
      for scorer in RECOMMENDED_VARIANT_SCORERS.values()
      if organism in SUPPORTED_ORGANISMS[scorer.base_variant_scorer]
      and (
          cell_conditioning
          or scorer.requested_output not in CELL_CONDITIONING_ONLY_OUTPUT_TYPES
      )
  ]


def tidy_anndata(
    adata: anndata.AnnData,
    match_gene_strand: bool = True,
    include_extended_metadata: bool = True,
    filter_protein_coding: bool = False,
) -> pd.DataFrame:
  """Formats an `AnnData` score as a tidy DataFrame.

  This function converts the score output from an AnnData object into a
  long-format pandas DataFrame, where each row represents:

    - For non-gene-centric variant scorers: Score for a variant-track pair.
    - For non-gene-centric interval scorers: Score for an interval-track pair.
    - For gene-centric variant/interval scoring: Score for a
      variant/interval-gene-track combination.

  Args:
    adata: An AnnData object containing scores.
    match_gene_strand: If True (and using gene-centric scoring), rows with
      mismatched gene and track strands are removed.
    include_extended_metadata: If True, includes additional columns derived from
      metadata specific to the output type, such as biosample name and type,
      transcription factor, and histone mark, if available. If False, only
      includes minimal metadata columns required to unique identify a track
      within a given output type: track_name and track_strand.
    filter_protein_coding: If True, filters the DataFrame to retain only rows
      where `gene_type == 'protein_coding'`.

  Returns:
    A pandas DataFrame with one score per row. The DataFrame includes
    columns for variant ID (if applicable), scored interval, gene information
    (if applicable), output type, variant/interval scorer, track name,
    track strand, the track imputation mode, and raw score. Additional metadata
    such as biosample name and type, histone and transcription factor marks are
    also returned (where available). See `tidy_scores` for
    more details on the returned columns.

  Raises:
    ValueError: If the input is not an AnnData object.
  """
  if not isinstance(adata, anndata.AnnData):
    raise ValueError('Invalid input type. Must be an AnnData object.')

  # Columns to include from the gene metadata. If the column did not
  # exist in the original metadata (or if the scores do not have a concept
  # of a gene), a column of all Nones is added.
  gene_columns = [
      'gene_id',
      'gene_name',
      'gene_type',
      'gene_strand',
      'junction_Start',
      'junction_End',
  ]
  # Columns to include from the CpG metadata. Only populated for scores that
  # are indexed by CpG site rather than by gene.
  cpg_columns = []
  if 'gene_id' in adata.obs and 'strand' in adata.obs:
    # Scores are for a gene-based scorer.
    obs = adata.obs.rename({'strand': 'gene_strand'}, axis=1)
    obs['gene_id'] = obs['gene_id'].str.split('.', expand=True)[0]  # Depatch.
    for col in gene_columns:
      if col not in obs:
        obs[col] = None
  elif 'cpg_position' in adata.obs:
    # Scores are for a CpG-based scorer, with one row per scored CpG site.
    obs = adata.obs.copy()
    cpg_columns = list(obs.columns)
    for col in gene_columns:
      obs[col] = None
  elif adata.X.shape[0] == 0:  # pyrefly: ignore[missing-attribute]
    # Scores are empty, so we return an empty dataframe.
    return pd.DataFrame()
  elif (
      adata.X.shape[0] == 1  # pyrefly: ignore[missing-attribute]
      and adata.obs.empty  # pyrefly: ignore[missing-attribute]
  ):
    # Scores are for a non-gene-based scorer.
    obs = pd.DataFrame([[None] * len(gene_columns)], columns=gene_columns)
  else:
    raise ValueError('Scores contain gene metadata but no gene_id/strand.')

  # Columns to include from the track metadata. Only added if the column
  # appears in the original metadata.
  var = adata.var.rename(
      {'strand': 'track_strand', 'name': 'track_name'}, axis=1
  )
  track_columns = ['track_name', 'track_strand', 'imputation_mode']
  if include_extended_metadata:
    track_columns.extend([c for c in var.columns if c not in track_columns])

  # Construct score dataframe.
  df = pd.merge(var, obs, how='cross')
  df['scored_interval'] = adata.uns['interval']

  # Add id and scorer information depending on the type of score.
  if 'variant_scorer' in adata.uns:
    df['variant_id'] = adata.uns['variant']
    df['output_type'] = adata.uns['variant_scorer'].requested_output.name
    df['variant_scorer'] = str(adata.uns['variant_scorer'])
    id_columns = ['variant_id', 'scored_interval']
    scorer_columns = ['output_type', 'variant_scorer']
  elif 'interval_scorer' in adata.uns:
    df['output_type'] = adata.uns['interval_scorer'].requested_output.name
    df['interval_scorer'] = str(adata.uns['interval_scorer'])
    id_columns = ['scored_interval']
    scorer_columns = ['output_type', 'interval_scorer']
  else:
    raise ValueError(
        'Invalid scores, expected either variant_scorer or interval_scorer.'
    )

  # Formatting final touches.
  columns = (
      id_columns + gene_columns + cpg_columns + scorer_columns + track_columns
  )
  df = df.loc[:, columns]
  df['raw_score'] = adata.X.T.reshape((  # pyrefly: ignore[missing-attribute]
      -1,
  ))  # pyrefly: ignore[missing-attribute]

  if 'quantiles' in adata.layers:  # pyrefly: ignore[not-iterable]
    quantiles_layer = adata.layers['quantiles']  # pyrefly: ignore[bad-index]
    df['quantile_score'] = quantiles_layer.T.reshape((-1,))

  # Remove entries where DNA strands are mismatched. Note that we still retain
  # all tracks that have strand '.' which indicates the data is unstranded.
  if match_gene_strand:
    mismatched_strands = (df['gene_strand'] == '+') & (
        df['track_strand'] == '-'
    ) | (df['gene_strand'] == '-') & (df['track_strand'] == '+')
    df = df[~mismatched_strands].reset_index(drop=True)

  if filter_protein_coding:
    if 'gene_type' not in df.columns:
      raise ValueError("Column 'gene_type' not found in score DataFrame.")
    df = df[df['gene_type'] == 'protein_coding'].reset_index(drop=True)

  return df.reset_index(drop=True)


def tidy_scores(
    scores: Sequence[anndata.AnnData] | Sequence[Sequence[anndata.AnnData]],
    match_gene_strand: bool = True,
    include_extended_metadata: bool = True,
    filter_protein_coding: bool = False,
) -> pd.DataFrame | None:
  """Formats scores into a tidy (long) pandas DataFrame.

  This function reformats variant or interval scores into a more readable
  DataFrame with one score per row. This function supports both scores generated
  from variant scorers (e.g., the output of `score_variant` or `score_variants`)
  or interval scorers (e.g., the output of `score_interval` or
  `score_intervals`).

  It handles both variant/interval-centric scoring (producing one score row per
  variant/interval-track pair) and gene-centric scoring (one score row per
  variant/interval-gene-track combination).

  The function accepts these score input types:

  - A sequence of AnnData objects (e.g., output of `score_variant` with one or
    more scorers).
  - A nested sequence of AnnData objects (e.g., output of `score_variants`
    with multiple variants).

  Scores from multiple scorers or multiple variants/intervals are concatenated
  together into a single pandas DataFrame, containing the union of all
  applicable columns across scorers.

  Args:
    scores: Scoring output as either a sequence of AnnData objects, or a nested
      sequence of AnnData objects (for example, the outputs of `score_variant`
      and `score_variants`, respectively).
    match_gene_strand: If True (and using gene-centric scoring), rows with
      mismatched gene and track strands are removed.
    include_extended_metadata: Argument passed to `tidy_anndata` to include
      additional metadata columns where available.
    filter_protein_coding: Argument passed to `tidy_anndata` to filter rows to
      only those where `gene_type == 'protein_coding'`.

  Returns:
    One row per score, with the columns below.

      - variant_id (when applicable): Variant of interest (e.g.
        chr22:36201698:A>C).
      - scored_interval: Genomic interval scored (e.g. chr22:36100000-36300000).
      - gene_id: ENSEMBL gene identifier without version number.
        (e.g. ENSG00000100342), or None if not applicable.
      - gene_name: HGNC gene symbol (e.g. APOL1), or None if not applicable.
      - gene_type: Gene biotype (e.g. protein_coding, lncRNA), or None if not
        applicable.
      - gene_strand: Strand of the gene ('+', '-', '.'), or None if not
        applicable.
      - output_type: Type of the output from the model (e.g. RNA_SEQ, DNASE).
      - interval_scorer (when applicable): Name of the interval scorer used.
      - variant_scorer (when applicable): Name of the variant scorer used.
      - track_name: Name of the output track (e.g. UBERON:0036149 total
        RNA-seq).
      - track_strand: Strand of the track ('+', '-', or '.').
      - imputation_mode: The imputation mode of the track ('observed',
        'imputed_supported' or 'imputed_exploratory').
      - biosample_curie: A '_' delimited string of ontology CURIEs that
        represent this biosample.
      - biosample_type: Type of biosample (e.g. 'tissue' or 'primary cell').
      - biosample_name: Name of the biosample.
      - raw_score: Raw variant score.
      - quantile_score (when applicable): Quantile score.

      Optional columns:

      - Assay title: Subtype of the assay (e.g., 'total RNA-seq').
      - data_source: Data source for the track (e.g. 'encode', 'gtex').
      - histone_mark: Name of the histological mark (e.g. 'H3K4ME3').
      - transcription_factor: Name of the transcription factor (e.g. 'CTCF').
      - splice_site_type: The type of splice site ('donor', 'acceptor').
      - suspension_type: The suspension type for single-cell assays ('nucleus',
        'cell').
      - endedness: The sequence reading endedness ('single', 'paired').
      - genetically_modified: Boolean indicating whether the sample was
        genetically modified.
      - cellxgene_dataset_id: The CELLxGENE dataset ID if applicable.
      - paper_doi: The DOI of the dataset paper if applicable.

  Raises:
    ValueError: If the input is not a valid type (sequence of AnnData or nested
      sequence of AnnDatas).
  """
  if isinstance(scores, Sequence):
    tidied_anndata = [
        tidy_anndata(
            adata,
            match_gene_strand=match_gene_strand,
            include_extended_metadata=include_extended_metadata,
            filter_protein_coding=filter_protein_coding,
        )
        for adata in itertools.chain.from_iterable(
            scores  # pyrefly: ignore[bad-argument-type]
        )
    ]
    if not tidied_anndata:
      return None

    concatenated_df = pd.concat(tidied_anndata, axis=0, ignore_index=True)
    # Put the raw_score (and quantile_score when applicable) columns last.
    last_columns = ['raw_score']
    if 'quantile_score' in concatenated_df.columns:
      last_columns.append('quantile_score')
    return concatenated_df[
        [col for col in concatenated_df.columns if col not in last_columns]
        + last_columns
    ]
  else:
    raise ValueError(
        'Invalid input type. Must be sequence of AnnDatas or nested sequence of'
        ' AnnDatas.'
    )
