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

"""Module containing interval scorer dataclasses for interval scoring."""

import dataclasses
import enum
from typing import TypeVar

import immutabledict

from alphagenome.models import dna_output
from alphagenome.protos import dna_model_pb2


class IntervalAggregationType(enum.Enum):
  """Enum indicating the type of interval scorer aggregation.

  Attributes:
    MEAN: Mean across positions.
    SUM: Sum across positions.

  Methods:
    to_proto: Converts the aggregation type to its corresponding proto enum.
  """

  MEAN = dna_model_pb2.IntervalAggregationType.INTERVAL_AGGREGATION_TYPE_MEAN
  SUM = dna_model_pb2.IntervalAggregationType.INTERVAL_AGGREGATION_TYPE_SUM

  def to_proto(self) -> dna_model_pb2.IntervalAggregationType:
    return self.value

  def __repr__(self) -> str:
    return self.name


class BaseIntervalScorer(enum.Enum):
  """Enum indicating the type of interval scoring.

  Attributes:
    GENE_MASK: Gene-mask scoring (e.g., aggregation of predictions within genes)
  """

  GENE_MASK = enum.auto()


SUPPORTED_OUTPUT_TYPES = immutabledict.immutabledict({
    BaseIntervalScorer.GENE_MASK: [
        dna_output.OutputType.ATAC,
        dna_output.OutputType.CAGE,
        dna_output.OutputType.CHIP_HISTONE,
        dna_output.OutputType.CHIP_TF,
        dna_output.OutputType.DNASE,
        dna_output.OutputType.RNA_SEQ,
        dna_output.OutputType.SPLICE_SITES,
        dna_output.OutputType.SPLICE_SITE_USAGE,
    ],
})

SUPPORTED_WIDTHS = immutabledict.immutabledict({
    BaseIntervalScorer.GENE_MASK: [None, 501, 2001, 10_001, 100_001, 200_001],
})


@dataclasses.dataclass(frozen=True)
class GeneMaskScorer:
  """Interval scorer for gene-mask scoring.

  Attributes:
    requested_output: The requested output type (e.g. ATAC, DNASE, etc.)
    width: The width of the target interval to include in the aggregation.
    aggregation_type: The type of aggregation to perform.
    base_interval_scorer: The base interval scorer.
    name: The name of the scorer (a composite of the above attributes that
      uniquely identifies a scorer combination).

  Methods:
    to_proto: Converts the scorer to its corresponding proto message.

  Raises:
    ValueError: If the requested output is not supported.
  """

  requested_output: dna_output.OutputType
  width: int | None
  aggregation_type: IntervalAggregationType

  @property
  def base_interval_scorer(self) -> BaseIntervalScorer:
    return BaseIntervalScorer.GENE_MASK

  @property
  def name(self) -> str:
    return str(self)

  def __post_init__(self):
    if (
        self.requested_output
        not in SUPPORTED_OUTPUT_TYPES[self.base_interval_scorer]
    ):
      raise ValueError(
          f'Unsupported requested output: {self.requested_output}. Supported'
          f' output types: {SUPPORTED_OUTPUT_TYPES[self.base_interval_scorer]}'
      )
    if self.width not in SUPPORTED_WIDTHS[self.base_interval_scorer]:
      raise ValueError(
          f'Unsupported width: {self.width}. Supported widths:'
          f' {SUPPORTED_WIDTHS[self.base_interval_scorer]}'
      )

  def to_proto(self) -> dna_model_pb2.IntervalScorer:
    return dna_model_pb2.IntervalScorer(
        gene_mask=dna_model_pb2.GeneMaskIntervalScorer(
            requested_output=self.requested_output.to_proto(),
            width=self.width,
            aggregation_type=self.aggregation_type.to_proto(),
        )
    )


# TypeVar for all interval scorer types.
IntervalScorerTypes = TypeVar(
    'IntervalScorerTypes',
    bound=GeneMaskScorer,
)

# A dict of interval scorers, with our recommended settings for a wide range
# of different use cases and settings.
RECOMMENDED_INTERVAL_SCORERS = {
    'RNA_SEQ': GeneMaskScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
        width=200_001,
        aggregation_type=IntervalAggregationType.MEAN,
    ),
}
