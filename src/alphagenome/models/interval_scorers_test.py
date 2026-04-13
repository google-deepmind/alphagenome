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

from absl.testing import absltest, parameterized

from alphagenome.models import dna_output, interval_scorers
from alphagenome.protos import dna_model_pb2


class IntervalScorersTest(parameterized.TestCase):

  def test_aggregation_type_to_proto(self):
    expected = [
        dna_model_pb2.IntervalAggregationType.INTERVAL_AGGREGATION_TYPE_MEAN,
        dna_model_pb2.IntervalAggregationType.INTERVAL_AGGREGATION_TYPE_SUM,
    ]

    for interval_scorer_aggregation_type, expected_proto in zip(
        interval_scorers.IntervalAggregationType, expected, strict=True
    ):
      self.assertEqual(
          interval_scorer_aggregation_type.to_proto(), expected_proto
      )

  def test_interval_scorers_to_proto(self):
    aggregation_type = (
        dna_model_pb2.IntervalAggregationType.INTERVAL_AGGREGATION_TYPE_MEAN
    )
    expected = [
        dna_model_pb2.IntervalScorer(
            gene_mask=dna_model_pb2.GeneMaskIntervalScorer(
                requested_output=dna_model_pb2.OUTPUT_TYPE_RNA_SEQ,
                width=10_001,
                aggregation_type=aggregation_type,
            )
        ),
    ]
    scorers = [
        interval_scorers.GeneMaskScorer(
            requested_output=dna_output.OutputType.RNA_SEQ,
            width=10_001,
            aggregation_type=interval_scorers.IntervalAggregationType.MEAN,
        ),
    ]
    for interval_scorer, expected_proto in zip(scorers, expected):
      self.assertEqual(interval_scorer.to_proto(), expected_proto)

  def test_unsupported_output_type_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported requested output'):
      _ = interval_scorers.GeneMaskScorer(
          requested_output=dna_output.OutputType.CONTACT_MAPS,
          width=10_001,
          aggregation_type=interval_scorers.IntervalAggregationType.MEAN,
      )

  def test_unsupported_width_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported width'):
      _ = interval_scorers.GeneMaskScorer(
          requested_output=dna_output.OutputType.RNA_SEQ,
          width=1234,
          aggregation_type=interval_scorers.IntervalAggregationType.MEAN,
      )


if __name__ == '__main__':
  absltest.main()
