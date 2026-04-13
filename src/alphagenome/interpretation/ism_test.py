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

import numpy as np
from absl.testing import absltest, parameterized

from alphagenome.data import genome
from alphagenome.interpretation import ism


class IsmTest(parameterized.TestCase):

  def test_ism_variants(self):
    sequence = 'ACG'
    interval = genome.Interval('chr1', 0, len(sequence))
    variants = ism.ism_variants(interval, sequence=sequence)
    expected = [
        genome.Variant('chr1', 1, 'A', 'C'),
        genome.Variant('chr1', 1, 'A', 'G'),
        genome.Variant('chr1', 1, 'A', 'T'),
        genome.Variant('chr1', 2, 'C', 'A'),
        genome.Variant('chr1', 2, 'C', 'G'),
        genome.Variant('chr1', 2, 'C', 'T'),
        genome.Variant('chr1', 3, 'G', 'A'),
        genome.Variant('chr1', 3, 'G', 'C'),
        genome.Variant('chr1', 3, 'G', 'T'),
    ]
    self.assertEqual(variants, expected)

  def test_ism_variants_length_mismatch(self):
    sequence = 'ACG'
    interval = genome.Interval('chr1', 0, 2)
    with self.assertRaises(ValueError):
      ism.ism_variants(interval, sequence=sequence)

  @parameterized.parameters([True, False])
  def test_ism_variants_with_interval_with_n(self, skip_n):
    sequence = 'NCG'
    interval = genome.Interval('chr1', 0, len(sequence))
    variants = ism.ism_variants(interval, sequence=sequence, skip_n=skip_n)
    expected_n = [
        genome.Variant('chr1', 1, 'N', 'A'),
        genome.Variant('chr1', 1, 'N', 'C'),
        genome.Variant('chr1', 1, 'N', 'G'),
        genome.Variant('chr1', 1, 'N', 'T'),
    ]
    expected = [
        genome.Variant('chr1', 2, 'C', 'A'),
        genome.Variant('chr1', 2, 'C', 'G'),
        genome.Variant('chr1', 2, 'C', 'T'),
        genome.Variant('chr1', 3, 'G', 'A'),
        genome.Variant('chr1', 3, 'G', 'C'),
        genome.Variant('chr1', 3, 'G', 'T'),
    ]
    self.assertEqual(variants, expected if skip_n else expected_n + expected)

  @parameterized.parameters(dict(multiply_by_sequence=[True, False]))
  def test_ism_scores(self, multiply_by_sequence):
    # 'ACG'
    variants = [
        genome.Variant('chr1', 1, 'A', 'C'),
        genome.Variant('chr1', 1, 'A', 'G'),
        genome.Variant('chr1', 1, 'A', 'T'),
        genome.Variant('chr1', 2, 'C', 'A'),
        genome.Variant('chr1', 2, 'C', 'G'),
        genome.Variant('chr1', 2, 'C', 'T'),
        genome.Variant('chr1', 3, 'G', 'A'),
        genome.Variant('chr1', 3, 'G', 'C'),
        genome.Variant('chr1', 3, 'G', 'T'),
    ]
    ism_matrix = ism.ism_matrix(
        variant_scores=[1] * len(variants),
        variants=variants,
        multiply_by_sequence=multiply_by_sequence,
    )
    if multiply_by_sequence:
      expect = np.array(
          [
              [-1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, -1, 0],
          ],
          dtype=np.float32,
      )
      np.testing.assert_array_equal(ism_matrix, expect)
    else:
      expect = (
          np.array(
              [
                  [0, 1, 1, 1],
                  [1, 0, 1, 1],
                  [1, 1, 0, 1],
              ],
              dtype=np.float32,
          )
          - 1
      )
      np.testing.assert_array_equal(ism_matrix, expect)

  def test_ism_scores_subset(self):
    variants = [
        genome.Variant('chr1', 1, 'A', 'C'),
        genome.Variant('chr1', 1, 'A', 'G'),
        genome.Variant('chr1', 1, 'A', 'T'),
        genome.Variant('chr1', 2, 'C', 'A'),
        genome.Variant('chr1', 2, 'C', 'G'),
        genome.Variant('chr1', 2, 'C', 'T'),
        genome.Variant('chr1', 3, 'G', 'A'),
        genome.Variant('chr1', 3, 'G', 'C'),
        genome.Variant('chr1', 3, 'G', 'T'),
    ]
    ism_matrix = ism.ism_matrix(
        variant_scores=[1] * len(variants),
        variants=variants,
        interval=genome.Interval('chr1', 1, 3),
        multiply_by_sequence=False,
    )

    expect = (
        np.array(
            [
                [1, 0, 1, 1],
                [1, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        - 1
    )
    np.testing.assert_array_equal(ism_matrix, expect)

  def test_ism_scores_missing_variants(self):
    variants = [
        genome.Variant('chr1', 1, 'A', 'C'),
        genome.Variant('chr1', 1, 'A', 'G'),
        genome.Variant('chr1', 1, 'A', 'T'),
        genome.Variant('chr1', 2, 'C', 'A'),
        genome.Variant('chr1', 2, 'C', 'G'),
        genome.Variant('chr1', 2, 'C', 'T'),
        genome.Variant('chr1', 3, 'G', 'A'),
        genome.Variant('chr1', 3, 'G', 'C'),
        genome.Variant('chr1', 3, 'G', 'T'),
    ]
    with self.assertRaises(ValueError):
      # Out of bounds.
      ism.ism_matrix(
          variant_scores=[1] * len(variants),
          variants=variants,
          interval=genome.Interval('chr1', 1, 4),
          multiply_by_sequence=False,
      )

    output = ism.ism_matrix(
        variant_scores=[1] * len(variants),
        variants=variants,
        interval=genome.Interval('chr1', 1, 4),
        multiply_by_sequence=False,
        require_fully_filled=False,
    )
    np.testing.assert_array_equal(output[-1], 0)


if __name__ == '__main__':
  absltest.main()
