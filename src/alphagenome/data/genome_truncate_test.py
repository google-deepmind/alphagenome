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

"""Reference-boundary regression tests for Interval.truncate."""

import sys

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome


class IntervalTruncateTest(parameterized.TestCase):

  @parameterized.product(
      case=(
          ((-20, -10), (0, 0)),
          ((20, 30), (10, 10)),
          ((-5, -5), (0, 0)),
          ((15, 15), (10, 10)),
          ((-5, 0), (0, 0)),
          ((10, 15), (10, 10)),
          ((-5, 5), (0, 5)),
          ((5, 15), (5, 10)),
          ((-5, 15), (0, 10)),
          ((2, 8), (2, 8)),
          ((0, 0), (0, 0)),
          ((10, 10), (10, 10)),
      ),
      strand=genome.STRAND_OPTIONS,
  )
  def test_truncation_preserves_interval_invariants_and_roundtrips(
      self, case, strand
  ):
    bounds, expected = case
    source = genome.Interval(
        "chr1",
        *bounds,
        strand=strand,
        name="example",
        info={"labels": ["source"]},
    )
    before = source.copy()
    actual = source.truncate(10)
    self.assertEqual((actual.start, actual.end), expected)
    self.assertGreaterEqual(actual.width, 0)
    self.assertTrue(actual.within_reference(10))
    self.assertEqual(actual.strand, strand)
    self.assertEqual(actual.name, source.name)
    self.assertEqual(actual.info, source.info)
    self.assertIsNot(actual, source)
    self.assertEqual(actual.truncate(10), actual)
    self.assertEqual(genome.Interval.from_str(str(actual)), actual)
    self.assertEqual(genome.Interval.from_proto(actual.to_proto()), actual)
    actual.info["labels"].append("changed")
    self.assertEqual(source.info, before.info)
    self.assertEqual(source, before)

  @parameterized.parameters(0, -1)
  def test_nonpositive_reference_length_is_still_rejected(self, length):
    with self.assertRaisesRegex(ValueError, "Reference length"):
      genome.Interval("chr1", 1, 2).truncate(length)

  def test_default_reference_and_large_coordinates_do_not_overflow(self):
    for bounds, expected in (
        ((-10, -5), (0, 0)),
        ((4, 12), (4, 12)),
        ((sys.maxsize + 1, sys.maxsize + 5), (sys.maxsize, sys.maxsize)),
    ):
      with self.subTest(bounds=bounds):
        actual = genome.Interval("chr1", *bounds).truncate()
        self.assertEqual((actual.start, actual.end), expected)


if __name__ == "__main__":
  absltest.main()
