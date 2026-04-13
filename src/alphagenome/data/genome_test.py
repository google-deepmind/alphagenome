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

from unittest import mock

import numpy as np
from absl.testing import absltest, parameterized

from alphagenome.data import genome
from alphagenome.protos import dna_model_pb2


class StrandTest(parameterized.TestCase):

  @parameterized.parameters(
      list(
          zip(
              genome.STRAND_OPTIONS,
              (
                  genome.Strand.POSITIVE,
                  genome.Strand.NEGATIVE,
                  genome.Strand.UNSTRANDED,
              ),
          )
      )
  )
  def test_from_str(self, strand_str, expected):
    strand = genome.Strand.from_str(strand_str)
    self.assertEqual(strand, expected)

  def test_invalid_strand_str(self):
    with self.assertRaises(ValueError):
      genome.Strand.from_str('foo')

  @parameterized.parameters(
      (genome.Strand.POSITIVE,),
      (genome.Strand.NEGATIVE,),
      (genome.Strand.UNSTRANDED,),
  )
  def test_from_proto(self, strand):
    proto = genome.Strand.to_proto(strand)
    round_trip = genome.Strand.from_proto(proto)
    self.assertEqual(strand, round_trip)

  def test_invalid_strand_proto(self):
    with self.assertRaises(ValueError):
      genome.Strand.from_proto(dna_model_pb2.Strand.STRAND_UNSPECIFIED)


class IntervalTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(chromosome='chr1', start=0, end=100, strand='-'),
      dict(chromosome='chr2', start=100, end=200, strand='.'),
      dict(chromosome='chr3', start=100, end=200, strand='+'),
  )
  def test_proto_roundtrip(self, chromosome, start, end, strand):
    interval = genome.Interval(chromosome, start, end, strand)
    proto = interval.to_proto()
    round_trip = genome.Interval.from_proto(proto)
    self.assertEqual(interval, round_trip)

  def test_init_valid(self):
    genome.Interval('chr1', 10, 10)
    genome.Interval(
        'chr1',
        10,
        20,
        name='test',
        info=dict(a=1, b='b', c=[1, 2], d=dict(e=1)),
    )

  def test_init_raises_value_error(self):
    with self.assertRaises(ValueError):
      genome.Interval('chr1', 5, 1)  # start < end

  def test_attributes(self):
    interval = genome.Interval('chr1', 10, 20, strand='-')
    self.assertEqual(interval.start, 10)
    self.assertEqual(interval.end, 20)
    self.assertEqual(interval.chromosome, 'chr1')
    self.assertEqual(interval.name, '')
    self.assertIsInstance(interval.info, dict)
    self.assertEmpty(interval.info)
    interval.info['test'] = 10
    self.assertEqual(interval.info['test'], 10)
    self.assertTrue(interval.negative_strand)

    self.assertEqual(interval.width, 10)

  def test_copy(self):
    # make sure the original got unchanged
    interval = genome.Interval('chr1', 10, 20, strand='-')
    i2 = interval.copy()
    i2.info['test'] = 10
    self.assertEqual(i2.info['test'], 10)
    self.assertEmpty(interval.info)

  def test_serialization(self):
    interval = genome.Interval('chr1', 0, 100, '-')
    self.assertEqual(interval, genome.Interval.from_str('chr1:0-100:-'))
    self.assertEqual(str(interval), 'chr1:0-100:-')

  def test_serialization_no_strand(self):
    # Serialized intervals are allowed to be missing a strand, in which case
    # they are treated as unstranded. Serializing an interval always adds the
    # strand.
    interval = genome.Interval('chr1', 0, 100)
    self.assertEqual(interval, genome.Interval.from_str('chr1:0-100'))
    self.assertEqual(interval, genome.Interval.from_str('chr1:0-100:.'))
    self.assertEqual(str(interval), 'chr1:0-100:.')

  def test_from_to_string(self):
    interval = genome.Interval('chr1', 10, 20, strand='-')
    self.assertIsInstance(str(interval), str)
    self.assertEqual(interval, genome.Interval.from_str(str(interval)))

  def test_from_to_string_negative_start(self):
    interval = genome.Interval('chr1', -10, 20, strand='-')
    self.assertIsInstance(str(interval), str)
    self.assertEqual(interval, genome.Interval.from_str(str(interval)))

  def test_from_to_string_negative_end(self):
    interval = genome.Interval('chr1', -20, -10, strand='-')
    self.assertIsInstance(str(interval), str)
    self.assertEqual(interval, genome.Interval.from_str(str(interval)))

  def test_overlaps_overlapping_at_start(self):
    self.assertTrue(
        genome.Interval('chr1', 1, 5).overlaps(genome.Interval('chr1', 3, 7))
    )

  def test_overlaps_overlapping_at_end(self):
    self.assertTrue(
        genome.Interval('chr1', 1, 5).overlaps(genome.Interval('chr1', 0, 2))
    )

  def test_overlaps_overlapping_contains(self):
    self.assertTrue(
        genome.Interval('chr1', 1, 5).overlaps(genome.Interval('chr1', 0, 6))
    )

  def test_overlaps_nonoverlapping_at_start(self):
    self.assertFalse(
        genome.Interval('chr1', 1, 5).overlaps(genome.Interval('chr1', 0, 1))
    )

  def test_overlaps_nonoverlapping_at_end(self):
    self.assertFalse(
        genome.Interval('chr1', 1, 5).overlaps(genome.Interval('chr1', 5, 6))
    )

  def test_overlaps_nonoverlapping_different_chromosome(self):
    self.assertFalse(
        genome.Interval('chr1', 1, 5).overlaps(genome.Interval('chr2', 0, 6))
    )

  def test_contains_small_within_larger(self):
    self.assertTrue(
        genome.Interval('chr1', 1, 5).contains(genome.Interval('chr1', 2, 3))
    )

  def test_contains_large_within_small(self):
    self.assertFalse(
        genome.Interval('chr1', 2, 3).contains(genome.Interval('chr1', 1, 5))
    )

  def test_contains_same_size(self):
    self.assertTrue(
        genome.Interval('chr1', 1, 5).contains(genome.Interval('chr1', 1, 5))
    )

  def test_contains_overlapping_end(self):
    self.assertFalse(
        genome.Interval('chr1', 1, 5).contains(genome.Interval('chr1', 4, 9))
    )

  def test_contains_overlapping_start(self):
    self.assertFalse(
        genome.Interval('chr1', 4, 10).contains(genome.Interval('chr1', 1, 5))
    )

  def test_contains_non_overlapping_end(self):
    self.assertFalse(
        genome.Interval('chr1', 1, 5).contains(genome.Interval('chr1', 6, 9))
    )

  def test_contains_non_overlapping_start(self):
    self.assertFalse(
        genome.Interval('chr1', 6, 9).contains(genome.Interval('chr1', 1, 5))
    )

  def test_contains_different_chromosome(self):
    self.assertFalse(
        genome.Interval('chr1', 1, 5).contains(genome.Interval('chr2', 1, 5))
    )

  def test_shift_dont_use_strand(self):
    interval = genome.Interval('chr1', 10, 20, strand='-')
    i2 = interval.shift(10, use_strand=False)

    # original unchanged
    self.assertEqual(interval.start, 10)
    self.assertEqual(interval.end, 20)

    self.assertEqual(i2.start, 20)
    self.assertEqual(i2.end, 30)

  def test_shift_use_strand(self):
    interval = genome.Interval('chr1', 10, 20, strand='-')
    i2 = interval.shift(10)  # use_strand = True by default.
    self.assertEqual(i2.start, 0)
    self.assertEqual(i2.end, 10)

    # Shift outside of the reference.
    self.assertFalse(interval.shift(20, use_strand=True).within_reference())

    i2 = interval.shift(15, use_strand=True).truncate()
    self.assertEqual(i2.start, 0)
    self.assertEqual(i2.end, 5)

    self.assertEqual(interval.center(), 15)

  def test_pad_does_not_mutate(self):
    interval = genome.Interval('chr1', 100, 200)
    self.assertIsNot(interval, interval.pad(0, 0))

  def test_pad_unstranded(self):
    self.assertEqual(
        genome.Interval('chr1', 100, 200, '.').pad(20, 10, use_strand=False),
        genome.Interval('chr1', 80, 210, '.'),
    )
    self.assertEqual(
        genome.Interval('chr1', 100, 200, '+').pad(20, 10, use_strand=False),
        genome.Interval('chr1', 80, 210, '+'),
    )
    self.assertEqual(
        genome.Interval('chr1', 100, 200, '+').pad(20, 10, use_strand=False),
        genome.Interval('chr1', 80, 210, '+'),
    )

  def test_pad_stranded(self):
    self.assertEqual(
        genome.Interval('chr1', 100, 200, '.').pad(20, 10, use_strand=True),
        genome.Interval('chr1', 80, 210, '.'),
    )
    self.assertEqual(
        genome.Interval('chr1', 100, 200, '+').pad(20, 10, use_strand=True),
        genome.Interval('chr1', 80, 210, '+'),
    )
    # Negative strand interval padding behaves differently when use_strand=True
    self.assertEqual(
        genome.Interval('chr1', 100, 200, '-').pad(20, 10, use_strand=True),
        genome.Interval('chr1', 90, 220, '-'),
    )

  def test_boundary_shift_positive_strand(self):
    interval = genome.Interval('chr1', 10, 20, strand='+')
    resized = interval.boundary_shift(1, -2)
    self.assertEqual(resized.start, 11)
    self.assertEqual(resized.end, 18)

  def test_boundary_shift_negative_strand(self):
    interval = genome.Interval('chr1', 10, 20, strand='-')
    resized = interval.boundary_shift(1, -2)
    self.assertEqual(resized.start, 12)
    self.assertEqual(resized.end, 19)

  def test_resize_positive_strand_odd_width(self):
    interval = genome.Interval('chr1', 10, 20, strand='+')
    i2 = interval.resize(11)
    self.assertEqual(i2.start, 9)
    self.assertEqual(i2.end, 20)

  def test_resize_positive_strand_even_width(self):
    interval = genome.Interval('chr1', 10, 20, strand='+')
    i2 = interval.resize(12)
    self.assertEqual(i2.start, 9)
    self.assertEqual(i2.end, 21)

  def test_resize_negative_strand_odd_width(self):
    interval = genome.Interval('chr1', 10, 20, strand='-')
    i2 = interval.resize(11)
    self.assertEqual(i2.start, 10)
    self.assertEqual(i2.end, 21)

  def test_resize_negative_strand_even_width(self):
    interval = genome.Interval('chr1', 10, 20, strand='-')
    i2 = interval.resize(12)
    self.assertEqual(i2.start, 9)
    self.assertEqual(i2.end, 21)

    i2 = interval.resize(9)
    self.assertEqual(i2.start, 11)
    self.assertEqual(i2.end, 20)

  @parameterized.parameters(('+', '-'), ('-', '+'))
  def test_swap_strand(self, before, after):
    interval = genome.Interval('chr1', 10, 20, strand=before)
    self.assertEqual(interval.swap_strand().strand, after)
    self.assertEqual(interval.strand, before)

  @parameterized.parameters(('+', '.'), ('-', '.'), ('.', '.'))
  def test_unstranded(self, before, after):
    interval = genome.Interval('chr1', 10, 20, strand=before)
    self.assertEqual(interval.as_unstranded().strand, after)
    self.assertEqual(interval.strand, before)

  def test_swap_strand_unstranded(self):
    interval = genome.Interval('chr1', 10, 20, strand='.')
    with self.assertRaises(ValueError):
      interval.swap_strand()

  @parameterized.parameters(
      (0, True), (0, False), (1, True), (1, False), (2, True), (2, False)
  )
  def test_center_even(self, shift, use_strand):
    """Checks even-width intervals."""
    interval = genome.Interval('chr1', 10 + shift, 20 + shift, strand='-')
    self.assertEqual(interval.center(use_strand=use_strand), 15 + shift)

  @parameterized.parameters(0, 1, 2)
  def test_center_odd_positive_strand(self, shift):
    """Checks odd-width intervals."""
    interval = genome.Interval('chr1', 9 + shift, 20 + shift, strand='+')
    self.assertEqual(interval.center(use_strand=False), 15 + shift)
    self.assertEqual(interval.center(use_strand=True), 15 + shift)

  @parameterized.parameters(0, 1, 2)
  def test_center_odd_negative_strand(self, shift):
    interval = genome.Interval('chr1', 9 + shift, 20 + shift, strand='-')
    self.assertEqual(interval.center(use_strand=False), 15 + shift)
    self.assertEqual(interval.center(use_strand=True), 14 + shift)

  @parameterized.parameters(
      (10, 13, '+'),  # even -> odd
      (10, 14, '+'),  # even -> even
      (11, 13, '+'),  # odd -> odd
      (11, 14, '+'),  # odd -> even
      (10, 13, '-'),  # negative strand
      (10, 14, '-'),
      (11, 13, '-'),
      (11, 14, '-'),
  )
  def test_center_constant_at_resize(self, start_width, resize_width, strand):
    """Tests that the center remains the same if we resize."""
    interval = genome.Interval('chr1', 10, 10 + start_width, strand=strand)
    self.assertEqual(interval.resize(resize_width).center(), interval.center())

  def test_intersect_overlaps(self):
    a = genome.Interval('chr1', 1, 5)
    b = genome.Interval('chr1', 3, 7)
    self.assertEqual(a.intersect(b), genome.Interval('chr1', 3, 5))

  def test_intersect_no_overlap(self):
    a = genome.Interval('chr1', 1, 5)
    b = genome.Interval('chr1', 7, 10)
    self.assertIsNone(a.intersect(b))

  def test_coverage(self):
    a = genome.Interval('chr1', 1, 5)
    b = genome.Interval('chr1', 3, 7)
    c = genome.Interval('chr1', 4, 7)
    d = genome.Interval('chr1', 1, 2)
    np.testing.assert_array_equal(a.coverage([b, c, d]), [1, 0, 1, 2])
    np.testing.assert_array_equal(a.coverage([b, c, d], bin_size=2), [1, 3])
    np.testing.assert_array_equal(a.coverage([b, c, d], bin_size=4), [4])

  def test_coverage_raises(self):
    a = genome.Interval('chr1', 1, 5)
    b = genome.Interval('chr1', 3, 7)
    c = genome.Interval('chr1', 4, 7)
    d = genome.Interval('chr1', 1, 2)
    with self.assertRaises(ValueError):
      a.coverage([b, c, d], bin_size=0)
    with self.assertRaises(ValueError):
      a.coverage([b, c, d], bin_size=-1)
    with self.assertRaises(ValueError):
      a.coverage([b, c, d], bin_size=3)

  def test_overlap_ranges(self):
    a = genome.Interval('chr1', 1, 5)
    b = genome.Interval('chr1', 3, 7)
    c = genome.Interval('chr1', 2, 4)
    d = genome.Interval('chr1', 0, 2)
    e = genome.Interval('chr1', 5, 10)
    f = genome.Interval('chr2', 1, 5)
    np.testing.assert_array_equal(
        a.overlap_ranges([b, c, d, e, f]),
        np.array(
            [
                (2, 4),
                (1, 3),
                (0, 1),
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        a.overlap_ranges([f]), np.empty((0, 2), dtype=np.int32)
    )

  def test_binary_mask(self):
    a = genome.Interval('chr1', 1, 5)
    b = genome.Interval('chr1', 3, 7)
    c = genome.Interval('chr1', 4, 7)
    d = genome.Interval('chr1', 1, 2)
    np.testing.assert_array_equal(
        a.binary_mask([b, c, d]), [True, False, True, True]
    )
    np.testing.assert_array_equal(
        a.binary_mask([b, c, d], bin_size=2), [True, True]
    )

  def test_binary_mask2(self):
    interval = genome.Interval('chr1', 0, 6)
    intervals = [
        genome.Interval('chr1', 1, 2),
        genome.Interval('chr1', 2, 3),
        genome.Interval('chr1', 3, 4),
    ]
    mask = interval.binary_mask(intervals, bin_size=1)
    self.assertSequenceEqual(
        [False, True, True, True, False, False], list(mask)
    )

    mask = interval.binary_mask(intervals, bin_size=2)
    self.assertSequenceEqual([True, True, False], list(mask))

  def test_coverage_stranded(self):
    a = genome.Interval('chr1', 1, 5)
    a_reversed = genome.Interval('chr1', 1, 5, strand='-')
    b = genome.Interval('chr1', 3, 7)
    c = genome.Interval('chr1', 4, 7)
    d = genome.Interval('chr1', 1, 2, strand='-')
    np.testing.assert_array_equal(
        a.coverage_stranded([b, c, d]), [[0, 1], [0, 0], [1, 0], [2, 0]]
    )
    np.testing.assert_array_equal(
        a_reversed.coverage_stranded([b, c, d]),
        [[0, 2], [0, 1], [0, 0], [1, 0]],
    )

  def test_binary_mask_stranded(self):
    interval = genome.Interval('chr1', 0, 6)
    intervals = [
        genome.Interval('chr1', 1, 2),
        genome.Interval('chr1', 2, 3),
        genome.Interval('chr1', 3, 4),
    ]

    mask = interval.binary_mask_stranded(intervals, bin_size=2)
    np.testing.assert_array_equal(
        [[True, False], [True, False], [False, False]], mask
    )

  def test_binary_mask_stranded2(self):
    interval = genome.Interval('chr1', 0, 6, strand='-')
    intervals = [
        genome.Interval('chr1', 1, 2, strand='+'),
        genome.Interval('chr1', 2, 3, strand='+'),
        genome.Interval('chr1', 3, 4, strand='-'),
    ]

    mask = interval.binary_mask_stranded(intervals, bin_size=2)
    np.testing.assert_array_equal(
        [[False, False], [True, True], [False, True]], mask
    )


class VariantTest(parameterized.TestCase):

  def test_proto_roundtrip(self):
    variant = genome.Variant('chr1', 10, 'C', 'T')
    proto = variant.to_proto()
    round_trip = genome.Variant.from_proto(proto)
    self.assertEqual(variant, round_trip)

  def test_init_valid(self):
    genome.Variant('chr1', 10, 'C', 'T')
    genome.Variant('chr1', 10, 'C', 'T', name='test')
    genome.Variant(
        'chr1',
        10,
        'C',
        'T',
        name='test',
        info=dict(a=1, b='b', c=[1, 2], d=dict(e=1)),
    )

  @parameterized.parameters(
      ('chr1', -20, 'C', 'T', 'Position has to be >=1.'),
      ('chr1', 1, 'Z', 'T', 'Invalid reference bases'),
      ('chr1', 1, 'A', 'Z', 'Invalid alternate bases'),
  )
  def test_init_raises_value_error(
      self, chromosome, position, ref_bases, alt_bases, expected_error
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      genome.Variant(chromosome, position, ref_bases, alt_bases)

  def test_attributes(self):
    v = genome.Variant('chr1', 10, 'C', 'T')
    self.assertEqual(v.start, 9)
    self.assertEqual(v.chromosome, 'chr1')
    self.assertEqual(v.position, 10)
    self.assertEqual(v.reference_bases, 'C')
    self.assertEqual(v.alternate_bases, 'T')

  def test_attributes_info_assignment(self):
    v = genome.Variant('chr1', 10, 'C', 'T')
    self.assertIsInstance(v.info, dict)
    v.info['test'] = 10
    self.assertEqual(v.info['test'], 10)

  def test_copy(self):
    # Make sure the original got unchanged.
    v = genome.Variant('chr1', 10, 'C', 'T', info=dict(test=10))
    v2 = v.copy()
    v.info['test'] = 20
    self.assertEqual(v2.info['test'], 10)
    self.assertEqual(v.info['test'], 20)

  @parameterized.parameters(
      ('chr22:1024:A>C', genome.VariantFormat.DEFAULT),
      ('chr22_1024_A_C_b38', genome.VariantFormat.GTEX),
      ('22_1024_A_C', genome.VariantFormat.OPEN_TARGETS),
      ('22:1024:A:C', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('22-1024-A-C', genome.VariantFormat.GNOMAD),
  )
  def test_from_str_single_base(self, string, variant_format):
    expected_variant = genome.Variant(
        chromosome='chr22',
        position=1024,
        reference_bases='A',
        alternate_bases='C',
    )
    actual_variant = genome.Variant.from_str(
        string, variant_format=variant_format
    )
    self.assertEqual(expected_variant.chromosome, actual_variant.chromosome)
    self.assertEqual(expected_variant.position, actual_variant.position)
    self.assertEqual(
        expected_variant.reference_bases, actual_variant.reference_bases
    )
    self.assertEqual(
        expected_variant.alternate_bases, actual_variant.alternate_bases
    )

  @parameterized.parameters(
      ('chrX:1024:AGTC>C', genome.VariantFormat.DEFAULT),
      ('chrX_1024_AGTC_C', genome.VariantFormat.GTEX),
      ('X_1024_AGTC_C', genome.VariantFormat.OPEN_TARGETS),
      ('X:1024:AGTC:C', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('X-1024-AGTC-C', genome.VariantFormat.GNOMAD),
  )
  def test_from_str_multiple_bases(self, string, variant_format):
    expected_variant = genome.Variant(
        chromosome='chrX',
        position=1024,
        reference_bases='AGTC',
        alternate_bases='C',
    )
    actual_variant = genome.Variant.from_str(
        string, variant_format=variant_format
    )
    self.assertEqual(expected_variant.chromosome, actual_variant.chromosome)
    self.assertEqual(expected_variant.position, actual_variant.position)
    self.assertEqual(
        expected_variant.reference_bases, actual_variant.reference_bases
    )
    self.assertEqual(
        expected_variant.alternate_bases, actual_variant.alternate_bases
    )

  @parameterized.parameters(
      ('chrM:1024:AGTC>', genome.VariantFormat.DEFAULT),
      ('chrM_1024_AGTC__hg38', genome.VariantFormat.GTEX),
      ('M_1024_AGTC_', genome.VariantFormat.OPEN_TARGETS),
      ('M:1024:AGTC:', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('M-1024-AGTC-', genome.VariantFormat.GNOMAD),
  )
  def test_from_str_deletion(self, string, variant_format):
    expected_variant = genome.Variant(
        chromosome='chrM',
        position=1024,
        reference_bases='AGTC',
        alternate_bases='',
    )
    actual_variant = genome.Variant.from_str(
        string, variant_format=variant_format
    )
    self.assertEqual(expected_variant.chromosome, actual_variant.chromosome)
    self.assertEqual(expected_variant.position, actual_variant.position)
    self.assertEqual(
        expected_variant.reference_bases, actual_variant.reference_bases
    )
    self.assertEqual(
        expected_variant.alternate_bases, actual_variant.alternate_bases
    )

  @parameterized.parameters(
      ('chrM:1024:>AGTC', genome.VariantFormat.DEFAULT),
      ('chrM_1024__AGTC_hg38', genome.VariantFormat.GTEX),
      ('M_1024__AGTC', genome.VariantFormat.OPEN_TARGETS),
      ('M:1024::AGTC', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('M-1024--AGTC', genome.VariantFormat.GNOMAD),
  )
  def test_from_str_insertion(self, string, variant_format):
    expected_variant = genome.Variant(
        chromosome='chrM',
        position=1024,
        reference_bases='',
        alternate_bases='AGTC',
    )
    actual_variant = genome.Variant.from_str(
        string, variant_format=variant_format
    )
    self.assertEqual(expected_variant.chromosome, actual_variant.chromosome)
    self.assertEqual(expected_variant.position, actual_variant.position)
    self.assertEqual(
        expected_variant.reference_bases, actual_variant.reference_bases
    )
    self.assertEqual(
        expected_variant.alternate_bases, actual_variant.alternate_bases
    )

  @parameterized.parameters(
      # Invalid chromosomes.
      ('chrR:1024:A>C', genome.VariantFormat.DEFAULT),
      ('chrR_1024_A_C', genome.VariantFormat.GTEX),
      ('R_1024_A_C', genome.VariantFormat.OPEN_TARGETS),
      ('R:1024:A:C', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('R-1024-A-C', genome.VariantFormat.GNOMAD),
      # Missing chromosomes.
      (':1024:A>C', genome.VariantFormat.DEFAULT),
      ('_1024_A_C', genome.VariantFormat.GTEX),
      ('_1024_A_C', genome.VariantFormat.OPEN_TARGETS),
      (':1024:A:C', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('-1024-A-C', genome.VariantFormat.GNOMAD),
      # Missing positions.
      ('chr22::A>C', genome.VariantFormat.DEFAULT),
      ('chr22__A_C', genome.VariantFormat.GTEX),
      ('22__A_C', genome.VariantFormat.OPEN_TARGETS),
      ('22::A:C', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('22--A-C', genome.VariantFormat.GNOMAD),
      # Missing chr prefixes.
      ('22:1024:A>C', genome.VariantFormat.DEFAULT),
      ('22_1024_A_C', genome.VariantFormat.GTEX),
      # Unexpected chr prefixes.
      ('chr22_1024_A_C', genome.VariantFormat.OPEN_TARGETS),
      ('chr22:1024:A:C', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('chr22-1024-A-C', genome.VariantFormat.GNOMAD),
      # Unexpected builds.
      ('chr22:1024:A>C_b38', genome.VariantFormat.DEFAULT),
      ('22_1024_A_C_b', genome.VariantFormat.OPEN_TARGETS),
      ('22:1024:A:C_hg38', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      # Mismatched formats.
      ('22-1024-A-C_b38', genome.VariantFormat.GNOMAD),
      ('chr22_1024_A_C', genome.VariantFormat.DEFAULT),
      ('chr22:1024:A>C', genome.VariantFormat.GTEX),
      ('chr22:1024:A>C', genome.VariantFormat.OPEN_TARGETS),
      ('chr22:1024:A>C', genome.VariantFormat.OPEN_TARGETS_BIGQUERY),
      ('chr22:1024:A>C', genome.VariantFormat.GNOMAD),
  )
  def test_from_str_invalid(self, string, variant_format):
    with self.assertRaisesRegex(
        ValueError, f'Invalid format for variant string: {string}'
    ):
      genome.Variant.from_str(string, variant_format=variant_format)

  def test_from_to_string(self):
    v = genome.Variant('chr1', 10, 'C', 'T')
    self.assertIsInstance(str(v), str)
    self.assertEqual(v, genome.Variant.from_str(str(v)))

  def test_from_to_dict(self):
    v = genome.Variant('chr1', 10, 'C', 'T')
    d = v.to_dict()
    self.assertIsInstance(d, dict)
    self.assertEqual(v, genome.Variant.from_dict(d))

  def test_split_alternate_longer(self):
    # Case #1: alternate_bases longer than reference_bases.
    v = genome.Variant('chr1', 3, 'AC', 'TGTC')
    # Anchor before the variant.
    self.assertEqual(
        v.split(2), (None, genome.Variant('chr1', 3, 'AC', 'TGTC'))
    )
    # Anchor at the variant (between A and C in the reference).
    # Note: start=2, end=4.
    self.assertEqual(
        v.split(3),
        (
            genome.Variant('chr1', 3, 'A', 'T'),
            genome.Variant('chr1', 4, 'C', 'GTC'),
        ),
    )
    # Anchor after the variant.
    for split in [4, 5, 6]:
      self.assertEqual(
          v.split(split), (genome.Variant('chr1', 3, 'AC', 'TGTC'), None)
      )

  def test_split_reference_longer(self):
    # Case #2: reference_bases longer than alternate_bases.
    v = genome.Variant('chr1', 3, 'TGTC', 'AC')
    # Anchor before the variant.
    self.assertEqual(
        v.split(2), (None, genome.Variant('chr1', 3, 'TGTC', 'AC'))
    )
    # Anchor at the variant.
    self.assertEqual(
        v.split(3),
        (
            genome.Variant('chr1', 3, 'T', 'A'),
            genome.Variant('chr1', 4, 'GTC', 'C'),
        ),
    )
    self.assertEqual(
        v.split(4),
        (
            genome.Variant('chr1', 3, 'TG', 'AC'),
            genome.Variant('chr1', 5, 'TC', ''),
        ),
    )
    self.assertEqual(
        v.split(5),
        (
            genome.Variant('chr1', 3, 'TGT', 'AC'),
            genome.Variant('chr1', 6, 'C', ''),
        ),
    )
    # Anchor after the variant.
    self.assertEqual(
        v.split(6), (genome.Variant('chr1', 3, 'TGTC', 'AC'), None)
    )

  @parameterized.parameters(
      ('A', 'C', True),
      ('AC', 'A', False),
      ('A', 'AC', False),
      ('AC', 'GT', False),
      ('', 'A', False),
  )
  def test_is_snv(self, ref, alt, expected):
    v = genome.Variant('chr1', 10, ref, alt)
    self.assertEqual(v.is_snv, expected)

  @parameterized.parameters(
      ('A', 'C', False),
      ('AC', 'A', True),
      ('A', 'AC', False),
      ('AC', 'GT', False),
      ('', 'A', False),
      ('A', '', True),
  )
  def test_is_deletion(self, ref, alt, expected):
    v = genome.Variant('chr1', 10, ref, alt)
    self.assertEqual(v.is_deletion, expected)

  @parameterized.parameters(
      ('A', 'C', False),
      ('AC', 'A', False),
      ('A', 'AC', True),
      ('AC', 'GT', False),
      ('', 'A', True),
  )
  def test_is_insertion(self, ref, alt, expected):
    v = genome.Variant('chr1', 10, ref, alt)
    self.assertEqual(v.is_insertion, expected)

  @parameterized.parameters(
      ('A', 'C', False),
      ('AC', 'A', True),
      ('A', 'AC', True),
      ('AC', 'GT', False),
      ('', 'A', True),
      ('A', '', True),
  )
  def test_is_indel(self, ref, alt, expected):
    v = genome.Variant('chr1', 10, ref, alt)
    self.assertEqual(v.is_indel, expected)

  @parameterized.parameters(
      ('A', 'C', False),
      ('AC', 'GTA', True),
      ('', 'AAA', False),
      ('AAAA', 'C', False),
      ('ACG', 'ACGTACGTN', False),
  )
  def test_is_frameshift(self, ref, alt, expected):
    v = genome.Variant('chr1', 10, ref, alt)
    self.assertEqual(v.is_frameshift, expected)

  @parameterized.parameters(
      ('', 'C' * 50, True),
      ('A' * 51, 'C', True),
      ('A', 'AC', False),
  )
  def test_is_structural(self, ref, alt, expected):
    v = genome.Variant('chr1', 10, ref, alt)
    self.assertEqual(v.is_structural, expected)

  @parameterized.parameters(
      ('chr1:1:AAA>GGG', 'TTTNNNN', 'chr1:1:AAA>GGG'),
      ('chr1:1:ATCG>ATCC', 'ATCGNNN', 'chr1:4:G>C'),
      ('chr1:1:ATCG>ATC', 'ATCG', 'chr1:3:CG>C'),
      ('chr1:1:ACTG>GGTG', 'NNNN', 'chr1:1:ACTG>GGTG'),
      ('chr1:1:C>AAA', 'CCC', 'chr1:1:C>AAA'),
      ('chr1:5:C>', 'AAAAA', 'chr1:4:AC>A'),
      ('chr1:2:A>A', 'AAAAAA', 'chr1:2:A>A'),
      ('chr1:1:GAT>GT', 'GAT', 'chr1:1:GA>G'),
      ('chr1:1:GAT>GT', 'GAN', 'chr1:2:AT>T'),
      # Examples from paper.
      ('chr1:6:CAC>C', 'GGGCACACACAGGG', 'chr1:3:GCA>G'),
      ('chr2:3:GCACA>GCA', 'GGGCACACACAGGG', 'chr2:3:GCA>G'),
      ('chr3:2:GGCA>GG', 'GGGCACACACAGGG', 'chr3:3:GCA>G'),
      ('chr4:3:GCA>G', 'GGGCACACACAGGG', 'chr4:3:GCA>G'),
  )
  def test_normalize_variant(
      self, variant: str, reference_sequence: str, expected: str
  ):
    mock_extractor = mock.create_autospec(genome._FastaExtractorType)  # pylint: disable=protected-access
    variant = genome.Variant.from_str(variant)

    def _extract(interval):
      return reference_sequence[interval.start : interval.end]

    mock_extractor.extract.side_effect = _extract

    result = genome.normalize_variant(variant, mock_extractor)
    self.assertEqual(str(result), expected)


class JunctionTest(parameterized.TestCase):

  def test_init_valid(self):
    genome.Junction(chromosome='chr1', start=10, end=10, strand='+')
    genome.Junction(chromosome='chr1', start=10, end=20, strand='-', k=1)

  def test_unstranded_junction_raises_value_error(self):
    with self.assertRaises(ValueError):
      genome.Junction(chromosome='chr1', start=10, end=10)

  def test_invalid_start_end_raises_value_error(self):
    with self.assertRaises(ValueError):
      genome.Junction(chromosome='chr1', start=10, end=1, strand='+')

  def test_attributes(self):
    junction = genome.Junction(
        'chr1',
        10,
        20,
        strand='-',
        info=dict(a=1, b='b', c=[1, 2], d=dict(e=1)),
        k=1,
    )
    self.assertEqual(junction.start, 10)
    self.assertEqual(junction.end, 20)
    self.assertEqual(junction.chromosome, 'chr1')
    self.assertEqual(junction.name, '')
    self.assertIsInstance(junction.info, dict)
    self.assertEqual(junction.info['a'], 1)
    self.assertTrue(junction.negative_strand)
    self.assertEqual(junction.width, 10)
    self.assertEqual(junction.k, 1)

  def test_copy(self):
    junction = genome.Junction('chr1', 10, 20, strand='-', k=1)
    junction_copy = junction.copy()
    junction_copy.info['test'] = 10
    self.assertEqual(junction_copy.info['test'], 10)
    self.assertEmpty(junction.info)
    self.assertEqual(junction.k, junction_copy.k)

  def test_acceptor_donor(self):
    junction = genome.Junction('chr1', 10, 20, strand='-')
    self.assertEqual(junction.acceptor, 10)
    self.assertEqual(junction.donor, 20)

    junction = genome.Junction('chr1', 10, 20, strand='+')
    self.assertEqual(junction.acceptor, 20)
    self.assertEqual(junction.donor, 10)

  def test_dinucleotide_region(self):
    junction = genome.Junction('chr1', 10, 20, strand='-')
    self.assertEqual(
        junction.dinucleotide_region(),
        (
            genome.Interval('chr1', 10, 12, '-'),
            genome.Interval('chr1', 18, 20, '-'),
        ),
    )

  def test_acceptor_donor_region(self):
    junction = genome.Junction('chr1', 10, 20, strand='-')
    self.assertEqual(
        junction.acceptor_region(),
        genome.Interval('chr1', -240, 260, '-'),
    )
    self.assertEqual(
        junction.donor_region(),
        genome.Interval('chr1', -230, 270, '-'),
    )


class IntervalIntersectionTest(parameterized.TestCase):

  def _parse_intervals(self, intervals):
    if not intervals:
      return []
    return [genome.Interval.from_str(i) for i in intervals.split(',')]

  @parameterized.named_parameters(
      ('no_overlap_1', 'chr1:0-100', 'chr2:0-100', ''),
      ('no_overlap_2', 'chr1:0-100', 'chr1:200-300', ''),
      ('overlap_1', 'chr1:0-100', 'chr1:50-150', 'chr1:50-100'),
      (
          'overlap_2',
          'chr1:0-100,chr2:100-200',
          'chr2:50-150,chr1:50-150',
          'chr1:50-100,chr2:100-150',
      ),
      (
          'overlap_3',
          'chr1:0-100,chr1:150-200',
          'chr1:50-200',
          'chr1:50-100,chr1:150-200',
      ),
      (
          'point_intervals',
          'chr1:10-10,chr1:15-15,chr1:20-20',
          'chr1:5-15',
          'chr1:10-10,chr1:15-15',
      ),
      ('point_intervals_2', 'chr1:10-10,chr1:10-20', 'chr1:5-15', 'chr1:10-15'),
      ('point_intervals_3', 'chr1:0-10,chr1:10-10', 'chr1:5-15', 'chr1:5-10'),
      (
          'abutting_intervals',
          'chr1:10-20,chr1:20-30',
          'chr1:15-25',
          'chr1:15-25',
      ),
      ('abutting_intervals_2', 'chr1:0-100', 'chr1:100-200', 'chr1:100-100'),
  )
  def test_intersection(self, lhs, rhs, expected):
    with self.subTest('forward'):
      self.assertSameElements(
          self._parse_intervals(expected),
          list(
              genome.intersect_intervals(
                  self._parse_intervals(lhs), self._parse_intervals(rhs)
              )
          ),
      )
    with self.subTest('reverse'):
      self.assertSameElements(
          self._parse_intervals(expected),
          list(
              genome.intersect_intervals(
                  self._parse_intervals(rhs), self._parse_intervals(lhs)
              )
          ),
      )


class IntervalUnionTest(parameterized.TestCase):

  def _parse_intervals(self, intervals):
    if not intervals:
      return []
    return [genome.Interval.from_str(i) for i in intervals.split(',')]

  @parameterized.named_parameters(
      ('no_overlap_1', 'chr1:0-100', 'chr2:0-100', 'chr1:0-100,chr2:0-100'),
      ('no_overlap_2', 'chr1:0-100', 'chr1:200-300', 'chr1:0-100,chr1:200-300'),
      ('overlap_1', 'chr1:0-100', 'chr1:50-150', 'chr1:0-150'),
      (
          'overlap_2',
          'chr1:0-100,chr2:100-200',
          'chr2:50-150,chr1:50-150',
          'chr1:0-150,chr2:50-200',
      ),
      ('overlap_3', 'chr1:0-100,chr1:150-200', 'chr1:50-200', 'chr1:0-200'),
      (
          'point_intervals',
          'chr1:10-10,chr1:15-15,chr1:20-20',
          'chr1:5-15',
          'chr1:5-15,chr1:20-20',
      ),
      ('point_intervals_2', 'chr1:10-10,chr1:10-20', 'chr1:5-15', 'chr1:5-20'),
      ('point_intervals_3', 'chr1:0-10,chr1:10-10', 'chr1:5-15', 'chr1:0-15'),
      (
          'abutting_intervals',
          'chr1:10-20,chr1:20-30',
          'chr1:15-25',
          'chr1:10-30',
      ),
      ('abutting_intervals_2', 'chr1:0-100', 'chr1:100-200', 'chr1:0-200'),
  )
  def test_union(self, lhs, rhs, expected):
    with self.subTest('forward'):
      self.assertSameElements(
          self._parse_intervals(expected),
          list(
              genome.union_intervals(
                  self._parse_intervals(lhs), self._parse_intervals(rhs)
              )
          ),
      )
    with self.subTest('reverse'):
      self.assertSameElements(
          self._parse_intervals(expected),
          list(
              genome.union_intervals(
                  self._parse_intervals(rhs), self._parse_intervals(lhs)
              )
          ),
      )


class MergeIntervalsTest(parameterized.TestCase):

  def test_empty_list(self):
    self.assertEmpty(genome.merge_overlapping_intervals([]))

  def test_one_interval(self):
    intervals = [genome.Interval('chr1', 1, 100)]
    self.assertListEqual(
        genome.merge_overlapping_intervals(intervals), intervals
    )

  def test_no_overlap(self):
    intervals = [
        genome.Interval('chr1', 1, 100),
        genome.Interval('chr1', 101, 200),
    ]
    self.assertListEqual(
        genome.merge_overlapping_intervals(intervals), intervals
    )

  def test_interval_within(self):
    intervals = [
        genome.Interval('chr1', 1, 100),
        genome.Interval('chr1', 10, 100),
    ]
    self.assertListEqual(
        genome.merge_overlapping_intervals(intervals), [intervals[0]]
    )

  def test_single_overlap(self):
    intervals = [
        genome.Interval('chr1', 1, 100),
        genome.Interval('chr1', 90, 200),
        genome.Interval('chr1', 300, 400),
    ]
    self.assertListEqual(
        genome.merge_overlapping_intervals(intervals),
        [genome.Interval('chr1', 1, 200), intervals[-1]],
    )

  def test_multiple_overlap(self):
    intervals = [
        genome.Interval('chr1', 1, 100),
        genome.Interval('chr1', 100, 200),
        genome.Interval('chr1', 200, 300),
        genome.Interval('chr1', 500, 700),
        genome.Interval('chr1', 700, 800),
        genome.Interval('chr1', 750, 801),
    ]
    self.assertListEqual(
        genome.merge_overlapping_intervals(intervals),
        [genome.Interval('chr1', 1, 300), genome.Interval('chr1', 500, 801)],
    )

  def test_multiple_overlap_unsorted(self):
    intervals = [
        genome.Interval('chr1', 1, 100),
        genome.Interval('chr1', 200, 300),
        genome.Interval('chr1', 700, 800),
        genome.Interval('chr1', 750, 801),
        genome.Interval('chr1', 500, 700),
        genome.Interval('chr1', 100, 200),
    ]
    self.assertListEqual(
        genome.merge_overlapping_intervals(intervals),
        [genome.Interval('chr1', 1, 300), genome.Interval('chr1', 500, 801)],
    )

  def test_info_field(self):
    original_intervals = [
        genome.Interval(
            'chr1', 1, 100, info={'transcript_type': 'protein_coding'}
        ),
        genome.Interval('chr1', 101, 200),
    ]
    merged_intervals = genome.merge_overlapping_intervals(original_intervals)
    self.assertListEqual(merged_intervals, original_intervals)
    for interval in merged_intervals:
      self.assertEmpty(interval.info)


if __name__ == '__main__':
  absltest.main()
