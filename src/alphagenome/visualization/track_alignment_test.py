"""Tests for track_alignment."""

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.visualization import track_alignment
import numpy as np
import pandas as pd


class TrackAlignmentTest(parameterized.TestCase):

  def test_align_tracks_deletion_1bp(self):
    # Setup simple REF track
    resolution = 1
    ref_start = 100
    ref_len = 10
    ref_vals = np.arange(ref_len, dtype=np.float32).reshape(
        ref_len, 1
    )  # [0, 1, 2, ..., 9]
    ref_metadata = pd.DataFrame({'name': ['t1'], 'strand': ['.']})
    ref_interval = genome.Interval('chr1', ref_start, ref_start + ref_len)
    ref_track = track_data.TrackData(
        ref_vals, ref_metadata, resolution=resolution, interval=ref_interval
    )

    # Variant: Deletion at 103 (0-based index of first ref base).
    # ref="ABC" (length 3), alt="A" (length 1).
    # Variant start = 103.
    # Gap should correspond to REF bases at 103+1=104 and 103+2=105.

    # Construct ALT track.
    # Sequence: 100..102 (3) | 103 (1) | 106..109 (4)
    # Total length: 3 + 1 + 4 = 8.
    alt_vals = np.array(
        [
            0,
            1,
            2,  # 100, 101, 102
            3,  # 103 (matches REF 103)
            6,
            7,
            8,
            9,  # 106, 107, 108, 109
        ],
        dtype=np.float32,
    ).reshape(8, 1)

    alt_len = len(alt_vals)
    alt_interval = genome.Interval(
        'chr1', ref_start, ref_start + alt_len
    )
    alt_track = track_data.TrackData(
        alt_vals, ref_metadata, resolution=resolution, interval=alt_interval
    )

    variant = genome.Variant(
        chromosome='chr1',
        position=104,  # 1-based. 104 -> index 103 0-based.
        reference_bases='ATG',  # len 3
        alternate_bases='A',  # len 1
    )

    ref_aligned, alt_aligned = track_alignment.align_ref_alt_tracks(
        ref_track, alt_track, variant
    )

    self.assertEqual(ref_aligned.interval, alt_aligned.interval)
    self.assertEqual(ref_aligned.values.shape, alt_aligned.values.shape)

    # Check values
    # 0, 1, 2 should match
    np.testing.assert_array_almost_equal(
        alt_aligned.values[:3], [[0], [1], [2]]
    )
    # 3 should match
    np.testing.assert_array_almost_equal(alt_aligned.values[3:4], [[3]])

    # Gap: indices 4 and 5 (positions 104, 105).
    self.assertTrue(np.isnan(alt_aligned.values[4]))
    self.assertTrue(np.isnan(alt_aligned.values[5]))

    # After gap: 6, 7, 8, 9. Index 6, 7, 8, 9.
    np.testing.assert_array_almost_equal(
        alt_aligned.values[6:], [[6], [7], [8], [9]]
    )

  def test_align_tracks_insertion_1bp(self):
    resolution = 1
    ref_start = 100
    ref_len = 10
    ref_vals = np.arange(ref_len, dtype=np.float32).reshape(ref_len, 1)
    ref_metadata = pd.DataFrame({'name': ['t1'], 'strand': ['.']})
    ref_interval = genome.Interval('chr1', ref_start, ref_start + ref_len)
    ref_track = track_data.TrackData(
        ref_vals, ref_metadata, resolution=resolution, interval=ref_interval
    )

    # Insertion: ref=A, alt=ATG. len=1, len=3.
    # at 103.
    # ALT vals:
    # 0, 1, 2
    # 3.0, 3.3, 3.6 (inserted/expanded region)
    # 4, 5, 6, 7, 8, 9
    alt_vals = np.array(
        [0, 1, 2, 3, 3.3, 3.6, 4, 5, 6, 7, 8, 9], dtype=np.float32
    ).reshape(12, 1)

    alt_interval = genome.Interval('chr1', ref_start, ref_start + 12)
    alt_track = track_data.TrackData(
        alt_vals, ref_metadata, resolution=resolution, interval=alt_interval
    )

    variant = genome.Variant(
        chromosome='chr1',
        position=104,  # 1-based. index 103.
        reference_bases='A',
        alternate_bases='ATG',
    )

    ref_aligned, alt_aligned = track_alignment.align_ref_alt_tracks(
        ref_track, alt_track, variant
    )

    self.assertEqual(
        ref_aligned.values.shape, alt_aligned.values.shape
    )  # Should be 10 back.

    # 0..2 match
    np.testing.assert_array_almost_equal(
        alt_aligned.values[:3], [[0], [1], [2]]
    )

    # At 3 (103): We expect compression.
    # Points 3, 3.3, 3.6 map to [3, 4) in REF.
    # REF index 3 corresponds to native 3.
    # REF index 4 corresponds to native 4 (value 4).
    # The intermediate values might be skipped or interpolated depending on grid.
    np.testing.assert_array_almost_equal(alt_aligned.values[3], [3.3])
    np.testing.assert_array_almost_equal(alt_aligned.values[4], [4])

  def test_align_tracks_deletion_coarse_resolution(self):
    # Resolution 10.
    # Ref len 100. 10 bins.
    res = 10
    ref_interval = genome.Interval('chr1', 1000, 1100)
    ref_vals = np.zeros((10, 1), dtype=np.float32)
    ref_metadata = pd.DataFrame({'name': ['t1'], 'strand': ['.']})
    ref_track = track_data.TrackData(
        ref_vals, ref_metadata, resolution=res, interval=ref_interval
    )

    # Deletion of 20bp at 1040. (2 bins).
    # REF: 1040-1060 deleted.
    # Variant: pos 1041 (1-based), so index 1040.
    variant = genome.Variant(
        'chr1', 1041, 'A' * 21, 'A'
    )  # 21 bases -> 1 base. Delta 20.

    # ALT track: length 80. (8 bins).
    alt_vals = np.ones((8, 1), dtype=np.float32)
    alt_interval = genome.Interval('chr1', 1000, 1080)
    alt_track = track_data.TrackData(
        alt_vals, ref_metadata, resolution=res, interval=alt_interval
    )

    ref_aligned, alt_aligned = track_alignment.align_ref_alt_tracks(
        ref_track, alt_track, variant
    )

    # Bins 1040, 1050 are in the deleted region [1041, 1061).
    # Centers 1045, 1055 fall inside.
    self.assertTrue(np.isnan(alt_aligned.values[4]))
    self.assertTrue(np.isnan(alt_aligned.values[5]))
    self.assertEqual(alt_aligned.values[0, 0], 1.0)
    self.assertEqual(alt_aligned.values[7, 0], 1.0)


if __name__ == '__main__':
  absltest.main()
