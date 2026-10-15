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


"""Reverse complementation requires a strand partner for each track name."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome
from alphagenome.data import track_data
import numpy as np
import pandas as pd


def make_data(pairs, rank=1):
  metadata = pd.DataFrame(pairs, columns=['name', 'strand'])
  metadata.index = np.arange(len(pairs)) * 3 + 7
  shape = (4,) * rank + (len(pairs),)
  values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
  interval = genome.Interval('chr1', 10, 18, strand='+') if rank else None
  return track_data.TrackData(values, metadata, resolution=2, interval=interval)


class TrackReversePairsTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 2)
  def test_mixed_paired_and_unpaired_names_are_rejected_even_with_balanced_counts(
      self, rank
  ):
    pairs = [('paired', '+'), ('paired', '-'), ('one', '+'), ('other', '-')]
    for order in itertools.permutations(pairs):
      data = make_data(order, rank)
      with self.assertRaisesRegex(ValueError, 'Not all stranded tracks'):
        data.reverse_complement()

  @parameterized.product(rank=[0, 1, 2], num_tracks=[0, 1, 3])
  def test_unstranded_tracks_only_reverse_the_positional_axes(
      self, rank, num_tracks
  ):
    data = make_data([(str(i), '.') for i in range(num_tracks)], rank)
    actual = data.reverse_complement()
    expected = np.flip(data.values, axis=tuple(data.positional_axes))
    np.testing.assert_array_equal(actual.values, expected)
    pd.testing.assert_frame_equal(actual.metadata, data.metadata)
    self.assertEqual(actual.resolution, data.resolution)
    if rank:
      self.assertEqual(actual.interval, data.interval.swap_strand())
    else:
      self.assertIsNone(actual.interval)

  @parameterized.parameters(0, 1, 2)
  def test_valid_pairs_match_name_lookup_and_round_trip(self, rank):
    pairs = [('b', '-'), ('a', '+'), ('plain', '.'), ('b', '+'), ('a', '-')]
    data = make_data(pairs, rank)
    original = data.values.copy()
    lookup = {key: i for i, key in enumerate(pairs)}
    indices = [
        lookup[(name, {'+': '-', '-': '+', '.': '.'}[strand])]
        for name, strand in pairs
    ]
    actual = data.reverse_complement()
    expected = np.flip(data.values, axis=tuple(data.positional_axes))[
        ..., indices
    ]
    np.testing.assert_array_equal(actual.values, expected)
    pd.testing.assert_frame_equal(actual.metadata, data.metadata.iloc[indices])
    restored = actual.reverse_complement()
    np.testing.assert_array_equal(restored.values, data.values)
    pd.testing.assert_frame_equal(restored.metadata, data.metadata)
    self.assertEqual(restored.interval, data.interval)
    np.testing.assert_array_equal(data.values, original)

  def test_all_unpaired_names_still_raise(self):
    with self.assertRaises(ValueError):
      make_data([('a', '+'), ('b', '-')]).reverse_complement()


if __name__ == '__main__':
  absltest.main()
