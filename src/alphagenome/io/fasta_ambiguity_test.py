# Copyright 2026 Google LLC.
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

"""DNA reverse complements preserve ambiguous-base meaning and case."""

import itertools
import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome
from alphagenome.io import fasta
import pyfaidx

_BASE_SETS = {
    'A': 'A',
    'C': 'C',
    'G': 'G',
    'T': 'T',
    'R': 'AG',
    'Y': 'CT',
    'S': 'CG',
    'W': 'AT',
    'K': 'GT',
    'M': 'AC',
    'B': 'CGT',
    'D': 'AGT',
    'H': 'ACT',
    'V': 'ACG',
    'N': 'ACGT',
}
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def reference(sequence):
  reverse_lookup = {
      frozenset(bases): code for code, bases in _BASE_SETS.items()
  }
  output = []
  for code in reversed(sequence):
    result = reverse_lookup[
        frozenset(_COMPLEMENT[b] for b in _BASE_SETS[code.upper()])
    ]
    output.append(result.lower() if code.islower() else result)
  return ''.join(output)


class ReverseComplementAmbiguityTest(parameterized.TestCase):

  @parameterized.product(code=list(_BASE_SETS), lowercase=[False, True])
  def test_complement_represents_the_complemented_base_set(
      self, code, lowercase
  ):
    sequence = code.lower() if lowercase else code
    self.assertEqual(fasta.reverse_complement(sequence), reference(sequence))

  def test_every_pair_preserves_order_and_base_meaning(self):
    for pair in itertools.product(_BASE_SETS, repeat=2):
      sequence = ''.join(pair)
      self.assertEqual(fasta.reverse_complement(sequence), reference(sequence))

  def test_mixed_case_round_trip_preserves_soft_masking(self):
    sequence = 'AcGtRySwKmBdHvN'
    self.assertEqual(fasta.reverse_complement(sequence), reference(sequence))
    self.assertEqual(
        fasta.reverse_complement(fasta.reverse_complement(sequence)), sequence
    )
    self.assertEqual(fasta.reverse_complement(''), '')

  @parameterized.parameters((0, 15), (-2, 17), (2, 10))
  def test_negative_strand_fasta_extraction_complements_ambiguities(
      self, start, end
  ):
    sequence = 'ACGTRYSWKMBDHVN'
    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'example.fa'
      path.write_text('>chr1\n' + sequence + '\n')
      indexed = pyfaidx.Fasta(path)
      indexed.close()
      extractor = fasta.FastaExtractor(path)
      forward = extractor.extract(
          genome.Interval('chr1', start, end, strand='+')
      )
      reverse = extractor.extract(
          genome.Interval('chr1', start, end, strand='-')
      )
      expected = (
          'N' * max(-start, 0)
          + sequence[max(start, 0) : min(end, len(sequence))]
          + 'N' * max(end - len(sequence), 0)
      )
      self.assertEqual(forward, expected)
      self.assertEqual(reverse, reference(expected))
      self.assertLen(reverse, end - start)


if __name__ == '__main__':
  absltest.main()
