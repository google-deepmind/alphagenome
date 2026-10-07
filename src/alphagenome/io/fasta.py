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

"""Extractors for working with FASTA files."""

import os

from alphagenome.data import genome
import fsspec
import pyfaidx


def reverse_complement(sequence: str) -> str:
  """Returns the reverse complement of a DNA sequence string."""
  return pyfaidx.complement(sequence)[::-1]


class FastaExtractor:
  """FASTA file extractor."""

  def __init__(self, fasta_path: str | os.PathLike[str]):
    self._faidx = pyfaidx.Faidx(
        fsspec.open(fasta_path),
        as_raw=True,
        mutable=False,
        build_index=False,
        sequence_always_upper=True,
    )

  def extract(self, interval: genome.Interval) -> str:
    """Returns the FASTA sequence in some given interval as a string.

    Args:
        interval: the interval to query.

    Returns:
        sequence of requested interval.
    """
    if (chromosome := self._faidx.index.get(interval.chromosome)) is None:
      raise ValueError(f'Chromosome "{interval.chromosome}" not found.')

    chromosome_length = chromosome.rlen

    if interval.start >= chromosome_length or interval.end < 0:
      raise ValueError(f'Interval fully out of bounds. {interval=}')
    elif interval.within_reference(chromosome_length):
      sequence = self._faidx.fetch(
          interval.chromosome, interval.start + 1, interval.end
      )
    else:
      start_padding = 'N' * max(-interval.start, 0)
      end_padding = 'N' * max(interval.end - chromosome_length, 0)
      trimmed_interval = interval.truncate(chromosome_length)
      sequence = self._faidx.fetch(
          interval.chromosome, trimmed_interval.start + 1, trimmed_interval.end
      )
      sequence = start_padding + sequence + end_padding

    if interval.negative_strand:
      return reverse_complement(sequence)
    else:
      return sequence

  @property
  def sequence_names(self) -> list[str]:
    """Returns the list of all sequence (chromosome) names in the FASTA."""
    return list(self._faidx.index.keys())

  def get_length_for_sequence_name(self, name: str) -> int:
    """Returns the length of the named sequence."""
    if name not in self._faidx.index:
      raise ValueError(f'Chromosome "{name}" not found.')
    return self._faidx.index[name].rlen

  def __contains__(self, name: str) -> bool:
    """Returns True if the FASTA contains a sequence with the given name."""
    return name in self._faidx.index
