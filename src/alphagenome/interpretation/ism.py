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
"""In-silico mutagenesis (ISM) functions for sequence interpretation."""

from collections.abc import Sequence

import numpy as np

from alphagenome.data import genome


def ism_variants(
    interval: genome.Interval,
    sequence: str,
    vocabulary: str = 'ACGT',
    skip_n: bool = False,
) -> list[genome.Variant]:
  """Create a list of all possible single nucleotide variants for an interval.

  Args:
    interval: Interval for which to generate the variants.
    sequence: Sequence extracted from the reference genome at the `interval`.
      Needs to have the same length as `interval.width`.
    vocabulary: Vocabulary of possible alternative bases contained in
      `sequence`.
    skip_n: If True, skip the N bases.

  Returns:
    List of all possible single nucleotide variants for a genomic interval.
  """
  if len(sequence) != interval.width:
    raise ValueError('Sequence must have the same length as interval.')
  variants = []
  for position in range(interval.width):
    reference_base = sequence[position]
    if skip_n and reference_base == 'N':
      continue
    for alternative_base in vocabulary:
      if reference_base == alternative_base:
        continue
      variants.append(
          genome.Variant(
              chromosome=interval.chromosome,
              reference_bases=reference_base,
              alternate_bases=alternative_base,
              position=interval.start + position + 1,
          )
      )
  return variants


def ism_matrix(
    variant_scores: Sequence[float],
    variants: Sequence[genome.Variant],
    interval: genome.Interval | None = None,
    multiply_by_sequence: bool = True,
    vocabulary: str = 'ACGT',
    require_fully_filled: bool = True,
) -> np.ndarray:
  """Construct the ISM (position, base) matrix from individual ISM scores.

  This function returns the relative effect of the variants compared to the
  per-position average: score[position, base] - mean(score[position, :]]).

  Args:
    variant_scores: Variant effect scores corresponding to the variants. These
      could be obtained from the score_variants() output summarised to a single
      scalar. Summarisation could be for example be obtained by selecting a
      specific variant scorer output and extract a specific value from the
      (variant, track) matrix.
    variants: Sequence of variants used to transform into the ISM matrix.
    interval: Interval for which to get the contribution scores. All variants
      need to be contained within that interval. If None, it will be
      automatically inferred from variants.
    multiply_by_sequence: If True, only return non-zero values at
      one-hot-encoded reference genome sequence bases.
    vocabulary: Vocabulary of possible alternative bases contained in
      `sequence`. The order determines the column order of the returned matrix.
    require_fully_filled: If True, raise an error if not all positions are
      covered by variants.

  Returns:
    Matrix of shape (interval.width, 4) containing variant scores.
  """
  if len(variants) != len(variant_scores):
    raise ValueError(
        'Variants and variant_scores need to have the same length.'
    )
  if interval is None:
    interval = genome.Interval(
        chromosome=variants[0].chromosome,
        start=min(variant.start for variant in variants),
        end=max(variant.end for variant in variants),
    )
  scores = np.zeros((interval.width, 4), dtype=np.float32)
  filled = np.zeros((interval.width, 4), dtype=bool)
  base_index = {base: i for i, base in enumerate(vocabulary)}

  for variant, score in zip(variants, variant_scores, strict=True):
    if len(variant.alternate_bases) != 1 or len(variant.reference_bases) != 1:
      # Only looking for single nucleotide variants.
      continue
    if not interval.contains(variant.reference_interval):
      continue
    position = variant.start - interval.start
    scores[position, base_index[variant.alternate_bases]] = score
    filled[position, base_index[variant.alternate_bases]] = True

  # Check that all positions were covered by variants.
  if (filled.sum(axis=-1) == 0).any() and require_fully_filled:
    missing_positions = list(
        np.where(filled.sum(axis=-1) == 0)[0] + interval.start + 1
    )
    raise ValueError(
        'No variants were found for the (1-based) positions on chromosome '
        f'{interval.chromosome}: {missing_positions}'
    )
  elif (filled.sum(axis=-1) == 4).any():
    full_positions = list(
        np.where(filled.sum(axis=-1) == 4)[0] + interval.start + 1
    )
    raise ValueError(
        'Some variants were found with 4 different alternative bases for the'
        ' position instead of 3. List of positions on chromosome '
        f'{interval.chromosome}: {full_positions}'
    )
  elif (filled.sum(axis=-1) != 3).any() and require_fully_filled:
    missing_positions = list(
        np.where(filled.sum(axis=-1) != 3)[0] + interval.start + 1
    )
    raise ValueError(
        'Some variants were found with only 1 or 2 alternative bases for the'
        ' position instead of 3. List of positions on chromosome '
        f'{interval.chromosome}: {missing_positions}'
    )

  scores -= np.sum(scores, axis=-1, keepdims=True) / (len(vocabulary) - 1)
  if multiply_by_sequence:
    scores = scores * (~filled)
  return scores
