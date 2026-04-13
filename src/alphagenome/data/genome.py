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

"""Utilities for working with genome-related objects such as intervals."""

import collections
import copy
import dataclasses
import enum
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from typing_extensions import Self

from alphagenome.protos import dna_model_pb2

STRAND_POSITIVE = '+'  # Also called forward strand, 5'->3' direction.
STRAND_NEGATIVE = '-'  # Also called negative strand, 3'->5' direction.
STRAND_UNSTRANDED = '.'
STRAND_OPTIONS = (STRAND_POSITIVE, STRAND_NEGATIVE, STRAND_UNSTRANDED)
_INTERVAL_START_END_REGEX = re.compile(r'(-?\d+)-(-?\d+)')
VALID_VARIANT_BASES = frozenset('ACGTN')


class Strand(enum.IntEnum):
  """Represents the strand of a DNA sequence.

  This enum defines the possible strands for a DNA sequence:

    * `POSITIVE`:  The forward strand (5' to 3').
    * `NEGATIVE`: The reverse strand (3' to 5').
    * `UNSTRANDED`:  The strand is not specified.
  """

  POSITIVE = enum.auto()
  NEGATIVE = enum.auto()
  UNSTRANDED = enum.auto()

  def __str__(self):
    match self:
      case Strand.POSITIVE:
        return STRAND_POSITIVE
      case Strand.NEGATIVE:
        return STRAND_NEGATIVE
      case Strand.UNSTRANDED:
        return STRAND_UNSTRANDED

  @classmethod
  def from_str(cls, strand: str) -> Self:
    match strand:
      case '+':
        return cls.POSITIVE
      case '-':
        return cls.NEGATIVE
      case '.':
        return cls.UNSTRANDED
      case _:
        raise ValueError(f'Strand needs to be in {STRAND_OPTIONS}')

  def to_proto(self) -> dna_model_pb2.Strand:
    match self:
      case Strand.POSITIVE:
        return dna_model_pb2.Strand.STRAND_POSITIVE
      case Strand.NEGATIVE:
        return dna_model_pb2.Strand.STRAND_NEGATIVE
      case Strand.UNSTRANDED:
        return dna_model_pb2.Strand.STRAND_UNSTRANDED

  @classmethod
  def from_proto(cls, strand: dna_model_pb2.Strand) -> Self:
    match strand:
      case dna_model_pb2.Strand.STRAND_POSITIVE:
        return cls.POSITIVE
      case dna_model_pb2.Strand.STRAND_NEGATIVE:
        return cls.NEGATIVE
      case dna_model_pb2.Strand.STRAND_UNSTRANDED:
        return cls.UNSTRANDED
      case _:
        raise ValueError(f'Strand needs to be in {STRAND_OPTIONS}')


PYRANGES_INTERVAL_COLUMNS = ('Chromosome', 'Start', 'End', 'Strand', 'Name')


@dataclasses.dataclass(order=True)
class Interval:
  """Represents a genomic interval.

  A genomic interval is a region on a chromosome defined by a start and end
  position. This class provides methods for manipulating and comparing
  intervals, and for calculating coverage and overlap.

  Attributes:
    chromosome: The chromosome name (e.g., 'chr1', '1').
    start: The 0-based start position.
    end: The 0-based end position (must be greater than or equal to start).
    strand: The strand of the interval ('+', '-', or '.'). Defaults to '.'
      (unstranded).
    name: An optional name for the interval.
    info: An optional dictionary to store additional information.
    negative_strand: True if the interval is on the negative strand, False
      otherwise.
    width: The width of the interval (end - start).
  """

  chromosome: str
  start: int
  end: int
  strand: str = STRAND_UNSTRANDED
  name: str = dataclasses.field(default='', compare=False, hash=False)
  info: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False, hash=False
  )

  def __post_init__(self):
    if self.end < self.start:
      raise ValueError('end < start. Interval: ' + str(self))
    if self.strand not in STRAND_OPTIONS:
      raise ValueError(
          f'Strand needs to be in {STRAND_OPTIONS}, found {self.strand}.'
      )

  @property
  def negative_strand(self) -> bool:
    """Returns True if interval is on the negative strand, False otherwise."""
    return self.strand == STRAND_NEGATIVE

  @property
  def width(self) -> int:
    """Returns the width of the interval."""
    return self.end - self.start

  def copy(self) -> Self:
    """Returns a deep copy of the interval."""
    return copy.deepcopy(self)

  def __str__(self) -> str:
    """Returns a string representation of the interval."""
    return f'{self.chromosome}:{self.start}-{self.end}:{self.strand}'

  @classmethod
  def from_str(cls, string: str) -> Self:
    """Creates an Interval from a string (e.g., 'chr1:100-200:+')."""
    chromosome, interval, *strand = string.split(':', maxsplit=2)
    if strand:
      strand = strand[0]
    else:
      strand = STRAND_UNSTRANDED
    # Get start and end from the interval string.
    match = _INTERVAL_START_END_REGEX.fullmatch(interval)
    if match:
      start, end = int(match.group(1)), int(match.group(2))
    else:
      raise ValueError(f'Invalid interval: {string}')
    return cls(
        chromosome=chromosome, start=int(start), end=int(end), strand=strand
    )

  def to_proto(self) -> dna_model_pb2.Interval:
    """Converts the interval to a protobuf message."""
    return dna_model_pb2.Interval(
        chromosome=self.chromosome,
        start=self.start,
        end=self.end,
        strand=Strand.from_str(self.strand).to_proto(),
    )

  @classmethod
  def from_proto(cls, proto: dna_model_pb2.Interval) -> Self:
    """Creates an Interval from a protobuf message."""
    return cls(
        chromosome=proto.chromosome,
        start=proto.start,
        end=proto.end,
        strand=str(Strand.from_proto(proto.strand)),
    )

  def to_interval_dict(self) -> dict[str, str | int]:
    """Converts the interval to a dictionary."""
    return dict(
        chrom=self.chromosome,
        start=self.start,
        end=self.end,
        strand=self.strand,
    )

  @classmethod
  def from_interval_dict(cls, interval: Mapping[str, str | int]) -> Self:
    """Creates an Interval from a dictionary."""
    return cls(
        chromosome=str(interval['chrom']),
        start=int(interval['start']),
        end=int(interval['end']),
        strand=str(interval.get('strand', STRAND_UNSTRANDED)),
    )

  @classmethod
  def from_pyranges_dict(
      cls, row: Mapping[str, Any], ignore_info: bool = False
  ) -> 'Interval':
    """Creates an Interval from a pyranges-like dictionary.

    This method constructs an `Interval` object from a dictionary that follows
    the pyranges format, such as a row from a :class:`pandas.DataFrame`
    converted to a dict.

    The dictionary should have the following keys:

      * 'Chromosome': The chromosome name.
      * 'Start': The start position.
      * 'End': The end position.
      * 'Strand': The strand (optional, defaults to unstranded).
      * 'Name': The interval name (optional).

    Any other keys in the dictionary will be added to the `info` attribute of
    the `Interval` object, unless `ignore_info` is set to True.

    Args:
      row: A dictionary containing interval data.
      ignore_info: If True, any keys in the dictionary that are not part of the
        standard pyranges columns ('Chromosome', 'Start', 'End', 'Strand',
        'Name') will not be added to the `info` attribute.

    Returns:
      An `Interval` object created from the input dictionary.
    """
    if ignore_info:
      info = {}
    else:
      info = {
          k: v for k, v in row.items() if k not in PYRANGES_INTERVAL_COLUMNS
      }

    return cls(
        chromosome=str(row['Chromosome']),
        start=int(row['Start']),
        end=int(row['End']),
        strand=str(row.get('Strand', STRAND_UNSTRANDED)),
        name=str(row.get('Name', '')),
        info=info,
    )

  def to_pyranges_dict(self) -> dict[str, int | str]:
    """Converts the interval to a pyranges-like dictionary."""
    return {
        'Chromosome': self.chromosome,
        'Start': self.start,
        'End': self.end,
        'Name': self.name,
        'Strand': self.strand,
        **self.info,
    }

  def swap_strand(self) -> Self:
    """Swaps the strand of the interval."""
    obj = self.copy()
    if obj.strand == STRAND_POSITIVE:
      obj.strand = STRAND_NEGATIVE
    elif obj.strand == STRAND_NEGATIVE:
      obj.strand = STRAND_POSITIVE
    elif obj.strand == STRAND_UNSTRANDED:
      raise ValueError('Cannot swap unstranded intervals.')
    return obj

  def as_unstranded(self) -> Self:
    """Returns an unstranded copy of the interval."""
    obj = self.copy()
    obj.strand = STRAND_UNSTRANDED
    return obj

  def within_reference(self, reference_length: int = sys.maxsize) -> bool:
    """Checks if the interval is within the valid reference range."""
    return self.start >= 0 and self.end <= reference_length

  def truncate(self, reference_length: int = sys.maxsize) -> Self:
    """Truncates the interval to fit within the valid reference range."""
    obj = self.copy()
    if reference_length <= 0:
      raise ValueError('Reference length should be larger than 0.')
    if self.within_reference(reference_length):
      return obj
    else:
      obj.start = max(self.start, 0)
      obj.end = min(self.end, reference_length)
      return obj

  def center(self, use_strand: bool = True) -> int:
    """Computes the center of the interval.

    For intervals with an odd width, the center is rounded up for
    positive/unstranded intervals and rounded down for negative strand
    intervals.

    If `use_strand` is True and the interval is on the negative strand, the
    center is calculated differently to maintain consistency when stacking
    sequences from different intervals oriented in the forward strand direction.
    This ensures that the relative distance between the interval's upstream
    boundary and its center is preserved.

    Args:
      use_strand: If True, the strand of the interval is considered when
        calculating the center.

    Returns:
      The integer representing the center position of the interval.

    Examples:
      >>> Interval('1', 1, 3, '+').center()
      2
      >>> Interval('1', 1, 3, '-').center()  # Strand doesn't matter.
      2
      >>> Interval('1', 1, 4, '+').center()
      3
      >>> Interval('1', 1, 4, '-').center()
      2
      >>> Interval('1', 1, 4, '-').center()
      2
      >>> Interval('1', 1, 4, '+').center(use_strand=False)
      3
      >>> Interval('1', 1, 2, '-').center()
      1
    """
    center = (self.start + self.end) // 2
    if use_strand and self.negative_strand:
      return center
    else:
      return center + self.width % 2

  def shift(self, offset: int, use_strand: bool = True) -> Self:
    """Shifts the interval by the given offset.

    Args:
      offset: The amount to shift the interval.
      use_strand: If True, the shift direction is reversed for negative strand
        intervals.

    Returns:
      A new shifted interval.
    """
    obj = self.copy()
    if use_strand and self.negative_strand:
      offset = -offset
    obj.start = self.start + offset
    obj.end = self.end + offset
    return obj

  def boundary_shift(
      self, start_offset: int = 0, end_offset: int = 0, use_strand: bool = True
  ) -> Self:
    """Extends or shrinks the interval by adjusting the positions with padding.

    Args:
      start_offset: The amount to shift the start position.
      end_offset: The amount to shift the end position.
      use_strand: If True, the offsets are applied in reverse for negative
        strand intervals.

    Returns:
      A new interval with adjusted boundaries.
    """
    return self.pad(-start_offset, end_offset, use_strand=use_strand)

  def pad(
      self, start_pad: int, end_pad: int, *, use_strand: bool = True
  ) -> Self:
    """Pads the interval by adding the specified padding to the start and end.

    Args:
      start_pad: The amount of padding to add to the start.
      end_pad: The amount of padding to add to the end.
      use_strand: If True, padding is applied in reverse for negative strand
        intervals.

    Returns:
      A new padded interval.
    """
    obj = self.copy()
    obj.pad_inplace(start_pad, end_pad, use_strand=use_strand)
    return obj

  def pad_inplace(
      self, start_pad: int, end_pad: int, *, use_strand: bool = True
  ):
    """Pads the interval in place by adding padding to the start and end.

    Args:
      start_pad: The amount of padding to add to the start.
      end_pad: The amount of padding to add to the end.
      use_strand: If True, padding is applied in reverse for negative strand
        intervals.
    """
    if use_strand and self.strand == '-':
      start_pad, end_pad = end_pad, start_pad
    self.start -= start_pad
    self.end += end_pad
    if self.width < 0:
      raise ValueError('Resulting interval has negative length')

  def resize(self, width: int, use_strand: bool = True) -> Self:
    """Resizes the interval to a new width, centered around the original center.

    Args:
      width: The new width of the interval.
      use_strand: If True, resizing considers the strand orientation.

    Returns:
      A new resized interval.
    """
    obj = self.copy()
    obj.resize_inplace(width, use_strand)
    return obj

  def resize_inplace(self, width: int, use_strand: bool = True) -> None:
    """Resizes the interval in place, centered around the original center.

    Args:
      width: The new width of the interval.
      use_strand: If True, resizing considers the strand orientation.
    """
    if width < 0:
      raise ValueError(f'Width needs to be > 0. Found: {width}.')

    if width is None or self.width == width:
      return

    center = self.center()
    if use_strand and self.negative_strand:
      self.start = center - width // 2
      self.end = center + (width + 1) // 2
    else:
      self.start = center - (width + 1) // 2
      self.end = center + width // 2
    assert self.width == width

  def overlaps(self, interval: Self) -> bool:
    """Checks if this interval overlaps with another interval."""
    return (
        self.chromosome == interval.chromosome
        and self.start < interval.end
        and interval.start < self.end
    )

  def contains(self, interval: Self) -> bool:
    """Checks if this interval completely contains another interval."""
    return (
        self.chromosome == interval.chromosome
        and self.start <= interval.start
        and self.end >= interval.end
    )

  def intersect(self, interval: Self) -> Self | None:
    """Returns the intersection of this interval with another interval."""
    output = self.copy()
    if not self.overlaps(interval):
      return None
    output.start = max(self.start, interval.start)
    output.end = min(self.end, interval.end)
    return output

  def coverage(
      self, intervals: Sequence[Self], *, bin_size: int = 1
  ) -> np.ndarray:
    """Computes coverage track from sequence of intervals overlapping interval.

    This method calculates the coverage of this interval by a set of other
    intervals. The coverage is defined as the number of intervals that overlap
    each position within this interval.

    The `bin_size` parameter allows you to bin the coverage into equal-sized
    windows. This can be useful for summarizing coverage over larger regions.
    If `bin_size` is 1, the coverage is calculated at single-base resolution.

    Args:
      intervals: A sequence of `Interval` objects that may overlap this
        interval.
      bin_size: The size of the bins used to calculate coverage. Must be a
        positive integer that divides the width of the interval.

    Returns:
      A 1D numpy array representing the coverage track. The length of the array
      is `self.width // bin_size`. Each element in the array represents the
      summed coverage within the corresponding bin.

    Raises:
      ValueError: If `bin_size` is not a positive integer or if the interval
        width is not divisible by `bin_size`.
    """
    if bin_size <= 0:
      raise ValueError('bin_size needs to be larger or equal to 1.')
    if self.width % bin_size != 0:
      raise ValueError(
          f'interval width {self.width} needs to be divisible '
          f'by bin_size {bin_size}.'
      )
    output = np.zeros((self.width,), dtype=np.int32)
    for interval in intervals:
      if not self.overlaps(interval):
        continue
      relative_start = max(interval.start - self.start, 0)
      relative_end = min(interval.end - self.start, self.width)
      output[relative_start:relative_end] += 1
    if bin_size > 1:
      return output.reshape((self.width // bin_size, bin_size)).sum(axis=-1)
    else:
      return output

  def overlap_ranges(
      self,
      intervals: Sequence[Self],
  ) -> np.ndarray:
    """Returns overlapping ranges from intervals overlapping this interval.

    Args:
      intervals: Sequence of candidate intervals to test for overlap.

    Returns:
      2D numpy array indicating the start and end of the overlapping ranges.
    """
    output = []
    for interval in intervals:
      if not self.overlaps(interval):
        continue
      relative_start = max(interval.start - self.start, 0)
      relative_end = min(interval.end - self.start, self.width)
      output.append([relative_start, relative_end])

    return (
        np.asarray(output, dtype=np.int32)
        if output
        else np.empty((0, 2), dtype=np.int32)
    )

  def binary_mask(
      self, intervals: Sequence[Self], bin_size: int = 1
  ) -> np.ndarray:
    """Boolean mask True if any interval overlaps the bin: coverage > 0."""
    return self.coverage(intervals, bin_size=bin_size) > 0

  def coverage_stranded(
      self, intervals: Sequence[Self], *, bin_size: int = 1
  ) -> np.ndarray:
    """Computes a coverage track from intervals overlapping this interval.

    This method considers the strand information of both self and intervals.

    Args:
      intervals: Sequence of intervals possibly overlapping self.
      bin_size: Resolution at which to bin the output coverage track. Coverage
        within each bin (if larger than 1) will be summarized using sum().

    Returns:
      Numpy array of shape (self.width // bin_size, 2) where output[:, 0]
      represents coverage for intervals on the same strand as self and
      output[:, 1] represents coverage of intervals on the opposite strand.
    """
    # Split intervals based on strand.
    forward_intervals = []
    reverse_intervals = []
    for interval in intervals:
      if interval.negative_strand:
        reverse_intervals.append(interval)
      else:
        forward_intervals.append(interval)

    coverage = np.stack(
        [
            self.coverage(forward_intervals, bin_size=bin_size),
            self.coverage(reverse_intervals, bin_size=bin_size),
        ],
        axis=-1,
    )
    if self.negative_strand:
      return coverage[::-1, ::-1]
    else:
      return coverage

  def binary_mask_stranded(
      self, intervals: Sequence[Self], bin_size: int = 1
  ) -> np.ndarray:
    """Boolean mask True if any interval overlaps the bin: coverage > 0."""
    return self.coverage_stranded(intervals, bin_size=bin_size) > 0


_DEFAULT_REGEX = re.compile(r'(chr(?:X|Y|M|\d+)):(\d+):([ACGTN]*)>([ACGTN]*)')
_GTEX_REGEX = re.compile(
    r'(chr(?:X|Y|M|\d+))_(\d+)_([ACGTN]*)_([ACGTN]*)_?[a-zA-Z0-9]*'
)
_OPEN_TARGETS_REGEX = re.compile(r'((?:X|Y|M|\d+))_(\d+)_([ACGTN]*)_([ACGTN]*)')
_OPEN_TARGETS_BIGQUERY_REGEX = re.compile(
    r'((?:X|Y|M|\d+)):(\d+):([ACGTN]*):([ACGTN]*)'
)
_GNOMAD_REGEX = re.compile(r'((?:X|Y|M|\d+))-(\d+)-([ACGTN]*)-([ACGTN]*)')


class VariantFormat(enum.Enum):
  """A format for parsing a string into a Variant object.

  This is used to convert from a string to a formal Variant object.
  Note that it does not perform any validation (e.g. it does not verify that the
  reference allele corresponds to the reference genome or that the
  position is valid).

  Example formats:
    DEFAULT: chr22:1024:A>C
    GTEX: chr22_1024_A_C_b38 (build suffix is optional)
    OPEN_TARGETS: 22_1024_A_C (chr prefix is omitted)
    OPEN_TARGETS_BIGQUERY: 22:1024:A:C (chr prefix is omitted)
    GNOMAD: 22-1024-A-C (chr prefix is omitted)
  """

  DEFAULT = 'default'
  GTEX = 'gtex'
  OPEN_TARGETS = 'open_targets'
  OPEN_TARGETS_BIGQUERY = 'open_targets_bigquery'
  GNOMAD = 'gnomad'

  def to_regex(self) -> re.Pattern[str]:
    """Returns a regular expression for the variant format."""
    match self:
      case VariantFormat.DEFAULT:
        return _DEFAULT_REGEX
      case VariantFormat.GTEX:
        return _GTEX_REGEX
      case VariantFormat.OPEN_TARGETS:
        return _OPEN_TARGETS_REGEX
      case VariantFormat.OPEN_TARGETS_BIGQUERY:
        return _OPEN_TARGETS_BIGQUERY_REGEX
      case VariantFormat.GNOMAD:
        return _GNOMAD_REGEX


@dataclasses.dataclass
class Variant:
  """Represents a genomic variant/mutation.

  Differs from the Variant definition in a VCF file, which allows
  for multiple alternative bases and contains sample information. This
  `Variant` class does not include sample information or variant call
  quality information.

  Attributes:
    chromosome: The chromosome name (e.g., 'chr1', '1').
    position: The 1-based position of the variant on the chromosome.
    reference_bases: The reference base(s) at the variant position. Most
      frequently (not always!), these correspond to the sequence in the
      reference genome at positions: [position, ..., position +
      len(reference_bases) - 1]
    alternate_bases: The alternate base(s) that replace the reference. For
      example, if sequence='ACT', position=2, reference_bases='C',
      alternate_bases='TG', then the actual (alternate) sequence would be ATGT.
    name: An optional name for the variant (e.g., a dbSNP ID like rs206437).
    info: An optional dictionary for additional variant information.
  """

  chromosome: str
  position: int
  reference_bases: str
  alternate_bases: str
  name: str = dataclasses.field(default='', compare=False, hash=False)
  info: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False, hash=False
  )

  def __post_init__(self):
    """Validates the variant's position."""
    if self.position < 1:
      raise ValueError(f'Position has to be >=1. Found: {self.position}.')
    if not set(self.reference_bases).issubset(VALID_VARIANT_BASES):
      raise ValueError(
          f'Invalid reference bases: "{self.reference_bases}". Must only'
          ' contain "ACGTN".'
      )
    if not set(self.alternate_bases).issubset(VALID_VARIANT_BASES):
      raise ValueError(
          f'Invalid alternate bases: "{self.alternate_bases}". Must only'
          ' contain "ACGTN".'
      )

  def __str__(self):
    """Returns a string representation of the variant."""
    ref_alt = f'{self.reference_bases}>{self.alternate_bases}'
    return f'{self.chromosome}:{self.position}:{ref_alt}'

  def as_truncated_str(self, max_length: int = 50):
    """Truncates the variant str's ref and alt bases to the given max length."""

    def _truncate(s: str):
      if len(s) <= max_length:
        return s
      else:
        return s[: max_length // 2] + '...' + s[-max_length // 2 :]

    return (
        f'{self.chromosome}:{self.position}:'
        f'{_truncate(self.reference_bases)}>{_truncate(self.alternate_bases)}'
    )

  @property
  def start(self) -> int:
    """Returns the 0-based start position of the variant."""
    return self.position - 1

  @property
  def end(self) -> int:
    """Returns the 0-based end position of the variant."""
    return self.start + len(self.reference_bases)

  @property
  def reference_interval(self) -> Interval:
    """Returns an `Interval` for the variant's reference sequence."""
    return Interval(self.chromosome, self.start, self.end)

  def reference_overlaps(self, interval: Interval) -> bool:
    """Checks if the variant's reference overlaps with the interval."""
    return interval.overlaps(Interval(self.chromosome, self.start, self.end))

  def alternate_overlaps(self, interval: Interval) -> bool:
    """Checks if the variant's alternate overlaps with the interval."""
    return interval.overlaps(
        Interval(
            self.chromosome, self.start, self.start + len(self.alternate_bases)
        )
    )

  @property
  def is_snv(self) -> bool:
    """Return if the variant is a Single Nucleotide Variant (SNV)."""
    return len(self.reference_bases) == 1 and len(self.alternate_bases) == 1

  @property
  def is_deletion(self) -> bool:
    """Return if the variant is a deletion."""
    return len(self.reference_bases) > len(self.alternate_bases)

  @property
  def is_insertion(self) -> bool:
    """Return if the variant is an insertion."""
    return len(self.reference_bases) < len(self.alternate_bases)

  @property
  def is_frameshift(self) -> bool:
    """Return if the variant is a frameshift."""
    indel_size = abs(len(self.reference_bases) - len(self.alternate_bases))
    return indel_size > 0 and indel_size % 3 != 0

  @property
  def is_indel(self) -> bool:
    """Return if the variant is an insertion or deletion."""
    return self.is_insertion or self.is_deletion

  @property
  def is_structural(self) -> bool:
    """Return if the variant is a structural variant."""
    indel_size = abs(len(self.reference_bases) - len(self.alternate_bases))
    return indel_size >= 50

  def copy(self) -> Self:
    """Returns a deep copy of the variant."""
    return copy.deepcopy(self)

  @classmethod
  def from_str(
      cls, string: str, variant_format: VariantFormat = VariantFormat.DEFAULT
  ) -> Self:
    """Creates a `Variant` from a string representation.

    Args:
      string: The string representation.
      variant_format: The format of the variant string. By default, this uses
        "chromosome:position:ref>alt" (for example, "chr1:1024:A>C"). See
        VariantFormat for alternate formatting options.

    Returns:
      A `Variant` object.
    """
    result = re.fullmatch(variant_format.to_regex(), string)
    if result is None:
      raise ValueError(f'Invalid format for variant string: {string}')

    chromosome, position, reference, alternate = result.groups()
    # Add chr prefix if not already present.
    if not chromosome.startswith('chr'):
      chromosome = f'chr{chromosome}'
    return cls(
        chromosome=chromosome,
        position=int(position),
        reference_bases=reference,
        alternate_bases=alternate,
    )

  def to_dict(self) -> dict[str, Any]:
    """Converts the variant to a dictionary."""
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, dictionary: Mapping[str, Any] | Self) -> Self:
    """Creates a `Variant` from a dictionary."""
    return cls(**dictionary)  # pytype: disable=bad-return-type

  def to_proto(self) -> dna_model_pb2.Variant:
    """Converts the variant to a protobuf message."""
    return dna_model_pb2.Variant(
        chromosome=self.chromosome,
        position=self.position,
        reference_bases=self.reference_bases,
        alternate_bases=self.alternate_bases,
    )

  @classmethod
  def from_proto(cls, variant: dna_model_pb2.Variant) -> Self:
    """Creates a `Variant` from a protobuf message."""
    return cls(
        chromosome=variant.chromosome,
        position=variant.position,
        reference_bases=variant.reference_bases,
        alternate_bases=variant.alternate_bases,
    )

  def split(self, anchor: int) -> tuple[Self | None, Self | None]:
    """Splits the variant into two at the anchor point.

    If the anchor point falls within the variant's reference sequence, the
    variant is split into two new variants: one upstream of the anchor and one
    downstream. If the anchor is outside the variant's reference sequence,
    the original variant is returned on the appropriate side, and None on the
    other.

    Example:
      position=       3
      ref:       ...[ A C ]...
      alt:       .....T G T C
      anchor=3         |
      returns:    (chr1:3:A>T, chr1:4:C>GTC)

    Args:
      anchor: The 0-based anchor point to split the variant.

    Returns:
      A tuple of the upstream and downstream variants. If the variant is only
      on one side of the anchorpoint, then None is returned.
    """
    if anchor <= self.start:
      return None, self.copy()
    elif anchor >= self.end:
      return self.copy(), None
    else:
      mid = anchor - self.start
      upstream, downstream = self.copy(), self.copy()
      upstream.reference_bases = self.reference_bases[:mid]
      upstream.alternate_bases = self.alternate_bases[:mid]

      downstream.position = anchor + 1
      downstream.reference_bases = self.reference_bases[mid:]
      downstream.alternate_bases = self.alternate_bases[mid:]
      return upstream, downstream


@dataclasses.dataclass
class Junction(Interval):
  """Represents a splice junction.

  A splice junction is a point in a pre-mRNA transcript where an intron is
  removed and exons are joined during RNA splicing. This class inherits from
  `Interval` and adds properties and methods specific to splice junctions.

  Attributes:
    chromosome: The chromosome name.
    start: The 0-based start position of the junction.
    end: The 0-based end position of the junction.
    strand: The strand of the junction ('+' or '-').
    name: An optional name for the junction.
    info: An optional dictionary to store additional information.
    k: An optional integer representing the number of reads supporting the
      splice junction.

  Raises:
    ValueError: If the strand is unstranded.
  """

  k: int | None = None

  def __post_init__(self):
    """Validates that the junction is stranded."""
    super().__post_init__()
    if self.strand == STRAND_UNSTRANDED:
      raise ValueError('Junctions must be stranded.')

  @property
  def acceptor(self) -> int:
    """Returns the acceptor site position."""
    return self.start if self.strand == STRAND_NEGATIVE else self.end

  @property
  def donor(self) -> int:
    """Returns the donor site position."""
    return self.end if self.strand == STRAND_NEGATIVE else self.start

  def dinucleotide_region(self) -> tuple[Interval, Interval]:
    """Returns the dinucleotide regions around acceptor and donor sites."""
    return (
        Interval(
            self.chromosome, self.start, self.start + 2, strand=self.strand
        ),
        Interval(self.chromosome, self.end - 2, self.end, strand=self.strand),
    )

  def acceptor_region(self, overhang: tuple[int, int] = (250, 250)) -> Interval:
    """Returns the region around the acceptor site with overhang."""
    return Interval(
        self.chromosome, self.acceptor, self.acceptor, strand=self.strand
    ).pad(start_pad=overhang[0], end_pad=overhang[1])

  def donor_region(self, overhang: tuple[int, int] = (250, 250)) -> Interval:
    """Returns the region around the donor site with overhang."""
    return Interval(
        self.chromosome, self.donor, self.donor, strand=self.strand
    ).pad(start_pad=overhang[0], end_pad=overhang[1])


class _FastaExtractorType(Protocol):
  """Protocol definition for extracting intervals from a Fasta file."""

  def extract(self, interval: Interval) -> str:
    """Extract and return the DNA sequence for a given interval."""


def _prefix_length(*sequences) -> int:
  """Returns the length of the common prefix for a sequence of strings."""
  i = 0
  for chars in zip(*sequences, strict=False):
    if all(c == chars[0] for c in chars):
      i += 1
    else:
      break
  return i


def normalize_variant(
    variant: Variant, extractor: _FastaExtractorType
) -> Variant:
  """Normalize a Variant by left-aligning the reference and alternate bases.

  Normalization applied following algorithm described in
  https://doi.org/10.1093/bioinformatics/btv112.

  Args:
    variant: The Variant to normalize.
    extractor: The FastaExtractor to use for extracting the reference sequence.

  Returns:
    The normalized Variant.
  """
  if variant.is_snv:
    return variant

  chromosome = variant.chromosome
  genome_sequence = extractor.extract(
      Interval(
          chromosome,
          variant.start,
          variant.start
          + max(len(variant.reference_bases), len(variant.alternate_bases)),
      )
  )

  position = variant.position
  alleles = [genome_sequence, variant.reference_bases, variant.alternate_bases]

  # Remove any common suffix from the alleles.
  finished = False
  while not finished:
    suffix_length = _prefix_length(*[reversed(a) for a in alleles])
    if suffix_length > 0:
      alleles = [a[:-suffix_length] for a in alleles]

    # If any alleles are empty, extend all alleles by 1 nucleotide to the left.
    if not all(alleles):
      position -= 1
      base = extractor.extract(Interval(chromosome, position - 1, position))[0]
      alleles = [base + a for a in alleles]
    else:
      # Finished iff we haven't shifted alleles and don't have a common suffix.
      finished = suffix_length == 0

  # Left-align the variant to the genome, ensuring at least 1 bp of context.
  max_prefix_length = len(min(alleles)) - 1
  prefix_length = _prefix_length(*alleles)
  i = min(max_prefix_length, prefix_length)

  _, reference_bases, alternate_bases = alleles
  return Variant(
      chromosome=chromosome,
      position=position + i,
      reference_bases=reference_bases[i:],
      alternate_bases=alternate_bases[i:],
  )


def _split_intervals(
    intervals: Iterable[Interval], marker: int, bounds: list[tuple[int, int]]
):
  """Splits intervals into start and end points with markers."""
  for i in intervals:
    bounds.append((i.start, +marker))
    bounds.append((i.end, -marker))


def _group_by_chromosome(
    intervals: Iterable[Interval],
) -> dict[str, list[Interval]]:
  """Groups intervals by chromosome."""
  interval_map = collections.defaultdict(list)
  for i in intervals:
    interval_map[i.chromosome].append(i)
  return dict(interval_map)


def intersect_intervals(
    lhs: Iterable[Interval],
    rhs: Iterable[Interval],
    *,
    result_strand: str = '.',
) -> Iterator[Interval]:
  """Generates the intersection of two interval sets.

  Point ranges (width == 0) are considered to intersect any range which contains
  them.

  In these examples, the intersection is a point range (2,2)
  ```
      1  2  3  4
    ..|..|..|..|..
          <>        (start=2, end=2)
    ------->        (..., end=2)
  ```

  ```
      1  2  3  4
    ..|..|..|..|..
          <>        (start=2, end=2)
          <-------  (start=2, end=...)
  ```

  For consistency, this means that the following results in a point
  intersection:

  ```
      1  2  3  4
    ..|..|..|..|..
    ------->        (start=..., end=2)
          <-------  (start=2, end=...)
  ```

  Args:
    lhs: A set of intervals.
    rhs: A set of intervals.
    result_strand: The strand for the resulting intervals.

  Yields:
    The intersection of intervals. Overlapping intervals within either `lhs`
      or `rhs` are implicitly unioned.
  """

  def _intersect(lhs, rhs, chrom):
    """Calculates the intersection for a specific chromosome.

    Deconstructs two sets of intervals into (start, +k) and (end, -k) positions,
    where k is different for each set, then sorts all interval endpoints. By
    walking the sorted endpoints and accumulating the k's, we can work out when
    we enter or exit an interval from either set.

    This allows emitting intervals corresponding to the intersection, and with
    appropriate k's, detecting when we enter an interval in either set before
    exiting the previous.

    Args:
      lhs: The first list of `Interval` objects.
      rhs: The second list of `Interval` objects.
      chrom: The chromosome name.

    Yields:
      `Interval` objects representing the intersections.
    """
    bounds = []
    _split_intervals(lhs, 0x00001, bounds)
    _split_intervals(rhs, 0x10000, bounds)
    accum = 0
    start = None
    for pos, delta in sorted(bounds, key=lambda x: (x[0], -x[1])):
      old_accum = accum
      accum += delta
      if accum & 0xFFFF and accum & 0xFFFF0000:
        if start is None:
          start = pos
      elif old_accum & 0xFFFF and old_accum & 0xFFFF0000:
        yield Interval(chrom, start, pos, result_strand)
        start = None

  lhs = _group_by_chromosome(lhs)
  rhs = _group_by_chromosome(rhs)
  for chromosome in set(lhs) & set(rhs):
    yield from _intersect(lhs[chromosome], rhs[chromosome], chromosome)


def union_intervals(
    lhs: Iterable[Interval],
    rhs: Iterable[Interval],
    *,
    result_strand: str = '.',
) -> Iterator[Interval]:
  """Generates the union of two interval sets.

  Args:
    lhs: A non-overlapping set of intervals.
    rhs: A non-overlapping set of intervals.
    result_strand: The strand for the resulting intervals.

  Yields:
    The union of intervals. Any position covered by an interval in `lhs` or
    `rhs` is covered in the result.
  """

  def _union(lhs, rhs, chrom):
    """Calculates the union for a specific chromosome, analog of _intersect."""
    bounds = []
    _split_intervals(lhs, 0x00001, bounds)
    _split_intervals(rhs, 0x10000, bounds)
    accum = 0
    start = end = None
    for pos, delta in sorted(bounds, key=lambda x: (x[0], -x[1])):
      if accum == 0 and end is not None and end < pos:
        # Delay generating an interval until after the observed end point.
        # This merges abutting ranges.
        yield Interval(chrom, start, end, result_strand)
        start = end = None
      accum += delta
      if accum:
        if start is None:
          start = pos
      else:
        assert start is not None
        end = pos
    if end is not None:
      yield Interval(chrom, start, end, result_strand)

  lhs = _group_by_chromosome(lhs)
  rhs = _group_by_chromosome(rhs)
  for chromosome in set(lhs) | set(rhs):
    yield from _union(
        lhs.get(chromosome, []), rhs.get(chromosome, []), chromosome
    )


def merge_overlapping_intervals(
    intervals: Sequence[Interval],
) -> list[Interval]:
  """Merges overlapping intervals and returns a sorted list.

  Args:
    intervals: A sequence of intervals with the same strand.

  Returns:
    A new sorted list of merged intervals.
  """
  if not intervals:
    return []
  assert all(i.strand == intervals[0].strand for i in intervals)
  return list(union_intervals(intervals, [], result_strand=intervals[0].strand))
