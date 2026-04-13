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

"""Utilities for working with transcripts."""

import collections
import copy
import dataclasses
import functools
import sys
from typing import Any

import pandas as pd

from alphagenome.data import genome

MITOCHONDRIAL_CHROMS = ['M', 'chrM', 'MT']


@dataclasses.dataclass(frozen=True)
class Transcript:
  """Represents transcript object containing attributes from a GTF file.

  A transcript is a region of DNA that encodes a single RNA molecule. The
  Transcript dataclass contains attributes that describe the structure and
  content of a transcript, namely:

  Attributes:
    exons: A list of `genome.Interval`s representing exons within transcript.
      Each `Transcript` must contain exons.
    cds: An optional list of `genome.Interval`s representing coding sequences
      (CDS) within a transcript. CDS include start codon and exclude stop codon.
    start_codon: An optional list of `genome.Interval`s representing a single
      start codon. Start codons can be split by introns, therefore might have
      more than one genomic interval. Some coding transcripts are missing start
      codons, e.g., ENST00000455638.6.
    stop_codon: An optional list of `genome.Interval`s representing a single
      stop codon. Stop codon can be split by introns, therefore might have more
      than one genomic interval. Some transcripts coding transcripts are missing
      stop codons, e.g., ENST00000574051.5.
    transcript_id: An optional string representing a transcript id.
    gene_id: An optional string representing a gene id.
    protein_id: An optional string representing a protein id which is encoded by
      the transcript.
    uniprot_id: An optional UniprotKB-AC id string.
    info: a dictionary of additional information on a transcript.
    chromosome: chromosome name on which the transcript is present. Must be the
      same for all genomic intervals within a transcript.
    is_mitochondrial: whether the transcript is on the mitochondria chromosome.
    strand_int: strand on which transcript is present as an int. -1 for negative
      strand +1 for positive strand
    strand: strand (positive or negative) on which transcript is present. Must
      be the same for all genomic intervals within a transcript.
    is_negative_strand: a boolean value indicating whether transcript is on
      negative strand.
    is_positive_strand: a boolean value indicating whether transcript is on
      positive strand.
    transcript_interval: a genomic interval of a transcript.
    selenocysteines: a list of intervals where selenocysteines are present
      within a transcript.
    selenocysteine_pos_in_protein: a list of 0-based positions of
      selenocysteines in protein encoded by the transcript.
    is_coding: a value indicating whether a `Transcript` contains coding
      sequences (CDS) or not.
    cds_including_stop_codon: a list of CDS and stop_codon intervals with
      overlapping intervals merged.
    utr5: A list of genomic intervals representing 5' untranslated region. 5'
      UTR doesn't include start codon. There may be no 5' UTR present in the
      transcript or UTRs can be split by introns.
    utr3: A list of genomic intervals representing 3' untranslated region. 3'
      UTR doesn't include stop codon. There may be no 3' UTR present in the
      transcript or UTRs can be split by introns.
    splice_regions: a list of splice regions within a transcript.
    splice_donor_sites: a list of splice donor sites. Commonly, the RNA sequence
      that is removed begins with the dinucleotide GU at its 5′ end.
    splice_acceptor_sites: a list of splice acceptor sites. Commonly, the RNA
      sequence that is removed ends with AG at its 3′ end.
    splice_donors: a list of splice donors. The first nucleotide of the intron
      (0-based).
    splice_acceptors: a list of splice acceptors. The last nucleotide of the
      intron (0-based).
  """

  exons: list[genome.Interval]
  cds: list[genome.Interval] | None = None
  start_codon: list[genome.Interval] | None = None
  stop_codon: list[genome.Interval] | None = None
  transcript_id: str | None = dataclasses.field(compare=False, default=None)
  gene_id: str | None = dataclasses.field(compare=False, default=None)
  protein_id: str | None = dataclasses.field(compare=False, default=None)
  uniprot_id: str | None = dataclasses.field(compare=False, default=None)
  info: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False, hash=False
  )

  def offset_in_cds(self, genome_position: int) -> int | None:
    """Return the offset within the set of CDS exons of `genome_position`.

    Args:
      genome_position: A coordinate presumed to be on the same chromosome as
        this transcript.

    Returns:
      The offset of `genome_position` from the start of the CDS, accounting for
      strand, or None, if `genome_position` does not overlap the CDS.
    """
    offset = 0
    for cds_exon in self.cds_including_stop_codon[:: self.strand_int]:
      if cds_exon.start <= genome_position < cds_exon.end:
        if self.is_positive_strand:
          return offset + genome_position - cds_exon.start
        else:
          return offset + cds_exon.end - genome_position - 1
      offset += cds_exon.width
    return None

  @property
  def chromosome(self) -> str:
    """Gets the chromosome name on which the transcript is present.

    Returns:
      The chromosome name.
    """
    return self.exons[0].chromosome

  @property
  def is_mitochondrial(self) -> bool:
    """Gets whether the transcript is on the mitochondria chromosome.

    Returns:
      True if the transcript is on the mitochondria chromosome, False otherwise.
    """
    return self.chromosome in MITOCHONDRIAL_CHROMS

  # TODO: b/376466056 - Unify strand representations.
  @property
  def strand_int(self) -> int:
    """Gets the strand as an integer.

    Returns:
      -1 for negative strand, +1 for positive strand, 0 for unknown.
    """
    return {genome.STRAND_NEGATIVE: -1, genome.STRAND_POSITIVE: +1}.get(
        self.strand, 0
    )

  @property
  def strand(self) -> str:
    """Gets the strand on which the transcript is present.

    Returns:
      The strand (positive or negative).
    """
    return self.exons[0].strand

  @property
  def is_positive_strand(self) -> bool:
    return self.strand == genome.STRAND_POSITIVE

  @property
  def is_negative_strand(self) -> bool:
    return self.strand == genome.STRAND_NEGATIVE

  @functools.cached_property
  def transcript_interval(self) -> genome.Interval:
    """Gets a genomic interval of a transcript.

    Returns:
      A genomic interval of a transcript where transcript start is equal to the
      first exon start and transcript end is equal to the last exon end.
    """
    return genome.Interval(
        self.chromosome,
        self.exons[0].start,
        self.exons[-1].end,
        strand=self.strand,
    )

  @functools.cached_property
  def selenocysteines(self) -> list[genome.Interval]:
    if 'selenocysteines' not in self.info:
      return []
    return self.info['selenocysteines']

  @functools.cached_property
  def selenocysteine_pos_in_protein(self) -> list[int]:
    # 0-based
    selenocystein_pos = []
    for selenocysteine in self.selenocysteines:
      if selenocysteine.info['cds_offset'] is None:
        raise ValueError(
            'Transcript cannot be translated due to '
            'bad input data of selenocysteines.'
        )
      selenocystein_pos.append(selenocysteine.info['cds_offset'] // 3)
    return selenocystein_pos

  @functools.cached_property
  def introns(self) -> list[genome.Junction]:
    """Get a list of intron intervals.

    Returns:
      A list of genomic intervals representing introns, where a single intron
      junction is an interval spanning between two adjacent exons.
    """
    intron_intervals = []
    for i in range(1, len(self.exons)):
      intron_intervals.append(
          genome.Junction(
              self.chromosome,
              self.exons[i - 1].end,
              self.exons[i].start,
              strand=self.strand,
          )
      )
    return intron_intervals

  @functools.cached_property
  def is_coding(self) -> bool:
    return bool(self.cds)

  @functools.cached_property
  def cds_including_stop_codon(self) -> list[genome.Interval]:
    """Obtains coding sequences including stop codon.

    By default gtf files exclude stop codons from CDS while gff include stop
    codons within coding sequences.
    """
    if not self.is_coding:
      return []
    return genome.merge_overlapping_intervals(
        self.cds + (self.stop_codon or [])
    )

  @functools.cached_property
  def utr5(self) -> list[genome.Interval]:
    return self._get_utr(self.strand != genome.STRAND_NEGATIVE)

  @functools.cached_property
  def utr3(self) -> list[genome.Interval]:
    return self._get_utr(self.strand == genome.STRAND_NEGATIVE)

  def _get_utr(self, before: bool) -> list[genome.Interval]:
    """Gets the UTRs located before/after first/last coding sequence."""
    utrs = []
    if not self.cds:
      return utrs

    merged_cds_stop = genome.merge_overlapping_intervals(
        self.cds + (self.stop_codon or [])
    )

    if before:
      start, end = 0, merged_cds_stop[0].start
    else:
      start, end = merged_cds_stop[-1].end, sys.maxsize

    valid_interval = genome.Interval(
        self.chromosome, start, end, strand=self.strand
    )

    for exon in filter(lambda x: x.overlaps(valid_interval), self.exons):
      intersect = valid_interval.intersect(exon)
      if intersect:
        utrs.append(intersect)
    return utrs

  # TODO: b/376465275 - deal with cases where exon shorter than 3 bp length
  # TODO: b/376465275 - deal with cases where intron is shorther than 4 bp
  @functools.cached_property
  def splice_regions(self) -> list[genome.Interval]:
    """Obtains and returns splice regions of a transcript.

    splice region (SO:0001630) is "within 1-3 bases of the exon or 3-8 bases of
    the intron.
    """
    if not self.introns:
      return []
    splice_regions = []

    for intron, prev_exon, next_exon in zip(
        self.introns, self.exons[:-1], self.exons[1:]
    ):
      if prev_exon.width > 2:
        splice_regions.append(
            genome.Interval(
                self.chromosome,
                intron.start - 3,
                intron.start,
                strand=self.strand,
            )
        )

      if next_exon.width > 2:
        splice_regions.append(
            genome.Interval(
                self.chromosome, intron.end, intron.end + 3, strand=self.strand
            )
        )

      if intron.width > 4:
        splice_regions.append(
            genome.Interval(
                self.chromosome,
                intron.start + 2,
                min(intron.start + 8, intron.end - 2),
                strand=self.strand,
            )
        )
        splice_regions.append(
            genome.Interval(
                self.chromosome,
                max(intron.end - 8, intron.start + 2),
                intron.end - 2,
                strand=self.strand,
            )
        )
    return genome.merge_overlapping_intervals(splice_regions)

  @functools.cached_property
  def splice_donor_sites(self) -> list[genome.Interval]:
    return self._get_splice_sites(False, intron_overhang=2, exon_overhang=0)

  @functools.cached_property
  def splice_acceptor_sites(self) -> list[genome.Interval]:
    return self._get_splice_sites(True, intron_overhang=2, exon_overhang=0)

  @functools.cached_property
  def splice_donors(self) -> list[genome.Interval]:
    # To be consistent with the splice sites defined by intron start and end,
    # the overhang for donor and acceptor are different.
    return self._get_splice_sites(  # pytype: disable=bad-return-type  # enable-cached-property
        False, intron_overhang=1, exon_overhang=0
    )

  @functools.cached_property
  def splice_acceptors(self) -> list[genome.Interval]:
    return self._get_splice_sites(  # pytype: disable=bad-return-type  # enable-cached-property
        True, intron_overhang=0, exon_overhang=1
    )

  # TODO: b/376465275 - deal with cases where intron shorter than 4 bp length.
  def _get_splice_sites(
      self, acceptor: bool, intron_overhang: int, exon_overhang: int
  ) -> list[genome.Interval]:
    """Obtains splice acceptor/donor intervals.

    https://www.nature.com/scitable/topicpage/rna-splicing-introns-exons-and-spliceosome-12375/#:~:text=Introns%20are%20removed%20from%20primary,AG%20at%20its%203%E2%80%B2%20end
    Introns are removed from primary transcripts by cleavage at conserved
    sequences called splice sites. These sites are found at the 5′ and 3′
    ends of introns. Most commonly, the RNA sequence that is removed begins with
    the dinucleotide GU at its 5′ end, and ends with AG at its 3′ end.

    if - strand, the end two bases of intron are splice donor bases,
    if +, then the start two bases.

    Args:
      acceptor: value indicating whether splice acceptor or donor should be
        obtained.
      intron_overhang: bases into the intron.
      exon_overhang: bases into the exon.

    Returns:
      List of intervals of splice acceptor/donor sites.
    """
    if not self.introns:
      return []
    splice_sites = []
    for intron in self.introns:  # pylint:disable=not-an-iterable
      if intron.width < 4:
        continue
      if self.is_negative_strand != acceptor:
        splice = genome.Interval(
            self.chromosome,
            intron.end - intron_overhang,
            intron.end + exon_overhang,
            strand=self.strand,
        )
      else:
        splice = genome.Interval(
            self.chromosome,
            intron.start - exon_overhang,
            intron.start + intron_overhang,
            strand=self.strand,
        )
      splice_sites.append(splice)
    return splice_sites

  def __post_init__(self):
    if not self.exons:
      raise ValueError('Transcript must contain at least one exon.')

    for exon in self.exons:
      if exon.strand != self.strand or exon.chromosome != self.chromosome:
        raise ValueError(
            'Transcript intervals are inconsistent. All intervals of a '
            'transcript should have same strand and chromosome.'
        )

    if self.cds:
      # first exons can be part of UTR. Searching for the first coding exon.
      index = 0
      for exon in self.exons:
        # if overlaps, will check whether exon contains it in the latter loop.
        if exon.overlaps(self.cds[0]):
          break
        index += 1
      if index == len(self.exons) or len(self.cds) + index > len(self.exons):
        raise ValueError(
            'The number of coding exons must be the same as CDS '
            'and CDS cannot be outside of the exon intervals.'
        )

      # checks each subsequent exon is coding
      for seq, exon in zip(self.cds, self.exons[index : index + len(self.cds)]):
        if not exon.contains(seq):
          raise ValueError(
              'The number of coding exons must be the same as CDS '
              'and CDS cannot be outside of the exon intervals.'
          )
        if seq.strand != self.strand:
          raise ValueError(
              'Transcript intervals are inconsistent. All intervals of a '
              'transcript should have same strand and chromosome.'
          )

    for sc in self.selenocysteines:  # pylint:disable=not-an-iterable
      sc_pos = sc.end - 1 if sc.negative_strand else sc.start
      sc.info['cds_offset'] = self.offset_in_cds(sc_pos)

  def __len__(self):
    return self.transcript_interval.width

  @classmethod
  def from_gtf_df(
      cls,
      transcript_df: pd.DataFrame,
      ignore_info: bool = True,
      fix_truncation: bool = False,
  ) -> 'Transcript':
    """Initialises Trancript object from a given transcript dataframe.

    Args:
      transcript_df: Dataframe representing a transcript. The dataframe must
        contain a single transcript.
      ignore_info: If True, other columns in transcript_df won't be added to the
        info field, except transcript_type and selenocysteines.
      fix_truncation: Whether or not apply truncation fixation to CDS.

    Returns:
      Initialised Transcript object.
    Raises:
      ValueError: if the dataframe provided is invalid (no or more than one
      transcript, transcript has inconsistent strand or chromosome,
      transcript doesn't contain exons, CDS are not within exons, etc.)
    """
    if transcript_df.empty:
      raise ValueError('transcript_df is empty')
    if 'Feature' not in transcript_df:
      raise ValueError('transcript_df must contain Feature column.')

    if (
        'transcript_id' in transcript_df
        and len(transcript_df.transcript_id.unique()) > 1
    ):
      raise ValueError('transcript_df should only contain a single transcript.')

    # Convert rows to genome.Interval list.
    transcript_df = transcript_df.sort_values(by='Start')
    intervals_per_feature = collections.defaultdict(list)
    exon_row = None
    for _, row in transcript_df.iterrows():
      interval = genome.Interval.from_pyranges_dict(
          row, ignore_info=True
      )  # pytype: disable=wrong-arg-types  # pandas-drop-duplicates-overloads
      if row.Feature in ['CDS', 'stop_codon']:
        interval.info['frame'] = int(row.Frame)
      if exon_row is None and row.Feature == 'exon':
        exon_row = row
      intervals_per_feature[row.Feature].append(interval)

    if exon_row is None:
      raise ValueError('transcript_df must contain at least one exon')

    # Seed info.
    if ignore_info:
      info = {}
    else:
      skip = list(genome.PYRANGES_INTERVAL_COLUMNS) + [
          'Feature',
          'transcript_type',
          'Selenocysteines',
          'gene_type',
      ]
      info = {k: v for k, v in exon_row.items() if k not in skip}

    if 'transcript_type' in exon_row:
      info['transcript_type'] = exon_row['transcript_type']

    if 'Selenocysteine' in intervals_per_feature:
      info['selenocysteines'] = intervals_per_feature['Selenocysteine']

    if 'gene_type' in exon_row:
      info['gene_type'] = exon_row['gene_type']

    transcript_obj = cls(
        exons=intervals_per_feature['exon'],
        cds=intervals_per_feature.get('CDS', None),
        start_codon=intervals_per_feature.get('start_codon', None),
        stop_codon=intervals_per_feature.get('stop_codon', None),
        transcript_id=exon_row.get('transcript_id', None),
        gene_id=exon_row.get('gene_id', None),
        protein_id=exon_row.get('protein_id', None),
        uniprot_id=exon_row.get('uniprot_id', None),
        info=info,
    )
    if fix_truncation:
      return Transcript.fix_truncation(transcript_obj)
    return transcript_obj

  @classmethod
  def fix_truncation(cls, transcript: 'Transcript') -> 'Transcript':
    """Fixes CDS start and stop positions to be within coding frame.

    Args:
      transcript: a transcript to fix.

    Returns:
      New transcript with set start/stop codons and fixed CDS if the total
      length of CDS is > 6. Returns a copy of original transcript otherwise.
    """
    cds = sorted(
        (transcript.cds or []) + (transcript.stop_codon or []),
        key=lambda x: x.start,
    )
    cdna_len = sum(seq.width for seq in cds)
    if cdna_len < 7:
      return copy.deepcopy(transcript)

    positive_strand = transcript.is_positive_strand
    try:
      frame = cds[0 if positive_strand else -1].info['frame']
    except KeyError as key_error:
      raise KeyError(
          'CDS intervals are missing frame information,'
          ' truncations cannot be deduced.'
      ) from key_error
    frame_last = (cdna_len - frame) % 3

    cds, start_codon = cls._fix_coding_frame(
        five_prime=True,
        beginning=positive_strand,
        frame=frame,
        cds_transcript=cds,
    )

    cds, stop_codon = cls._fix_coding_frame(
        five_prime=False,
        beginning=not positive_strand,
        frame=frame_last,
        cds_transcript=cds,
    )

    return cls(
        exons=[exon.copy() for exon in transcript.exons],
        cds=cds,
        start_codon=start_codon,
        stop_codon=stop_codon,
        transcript_id=transcript.transcript_id,
        gene_id=transcript.gene_id,
        protein_id=transcript.protein_id,
        uniprot_id=transcript.uniprot_id,
        info={**transcript.info, 'truncation_fixed': True},
    )

  @classmethod
  def _fix_coding_frame(
      cls,
      five_prime: bool,
      beginning: bool,
      frame: int,
      cds_transcript: list[genome.Interval],
  ) -> tuple[list[genome.Interval], list[genome.Interval]]:
    """Fixes coding frame for a transcript and returns a new start/stop codon.

    Args:
      five_prime: a value indicating whether a 5' end to be fixed.
      beginning: indicates whether the beginning or the end of the transript to
        be fixed.
      frame: a current coding frame (0, 1, 2). For fixing 5' end, it is a
        position at which the first full codon starts within a cds. For fixing
        3' end, this indicates where the last full codon ends within a cds.
      cds_transcript: a list of coding sequence intervals.

    Returns:
      A pair of lists where the first item is a list of CDS with fixed coding
      frame and the second item is a start/stop codon.
    """

    def shorten_intervals(
        cds_transcript: list[genome.Interval],
        beginning: bool,
        frame: int,
    ) -> list[genome.Interval]:
      cds = [interval.copy() for interval in cds_transcript]
      index = 0 if beginning else -1
      while frame > 0:
        if cds[index].width > frame:
          if beginning:
            cds[index].start += frame
          else:
            cds[index].end -= frame
          frame = 0
        else:
          frame -= cds[index].width
          del cds[index]
      return cds

    # Fix CDS coding frame.
    fixed_cds = shorten_intervals(cds_transcript, beginning, frame)

    # Set codon.
    codon_bases = 3
    fixed_codon = []
    for seq in fixed_cds if beginning else fixed_cds[::-1]:
      if codon_bases == 0:
        break
      start, end = seq.start, seq.end
      if seq.width > codon_bases and beginning:
        end = seq.start + codon_bases
      elif seq.width > codon_bases:
        start = seq.end - codon_bases
      interval = genome.Interval(seq.chromosome, start, end, strand=seq.strand)
      fixed_codon.append(interval)
      codon_bases -= interval.width

    if five_prime:
      fixed_cds[0 if beginning else -1].info['frame'] = 0
    else:
      # Remove newly set stop interval from cds.
      fixed_cds = shorten_intervals(fixed_cds, beginning, 3)
    return fixed_cds, fixed_codon


class _RangeExtractor:
  """Range extractor from gtf df."""

  def __init__(self, df: pd.DataFrame):
    self._df_start_end = {
        chromosome: (dfc, dfc['Start'].values, dfc['End'].values)
        for chromosome, dfc in df.groupby('Chromosome')
    }
    self._df_empty = df.iloc[:0]

  def extract(self, interval: genome.Interval) -> pd.DataFrame:
    """Finds all rows that contain the input Interval.

    Args:
      interval: query Interval

    Returns:
    a dataframe containing genome intervals that contain the query Interval
    """
    if interval.chromosome not in self._df_start_end:
      return self._df_empty
    else:
      dfc, start, end = self._df_start_end[interval.chromosome]
      start_contained = (interval.start <= start) & (start <= interval.end)
      end_contained = (interval.start <= end) & (end <= interval.end)
      interval_contained = (interval.start >= start) & (end >= interval.end)
      return dfc[start_contained | end_contained | interval_contained]


class TranscriptExtractor:
  """Transcript extractor from gtf."""

  def __init__(self, gtf_df: pd.DataFrame) -> None:
    """Init.

    Args:
      gtf_df: pd.DataFrame of GENCODE GTF entries containing transcript
        annotation. Must contain columns 'Chromosome', 'Start', 'End', 'Strand',
        'Feature', and 'transcript_id'.
    """
    self._transcript_extractor = _RangeExtractor(
        gtf_df[gtf_df.Feature == 'transcript'][
            ['Chromosome', 'Start', 'End', 'Strand', 'transcript_id']
        ]
    )
    self._transcript_indexed_gtf = gtf_df.set_index('transcript_id')
    self._transcript_from_id_cache = None

  def cache_transcripts(self) -> None:
    """Speed up extract() by converting GTF to dictionary of Transcripts.

    This may take ca 11 minutes on the full human genome GTF of 84k protein
    coding transcripts and 15 s on chr22 (1.5k transcripts).

    Running cache_transcripts() will speed up .extract() by ca 5-10x:
    (11 ms vs 65 ms tested on chr22, or 15 ms vs 160 ms on whole genome).
    """
    self._transcript_from_id_cache = self._transcripts_from_gtf(
        self._transcript_indexed_gtf.reset_index()
    )

  def _transcripts_from_gtf(
      self,
      gtf_df: pd.DataFrame,
  ) -> dict[str, Transcript]:
    return (
        {  # pytype: disable=bad-return-type  # pandas-drop-duplicates-overloads
            transcript_id: (
                Transcript.fix_truncation(
                    Transcript.from_gtf_df(gtf_subset, ignore_info=False)
                )
            )
            for transcript_id, gtf_subset in gtf_df.groupby('transcript_id')
        }
    )

  def extract(self, interval: genome.Interval) -> list[Transcript]:
    """Extract transcripts overlapping an interval.

    Args:
      interval: Interval used to overlap with transcripts.

    Returns:
      List of transcript overlapping `interval`.
    """
    gtf_df_within_interval = self._transcript_extractor.extract(interval)
    if gtf_df_within_interval.empty:
      return []

    transcript_ids = gtf_df_within_interval.transcript_id.dropna().unique()
    if self._transcript_from_id_cache is not None:
      return [
          self._transcript_from_id_cache[transcript_id]
          for transcript_id in transcript_ids
      ]
    else:
      transcript_gtfs = self._transcript_indexed_gtf.loc[
          transcript_ids
      ].reset_index()
      return list(self._transcripts_from_gtf(transcript_gtfs).values())
