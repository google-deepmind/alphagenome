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

"""Utilities for working with gene annotations (e.g., GTFs)."""

import enum
from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphagenome.data import genome


@enum.unique
class TranscriptType(enum.Enum):
  """Valid Transcript types available in the GENCODE GTF."""

  IG_C_GENE = 'IG_C_gene'
  IG_C_PSEUDOGENE = 'IG_C_pseudogene'
  IG_D_GENE = 'IG_D_gene'
  IG_J_GENE = 'IG_J_gene'
  IG_J_PSEUDOGENE = 'IG_J_pseudogene'
  IG_V_GENE = 'IG_V_gene'
  IG_V_PSEUDOGENE = 'IG_V_pseudogene'
  IG_PSEUDOGENE = 'IG_pseudogene'
  MT_RRNA = 'Mt_rRNA'
  MT_TRNA = 'Mt_tRNA'
  TEC = 'TEC'
  TR_C_GENE = 'TR_C_gene'
  TR_D_GENE = 'TR_D_gene'
  TR_J_GENE = 'TR_J_gene'
  TR_J_PSEUDOGENE = 'TR_J_pseudogene'
  TR_V_GENE = 'TR_V_gene'
  TR_V_PSEUDOGENE = 'TR_V_pseudogene'
  ARTIFACT = 'artifact'
  LNCRNA = 'lncRNA'
  MIRNA = 'miRNA'
  MISC_RNA = 'misc_RNA'
  NON_STOP_DECAY = 'non_stop_decay'
  NONSENSE_MEDIATED_DECAY = 'nonsense_mediated_decay'
  PROCESSED_PSEUDOGENE = 'processed_pseudogene'
  PROCESSED_TRANSCRIPT = 'processed_transcript'
  PROTEIN_CODING = 'protein_coding'
  PROTEIN_CODING_CDS_NOT_DEFINED = 'protein_coding_CDS_not_defined'
  PROTEIN_CODING_LOF = 'protein_coding_LoF'
  RRNA = 'rRNA'
  RRNA_PSEUDOGENE = 'rRNA_pseudogene'
  RETAINED_INTRON = 'retained_intron'
  RIBOZYME = 'ribozyme'
  SRNA = 'sRNA'
  SCRNA = 'scRNA'
  SCARNA = 'scaRNA'
  SNRNA = 'snRNA'
  SNORNA = 'snoRNA'
  TRANSCRIBED_PROCESSED_PSEUDOGENE = 'transcribed_processed_pseudogene'
  TRANSCRIBED_UNITARY_PSEUDOGENE = 'transcribed_unitary_pseudogene'
  TRANSCRIBED_UNPROCESSED_PSEUDOGENE = 'transcribed_unprocessed_pseudogene'
  TRANSLATED_PROCESSED_PSEUDOGENE = 'translated_processed_pseudogene'
  UNITARY_PSEUDOGENE = 'unitary_pseudogene'
  UNPROCESSED_PSEUDOGENE = 'unprocessed_pseudogene'
  VAULT_RNA = 'vault_RNA'


def extract_tss(gtf: pd.DataFrame, feature: str = 'transcript') -> pd.DataFrame:
  """Extract transcription start sites (TSS) from a DataFrame.

  Args:
    gtf: pd.DataFrame containing gene annotation.
    feature: Feature in the GTF file to use (either transcript or gene).

  Returns:
    pd.DataFrame containing transcription start sites as zero-width point
    intervals (Start == End, 0-based).
  """
  tss = gtf[(gtf.Feature == feature)].copy()

  # Remove the extra base to make it width=0.
  # .....[)TRANSCRIPT (strand = +)
  # TPIRCSNART[)..... (strand = -)
  new_start = np.where(tss.Strand == '-', tss.End, tss.Start)
  tss.Start = new_start
  tss.End = new_start

  return tss


def filter_transcript_type(
    gtf: pd.DataFrame,
    transcript_types: tuple[TranscriptType, ...] | None = None,
) -> pd.DataFrame:
  """Filter GTF entries by transcript types.

  This function takes a GTF DataFrame and a list of transcript types and returns
  a new DataFrame containing only the transcripts with the specified types.

  The GTF DataFrame must contain a column named 'transcript_type' or
  'transcript_biotype'. The function will raise a ValueError if neither of these
  columns is present.

  Args:
    gtf: pd.DataFrame or pyranges.PyRanges.
    transcript_types: List of valid transcript types to use for filtering.

  Returns:
    pd.DataFrame of GENCODE GTF entries subset to rows with the requested
    transcript types.
  """
  if transcript_types is not None:
    transcript_types_str = [x.value for x in transcript_types]
    if 'transcript_type' in gtf.columns:
      gtf = gtf[gtf.transcript_type.isin(transcript_types_str)]
    elif 'transcript_biotype' in gtf.columns:
      gtf = gtf[gtf.transcript_biotype.isin(transcript_types_str)]
    else:
      raise ValueError('transcript_type or transcript_biotype not in gtf.')
  return gtf


def filter_protein_coding(
    gtf: pd.DataFrame, include_gene_entries: bool = False
) -> pd.DataFrame:
  """Filter GTF entries to only protein-coding genes.

  Args:
    gtf: pd.DataFrame of GENCODE GTF entries. This data frame must contain a
      column named 'transcript_type' or 'transcript_biotype'.
    include_gene_entries: Whether to include gene entries in addition to
      transcript entries.

  Returns:
    pd.DataFrame of GENCODE GTF entries subset to rows with protein-coding
    genes.
  """
  if include_gene_entries:
    if 'gene_type' in gtf.columns:
      gtf = gtf[gtf.gene_type == TranscriptType.PROTEIN_CODING.value]
    else:
      raise ValueError('gene_type not in gtf.')
  else:
    gtf = filter_transcript_type(gtf, (TranscriptType.PROTEIN_CODING,))
  return gtf


def filter_to_longest_transcript(
    gtf: pd.DataFrame,
) -> pd.DataFrame:
  """Filter GTF entries to only the longest transcript per gene.

  Args:
    gtf: pd.DataFrame of GENCODE GTF entries. Must contain columns 'Feature',
      'End', 'Start', 'gene_id', and 'transcript_id'.

  Returns:
    pd.DataFrame of GENCODE GTF entries subset to rows with the longest
    transcript per gene.
  """
  lengths = gtf[gtf['Feature'] == 'transcript'].reset_index(drop=True)
  lengths['transcript_length'] = lengths['End'] - lengths['Start'] + 1

  # Identify longest transcripts per gene_id.
  longest_transcripts = lengths.loc[
      lengths.groupby('gene_id')['transcript_length'].idxmax()
  ]

  return gtf[gtf['transcript_id'].isin(longest_transcripts['transcript_id'])]


def filter_to_mane_select_transcript(gtf: pd.DataFrame) -> pd.DataFrame:
  """Filter GTF entries to only the MANE select transcript.

  Note that the MANE_Select tag only exists for the human GTF file.

  Args:
    gtf: pd.DataFrame of GENCODE GTF entries. Must contain columns 'tag'.

  Returns:
    pd.DataFrame of GENCODE GTF entries subset to rows representing MANE
    select transcripts, which are transcripts that are well-supported,
    conserved, and expressed.
  """
  filtered_gtf = gtf[gtf['tag'].fillna('').str.contains('MANE_Select')]
  if filtered_gtf.empty:
    raise ValueError(
        'No MANE_Select transcripts found in the GTF, possibly due to non-human'
        ' GTF.'
    )
  return filtered_gtf


def filter_transcript_support_level(
    gtf: pd.DataFrame,
    transcript_support_levels: str | Sequence[str],
) -> pd.DataFrame:
  """Filter GTF to only transcripts with specific GENCODE support levels.

  As documented in the [Ensembl
  glossary](https://www.ensembl.org/Help/Glossary),
  the transcript support level (TSL) indicates the degree of evidence that was
  used to construct the transcript.

  As taken from the glossary, the levels are:

  | Transcript support level | Description |  |
  |---|---|---|
  | 1 | A transcript where all splice junctions are supported by at least one
  non-suspect mRNA. |
  | 2 | A transcript where the best supporting mRNA is flagged as suspect or the
  support is from multiple ESTs |
  | 3 | A transcript where the only support is from a single EST |
  | 4 | A transcript where the best supporting EST is flagged as suspect |
  | 5 | A transcript where no single transcript supports the model structure. |
  | NA | A transcript that was not analysed for TSL. |

  Args:
    gtf: pd.DataFrame of GENCODE GTF entries. Must contain column
      'transcript_support_level'.
    transcript_support_levels: List of valid transcript support levels to use
      for filtering. This must be a subset of ['1', '2', '3', '4', '5']. Can
      also be single string.

  Returns:
    pd.DataFrame exactly as provided, but subset to rows with the specified
    support level(s).

  Transcripts are scored by GENCODE according to how well mRNA and EST
  alignments
  match over its full length. Valid levels are:
  '1': All splice junctions of the transcript are supported by at least one
  non-suspect mRNA.
  '2': The best supporting mRNA is flagged as suspect or the support is from
  multiple ESTs.
  '3': The only support is from a single EST.
  '4': The best supporting EST is flagged as suspect.
  '5': No single transcript supports the model structure.
  'NA': The transcript was not analyzed (not supported by this filter function).

  See GENCODE GTF format documentation for further details:
  https://www.gencodegenes.org/pages/data_format.html
  """
  if isinstance(transcript_support_levels, str):
    transcript_support_levels = list(transcript_support_levels)

  supported_tsls = {'1', '2', '3', '4', '5'}
  if not set(transcript_support_levels).issubset(supported_tsls):
    raise ValueError(
        f'transcript_support_level must be one of {supported_tsls}, but was'
        f' {transcript_support_levels}'
    )
  return gtf[gtf.transcript_support_level.isin(transcript_support_levels)]


def upgrade_annotation_ids(
    old_ids: pd.Series, new_ids: pd.Series, patchless: bool = False
) -> pd.Series:
  """Upgrade or add transcript id patch version to Ensembl IDs.

  This function works by

  1. Dropping the patch version from `old_ids` and `new_ids`
  2. Merging the two on the patch-less ids.
  3. Returning the result of the merge as a pd.Series, with the 'old_ids' as
  the index and the 'new_ids' as the values.

  Ensembl patch versions have two formats: ENST####.<patch>_PAR_Y or
  ENST####.<patch>. Both are handled.

  The function will raise a ValueError if either `old_ids` or `new_ids` result
  in duplicates after dropping the patch version.

  Examples:
  * If the old ids are ENST00010.1 and the new ids are ENST00010.3,
  then the mapping will be ENST00010.1 -> ENST00010.3.
  * If the old ids are ENST00010 and the new ids are ENST00010.3, then the
  mapping will be ENST00010 -> ENST00010.3.

  Args:
    old_ids: A pd.Series of Ensembl transcript or gene ids with older or missing
      version/patch numbers. The index of the series is ignored.
    new_ids: A pd.Series of transcript or gene ids with newer version/patch
      numbers. The index of the series is ignored.
    patchless: whether old_ids are missing patch.

  Returns:
    A pd.Series, with the 'old_ids' as the index and the 'new_ids' as the
    values.
  """
  new_ids = new_ids.drop_duplicates()

  def drop_version(x):
    """Drop the patch version from an Ensembl ID."""
    if not x.str.contains('.', regex=False).all():
      raise ValueError('All ids need to contain the patch version.')

    id_split = x.str.partition('.')

    # Retain anything after _ such as PAR_Y for ENST####.<patch>_PAR_Y.
    return id_split[0] + id_split[2].str.partition('_')[2]

  old_ids_nopatch = old_ids if patchless else drop_version(old_ids)
  new_ids_nopatch = drop_version(new_ids)
  assert (
      not old_ids_nopatch.duplicated().any()
  ), 'old_ids not unique without version'
  assert (
      not new_ids_nopatch.duplicated().any()
  ), 'new_ids not unique without version'
  df = pd.merge(
      pd.DataFrame({'old': old_ids.values, 'no_version': old_ids_nopatch}),
      pd.DataFrame({
          'new': new_ids.values,
          'no_version': new_ids_nopatch,
      }),
      on='no_version',
      how='left',
  )
  return pd.Series(
      df.set_index('old').loc[old_ids].new.values, index=old_ids.index
  )


def get_gene_intervals(
    gtf: pd.DataFrame,
    gene_symbols: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
) -> list[genome.Interval]:
  """Returns a list of stranded `genome.Interval`s for the given identifiers.

  Args:
    gtf: pd.DataFrame of GENCODE GTF entries. Must contain columns 'Feature',
      'gene_name', 'gene_id', 'Chromosome', 'Start', 'End', and 'Strand'.
    gene_symbols: A sequence of gene names or gene symbols (e.g., ['EGFR',
      'TNF', 'TP53']). Matching is case-insensitive.
    gene_ids: A sequence of Ensembl gene IDs, which can be patched (e.g.
      ['ENSG00000141510.17']) or unpatched (e.g., ['ENSG00000141510']). Matching
      is done on unpatched IDs.

  Returns:
    A list of `genome.Interval`s for the given identifiers. The
    returned list of intervals is in the same order as the input gene
    identifiers.

  Raises:
    ValueError: If neither or both gene_symbols and gene_ids are set, or if no
      interval or multiple intervals are found for any of the given gene
      identifiers.
  """
  if (gene_symbols is None) == (gene_ids is None):
    raise ValueError('Exactly one of gene_symbols or gene_ids must be set.')

  gtf_genes = gtf[gtf['Feature'] == 'gene'].copy()

  if gene_symbols is not None:
    id_col = 'gene_name'
    input_ids = gene_symbols
    process_fn = lambda s: s.str.upper()
  else:
    id_col = 'gene_id'
    input_ids = gene_ids
    process_fn = lambda s: s.str.split('.', n=1).str[0]

  processed_input_ids = process_fn(pd.Series(input_ids, dtype=str))
  gtf_genes['processed_id'] = process_fn(gtf_genes[id_col])

  # Filter the GTF to only the genes that are in the input IDs.
  gtf_subset = gtf_genes[
      gtf_genes['processed_id'].isin(processed_input_ids.unique())
  ]

  dup_mask = gtf_subset['processed_id'].duplicated(keep=False)
  if dup_mask.any():
    offending_ids = gtf_subset.loc[dup_mask, id_col].unique()
    raise ValueError(
        'Multiple intervals found for gene(s):'
        f' {", ".join(sorted(offending_ids))}.'
    )

  # Create a lookup map from processed_id to GTF data.
  # Use reindex to order genes by input and insert NaNs for missing genes.
  gtf_map = gtf_subset.set_index('processed_id')
  result_df = gtf_map.reindex(processed_input_ids)

  missing_mask = result_df['Chromosome'].isnull()
  if missing_mask.any():
    missing_ids = pd.Series(input_ids)[missing_mask.values].unique()
    raise ValueError(
        f'No interval found for gene(s): {", ".join(sorted(missing_ids))}.'
    )

  # Add original identifiers to the result for the 'name' field of the interval.
  result_df[id_col] = list(input_ids)

  return [
      genome.Interval(
          chromosome=row.Chromosome,
          start=row.Start,
          end=row.End,
          strand=row.Strand,
          name=getattr(row, id_col),
      )
      for row in result_df.itertuples()
  ]


def get_gene_interval(
    gtf: pd.DataFrame,
    gene_symbol: str | None = None,
    gene_id: str | None = None,
) -> genome.Interval:
  """Returns a stranded `genome.Interval` given a gene identifier.

  Either gene_symbol or gene_id must be set, but not both.

  Args:
    gtf: pd.DataFrame of GENCODE GTF entries. Must contain columns 'Feature',
      'gene_name', 'gene_id', 'Chromosome', 'Start', 'End', and 'Strand'.
    gene_symbol: A gene name or gene symbol (e.g., 'EGFR', 'TNF', 'TP53')
    gene_id: An Ensembl gene ID, which can be patched (e.g.
      'ENSG00000141510.17') or unpatched (e.g., 'ENSG00000141510').

  Returns:
    A `genome.Interval` for the given gene identifier.

  Raises:
    ValueError: If neither or both gene_symbol and gene_id are set, or if no
      interval or multiple intervals are found for the given gene identifier.
  """
  if sum(x is not None for x in [gene_symbol, gene_id]) != 1:
    raise ValueError('Exactly one of gene_symbol or gene_id must be set.')

  return get_gene_intervals(
      gtf,
      [gene_symbol] if gene_symbol else None,
      [gene_id] if gene_id else None,
  )[0]
