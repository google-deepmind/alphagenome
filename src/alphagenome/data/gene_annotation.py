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

from collections.abc import Sequence
import enum
from typing import Literal, Optional, Union, Tuple, List

from alphagenome.data import genome
import numpy as np
import pandas as pd


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


class GTFError(Exception):
    """Base exception for GTF processing errors."""
    pass


class MissingColumnError(GTFError):
    """Raised when required columns are missing from GTF."""
    pass


class AmbiguousGeneError(GTFError):
    """Raised when multiple intervals found for a gene."""
    pass


class EmptyInputError(GTFError):
    """Raised when input sequences are empty."""
    pass


class InvalidIDError(GTFError):
    """Raised when IDs have invalid format."""
    pass


def _validate_gtf_columns(
    gtf: pd.DataFrame, 
    required_columns: Sequence[str], 
    operation: str
) -> None:
    """Validate that GTF DataFrame has required columns.
    
    Args:
        gtf: GTF DataFrame to validate.
        required_columns: Columns that must be present.
        operation: Name of operation for error message.
        
    Raises:
        MissingColumnError: If any required column is missing.
    """
    missing = [col for col in required_columns if col not in gtf.columns]
    if missing:
        raise MissingColumnError(
            f"Operation '{operation}' requires columns {missing} "
            f"but they are not in GTF. Available columns: {list(gtf.columns)}"
        )


def extract_tss(
    gtf: pd.DataFrame, 
    feature: Literal['transcript', 'gene'] = 'transcript'
) -> pd.DataFrame:
    """Extract transcription start sites (TSS) from a DataFrame.
    
    Args:
        gtf: pd.DataFrame containing gene annotation. Must contain 'Feature',
            'Start', 'End', and 'Strand' columns.
        feature: Feature in the GTF file to use (either 'transcript' or 'gene').
            
    Returns:
        pd.DataFrame containing transcription start sites (width=0, 0-based).
        
    Raises:
        MissingColumnError: If required columns are missing.
        ValueError: If feature is not 'transcript' or 'gene'.
    """
    if feature not in ('transcript', 'gene'):
        raise ValueError(f"feature must be 'transcript' or 'gene', got '{feature}'")
    
    _validate_gtf_columns(
        gtf, ['Feature', 'Start', 'End', 'Strand'], 'extract_tss'
    )
    
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
    transcript_types: Optional[Tuple[TranscriptType, ...]] = None,
) -> pd.DataFrame:
    """Filter GTF entries by transcript types.
    
    This function takes a GTF DataFrame and a list of transcript types and returns
    a new DataFrame containing only the transcripts with the specified types.
    
    The GTF DataFrame must contain a column named 'transcript_type' or
    'transcript_biotype'. The function will raise a ValueError if neither of these
    columns is present.
    
    Args:
        gtf: pd.DataFrame containing GTF entries.
        transcript_types: Tuple of valid transcript types to use for filtering.
            If None or empty, returns the input GTF unchanged.
            
    Returns:
        pd.DataFrame of GTF entries subset to rows with the requested
        transcript types.
        
    Raises:
        MissingColumnError: If neither 'transcript_type' nor 'transcript_biotype'
            column is present.
        ValueError: If transcript_types is an empty tuple.
    """
    if transcript_types is None:
        return gtf.copy() if gtf is not None else gtf
    
    if not transcript_types:
        raise ValueError("transcript_types cannot be an empty tuple")
    
    transcript_types_str = [x.value for x in transcript_types]
    
    if 'transcript_type' in gtf.columns:
        gtf_filtered = gtf[gtf.transcript_type.isin(transcript_types_str)]
    elif 'transcript_biotype' in gtf.columns:
        gtf_filtered = gtf[gtf.transcript_biotype.isin(transcript_types_str)]
    else:
        raise MissingColumnError(
            "Neither 'transcript_type' nor 'transcript_biotype' column found in GTF. "
            f"Available columns: {list(gtf.columns)}"
        )
    
    return gtf_filtered


def filter_protein_coding(
    gtf: pd.DataFrame, 
    include_gene_entries: bool = False
) -> pd.DataFrame:
    """Filter GTF entries to only protein-coding genes.
    
    Args:
        gtf: pd.DataFrame of GTF entries.
        include_gene_entries: Whether to include gene entries in addition to
            transcript entries.
            
    Returns:
        pd.DataFrame of GTF entries subset to rows with protein-coding genes.
        
    Raises:
        MissingColumnError: If required columns are missing.
    """
    if include_gene_entries:
        _validate_gtf_columns(gtf, ['gene_type'], 'filter_protein_coding')
        gtf_filtered = gtf[gtf.gene_type == TranscriptType.PROTEIN_CODING.value]
    else:
        gtf_filtered = filter_transcript_type(gtf, (TranscriptType.PROTEIN_CODING,))
    
    return gtf_filtered


def filter_to_longest_transcript(
    gtf: pd.DataFrame,
) -> pd.DataFrame:
    """Filter GTF entries to only the longest transcript per gene.
    
    Args:
        gtf: pd.DataFrame of GTF entries. Must contain columns 'Feature',
            'End', 'Start', 'gene_id', and 'transcript_id'.
            
    Returns:
        pd.DataFrame of GTF entries subset to rows with the longest
        transcript per gene.
        
    Raises:
        MissingColumnError: If required columns are missing.
    """
    _validate_gtf_columns(
        gtf, ['Feature', 'Start', 'End', 'gene_id', 'transcript_id'], 
        'filter_to_longest_transcript'
    )
    
    # Get transcript entries and calculate lengths
    transcript_mask = gtf['Feature'] == 'transcript'
    if not transcript_mask.any():
        raise ValueError("No transcript entries found in GTF")
    
    lengths = gtf[transcript_mask].copy()
    lengths['transcript_length'] = lengths['End'] - lengths['Start'] + 1

    # Identify longest transcripts per gene_id
    longest_transcripts = lengths.loc[
        lengths.groupby('gene_id')['transcript_length'].idxmax()
    ]

    return gtf[gtf['transcript_id'].isin(longest_transcripts['transcript_id'])]


def filter_to_mane_select_transcript(gtf: pd.DataFrame) -> pd.DataFrame:
    """Filter GTF entries to only the MANE select transcript.
    
    Note that the MANE_Select tag only exists for the human GTF file.
    
    Args:
        gtf: pd.DataFrame of GTF entries. Must contain column 'tag'.
            
    Returns:
        pd.DataFrame of GTF entries subset to rows representing MANE
        select transcripts.
        
    Raises:
        MissingColumnError: If 'tag' column is missing.
        ValueError: If no MANE_Select transcripts found.
    """
    _validate_gtf_columns(gtf, ['tag'], 'filter_to_mane_select_transcript')
    
    if 'tag' not in gtf.columns:
        # Handle case where tag might be missing but allow NaN values
        gtf['tag'] = gtf.get('tag', pd.Series([], dtype=str))
    
    filtered_gtf = gtf[gtf['tag'].fillna('').str.contains('MANE_Select')]
    if filtered_gtf.empty:
        raise ValueError(
            'No MANE_Select transcripts found in the GTF. '
            'This could be due to: '
            '1) Non-human GTF file, '
            '2) Different annotation version, '
            '3) Missing or different tag column name.'
        )
    return filtered_gtf


def filter_transcript_support_level(
    gtf: pd.DataFrame,
    transcript_support_levels: Union[str, Sequence[str]],
) -> pd.DataFrame:
    """Filter GTF to only transcripts with specific GENCODE support levels.
    
    As documented in the Ensembl glossary, the transcript support level (TSL)
    indicates the degree of evidence that was used to construct the transcript.
    
    Args:
        gtf: pd.DataFrame of GTF entries. Must contain column
            'transcript_support_level'.
        transcript_support_levels: Valid transcript support levels to use
            for filtering. Must be a subset of {'1', '2', '3', '4', '5'}.
            
    Returns:
        pd.DataFrame exactly as provided, but subset to rows with the specified
        support level(s).
        
    Raises:
        MissingColumnError: If 'transcript_support_level' column is missing.
        ValueError: If invalid support levels are provided.
    """
    _validate_gtf_columns(
        gtf, ['transcript_support_level'], 'filter_transcript_support_level'
    )
    
    if isinstance(transcript_support_levels, str):
        transcript_support_levels = [transcript_support_levels]
    
    supported_tsls = {'1', '2', '3', '4', '5'}
    invalid_levels = set(transcript_support_levels) - supported_tsls
    
    if invalid_levels:
        raise ValueError(
            f"Invalid transcript support level(s): {sorted(invalid_levels)}. "
            f"Must be one of: {sorted(supported_tsls)}"
        )
    
    return gtf[gtf.transcript_support_level.isin(transcript_support_levels)]


def _strip_version_from_id(id_series: pd.Series) -> pd.Series:
    """Strip version/patch suffix from Ensembl IDs.
    
    Handles formats:
    - ENST00000123456.1 -> ENST00000123456
    - ENST00000123456.1_PAR_Y -> ENST00000123456_PAR_Y
    
    Args:
        id_series: Series of Ensembl IDs.
        
    Returns:
        Series of IDs with version stripped.
        
    Raises:
        InvalidIDError: If any ID doesn't contain a dot (invalid format).
    """
    # Check if all IDs have the expected format
    if not id_series.str.contains('.', regex=False).all():
        invalid_ids = id_series[~id_series.str.contains('.', regex=False)]
        raise InvalidIDError(
            f"All IDs must contain a dot separator (e.g., ENSG00000123456.1). "
            f"Invalid IDs: {invalid_ids.tolist()[:10]}"  # Show first 10
        )
    
    # Split on first dot, keep part after underscore if present
    id_split = id_series.str.partition('.')
    return id_split[0] + id_split[2].str.partition('_')[2]


def upgrade_annotation_ids(
    old_ids: pd.Series, 
    new_ids: pd.Series, 
    patchless: bool = False
) -> pd.Series:
    """Upgrade or add transcript id patch version to Ensembl IDs.
    
    This function works by:
    1. Dropping the patch version from `old_ids` and `new_ids` if not patchless
    2. Merging the two on the patch-less ids
    3. Returning the result of the merge as a pd.Series
    
    Examples:
    * If the old ids are ENST00010.1 and the new ids are ENST00010.3,
      then the mapping will be ENST00010.1 -> ENST00010.3.
    * If patchless=True and old ids are ENST00010 and new ids are ENST00010.3,
      then the mapping will be ENST00010 -> ENST00010.3.
    
    Args:
        old_ids: A pd.Series of Ensembl transcript or gene ids with older or
            missing version/patch numbers.
        new_ids: A pd.Series of transcript or gene ids with newer version/patch
            numbers.
        patchless: If True, assumes both old_ids and new_ids are already
            patchless. If False, strips versions from both.
            
    Returns:
        A pd.Series with the same index as old_ids, containing the upgraded IDs.
        If an old ID cannot be mapped, the value will be NaN.
        
    Raises:
        InvalidIDError: If IDs don't have expected format when patchless=False.
        ValueError: If duplicate IDs are found after stripping versions.
    """
    # Input validation
    if old_ids.empty or new_ids.empty:
        raise EmptyInputError("old_ids and new_ids cannot be empty")
    
    # Remove duplicates while preserving first occurrence
    old_ids_dedup = old_ids[~old_ids.duplicated()]
    new_ids_dedup = new_ids[~new_ids.duplicated()]
    
    if patchless:
        old_ids_nopatch = old_ids_dedup
        new_ids_nopatch = new_ids_dedup
    else:
        old_ids_nopatch = _strip_version_from_id(old_ids_dedup)
        new_ids_nopatch = _strip_version_from_id(new_ids_dedup)
    
    # Check for duplicates in version-stripped IDs
    if old_ids_nopatch.duplicated().any():
        raise ValueError(
            "old_ids not unique after stripping version. "
            f"Duplicates: {old_ids_dedup[old_ids_nopatch.duplicated()].tolist()}"
        )
    if new_ids_nopatch.duplicated().any():
        raise ValueError(
            "new_ids not unique after stripping version. "
            f"Duplicates: {new_ids_dedup[new_ids_nopatch.duplicated()].tolist()}"
        )
    
    # Create DataFrames for merging
    old_df = pd.DataFrame({
        'old': old_ids_dedup.values,
        'no_version': old_ids_nopatch.values
    })
    
    new_df = pd.DataFrame({
        'new': new_ids_dedup.values,
        'no_version': new_ids_nopatch.values
    })
    
    # Merge on version-stripped IDs
    merged = pd.merge(old_df, new_df, on='no_version', how='left')
    
    # Create mapping dictionary for efficient lookup
    mapping = dict(zip(merged['old'], merged['new']))
    
    # Apply mapping to original old_ids (preserving duplicates and order)
    result = old_ids.map(mapping)
    
    return result


def get_gene_intervals(
    gtf: pd.DataFrame,
    gene_symbols: Optional[Sequence[str]] = None,
    gene_ids: Optional[Sequence[str]] = None,
) -> List[genome.Interval]:
    """Returns a list of stranded `genome.Interval`s for the given identifiers.
    
    Args:
        gtf: pd.DataFrame of GTF entries. Must contain columns 'Feature',
            'gene_name', 'gene_id', 'Chromosome', 'Start', 'End', and 'Strand'.
        gene_symbols: A sequence of gene names or gene symbols 
            (e.g., ['EGFR', 'TNF', 'TP53']). Matching is case-insensitive.
        gene_ids: A sequence of Ensembl gene IDs, which can be patched or
            unpatched. Matching is done on unpatched IDs.
            
    Returns:
        A list of `genome.Interval`s for the given identifiers. The
        returned list of intervals is in the same order as the input gene
        identifiers.
        
    Raises:
        EmptyInputError: If gene_symbols or gene_ids is empty.
        AmbiguousGeneError: If multiple intervals found for any gene identifier.
        MissingColumnError: If required columns are missing.
    """
    # Validate exactly one identifier type is provided
    if (gene_symbols is None) == (gene_ids is None):
        raise ValueError('Exactly one of gene_symbols or gene_ids must be set.')
    
    # Validate input sequences are not empty
    if gene_symbols is not None:
        if not gene_symbols:
            raise EmptyInputError('gene_symbols cannot be empty.')
        input_ids = list(gene_symbols)
        id_col = 'gene_name'
        def process_fn(s: pd.Series) -> pd.Series:
            return s.str.upper()
    else:
        if not gene_ids:
            raise EmptyInputError('gene_ids cannot be empty.')
        input_ids = list(gene_ids)
        id_col = 'gene_id'
        def process_fn(s: pd.Series) -> pd.Series:
            return s.str.split('.', n=1).str[0]
    
    # Validate GTF has required columns
    _validate_gtf_columns(
        gtf, ['Feature', 'Chromosome', 'Start', 'End', 'Strand', id_col],
        'get_gene_intervals'
    )
    
    # Filter to gene entries only
    gene_mask = gtf['Feature'] == 'gene'
    if not gene_mask.any():
        raise ValueError("No gene entries found in GTF")
    
    gtf_genes = gtf[gene_mask].copy()
    
    # Process identifiers for matching
    input_series = pd.Series(input_ids, dtype=str)
    processed_input_ids = process_fn(input_series)
    
    # Handle NaN/None values in input
    if processed_input_ids.isnull().any():
        nan_indices = processed_input_ids.index[processed_input_ids.isnull()]
        raise ValueError(
            f"Input identifiers contain NaN/None values at indices: {nan_indices.tolist()}"
        )
    
    gtf_genes['processed_id'] = process_fn(gtf_genes[id_col])
    
    # Filter to genes in input
    gtf_subset = gtf_genes[
        gtf_genes['processed_id'].isin(processed_input_ids.unique())
    ]
    
    # Check for ambiguous genes (multiple intervals for same identifier)
    dup_mask = gtf_subset['processed_id'].duplicated(keep=False)
    if dup_mask.any():
        offending_ids = gtf_subset.loc[dup_mask, id_col].unique()
        raise AmbiguousGeneError(
            f'Multiple intervals found for gene(s): {", ".join(sorted(offending_ids))}.'
        )
    
    # Create lookup map and reindex by processed input IDs
    gtf_map = gtf_subset.set_index('processed_id')
    result_df = gtf_map.reindex(processed_input_ids.values)  # Use .values, not index
    
    # Check for missing genes
    missing_mask = result_df['Chromosome'].isnull()
    if missing_mask.any():
        missing_indices = missing_mask.index[missing_mask]
        missing_original_ids = [input_ids[i] for i in missing_indices]
        raise AmbiguousGeneError(
            f'No interval found for gene(s): {", ".join(sorted(missing_original_ids))}.'
        )
    
    # Create intervals in the same order as input
    intervals = []
    for idx, row in enumerate(result_df.itertuples()):
        intervals.append(
            genome.Interval(
                chromosome=row.Chromosome,
                start=row.Start,
                end=row.End,
                strand=row.Strand,
                name=input_ids[idx],  # Use original identifier as name
            )
        )
    
    return intervals


def get_gene_interval(
    gtf: pd.DataFrame,
    gene_symbol: Optional[str] = None,
    gene_id: Optional[str] = None,
) -> genome.Interval:
    """Returns a stranded `genome.Interval` given a single gene identifier.
    
    Args:
        gtf: pd.DataFrame of GTF entries.
        gene_symbol: A gene name or gene symbol (e.g., 'EGFR', 'TNF', 'TP53').
        gene_id: An Ensembl gene ID, which can be patched or unpatched.
            
    Returns:
        A `genome.Interval` for the given gene identifier.
        
    Raises:
        ValueError: If neither or both gene_symbol and gene_id are set, or if
            no interval or multiple intervals are found.
    """
    if sum(x is not None for x in [gene_symbol, gene_id]) != 1:
        raise ValueError('Exactly one of gene_symbol or gene_id must be set.')
    
    intervals = get_gene_intervals(
        gtf,
        [gene_symbol] if gene_symbol else None,
        [gene_id] if gene_id else None,
    )
    
    return intervals[0]
