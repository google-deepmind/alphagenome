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

"""Comprehensive utilities for working with gene annotations (GTF/GFF3 format)."""

import enum
import re
from collections.abc import Sequence
from typing import Optional, Union, List, Dict, Any, Literal, Tuple
import warnings

import numpy as np
import pandas as pd


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class GTFAnnotationError(Exception):
    """Base exception for all GTF annotation errors."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message
    
    def __str__(self) -> str:
        if self.context:
            context_str = ', '.join(f"{k}: {v}" for k, v in self.context.items())
            return f"{self.message} [Context: {context_str}]"
        return self.message


class MissingColumnError(GTFAnnotationError):
    """Raised when required columns are missing from GTF."""
    def __init__(self, missing_columns: List[str], available_columns: List[str], 
                 operation: str = None):
        context = {
            'missing_columns': missing_columns,
            'available_columns': available_columns,
            'operation': operation
        }
        message = (f"Missing required columns: {missing_columns}. "
                  f"Available columns: {available_columns}")
        if operation:
            message = f"Operation '{operation}' failed: {message}"
        super().__init__(message, context)


class AmbiguousGeneError(GTFAnnotationError):
    """Raised when multiple intervals found for a gene identifier."""
    def __init__(self, gene_identifier: str, found_entries: List[str] = None):
        context = {
            'gene_identifier': gene_identifier,
            'found_entries': found_entries or []
        }
        message = f"Multiple intervals found for gene: {gene_identifier}"
        if found_entries:
            message += f". Found {len(found_entries)} entries"
        super().__init__(message, context)


class GeneNotFoundError(GTFAnnotationError):
    """Raised when a gene identifier is not found in the GTF."""
    def __init__(self, gene_identifier: str, searched_columns: List[str] = None):
        context = {
            'gene_identifier': gene_identifier,
            'searched_columns': searched_columns or []
        }
        message = f"Gene not found: {gene_identifier}"
        if searched_columns:
            message += f" (searched in: {searched_columns})"
        super().__init__(message, context)


class InvalidInputError(GTFAnnotationError):
    """Raised when input parameters are invalid."""
    pass


class InvalidIDFormatError(GTFAnnotationError):
    """Raised when Ensembl IDs have invalid format."""
    def __init__(self, invalid_ids: List[str], expected_format: str = None):
        context = {
            'invalid_ids': invalid_ids[:10],  # Show first 10
            'total_invalid': len(invalid_ids),
            'expected_format': expected_format
        }
        message = f"Invalid ID format for: {invalid_ids[:5]}"
        if len(invalid_ids) > 5:
            message += f" and {len(invalid_ids) - 5} more"
        if expected_format:
            message += f". Expected format: {expected_format}"
        super().__init__(message, context)


class EmptyDataError(GTFAnnotationError):
    """Raised when input data is empty."""
    def __init__(self, data_name: str, operation: str = None):
        context = {
            'data_name': data_name,
            'operation': operation
        }
        message = f"{data_name} cannot be empty"
        if operation:
            message = f"Operation '{operation}' failed: {message}"
        super().__init__(message, context)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def _validate_gtf_dataframe(
    gtf: pd.DataFrame,
    required_columns: List[str],
    operation: str = None,
    allow_empty: bool = False
) -> None:
    """
    Validate GTF DataFrame structure and content.
    
    Args:
        gtf: DataFrame to validate.
        required_columns: Columns that must be present.
        operation: Name of operation for error context.
        allow_empty: Whether empty DataFrame is allowed.
        
    Raises:
        EmptyDataError: If DataFrame is empty and allow_empty=False.
        MissingColumnError: If required columns are missing.
        ValueError: If DataFrame contains invalid data.
    """
    if gtf is None:
        raise EmptyDataError("GTF DataFrame", operation)
    
    if not allow_empty and gtf.empty:
        raise EmptyDataError("GTF DataFrame", operation)
    
    # Check required columns
    missing = [col for col in required_columns if col not in gtf.columns]
    if missing:
        raise MissingColumnError(
            missing_columns=missing,
            available_columns=list(gtf.columns),
            operation=operation
        )
    
    # Validate column data types for key columns
    if 'Start' in gtf.columns and 'End' in gtf.columns:
        if not pd.api.types.is_numeric_dtype(gtf['Start']):
            raise ValueError("'Start' column must contain numeric values")
        if not pd.api.types.is_numeric_dtype(gtf['End']):
            raise ValueError("'End' column must contain numeric values")
        
        # Convert to int if possible
        if pd.api.types.is_float_dtype(gtf['Start']):
            if gtf['Start'].notna().any() and (gtf['Start'].dropna() % 1 != 0).any():
                raise ValueError("'Start' column contains non-integer floats")
            gtf['Start'] = gtf['Start'].astype('Int64')
        
        if pd.api.types.is_float_dtype(gtf['End']):
            if gtf['End'].notna().any() and (gtf['End'].dropna() % 1 != 0).any():
                raise ValueError("'End' column contains non-integer floats")
            gtf['End'] = gtf['End'].astype('Int64')
        
        # Check for invalid intervals
        invalid_intervals = gtf[gtf['End'] < gtf['Start']]
        if not invalid_intervals.empty:
            raise ValueError(
                f"Found {len(invalid_intervals)} intervals where End < Start. "
                f"First example: {invalid_intervals.iloc[0].to_dict()}"
            )
    
    if 'Strand' in gtf.columns:
        valid_strands = {'+', '-', '.', '?'}
        invalid_mask = ~gtf['Strand'].isin(valid_strands)
        invalid_strands = gtf.loc[invalid_mask, 'Strand'].dropna().unique().tolist()
        if invalid_strands:
            raise ValueError(
                f"Invalid strand values: {invalid_strands}. "
                f"Valid values are: {sorted(valid_strands)}"
            )


def _validate_gene_identifiers(
    gene_symbols: Optional[List[str]] = None,
    gene_ids: Optional[List[str]] = None
) -> Tuple[str, List[str]]:
    """
    Validate gene identifier inputs and return identifier type and list.
    
    Args:
        gene_symbols: List of gene symbols.
        gene_ids: List of gene IDs.
        
    Returns:
        Tuple of (identifier_type, identifier_list).
        
    Raises:
        InvalidInputError: If inputs are invalid.
    """
    if (gene_symbols is None) == (gene_ids is None):
        raise InvalidInputError(
            "Exactly one of gene_symbols or gene_ids must be provided",
            context={
                'gene_symbols_provided': gene_symbols is not None,
                'gene_ids_provided': gene_ids is not None
            }
        )
    
    if gene_symbols is not None:
        if not gene_symbols:
            raise EmptyDataError("gene_symbols list")
        if not all(isinstance(g, str) and g.strip() for g in gene_symbols):
            invalid = [g for g in gene_symbols 
                      if not isinstance(g, str) or not g.strip()]
            raise InvalidInputError(
                "All gene symbols must be non-empty strings",
                context={'invalid_symbols': invalid}
            )
        return 'gene_name', gene_symbols
    
    if gene_ids is not None:
        if not gene_ids:
            raise EmptyDataError("gene_ids list")
        if not all(isinstance(g, str) and g.strip() for g in gene_ids):
            invalid = [g for g in gene_ids 
                      if not isinstance(g, str) or not g.strip()]
            raise InvalidInputError(
                "All gene IDs must be non-empty strings",
                context={'invalid_ids': invalid}
            )
        return 'gene_id', gene_ids


def _validate_ensembl_ids(
    ids: pd.Series,
    require_version: bool = False,
    id_type: str = "Ensembl ID"
) -> pd.Series:
    """
    Validate Ensembl ID format.
    
    Args:
        ids: Series of Ensembl IDs.
        require_version: Whether version number is required.
        id_type: Type of ID for error messages.
        
    Returns:
        Cleaned Series of IDs.
        
    Raises:
        InvalidIDFormatError: If IDs have invalid format.
        EmptyDataError: If IDs are empty.
    """
    if ids.empty:
        raise EmptyDataError(f"{id_type} series")
    
    # Remove whitespace and handle NaN
    ids_clean = ids.astype(str).str.strip()
    
    if require_version:
        # Check for version number (e.g., ENSG00000141510.17)
        pattern = r'^ENS[TG]\d{11}\.\d+'
        invalid_mask = ~ids_clean.str.contains(pattern, na=False)
    else:
        # Basic Ensembl ID pattern (with or without version)
        pattern = r'^ENS[TG]\d{11}(\.\d+)?'
        invalid_mask = ~ids_clean.str.contains(pattern, na=False)
    
    invalid_ids = ids_clean[invalid_mask].dropna().tolist()
    if invalid_ids:
        raise InvalidIDFormatError(
            invalid_ids,
            expected_format="ENS[TG]XXXXXXXXXXX[.Y] (e.g., ENSG00000141510.17)"
        )
    
    return ids_clean


def _validate_transcript_types(
    transcript_types: Optional[Tuple['TranscriptType', ...]]
) -> None:
    """
    Validate transcript types input.
    
    Args:
        transcript_types: Tuple of transcript types.
        
    Raises:
        InvalidInputError: If transcript_types is invalid.
    """
    if transcript_types is not None:
        if not transcript_types:
            raise InvalidInputError(
                "transcript_types cannot be an empty tuple",
                context={'provided_types': transcript_types}
            )
        invalid_types = [t for t in transcript_types 
                        if not isinstance(t, TranscriptType)]
        if invalid_types:
            raise InvalidInputError(
                f"Invalid transcript types: {invalid_types}",
                context={'valid_types': [t.value for t in TranscriptType]}
            )


def _validate_tsl_levels(
    tsl_levels: Union[str, Sequence[str]]
) -> List[str]:
    """
    Validate transcript support levels.
    
    Args:
        tsl_levels: TSL level(s) to validate.
        
    Returns:
        List of validated TSL levels.
        
    Raises:
        InvalidInputError: If TSL levels are invalid.
    """
    valid_levels = {'1', '2', '3', '4', '5'}
    
    if isinstance(tsl_levels, str):
        tsl_levels = [tsl_levels]
    
    invalid_levels = set(tsl_levels) - valid_levels
    if invalid_levels:
        raise InvalidInputError(
            f"Invalid TSL levels: {sorted(invalid_levels)}",
            context={
                'valid_levels': sorted(valid_levels),
                'provided_levels': tsl_levels
            }
        )
    
    return list(tsl_levels)


# ============================================================================
# TRANSCRIPT TYPE ENUM
# ============================================================================

@enum.unique
class TranscriptType(enum.Enum):
    """Valid Transcript types available in the GENCODE GTF."""
    
    # Protein coding
    PROTEIN_CODING = 'protein_coding'
    PROTEIN_CODING_CDS_NOT_DEFINED = 'protein_coding_CDS_not_defined'
    PROTEIN_CODING_LOF = 'protein_coding_LoF'
    
    # Non-coding RNA
    LNCRNA = 'lncRNA'
    MIRNA = 'miRNA'
    SNORNA = 'snoRNA'
    SNRNA = 'snRNA'
    RRNA = 'rRNA'
    MT_RRNA = 'Mt_rRNA'
    MT_TRNA = 'Mt_tRNA'
    TR_C_GENE = 'TR_C_gene'
    TR_D_GENE = 'TR_D_gene'
    TR_J_GENE = 'TR_J_gene'
    TR_V_GENE = 'TR_V_gene'
    SCRNA = 'scRNA'
    SRNA = 'sRNA'
    MISC_RNA = 'misc_RNA'
    VAULT_RNA = 'vault_RNA'
    RIBOZYME = 'ribozyme'
    SCARNA = 'scaRNA'
    TEC = 'TEC'
    ARTIFACT = 'artifact'
    
    # Pseudogenes
    PROCESSED_PSEUDOGENE = 'processed_pseudogene'
    UNPROCESSED_PSEUDOGENE = 'unprocessed_pseudogene'
    UNITARY_PSEUDOGENE = 'unitary_pseudogene'
    TRANSCRIBED_PROCESSED_PSEUDOGENE = 'transcribed_processed_pseudogene'
    TRANSCRIBED_UNPROCESSED_PSEUDOGENE = 'transcribed_unprocessed_pseudogene'
    TRANSCRIBED_UNITARY_PSEUDOGENE = 'transcribed_unitary_pseudogene'
    TRANSLATED_PROCESSED_PSEUDOGENE = 'translated_processed_pseudogene'
    IG_C_PSEUDOGENE = 'IG_C_pseudogene'
    IG_J_PSEUDOGENE = 'IG_J_pseudogene'
    IG_V_PSEUDOGENE = 'IG_V_pseudogene'
    IG_PSEUDOGENE = 'IG_pseudogene'
    TR_J_PSEUDOGENE = 'TR_J_pseudogene'
    TR_V_PSEUDOGENE = 'TR_V_pseudogene'
    RRNA_PSEUDOGENE = 'rRNA_pseudogene'
    
    # Immunoglobulin genes
    IG_C_GENE = 'IG_C_gene'
    IG_D_GENE = 'IG_D_gene'
    IG_J_GENE = 'IG_J_gene'
    IG_V_GENE = 'IG_V_gene'
    
    # Other
    PROCESSED_TRANSCRIPT = 'processed_transcript'
    RETAINED_INTRON = 'retained_intron'
    NON_STOP_DECAY = 'non_stop_decay'
    NONSENSE_MEDIATED_DECAY = 'nonsense_mediated_decay'


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_tss(
    gtf: pd.DataFrame, 
    feature: Literal['transcript', 'gene'] = 'transcript'
) -> pd.DataFrame:
    """
    Extract transcription start sites (TSS) from a DataFrame.
    
    Args:
        gtf: pd.DataFrame containing gene annotation.
        feature: Feature in the GTF file to use (either 'transcript' or 'gene').
            
    Returns:
        pd.DataFrame containing transcription start sites (width=0, 0-based).
        
    Raises:
        MissingColumnError: If required columns are missing.
        ValueError: If feature is not 'transcript' or 'gene'.
        
    Example:
        >>> tss = extract_tss(gtf_data, feature='transcript')
        >>> print(tss[['Chromosome', 'Start', 'End', 'Strand']].head())
    """
    if feature not in ('transcript', 'gene'):
        raise ValueError(
            f"feature must be 'transcript' or 'gene', got '{feature}'",
            context={'valid_features': ['transcript', 'gene']}
        )
    
    _validate_gtf_dataframe(
        gtf, 
        required_columns=['Feature', 'Start', 'End', 'Strand'],
        operation='extract_tss'
    )
    
    # Check if feature exists in the DataFrame
    if feature not in gtf['Feature'].unique():
        raise ValueError(
            f"Feature '{feature}' not found in GTF. Available features: {gtf['Feature'].unique().tolist()}",
            context={'available_features': gtf['Feature'].unique().tolist()}
        )
    
    tss = gtf[gtf['Feature'] == feature].copy()
    
    if tss.empty:
        warnings.warn(
            f"No '{feature}' entries found in GTF. Returning empty DataFrame.",
            UserWarning
        )
        return tss
    
    # Remove the extra base to make it width=0.
    # .....[)TRANSCRIPT (strand = +)
    # TPIRCSNART[)..... (strand = -)
    new_start = np.where(tss['Strand'] == '-', tss['End'], tss['Start'])
    tss['Start'] = new_start
    tss['End'] = new_start
    
    # Add TSS metadata
    tss['TSS_derived_from'] = feature
    
    return tss


def filter_transcript_type(
    gtf: pd.DataFrame,
    transcript_types: Optional[Tuple[TranscriptType, ...]] = None,
) -> pd.DataFrame:
    """
    Filter GTF entries by transcript types.
    
    This function takes a GTF DataFrame and a list of transcript types and returns
    a new DataFrame containing only the transcripts with the specified types.
    
    Args:
        gtf: pd.DataFrame containing GTF entries.
        transcript_types: Tuple of valid transcript types to use for filtering.
            If None, returns the input GTF unchanged.
            
    Returns:
        pd.DataFrame of GTF entries subset to rows with the requested
        transcript types.
        
    Raises:
        MissingColumnError: If neither 'transcript_type' nor 'transcript_biotype'
            column is present.
        InvalidInputError: If transcript_types is an empty tuple.
        
    Example:
        >>> protein_coding = filter_transcript_type(gtf_data, (TranscriptType.PROTEIN_CODING,))
        >>> lnc_and_mirna = filter_transcript_type(gtf_data, (TranscriptType.LNCRNA, TranscriptType.MIRNA))
    """
    if transcript_types is None:
        return gtf.copy() if gtf is not None else gtf
    
    _validate_transcript_types(transcript_types)
    
    transcript_types_str = [t.value for t in transcript_types]
    
    # Try to find transcript type column
    transcript_type_col = None
    for col in ['transcript_type', 'transcript_biotype', 'gene_type', 'biotype']:
        if col in gtf.columns:
            transcript_type_col = col
            break
    
    if transcript_type_col is None:
        raise MissingColumnError(
            missing_columns=['transcript_type', 'transcript_biotype'],
            available_columns=list(gtf.columns),
            operation='filter_transcript_type'
        )
    
    # Filter by transcript type
    mask = gtf[transcript_type_col].isin(transcript_types_str)
    gtf_filtered = gtf[mask].copy()
    
    if gtf_filtered.empty:
        warnings.warn(
            f"No entries found with transcript types: {transcript_types_str}. "
            "Returning empty DataFrame.",
            UserWarning
        )
    
    return gtf_filtered


def filter_protein_coding(
    gtf: pd.DataFrame, 
    include_gene_entries: bool = False
) -> pd.DataFrame:
    """
    Filter GTF entries to only protein-coding genes.
    
    Args:
        gtf: pd.DataFrame of GTF entries.
        include_gene_entries: Whether to include gene entries in addition to
            transcript entries.
            
    Returns:
        pd.DataFrame of GTF entries subset to rows with protein-coding genes.
        
    Raises:
        MissingColumnError: If required columns are missing.
        
    Example:
        >>> protein_coding_genes = filter_protein_coding(gtf_data, include_gene_entries=True)
    """
    if include_gene_entries:
        if 'gene_type' not in gtf.columns:
            raise MissingColumnError(
                missing_columns=['gene_type'],
                available_columns=list(gtf.columns),
                operation='filter_protein_coding'
            )
        gtf_filtered = gtf[gtf['gene_type'] == TranscriptType.PROTEIN_CODING.value]
    else:
        gtf_filtered = filter_transcript_type(gtf, (TranscriptType.PROTEIN_CODING,))
    
    return gtf_filtered


def filter_to_longest_transcript(gtf: pd.DataFrame) -> pd.DataFrame:
    """
    Filter GTF entries to only the longest transcript per gene.
    
    Args:
        gtf: pd.DataFrame of GTF entries. Must contain columns 'Feature',
            'Start', 'End', 'gene_id', and 'transcript_id'.
            
    Returns:
        pd.DataFrame of GTF entries subset to rows with the longest
        transcript per gene.
        
    Raises:
        MissingColumnError: If required columns are missing.
        ValueError: If no transcript entries found.
        
    Example:
        >>> longest_transcripts = filter_to_longest_transcript(gtf_data)
    """
    _validate_gtf_dataframe(
        gtf,
        required_columns=['Feature', 'Start', 'End', 'gene_id', 'transcript_id'],
        operation='filter_to_longest_transcript'
    )
    
    # Get transcript entries
    transcript_mask = gtf['Feature'] == 'transcript'
    if not transcript_mask.any():
        raise ValueError(
            "No transcript entries found in GTF",
            context={'available_features': gtf['Feature'].unique().tolist()}
        )
    
    lengths = gtf[transcript_mask].copy()
    
    # Calculate transcript length (inclusive of both ends)
    lengths['transcript_length'] = lengths['End'] - lengths['Start'] + 1
    
    # Check for negative or zero lengths
    invalid_lengths = lengths[lengths['transcript_length'] <= 0]
    if not invalid_lengths.empty:
        warnings.warn(
            f"Found {len(invalid_lengths)} transcripts with non-positive length. "
            "These will be excluded from longest transcript selection.",
            UserWarning
        )
        lengths = lengths[lengths['transcript_length'] > 0]
    
    if lengths.empty:
        raise ValueError(
            "No valid transcript entries with positive length found in GTF",
            context={'total_transcripts': transcript_mask.sum()}
        )
    
    # Identify longest transcripts per gene_id
    # Use idxmax to get index of longest transcript per gene
    longest_idx = lengths.groupby('gene_id')['transcript_length'].idxmax()
    
    # Handle cases where idxmax might return NaN for empty groups
    longest_idx = longest_idx.dropna()
    
    if longest_idx.empty:
        raise ValueError(
            "Could not determine longest transcripts - check gene_id values",
            context={'unique_genes': lengths['gene_id'].nunique()}
        )
    
    longest_transcripts = lengths.loc[longest_idx]
    
    # Return all rows for the selected transcripts
    selected_transcript_ids = longest_transcripts['transcript_id'].unique()
    result = gtf[gtf['transcript_id'].isin(selected_transcript_ids)].copy()
    
    # Add metadata about the filtering
    result.attrs['filtering_note'] = 'longest_transcript_per_gene'
    result.attrs['genes_processed'] = lengths['gene_id'].nunique()
    result.attrs['transcripts_selected'] = len(selected_transcript_ids)
    
    return result


def filter_to_mane_select_transcript(gtf: pd.DataFrame) -> pd.DataFrame:
    """
    Filter GTF entries to only the MANE select transcript.
    
    Note: The MANE_Select tag primarily exists for human GTF files.
    
    Args:
        gtf: pd.DataFrame of GTF entries. Must contain column 'tag' or similar.
            
    Returns:
        pd.DataFrame of GTF entries subset to rows representing MANE
        select transcripts.
        
    Raises:
        MissingColumnError: If 'tag' column is missing and cannot be inferred.
        ValueError: If no MANE_Select transcripts found.
        
    Example:
        >>> mane_transcripts = filter_to_mane_select_transcript(human_gtf)
    """
    # Look for tag column (might have different names in different annotations)
    tag_column = None
    for possible_tag in ['tag', 'Tag', 'tags', 'Tags', 'note', 'Note']:
        if possible_tag in gtf.columns:
            tag_column = possible_tag
            break
    
    if tag_column is None:
        # Check if any column contains 'MANE' string in values
        mane_columns = []
        for col in gtf.columns:
            if gtf[col].astype(str).str.contains('MANE', case=False, na=False).any():
                mane_columns.append(col)
        
        if mane_columns:
            warnings.warn(
                f"No explicit 'tag' column found, but MANE data appears in columns: {mane_columns}. "
                "Trying to extract from these columns.",
                UserWarning
            )
            # Create a combined tag column
            gtf = gtf.copy()
            gtf['_combined_tags'] = gtf[mane_columns].apply(
                lambda row: ';'.join(row.dropna().astype(str)), axis=1
            )
            tag_column = '_combined_tags'
        else:
            raise MissingColumnError(
                missing_columns=['tag'],
                available_columns=list(gtf.columns),
                operation='filter_to_mane_select_transcript'
            )
    
    # Filter for MANE Select transcripts
    mask = gtf[tag_column].astype(str).str.contains('MANE_Select', case=False, na=False)
    filtered_gtf = gtf[mask].copy()
    
    if filtered_gtf.empty:
        # Try alternative MANE tags
        alternative_patterns = ['MANE_select', 'MANE Select', 'MANE-Select']
        for pattern in alternative_patterns:
            mask = gtf[tag_column].astype(str).str.contains(pattern, case=False, na=False)
            if mask.any():
                filtered_gtf = gtf[mask].copy()
                warnings.warn(
                    f"Found MANE transcripts using alternative pattern: '{pattern}'",
                    UserWarning
                )
                break
        
        if filtered_gtf.empty:
            raise ValueError(
                "No MANE_Select transcripts found in the GTF.",
                context={
                    'tag_column_used': tag_column,
                    'sample_tags': gtf[tag_column].dropna().unique()[:5].tolist() if tag_column in gtf.columns else []
                }
            )
    
    # Clean up temporary column if created
    if tag_column == '_combined_tags' and '_combined_tags' in filtered_gtf.columns:
        filtered_gtf = filtered_gtf.drop(columns=['_combined_tags'])
    
    filtered_gtf.attrs['mane_filter_applied'] = True
    filtered_gtf.attrs['mane_transcripts_found'] = len(filtered_gtf['transcript_id'].unique())
    
    return filtered_gtf


def filter_transcript_support_level(
    gtf: pd.DataFrame,
    transcript_support_levels: Union[str, Sequence[str]],
) -> pd.DataFrame:
    """
    Filter GTF to only transcripts with specific GENCODE support levels.
    
    Transcript Support Level (TSL) indicates the degree of evidence used to
    construct the transcript. Lower numbers indicate higher quality evidence.
    
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
        InvalidInputError: If invalid support levels are provided.
        
    Example:
        >>> high_quality = filter_transcript_support_level(gtf_data, ['1', '2'])
    """
    _validate_gtf_dataframe(
        gtf,
        required_columns=['transcript_support_level'],
        operation='filter_transcript_support_level'
    )
    
    tsl_levels = _validate_tsl_levels(transcript_support_levels)
    
    # Filter for specified TSL levels
    mask = gtf['transcript_support_level'].isin(tsl_levels)
    filtered_gtf = gtf[mask].copy()
    
    if filtered_gtf.empty:
        warnings.warn(
            f"No transcripts found with TSL levels: {tsl_levels}. "
            f"Available TSL levels: {gtf['transcript_support_level'].dropna().unique().tolist()}",
            UserWarning
        )
    
    filtered_gtf.attrs['tsl_filter_applied'] = True
    filtered_gtf.attrs['tsl_levels'] = tsl_levels
    filtered_gtf.attrs['transcripts_retained'] = len(filtered_gtf)
    
    return filtered_gtf


def _strip_ensembl_version(id_series: pd.Series) -> pd.Series:
    """
    Strip version/patch suffix from Ensembl IDs.
    
    Handles formats:
    - ENSG00000123456.1 -> ENSG00000123456
    - ENSG00000123456.1_PAR_Y -> ENSG00000123456_PAR_Y
    
    Args:
        id_series: Series of Ensembl IDs.
        
    Returns:
        Series of IDs with version stripped.
        
    Raises:
        InvalidIDFormatError: If any ID doesn't contain expected format.
    """
    if id_series.empty:
        return id_series
    
    # Make a copy to avoid modifying original
    ids = id_series.copy().astype(str).str.strip()
    
    # Handle IDs without versions gracefully
    def strip_single_id(id_str: str) -> str:
        if pd.isna(id_str):
            return id_str
        
        # Check if it looks like an Ensembl ID
        if not re.match(r'^ENS[TG]\d', id_str):
            return id_str  # Return as-is for non-Ensembl IDs
        
        # Split on first dot
        parts = id_str.split('.', 1)
        if len(parts) == 1:
            return id_str  # No version to strip
        
        base_id = parts[0]
        remainder = parts[1]
        
        # Check for and preserve PAR suffix
        if '_PAR_' in remainder:
            # Keep everything after version number before _PAR_
            par_parts = remainder.split('_PAR_', 1)
            if len(par_parts) == 2:
                return f"{base_id}_PAR_{par_parts[1]}"
        
        return base_id
    
    return ids.apply(strip_single_id)


def upgrade_annotation_ids(
    old_ids: pd.Series, 
    new_ids: pd.Series, 
    patchless: bool = False,
    strict: bool = True
) -> pd.Series:
    """
    Upgrade or add transcript id patch version to Ensembl IDs.
    
    This function maps old Ensembl IDs to new versions by stripping version
    numbers and matching on the base ID.
    
    Args:
        old_ids: A pd.Series of Ensembl transcript or gene ids with older or
            missing version/patch numbers.
        new_ids: A pd.Series of transcript or gene ids with newer version/patch
            numbers.
        patchless: If True, assumes both old_ids and new_ids are already
            patchless. If False, strips versions from both before matching.
        strict: If True, raises errors for duplicates and unmapped IDs.
            If False, issues warnings instead.
            
    Returns:
        A pd.Series with the same index as old_ids, containing the upgraded IDs.
        Unmapped IDs will be NaN if strict=False.
        
    Raises:
        EmptyDataError: If old_ids or new_ids are empty.
        InvalidIDFormatError: If IDs don't have expected format when patchless=False.
        ValueError: If duplicate IDs are found after stripping versions (strict mode).
        
    Example:
        >>> old = pd.Series(['ENSG00000141510.1', 'ENSG00000141510.2'])
        >>> new = pd.Series(['ENSG00000141510.17', 'ENSG00000141510.18'])
        >>> upgraded = upgrade_annotation_ids(old, new)
        >>> print(upgraded)
    """
    # Input validation
    if old_ids.empty:
        raise EmptyDataError("old_ids series")
    if new_ids.empty:
        raise EmptyDataError("new_ids series")
    
    # Store original index
    original_index = old_ids.index
    
    # Remove duplicates while preserving order
    old_ids_dedup = old_ids[~old_ids.duplicated()]
    new_ids_dedup = new_ids[~new_ids.duplicated()]
    
    if patchless:
        old_ids_nopatch = old_ids_dedup.astype(str).str.strip()
        new_ids_nopatch = new_ids_dedup.astype(str).str.strip()
    else:
        # Try to strip versions, but be lenient about format
        old_ids_nopatch = _strip_ensembl_version(old_ids_dedup)
        new_ids_nopatch = _strip_ensembl_version(new_ids_dedup)
    
    # Check for duplicates in version-stripped IDs
    old_duplicates = old_ids_nopatch[old_ids_nopatch.duplicated()]
    new_duplicates = new_ids_nopatch[new_ids_nopatch.duplicated()]
    
    if not old_duplicates.empty:
        error_msg = (
            "old_ids not unique after stripping version. "
            f"Duplicates: {old_ids_dedup[old_ids_nopatch.duplicated()].tolist()}"
        )
        if strict:
            raise ValueError(error_msg)
        else:
            warnings.warn(error_msg, UserWarning)
            # Keep first occurrence of duplicates
            old_ids_nopatch = old_ids_nopatch[~old_ids_nopatch.duplicated()]
            old_ids_dedup = old_ids_dedup[old_ids_nopatch.index]
    
    if not new_duplicates.empty:
        error_msg = (
            "new_ids not unique after stripping version. "
            f"Duplicates: {new_ids_dedup[new_ids_nopatch.duplicated()].tolist()}"
        )
        if strict:
            raise ValueError(error_msg)
        else:
            warnings.warn(error_msg, UserWarning)
            # Keep first occurrence of duplicates
            new_ids_nopatch = new_ids_nopatch[~new_ids_nopatch.duplicated()]
            new_ids_dedup = new_ids_dedup[new_ids_nopatch.index]
    
    # Create DataFrames for merging
    old_df = pd.DataFrame({
        'old_original': old_ids_dedup.values,
        'old_stripped': old_ids_nopatch.values
    })
    
    new_df = pd.DataFrame({
        'new_original': new_ids_dedup.values,
        'new_stripped': new_ids_nopatch.values
    })
    
    # Merge on stripped IDs
    merged = pd.merge(
        old_df, 
        new_df, 
        left_on='old_stripped', 
        right_on='new_stripped', 
        how='left'
    )
    
    # Create mapping dictionary
    mapping = dict(zip(merged['old_original'], merged['new_original']))
    
    # Apply mapping to original old_ids (preserving order and duplicates)
    result = old_ids.map(mapping)
    
    # Check for unmapped IDs
    unmapped_mask = result.isna()
    if unmapped_mask.any() and strict:
        unmapped_ids = old_ids[unmapped_mask].unique().tolist()
        raise ValueError(
            f"Could not map {unmapped_mask.sum()} IDs to new version. "
            f"Unmapped IDs: {unmapped_ids[:10]}",
            context={
                'unmapped_count': unmapped_mask.sum(),
                'total_ids': len(old_ids),
                'mapping_rate': f"{100*(1 - unmapped_mask.mean()):.1f}%"
            }
        )
    elif unmapped_mask.any():
        warnings.warn(
            f"{unmapped_mask.sum()} IDs could not be mapped to new version",
            UserWarning
        )
    
    # Restore original index
    result.index = original_index
    
    return result


def get_gene_intervals(
    gtf: pd.DataFrame,
    gene_symbols: Optional[Sequence[str]] = None,
    gene_ids: Optional[Sequence[str]] = None,
    require_strand: bool = True,
    allow_multiple: bool = False
) -> List[Dict[str, Any]]:
    """
    Returns genomic intervals for the given gene identifiers.
    
    Args:
        gtf: pd.DataFrame of GTF entries. Must contain columns 'Feature',
            'gene_name', 'gene_id', 'Chromosome', 'Start', 'End', and 'Strand'.
        gene_symbols: A sequence of gene names or gene symbols 
            (e.g., ['EGFR', 'TNF', 'TP53']). Matching is case-insensitive.
        gene_ids: A sequence of Ensembl gene IDs, which can be patched or
            unpatched. Matching is done on unpatched IDs.
        require_strand: Whether to require strand information.
        allow_multiple: If True, returns all matches for ambiguous genes.
            If False, raises AmbiguousGeneError.
            
    Returns:
        A list of dictionaries with interval information for each gene.
        Each dictionary contains: chromosome, start, end, strand, gene_id,
        gene_name, and source_row (the original DataFrame row as dict).
        
    Raises:
        InvalidInputError: If inputs are invalid.
        GeneNotFoundError: If gene identifiers are not found.
        AmbiguousGeneError: If multiple intervals found and allow_multiple=False.
        MissingColumnError: If required columns are missing.
        
    Example:
        >>> intervals = get_gene_intervals(gtf_data, gene_symbols=['TP53', 'EGFR'])
        >>> for interval in intervals:
        ...     print(f"Gene: {interval['gene_name']}, "
        ...           f"Location: {interval['chromosome']}:{interval['start']}-{interval['end']}")
    """
    # Validate inputs
    id_type, input_ids = _validate_gene_identifiers(gene_symbols, gene_ids)
    
    # Validate GTF structure
    required_cols = ['Feature', 'Chromosome', 'Start', 'End', 'gene_id', 'gene_name']
    if require_strand:
        required_cols.append('Strand')
    
    _validate_gtf_dataframe(
        gtf,
        required_columns=required_cols,
        operation='get_gene_intervals'
    )
    
    # Filter to gene entries
    gene_mask = gtf['Feature'] == 'gene'
    if not gene_mask.any():
        raise ValueError(
            "No gene entries found in GTF",
            context={'available_features': gtf['Feature'].unique().tolist()}
        )
    
    gtf_genes = gtf[gene_mask].copy()
    
    # Prepare identifiers for matching
    input_series = pd.Series(input_ids, dtype=str)
    
    # Define processing function based on identifier type
    if id_type == 'gene_name':
        # Case-insensitive matching for gene symbols
        input_processed = input_series.str.upper().str.strip()
        gtf_genes['processed_id'] = gtf_genes['gene_name'].astype(str).str.upper().str.strip()
        original_id_col = 'gene_name'
    else:  # gene_id
        # Strip version from gene IDs for matching
        def strip_gene_id(gid: str) -> str:
            if pd.isna(gid):
                return gid
            parts = str(gid).split('.', 1)
            return parts[0].strip()
        
        input_processed = input_series.apply(strip_gene_id)
        gtf_genes['processed_id'] = gtf_genes['gene_id'].apply(strip_gene_id)
        original_id_col = 'gene_id'
    
    # Filter to genes in input
    mask = gtf_genes['processed_id'].isin(input_processed)
    gtf_subset = gtf_genes[mask].copy()
    
    if gtf_subset.empty:
        # Try to provide helpful error message
        sample_available = gtf_genes[original_id_col].dropna().unique()[:10].tolist()
        raise GeneNotFoundError(
            f"No genes found for identifiers: {input_ids[:5]}...",
            searched_columns=[original_id_col]
        )
    
    # Group by processed ID to check for duplicates
    grouped = gtf_subset.groupby('processed_id')
    
    intervals = []
    unmatched_ids = []
    
    for input_id, processed_id in zip(input_ids, input_processed):
        if processed_id not in grouped.groups:
            unmatched_ids.append(input_id)
            continue
        
        group_indices = grouped.groups[processed_id]
        group_size = len(group_indices)
        
        if group_size > 1 and not allow_multiple:
            # Multiple matches found
            matching_rows = gtf_genes.loc[group_indices]
            found_entries = matching_rows[original_id_col].unique().tolist()
            raise AmbiguousGeneError(
                gene_identifier=input_id,
                found_entries=found_entries
            )
        
        # Handle multiple matches if allowed
        for idx in (group_indices if allow_multiple else [group_indices[0]]):
            row = gtf_genes.loc[idx]
            
            interval = {
                'chromosome': str(row['Chromosome']),
                'start': int(row['Start']),
                'end': int(row['End']),
                'strand': row['Strand'] if 'Strand' in row else None,
                'gene_id': str(row['gene_id']),
                'gene_name': str(row['gene_name']) if pd.notna(row['gene_name']) else None,
                'original_identifier': input_id,
                'identifier_type': id_type,
                'source_row': row.to_dict()
            }
            
            # Add additional metadata if available
            for col in ['gene_type', 'gene_biotype', 'gene_source']:
                if col in row and pd.notna(row[col]):
                    interval[col] = str(row[col])
            
            intervals.append(interval)
    
    # Report unmatched genes
    if unmatched_ids:
        warnings.warn(
            f"Could not find intervals for {len(unmatched_ids)} genes: {unmatched_ids[:10]}",
            UserWarning
        )
    
    return intervals


def get_gene_interval(
    gtf: pd.DataFrame,
    gene_symbol: Optional[str] = None,
    gene_id: Optional[str] = None,
    require_strand: bool = True
) -> Dict[str, Any]:
    """
    Returns genomic interval for a single gene identifier.
    
    Args:
        gtf: pd.DataFrame of GTF entries.
        gene_symbol: A gene name or gene symbol (e.g., 'EGFR', 'TNF', 'TP53').
        gene_id: An Ensembl gene ID, which can be patched or unpatched.
        require_strand: Whether to require strand information.
            
    Returns:
        A dictionary with interval information for the gene.
        
    Raises:
        InvalidInputError: If neither or both gene_symbol and gene_id are set.
        GeneNotFoundError: If gene identifier is not found.
        AmbiguousGeneError: If multiple intervals found.
        
    Example:
        >>> tp53_interval = get_gene_interval(gtf_data, gene_symbol='TP53')
        >>> print(f"TP53: {tp53_interval['chromosome']}:"
        ...       f"{tp53_interval['start']}-{tp53_interval['end']}")
    """
    if sum(x is not None for x in [gene_symbol, gene_id]) != 1:
        raise InvalidInputError(
            "Exactly one of gene_symbol or gene_id must be provided",
            context={'gene_symbol': gene_symbol, 'gene_id': gene_id}
        )
    
    intervals = get_gene_intervals(
        gtf,
        gene_symbols=[gene_symbol] if gene_symbol else None,
        gene_ids=[gene_id] if gene_id else None,
        require_strand=require_strand,
        allow_multiple=False
    )
    
    if not intervals:
        identifier = gene_symbol or gene_id
        raise GeneNotFoundError(
            f"Gene not found: {identifier}",
            searched_columns=['gene_name' if gene_symbol else 'gene_id']
        )
    
    return intervals[0]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_gtf(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a GTF file into a pandas DataFrame.
    
    Args:
        filepath: Path to the GTF file.
        nrows: Number of rows to read (for testing).
        
    Returns:
        pd.DataFrame with GTF data.
        
    Example:
        >>> gtf_data = load_gtf('gencode.v44.annotation.gtf.gz')
    """
    # GTF column names
    gtf_columns = [
        'Chromosome', 'Source', 'Feature', 'Start', 'End', 
        'Score', 'Strand', 'Frame', 'Attributes'
    ]
    
    try:
        gtf = pd.read_csv(
            filepath,
            sep='\t',
            comment='#',
            header=None,
            names=gtf_columns,
            low_memory=False,
            nrows=nrows
        )
    except Exception as e:
        raise IOError(f"Failed to load GTF file: {filepath}") from e
    
    # Parse attributes column
    gtf = _parse_gtf_attributes(gtf)
    
    return gtf


def _parse_gtf_attributes(gtf: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Attributes column of a GTF into separate columns.
    
    Args:
        gtf: DataFrame with 'Attributes' column.
        
    Returns:
        DataFrame with parsed attributes as separate columns.
    """
    if 'Attributes' not in gtf.columns:
        return gtf
    
    def parse_attributes(attr_str: str) -> Dict[str, str]:
        """Parse GTF attribute string into dictionary."""
        if pd.isna(attr_str):
            return {}
        
        attributes = {}
        # Split by semicolon, handling quoted values
        parts = [p.strip() for p in attr_str.split(';') if p.strip()]
        
        for part in parts:
            if ' ' in part:
                # Standard format: key "value"
                key, value = part.split(' ', 1)
                # Remove quotes if present
                value = value.strip('"')
                attributes[key] = value
            elif '=' in part:
                # Alternative format: key=value
                key, value = part.split('=', 1)
                attributes[key] = value
        
        return attributes
    
    # Parse all attributes
    parsed_attrs = gtf['Attributes'].apply(parse_attributes)
    
    # Extract common attributes
    common_keys = set()
    for attrs in parsed_attrs:
        common_keys.update(attrs.keys())
    
    # Create columns for common attributes
    for key in common_keys:
        gtf[key] = parsed_attrs.apply(lambda x: x.get(key))
    
    return gtf


def validate_gtf_structure(gtf: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate GTF structure and provide summary statistics.
    
    Args:
        gtf: GTF DataFrame to validate.
        
    Returns:
        Dictionary with validation results and statistics.
        
    Example:
        >>> stats = validate_gtf_structure(gtf_data)
        >>> print(f"Features: {stats['features']}")
        >>> print(f"Genes: {stats['gene_count']}")
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        # Basic structure validation
        required_cols = ['Chromosome', 'Feature', 'Start', 'End', 'Strand']
        missing = [col for col in required_cols if col not in gtf.columns]
        
        if missing:
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Missing required columns: {missing}"
            )
        
        # Statistics
        stats = {
            'total_rows': len(gtf),
            'features': gtf['Feature'].value_counts().to_dict() if 'Feature' in gtf.columns else {},
            'chromosomes': gtf['Chromosome'].nunique() if 'Chromosome' in gtf.columns else 0,
            'strand_distribution': gtf['Strand'].value_counts().to_dict() if 'Strand' in gtf.columns else {},
        }
        
        # Gene/transcript counts if available
        if 'gene_id' in gtf.columns:
            stats['gene_count'] = gtf['gene_id'].nunique()
        if 'transcript_id' in gtf.columns:
            stats['transcript_count'] = gtf['transcript_id'].nunique()
        
        validation_results['statistics'] = stats
        
        # Data quality checks
        if 'Start' in gtf.columns and 'End' in gtf.columns:
            invalid_intervals = gtf[gtf['End'] < gtf['Start']]
            if not invalid_intervals.empty:
                validation_results['warnings'].append(
                    f"Found {len(invalid_intervals)} intervals with End < Start"
                )
        
        # Check for common issues
        if gtf['Chromosome'].astype(str).str.contains('chr', case=False).any():
            validation_results['warnings'].append(
                "Chromosome names contain 'chr' prefix"
            )
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Validation error: {str(e)}")
    
    return validation_results


# ============================================================================
# MAIN MODULE EXPORTS
# ============================================================================

__all__ = [
    # Core classes
    'TranscriptType',
    
    # Exceptions
    'GTFAnnotationError',
    'MissingColumnError',
    'AmbiguousGeneError',
    'GeneNotFoundError',
    'InvalidInputError',
    'InvalidIDFormatError',
    'EmptyDataError',
    
    # Main functions
    'extract_tss',
    'filter_transcript_type',
    'filter_protein_coding',
    'filter_to_longest_transcript',
    'filter_to_mane_select_transcript',
    'filter_transcript_support_level',
    'upgrade_annotation_ids',
    'get_gene_intervals',
    'get_gene_interval',
    
    # Helper functions
    'load_gtf',
    'validate_gtf_structure',
]

# Version information
__version__ = '1.0.0'
__author__ = 'Google LLC'
__license__ = 'Apache 2.0'
