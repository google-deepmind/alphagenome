# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0]

### Added

-   Add AlphaGenome Atlas API for programmatic access to Atlas scores.

## [0.8.0]

### Added

-   Color argument to sashimi plot component.
-   Tutorials on haplotype analysis and deriving PSI values.
-   Helper methods `with_name_suffix` and `with_metadata_column` to track data.

### Changed

-   Sped up `plot_transcripts` by removing redundant drawing passes.
-   Code cleanup and improved type hints, generating .pyi files for protocol
    buffers.

## [0.7.0]

### Added

-   Fasta extractor for reading DNA sequences.
-   Colab to demonstrate how to score splicing variants.

### Changed

-   Merge stranded tracks for gene-centric scoring.
-   Better transcript arrow plotting.

## [0.6.1]

### Changed

-   Speed up seqlogo plotting 100x.
-   Filter splice junctions when calling `filter_by_strand` of model Output
    dataclass. Previously only tracks would be correctly filtered.
-   When plotting junctions, don't calculate k-threshold if there are no
    junctions to plot.

### Fixed

-   `IndexError` when plotting annotations with no labels.
-   `KeyError` when tidying variant scores with no values.

## [0.6.0]

### Added

-   Support for making interval scoring requests with no center mask.
-   Additional `Variant` properties for if the variant is an indel, has
    frameshift or is a structured variant.
-   Options to set y-axis bounds, ticks and tick labels.
-   Support for processing splice sites from GTF.

### Changed

-   Reduce peak memory usage by 50% when unpacking tensors.
-   Update citation to official Nature paper.

## [0.5.1]

### Changed

-   Move ModelVersion enum to `dna_model` base class.
-   Add less-than operator to Organism enum.
-   Make `OutputMetadata` keyword-only, to better support derived classes.

### Removed

-   Remove conversion from NumPy scalar to int in `Interval`.

## [0.5.0]

### Added

-   Support for performing in-silico mutagenesis (ISM) on the alternate allele.
    This enables the reproduction of e.g.
    [Figure 4b in our pre-print](https://doi.org/10.1101/2025.06.25.661532).
-   Add missing extended columns when calling `tidy_anndata`.

### Removed

-   Remove support for 2kb DNA sequence lengths. This is due to AlphaGenome not
    performing well with very short sequence lengths (see
    [Figure 7 of our pre-print](https://doi.org/10.1101/2025.06.25.661532) for
    details).

## [0.4.0]

### Added

-   Add `filter_to_mane_select_transcript` to subset a GENCODE GTF to include
    only entries corresponding to MANE select transcripts.
-   Add `from_outputs` class method for creating `OutputMetadata` object from a
    set of outputs.

### Changed

-   Update GTF processing script to include duplicate attributes and support
    downloading source GTF from a URL.

## [0.3.0]

### Added

-   Add `get_gene_intervals` to retrieve multiple gene intervals.
-   Implement `__getitem__` on `TrackData` to generalize filter/slice methods.
-   Add `normalize_variant` function to normalize variants with the underlying
    assembly.
-   Add missing "Assay title", "data_source" and "biosample" columns to splice
    junction metadata.
-   Add splice junction section to API docs.

### Changed

-   Update quick start notebook to not use shorter, less performant sequence
    lengths.
-   Update documentation on ChIP-TF and Histone units.
-   Move some protocol buffer conversion functions from data to models
    directory.
-   Include link in README license section to API terms.

## [0.2.0]

### Added

-   Add `is_insertion` and `is_deletion` properties to `Variant`.
-   Add `DnaModel` abstract base class.
-   Add support for center mask scoring over the entire sequence by passing
    `None` for width.

### Changed

-   Move RPC requests and responses to `dna_model_service.proto`.
-   Move functionality to convert `TrackData` to/from protocol buffers to
    utility module.

## [0.1.0]

### Added

-   Add `L2_DIFF_LOG1P` variant scoring aggregation type.
-   Add `is_snv` property to `Variant`.
-   Add non-zero mean track metadata field to model output metadata.
-   Add optional interval argument to `predict_sequence`.

## [0.0.2]

### Added

-   `colab_utils` module to wrap reading API keys from environment variables or
    Google Colab secrets.

## [0.0.1]

Initial release.
