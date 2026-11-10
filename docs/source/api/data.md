# Data

Classes and utilities for manipulating genomics data.

```{eval-rst}
.. currentmodule:: alphagenome
```

## Fold intervals

```{eval-rst}

.. autosummary::
    :toctree: generated

    data.fold_intervals.Subset
    data.fold_intervals.get_all_folds
    data.fold_intervals.get_fold_names
    data.fold_intervals.get_fold_intervals
```

## Genome

```{eval-rst}

.. autosummary::
    :toctree: generated

    data.genome.Strand
    data.genome.VariantFormat
    data.genome.Interval
    data.genome.Variant
    data.genome.Junction
    data.genome.normalize_variant
    data.genome.intersect_intervals
    data.genome.merge_overlapping_intervals
    data.genome.union_intervals
```

## Gene annotation

```{eval-rst}

.. autosummary::
    :toctree: generated

    data.gene_annotation.TranscriptType
    data.gene_annotation.extract_tss
    data.gene_annotation.filter_transcript_type
    data.gene_annotation.filter_protein_coding
    data.gene_annotation.filter_to_longest_transcript
    data.gene_annotation.filter_to_mane_select_transcript
    data.gene_annotation.filter_transcript_support_level
    data.gene_annotation.get_gene_interval
    data.gene_annotation.get_gene_intervals
```

## FASTA Extractor

```{eval-rst}

.. autosummary::
    :toctree: generated

    io.fasta.FastaExtractor
    io.fasta.reverse_complement
```

## Junction data

```{eval-rst}

.. autosummary::
    :toctree: generated

    data.junction_data.JunctionData
    data.junction_data.get_junctions_to_plot
```

## Ontology

```{eval-rst}

.. autosummary::
    :toctree: generated

    data.ontology.OntologyType
    data.ontology.OntologyTerm
    data.ontology.from_curie
    data.ontology.from_curies
```

## Track data

```{eval-rst}

.. autosummary::
    :toctree: generated

    data.track_data.AggregationType
    data.track_data.TrackData
    data.track_data.concat
    data.track_data.interleave
```

## Gene expression data

```{eval-rst}

.. autosummary::
    :toctree: generated

    data.gene_expression_data.GeneExpressionData
    data.gene_expression_data.concat
```

## Transcript

```{eval-rst}

.. autosummary::
    :toctree: generated

    data.transcript.Transcript
    data.transcript.TranscriptExtractor
```
