# Migrating from AlphaGenome 1

Both models are available through the `alphagenome` package:

```python
from alphagenome.models.v2 import dna_client  # AlphaGenome 2 (recommended)
from alphagenome.models.v1 import dna_client as dna_client_v1  # AlphaGenome 1
```

## When to use each model

*   **Use AlphaGenome 2** for most analyses: predicting tracks or scoring
    variants in a reference condition (`condition=`) or under expression
    conditioning on your own human scRNA-seq count matrix
    (`cell_conditioning=`), predicting single-cell accessibility (`SC_ATAC`),
    denoised single-cell gene expression (`SC_GENE_EXPRESSION_DENOISED`),
    protocol-specific single-cell gene expression (`SC_GENE_EXPRESSION`), or CpG
    methylation (`METHYLATION`), computing input gradients for enhancer-to-gene
    (E2G) attribution (`predict_interval_gradients`), encoding single-cell
    profiles (`encode_cells`), or using cross-modality track imputation.
*   **Use AlphaGenome 1** to reproduce results from the AlphaGenome paper
    {cite:p}`alphagenome`, score variants across all tissues and cell types at
    once without choosing a condition, or predict PRO-cap tracks.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->
```{figure} /_static/model_overview_v2.png
:width: 100%
:alt: Overview of AlphaGenome 2
:name: ag2-overview-figure

**AlphaGenome 2 architecture.** Takes a DNA sequence of up to 1 Mb, the species,
and a condition (a reference condition, or a cell embedding from the cell
encoder under expression conditioning) to predict observed and imputed coverage
tracks, gene expression, splicing, and contact maps.
```

```{figure} /_static/model_overview.png
:width: 100%
:alt: Overview of AlphaGenome 1
:name: ag1-overview-figure

**AlphaGenome 1 architecture.** Takes a DNA sequence of up to 1 Mb and predicts
observed multi-task coverage tracks, splicing, and pairwise contact maps across
human and mouse assays.
```
<!-- mdformat on -->

## API migration reference

Both models share the same core prediction and scoring methods; the differences
are in how you choose conditions, discover metadata, and read scores.

### Conditions

```{list-table}
:header-rows: 1

*   -
    - AlphaGenome 1
    - AlphaGenome 2
*   - Predict or score one variant or interval
    - `predict_*`: optional `ontology_terms` list (pass `None` for all tissues
      and cell types); `score_*`: no `ontology_terms` argument (all tissues and
      cell types are always scored)
    - Required: one reference condition (`condition=`) **or** expression
      conditioning on your own human scRNA-seq counts (`cell_conditioning=`
      plus a `cell_condition_name=` label; up to 256 cells per call, raw unique
      molecular identifier (UMI) counts with Ensembl IDs)
*   - Batch methods (`predict_sequences`, `predict_intervals`,
      `predict_variants`, `score_intervals`, `score_variants`, and in
      AlphaGenome 2 `predict_interval_gradients_batch`)
    - `predict_*`: optional `ontology_terms` list (pass `None` for all tissues
      and cell types) applied across all items; `score_*`: no `ontology_terms`
      argument (`predict_interval_gradients_batch` not available)
    - One reference condition (`conditions=`) **or** expression-conditioning
      input (`cell_conditionings=` plus `cell_condition_names=`) per item.
      Pass `conditions=` and `variant_scorers=` as keywords when porting
      AlphaGenome 1 `score_variants` calls.
*   - Condition format
    - Single ontology CURIE
    - Composite condition (ontology CURIEs joined by underscores), single
      CURIE, or a `ModelCondition`
```

### Metadata

```{list-table}
:header-rows: 1

*   -
    - AlphaGenome 1
    - AlphaGenome 2
*   - What can I predict?
    - `dna_model.output_metadata()`
    - `dna_model.supported_conditions()`, one row per condition
*   - Which tracks were measured?
    - Same table
    - `dna_model.observed_metadata()`
*   - Biosample column
    - `ontology_curie`
    - `biosample_curie`, plus an `imputation_mode` for every track
```

### Output types

**Added in AlphaGenome 2** (see
{doc}`Model output metadata <../exploring_model_metadata>` for descriptions):

*   `SC_GENE_EXPRESSION_DENOISED`: denoised single-cell gene expression under
    `cell_conditioning=`, one channel per strand in CPM
*   `SC_GENE_EXPRESSION`
*   `SC_ATAC`
*   `METHYLATION`
*   `CELL_EMBEDDINGS`

**Changed in AlphaGenome 2**

*   `SPLICE_SITES`: the 4 condition-agnostic tracks of AlphaGenome 1 are now
    condition-specific, with 5 channels per condition (`donor +`, `acceptor +`,
    `donor -`, `acceptor -`, `padding .`).
*   `CONTACT_MAPS`: targets are not coarse-grained, so values are not directly
    comparable to AlphaGenome 1 contact maps.

**Removed in AlphaGenome 2**

*   `PROCAP`: use `CAGE` for transcription start site activity.

### Variant scores

```{list-table}
:header-rows: 1

*   -
    - AlphaGenome 1
    - AlphaGenome 2
*   - Score columns
    - `raw_score` and `quantile_score`
    - `raw_score` only (no `quantile_score` column)
```

<!-- mdformat off(Retain MyST dropdown syntax.) -->
:::{dropdown} How empirical quantile scores work in AlphaGenome 1
:name: quantile-scores-ag1

Different output types yield raw variant scores on different scales (for
example, splice site usage scores lie in $[0, 1]$ whereas RNA-seq log-fold
change scores are signed and not restricted to that range). AlphaGenome 1
(`alphagenome.models.v1`) estimates a background distribution for each
recommended scorer and track across ~300,000 common variants (MAF > 0.01 in any
gnomAD v3 population; see {cite:p}`alphagenome`) and converts each raw score
into a percentile rank bounded by ±0.99999.

For signed variant scorers, $[0, 1]$ rank probabilities are linearly mapped to
$[-1, 1]$ so that negative scores map near $-1$, scores near zero map to $0$,
and strongly positive scores map near $+1$.
:::
<!-- mdformat on -->

## Key behavioral differences

*   **One condition per call:** With AlphaGenome 2, each call to
    `predict_sequence`, `predict_interval`, `predict_variant`, or
    `score_variant` predicts for one condition; loop over conditions when
    comparing multiple tissues or cell types (or pass one reference condition or
    expression-conditioning input per item to the batch methods).
*   **Complete channel output with per-track imputation labels:** For every
    condition, AlphaGenome 2 predicts all channels across requested output types
    and labels each track in `.metadata['imputation_mode']` as `observed`,
    `imputed_supported`, or `imputed_exploratory` (see
    [Navigating imputed tracks](imputation)).
