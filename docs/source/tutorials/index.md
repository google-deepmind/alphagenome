# Tutorials

```{toctree}
:maxdepth: 1
:hidden:

../colabs/v2/choosing_a_condition
../colabs/v2/selecting_a_genomic_region
../colabs/v2/predict_interval
../colabs/v2/predict_variant
../colabs/v2/score_variant
../colabs/v2/plotting_predictions
../colabs/v2/conditioning_on_single_cell_data
../colabs/v2/imputing_unmeasured_tracks
../colabs/v2/splicing_variant
../colabs/v2/haplotypes
../colabs/v2/ISM
../colabs/v2/mouse_predictions
../colabs/v1/quick_start
../colabs/v1/deriving_PSI_values
```

Tutorials are hands-on notebooks. The core workflow teaches the AlphaGenome 2
API in order: choose a condition, select a genomic region, predict, and score.
Each topic tutorial then covers one capability. For end-to-end biological
analyses, see {doc}`Vignettes <../vignettes/index>`.

## Core workflow

<!-- mdformat off(Turn off mdformat to retain grid card syntax.) -->

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} 1. Choosing a condition
:link: ../colabs/v2/choosing_a_condition
:link-type: doc

Search `supported_conditions`, check which output types are observed or
imputed for a condition, and see when to use expression conditioning instead.
:::

:::{grid-item-card} 2. Selecting a genomic region
:link: ../colabs/v2/selecting_a_genomic_region
:link-type: doc

Construct and resize `Interval` and `Variant` inputs around coordinates,
GENCODE genes, or variants.
:::

:::{grid-item-card} 3. Predicting tracks for an interval or sequence
:link: ../colabs/v2/predict_interval
:link-type: doc

Predict bulk and pseudobulk tracks for a genomic interval, and bulk tracks for
a custom DNA sequence.
:::

:::{grid-item-card} 4. Predicting a variant
:link: ../colabs/v2/predict_variant
:link-type: doc

Predict `REF` and `ALT` bulk and single-cell outputs for a genetic variant and
plot them together.
:::

:::{grid-item-card} 5. Scoring a variant
:link: ../colabs/v2/score_variant
:link-type: doc

Quantify variant effects with recommended and custom variant scorers, and
flatten the results with `tidy_scores`.
:::

::::

## Topics

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} Plotting predictions
:link: ../colabs/v2/plotting_predictions
:link-type: doc

Compose multi-panel plots across output types: bulk coverage, single-cell gene
expression, chromatin accessibility, histone and TF ChIP-seq, and contact maps.
:::

:::{grid-item-card} Conditioning on your own single-cell data
:link: ../colabs/v2/conditioning_on_single_cell_data
:link-type: doc

Condition predictions and scores on your own `AnnData` single-cell gene
expression counts.
:::

:::{grid-item-card} Working with imputed tracks
:link: ../colabs/v2/imputing_unmeasured_tracks
:link-type: doc

Check which tracks are observed or imputed for a condition, filter by
`imputation_mode`, and label imputed tracks in plots.
:::

:::{grid-item-card} Analyzing a splicing variant
:link: ../colabs/v2/splicing_variant
:link-type: doc

Predict, plot, and score splice site, splice site usage, and splice junction
changes for a variant.
:::

:::{grid-item-card} Analyzing haplotypes
:link: ../colabs/v2/haplotypes
:link-type: doc

Predict and compare combined multi-variant haplotype effects against individual
single-variant predictions.
:::

:::{grid-item-card} Highlighting important regions with ISM
:link: ../colabs/v2/ISM
:link-type: doc

Run *in silico* mutagenesis (ISM) across an interval to find the bases that
drive a prediction.
:::

:::{grid-item-card} Making mouse predictions
:link: ../colabs/v2/mouse_predictions
:link-type: doc

Load mouse gene annotations and make predictions for the mouse genome.
:::

::::

## AlphaGenome 1

AlphaGenome 1 is the original model described in the AlphaGenome paper
{cite:p}`alphagenome`. See {doc}`Getting started <../getting_started>` for when
to use it and {doc}`Migrating from AlphaGenome 1 <../user_guides/ag1_vs_ag2>`
for how it differs from AlphaGenome 2.

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} AlphaGenome 1 quick start (legacy)
:link: ../colabs/v1/quick_start
:link-type: doc

Browse AlphaGenome 1 metadata, then predict and score a variant across all
observed tracks.
:::

:::{grid-item-card} Deriving PSI values from splice junctions
:link: ../colabs/v1/deriving_PSI_values
:link-type: doc

Compute percent spliced in (PSI5 and PSI3) values from predicted splice
junction counts.
:::

::::

<!-- mdformat on -->
