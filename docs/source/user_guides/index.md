# User guide

Concepts behind AlphaGenome inputs, outputs, imputed tracks, and variant
scoring.

```{toctree}
:maxdepth: 1
:hidden:

../colabs/v2/key_objects
../exploring_model_metadata
imputation
../variant_scoring
../visualization_basics
ag1_vs_ag2
```

<!-- mdformat off(Turn off mdformat to retain grid card syntax.) -->

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Key objects
:link: ../colabs/v2/key_objects
:link-type: doc

Intervals, variants, prediction containers (`TrackData`, `GeneExpressionData`,
`InputGradientsOutput`), variant-score `AnnData` objects, and a glossary of core
terms.
:::

:::{grid-item-card} Model output metadata
:link: ../exploring_model_metadata
:link-type: doc

Output types, units, resolutions, channels, and track metadata across
AlphaGenome 2 and AlphaGenome 1.
:::

:::{grid-item-card} Navigating imputed tracks
:link: imputation
:link-type: doc

What the `observed`, `imputed_supported`, and `imputed_exploratory` labels mean,
and pitfalls when using imputed tracks.
:::

:::{grid-item-card} How variant scoring works
:link: ../variant_scoring
:link-type: doc

Spatial masks, indel alignment, and `ALT - REF` aggregation across scorers.
:::

:::{grid-item-card} Visualization basics
:link: ../visualization_basics
:link-type: doc

How `plot_components.plot`, track and expression components, enhancer arcs,
contact maps, and interval or variant annotations fit together.
:::

:::{grid-item-card} Migrating from AlphaGenome 1
:link: ag1_vs_ag2
:link-type: doc

API translation table, behavioral changes between AlphaGenome 1 and
AlphaGenome 2, and when to use each model.
:::

::::

<!-- mdformat on -->
