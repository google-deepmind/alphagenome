---
html_theme.sidebar_secondary.remove: true
---

# AlphaGenome documentation

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->
```{raw} html
<div class="ag-hero">
  <div class="ag-hero-text">
    <h2>AlphaGenome</h2>
    <p>
      Programmatic access to the AlphaGenome family of models and the
      AlphaGenome Atlas, developed by Google DeepMind. Predict the regulatory
      effects of DNA sequences and genetic variants, or query pre-computed
      variant effects across the human genome.
    </p>
    <div class="ag-hero-buttons">
      <a class="ag-btn ag-btn-primary" href="getting_started.html">Get started</a>
      <a class="ag-btn" href="https://alphagenome.google/api">Get an API key</a>
    </div>
  </div>
  <video class="ag-hero-video" autoplay loop muted playsinline
    disablepictureinpicture preload="metadata"
    poster="_static/hero_poster.jpg" aria-hidden="true">
    <source src="https://storage.googleapis.com/gdm-deepmind-com-prod-public/media/gkz58pvCcO8k2KhK/alphagenome_web_banner.webm#t=0.1"
      type="video/webm">
  </video>
</div>
```

::::{grid} 1 1 3 3
:gutter: 3
:class-container: ag-features

:::{grid-item-card} AlphaGenome 2
:shadow: none
:class-card: sd-border-0

Our latest model. Improved cell-type specificity and generalizes to your own
scRNA-seq data. Imputes unobserved modalities.
:::

:::{grid-item-card} AlphaGenome Atlas
:shadow: none
:class-card: sd-border-0

Genome-wide *in silico* mutagenesis across 9 billion variants, with
pre-computed scores and the AlphaGenome Variant Impact (AVI) score for
coding and non-coding variant prioritization. Available on a
[zero-code website](https://alphagenome.google/atlas) and through the API.
:::

:::{grid-item-card} AlphaGenome 1
:shadow: none
:class-card: sd-border-0

The original model described in the AlphaGenome paper {cite:p}`alphagenome`.
:::
::::

::::{grid} 1 2 4 4
:gutter: 3
:class-container: ag-tiles

:::{grid-item-card} {octicon}`rocket;2em` Getting started
:link: getting_started
:link-type: doc
:shadow: none
:class-card: ag-tile
:::

:::{grid-item-card} {octicon}`book;2em` Tutorials
:link: tutorials/index
:link-type: doc
:shadow: none
:class-card: ag-tile
:::

:::{grid-item-card} {octicon}`mortar-board;2em` User guide
:link: user_guides/index
:link-type: doc
:shadow: none
:class-card: ag-tile
:::

:::{grid-item-card} {octicon}`beaker;2em` Vignettes
:link: vignettes/index
:link-type: doc
:shadow: none
:class-card: ag-tile
:::
::::
<!-- mdformat on -->

The API is offered free of charge for
[non-commercial use](https://alphagenome.google/terms). Query rates vary with
demand. Atlas lookups allow higher rates than live model predictions, which suit
analyses of thousands of predictions rather than millions.

```{toctree}
:maxdepth: 2
:hidden: False

installation
getting_started
tutorials/index
user_guides/index
vignettes/index
api/index
faqs
resources
references
Community <https://www.alphagenomecommunity.com>
```
