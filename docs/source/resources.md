# Resources

## Videos

These videos were recorded for AlphaGenome 1. Core concepts such as intervals,
variants, output modalities, variant scoring, and plotting carry over to
AlphaGenome 2; see [Migrating from AlphaGenome 1](user_guides/ag1_vs_ag2) for
API differences.

<!-- mdformat off(Retain sphinx-design grid syntax.) -->
::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} AlphaGenome 101 tutorial
:class-body: ag-video-card-body

{bdg-primary}`Tutorial` {bdg-secondary}`AlphaGenome 1`
^^^
Walkthrough of AlphaGenome 1's multimodal sequence predictions, variant effect
scoring, and Python API workflow.

```{raw} html
<div class="ag-video-wrapper">
  <iframe
    class="ag-video-embed"
    src="https://www.youtube-nocookie.com/embed/Xbvloe13nak"
    title="AlphaGenome 101 tutorial"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    allow="fullscreen; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>
```
+++
{octicon}`video;0.9em;sd-mr-1`
[Watch on YouTube](https://youtu.be/Xbvloe13nak)
:::

:::{grid-item-card} AlphaGenome author roundtable
:class-body: ag-video-card-body

{bdg-primary}`Roundtable` {bdg-secondary}`AlphaGenome 1`
^^^
Google DeepMind authors discuss the biological and engineering design behind
AlphaGenome 1, the *Nature* paper, and the API.

```{raw} html
<div class="ag-video-wrapper">
  <iframe
    class="ag-video-embed"
    src="https://www.youtube-nocookie.com/embed/V8lhUqKqzUc"
    title="AlphaGenome author roundtable"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    allow="fullscreen; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>
```
+++
{octicon}`video;0.9em;sd-mr-1`
[Watch on YouTube](https://www.youtube.com/watch?v=V8lhUqKqzUc)
:::

::::
<!-- mdformat on -->

## Publications

<!-- mdformat off(Retain sphinx-design grid syntax.) -->
::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} AlphaGenome
:link: https://www.nature.com/articles/s41586-025-10014-0
:link-type: url

{bdg-primary}`Paper` *Nature* · 2026
^^^
**Advancing regulatory variant effect prediction with AlphaGenome**

Avsec *et al.* Introduces AlphaGenome 1, a unified 1 Mb DNA sequence model, and
benchmarks multimodal predictions and variant effect scoring across gene
expression, splicing, chromatin features, and contact maps.
+++
{octicon}`link-external;0.9em;sd-mr-1` doi:10.1038/s41586-025-10014-0
:::

:::{grid-item-card} AlphaGenome Atlas
:link: https://doi.org/10.64898/2026.09.16.26363192
:link-type: url

{bdg-primary}`Preprint` *medRxiv* · 2026
^^^
**AlphaGenome Atlas: *in silico* mutagenesis of the entire human genome
improves prioritization and interpretation of non-coding variants**

Cheng *et al.* Presents genome-wide *in silico* mutagenesis across 9 billion
human variants and the AlphaGenome Variant Impact (AVI) score for variant
prioritization.
+++
{octicon}`link-external;0.9em;sd-mr-1` doi:10.64898/2026.09.16.26363192
:::

::::
<!-- mdformat on -->

See {ref}`How should I cite AlphaGenome? <citing-alphagenome>` for BibTeX
citation entries.

## Agent skills

The
[Google DeepMind Science skills](https://github.com/google-deepmind/science-skills)
repository provides skills for [Google Antigravity](https://antigravity.google/)
to run AlphaGenome and AlphaGenome Atlas workflows from natural-language
prompts. To enable the Science plugin in Google Antigravity, open
**Customizations** > **Build with Google** > **Science** > **Add**.

<!-- mdformat off(Retain sphinx-design grid syntax.) -->
::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`mark-github;1em;sd-mr-1` science-skills
:link: https://github.com/google-deepmind/science-skills
:link-type: url

{bdg-primary}`Agent skills` {bdg-secondary}`Antigravity`
^^^
Google DeepMind repository of scientific skills for Google Antigravity,
including AlphaGenome skills for single-variant regulatory and splicing
analysis via the API, variant prioritization and annotation with AlphaGenome
Variant Impact (AVI) scores, and generating deep links to the AlphaGenome Atlas
web application.
+++
{octicon}`link-external;0.9em;sd-mr-1` google-deepmind/science-skills
:::

::::
<!-- mdformat on -->

## Model weights and code

The `alphagenome` package queries the hosted API and requires no local GPU or
TPU hardware. If you want to download and run the AlphaGenome model weights
locally, see the repository below.

<!-- mdformat off(Retain sphinx-design grid syntax.) -->
::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`mark-github;1em;sd-mr-1` alphagenome_research
:link: https://github.com/google-deepmind/alphagenome_research
:link-type: url

{bdg-primary}`Model weights` {bdg-secondary}`JAX`
^^^
Google DeepMind research repository for using the AlphaGenome model weights,
including the JAX reference implementation, variant scorers, and evaluation
notebooks.
+++
{octicon}`link-external;0.9em;sd-mr-1` google-deepmind/alphagenome_research
:::

::::
<!-- mdformat on -->

## Data downloads

To download precomputed AlphaGenome artifacts, visit the downloads page below.

<!-- mdformat off(Retain sphinx-design grid syntax.) -->
::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`download;1em;sd-mr-1` AlphaGenome data downloads
:link: https://deepmind.google.com/science/alphagenome/downloads
:link-type: url

{bdg-primary}`Data`
^^^
Precomputed AlphaGenome artifacts, including variant effect scores, AVI scores,
and enhancer-gene links.
+++
{octicon}`link-external;0.9em;sd-mr-1`
deepmind.google.com/science/alphagenome/downloads
:::

::::
<!-- mdformat on -->
