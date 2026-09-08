# Exploring the genome with AlphaGenome

This API provides access to AlphaGenome, Google DeepMind’s unifying model for
deciphering the regulatory code within DNA sequences.

AlphaGenome offers multimodal predictions, encompassing diverse functional
outputs such as gene expression, splicing patterns, chromatin features, and
contact maps (see diagram below). The model analyzes DNA sequences of up to 1
million base pairs in length and can deliver predictions at single base-pair
resolution for most outputs. AlphaGenome achieves state-of-the-art performance
across a range of genomic prediction benchmarks, including numerous diverse
variant effect prediction tasks (detailed in {cite:p}`alphagenome`).

AlphaGenome API also provides access to the ‘Atlas’ dataset. AlphaGenome Atlas
is a resource of pre-computed variant effect score predictions, AlphaGenome
Variant Impact (AVI) score, and feature importances across the entire human
genome. Learn more
[here](https://deepmind.google.com/science/alphagenome/atlas).

The API is offered as a free service for
[non-commercial use](https://deepmind.google.com/science/alphagenome/terms).
Query rates vary based on demand – the pre-computed Atlas predictions will
typically have a larger query rate. Other predictions are better suited for
smaller to medium-scale analyses such as analysing a limited number of genomic
regions or variants requiring 1000s of predictions, but is likely not suitable
for large scale analyses requiring more than 1 million predictions.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->
```{figure} /_static/model_overview.png
:width: 600px
:alt: overview of AlphaGenome
:name: overview-figure
```
<!-- mdformat on -->

## Getting started

You can get started by
[getting an API key](https://deepmind.google.com/science/alphagenome), and
following our [Quick Start Guide](./colabs/quick_start.ipynb), or watching our
[AlphaGenome 101 tutorial](https://youtu.be/Xbvloe13nak). Please also check out
our installation guide, tutorials with comprehensive overviews of plotting,
variant scoring and other use cases, and our API reference documentation.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->
::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card}
:link: installation
:link-type: doc
Installation
^^^^^^^^^^^^

Install `alphagenome` locally.
:::

:::{grid-item-card}
:link: tutorials/index
:link-type: doc
Tutorials
^^^^^^^^^
The tutorials walk through example usage of the AlphaGenome model.
:::

:::{grid-item-card}
:link: api/index
:link-type: doc
API reference
^^^^^^^^^^^^^

Reference documentation for the `alphagenome` package.
:::
::::
<!-- mdformat on -->

``` {toctree}
:maxdepth: 2
:hidden: False

../colabs/quick_start
installation
api/index
tutorials/index
user_guides/index
faqs
Community <https://www.alphagenomecommunity.com>
references
```
