# Getting started

This page helps you pick the right tool and shows the AlphaGenome 2 workflow in
a single example. To make API calls you need an
[API key](https://alphagenome.google/api) and the `alphagenome` package (see
[installation](installation)).

## Which tool do I need?

```{list-table}
:header-rows: 1

*   - If you want to…
    - Use
*   - Predict tracks or score variants in a reference tissue or cell type, or
      with expression conditioning given your own scRNA-seq data
    - **AlphaGenome 2** (recommended), continue below
*   - Look up pre-computed effects for human variants without running the model
    - **AlphaGenome Atlas**, on the
      [zero-code website](https://alphagenome.google/atlas) or the
      [Atlas API](api/atlas)
*   - Reproduce results from the AlphaGenome paper {cite:p}`alphagenome`, score
      variants across all tissues and cell types at once without choosing a
      condition, or predict PRO-cap
    - **AlphaGenome 1**, see
      [Migrating from AlphaGenome 1](user_guides/ag1_vs_ag2) and the
      [AlphaGenome 1 quick start](colabs/v1/quick_start)
```

Each AlphaGenome 2 call predicts for one condition: either a reference condition
(`condition=`, including composite conditions such as
`UBERON:0000178_CL:0001054`) or expression conditioning given up to 256 cells
from your own scRNA-seq data (`cell_conditioning=` plus `cell_condition_name=`).
To compare conditions, loop over them or pass one condition per item
(`conditions=` or `cell_conditionings=` plus `cell_condition_names=`) to batch
methods such as `predict_variants` and `score_variants`.

## The AlphaGenome 2 workflow

Most AlphaGenome 2 analyses follow four steps:

1.  **Choose a condition**: a reference tissue or cell type, or your own human
    cells (expression conditioning).
2.  **Select a genomic region**: the interval, and optionally the variant, to
    predict.
3.  **Predict or score**: predict tracks for an interval or sequence, predict
    the effect of a variant, or score a variant.
4.  **Plot** or inspect the predictions.

The example below predicts and scores a variant's effect on *GLB1L* expression,
using either a reference condition (CD14-positive monocytes in blood) or
expression conditioning given your own scRNA-seq data:

`````{tab-set}
````{tab-item} Reference condition
:sync: reference-condition

```python
from alphagenome.data import genome
from alphagenome.models.v2 import dna_client
from alphagenome.models.v2 import variant_scorers
from alphagenome.visualization import plot_components

dna_model = dna_client.create('YOUR_API_KEY')

# 1. Choose a condition: CD14-positive monocytes in blood.
condition = 'UBERON:0000178_CL:0001054'

# 2. Select a genomic region: center a 1 Mb interval on the variant.
variant = genome.Variant(
    chromosome='chr2',
    position=219243701,
    reference_bases='C',
    alternate_bases='T',
)
interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

# 3a. Predict REF and ALT RNA-seq tracks across all channels.
output = dna_model.predict_variant(
    interval=interval,
    variant=variant,
    requested_outputs=[dna_client.OutputType.RNA_SEQ],
    condition=condition,
)

# 3b. Score the variant effect across genes and filter to GLB1L.
scores = dna_model.score_variant(
    interval=interval,
    variant=variant,
    variant_scorers=[variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']],
    condition=condition,
)
tidy = variant_scorers.tidy_scores([scores]).query('gene_name == "GLB1L"')

# 4. Plot observed REF versus ALT tracks on the negative strand, where
#    GLB1L lies.
ref_tracks = output.reference.filter_to_observed().rna_seq
alt_tracks = output.alternate.filter_to_observed().rna_seq
plot_components.plot(
    [
        plot_components.OverlaidTracks(
            tdata={
                'REF': ref_tracks.filter_to_negative_strand(),
                'ALT': alt_tracks.filter_to_negative_strand(),
            },
            colors={'REF': '#22AAE1', 'ALT': 'red'},
        )
    ],
    interval=variant.reference_interval.resize(2**15),
    annotations=[plot_components.VariantAnnotation([variant])],
)
```
````

````{tab-item} Expression conditioning
:sync: expression-conditioning

```python
from alphagenome.data import genome
from alphagenome.models.v2 import dna_client
from alphagenome.models.v2 import variant_scorers
from alphagenome.visualization import plot_components

dna_model = dna_client.create('YOUR_API_KEY')

# 1. Choose a condition: up to 256 human cells per call as raw integer UMI
#    counts (AnnData with Ensembl gene IDs in var_names), passed as
#    `cell_conditioning` with a `cell_condition_name` label instead of
#    `condition`.
cells = ...

# 2. Select a genomic region: center a 1 Mb interval on the variant.
variant = genome.Variant(
    chromosome='chr2',
    position=219243701,
    reference_bases='C',
    alternate_bases='T',
)
interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

# 3a. Predict REF and ALT denoised single-cell gene expression (one channel
#     per strand in CPM).
output = dna_model.predict_variant(
    interval=interval,
    variant=variant,
    requested_outputs=[dna_client.OutputType.SC_GENE_EXPRESSION_DENOISED],
    cell_conditioning=cells,
    # Names the condition in output metadata.
    cell_condition_name='custom_cells',
)

# 3b. Score denoised single-cell gene expression conditioned on your cells.
scores = dna_model.score_variant(
    interval=interval,
    variant=variant,
    variant_scorers=[
        variant_scorers.RECOMMENDED_VARIANT_SCORERS[
            'SC_GENE_EXPRESSION_DENOISED'
        ]
    ],
    cell_conditioning=cells,
    cell_condition_name='custom_cells',
)
tidy = variant_scorers.tidy_scores([scores]).query('gene_name == "GLB1L"')

# 4. Plot REF versus ALT denoised single-cell gene expression on a log scale
#    around GLB1L.
ref_expression = output.reference.sc_gene_expression_denoised
alt_expression = output.alternate.sc_gene_expression_denoised
plot_components.plot(
    [
        plot_components.OverlaidGeneExpression(
            gdata={
                'REF': ref_expression.filter_to_negative_strand(),
                'ALT': alt_expression.filter_to_negative_strand(),
            },
            colors={'REF': '#22AAE1', 'ALT': 'red'},
            log_scale=True,
        )
    ],
    interval=variant.reference_interval.resize(2**15),
    annotations=[plot_components.VariantAnnotation([variant])],
)
```
````
`````

## Next steps

*   Start the ordered [Tutorials](tutorials/index) beginning with
    [Choosing a condition](colabs/v2/choosing_a_condition), or jump directly to
    [Conditioning on your own single-cell data](colabs/v2/conditioning_on_single_cell_data).
*   Browse [Vignettes](vignettes/index) for end-to-end biological analyses.
*   Read the [User guide](user_guides/index) for data objects, metadata, and how
    variant scoring works.
