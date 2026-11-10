# Visualization basics

AlphaGenome predicts several output types with different data shapes and
resolutions: 1D genomic tracks (`TrackData`), sparse splice junctions
(`JunctionData`), 2D chromatin contact maps (`TrackData`), and, in AlphaGenome 2
only, gene-level single-cell expression tables (`GeneExpressionData`). The
{doc}`alphagenome.visualization <api/visualization>` package builds multi-panel
`matplotlib` figures that align these outputs along a shared genomic coordinate
axis.

```{tip}
For worked examples across every modality, see the
{doc}`Plotting predictions <colabs/v2/plotting_predictions>` tutorial.
```

## Assembling a figure with `plot`

{func}`~alphagenome.visualization.plot_components.plot` takes a sequence of
components, a target {class}`~alphagenome.data.genome.Interval`, and optional
annotations, and returns a {class}`matplotlib.figure.Figure`:

```python
from alphagenome.visualization import plot_components

fig = plot_components.plot(
    components=[
        plot_components.TranscriptAnnotation(transcripts),
        plot_components.Tracks(output.atac),
        plot_components.GeneExpression(output.sc_gene_expression_denoised),
    ],
    interval=interval,
    annotations=[plot_components.VariantAnnotation([variant])],
)
```

Each component in `components` renders as one vertically stacked panel (or one
panel per track) with its own y-axis and a shared base-pair x-axis cropped to
`interval` ({ref}`figure <viz-figure>`).

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->

```{figure} /_static/visualization_overview.png
:height: 600px
:alt: Overview of plot_components layout and arguments
:name: viz-figure

Anatomy of a `plot_components.plot` figure. Blue labels show component and
annotation classes; purple labels show `plot()` arguments that control
figure-wide layout.
```

<!-- mdformat on -->

## Components and annotations

Each component class wraps a model output container (or custom data placed in
`TrackData`, `GeneExpressionData`, or `JunctionData`) and controls panel-level
aesthetics such as colors, y-axis scaling (`shared_y_scale`, `global_ylims`),
and y-axis label templates (`ylabel_template`).

```{list-table}
:header-rows: 1
:name: viz-table

*   - Class
    - Input container
    - Recommended outputs
    - Description
*   - {class}`~alphagenome.visualization.plot_components.Tracks`
    - `TrackData` (1D)
    - Positional tracks (`ATAC`, `DNASE`, `RNA_SEQ`, `CAGE`, `CHIP_HISTONE`,
      `CHIP_TF`, `SPLICE_SITES`, `SPLICE_SITE_USAGE`; `METHYLATION` and
      `SC_ATAC` in AlphaGenome 2 only; `PROCAP` in AlphaGenome 1 only)
    - Line or filled-area profile across genomic coordinates, one subplot per
      track.
*   - {class}`~alphagenome.visualization.plot_components.OverlaidTracks`
    - `Mapping[str, TrackData]`
    - `REF` vs. `ALT` positional tracks
    - Overlays two or more `TrackData` objects with matching metadata on the
      same axes to compare variant alleles or conditions.
*   - {class}`~alphagenome.visualization.plot_components.GeneExpression`
    - `GeneExpressionData`
    - `SC_GENE_EXPRESSION_DENOISED`, `SC_GENE_EXPRESSION` (AlphaGenome 2 only)
    - Horizontal gene-span bars showing scalar gene-level expression (in CPM, or
      `log1p` with `log_scale=True`) across genes overlapping the interval.
*   - {class}`~alphagenome.visualization.plot_components.OverlaidGeneExpression`
    - `Mapping[str, GeneExpressionData]`
    - `REF` vs. `ALT` gene-level expression (AlphaGenome 2 only)
    - Overlays two or more `GeneExpressionData` objects on the same axes to
      compare `REF` and `ALT` gene expression across an interval.
*   - {class}`~alphagenome.visualization.plot_components.Sashimi`
    - `JunctionData`
    - `SPLICE_JUNCTIONS`
    - Arcs connecting splice donor and acceptor coordinates, with arc thickness
      and optional labels showing normalized junction read counts.
*   - {class}`~alphagenome.visualization.plot_components.EnhancerArcs`
    - `DataFrame`
    - Enhancer-to-gene links from
      {func}`~alphagenome.interpretation.enhancer_scoring.get_active_enhancers`
      (AlphaGenome 2 only)
    - Cubic Bezier arcs connecting candidate enhancer intervals to target gene
      TSS coordinates, optionally colored or scaled by enhancer score.
*   - {class}`~alphagenome.visualization.plot_components.ContactMaps`
    - `TrackData` (2D)
    - `CONTACT_MAPS`
    - Pairwise 2D heatmap of chromatin contact frequencies across genomic bins.
*   - {class}`~alphagenome.visualization.plot_components.ContactMapsDiff`
    - `TrackData` (2D)
    - `ALT - REF` `CONTACT_MAPS`
    - Pairwise 2D heatmap with a zero-centered diverging colormap (`RdBu_r`) for
      contact map differences.
*   - {class}`~alphagenome.visualization.plot_components.SeqLogo`
    - `np.ndarray` `(S, A)` + `Interval`
    - ISM contribution scores
    - Nucleotide sequence logo where letter heights reflect per-base
      attribution or *in silico* mutagenesis scores.
*   - {class}`~alphagenome.visualization.plot_components.TranscriptAnnotation`
    - `Sequence[Transcript]`
    - GENCODE / GTF transcripts
    - Horizontal gene and isoform models showing exons, introns, coding
      regions, and strand direction.
*   - {class}`~alphagenome.visualization.plot_components.VariantAnnotation`
    - `Sequence[Variant]`
    - Variant positions
    - Vertical highlight lines spanning all panels at each variant coordinate,
      with optional angled labels at the top of the figure.
*   - {class}`~alphagenome.visualization.plot_components.IntervalAnnotation`
    - `Sequence[Interval]`
    - Candidate regulatory elements, exons, or ROIs
    - Shaded vertical bands spanning all panels across each genomic interval,
      with optional labels.
```

## Custom components

To define a custom panel type or cross-panel overlay, subclass
{class}`~alphagenome.visualization.plot_components.AbstractComponent`
(implementing `num_axes`, `get_ax_height`, and `plot_ax`) or
{class}`~alphagenome.visualization.plot_components.AbstractAnnotation`
(implementing `plot_ax` and `plot_labels`).
