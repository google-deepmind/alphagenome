# Visualization basics

<!-- disableFinding(LINK_ID) -->

AlphaGenome predicts a variety of output types with different data shapes and
biological interpretations ([table](#viz-table)). We provide
[`alphagenome.visualization`](project:api/visualization.md) to generate
matplotlib figures from model API outputs, which we outline here.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->

```{tip}
See the {doc}`visualizing predictions tutorial </colabs/visualization_modality_tour>`
for worked examples of plotting different modalities.
```

<!-- mdformat on -->

## Plot

The key function, {func}`~alphagenome.visualization.plot_components.plot`, takes
as input a list of components and returns a {class}`matplotlib.figure.Figure`.

## Components

A component is a light wrapper around a model output (such as predicted genomic
tracks, splice junctions, etc) and specifies plot aesthetics. Each component
maps to one vertically stacked subplot in the final figure (see blue text in the
[figure](#viz-figure)). Each component has an independent y-axis but shares a
common x-axis, corresponding to the length of the DNA interval, in base pairs
(bp).

Several default components are available, each designed to best visually
represent different modalities and data shapes returned by the model API (see
[table](#viz-table)).

## Annotations

Additional figure elements specific to the DNA interval, but outside of
components -- such as locations of promoters or variants -- can be overlaid via
a list of annotations that are passed to
{func}`~alphagenome.visualization.plot_components.plot`.

## Aligning REF and ALT tracks

When visualizing long variants (such as large deletions or insertions), the REF and ALT tracks may have different lengths and coordinate systems. To correctly overlay them using {class}`~alphagenome.visualization.plot_components.OverlaidTracks`, you should first align them to a common coordinate system (usually REF).

We provide a utility function for this purpose:

```python
from alphagenome.visualization import track_alignment

# ... generate ref_track, alt_track, and variant ...

ref_aligned, alt_aligned = track_alignment.align_ref_alt_tracks(
    ref_track, alt_track, variant
)

# Now plot using OverlaidTracks
# ...
```

This utility interpolates the ALT track onto the REF coordinate grid. Deleted regions are filled with NaNs (appearing as gaps), and insertions are compressed. This ensures that the tracks are visually aligned.

## Custom plotting

For users interested in configuring novel components, extend the
{func}`~alphagenome.visualization.plot_components.AbstractComponent` and
{func}`~alphagenome.visualization.plot_components.AbstractAnnotation` base
classes.

Any other data supplied by the user can be visualized using this library as is,
as long as it is provided to
[`plot_components`](project:api/visualization.md#plot-components) in the format
required e.g. {class}`~alphagenome.data.track_data.TrackData` for
{class}`~alphagenome.visualization.plot_components.Tracks`.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->

```{figure} /_static/visualization_overview.png
:height: 600px
:alt: visualization library description/overview
:name: viz-figure

Illustrative diagram of visualization library. Blue text indicates
[`plot_components`](<project:api/visualization.md#plot-components>) classes, and purple text indicates arguments to
[`plot_components`](<project:api/visualization.md#plot-components>) that adjust figure-wide aesthetics
```

```{list-table} Plotting components and annotation classes.
:widths: 10 30 10 10 30 10
:header-rows: 1
:name: viz-table

*   - Component name plot\_components.\*
    -   Description
    -   Example figure
    -   Data shape supported
    -   Recommended model outputs
    -   Good for visualising variants?
*   - {class}`~alphagenome.visualization.plot_components.Tracks`
    -   A line-plot visualizing a scalar value at each genomic position (or
        coarser resolution) e.g. predictions of RNA\_SEQ for a specific
    -   Colab cell
    -   1D
    -   All except SPLICE\_JUNCTIONS; CONTACT\_MAPS
    -   No
*   - {class}`~alphagenome.visualization.plot_components.OverlaidTracks`
    -   A line-plot as for Tracks, but with two separate lines on the same axis
        with different colors e.g. predictions of RNA\_SEQ for the Reference and
        Alternative sequence defined by a variant.
    -   Colab cell
    -   1D x 2
    -   All except SPLICE\_JUNCTIONS; CONTACT\_MAPS
    -   Yes
*   - {class}`~alphagenome.visualization.plot_components.Sashimi`
    -   A series of arcs, each representing a scalar value for a pair of genomic
        positions (e.g. splice junctions). The thickness of the arcs are
        determined by the relative sizes of the scalars.
    -   Colab cell
    -   2D (sparse)
    -   SPLICE\_JUNCTIONS
    -   Yes
*   - {class}`~alphagenome.visualization.plot_components.SeqLogo`
    -   A sequence of letters (bases) with heights corresponding to a single
        scalar value per genomic position (e.g. from contribution scores).
    -   Colab cell
    -   1D \+ sequence
    -   ISM contribution scores
    -   Yes
*   - {class}`~alphagenome.visualization.plot_components.ContactMaps`
    -   A heatmap visualizing a matrix of scalars (e.g. predicted DNA-DNA
        contacts), one for each pair of genomic positions in an interval.
    -   Colab cell
    -   2D
    -   CONTACT\_MAPS
    -   No
*   - {class}`~alphagenome.visualization.plot_components.ContactMapsDiff`
    -   A heatmap as for ContactMaps, but with a diverging color map centered on
        zero (white) to represent values derived from differences (e.g. ALT \-
        REF)
    -   Colab cell
    -   2D
    -   CONTACT\_MAPS
    -   Yes
*   - {class}`~alphagenome.visualization.plot_components.TranscriptAnnotation`
    -   Horizontal lines representing locations of transcripts. Exons, introns,
        untranslated regions, and direction of transcription are indicated by
        differences in line thickness.
    -   Colab cell
    -   Interval(s)
    -   N/A
    -   No
*   - {class}`~alphagenome.visualization.plot_components.VariantAnnotation`
    -   A semi-transparent rectangle (or vertical line if a variant) spanning
        all plot components, indicating the location of an interval (or
        variant). The interval (variant) is optionally labeled.
    -   Colab cell
    -   Interval(s) or Variant(s)
    -   N/A
    -   Yes
*   - {class}`~alphagenome.visualization.plot_components.AbstractComponent`
    -   This is an abstract class, which is the parent class of most
        plot\_components.\*. A user can define their own component class,
        provided it adheres to the structure specified by AbstractComponent. The
        workhorse method is plot\_ax(), which populates a matplotlib.axes.Axes
        object with visuals defined by the input data.
    -   N/A
    -   N/A
    -   N/A
    -   N/A
```

<!-- mdformat on -->
