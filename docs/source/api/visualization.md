# Visualization

Plotting tools for AlphaGenome outputs. Most figures are built with
`plot_components.plot` and component classes such as `Tracks` and
`TranscriptAnnotation`; `plot` holds lower-level matplotlib helpers.

## Plot

```{eval-rst}
.. currentmodule:: alphagenome
```

```{eval-rst}

.. autosummary::
    :toctree: generated

    visualization.plot.seqlogo
    visualization.plot.plot_contact_map
    visualization.plot.plot_track
    visualization.plot.plot_tracks
    visualization.plot.sashimi_plot
    visualization.plot.enhancer_arcs_plot
    visualization.plot.pad_track
```

(visualization/plot-components)=

## Plot components

```{eval-rst}

.. autosummary::
    :toctree: generated

    visualization.plot_components.plot
    visualization.plot_components.AbstractComponent
    visualization.plot_components.Tracks
    visualization.plot_components.OverlaidTracks
    visualization.plot_components.ContactMaps
    visualization.plot_components.ContactMapsDiff
    visualization.plot_components.GeneExpression
    visualization.plot_components.OverlaidGeneExpression
    visualization.plot_components.TranscriptAnnotation
    visualization.plot_components.SeqLogo
    visualization.plot_components.Sashimi
    visualization.plot_components.FeatureContribution
    visualization.plot_components.ScoreBar
    visualization.plot_components.EnhancerArcs
    visualization.plot_components.EmptyComponent
    visualization.plot_components.AbstractAnnotation
    visualization.plot_components.IntervalAnnotation
    visualization.plot_components.VariantAnnotation
```

## Plot transcripts

```{eval-rst}

.. autosummary::
    :toctree: generated

    visualization.plot_transcripts.TranscriptStyle
    visualization.plot_transcripts.TranscriptStylePreset
    visualization.plot_transcripts.plot_transcripts
    visualization.plot_transcripts.draw_interval
    visualization.plot_transcripts.draw_transcript
```
