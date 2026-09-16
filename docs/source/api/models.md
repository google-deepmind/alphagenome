# Models

AlphaGenome client and variant scorers.

## DNA Client

```{eval-rst}
.. currentmodule:: alphagenome
```

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.dna_client.create
    models.dna_client.ModelVersion
    models.dna_client.Organism
    models.dna_client.validate_sequence_length
    models.dna_client.DnaClient
```

## DNA Output

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.dna_output.OutputType
    models.dna_output.Output
    models.dna_output.OutputMetadata
    models.dna_output.VariantOutput
```

## Variant Scorers

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.variant_scorers.AggregationType
    models.variant_scorers.BaseVariantScorer
    models.variant_scorers.CenterMaskScorer
    models.variant_scorers.ContactMapScorer
    models.variant_scorers.GeneMaskLFCScorer
    models.variant_scorers.GeneMaskActiveScorer
    models.variant_scorers.GeneMaskSplicingScorer
    models.variant_scorers.PolyadenylationScorer
    models.variant_scorers.SpliceJunctionScorer
    models.variant_scorers.get_recommended_scorers
    models.variant_scorers.tidy_anndata
    models.variant_scorers.tidy_scores
```
