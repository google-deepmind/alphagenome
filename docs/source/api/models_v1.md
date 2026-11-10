# AlphaGenome 1

Client, outputs and scorers for AlphaGenome 1.

## DNA client

```{eval-rst}
.. currentmodule:: alphagenome
```

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v1.dna_client.create
    models.v1.dna_client.ModelVersion
    models.v1.dna_client.Organism
    models.v1.dna_client.validate_sequence_length
    models.v1.dna_client.DnaClient
```

### Sequence lengths

Supported input sequence lengths, for use with `sequence_length` arguments and
{func}`~alphagenome.models.v1.dna_client.validate_sequence_length`.

```{eval-rst}

.. autodata:: alphagenome.models.v1.dna_client.SEQUENCE_LENGTH_16KB

.. autodata:: alphagenome.models.v1.dna_client.SEQUENCE_LENGTH_100KB

.. autodata:: alphagenome.models.v1.dna_client.SEQUENCE_LENGTH_500KB

.. autodata:: alphagenome.models.v1.dna_client.SEQUENCE_LENGTH_1MB
```

## DNA output

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v1.dna_output.OutputType
    models.v1.dna_output.Output
    models.v1.dna_output.OutputMetadata
    models.v1.dna_output.VariantOutput
```

## Variant scorers

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v1.variant_scorers.AggregationType
    models.v1.variant_scorers.BaseVariantScorer
    models.v1.variant_scorers.CenterMaskScorer
    models.v1.variant_scorers.ContactMapScorer
    models.v1.variant_scorers.GeneMaskLFCScorer
    models.v1.variant_scorers.GeneMaskActiveScorer
    models.v1.variant_scorers.GeneMaskSplicingScorer
    models.v1.variant_scorers.PolyadenylationScorer
    models.v1.variant_scorers.SpliceJunctionScorer
    models.v1.variant_scorers.get_recommended_scorers
    models.v1.variant_scorers.tidy_anndata
    models.v1.variant_scorers.tidy_scores
```

```{eval-rst}

.. autodata:: alphagenome.models.v1.variant_scorers.RECOMMENDED_VARIANT_SCORERS
   :no-value:

.. autodata:: alphagenome.models.v1.variant_scorers.VariantScorerTypes
   :no-value:
```

## Interval scorers

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v1.interval_scorers.IntervalAggregationType
    models.v1.interval_scorers.BaseIntervalScorer
    models.v1.interval_scorers.GeneMaskScorer

.. py:data:: alphagenome.models.v1.interval_scorers.IntervalScorerTypes

   Union type alias of supported AlphaGenome 1 interval scorer configurations.
```
