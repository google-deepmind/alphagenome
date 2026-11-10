# AlphaGenome 2

Client, endpoints, outputs, conditions and scorers for AlphaGenome 2.

## DNA client

```{eval-rst}
.. currentmodule:: alphagenome
```

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v2.dna_client.create
    models.v2.dna_client.ModelVersion
    models.v2.dna_client.Organism
    models.v2.dna_client.validate_sequence_length
    models.v2.dna_client.DnaClient
```

### Sequence lengths

Supported input sequence lengths, for use with `sequence_length` arguments and
{func}`~alphagenome.models.v2.dna_client.validate_sequence_length`. Input
gradient endpoints support only `SEQUENCE_LENGTH_100KB` and
`SEQUENCE_LENGTH_1MB`.

```{eval-rst}

.. autodata:: alphagenome.models.v2.dna_client.SEQUENCE_LENGTH_16KB

.. autodata:: alphagenome.models.v2.dna_client.SEQUENCE_LENGTH_100KB

.. autodata:: alphagenome.models.v2.dna_client.SEQUENCE_LENGTH_500KB

.. autodata:: alphagenome.models.v2.dna_client.SEQUENCE_LENGTH_1MB
```

### Limits

```{eval-rst}

.. autodata:: alphagenome.models.v2.dna_client.MAX_CONDITIONING_CELLS
```

### Endpoints

The {class}`~alphagenome.models.v2.dna_client.DnaClient` exposes the following
endpoints:

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->

| Method | Description |
| :--- | :--- |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_sequence` | Predict tracks for a raw DNA sequence (`predict_sequences` for batches). |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_interval` | Predict tracks for a reference genome interval (`predict_intervals` for batches). |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_variant` | Predict tracks for the REF and ALT alleles of a variant (`predict_variants` for batches). |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.score_variant` | Score the effect of a variant (`score_variants` for batches). |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.score_interval` | Score an interval (`score_intervals` for batches). |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.score_ism_variants` | Score all single-base variants in an interval (*in silico* mutagenesis). |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_interval_gradients` | Compute input gradients of a gene's predicted expression (`RNA_SEQ`, `SC_GENE_EXPRESSION`, or `SC_GENE_EXPRESSION_DENOISED`) over an interval centered on the gene's TSS (`predict_interval_gradients_batch` for batches). |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.encode_cells` | Encode raw single-cell UMI counts (human) into per-cell embeddings with the cell encoder, without running the sequence model. |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.supported_conditions` | List reference conditions with the imputation mode (`observed`, `imputed_supported`, or `imputed_exploratory`) of each output type. |
| {meth}`~alphagenome.models.v2.dna_client.DnaClient.observed_metadata` | Get the metadata for tracks observed during training. |

<!-- mdformat on -->

## DNA output

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v2.dna_output.OutputType
    models.v2.dna_output.Output
    models.v2.dna_output.OutputMetadata
    models.v2.dna_output.VariantOutput
    models.v2.dna_output.TargetGeneMetadata
    models.v2.dna_output.InputGradientsOutput
```

## Conditions

Reference conditions (`condition=`) are built with `model_condition`; expression
conditioning inputs (`cell_conditioning=`) are prepared with
`gene_counts_condition`.

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v2.model_condition.ModelCondition
    models.v2.model_condition.from_curies
    models.v2.gene_counts_condition.from_anndata
    models.v2.gene_counts_condition.subsample
```

## Variant scorers

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v2.variant_scorers.AggregationType
    models.v2.variant_scorers.BaseVariantScorer
    models.v2.variant_scorers.CenterMaskScorer
    models.v2.variant_scorers.ContactMapScorer
    models.v2.variant_scorers.GeneMaskLFCScorer
    models.v2.variant_scorers.GeneMaskActiveScorer
    models.v2.variant_scorers.GeneMaskSplicingScorer
    models.v2.variant_scorers.MethylationScorer
    models.v2.variant_scorers.PolyadenylationScorer
    models.v2.variant_scorers.SpliceJunctionScorer
    models.v2.variant_scorers.get_recommended_scorers
    models.v2.variant_scorers.tidy_anndata
    models.v2.variant_scorers.tidy_scores
```

```{eval-rst}

.. autodata:: alphagenome.models.v2.variant_scorers.RECOMMENDED_VARIANT_SCORERS
   :no-value:

.. autodata:: alphagenome.models.v2.variant_scorers.VariantScorerTypes
   :no-value:
```

## Interval scorers

```{eval-rst}

.. autosummary::
    :toctree: generated

    models.v2.interval_scorers.IntervalAggregationType
    models.v2.interval_scorers.BaseIntervalScorer
    models.v2.interval_scorers.GeneMaskScorer

.. py:data:: alphagenome.models.v2.interval_scorers.IntervalScorerTypes

   Union type alias of supported AlphaGenome 2 interval scorer configurations.
```
