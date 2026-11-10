# Interpretation

Helpers for interpreting model predictions via in silico mutagenesis (ISM) and
enhancer-to-gene (E2G) scoring:

-   **ISM**: `ism_variants` lists every single-base substitution in an interval,
    and `ism_matrix` turns per-variant scores from
    `DnaClient.score_ism_variants` into a position-by-base matrix.
-   **Enhancer scoring**: `tile_interval`, `score_enhancers`,
    `get_active_enhancers`, and `assign_consensus_enhancers` tile genomic
    intervals, score candidate elements from
    {meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_interval_gradients`
    (or
    {meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_interval_gradients_batch`)
    outputs, call active enhancer-gene links with modality-calibrated
    thresholds, and cluster called peaks into cross-biosample consensus
    intervals.

## ISM

```{eval-rst}
.. module:: alphagenome.interpretation
.. currentmodule:: alphagenome

.. autosummary::
    :toctree: generated

    interpretation.ism.ism_variants
    interpretation.ism.ism_matrix
```

## Enhancer scoring

Call and score candidate cis-regulatory enhancers from
{meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_interval_gradients`
(or
{meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_interval_gradients_batch`)
outputs and chromatin accessibility tracks. See {doc}`Mapping lineage-specific
enhancers at the BCL11A locus <../colabs/v2/enhancer_to_gene_bcl11a>` for an
end-to-end walkthrough.

```{eval-rst}
.. autosummary::
    :toctree: generated

    interpretation.enhancer_scoring.tile_interval
    interpretation.enhancer_scoring.score_enhancers
    interpretation.enhancer_scoring.get_active_enhancers
    interpretation.enhancer_scoring.assign_consensus_enhancers
```

```{eval-rst}
.. autodata:: alphagenome.interpretation.enhancer_scoring.DEFAULT_GRADIENT_SCORE_THRESHOLDS
   :no-value:
```
