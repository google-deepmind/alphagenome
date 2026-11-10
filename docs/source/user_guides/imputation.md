# Navigating imputed tracks

AlphaGenome 2 predicts every track for every condition, including tracks that
were never experimentally measured in that condition. Any imputed track is a
model-generated hypothesis rather than a replacement for measurement. Each track
carries an `imputation_mode` label indicating whether it was directly measured
in training (`observed`), imputed along a biologically informative and
benchmarked direction (`imputed_supported`), or imputed without sufficient
experimental constraints where performance may be unreliable
(`imputed_exploratory`).

## Modalities, channels, and tracks

Each output type predicts a fixed set of **channels**, and a **track** is one
channel in one condition. For most assays, channels are technical variants of
the same readout (such as sequencing protocol, polyA+ versus total RNA, or
forward `+` and reverse `-` strands); only in ChIP-seq (`CHIP_HISTONE` and
`CHIP_TF`) do channels correspond to distinct biological targets (histone marks
or transcription factors). Output types group into **modalities** such as
chromatin accessibility or gene expression (see
[Modalities, channels, and tracks](/exploring_model_metadata.md#modalities-channels-and-tracks)).
`imputation_mode` is a per-track label.

A condition rarely has every channel or output type in a modality measured: it
typically has one sequencing protocol or assay profiled rather than all variants
(and in ChIP-seq, only a subset of histone marks or transcription factors).

## Two kinds of imputation

<!-- mdformat off(Retain sphinx-design grid syntax.) -->
::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Within a modality
At least one assay or channel of the modality was measured in this condition,
and the model fills in the rest. For most assays, this imputes unmeasured
technical variants or related output types of the same readout (such as
unmeasured sequencing protocols, or ATAC-seq when DNase-seq was measured). For
ChIP-seq, where channels represent distinct targets, it also imputes unmeasured
histone marks or transcription factors.
:::

:::{grid-item-card} Across modalities
None of the modality's channels were measured in this condition, so every
channel is predicted from the other modalities that were measured. In bone
marrow T cells, which only have single-cell assays, all histone ChIP-seq
channels are imputed from scATAC-seq and scRNA-seq.
:::

::::
<!-- mdformat on -->

{doc}`Working with imputed tracks <../colabs/v2/imputing_unmeasured_tracks>`
walks through both cases at the *TCF7* locus.

## Imputation modes

Every imputed track is a hypothesis about an unmeasured experiment. The two
imputed modes distinguish whether the condition's measured assays carry the
biological information needed to support that hypothesis:

<!-- mdformat off(Retain sphinx-design grid syntax.) -->
::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {bdg-success}`observed`
**Measured in training**

This exact channel was experimentally measured in this condition and used for
training.
:::

:::{grid-item-card} {bdg-primary}`imputed_supported`
**Supported hypothesis**

Not measured here, but inferred from measured assays where the direction of
imputation makes biological sense and has been benchmarked (for example,
predicting histone marks from chromatin accessibility). Still a hypothesis, but
grounded in informative measurements from that condition.
:::

:::{grid-item-card} {bdg-warning}`imputed_exploratory`
**Unconstrained / unvalidated**

Not measured here, and the condition's measured assays do not reliably
constrain this readout (for example, predicting splicing or specific TF binding
from chromatin accessibility alone). Performance in this regime is not
benchmarked and can be poor or systematically misleading.
:::

::::
<!-- mdformat on -->

## Cross-modality rules

The same rules apply to unmeasured channels inside a partly measured output
type: unmeasured histone marks in blood are `imputed_supported`, while
unmeasured transcription factors are always `imputed_exploratory`. The tables
below list which measured modalities support each output type.

**Can be supported**

```{list-table}
:header-rows: 1

*   - Modality
    - Output types
    - Supported when the condition has measured
*   - Chromatin accessibility
    - `ATAC`, `DNASE`, `SC_ATAC`
    - Accessibility and/or gene expression
*   - Gene expression
    - `RNA_SEQ`, `SC_GENE_EXPRESSION`
    - Accessibility and/or gene expression
*   - Histone modification and CpG methylation
    - `CHIP_HISTONE`, `METHYLATION`
    - Accessibility (gene expression alone is not sufficient)
```

**Always exploratory when unmeasured**

```{list-table}
:header-rows: 1

*   - Modality
    - Output types
*   - TF binding
    - `CHIP_TF`
*   - Splicing
    - `SPLICE_SITES`, `SPLICE_SITE_USAGE`, `SPLICE_JUNCTIONS`
*   - Transcription start sites
    - `CAGE`
*   - 3D chromatin contacts
    - `CONTACT_MAPS`
```

<!-- mdformat off(Retain MyST admonition syntax.) -->
:::{admonition} Expression conditioning
:class: note

Under expression conditioning (`cell_conditioning=`), your human scRNA-seq
counts do not correspond to a reference condition, so no track is labeled
observed. Single-cell gene expression is predicted directly from your cells, and
other modalities are imputed from the cell embedding with the same reliability
rules as above.
:::
<!-- mdformat on -->

## Where the labels appear

<!-- mdformat off(Retain sphinx-design grid syntax.) -->
::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Per condition: supported_conditions()
One row per condition and one column per output type. Each column shows the best
label among that output type's channels, so **observed means at least one
channel was measured, not all of them**. If only CTCF was profiled, the
transcription factor column reads observed even though the other 777 factors are
exploratory.
:::

:::{grid-item-card} Per track: imputation_mode
Every track in a prediction or score table has its own label in the
imputation_mode metadata column. Filter on it to keep only measured tracks, or
to include supported imputation hypotheses while excluding exploratory tracks.
:::

::::
<!-- mdformat on -->

`````{tab-set}
````{tab-item} Track and expression outputs
```python
# Keep only directly measured tracks on an Output or GeneExpressionData object:
observed_rna = output.filter_to_observed().rna_seq
```
````

````{tab-item} Tidy variant scores
```python
# Keep measured tracks and supported imputation hypotheses:
tidy = variant_scorers.tidy_scores(scores)
supported_scores = tidy.query(
    'imputation_mode in ["observed", "imputed_supported"]'
)
```
````
`````

## Potential pitfalls

### Splicing effects from single-cell assays only

*   **No full-length junction supervision:** Many primary cell types in
    `supported_conditions()` (and all expression conditioning inputs via
    `cell_conditioning=`) only have 3' or 5' UMI scRNA-seq
    (`SC_GENE_EXPRESSION`) or scATAC-seq (`SC_ATAC`). Droplet end-tag reads do
    not span internal splice junctions, providing no direct training supervision
    on cell-type-specific exon inclusion.
*   **Plausible baseline boundaries without *trans*-acting regulation:** DNA
    splice donor (GT) and acceptor (AG) motifs let AlphaGenome 2 place
    constitutive exon boundaries that look realistic on REF tracks, while
    tissue-specific alternative splicing controlled by *trans*-acting
    RNA-binding proteins remains unconstrained.
*   **Exon-level read depth in imputed RNA-seq:** When `RNA_SEQ` is
    `imputed_supported` from single-cell assays alone, total gene-level
    expression magnitude is well supported, whereas relative coverage across
    individual exons should not be interpreted as validated isoform usage.

<!-- mdformat off(Retain MyST admonition syntax.) -->
:::{admonition} How to avoid splicing pitfalls
:class: tip

Restrict `SPLICE_SITES`, `SPLICE_SITE_USAGE`, and `SPLICE_JUNCTIONS` scoring to
conditions where splicing tracks (`imputation_mode == 'observed'`) or bulk
full-length RNA-seq (`RNA_SEQ`) were directly measured. In single-cell-only
conditions, score total transcriptional effects using
`SC_GENE_EXPRESSION_DENOISED` (under expression conditioning with
`cell_conditioning=`) or `SC_GENE_EXPRESSION` (when passing `condition=`).
:::
<!-- mdformat on -->

### False positives in TF ChIP-seq for unexpressed transcription factors

*   **No negative-control TF ChIP-seq training tracks:** TF ChIP-seq experiments
    are almost always run in cell types where the target transcription factor is
    expressed. Unmeasured (condition, TF) tracks were masked from the loss
    during training rather than given zero signal, so the model was never
    explicitly taught that unexpressed factors do not bind DNA.
*   **Open chromatin and sequence motif false positives:** Whenever a regulatory
    element or variant falls in open chromatin and matches the motif of a
    transcription factor (or an expressed paralog from the same motif family,
    such as ETS, GATA, or KLF/SP proteins), imputed `CHIP_TF` tracks and ALT
    minus REF variant scores can be large even when the gene encoding that
    factor is unexpressed in the condition.

<!-- mdformat off(Retain MyST admonition syntax.) -->
:::{admonition} Check whether the candidate TF is expressed in that context
:class: tip

Restrict `CHIP_TF` tracks and scores to `imputation_mode == 'observed'` whenever
possible. When inspecting `imputed_exploratory` `CHIP_TF` predictions or scores,
**verify that the candidate transcription factor gene is expressed in that
condition** using a matched expression profile for that cell type (such as your
input `AnnData`) or by querying AlphaGenome 2 gene expression on the genomic
interval of the factor gene:

```python
# Check expression of a candidate TF gene on its own 1 Mb genomic interval:
tf_pred = dna_model.predict_interval(
    interval=tf_gene_interval.resize(dna_client.SEQUENCE_LENGTH_1MB),
    requested_outputs=[dna_client.OutputType.SC_GENE_EXPRESSION],
    condition=condition,
)
sc_gex = tf_pred.sc_gene_expression.filter_to_observed()

# Or filter candidate TF ChIP-seq variant scores against a predefined set of TF
# gene symbols expressed in your condition (for example, from your own RNA-seq):
expressed_tf_symbols = {'SPI1', 'CEBPA', 'IRF8'}
valid_tf_scores = tidy[
    (tidy['output_type'] == 'CHIP_TF')
    & (
        (tidy['imputation_mode'] == 'observed')
        | tidy['transcription_factor'].isin(expressed_tf_symbols)
    )
]
```
:::
<!-- mdformat on -->
