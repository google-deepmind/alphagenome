# Model output metadata

Output types, units, resolutions, and track metadata columns for human
predictions across AlphaGenome 2 and AlphaGenome 1.

## AlphaGenome 2

AlphaGenome 2 defines 15 output types in
{class}`~alphagenome.models.v2.dna_output.OutputType` (see
[Migrating from AlphaGenome 1](user_guides/ag1_vs_ag2) for output types added or
changed relative to AlphaGenome 1).

### Modalities, channels, and tracks

A **modality** is a class of molecular readout (such as chromatin accessibility,
gene expression, histone modification, TF binding, splicing, CpG methylation, or
3D chromatin contacts) predicted by one or more output types (for example,
chromatin accessibility spans `ATAC`, `DNASE`, and `SC_ATAC`). Each output type
predicts a fixed set of **channels** shared across conditions. For most output
types, channels represent technical variants of the same assay (such as
sequencing protocol, polyA+ versus total RNA, or forward `+` and reverse `-`
strands), whereas for ChIP-seq (`CHIP_HISTONE` and `CHIP_TF`) each channel
corresponds to a specific histone mark or transcription factor target. A
**track** is one channel in one condition. For example:

*   Histone ChIP-seq (`CHIP_HISTONE`) has 43 channels covering 32 histone marks
    (H3K27ac, H3K4me3, H3K27me3, and so on) across single- and paired-end
    assays.
*   TF ChIP-seq (`CHIP_TF`) has 911 channels covering 778 transcription factors.
*   scATAC-seq (`SC_ATAC`) has 5 channels, one per sequencing protocol.
*   Stranded assays such as RNA-seq and CAGE have separate forward (`+`) and
    reverse (`-`) strand channels.

Each track carries an `imputation_mode` label (see
[Navigating imputed tracks](user_guides/imputation)).

### Output types

Use the tabs below to browse output types by modality. Counts reflect observed
human training tracks (`dna_model.observed_metadata()`).

`````{tab-set}

````{tab-item} Expression
```{list-table}
:header-rows: 1

*   - Output type
    - Measures
    - Units
    - Resolution
    - Biosamples
    - Tracks
*   - {attr}`RNA_SEQ <alphagenome.models.v2.dna_output.OutputType.RNA_SEQ>`
    - Bulk RNA-seq coverage (polyA+ and total RNA, stranded and unstranded)
    - Normalized read signal
    - 1 bp
    - 306
    - 738
*   - {attr}`CAGE <alphagenome.models.v2.dna_output.OutputType.CAGE>`
    - Cap Analysis of Gene Expression (CAGE) 5′-cap signal, concentrated at
      transcription start sites (also detected along exons)
    - Normalized read signal
    - 1 bp
    - 268
    - 550
*   - {attr}`SC_GENE_EXPRESSION
      <alphagenome.models.v2.dna_output.OutputType.SC_GENE_EXPRESSION>`
    - Protocol-specific single-cell RNA-seq expression
    - Counts per million (CPM)
    - Gene
    - 1943
    - 5028
*   - {attr}`SC_GENE_EXPRESSION_DENOISED
      <alphagenome.models.v2.dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED>`
    - Denoised single-cell RNA-seq expression (1 channel per strand)
    - CPM
    - Gene
    - Per call
    - 2
```

- `SC_GENE_EXPRESSION` has a separate track for each sequencing chemistry,
  suspension type (cell or nucleus), and strand, and is supported with both
  `condition=` and `cell_conditioning=`.
- `SC_GENE_EXPRESSION_DENOISED` gives one denoised expression estimate per gene
  and strand, without the per-protocol tracks of `SC_GENE_EXPRESSION`. It is
  human-only and supported exclusively under expression conditioning
  (`cell_conditioning=`).
````

````{tab-item} Accessibility
```{list-table}
:header-rows: 1

*   - Output type
    - Measures
    - Units
    - Resolution
    - Biosamples
    - Tracks
*   - {attr}`DNASE <alphagenome.models.v2.dna_output.OutputType.DNASE>`
    - DNase I hypersensitive site cleavage (DNase-seq)
    - Normalized insertion signal
    - 1 bp
    - 330
    - 385
*   - {attr}`ATAC <alphagenome.models.v2.dna_output.OutputType.ATAC>`
    - Bulk transposase-accessible chromatin (ATAC-seq)
    - Normalized insertion signal
    - 1 bp
    - 178
    - 179
*   - {attr}`SC_ATAC <alphagenome.models.v2.dna_output.OutputType.SC_ATAC>`
    - Single-cell ATAC-seq pseudobulks (5 assay protocols per condition)
    - Normalized insertion signal
    - 1 bp
    - 704
    - 731
```

- Single-cell datasets (`SC_ATAC` and `SC_GENE_EXPRESSION`) were processed with
  standard pipelines (such as CellRanger or the ENCODE single-cell pipeline) and
  pseudobulked by author cell-type labels aligned to ontology terms.
````

````{tab-item} Histone, TF, and methylation
```{list-table}
:header-rows: 1

*   - Output type
    - Measures
    - Units
    - Resolution
    - Biosamples
    - Tracks
*   - {attr}`CHIP_HISTONE
      <alphagenome.models.v2.dna_output.OutputType.CHIP_HISTONE>`
    - Histone modification ChIP-seq
    - Fold-change over control
    - 128 bp
    - 221
    - 1183
*   - {attr}`CHIP_TF <alphagenome.models.v2.dna_output.OutputType.CHIP_TF>`
    - Transcription factor binding ChIP-seq
    - Fold-change over control
    - 128 bp
    - 181
    - 1709
*   - {attr}`METHYLATION
      <alphagenome.models.v2.dna_output.OutputType.METHYLATION>`
    - CpG methylation across 2 channels (bulk WGBS and single-cell methylome)
    - Methylated fraction
    - 1 bp
    - 185
    - 190
```

- `CHIP_HISTONE` covers 32 histone marks (30 in human and 11 in mouse) across
  43 channels per condition (accounting for both single- and paired-end
  sequencing), summed per 128 bp bin.
- `CHIP_TF` covers 778 transcription factor targets (772 in human and 47 in
  mouse) across 911 channels per condition (accounting for both single- and
  paired-end sequencing), summed per 128 bp bin.
````

````{tab-item} Splicing
```{list-table}
:header-rows: 1

*   - Output type
    - Measures
    - Units
    - Resolution
    - Biosamples
    - Tracks
*   - {attr}`SPLICE_SITES
      <alphagenome.models.v2.dna_output.OutputType.SPLICE_SITES>`
    - Condition-specific splice site classification (5 channels per condition)
    - Predicted probability
    - 1 bp
    - 227
    - 1135
*   - {attr}`SPLICE_SITE_USAGE
      <alphagenome.models.v2.dna_output.OutputType.SPLICE_SITE_USAGE>`
    - Fraction of transcripts using each donor or acceptor site
    - Fraction
    - 1 bp
    - 227
    - 560
*   - {attr}`SPLICE_JUNCTIONS
      <alphagenome.models.v2.dna_output.OutputType.SPLICE_JUNCTIONS>`
    - Split-read counts across candidate donor-acceptor pairings
    - Normalized junction signal
    - 1 bp
    - 227
    - 280
```

- `SPLICE_SITES` predicts 5 channels per condition (`donor +`, `acceptor +`,
  `donor -`, `acceptor -`, and `padding .`) derived from all splicing assays in
  that condition.
````

````{tab-item} 3D contacts and cells
```{list-table}
:header-rows: 1

*   - Output type
    - Measures
    - Units
    - Resolution
    - Biosamples
    - Tracks
*   - {attr}`CONTACT_MAPS
      <alphagenome.models.v2.dna_output.OutputType.CONTACT_MAPS>`
    - Pairwise 3D chromatin contact frequency (Micro-C and Hi-C), normalized for
      distance decay
    - Log-fold over distance expectation
    - 2048 bp
    - 73
    - 73
*   - {attr}`CELL_EMBEDDINGS
      <alphagenome.models.v2.dna_output.OutputType.CELL_EMBEDDINGS>`
    - 256-dimensional cell embeddings from the cell encoder
      (`cell_conditioning=`)
    - Embedding vector
    - Cell
    - Per call
    - NA
```
````

`````

### Track metadata columns

To inspect the metadata table for all observed human training tracks:

```python
observed = dna_model.observed_metadata(
    organism=dna_client.Organism.HOMO_SAPIENS
)
```

Each attribute on `observed` (such as
{attr}`~alphagenome.models.v2.dna_output.OutputMetadata.rna_seq`) is a
{class}`~pandas.DataFrame` with one row per track. The columns are:

```{list-table}
:header-rows: 1

*   - Column
    - Applies to
    - Description
*   - name
    - All track outputs
    - Track identifier (for example, `CL:0000047 polyA plus RNA-seq`)
*   - strand
    - All track outputs
    - `+` (forward), `-` (reverse), or `.` (unstranded)
*   - biosample_curie
    - All track outputs
    - Ontology CURIE or composite condition (for example,
      `UBERON:0000178_CL:0000907`)
*   - biosample_name
    - All track outputs
    - Plain-text biosample description (for example, `motor neuron`)
*   - imputation_mode
    - Predicted outputs
    - `observed`, `imputed_supported`, or `imputed_exploratory`
*   - data_pipeline, data_source
    - Most bulk outputs
    - Preprocessing pipeline and source consortium
*   - paper_doi
    - Single-cell outputs
    - Publication DOI for the source single-cell dataset
*   - histone_mark
    - {py:class}`chip_histone
      <alphagenome.models.v2.dna_output.OutputMetadata.chip_histone>` outputs
    - Profiled histone mark (for example, `H3K27ac`)
*   - transcription_factor
    - {py:class}`chip_tf
      <alphagenome.models.v2.dna_output.OutputMetadata.chip_tf>` outputs
    - Assayed transcription factor target
*   - splice_site_type
    - {py:class}`splice_sites
      <alphagenome.models.v2.dna_output.OutputMetadata.splice_sites>` outputs
    - `donor`, `acceptor`, or `padding`
```

Every row of `observed_metadata()` is an observed training track. On predicted
outputs (`TrackData.metadata` and `GeneExpressionData.track_metadata`) and in
variant scores, `imputation_mode` takes all three values.
{func}`~alphagenome.models.v2.variant_scorers.tidy_scores` renames `name` and
`strand` to `track_name` and `track_strand` to distinguish them from `gene_name`
and `gene_strand`.

<!-- mdformat off(Retain MyST dropdown syntax.) -->
:::{dropdown} How stranded (`+`, `-`) and unstranded (`.`) assays differ
:name: stranded-assays-primer

Strand-specific transcriptional assays (stranded RNA-seq, CAGE,
`SC_GENE_EXPRESSION_DENOISED`, and `SC_GENE_EXPRESSION`) have separate forward
(`+`) and reverse (`-`) strand channels to distinguish overlapping antisense
transcripts. Chromatin accessibility and unstranded assays (ATAC-seq, DNase-seq,
and unstranded GTEx RNA-seq) report a single unstranded (`.`) track per channel.

For `SPLICE_JUNCTIONS`, strand is stored per junction rather than per track row,
so track metadata contains half as many rows as the total junction tracks.
:::
<!-- mdformat on -->

### Observed and imputed tracks

AlphaGenome 2 predicts all channels of the requested output types for the chosen
condition and labels each track in `metadata['imputation_mode']`. To browse all
reference conditions and see which output types have observed training data in
each condition:

```python
conditions = dna_model.supported_conditions(
    organism=dna_client.Organism.HOMO_SAPIENS
)
```

Each row is a reference condition, and each output-type column records the best
imputation mode across that output type's channels (`'observed'` means at least
one channel was measured). See
[Navigating imputed tracks](user_guides/imputation) for details on imputation
modes and the [Choosing a condition tutorial](colabs/v2/choosing_a_condition)
for interactive filtering.

## AlphaGenome 1

AlphaGenome 1 (`alphagenome.models.v1`) predicts 11 output types across human
and mouse assays. See the Methods section of the AlphaGenome paper
{cite:p}`alphagenome` for dataset processing and normalization details.

### Output types

```{list-table}
:header-rows: 1
:name: model_metadata-table1-target

*   - Output type
    - Measures
    - Units
    - Resolution
    - Biosamples
    - Tracks
*   - {attr}`RNA_SEQ <alphagenome.models.v1.dna_output.OutputType.RNA_SEQ>`
    - Bulk RNA-seq coverage (polyA+ and total RNA, stranded and unstranded)
    - Normalized read signal
    - 1 bp
    - 285
    - 667
*   - {attr}`CAGE <alphagenome.models.v1.dna_output.OutputType.CAGE>`
    - Cap Analysis of Gene Expression (CAGE) 5′-cap signal, concentrated at
      transcription start sites (also detected along exons)
    - Normalized read signal
    - 1 bp
    - 264
    - 546
*   - {attr}`PROCAP <alphagenome.models.v1.dna_output.OutputType.PROCAP>`
    - Precision run-on sequencing and capping (PRO-cap) at start sites
    - Normalized read signal
    - 1 bp
    - 6
    - 12
*   - {attr}`DNASE <alphagenome.models.v1.dna_output.OutputType.DNASE>`
    - DNase I hypersensitive site cleavage (DNase-seq)
    - Normalized insertion signal
    - 1 bp
    - 305
    - 305
*   - {attr}`ATAC <alphagenome.models.v1.dna_output.OutputType.ATAC>`
    - Bulk transposase-accessible chromatin (ATAC-seq)
    - Normalized insertion signal
    - 1 bp
    - 167
    - 167
*   - {attr}`CHIP_HISTONE
      <alphagenome.models.v1.dna_output.OutputType.CHIP_HISTONE>`
    - Histone modification ChIP-seq across 24 marks
    - Fold-change over control
    - 128 bp
    - 219
    - 1116
*   - {attr}`CHIP_TF <alphagenome.models.v1.dna_output.OutputType.CHIP_TF>`
    - Transcription factor ChIP-seq across 43 targets
    - Fold-change over control
    - 128 bp
    - 163
    - 1617
*   - {attr}`SPLICE_SITES
      <alphagenome.models.v1.dna_output.OutputType.SPLICE_SITES>`
    - Condition-agnostic donor and acceptor probabilities (4 tracks total)
    - Predicted probability
    - 1 bp
    - NA
    - 4
*   - {attr}`SPLICE_SITE_USAGE
      <alphagenome.models.v1.dna_output.OutputType.SPLICE_SITE_USAGE>`
    - Fraction of transcripts using each donor or acceptor site
    - Fraction
    - 1 bp
    - 282
    - 734
*   - {attr}`SPLICE_JUNCTIONS
      <alphagenome.models.v1.dna_output.OutputType.SPLICE_JUNCTIONS>`
    - Split-read counts across candidate donor-acceptor pairings
    - Normalized junction signal
    - 1 bp
    - 282
    - 734
*   - {attr}`CONTACT_MAPS
      <alphagenome.models.v1.dna_output.OutputType.CONTACT_MAPS>`
    - Coarse-grained pairwise 3D contact frequency {cite:p}`zhou2022sequence`
    - Log-fold over distance expectation
    - 2048 bp
    - 12
    - 28
```

### Track metadata columns

To access track metadata for AlphaGenome 1 human outputs:

```python
output_metadata = dna_model.output_metadata(
    organism=dna_client.Organism.HOMO_SAPIENS
)
```

Each output type attribute (for example, {py:class}`output_metadata.rna_seq
<alphagenome.models.v1.dna_output.OutputMetadata.rna_seq>`) is a
{class}`~pandas.DataFrame` with `name`, `strand`, `ontology_curie`, and
`biosample_name` columns (plus type-specific columns such as `gtex_tissue` on
`rna_seq` for [GTEx](https://gtexportal.org/home/samplingSitePage)
{cite:t}`gtex2020gtex` tissues; see the
[AlphaGenome 1 quick start](colabs/v1/quick_start)).

```{note}
For GTEx tissue `'Brain - Cerebellar hemisphere'`, metadata uses
[`UBERON:0002245`](https://www.ebi.ac.uk/ols4/ontologies/uberon/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FUBERON_0002245)
(cerebellar hemisphere) instead of the UBERON ID listed in the
[GTEx documentation](https://gtexportal.org/home/samplingSitePage)
(`UBERON:0002037`).
```
