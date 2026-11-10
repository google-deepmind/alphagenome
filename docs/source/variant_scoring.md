# How variant scoring works

AlphaGenome 2 scores a variant by predicting reference (REF) and alternate (ALT)
profiles in one condition, restricting both to a spatial mask, and summarizing
their difference per track (and per gene, junction, or CpG site where
applicable).

## Scoring pipeline

Each variant scorer transforms paired REF and ALT predictions into scalar scores
through four steps:

`````{tab-set}

````{tab-item} 1. Predict REF and ALT
Variant scoring begins by running forward passes on the REF and ALT sequences
across the requested genomic interval and output type (such as `RNA_SEQ`,
`ATAC`, or `SPLICE_SITES`).

```{figure} /_static/variant_scoring_ref_alt.png
:width: 500px
:alt: Predict REF and ALT profiles for a given output type.
:name: variant-scoring-1
```

:::{admonition} Indel coordinate alignment
:class: note

For insertions and deletions (indels), the ALT profile is aligned to REF
genomic coordinates before comparison. Inserted bases are summarized by taking
the maximum predicted signal across the inserted segment, while deleted bases
are assigned zero signal in ALT so both profiles share identical positions.
:::
````

````{tab-item} 2. Apply spatial mask
A spatial mask restricts scoring to the positions relevant to the assay,
discarding predictions outside the masked coordinates:

- **Variant-centered windows** capture local cis-regulatory changes around the
  variant: 501 bp for `ATAC`, `DNASE`, `CAGE`, and `CHIP_TF`; 1201 bp for
  `SC_ATAC` and shared CpG sites in `METHYLATION`; 2001 bp for broad histone
  marks in `CHIP_HISTONE`; and 1 Mb for 3D interactions in `CONTACT_MAPS`.
- **Gene annotations** from a GTF reference restrict scoring to exons
  (`RNA_SEQ`), gene bodies including introns (`SPLICE_SITES`,
  `SPLICE_SITE_USAGE`), top splice sites (`SPLICE_JUNCTIONS`), or 400-bp windows
  around 3' cleavage sites (`POLYADENYLATION`).

```{figure} /_static/variant_scoring_spatial_mask.png
:width: 500px
:alt: Apply a variant-centered or gene-centered spatial mask.
:name: variant-scoring-2
```
````

````{tab-item} 3. Aggregate and compare
Within the spatial mask, each scorer reduces positions along the sequence axis
(mean, sum, or max), applies any optional log or {math}`\ell_2` scaling, and
compares ALT against REF to produce one scalar value per track (and per gene or
CpG site when applicable).

```{figure} /_static/variant_scoring_spatial_compute.png
:width: 500px
:alt: Aggregate spatially across the mask and compute ALT minus REF.
:name: variant-scoring-3
```

:::{dropdown} How AggregationType names encode order of operations
Options in {class}`~alphagenome.models.v2.variant_scorers.AggregationType`
list operations from right to left, with the rightmost operation applied first
to the masked predictions:

- **DIFF_LOG2_SUM** sums predicted coverage across the window first, transforms
  with {math}`\log_2(\text{sum} + 1)`, and then takes ALT minus REF.
- **DIFF_SUM_LOG2** transforms each position with
  {math}`\log_2(\text{value} + 1)`, sums across the mask, and then takes ALT
  minus REF.
:::
````

````{tab-item} 4. Tidy and summarize
Raw scoring outputs are returned as `AnnData` objects per variant and scorer.
Calling {func}`~alphagenome.models.v2.variant_scorers.tidy_scores` flattens
them into a tidy table with `raw_score`, track metadata (including
`imputation_mode`), optional strand matching (`match_gene_strand`), and optional
filtering to protein-coding genes.

```{figure} /_static/variant_scoring_aggregate.png
:width: 500px
:alt: Select or aggregate scores across output tracks.
:name: variant-scoring-4
```
````

`````

## Recommended variant scorers

{data}`~alphagenome.models.v2.variant_scorers.RECOMMENDED_VARIANT_SCORERS`
provides 23 preconfigured scorers across expression, chromatin, splicing,
methylation, 3D contacts, and active-allele baseline activity.
{func}`~alphagenome.models.v2.variant_scorers.get_recommended_scorers` returns
the subset supported for a given organism, and includes
`SC_GENE_EXPRESSION_DENOISED` only when `cell_conditioning=True`:

`````{tab-set}

````{tab-item} Expression and polyadenylation
(gene-expression-rna-seq)=
Gene-expression scorers
({class}`~alphagenome.models.v2.variant_scorers.GeneMaskLFCScorer`) are
**signed** (ALT minus REF); `POLYADENYLATION`
({class}`~alphagenome.models.v2.variant_scorers.PolyadenylationScorer`) is
**unsigned**:

```{list-table}
:header-rows: 1

*   - Key
    - Mask
    - Score
*   - `RNA_SEQ`
    - Gene exons
    - {math}`\log(\text{mean}(\text{ALT}) + 0.001)`
      {math}`- \log(\text{mean}(\text{REF}) + 0.001)`
*   - `SC_GENE_EXPRESSION_DENOISED`
    - Gene row (`GeneExpressionData`)
    - {math}`\log(\text{ALT} + 0.001) - \log(\text{REF} + 0.001)` on gene-level
      CPM (one channel per strand; requires `cell_conditioning=`)
*   - `SC_GENE_EXPRESSION`
    - Gene row (`GeneExpressionData`)
    - {math}`\log(\text{ALT} + 0.001) - \log(\text{REF} + 0.001)` on gene-level
      CPM (one track per chemistry, suspension type, and strand)
*   - `POLYADENYLATION`
    - 400-bp windows at 3' cleavage sites
    - Max absolute log-fold change of distal/proximal PAS usage ratios
      {cite:p}`borzoi`
```
````

````{tab-item} Chromatin, TF, and methylation
All chromatin, transcription factor, and methylation scorers are **signed**
(positive when ALT increases signal or methylation relative to REF). Spatial
tracks use {class}`~alphagenome.models.v2.variant_scorers.CenterMaskScorer`;
methylation uses
{class}`~alphagenome.models.v2.variant_scorers.MethylationScorer`:

```{list-table}
:header-rows: 1

*   - Key
    - Mask
    - Score
*   - `ATAC`, `DNASE`, `CAGE`, `CHIP_TF`
    - 501 bp centered on variant
    - {math}`\log_2((\text{sum}(\text{ALT})+1)/(\text{sum}(\text{REF})+1))`
*   - `SC_ATAC`
    - 1201 bp centered on variant
    - {math}`\log_2((\text{sum}(\text{ALT})+1)/(\text{sum}(\text{REF})+1))`
*   - `CHIP_HISTONE`
    - 2001 bp centered on variant
    - {math}`\log_2((\text{sum}(\text{ALT})+1)/(\text{sum}(\text{REF})+1))`
*   - `METHYLATION`
    - 1201 bp centered on variant (shared CpGs)
    - Per-CpG
      {math}`\operatorname{logit}(\text{ALT})-\operatorname{logit}(\text{REF})`
      of the methylated fraction, for CpG sites present in both alleles (CpGs
      created or destroyed by the variant are omitted)
```
````

````{tab-item} RNA splicing and 3D contacts
(splicing-splice-site-usage)=
All splicing and 3D contact scorers
({class}`~alphagenome.models.v2.variant_scorers.GeneMaskSplicingScorer`,
{class}`~alphagenome.models.v2.variant_scorers.SpliceJunctionScorer`, and
{class}`~alphagenome.models.v2.variant_scorers.ContactMapScorer`) are
**unsigned** magnitudes:

```{list-table}
:header-rows: 1

*   - Key
    - Mask
    - Score
*   - `SPLICE_SITES`
    - Gene body (exons and introns)
    - {math}`\max\lvert\text{ALT} - \text{REF}\rvert` across donor and acceptor
      probabilities
*   - `SPLICE_SITE_USAGE`
    - Gene body (exons and introns)
    - {math}`\max\lvert\text{ALT} - \text{REF}\rvert` across splice site usage
      fractions
*   - `SPLICE_JUNCTIONS`
    - Top-k annotated and predicted splice sites
    - {math}`\max\lvert\log(\text{ALT}) - \log(\text{REF})\rvert` across
      junction read counts
*   - `CONTACT_MAPS`
    - 1 Mb centered on variant
    - Mean {math}`\lvert\text{ALT} - \text{REF}\rvert` across interactions with
      the variant bin
```
````

````{tab-item} Active-allele scorers
Active-allele scorers
({class}`~alphagenome.models.v2.variant_scorers.GeneMaskActiveScorer` and
{class}`~alphagenome.models.v2.variant_scorers.CenterMaskScorer`) quantify
baseline signal intensity rather than allelic change by taking the maximum of
the aggregated REF and ALT signals within the mask:

```{list-table}
:header-rows: 1

*   - Key
    - Mask
    - Score
*   - `RNA_SEQ_ACTIVE`
    - Gene exons
    - {math}`\max(\text{mean}(\text{ALT}), \text{mean}(\text{REF}))`
*   - `SC_GENE_EXPRESSION_ACTIVE`
    - Gene row (`GeneExpressionData`)
    - {math}`\max(\text{sum}(\text{ALT}), \text{sum}(\text{REF}))`
*   - `ATAC_ACTIVE`, `DNASE_ACTIVE`, `CAGE_ACTIVE`, `CHIP_TF_ACTIVE`
    - 501 bp centered on variant
    - {math}`\max(\text{sum}(\text{ALT}), \text{sum}(\text{REF}))`
*   - `SC_ATAC_ACTIVE`
    - 1201 bp centered on variant
    - {math}`\max(\text{sum}(\text{ALT}), \text{sum}(\text{REF}))`
*   - `CHIP_HISTONE_ACTIVE`
    - 2001 bp centered on variant
    - {math}`\max(\text{sum}(\text{ALT}), \text{sum}(\text{REF}))`
```
````

`````

## Next steps

<!-- mdformat off(Retain sphinx-design grid cards.) -->

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Scoring a variant
:link: colabs/v2/score_variant
:link-type: doc

Run recommended scorers on single variants and batches, tidy outputs, and
inspect top cell types and genes.
:::

:::{grid-item-card} Analyzing a splicing variant
:link: colabs/v2/splicing_variant
:link-type: doc

Compare splice site probabilities, splice site usage, and splice junction
read counts on pathogenic variants.
:::

:::{grid-item-card} Variant scorers API reference
:link: api/models_v2.html#variant-scorers
:link-type: url

Inspect all scorer classes, supported mask widths, and
{func}`~alphagenome.models.v2.variant_scorers.tidy_scores` options.
:::

::::

<!-- mdformat on -->
