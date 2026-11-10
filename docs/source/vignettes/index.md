# Vignettes

```{toctree}
:maxdepth: 1
:hidden:

../colabs/v2/finemapping_ets2
../colabs/v2/tal1_oncogenic_mutations
../colabs/v2/enhancer_to_gene_bcl11a
```

Vignettes are end-to-end biological case studies combining AlphaGenome 2 variant
scoring, track predictions, input gradients, and *in silico* mutagenesis with
published experimental results to interpret disease loci.

<!-- mdformat off(Turn off mdformat to retain grid card syntax.) -->

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} Fine-mapping the *ETS2* locus
:link: ../colabs/v2/finemapping_ets2
:link-type: doc

Prioritize the likely causal variant (rs2836882), its distal target gene
(*ETS2*), and the myeloid cell types that drive the chr21q22 inflammatory bowel
disease association.
:::

:::{grid-item-card} Dissecting non-coding *TAL1* mutations in T-ALL
:link: ../colabs/v2/tal1_oncogenic_mutations
:link-type: doc

Predict how T-ALL insertions and SNVs near *TAL1* remodel RNA-seq, DNase-seq,
and histone ChIP-seq tracks, and compare alleles from individuals with T-ALL
against length-matched random alleles at the same positions.
:::

:::{grid-item-card} Mapping lineage-specific enhancers at the *BCL11A* locus
:link: ../colabs/v2/enhancer_to_gene_bcl11a
:link-type: doc

Call and cluster enhancer-to-gene links across 8 hematopoietic differentiation
lineages with input gradients and chromatin accessibility, recovering the
erythroid-specific `+55/+58/+62 kb` intronic enhancers of *BCL11A*.
:::

::::

<!-- mdformat on -->
