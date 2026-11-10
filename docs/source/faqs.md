# FAQs

## Inputs and coordinates

(genome-builds)=

### Which genome assemblies are supported?

Human assembly hg38 (`GRCh38.p13.genome.fa`) and mouse assembly mm10
(`GRCm38.p6.genome.fa`). To convert coordinates from other assemblies (such as
hg19), use UCSC [LiftOver](https://genome.ucsc.edu/cgi-bin/hgLiftOver).

(variant-position-is-1-based)=

### What coordinate convention do Interval and Variant use?

{class}`~alphagenome.data.genome.Interval` uses half-open 0-based coordinates
(`[start, end)`), matching standard Python slicing. For compatibility with VCF
and dbSNP, {class}`~alphagenome.data.genome.Variant` takes a 1-based
{attr}`~alphagenome.data.genome.Variant.position` and exposes 0-based bounds via
`start` (`position - 1`) and `end`.

(supported-sequence-lengths)=

### Which sequence lengths are supported, and how does resize work?

AlphaGenome 2 accepts four input widths: `SEQUENCE_LENGTH_16KB` (16,384 bp),
`SEQUENCE_LENGTH_100KB` (131,072 bp), `SEQUENCE_LENGTH_500KB` (524,288 bp), and
`SEQUENCE_LENGTH_1MB` (1,048,576 bp; recommended for best accuracy);
{meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_interval_gradients`
and
{meth}`~alphagenome.models.v2.dna_client.DnaClient.predict_interval_gradients_batch`
accept only `SEQUENCE_LENGTH_100KB` and `SEQUENCE_LENGTH_1MB`. Calling
{func}`~alphagenome.data.genome.Interval.resize` expands or crops around the
interval center using real flanking reference genome sequence rather than
zero-padding.

(arbitrary-sequences)=

### Can I make predictions for arbitrary or non-human/mouse DNA sequences?

Yes, {func}`~alphagenome.models.v2.dna_client.DnaClient.predict_sequence`
accepts any sequence composed of `ACGTN` at a supported length. However, the
models were trained on the human and mouse reference genomes, and predictions
for other species, synthetic constructs, large structural variants, or heavily
N-padded sequences are not benchmarked.

## Conditions and outputs

(condition-vs-cell-conditioning)=

### Should I use a reference condition or expression conditioning?

Start with a reference condition (`condition=`). Reference conditions in
`dna_model.supported_conditions()` span a broad reference across human and mouse
tissues, primary cell types, and cell lines, and cover most use cases.
`supported_conditions()` lists human conditions by default; pass
`organism=dna_client.Organism.MUS_MUSCULUS` for mouse.

Expression conditioning (`cell_conditioning=` plus `cell_condition_name=`) lets
you apply the model to your own human scRNA-seq data without explicitly
fine-tuning it on your dataset: the model encodes up to 256 cells and predicts
for that cell state. It generalizes across many cell types, but if your data
differ substantially from the training distribution, the model may fail to
capture that variation.

See [Choosing a condition](colabs/v2/choosing_a_condition) and
[Conditioning on your own single-cell data](colabs/v2/conditioning_on_single_cell_data).

(imputation-modes)=

### What do observed, imputed_supported, and imputed_exploratory mean?

Every predicted track carries an `imputation_mode` label in `.metadata`:
`observed` (measured in training), `imputed_supported` (not measured, but
imputable from a measured modality in that condition), or `imputed_exploratory`
(not benchmarked; treat as a hypothesis). See
[Navigating imputed tracks](user_guides/imputation).

(multiple-conditions)=

### How do I compare predictions or scores across multiple conditions?

Each call to `predict_interval`, `predict_variant`, or `score_variant` takes one
condition (`condition=` or `cell_conditioning=`). To compare reference
conditions or expression-conditioning inputs across items, pass matching
sequences (`conditions=` or `cell_conditionings=` plus `cell_condition_names=`)
to {func}`~alphagenome.models.v2.dna_client.DnaClient.predict_variants`,
{func}`~alphagenome.models.v2.dna_client.DnaClient.score_variants`, or the other
batch methods.

(ag1-vs-ag2)=

### AlphaGenome 1 vs. AlphaGenome 2: which model should I use?

Use **AlphaGenome 2** (`alphagenome.models.v2`) for new analyses. Use
**AlphaGenome 1** (`alphagenome.models.v1`) to reproduce results from the
AlphaGenome paper {cite:p}`alphagenome`, score variants across all tissues and
cell types at once without choosing a condition, or predict PRO-cap. See
[Getting started](getting_started) and
[Migrating from AlphaGenome 1](user_guides/ag1_vs_ag2).

(strands-and-saving)=

### How are DNA strands labeled?

Track metadata records strand in `strand` as `+` (forward), `-` (reverse), or
`.` (unstranded; for example, ATAC-seq, DNase-seq, and unstranded GTEx RNA-seq).

(saving-outputs)=

### How do I save outputs?

Flatten variant score `AnnData` objects with
{func}`~alphagenome.models.v2.variant_scorers.tidy_scores` and export with
`DataFrame.to_csv(...)`, export `GeneExpressionData` with `.to_anndata()`, or
save `TrackData.values` NumPy arrays with `np.savez_compressed(...)`.

## Variant scoring

(indels-and-reference-bases)=

### Are insertions and deletions (indels) supported, and must reference_bases match the genome?

Yes. For scoring, the ALT prediction is aligned to REF coordinates following
SpliceAI {cite:p}`spliceai` (see [How variant scoring works](variant_scoring)).
`Variant` does not normalize indels; use
{func}`~alphagenome.data.genome.normalize_variant` to left-align them. In
{func}`~alphagenome.models.v2.dna_client.DnaClient.predict_variant`,
`reference_bases` and `alternate_bases` substitute directly into the reference
sequence without requiring `reference_bases` to match the hg38/mm10 allele.

(scoring-splicing-variants)=

### How should I score splicing variants?

Combine the maximum absolute scores across `SPLICE_SITES`, `SPLICE_SITE_USAGE`,
and `SPLICE_JUNCTIONS` (`/ 5`) on conditions with observed splicing or
full-length RNA-seq data. See
[Analyzing a splicing variant](colabs/v2/splicing_variant) for the full
walkthrough.

(quantile-score-nan)=

### Does AlphaGenome 2 report quantile scores?

No. AlphaGenome 2 scores report `raw_score` only; quantile scores calibrated
against common variants are available in AlphaGenome 1. See
{ref}`Migrating from AlphaGenome 1 <quantile-scores-ag1>`.

## Usage, licensing, and support

(terms-and-commercial)=

### What terms of use apply, and is commercial access available?

AlphaGenome is governed by the
[AlphaGenome Terms of Service](https://alphagenome.google/terms): direct API
outputs and AlphaGenome Atlas scores are for non-commercial research use only,
must not be used to train machine learning models, and must not be used for
clinical decision-making. For commercial and enterprise workflows, access
AlphaGenome on Google Cloud via
[Model Garden on Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/alphagenome)
or contact <alphagenome-cloud-external@google.com>.

(citing-alphagenome)=

### How should I cite AlphaGenome?

If you use AlphaGenome in your research, please cite:

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

```bibtex
@article{alphagenome,
  title={Advancing regulatory variant effect prediction with {AlphaGenome}},
  author={Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
  journal={Nature},
  volume={649},
  number={8099},
  pages={1206--1218},
  year={2026},
  doi={10.1038/s41586-025-10014-0},
  publisher={Nature Publishing Group UK London}
}
```

<!-- enableFinding(SNIPPET_INVALID_LANGUAGE) -->

If you use the AlphaGenome Atlas website or scores, please cite:

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

```bibtex
@article{alphagenome_atlas,
  title={{AlphaGenome} Atlas: in silico mutagenesis of the entire human genome improves prioritization and interpretation of non-coding variants},
  author={Cheng, Jun and Taylor, Kyle R. and Nicolaisen, Lauren and Pan, Joshua and Bycroft, Clare and Perino, Matteo and Ward, Tom and Hawkes, Gareth and Covil, Laura E. and Weilert, Melanie and Thomas, Raina W. and Latysheva, Natasha and Hirschmann, J. Maile and Chen, Xi Dawn and Beaumont, Robin N. and Chundru, V Kartik and Weedon, Michael N. and Bourdareau, Simon and Chu, Hoyin and Hariharan, Dhavanthi and Kagohara, Thais and Ten\'{o}rio, Lucas and Ushigome, Yosuke and Shearer, Courtney A. and Ikica, Barbara and Fang, Ada and Naciri, Mouad and Johnston, Victoria and Green, Richard and Wong, Lai Hong and Dutordoir, Vincent and Mottram, Anne and Gayoso, Adam and Arvaniti, Eirini and Rehm, Heidi L. and Chen, Fei and Lareau, Caleb A. and Wright, Caroline F. and O'Donnell-Luria, Anne and Zeitlinger, Julia and Kohli, Pushmeet and Avsec, {\v Z}iga},
  year = {2026},
  doi = {10.64898/2026.09.16.26363192},
  publisher = {Cold Spring Harbor Laboratory Press},
  journal = {medRxiv}
}
```

<!-- enableFinding(SNIPPET_INVALID_LANGUAGE) -->

(support-and-contact)=

### Where do I report issues or ask questions?

Report software issues on
[GitHub](https://github.com/google-deepmind/alphagenome) and post usage
questions on the
[AlphaGenome community forum](https://www.alphagenomecommunity.com). For direct
inquiries, email <alphagenome@google.com>.
