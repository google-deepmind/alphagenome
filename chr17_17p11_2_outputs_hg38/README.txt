17p11.2 Locus Analysis Results (hg38)
=====================================

Target locus (hg38): chr17:16700000-24100000:.
Equivalent hg19: chr17:14,890,000-22,230,032
Total chunks processed: 14
Chunk size: 524,288 bp

Files generated:
- complete_outputs.pkl: Full AlphaGenome output objects
- *_rna_seq.npy: RNA-seq prediction arrays
- *_dnase.npy: DNase hypersensitivity arrays
- *_atac.npy: ATAC-seq arrays (if available)

Load numpy arrays with: np.load("filename.npy")
Load complete outputs with: pickle.load(open("complete_outputs.pkl", "rb"))
