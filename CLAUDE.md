# AlphaGenome Project Guide

## Project Overview
AlphaGenome (by Google DeepMind) is a Python SDK for interacting with and visualizing genomic models. This fork integrates SNP (single nucleotide polymorphism) data with AlphaGenome's genomic tracks (ATAC, DNase, RNA-seq) to analyze genetic variants, map SNPs to genes, and score patient variants — with a focus on the 17p11.2 chromosomal locus.

## Technology Stack
- **Language**: Python 3.10+ (supports 3.10–3.13)
- **Build System**: Hatch (hatchling) with uv installer
- **Core Dependencies** (from pyproject.toml):
  - NumPy, Pandas, SciPy for computation
  - gRPC + Protocol Buffers for model serving
  - Matplotlib, Seaborn for visualization
  - anndata, intervaltree, pyarrow for genomic data
  - absl-py for flags/testing
- **Code Style**: pyink (Google style), 80-char lines, 2-space indent
- **Testing**: hatch test (pytest-based), with `conftest.py` configuring absl FLAGS
- **Data Formats**: `.npy` (numpy arrays), `.pkl` (pickle), `.tsv`/`.csv` (tabular), `.proto` (protobuf)
- **Genomic Tools**: LiftOver (hg19 to hg38 conversion)

## Project Structure

```
alphagenome/
├── src/alphagenome/           # Core SDK package
│   ├── data/                  # Genomic data loading
│   │   ├── genome.py          # Reference genome handling
│   │   ├── transcript.py      # Transcript data
│   │   ├── track_data.py      # Track data (ATAC, DNase, RNA-seq)
│   │   ├── junction_data.py   # Splice junction data
│   │   ├── fold_intervals.py  # Fold interval operations
│   │   ├── gene_annotation.py # Gene annotations
│   │   └── ontology.py        # Tissue/cell type ontology
│   ├── models/                # ML models and scoring
│   │   ├── dna_model.py       # Core DNA model
│   │   ├── dna_client.py      # gRPC client for model inference
│   │   ├── dna_output.py      # Model output handling
│   │   ├── variant_scorers.py # Variant effect scoring
│   │   ├── interval_scorers.py # Interval-based scoring
│   │   ├── track_data_utils.py # Track data utilities
│   │   └── junction_data_utils.py
│   ├── interpretation/        # Model interpretation
│   │   └── ism.py             # In-silico mutagenesis
│   ├── protos/                # Protocol Buffer definitions
│   │   ├── dna_model.proto
│   │   ├── dna_model_service.proto
│   │   └── tensor.proto
│   ├── visualization/         # Plotting
│   │   ├── plot.py
│   │   ├── plot_components.py
│   │   └── plot_transcripts.py
│   ├── colab_utils.py         # Colab notebook helpers
│   ├── tensor_utils.py        # Tensor manipulation utilities
│   └── typing.py              # Custom type definitions
├── colabs/                    # Jupyter notebooks for Google Colab
│   ├── quick_start.ipynb
│   ├── batch_variant_scoring.ipynb
│   ├── variant_scoring_ui.ipynb
│   └── ... (more notebooks)
├── scripts/                   # Utility scripts
│   └── process_gtf.py
├── docs/                      # Sphinx documentation
├── Knowledge-Graph-UI/        # Knowledge graph visualization (Next.js + Neo4j)
│   └── WARP.md                # Technical guide (architecture, env vars, components)
│
│ # Custom analysis scripts (project-specific)
├── analyze_patient_snps.py         # Main patient SNP analysis
├── snp2gene_mapping_pipeline.py    # SNP-to-gene mapping
├── snp2genotype_comprehensive_analysis.py  # Comprehensive genotype analysis
├── acquire_17p11_2_locus_hg38.py   # 17p11.2 locus data acquisition
├── alphagenome_snp_analyzer.py     # SNP analysis helper
├── alphagenome_interval_checker.py # Interval checking utility
├── snp_fasta_loader.py             # FASTA sequence loading for SNPs
├── display_all_genes_snps.py       # Gene/SNP display utility
│
│ # Data files
├── chr17_17p11_2_outputs_hg38/     # 17p11.2 locus data (14 chunks)
├── all_genes_list.txt              # Gene reference list
├── all_snps_list.txt               # SNP reference list
├── gene_patient_matrix.csv         # Gene-patient associations
├── snp2genotype_*_variants.tsv     # Variant categorization files
│
├── pyproject.toml                  # Build config & dependencies
├── Dockerfile                      # Docker container definition
├── alphagenome-docker.sh           # Docker run helper
├── hatch_build.py                  # Custom build hook (proto compilation)
└── conftest.py                     # Test configuration (absl FLAGS)
```

## Key Analysis Workflows

### 17p11.2 Locus Analysis
- 7Mb region on chromosome 17: `chr17:14890000-21890000`
- Data stored in 14 chunks with ATAC, DNase, and RNA-seq tracks
- Acquired via `acquire_17p11_2_locus_hg38.py`

### Patient SNP Analysis
```bash
python analyze_patient_snps.py
```

### SNP-to-Gene Mapping
```bash
python snp2gene_mapping_pipeline.py
```

### Comprehensive Genotype Analysis
```bash
python snp2genotype_comprehensive_analysis.py
```

## MCP Tool Integrations

This project has access to three MCP (Model Context Protocol) tool servers for biomedical research queries. These complement the genomic analysis workflows by connecting variants, genes, and loci to literature, drugs, and clinical data.

### bioRxiv / medRxiv
Search and retrieve preprints from bioRxiv (biology) and medRxiv (medical/health).
- **Search preprints** by category and date range (27 categories including genomics, genetics, neuroscience, cancer biology)
- **Get preprint details** by DOI (full abstract, authors, PDF URL, funding, publication status)
- **Search by funder** using ROR IDs (e.g., NIH `021nxhr62`, NSF `01cwqze88`)
- **Track published preprints** — find which preprints became peer-reviewed journal articles
- **Usage**: Search for recent research on 17p11.2 genes (e.g., RAI1, PMP22), Smith-Magenis syndrome, or Charcot-Marie-Tooth disease

### ChEMBL
Query the ChEMBL database for compounds, drug targets, and bioactivity data.
- **Compound search** by name, ChEMBL ID, or SMILES structure (with similarity/substructure search)
- **Drug search** by therapeutic indication (e.g., find approved drugs for neuropathy)
- **Target search** by protein name, gene symbol (e.g., EGFR, BRAF), or organism
- **Bioactivity data** — IC50, Ki, EC50 measurements for compound-target interactions
- **Mechanism of action** — how drugs interact with targets (inhibitor, agonist, etc.)
- **ADMET properties** — drug-likeness assessment (Lipinski rules, QED scores)
- **Usage**: Check if genes in the 17p11.2 locus have known drug targets; find compounds active against proteins encoded by SNP-affected genes

### ClinicalTrials.gov
Search the ClinicalTrials.gov database for clinical trials.
- **Search trials** by condition, intervention, sponsor, location, phase, and status
- **Get trial details** by NCT ID (eligibility criteria, endpoints, locations, contacts)
- **Search by sponsor** — find all trials from a specific company or institution
- **Search by eligibility** — match patient criteria (age, sex, biomarkers)
- **Find investigators** — search PIs and research sites by condition, institution, or location
- **Analyze endpoints** — primary/secondary outcome measures for a trial or across a therapeutic area
- **Usage**: Find clinical trials related to conditions associated with 17p11.2 variants (e.g., Smith-Magenis syndrome, hereditary neuropathy); identify investigators studying these conditions

## Knowledge-Graph-UI

A **Next.js 14 web application** for exploring Neo4j-based bioinformatics knowledge graphs. Serves as the interactive visualization frontend for AlphaGenome's SNP/gene data.

- **Tech Stack**: Next.js 14 (App Router), React, TypeScript, MUI, Cytoscape.js, Neo4j
- **Purpose**: Schema-driven UI for searching nodes (genes, proteins, diseases, compounds), visualizing relationships and shortest paths, and running enrichment analysis
- **Current Status**: Documentation only (`Knowledge-Graph-UI/WARP.md`); full source code in separate upstream repo
- **Multi-tenant**: Same codebase supports multiple deployments (CFDE, Enrichr KG, lncRNAlyzr, Harmonizome) via JSON schema configuration

### Key Environment Variables
| Variable | Purpose |
|---|---|
| `NEO4J_URL` / `NEO4J_V5_URL` | Neo4j database connection |
| `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j credentials |
| `NEXT_PUBLIC_SCHEMA` | External schema URL (overrides `public/schema.json`) |
| `NEXT_PUBLIC_ENRICHR_URL` | Enrichr API base URL |

### API Endpoints
- `GET /api/schema` — fetch UI schema
- `GET /api/initialize` — precompute colors/edges metadata
- `GET /api/knowledge_graph` — query graph with filters (Cypher-based)
- `POST /api/enrichment` — enrichment analysis via Enrichr
- `GET /api/docs` — OpenAPI spec

### Relationship to AlphaGenome
- Visualizes SNP-to-gene mappings and patient variant networks
- Can ingest SIF network files (e.g., the NIH merged SNP-to-gene mapping `.sif` file in the repo root)
- Provides interactive exploration of genes, SNPs, and regulatory tracks from the 17p11.2 region

### Development
```bash
cd Knowledge-Graph-UI
npm install
npm run dev    # localhost:3000
```

## Development

### Setup
```bash
# Install in development mode
pip install -e .[dev]

# Or use hatch
hatch env create
```

### Running Tests
```bash
# Run tests with hatch
hatch test

# Run tests across all Python versions
hatch test --all
```

### Code Formatting & Linting
```bash
# Check formatting (pyink, Google style)
hatch run check:format

# Lint
hatch run check:lint

# Both
hatch run check:all
```

### Docker
```bash
# Build and run
./alphagenome-docker.sh
```

## Environment
- Requires `.env` file with `ALPHAGENOME_API_KEY` for gRPC model inference
- Python 3.10+ required

## When Working on This Codebase
- **Architecture**: Respect the separation between data loading (`data/`), model inference (`models/`), interpretation (`interpretation/`), and visualization (`visualization/`)
- **Memory**: Be mindful of large numpy array operations — the 17p11.2 data has 14 chunks of multi-track arrays
- **Coordinates**: Maintain compatibility with hg38 coordinates; use LiftOver when handling hg19 data
- **Code Style**: Follow Google Python style — pyink formatting, 2-space indent, 80-char line limit
- **Testing**: Each module has a corresponding `*_test.py` file; maintain test coverage
- **Protobuf**: Proto files in `src/alphagenome/protos/` — compiled Python bindings are generated by `hatch_build.py`
- **Pipeline Impact**: When modifying analysis scripts, consider downstream effects on the SNP analysis pipeline and output files
- **Do not commit**: `.env`, `__pycache__/`, `*.pyc`, Excel temp files (`~$*`)
