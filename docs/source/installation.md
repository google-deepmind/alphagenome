# Installation

The `alphagenome` package is a lightweight Python client for the hosted
AlphaGenome models and the AlphaGenome Atlas. It runs on Linux, macOS, and
Windows, and no local GPU is required.

Model predictions and Atlas queries both need a free
[AlphaGenome API key](https://alphagenome.google/api).

## Install the package

Install the latest release from [PyPI](https://pypi.org/project/alphagenome)
inside a virtual environment:

`````{tab-set}
````{tab-item} pip
:sync: pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install alphagenome
```
````

````{tab-item} uv
:sync: uv

```bash
uv venv
source .venv/bin/activate
uv pip install alphagenome
```

Or add directly to an existing `uv` project:

```bash
uv add alphagenome
```
````

````{tab-item} conda
:sync: conda

```bash
conda create -n alphagenome python -y
conda activate alphagenome
pip install alphagenome
```
````

````{tab-item} From source
:sync: source

Clone the [GitHub repository](https://github.com/google-deepmind/alphagenome)
for an editable install:

```bash
git clone https://github.com/google-deepmind/alphagenome.git
cd alphagenome
pip install -e .
```

To pull upstream changes later:

```bash
git pull
pip install -e .
```
````
`````

## Google Colab

Every tutorial notebook starts with an install cell that installs `alphagenome`
(and any extra packages that notebook needs) in your Colab runtime:

```python
! pip install alphagenome
```

### Add API key to secrets

To authenticate from Google Colab notebooks without pasting your API key into
code cells:

1.  Open your Google Colab notebook and click the **Secrets** icon (key symbol)
    in the left sidebar.
2.  Click **+ Add new secret** and name it `ALPHA_GENOME_API_KEY`.
3.  Paste your [AlphaGenome API key](https://alphagenome.google/api) into the
    **Value** field.
4.  Toggle **Notebook access** on for the current notebook.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->

```{figure} /_static/secrets.png
:width: 600px
:alt: Adding ALPHA_GENOME_API_KEY in the Google Colab Secrets sidebar panel.
:name: secrets-screenshot
```
<!-- mdformat on -->

## Local API key configuration

When running Python scripts or Jupyter notebooks locally, export
`ALPHA_GENOME_API_KEY` in your shell environment (or add it to `~/.zshrc` /
`~/.bashrc`):

```bash
export ALPHA_GENOME_API_KEY="YOUR_API_KEY"
```

`colab_utils.get_api_key()` reads this environment variable first and falls back
to Colab Secrets, so the same code runs locally and in Colab.

## Verify your installation

Confirm that the package is installed and your API key works:

```python
from alphagenome import colab_utils
from alphagenome.models.v2 import dna_client

dna_model = dna_client.create(colab_utils.get_api_key())
print(len(dna_model.supported_conditions()), 'supported human conditions')
```

## Next steps

*   Continue to [Getting started](getting_started) to choose a tool and walk
    through an AlphaGenome 2 workflow.
*   Follow the interactive [Tutorials](tutorials/index) covering reference
    conditions, expression conditioning (`cell_conditioning=`), and imputed
    tracks.
