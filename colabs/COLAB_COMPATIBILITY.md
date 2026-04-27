# Google Colab Compatibility

The AlphaGenome SDK now includes built-in support for running code both in Google Colab and local environments using conditional imports.

## Overview

The `alphagenome.colab_utils` module now exports Google Colab-specific objects that work seamlessly in both environments:

- `data_table` - For enhanced dataframe display (Colab) or standard pandas display (local)
- `files` - For file upload/download operations (Colab) or mock operations (local)
- `IN_COLAB` - Boolean flag indicating if running in Google Colab

## Usage

### Basic Import Pattern

Instead of importing directly from `google.colab`:

```python
# ❌ Old way (breaks in local environment)
from google.colab import data_table, files
```

Import from `alphagenome.colab_utils`:

```python
# ✅ New way (works in both Colab and local)
from alphagenome.colab_utils import data_table, files, IN_COLAB
```

### Complete Example

```python
from alphagenome.colab_utils import data_table, files, IN_COLAB
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from alphagenome.visualization import plot_components
import pandas as pd

# The data_table and files objects work in both environments
print(f"Running in Google Colab: {IN_COLAB}")

# Configure data display
if IN_COLAB:
    data_table.DataTable.max_rows = 100_000
    data_table.enable_dataframe_formatter()
else:
    # Pandas display options are automatically set
    # but you can override them if needed
    pd.set_option('display.max_rows', 100_000)

# Create and display data
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
print(df)

# File operations
if IN_COLAB:
    # In Colab, this actually downloads the file
    files.download('output.csv')
else:
    # In local environment, prints helpful message
    files.download('output.csv')
```

## Behavior Differences

### In Google Colab

When `IN_COLAB == True`:
- `data_table` provides enhanced interactive dataframe display
- `data_table.enable_dataframe_formatter()` enables Colab's data table view
- `files.download(filename)` triggers browser download
- `files.upload()` opens file upload dialog

### In Local Environment

When `IN_COLAB == False`:
- `data_table` is a mock object that silently ignores method calls
- Pandas display options are automatically configured for better console output
- `files.download(filename)` prints a message about where to find the file
- `files.upload()` prints a message to use file paths directly

## Updating Existing Notebooks

To update existing notebooks that use `google.colab` imports:

1. Find all instances of `from google.colab import`
2. Replace with `from alphagenome.colab_utils import`
3. Add `IN_COLAB` to the import if you need environment-specific logic

Example:

```python
# Before
from google.colab import data_table, files
data_table.DataTable.max_rows = 100_000

# After
from alphagenome.colab_utils import data_table, files, IN_COLAB
data_table.DataTable.max_rows = 100_000  # Works in both environments!
```

## API Key Handling

The existing `get_api_key()` function continues to work in both environments:

```python
from alphagenome.colab_utils import get_api_key
from alphagenome.models import dna_client

# Tries environment variable first, then Colab secrets
api_key = get_api_key('ALPHA_GENOME_API_KEY')
model = dna_client.create(api_key)
```

## Benefits

1. **Write once, run anywhere**: Same code works in Colab and local Jupyter/Python
2. **No try/except blocks needed**: Conditional logic handled internally
3. **Graceful degradation**: Mock objects provide sensible fallback behavior
4. **Better local development**: Automatically configures pandas for readable console output

## Technical Details

The module checks for Google Colab availability at import time:

```python
try:
    from google.colab import data_table, files, userdata
    IN_COLAB = True
    # Configure Colab-specific settings
except ImportError:
    IN_COLAB = False
    # Provide mock objects and configure pandas
```

Mock objects implement the same interface as their Colab counterparts but provide appropriate behavior for local environments.