# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for preparing and converting AnnData cell conditioning inputs."""

from collections.abc import Sequence
import re

import anndata
import numpy as np
import scipy.sparse

from alphagenome.protos.v2 import dna_model_pb2

_ENSEMBL_REGEX = re.compile(r"^ENS[A-Z]*G[0-9]{11}$")


def clean_gene_id(gene_id: str) -> str:
  """Strips version suffix from ENSEMBL ID (e.g. ENSG000001.2 -> ENSG000001)."""
  return gene_id.split(".")[0]


def is_valid_ensembl_id(gene_id: str) -> bool:
  """Checks if a gene ID matches the ENSEMBL format (without version)."""
  return bool(_ENSEMBL_REGEX.match(gene_id))


def from_anndata(
    adata: anndata.AnnData,
    target: dna_model_pb2.CellConditioningInput | None = None,
) -> dna_model_pb2.CellConditioningInput:
  """Converts AnnData rows to a CellConditioningInput proto.

  This constructs the proto using bulk array slices for performance.

  Args:
    adata: An AnnData object. `adata.X` contains the count matrix, and
      `adata.var_names` contains the gene IDs.
    target: Optional CellConditioningInput proto to populate in-place (this
      avoids copying the potentially large submessage). If provided, the message
      will be Cleared and populated in-place without copying.

  Returns:
    A CellConditioningInput proto.
  """
  if adata.shape[0] == 0:
    raise ValueError("must have at least one cell")
  if adata.shape[1] == 0:
    raise ValueError("must have at least one gene")

  # Clean and validate gene IDs.
  gene_ids = [clean_gene_id(g) for g in adata.var_names]
  for g in gene_ids:
    if not is_valid_ensembl_id(g):
      raise ValueError(f"not a valid ENSEMBL ID: {g}")

  if len(set(gene_ids)) != len(gene_ids):
    raise ValueError("duplicate gene ID detected in var_names")

  # Convert matrix to CSR for fast slicing.
  x = adata.X
  if scipy.sparse.issparse(x):
    if not isinstance(x, scipy.sparse.csr_matrix):
      csr = x.tocsr()  # pyrefly: ignore[missing-attribute]
    else:
      csr = x
  else:
    csr = scipy.sparse.csr_matrix(x)

  # Ensure types match proto fields (int32).
  indices = csr.indices.astype(np.int32)
  data = csr.data.astype(np.int32)
  indptr = csr.indptr

  if np.any(data <= 0):
    raise ValueError("count must be > 0")

  if target is not None:
    conditioning = target
    conditioning.Clear()
    conditioning.gene_ids.extend(gene_ids)
  else:
    conditioning = dna_model_pb2.CellConditioningInput(gene_ids=gene_ids)

  for i in range(adata.shape[0]):
    s, e = indptr[i], indptr[i + 1]
    num_nonzero = e - s
    if num_nonzero == 0:
      raise ValueError(f"cell at index {i} has 0 non-zero genes")

    conditioning.cells.add(
        gene_indices=indices[s:e],
        gene_counts=data[s:e],
    )

  return conditioning


def subsample(
    adata: anndata.AnnData,
    n_cells: int,
    label_key: str,
    label_values: Sequence[str] | None = None,
    seed: int | None = None,
    copy: bool = True,
) -> dict[str, anndata.AnnData]:
  """Subsamples an AnnData object into a mapping of condition label to AnnData.

  For each label in `adata.obs[label_key]`, extracts up to `n_cells`
  observations by sampling integer positional indices from `adata.obs` first and
  then slicing `adata`, avoiding unnecessary copies or densification of `.X`.

  Args:
    adata: Input single-cell AnnData object.
    n_cells: Maximum number of cells to sample per label.
    label_key: Column name in `adata.obs` representing the condition or cell
      type label.
    label_values: Optional sequence of label values in `adata.obs[label_key]` to
      include. If None, all non-empty groups in `adata.obs[label_key]` are
      included.
    seed: Optional random seed for reproducible subsampling.
    copy: Whether to return a copy of each subsetted AnnData slice.

  Returns:
    A dictionary mapping condition label to subsampled AnnData with at most
    `n_cells` observations.

  Raises:
    KeyError: If `label_key` is not found in `adata.obs`.
    ValueError: If `n_cells` <= 0.
  """
  if n_cells <= 0:
    raise ValueError(f"n_cells must be positive, got {n_cells}")

  rng = np.random.default_rng(seed)
  grouped_indices = adata.obs.groupby(label_key, observed=True).indices

  target_labels = (
      list(grouped_indices.keys()) if label_values is None else label_values
  )

  result: dict[str, anndata.AnnData] = {}
  for label_val in target_labels:
    obs_indices = grouped_indices.get(label_val)
    if obs_indices is None or len(obs_indices) == 0:
      continue
    if len(obs_indices) > n_cells:
      selected = np.sort(rng.choice(obs_indices, size=n_cells, replace=False))
    else:
      selected = obs_indices
    sub_adata = adata[selected]
    result[str(label_val)] = sub_adata.copy() if copy else sub_adata

  return result
