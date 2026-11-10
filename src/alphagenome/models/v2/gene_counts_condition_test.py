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

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.models.v2 import gene_counts_condition

from alphagenome.protos.v2 import dna_model_pb2

import anndata
import numpy as np
import scipy.sparse


class GeneCountsConditionTest(parameterized.TestCase):

  def test_from_anndata_dense(self):
    # 2 cells, 3 genes (human IDs)
    x = np.array([[1, 0, 3], [0, 5, 0]], dtype=np.int32)
    var_names = ["ENSG00000139618.15", "ENSG00000141510", "ENSG00000100342"]
    adata = anndata.AnnData(X=x, var={"gene_id": var_names})
    adata.var_names = var_names

    proto = gene_counts_condition.from_anndata(adata)

    # Check gene IDs are cleaned and match
    self.assertEqual(
        proto.gene_ids,
        ["ENSG00000139618", "ENSG00000141510", "ENSG00000100342"],
    )
    self.assertLen(proto.cells, 2)

    # Cell 0: non-zero at 0 (count 1) and 2 (count 3)
    self.assertEqual(list(proto.cells[0].gene_indices), [0, 2])
    self.assertEqual(list(proto.cells[0].gene_counts), [1, 3])

    # Cell 1: non-zero at 1 (count 5)
    self.assertEqual(list(proto.cells[1].gene_indices), [1])
    self.assertEqual(list(proto.cells[1].gene_counts), [5])

  def test_from_anndata_sparse_csr(self):
    # Same as above but sparse CSR
    x_sparse = scipy.sparse.csr_matrix([[1, 0, 3], [0, 5, 0]], dtype=np.float32)
    var_names = ["ENSG00000139618", "ENSG00000141510", "ENSG00000100342"]
    adata = anndata.AnnData(X=x_sparse)
    adata.var_names = var_names

    proto = gene_counts_condition.from_anndata(adata)

    self.assertEqual(
        proto.gene_ids,
        ["ENSG00000139618", "ENSG00000141510", "ENSG00000100342"],
    )
    self.assertLen(proto.cells, 2)
    self.assertEqual(list(proto.cells[0].gene_indices), [0, 2])
    self.assertEqual(list(proto.cells[0].gene_counts), [1, 3])

  def test_from_anndata_mouse_ids(self):
    x = np.array([[10]], dtype=np.int32)
    var_names = ["ENSMUSG00000026354"]  # Mouse Ensembl gene ID
    adata = anndata.AnnData(X=x)
    adata.var_names = var_names

    proto = gene_counts_condition.from_anndata(adata)
    self.assertEqual(proto.gene_ids, ["ENSMUSG00000026354"])

  def test_from_anndata_empty_cells(self):
    adata = anndata.AnnData(X=np.empty((0, 5), dtype=np.int32))
    with self.assertRaisesRegex(ValueError, "must have at least one cell"):
      gene_counts_condition.from_anndata(adata)

  def test_from_anndata_empty_genes(self):
    adata = anndata.AnnData(X=np.empty((2, 0), dtype=np.int32))
    with self.assertRaisesRegex(ValueError, "must have at least one gene"):
      gene_counts_condition.from_anndata(adata)

  def test_from_anndata_invalid_id(self):
    adata = anndata.AnnData(X=np.array([[1]], dtype=np.int32))
    # Missing digits, not starting with ENS
    adata.var_names = ["NOT_A_GENE_ID"]
    with self.assertRaisesRegex(ValueError, "not a valid ENSEMBL ID"):
      gene_counts_condition.from_anndata(adata)

  def test_from_anndata_duplicate_ids(self):
    adata = anndata.AnnData(X=np.array([[1, 2]], dtype=np.int32))
    adata.var_names = [
        "ENSG00000100342.1",
        "ENSG00000100342.2",
    ]  # Both clean to same ID
    with self.assertRaisesRegex(ValueError, "duplicate gene ID detected"):
      gene_counts_condition.from_anndata(adata)

  def test_from_anndata_negative_count(self):
    adata = anndata.AnnData(X=np.array([[1, -1]], dtype=np.int32))
    adata.var_names = ["ENSG00000139618", "ENSG00000141510"]
    with self.assertRaisesRegex(ValueError, "count must be > 0"):
      gene_counts_condition.from_anndata(adata)

  def test_from_anndata_zero_count_cell(self):
    # One cell has no counts
    adata = anndata.AnnData(X=np.array([[1, 0], [0, 0]], dtype=np.int32))
    adata.var_names = ["ENSG00000139618", "ENSG00000141510"]
    with self.assertRaisesRegex(
        ValueError, "cell at index 1 has 0 non-zero genes"
    ):
      gene_counts_condition.from_anndata(adata)

  def test_from_anndata_with_target(self):
    x = np.array([[10]], dtype=np.int32)
    var_names = ["ENSG00000139618"]
    adata = anndata.AnnData(X=x)
    adata.var_names = var_names

    proto = dna_model_pb2.CellConditioningInput()
    # Pre-populate with some garbage to verify it gets Cleared
    proto.gene_ids.append("GARBAGE")

    returned_proto = gene_counts_condition.from_anndata(adata, target=proto)

    self.assertIs(returned_proto, proto)
    self.assertEqual(proto.gene_ids, ["ENSG00000139618"])
    self.assertLen(proto.cells, 1)

  def test_subsample_all_groups(self):
    x = scipy.sparse.csr_matrix(np.ones((10, 2), dtype=np.int32))
    obs = {"cell_type": ["typeA"] * 6 + ["typeB"] * 4}
    adata = anndata.AnnData(X=x, obs=obs)

    subsampled = gene_counts_condition.subsample(
        adata, n_cells=5, label_key="cell_type", seed=42
    )
    self.assertEqual(set(subsampled.keys()), {"typeA", "typeB"})
    self.assertEqual(subsampled["typeA"].shape, (5, 2))
    self.assertEqual(subsampled["typeB"].shape, (4, 2))
    self.assertTrue(scipy.sparse.issparse(subsampled["typeA"].X))

  def test_subsample_with_label_values(self):
    x = np.arange(12, dtype=np.int32).reshape(6, 2)
    obs = {"tissue": ["lung", "lung", "spleen", "spleen", "bm", "bm"]}
    adata = anndata.AnnData(X=x, obs=obs)

    # Sequence of label_values (including a non-existent one which is skipped)
    sub_seq = gene_counts_condition.subsample(
        adata,
        1,
        "tissue",
        ["spleen", "lung", "liver"],
        seed=0,
    )
    self.assertEqual(list(sub_seq.keys()), ["spleen", "lung"])
    self.assertEqual(sub_seq["spleen"].shape[0], 1)
    self.assertEqual(sub_seq["lung"].shape[0], 1)

  def test_subsample_reproducible_with_seed(self):
    x = np.arange(20, dtype=np.int32).reshape(10, 2)
    obs = {"cell_type": ["typeA"] * 10}
    adata = anndata.AnnData(X=x, obs=obs)

    sub1 = gene_counts_condition.subsample(
        adata, n_cells=4, label_key="cell_type", seed=123
    )
    sub2 = gene_counts_condition.subsample(
        adata, n_cells=4, label_key="cell_type", seed=123
    )
    np.testing.assert_array_equal(sub1["typeA"].X, sub2["typeA"].X)

  def test_subsample_errors(self):
    adata = anndata.AnnData(
        X=np.ones((2, 2), dtype=np.int32), obs={"cell_type": ["a", "b"]}
    )
    with self.assertRaises(KeyError):
      gene_counts_condition.subsample(adata, n_cells=5, label_key="missing")

    with self.assertRaisesRegex(ValueError, "n_cells must be positive"):
      gene_counts_condition.subsample(adata, n_cells=0, label_key="cell_type")


if __name__ == "__main__":
  absltest.main()
