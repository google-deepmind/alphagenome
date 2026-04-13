# Copyright 2024 Google LLC.
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

import anndata
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from alphagenome.models import dna_output, interval_scorers, variant_scorers
from alphagenome.protos import dna_model_pb2


class VariantScorersTest(parameterized.TestCase):

  def test_aggregation_type_to_proto(self):
    expected = [
        dna_model_pb2.AggregationType.AGGREGATION_TYPE_DIFF_MEAN,
        dna_model_pb2.AggregationType.AGGREGATION_TYPE_DIFF_SUM,
        dna_model_pb2.AggregationType.AGGREGATION_TYPE_DIFF_SUM_LOG2,
        dna_model_pb2.AggregationType.AGGREGATION_TYPE_DIFF_LOG2_SUM,
        dna_model_pb2.AggregationType.AGGREGATION_TYPE_L2_DIFF,
        dna_model_pb2.AggregationType.AGGREGATION_TYPE_L2_DIFF_LOG1P,
        dna_model_pb2.AggregationType.AGGREGATION_TYPE_ACTIVE_MEAN,
        dna_model_pb2.AggregationType.AGGREGATION_TYPE_ACTIVE_SUM,
    ]

    for variant_scorer_aggregation_type, expected_proto in zip(
        variant_scorers.AggregationType, expected, strict=True
    ):
      self.assertEqual(
          variant_scorer_aggregation_type.to_proto(), expected_proto
      )

  def test_variant_scorers_to_proto(self):
    expected = [
        dna_model_pb2.VariantScorer(
            center_mask=dna_model_pb2.CenterMaskScorer(
                width=10_001,
                aggregation_type=(
                    dna_model_pb2.AggregationType.AGGREGATION_TYPE_DIFF_MEAN
                ),
                requested_output=dna_model_pb2.OUTPUT_TYPE_RNA_SEQ,
            )
        ),
        dna_model_pb2.VariantScorer(
            center_mask=dna_model_pb2.CenterMaskScorer(
                width=10_001,
                aggregation_type=(
                    dna_model_pb2.AggregationType.AGGREGATION_TYPE_ACTIVE_MEAN
                ),
                requested_output=dna_model_pb2.OUTPUT_TYPE_PROCAP,
            )
        ),
        dna_model_pb2.VariantScorer(
            center_mask=dna_model_pb2.CenterMaskScorer(
                width=None,
                aggregation_type=(
                    dna_model_pb2.AggregationType.AGGREGATION_TYPE_DIFF_LOG2_SUM
                ),
                requested_output=dna_model_pb2.OUTPUT_TYPE_ATAC,
            )
        ),
        dna_model_pb2.VariantScorer(
            gene_mask=dna_model_pb2.GeneMaskLFCScorer(
                requested_output=dna_model_pb2.OUTPUT_TYPE_RNA_SEQ,
            )
        ),
        dna_model_pb2.VariantScorer(
            gene_mask_splicing=dna_model_pb2.GeneMaskSplicingScorer(
                width=10_001,
                requested_output=dna_model_pb2.OUTPUT_TYPE_SPLICE_SITE_USAGE,
            )
        ),
        dna_model_pb2.VariantScorer(
            pa_qtl=dna_model_pb2.PolyadenylationScorer()
        ),
        dna_model_pb2.VariantScorer(
            splice_junction=dna_model_pb2.SpliceJunctionScorer()
        ),
    ]
    scorers = [
        variant_scorers.CenterMaskScorer(
            width=10_001,
            aggregation_type=(variant_scorers.AggregationType.DIFF_MEAN),
            requested_output=dna_output.OutputType.RNA_SEQ,
        ),
        variant_scorers.CenterMaskScorer(
            width=10_001,
            aggregation_type=(variant_scorers.AggregationType.ACTIVE_MEAN),
            requested_output=dna_output.OutputType.PROCAP,
        ),
        variant_scorers.CenterMaskScorer(
            width=None,
            aggregation_type=(variant_scorers.AggregationType.DIFF_LOG2_SUM),
            requested_output=dna_output.OutputType.ATAC,
        ),
        variant_scorers.GeneMaskLFCScorer(
            requested_output=dna_output.OutputType.RNA_SEQ,
        ),
        variant_scorers.GeneMaskSplicingScorer(
            width=10_001,
            requested_output=dna_output.OutputType.SPLICE_SITE_USAGE,
        ),
        variant_scorers.PolyadenylationScorer(),
        variant_scorers.SpliceJunctionScorer(),
    ]
    for variant_scorer, expected_proto in zip(scorers, expected):
      self.assertEqual(variant_scorer.to_proto(), expected_proto)

  def test_unsupported_output_type_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported requested output'):
      _ = variant_scorers.CenterMaskScorer(
          width=10_001,
          aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
          requested_output=dna_output.OutputType.CONTACT_MAPS,
      )

  def test_unsupported_width_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported width'):
      _ = variant_scorers.CenterMaskScorer(
          width=1,
          aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
          requested_output=dna_output.OutputType.RNA_SEQ,
      )


class TestTidyScores(absltest.TestCase):
  """Test tidying scores."""

  def setUp(self):
    """Set up test data."""
    super().setUp()

    # Possible scorers.
    self.gene_scorer = variant_scorers.GeneMaskLFCScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
    )
    self.center_mask_scorer = variant_scorers.CenterMaskScorer(
        width=10_001,
        aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
        requested_output=dna_output.OutputType.DNASE,
    )
    self.paqtl_scorer = variant_scorers.PolyadenylationScorer()
    self.junction_scorer = variant_scorers.SpliceJunctionScorer()
    self.splicing_scorer_ssu = variant_scorers.GeneMaskSplicingScorer(
        dna_output.OutputType.SPLICE_SITE_USAGE, width=101
    )
    self.splicing_scorer_ss = variant_scorers.GeneMaskSplicingScorer(
        dna_output.OutputType.SPLICE_SITES, width=101
    )
    self.gene_interval_scorer = interval_scorers.GeneMaskScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
        width=10_001,
        aggregation_type=interval_scorers.IntervalAggregationType.MEAN,
    )

    # Gene-centric scoring output.
    self.adata_gene_centric = anndata.AnnData(
        X=np.array([[1, 2], [3, 4], [5, 6]]),
        # 3 genes in vicinity of the variant.
        obs=pd.DataFrame({
            'gene_id': [
                'ENSG00000100342',
                'ENSG00000123456',
                'ENSG00000123457',
            ],
            'gene_name': ['GENE1', 'GENE2', 'GENE3'],
            'gene_type': ['protein_coding', 'lncRNA', 'protein_coding'],
            'strand': ['+', '-', '-'],
        }),
        # Two RNA_SEQ tracks.
        var=pd.DataFrame({
            'name': [
                'UBERON:0002107 total RNA-seq',
                'UBERON:0006566 gtex Heart_Left_Ventricle polyA plus RNA-seq',
            ],
            'strand': ['+', '-'],
            'Assay title': ['total RNA-seq', 'polyA plus RNA-seq'],
            'ontology_curie': ['UBERON:0002107', 'UBERON:0006566'],
            'biosample_name': ['liver', 'left ventricle myocardium'],
            'biosample_type': ['tissue', 'tissue'],
            'biosample_life_stage': ['adult', 'embryonic'],
            'gtex_tissue': ['Liver', 'Heart_Left_Ventricle'],
            'endedness': ['paired', 'paired'],
        }),
        uns={
            'variant': 'chr1:100:A>C',
            'interval': 'chr1:90-110',
            'endedness': ['paired', 'paired'],
            'variant_scorer': self.gene_scorer,
        },
        layers={
            'quantiles': np.array([[1, 0], [-1, 0], [1, -1]]),
        },
    )

    # Variant-centric scoring output.
    self.adata_variant_centric = anndata.AnnData(
        # 4 DNASE tracks.
        X=np.array([[5, 6, 7, 8]]),
        obs=pd.DataFrame(index=[0]),
        var=pd.DataFrame({
            'name': [
                'UBERON:0001 dnase',
                'UBERON:0002 dnase',
                'UBERON:0003 dnase',
                'UBERON:0004 dnase',
            ],
            'strand': ['-', '+', '-', '.'],
            'Assay title': ['DNase-seq'] * 4,
            'ontology_curie': [
                'UBERON:0001',
                'UBERON:0002',
                'UBERON:0003',
                'UBERON:0004',
            ],
            'biosample_name': [
                'T-cell',
                'endodermal cell',
                'natural killer cell',
                'CD4-positive, alpha-beta memory T cell',
            ],
            'biosample_type': [
                'primary_cell',
                'in_vitro_differentiated_cells',
                'primary_cell',
                'primary_cell',
            ],
            'data_source': ['gtex', 'gtex', 'gtex', 'gtex'],
        }),
        uns={
            'variant': 'chr2:200:G>T',
            'interval': 'chr2:190-210',
            'variant_scorer': self.center_mask_scorer,
        },
    )
    # PolyadenylationScorer RNA_SEQ scoring output. This output has the same
    # format as a gene-centric scoring output so it is not really a special
    # case.
    self.adata_paqtl = anndata.AnnData(
        X=np.array([
            [
                5,
                6,
                7,
            ],
            [8, 9, 10],
        ]),
        obs=pd.DataFrame({
            'gene_id': ['ENSG00000100342', 'ENSG00000123456'],
            'gene_name': ['GENE1', 'GENE2'],
            'gene_type': ['protein_coding', 'lncRNA'],
            'strand': ['+', '-'],
        }),
        var=pd.DataFrame({
            'name': [
                'CL:0000047 polyA plus RNA-seq',
                'CL:0000047 polyA plus RNA-seq',
                'UBERON:0018115 polyA plus RNA-seq',
            ],
            'strand': ['+', '-', '.'],
            'Assay title': [
                'polyA plus RNA-seq',
                'polyA plus RNA-seq',
                'polyA plus RNA-seq',
            ],
            'ontology_curie': [
                'CL:0000047',
                'CL:0000047',
                'UBERON:0018115',
            ],
            'biosample_name': [
                'neuronal stem cell',
                'neuronal stem cell',
                'left renal pelvis',
            ],
            'biosample_type': [
                'in_vitro_differentiated_cells',
                'in_vitro_differentiated_cells',
                'tissue',
            ],
            'genetically_modified': [True, True, False],
            'gtex_tissue': ['', '', ''],
        }),
        uns={
            'variant': 'chr3:300:A>G',
            'interval': 'chr3:290-310',
            'variant_scorer': self.paqtl_scorer,
        },
    )
    # SpliceJunctionScorer SPLICE_JUNCTIONS scoring output.
    self.adata_junction = anndata.AnnData(
        X=np.array([
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]),
        obs=pd.DataFrame({
            'gene_id': ['ENSG00000100342', 'ENSG00000123456'],
            'gene_name': ['GENE1', 'GENE2'],
            'gene_type': ['protein_coding', 'lncRNA'],
            'strand': ['+', '-'],
            'junction_Start': [0, 1],
            'junction_End': [1, 2],
        }),
        var=pd.DataFrame({
            'name': [
                'delta_psi5:Brain_Cerebellum',
                'delta_psi3:Brain_Cerebellum',
                'delta_counts:Brain_Cerebellum',
                'ref_counts:Brain_Cerebellum',
            ],
            'strand': ['.', '.', '.', '.'],
            'gtex_tissue': [
                'Brain_Cerebellum',
                'Brain_Cerebellum',
                'Brain_Cerebellum',
                'Brain_Cerebellum',
            ],
        }),
        uns={
            'variant': 'chr3:300:A>G',
            'interval': 'chr3:290-310',
            'variant_scorer': self.junction_scorer,
        },
    )
    # GeneMaskSplicingScorer SPLICE_SITE_USAGE scoring output.
    self.adata_splicing_ssu = anndata.AnnData(
        X=np.array([[1, 2], [3, 4]]),
        obs=pd.DataFrame({
            'gene_id': ['ENSG00000100342', 'ENSG00000123456'],
            'gene_name': ['GENE1', 'GENE2'],
            'gene_type': ['protein_coding', 'lncRNA'],
            'strand': ['+', '-'],
        }),
        var=pd.DataFrame({
            'name': [
                'usage_Brain_Cerebellum',
                'usage_Brain_Cerebellum',
            ],
            'strand': ['+', '-'],
            'ontology_curie': ['UBERON:0002037', 'UBERON:0002037'],
            'biosample_name': ['cerebellum', 'cerebellum'],
            'gtex_tissue': [
                'Brain_Cerebellum',
                'Brain_Cerebellum',
            ],
        }),
        uns={
            'variant': 'chr3:300:A>G',
            'interval': 'chr3:290-310',
            'variant_scorer': self.splicing_scorer_ssu,
        },
    )
    # GeneMaskSplicingScorer SPLICE_SITES scoring output.
    self.adata_splicing_ss = anndata.AnnData(
        X=np.array([[1, 2, 3, 4, 5]]),
        obs=pd.DataFrame({
            'gene_id': ['ENSG00000100342'],
            'gene_name': ['GENE1'],
            'gene_type': ['protein_coding'],
            'strand': ['+'],
        }),
        var=pd.DataFrame({
            # There are only ever these 4 tracks for this scorer.
            'name': [
                'donor',
                'acceptor',
                'donor',
                'acceptor',
                'other',
            ],
            'strand': ['+', '+', '-', '-', '.'],
        }),
        uns={
            'variant': 'chr3:300:A>G',
            'interval': 'chr3:290-310',
            'variant_scorer': self.splicing_scorer_ss,
        },
    )

    self.extended_metadata_columns = set([
        'Assay title',
        'ontology_curie',
        'biosample_name',
        'biosample_type',
        'gtex_tissue',
        'transcription_factor',
        'histone_mark',
    ])
    # Interval scoring output.
    self.adata_interval_centric = anndata.AnnData(
        X=np.array([[1, 2], [3, 4], [5, 6]]),
        # 3 genes in vicinity of the interval.
        obs=pd.DataFrame({
            'gene_id': [
                'ENSG00000100342',
                'ENSG00000123456',
                'ENSG00000123457',
            ],
            'gene_name': ['GENE1', 'GENE2', 'GENE3'],
            'gene_type': ['protein_coding', 'lncRNA', 'protein_coding'],
            'strand': ['+', '-', '-'],
        }),
        # Two RNA_SEQ tracks.
        var=pd.DataFrame({
            'name': [
                'UBERON:0002107 total RNA-seq',
                'UBERON:0006566 gtex Heart_Left_Ventricle polyA plus RNA-seq',
            ],
            'strand': ['+', '-'],
            'Assay title': ['total RNA-seq', 'polyA plus RNA-seq'],
            'ontology_curie': ['UBERON:0002107', 'UBERON:0006566'],
            'biosample_name': ['liver', 'left ventricle myocardium'],
            'biosample_type': ['tissue', 'tissue'],
            'gtex_tissue': ['Liver', 'Heart_Left_Ventricle'],
        }),
        uns={
            'interval': 'chr1:90-110',
            'interval_scorer': self.gene_interval_scorer,
        },
    )

  def test_tidy_anndata_gene_centric(self):
    """Test tidying a single gene centric AnnData object."""
    df = variant_scorers.tidy_anndata(
        self.adata_gene_centric, match_gene_strand=False
    )
    self.assertLen(df, 6)  # 3 genes by 2 tracks.

    df = variant_scorers.tidy_anndata(self.adata_gene_centric)
    self.assertLen(df, 3)  # Removes mismatched strands.
    expected = pd.DataFrame({
        'variant_id': ['chr1:100:A>C', 'chr1:100:A>C', 'chr1:100:A>C'],
        'scored_interval': ['chr1:90-110', 'chr1:90-110', 'chr1:90-110'],
        'gene_id': ['ENSG00000100342', 'ENSG00000123456', 'ENSG00000123457'],
        'gene_name': ['GENE1', 'GENE2', 'GENE3'],
        'gene_type': ['protein_coding', 'lncRNA', 'protein_coding'],
        'gene_strand': ['+', '-', '-'],
        'junction_Start': [None] * 3,
        'junction_End': [None] * 3,
        'output_type': ['RNA_SEQ', 'RNA_SEQ', 'RNA_SEQ'],
        'variant_scorer': [str(self.gene_scorer)] * 3,
        'track_name': [
            'UBERON:0002107 total RNA-seq',
            'UBERON:0006566 gtex Heart_Left_Ventricle polyA plus RNA-seq',
            'UBERON:0006566 gtex Heart_Left_Ventricle polyA plus RNA-seq',
        ],
        'track_strand': ['+', '-', '-'],
        'Assay title': [
            'total RNA-seq',
            'polyA plus RNA-seq',
            'polyA plus RNA-seq',
        ],
        'ontology_curie': [
            'UBERON:0002107',
            'UBERON:0006566',
            'UBERON:0006566',
        ],
        'biosample_name': [
            'liver',
            'left ventricle myocardium',
            'left ventricle myocardium',
        ],
        'biosample_type': ['tissue', 'tissue', 'tissue'],
        'biosample_life_stage': ['adult', 'embryonic', 'embryonic'],
        'gtex_tissue': [
            'Liver',
            'Heart_Left_Ventricle',
            'Heart_Left_Ventricle',
        ],
        'endedness': ['paired', 'paired', 'paired'],
        'raw_score': [1, 4, 6],
        'quantile_score': [1, 0, -1],
    })
    pd.testing.assert_frame_equal(df, expected)
    # Test without extended metadata.
    df = variant_scorers.tidy_anndata(
        self.adata_gene_centric, include_extended_metadata=False
    )
    pd.testing.assert_frame_equal(
        df,
        expected[
            [c for c in df.columns if c not in self.extended_metadata_columns]
        ],
    )

  def test_tidy_anndata_variant_centric(self):
    """Test tidying a single center mask AnnData object."""
    df = variant_scorers.tidy_anndata(self.adata_variant_centric)
    self.assertLen(df, 4)  # 4 tracks.
    # Strand filtering has no effect on variant-centric scoring.
    df = variant_scorers.tidy_anndata(
        self.adata_variant_centric, match_gene_strand=False
    )
    self.assertLen(df, 4)

    expected = pd.DataFrame({
        'variant_id': [
            'chr2:200:G>T',
            'chr2:200:G>T',
            'chr2:200:G>T',
            'chr2:200:G>T',
        ],
        'scored_interval': [
            'chr2:190-210',
            'chr2:190-210',
            'chr2:190-210',
            'chr2:190-210',
        ],
        'gene_id': [None, None, None, None],
        'gene_name': [None, None, None, None],
        'gene_type': [None, None, None, None],
        'gene_strand': [None, None, None, None],
        'junction_Start': [None] * 4,
        'junction_End': [None] * 4,
        'output_type': ['DNASE', 'DNASE', 'DNASE', 'DNASE'],
        'variant_scorer': [str(self.center_mask_scorer)] * 4,
        'track_name': [
            'UBERON:0001 dnase',
            'UBERON:0002 dnase',
            'UBERON:0003 dnase',
            'UBERON:0004 dnase',
        ],
        'track_strand': ['-', '+', '-', '.'],
        'Assay title': ['DNase-seq', 'DNase-seq', 'DNase-seq', 'DNase-seq'],
        'ontology_curie': [
            'UBERON:0001',
            'UBERON:0002',
            'UBERON:0003',
            'UBERON:0004',
        ],
        'biosample_name': [
            'T-cell',
            'endodermal cell',
            'natural killer cell',
            'CD4-positive, alpha-beta memory T cell',
        ],
        'biosample_type': [
            'primary_cell',
            'in_vitro_differentiated_cells',
            'primary_cell',
            'primary_cell',
        ],
        'data_source': ['gtex', 'gtex', 'gtex', 'gtex'],
        'raw_score': [5, 6, 7, 8],
    })
    pd.testing.assert_frame_equal(df, expected)
    # Test without extended metadata.
    df = variant_scorers.tidy_anndata(
        self.adata_variant_centric, include_extended_metadata=False
    )
    pd.testing.assert_frame_equal(
        df,
        expected[
            [c for c in df.columns if c not in self.extended_metadata_columns]
        ],
    )

  def test_tidy_anndata_paqtl(self):
    """Test tidying a single paQTL scoring output AnnData object."""
    df = variant_scorers.tidy_anndata(self.adata_paqtl)

    expected = pd.DataFrame({
        'variant_id': [
            'chr3:300:A>G',
            'chr3:300:A>G',
            'chr3:300:A>G',
            'chr3:300:A>G',
        ],
        'scored_interval': [
            'chr3:290-310',
            'chr3:290-310',
            'chr3:290-310',
            'chr3:290-310',
        ],
        'gene_id': [
            'ENSG00000100342',
            'ENSG00000123456',
            'ENSG00000100342',
            'ENSG00000123456',
        ],
        'gene_name': ['GENE1', 'GENE2', 'GENE1', 'GENE2'],
        'gene_type': ['protein_coding', 'lncRNA', 'protein_coding', 'lncRNA'],
        'gene_strand': ['+', '-', '+', '-'],
        'junction_Start': [None] * 4,
        'junction_End': [None] * 4,
        'output_type': ['RNA_SEQ', 'RNA_SEQ', 'RNA_SEQ', 'RNA_SEQ'],
        'variant_scorer': [str(self.paqtl_scorer)] * 4,
        'track_name': [
            'CL:0000047 polyA plus RNA-seq',
            'CL:0000047 polyA plus RNA-seq',
            'UBERON:0018115 polyA plus RNA-seq',
            'UBERON:0018115 polyA plus RNA-seq',
        ],
        'track_strand': ['+', '-', '.', '.'],
        'Assay title': [
            'polyA plus RNA-seq',
            'polyA plus RNA-seq',
            'polyA plus RNA-seq',
            'polyA plus RNA-seq',
        ],
        'ontology_curie': [
            'CL:0000047',
            'CL:0000047',
            'UBERON:0018115',
            'UBERON:0018115',
        ],
        'biosample_name': [
            'neuronal stem cell',
            'neuronal stem cell',
            'left renal pelvis',
            'left renal pelvis',
        ],
        'biosample_type': [
            'in_vitro_differentiated_cells',
            'in_vitro_differentiated_cells',
            'tissue',
            'tissue',
        ],
        'gtex_tissue': ['', '', '', ''],
        'genetically_modified': [True, True, False, False],
        'raw_score': [5, 9, 7, 10],
    })
    pd.testing.assert_frame_equal(df, expected)
    # Test without extended metadata.
    df = variant_scorers.tidy_anndata(
        self.adata_paqtl, include_extended_metadata=False
    )
    pd.testing.assert_frame_equal(
        df,
        expected[
            [c for c in df.columns if c not in self.extended_metadata_columns]
        ],
    )

  def test_tidy_anndata_interval_centric(self):
    """Test tidying a single gene centric interval scoring AnnData object."""
    df = variant_scorers.tidy_anndata(
        self.adata_interval_centric, match_gene_strand=False
    )
    self.assertLen(df, 6)  # 3 genes by 2 tracks.

    df = variant_scorers.tidy_anndata(self.adata_interval_centric)
    self.assertLen(df, 3)  # Removes mismatched strands.
    expected = pd.DataFrame({
        'scored_interval': ['chr1:90-110', 'chr1:90-110', 'chr1:90-110'],
        'gene_id': ['ENSG00000100342', 'ENSG00000123456', 'ENSG00000123457'],
        'gene_name': ['GENE1', 'GENE2', 'GENE3'],
        'gene_type': ['protein_coding', 'lncRNA', 'protein_coding'],
        'gene_strand': ['+', '-', '-'],
        'junction_Start': [None] * 3,
        'junction_End': [None] * 3,
        'output_type': ['RNA_SEQ', 'RNA_SEQ', 'RNA_SEQ'],
        'interval_scorer': [str(self.gene_interval_scorer)] * 3,
        'track_name': [
            'UBERON:0002107 total RNA-seq',
            'UBERON:0006566 gtex Heart_Left_Ventricle polyA plus RNA-seq',
            'UBERON:0006566 gtex Heart_Left_Ventricle polyA plus RNA-seq',
        ],
        'track_strand': ['+', '-', '-'],
        'Assay title': [
            'total RNA-seq',
            'polyA plus RNA-seq',
            'polyA plus RNA-seq',
        ],
        'ontology_curie': [
            'UBERON:0002107',
            'UBERON:0006566',
            'UBERON:0006566',
        ],
        'biosample_name': [
            'liver',
            'left ventricle myocardium',
            'left ventricle myocardium',
        ],
        'biosample_type': ['tissue', 'tissue', 'tissue'],
        'gtex_tissue': [
            'Liver',
            'Heart_Left_Ventricle',
            'Heart_Left_Ventricle',
        ],
        'raw_score': [1, 4, 6],
    })
    pd.testing.assert_frame_equal(df, expected)
    # Test without extended metadata.
    df = variant_scorers.tidy_anndata(
        self.adata_interval_centric, include_extended_metadata=False
    )
    pd.testing.assert_frame_equal(
        df,
        expected[
            [c for c in df.columns if c not in self.extended_metadata_columns]
        ],
    )

  def test_tidy_anndata_junction(self):
    """Test tidying a single splice junction scoring output AnnData object."""
    df = variant_scorers.tidy_anndata(self.adata_junction)
    expected = pd.DataFrame({
        'variant_id': [
            'chr3:300:A>G',
            'chr3:300:A>G',
            'chr3:300:A>G',
            'chr3:300:A>G',
            'chr3:300:A>G',
            'chr3:300:A>G',
            'chr3:300:A>G',
            'chr3:300:A>G',
        ],
        'scored_interval': [
            'chr3:290-310',
            'chr3:290-310',
            'chr3:290-310',
            'chr3:290-310',
            'chr3:290-310',
            'chr3:290-310',
            'chr3:290-310',
            'chr3:290-310',
        ],
        'gene_id': [
            'ENSG00000100342',
            'ENSG00000123456',
            'ENSG00000100342',
            'ENSG00000123456',
            'ENSG00000100342',
            'ENSG00000123456',
            'ENSG00000100342',
            'ENSG00000123456',
        ],
        'gene_name': [
            'GENE1',
            'GENE2',
            'GENE1',
            'GENE2',
            'GENE1',
            'GENE2',
            'GENE1',
            'GENE2',
        ],
        'gene_type': [
            'protein_coding',
            'lncRNA',
            'protein_coding',
            'lncRNA',
            'protein_coding',
            'lncRNA',
            'protein_coding',
            'lncRNA',
        ],
        'gene_strand': ['+', '-', '+', '-', '+', '-', '+', '-'],
        'junction_Start': [0, 1, 0, 1, 0, 1, 0, 1],
        'junction_End': [1, 2, 1, 2, 1, 2, 1, 2],
        'output_type': [
            'SPLICE_JUNCTIONS',
            'SPLICE_JUNCTIONS',
            'SPLICE_JUNCTIONS',
            'SPLICE_JUNCTIONS',
            'SPLICE_JUNCTIONS',
            'SPLICE_JUNCTIONS',
            'SPLICE_JUNCTIONS',
            'SPLICE_JUNCTIONS',
        ],
        'variant_scorer': [str(self.junction_scorer)] * 8,
        'track_name': [
            'delta_psi5:Brain_Cerebellum',
            'delta_psi5:Brain_Cerebellum',
            'delta_psi3:Brain_Cerebellum',
            'delta_psi3:Brain_Cerebellum',
            'delta_counts:Brain_Cerebellum',
            'delta_counts:Brain_Cerebellum',
            'ref_counts:Brain_Cerebellum',
            'ref_counts:Brain_Cerebellum',
        ],
        'track_strand': ['.', '.', '.', '.', '.', '.', '.', '.'],
        'gtex_tissue': ['Brain_Cerebellum'] * 8,
        'raw_score': [5, 9, 6, 10, 7, 11, 8, 12],
    })
    pd.testing.assert_frame_equal(df, expected)
    # Test without extended metadata.
    df = variant_scorers.tidy_anndata(
        self.adata_junction, include_extended_metadata=False
    )
    pd.testing.assert_frame_equal(
        df,
        expected[
            [c for c in df.columns if c not in self.extended_metadata_columns]
        ],
    )

  def test_tidy_anndata_splicing_ssu(self):
    """Test tidying a single splicing ssu scoring output AnnData object."""
    df = variant_scorers.tidy_anndata(self.adata_splicing_ssu)
    expected = pd.DataFrame({
        'variant_id': [
            'chr3:300:A>G',
            'chr3:300:A>G',
        ],
        'scored_interval': [
            'chr3:290-310',
            'chr3:290-310',
        ],
        'gene_id': [
            'ENSG00000100342',
            'ENSG00000123456',
        ],
        'gene_name': [
            'GENE1',
            'GENE2',
        ],
        'gene_type': ['protein_coding', 'lncRNA'],
        'gene_strand': ['+', '-'],
        'junction_Start': [None] * 2,
        'junction_End': [None] * 2,
        'output_type': [
            'SPLICE_SITE_USAGE',
            'SPLICE_SITE_USAGE',
        ],
        'variant_scorer': [str(self.splicing_scorer_ssu)] * 2,
        'track_name': [
            'usage_Brain_Cerebellum',
            'usage_Brain_Cerebellum',
        ],
        'track_strand': [
            '+',
            '-',
        ],
        'ontology_curie': ['UBERON:0002037', 'UBERON:0002037'],
        'biosample_name': ['cerebellum', 'cerebellum'],
        'gtex_tissue': [
            'Brain_Cerebellum',
            'Brain_Cerebellum',
        ],
        'raw_score': [1, 4],
    })
    pd.testing.assert_frame_equal(df, expected)

  def test_tidy_anndata_splicing_sites(self):
    """Test tidying a single splicing sites scoring output AnnData object."""
    df = variant_scorers.tidy_anndata(self.adata_splicing_ss)
    expected = pd.DataFrame({
        'variant_id': ['chr3:300:A>G', 'chr3:300:A>G', 'chr3:300:A>G'],
        'scored_interval': ['chr3:290-310', 'chr3:290-310', 'chr3:290-310'],
        'gene_id': ['ENSG00000100342', 'ENSG00000100342', 'ENSG00000100342'],
        'gene_name': ['GENE1', 'GENE1', 'GENE1'],
        'gene_type': ['protein_coding', 'protein_coding', 'protein_coding'],
        'gene_strand': ['+', '+', '+'],
        'junction_Start': [None] * 3,
        'junction_End': [None] * 3,
        'output_type': ['SPLICE_SITES', 'SPLICE_SITES', 'SPLICE_SITES'],
        'variant_scorer': [str(self.splicing_scorer_ss)] * 3,
        'track_name': ['donor', 'acceptor', 'other'],
        'track_strand': ['+', '+', '.'],
        'raw_score': [1, 2, 5],
    })
    pd.testing.assert_frame_equal(df, expected)
    # Test without extended metadata.
    df = variant_scorers.tidy_anndata(
        self.adata_splicing_ss, include_extended_metadata=False
    )
    pd.testing.assert_frame_equal(
        df,
        expected[
            [c for c in df.columns if c not in self.extended_metadata_columns]
        ],
    )

  def test_tidy_anndata_empty(self):
    adata = anndata.AnnData(
        X=np.ones((0, 3)), obs=pd.DataFrame(columns=['gene_id', 'strand'])
    )
    result = variant_scorers.tidy_anndata(adata)
    pd.testing.assert_frame_equal(result, pd.DataFrame())

  def test_tidy_lists_of_anndata(self):
    # List of AnnDatas.
    anndata_list = [self.adata_gene_centric, self.adata_variant_centric]
    df_list = variant_scorers.tidy_scores(anndata_list)
    self.assertLen(df_list, 7)

    # List of list of AnnDatas.
    anndata_list = [[self.adata_gene_centric], [self.adata_variant_centric]]
    df_nested_list = variant_scorers.tidy_scores(anndata_list)
    self.assertLen(df_nested_list, 7)
    # The exact nesting pattern doesn't matter, the result is equivalent.
    pd.testing.assert_frame_equal(df_list, df_nested_list)

    # Longer list of list of AnnDatas.
    anndata_list = [
        [self.adata_gene_centric, self.adata_variant_centric],
        [self.adata_paqtl, self.adata_junction, self.adata_interval_centric],
    ]
    df_longer_nested_list = variant_scorers.tidy_scores(anndata_list)
    self.assertLen(df_longer_nested_list, 22)

    # Scores are empty.
    anndata_list = [anndata.AnnData(X=np.ones((0, 3)))]
    df_empty_list = variant_scorers.tidy_scores(anndata_list)
    self.assertIsNone(df_empty_list)


if __name__ == '__main__':
  absltest.main()
