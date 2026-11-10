# Copyright 2025 Google LLC.
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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import gene_expression_data
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.data import ontology
from alphagenome.data import track_data
from alphagenome.models.v2 import dna_output
from alphagenome.protos.v1 import dna_model_pb2 as dna_model_pb2_v1
from alphagenome.protos.v2 import dna_model_pb2
import numpy as np
import pandas as pd

_EXAMPLE_ONTOLOGY_TERM = 'UBERON:0002037'

_ATAC_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar', 'baz', 'buz', 'fux'],
        strand=['.'] * 5,
        imputation_mode=[
            'observed',
            'imputed',
            'observed',
            'imputed',
            'observed',
        ],
        padding=[False, False, False, False, True],
        ontology_curie=[
            _EXAMPLE_ONTOLOGY_TERM,
            _EXAMPLE_ONTOLOGY_TERM,
            _EXAMPLE_ONTOLOGY_TERM,
            'UBERON:0000005',
            None,
        ],
        extra=[1] * 5,
    )
)
_CAGE_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar', 'baz'],
        strand=['.'] * 3,
        imputation_mode=['observed', 'imputed', 'observed'],
        padding=[False] * 3,
        ontology_curie=[_EXAMPLE_ONTOLOGY_TERM] * 3,
        extra=[1] * 3,
    )
)
_DNASE_METADATA = pd.DataFrame(
    dict(
        name=['foo'],
        strand=['.'],
        imputation_mode=['observed'],
        padding=[False],
        extra=[1],
    )
)
_RNA_SEQ_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'foo', 'baz', 'buz'],
        strand=['+', '-', '.', '.'],
        imputation_mode=['observed', 'observed', 'imputed', 'observed'],
        ontology_curie=[_EXAMPLE_ONTOLOGY_TERM] * 4,
        padding=[False] * 4,
        extra=[1] * 4,
    )
)
_CHIP_HISTONE_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar', 'baz', 'buz', 'quz'],
        strand=['.'] * 5,
        padding=[False] * 5,
        extra=[1] * 5,
    )
)
_SPLICING_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar', 'foo', 'bar'],
        strand=['+'] * 2 + ['-'] * 2,
        padding=[False] * 4,
        extra=[1] * 4,
    )
)
_SC_GENE_EXPRESSION_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar'],
        strand=['+', '-'],
        imputation_mode=['observed', 'imputed'],
        padding=[False] * 2,
        ontology_curie=[_EXAMPLE_ONTOLOGY_TERM] * 2,
        extra=[1] * 2,
    )
)
_SC_ATAC_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar', 'baz'],
        strand=['.'] * 3,
        imputation_mode=['observed', 'observed', 'imputed'],
        padding=[False] * 3,
        ontology_curie=[_EXAMPLE_ONTOLOGY_TERM] * 3,
        extra=[1] * 3,
    )
)


def _create_track_data(
    sequence_length: int,
    metadata: pd.DataFrame,
    *,
    ontology_terms: list[ontology.OntologyTerm | None] | None = None,
) -> track_data.TrackData:
  metadata = metadata[~metadata['padding']]
  if ontology_terms is not None:
    if 'ontology_curie' in metadata.columns:
      metadata = metadata[
          metadata['ontology_curie'].isin(
              [o.ontology_curie for o in ontology_terms if o is not None]
          )
      ]
    else:
      metadata = metadata[slice(0, 0)]
  return track_data.TrackData(
      np.zeros(
          (sequence_length, len(metadata)),
          dtype=np.float32,
      ),
      metadata,
  )


def _get_output(sequence_length: int = 10):
  return dna_output.Output(
      atac=_create_track_data(sequence_length, _ATAC_METADATA),
      cage=_create_track_data(sequence_length, _CAGE_METADATA),
      dnase=_create_track_data(sequence_length, _DNASE_METADATA),
      rna_seq=_create_track_data(sequence_length, _RNA_SEQ_METADATA),
      chip_histone=None,
      chip_tf=None,
      splice_sites=None,
      splice_site_usage=None,
      sc_gene_expression=gene_expression_data.GeneExpressionData(
          values=np.zeros((2, 2), dtype=np.float32),
          track_metadata=_SC_GENE_EXPRESSION_METADATA,
          gene_metadata=pd.DataFrame({'gene_id': ['foo', 'bar']}),
          interval=genome.Interval('chr1', start=0, end=sequence_length),
      ),
      sc_atac=_create_track_data(sequence_length, _SC_ATAC_METADATA),
      splice_junctions=None,
      contact_maps=None,
  )


def _get_rnaseq_atac_output():
  """Manually specified RNA-seq and ATAC Outputs for testing diff and add."""
  atac_track_data_1 = track_data.TrackData(
      values=np.array(
          [
              [0.0, 0.2, 0.3, 0.4],
              [0.1, 0.3, 0.5, 0.6],
              [0.2, 0.4, 0.7, 0.8],
              [0.5, 0.1, 0.2, 0.6],
              [0.2, 0.4, 0.7, 0.8],
          ],
          dtype=np.float32,
      ),
      metadata=pd.DataFrame({
          'name': ['track_0', 'track_1', 'track_2', 'track_3'],
          'strand': '.',
          'tissue': ['HEART', 'BRAIN', 'BANANA', 'STOMACH'],
          'padding': [False, False, False, False],
      }),
  )

  atac_track_data_2 = track_data.TrackData(
      values=-1 * atac_track_data_1.values,
      metadata=atac_track_data_1.metadata,
  )

  rna_seq_track_data_1 = track_data.TrackData(
      values=np.array(
          [
              [1.0, 1.1, 1.2, 1.3, 1.4],
              [1.5, 1.6, 1.7, 1.8, 1.9],
              [1.2, 1.4, 1.7, 1.8, 1.9],
              [1.3, 1.4, 1.7, 1.8, 1.9],
              [2.0, 2.1, 2.2, 2.3, 2.4],
          ],
          dtype=np.float32,
      ),
      metadata=pd.DataFrame({
          'name': ['track_0', 'track_1', 'track_2', 'track_3', 'track_4'],
          'strand': ['+', '-', '-', '+', '.'],
          'tissue': ['BLADDER', 'PANCREAS', 'SKIN', 'PANCREAS', ''],
          'padding': [False, False, False, False, True],
      }),
  )

  rna_seq_track_data_2 = track_data.TrackData(
      values=-1 * rna_seq_track_data_1.values,
      metadata=rna_seq_track_data_1.metadata,
  )

  sc_gene_expression_1 = gene_expression_data.GeneExpressionData(
      values=np.ones((2, 2), dtype=np.float32),
      track_metadata=pd.DataFrame({
          'name': ['sc_track_0', 'sc_track_1'],
          'strand': ['+', '-'],
          'tissue': ['HEART', 'BRAIN'],
          'padding': [False, False],
      }),
      gene_metadata=pd.DataFrame({'gene_id': ['foo', 'bar']}),
      interval=genome.Interval('chr1', start=0, end=5),
  )
  sc_gene_expression_2 = gene_expression_data.GeneExpressionData(
      values=-1 * sc_gene_expression_1.values,
      track_metadata=sc_gene_expression_1.track_metadata,
      gene_metadata=sc_gene_expression_1.gene_metadata,
      interval=sc_gene_expression_1.interval,
  )

  # Build the 2 Output objects.
  output_1 = dna_output.Output(
      atac=atac_track_data_1,
      rna_seq=rna_seq_track_data_1,
      sc_gene_expression=sc_gene_expression_1,
      sc_atac=atac_track_data_1,
  )
  output_2 = dna_output.Output(
      atac=atac_track_data_2,
      rna_seq=rna_seq_track_data_2,
      sc_gene_expression=sc_gene_expression_2,
      sc_atac=atac_track_data_2,
  )
  output_3 = dna_output.Output(
      atac=atac_track_data_2,
      rna_seq=None,
      sc_gene_expression=None,
      sc_atac=atac_track_data_2,
  )
  return output_1, output_2, output_3


class OutputTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.output = _get_output()
    # Additional manually specified outputs for testing diff and add.
    self.output1, self.output2, self.output3 = _get_rnaseq_atac_output()

  def test_get(self):
    atac_output = self.output.get(dna_output.OutputType.ATAC)
    self.assertIsNotNone(atac_output)
    self.assertLen(
        _ATAC_METADATA[~_ATAC_METADATA['padding']],
        atac_output.num_tracks,
    )
    self.assertIsNone(self.output.get(dna_output.OutputType.CONTACT_MAPS))

  def test_map_over_outputs(self):

    def _fn(tdata, output_type):
      del output_type
      return tdata.select_tracks_by_index([0])

    filtered_output = self.output.map_over_outputs(_fn)

    atac_output = filtered_output.get(dna_output.OutputType.ATAC)
    self.assertIsNotNone(atac_output)
    self.assertEqual(atac_output.num_tracks, 1)

    cage_output = filtered_output.get(dna_output.OutputType.CAGE)
    self.assertIsNotNone(cage_output)
    self.assertEqual(cage_output.num_tracks, 1)

    dnase_output = filtered_output.get(dna_output.OutputType.DNASE)
    self.assertIsNotNone(dnase_output)
    self.assertEqual(dnase_output.num_tracks, 1)

    rna_seq_output = filtered_output.get(dna_output.OutputType.RNA_SEQ)
    self.assertIsNotNone(rna_seq_output)
    self.assertEqual(rna_seq_output.num_tracks, 1)

    sc_gene_expression_output = filtered_output.get(
        dna_output.OutputType.SC_GENE_EXPRESSION
    )
    self.assertIsNotNone(sc_gene_expression_output)
    self.assertEqual(sc_gene_expression_output.num_tracks, 1)

    self.assertIsNone(self.output.get(dna_output.OutputType.CONTACT_MAPS))

  def test_filter_to_strand_plus(self):
    filtered_output = self.output.filter_to_strand('+')

    atac_output = filtered_output.get(dna_output.OutputType.ATAC)
    self.assertIsNotNone(atac_output)
    self.assertEqual(atac_output.num_tracks, 0)

    cage_output = filtered_output.get(dna_output.OutputType.CAGE)
    self.assertIsNotNone(cage_output)
    self.assertEqual(cage_output.num_tracks, 0)

    dnase_output = filtered_output.get(dna_output.OutputType.DNASE)
    self.assertIsNotNone(dnase_output)
    self.assertEqual(dnase_output.num_tracks, 0)

    rna_seq_output = filtered_output.get(dna_output.OutputType.RNA_SEQ)
    self.assertIsNotNone(rna_seq_output)
    self.assertEqual(rna_seq_output.num_tracks, 1)

    sc_gene_expression_output = filtered_output.get(
        dna_output.OutputType.SC_GENE_EXPRESSION
    )
    self.assertIsNotNone(sc_gene_expression_output)
    self.assertEqual(sc_gene_expression_output.num_tracks, 1)

    self.assertIsNone(filtered_output.get(dna_output.OutputType.CONTACT_MAPS))

  def test_filter_to_strand_minus(self):
    filtered_output = self.output.filter_to_strand('-')

    atac_output = filtered_output.get(dna_output.OutputType.ATAC)
    self.assertIsNotNone(atac_output)
    self.assertEqual(atac_output.num_tracks, 0)

    cage_output = filtered_output.get(dna_output.OutputType.CAGE)
    self.assertIsNotNone(cage_output)
    self.assertEqual(cage_output.num_tracks, 0)

    dnase_output = filtered_output.get(dna_output.OutputType.DNASE)
    self.assertIsNotNone(dnase_output)
    self.assertEqual(dnase_output.num_tracks, 0)

    rna_seq_output = filtered_output.get(dna_output.OutputType.RNA_SEQ)
    self.assertIsNotNone(rna_seq_output)
    self.assertEqual(rna_seq_output.num_tracks, 1)

    sc_gene_expression_output = filtered_output.get(
        dna_output.OutputType.SC_GENE_EXPRESSION
    )
    self.assertIsNotNone(sc_gene_expression_output)
    self.assertEqual(sc_gene_expression_output.num_tracks, 1)

    self.assertIsNone(filtered_output.get(dna_output.OutputType.CONTACT_MAPS))

  def test_filter_to_strand_unstranded(self):
    filtered_output = self.output.filter_to_strand('.')

    atac_output = filtered_output.get(dna_output.OutputType.ATAC)
    self.assertIsNotNone(atac_output)
    self.assertLen(
        _ATAC_METADATA[~_ATAC_METADATA['padding']], atac_output.num_tracks
    )

    cage_output = filtered_output.get(dna_output.OutputType.CAGE)
    self.assertIsNotNone(cage_output)
    self.assertLen(
        _CAGE_METADATA[~_CAGE_METADATA['padding']],
        cage_output.num_tracks,
    )

    dnase_output = filtered_output.get(dna_output.OutputType.DNASE)
    self.assertIsNotNone(dnase_output)
    self.assertLen(
        _DNASE_METADATA[~_DNASE_METADATA['padding']],
        dnase_output.num_tracks,
    )

    rna_seq_output = filtered_output.get(dna_output.OutputType.RNA_SEQ)
    self.assertIsNotNone(rna_seq_output)
    self.assertEqual(rna_seq_output.num_tracks, 2)

    sc_atac_output = filtered_output.get(dna_output.OutputType.SC_ATAC)
    self.assertIsNotNone(sc_atac_output)
    self.assertLen(
        _SC_ATAC_METADATA[~_SC_ATAC_METADATA['padding']],
        sc_atac_output.num_tracks,
    )

    sc_gene_expression_output = filtered_output.get(
        dna_output.OutputType.SC_GENE_EXPRESSION
    )
    self.assertIsNotNone(sc_gene_expression_output)
    self.assertEqual(sc_gene_expression_output.num_tracks, 0)

    self.assertIsNone(filtered_output.get(dna_output.OutputType.CONTACT_MAPS))

  def test_filter_to_observed(self):
    self.assertIsNone(self.output.filter_to_observed().splice_junctions)

    junction_meta = pd.DataFrame({
        'name': ['junc_0', 'junc_1', 'junc_2'],
        'imputation_mode': ['observed', 'imputed_supported', 'observed'],
    })
    junctions = junction_data.JunctionData(
        junctions=np.array([genome.Interval('chr1', 2, 5, '+')]),
        values=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        metadata=junction_meta,
    )
    output = dataclasses.replace(self.output, splice_junctions=junctions)
    filtered_output = output.filter_to_observed()

    # Verify ATAC filtering.
    filtered_atac = filtered_output.get(dna_output.OutputType.ATAC)
    self.assertIsNotNone(filtered_atac)
    self.assertIsInstance(filtered_atac, track_data.TrackData)
    pd.testing.assert_frame_equal(
        filtered_atac.metadata,
        _ATAC_METADATA[
            (~_ATAC_METADATA['padding'])
            & (_ATAC_METADATA['imputation_mode'] == 'observed')
        ],
    )

    # Verify SC_GENE_EXPRESSION filtering.
    filtered_sc_gene = filtered_output.get(
        dna_output.OutputType.SC_GENE_EXPRESSION
    )
    self.assertIsNotNone(filtered_sc_gene)
    self.assertIsInstance(
        filtered_sc_gene, gene_expression_data.GeneExpressionData
    )
    pd.testing.assert_frame_equal(
        filtered_sc_gene.track_metadata,
        _SC_GENE_EXPRESSION_METADATA[
            (_SC_GENE_EXPRESSION_METADATA['imputation_mode'] == 'observed')
        ],
    )

    # Verify SPLICE_JUNCTIONS filtering.
    filtered_junctions = filtered_output.get(
        dna_output.OutputType.SPLICE_JUNCTIONS
    )
    self.assertIsNotNone(filtered_junctions)
    self.assertIsInstance(filtered_junctions, junction_data.JunctionData)
    pd.testing.assert_frame_equal(
        filtered_junctions.metadata,
        junction_meta.iloc[[0, 2]],
    )
    np.testing.assert_array_equal(
        filtered_junctions.values,
        np.array([[1.0, 3.0]], dtype=np.float32),
    )

  def test_filter_output_type(self):
    output = self.output.filter_output_type({
        dna_output.OutputType.ATAC,
        dna_output.OutputType.SC_ATAC,
        dna_output.OutputType.SC_GENE_EXPRESSION,
    })
    self.assertIsInstance(
        output.get(dna_output.OutputType.ATAC), track_data.TrackData
    )
    self.assertIsInstance(
        output.get(dna_output.OutputType.SC_ATAC), track_data.TrackData
    )
    self.assertIsInstance(
        output.get(dna_output.OutputType.SC_GENE_EXPRESSION),
        gene_expression_data.GeneExpressionData,
    )
    self.assertIsNone(output.get(dna_output.OutputType.CAGE))
    self.assertIsNone(output.get(dna_output.OutputType.DNASE))
    self.assertIsNone(output.get(dna_output.OutputType.CONTACT_MAPS))

  def test_output_diff(self):
    output_diff = self.output1 - self.output2
    self.assertIsInstance(output_diff.rna_seq, track_data.TrackData)
    self.assertIsInstance(output_diff.atac, track_data.TrackData)
    self.assertIsInstance(
        output_diff.sc_gene_expression, gene_expression_data.GeneExpressionData
    )

    # To fix failing build (since these fields can be None). Can't be list comp.
    assert self.output1.rna_seq is not None
    assert self.output2.rna_seq is not None
    assert output_diff.rna_seq is not None
    assert self.output1.atac is not None
    assert self.output2.atac is not None
    assert output_diff.atac is not None
    assert self.output1.sc_gene_expression is not None
    assert self.output2.sc_gene_expression is not None
    assert output_diff.sc_gene_expression is not None

    manual_diff_rnaseq = (
        self.output1.rna_seq.values - self.output2.rna_seq.values
    )
    np.testing.assert_array_equal(
        output_diff.rna_seq.values, manual_diff_rnaseq
    )

    manual_diff_atac = self.output1.atac.values - self.output2.atac.values
    np.testing.assert_array_equal(output_diff.atac.values, manual_diff_atac)

    manual_diff_sc_gene_expression = (
        self.output1.sc_gene_expression.values
        - self.output2.sc_gene_expression.values
    )
    np.testing.assert_array_equal(
        output_diff.sc_gene_expression.values, manual_diff_sc_gene_expression
    )

    # output3 has None for rna_seq. Where None is passed, just return None
    # rather than trying to do arithmetic.
    output_diff = self.output1 - self.output3
    self.assertIsNone(output_diff.rna_seq)
    self.assertIsNone(output_diff.sc_gene_expression)
    self.assertIsNotNone(output_diff.atac)

    output_diff = self.output3 - self.output1
    self.assertIsNone(output_diff.rna_seq)
    self.assertIsNone(output_diff.sc_gene_expression)
    self.assertIsNotNone(output_diff.atac)

  def test_output_add(self):
    output_sum = self.output1 + self.output2
    self.assertIsInstance(output_sum.rna_seq, track_data.TrackData)
    self.assertIsInstance(output_sum.atac, track_data.TrackData)
    self.assertIsInstance(
        output_sum.sc_gene_expression, gene_expression_data.GeneExpressionData
    )

    # To fix failing build (since these fields can be None). Can't be list comp.
    assert self.output1.rna_seq is not None
    assert self.output2.rna_seq is not None
    assert output_sum.rna_seq is not None
    assert self.output1.atac is not None
    assert self.output2.atac is not None
    assert output_sum.atac is not None
    assert self.output1.sc_gene_expression is not None
    assert self.output2.sc_gene_expression is not None
    assert output_sum.sc_gene_expression is not None

    manual_sum_rnaseq = (
        self.output1.rna_seq.values + self.output2.rna_seq.values
    )
    np.testing.assert_array_equal(output_sum.rna_seq.values, manual_sum_rnaseq)

    manual_sum_atac = self.output1.atac.values + self.output2.atac.values
    np.testing.assert_array_equal(output_sum.atac.values, manual_sum_atac)

    manual_sum_sc_gene_expression = (
        self.output1.sc_gene_expression.values
        + self.output2.sc_gene_expression.values
    )
    np.testing.assert_array_equal(
        output_sum.sc_gene_expression.values, manual_sum_sc_gene_expression
    )

    # output3 has None for rna_seq. Where None is passed, just return None
    # rather than trying to do arithmetic.
    output_sum = self.output1 + self.output3
    self.assertIsNone(output_sum.rna_seq)
    self.assertIsNone(output_sum.sc_gene_expression)
    self.assertIsNotNone(output_sum.atac)

    output_sum = self.output3 + self.output1
    self.assertIsNone(output_sum.rna_seq)
    self.assertIsNone(output_sum.sc_gene_expression)
    self.assertIsNotNone(output_sum.atac)

  def test_output_metadata_concatenate(self):
    output_metadata = dna_output.OutputMetadata(
        atac=track_data.TrackMetadata(
            {'name': [f'track_{i}' for i in range(32)], 'strand': ['.'] * 32}
        ),
        dnase=track_data.TrackMetadata(
            {'name': [f'track_{i}' for i in range(10)], 'strand': ['.'] * 10}
        ),
    )
    df = output_metadata.concatenate()
    self.assertLen(df, 42)

  def test_output_with_name_suffix(self):
    junctions = junction_data.JunctionData(
        junctions=np.array([genome.Interval('chr1', 10, 20, '+')]),
        values=np.array([[1.0]]),
        metadata=pd.DataFrame({'name': ['junc_0']}),
    )
    output = dataclasses.replace(self.output1, splice_junctions=junctions)
    suffixed = output.with_name_suffix('_suffix')
    assert suffixed.atac is not None
    assert self.output1.atac is not None
    self.assertListEqual(
        list(suffixed.atac.names),
        [f'{name}_suffix' for name in self.output1.atac.names],
    )
    assert suffixed.sc_gene_expression is not None
    assert self.output1.sc_gene_expression is not None
    self.assertListEqual(
        list(suffixed.sc_gene_expression.names),
        [f'{name}_suffix' for name in self.output1.sc_gene_expression.names],
    )
    assert suffixed.splice_junctions is not None
    self.assertListEqual(
        list(suffixed.splice_junctions.names),
        ['junc_0_suffix'],
    )

  def test_output_with_metadata_column(self):
    junctions = junction_data.JunctionData(
        junctions=np.array([genome.Interval('chr1', 10, 20, '+')]),
        values=np.array([[1.0]]),
        metadata=pd.DataFrame({'name': ['junc_0']}),
    )
    output = dataclasses.replace(self.output1, splice_junctions=junctions)
    updated = output.with_metadata_column('biosample_name', 'cell_0')
    assert updated.atac is not None
    self.assertIn('biosample_name', updated.atac.metadata.columns)
    self.assertTrue((updated.atac.metadata['biosample_name'] == 'cell_0').all())
    assert updated.sc_gene_expression is not None
    self.assertIn(
        'biosample_name', updated.sc_gene_expression.track_metadata.columns
    )
    self.assertTrue(
        (
            updated.sc_gene_expression.track_metadata['biosample_name']
            == 'cell_0'
        ).all()
    )
    assert updated.splice_junctions is not None
    self.assertIn('biosample_name', updated.splice_junctions.metadata.columns)
    self.assertTrue(
        (updated.splice_junctions.metadata['biosample_name'] == 'cell_0').all()
    )
    self.assertNotIn('biosample_name', junctions.metadata.columns)

  def test_target_gene_metadata_from_proto_and_validation(self):
    full_proto = dna_model_pb2.GeneExpressionMetadata(
        gene_id='ENSG00000119866',
        chromosome='chr2',
        start=60_450_520,
        end=60_553_654,
        strand=dna_model_pb2_v1.STRAND_NEGATIVE,
        name='BCL11A',
        type='protein_coding',
    )
    meta = dna_output.TargetGeneMetadata.from_proto(full_proto)
    self.assertEqual(
        meta,
        dna_output.TargetGeneMetadata(
            gene_id='ENSG00000119866',
            interval=genome.Interval('chr2', 60_450_520, 60_553_654, '-'),
            gene_name='BCL11A',
            gene_type='protein_coding',
        ),
    )

    minimal_proto = dna_model_pb2.GeneExpressionMetadata(
        gene_id='ENSG00000119866',
        chromosome='chr2',
        start=60_450_520,
        end=60_553_654,
    )
    minimal_meta = dna_output.TargetGeneMetadata.from_proto(minimal_proto)
    self.assertEqual(
        minimal_meta.interval,
        genome.Interval('chr2', 60_450_520, 60_553_654, '.'),
    )
    self.assertIsNone(minimal_meta.gene_name)
    self.assertIsNone(minimal_meta.gene_type)

    with self.assertRaisesRegex(ValueError, 'gene_id'):
      dna_output.TargetGeneMetadata(
          gene_id='', interval=genome.Interval('chr2', 100, 200)
      )

  def test_input_gradients_output_properties_and_validation(self):
    gradients = np.ones((4, 4), dtype=np.float32)
    interval = genome.Interval('chr2', 100, 104)
    gene_meta = dna_output.TargetGeneMetadata(
        gene_id='ENSG00000119866',
        interval=genome.Interval('chr2', 100, 200, '-'),
        gene_name='BCL11A',
    )
    grad_out = dna_output.InputGradientsOutput(
        gradients=gradients,
        interval=interval,
        scalar_score=1.5,
        target_gene=gene_meta,
        track_metadata=pd.DataFrame({
            'name': ['sc_rna_+', 'sc_rna_-'],
            'biosample_name': ['Erythroid', 'Erythroid'],
        }),
        output_type=dna_output.OutputType.SC_GENE_EXPRESSION,
    )
    self.assertEqual(grad_out.target_gene, gene_meta)
    self.assertEqual(grad_out.gene_id, 'ENSG00000119866')
    self.assertEqual(grad_out.gene_name, 'BCL11A')
    self.assertEqual(grad_out.biosample_name, 'Erythroid')

    no_name_out = dataclasses.replace(
        grad_out,
        target_gene=dna_output.TargetGeneMetadata(
            gene_id='ENSG00000119866',
            interval=genome.Interval('chr2', 100, 200, '-'),
        ),
    )
    self.assertIsNone(no_name_out.gene_name)

    for bad_shape in ((3, 4), (4, 3), (4,)):
      with self.assertRaisesRegex(ValueError, 'Expected gradients of shape'):
        dataclasses.replace(
            grad_out, gradients=np.ones(bad_shape, dtype=np.float32)
        )

    for bad_track_meta in (
        pd.DataFrame({'name': ['t1']}),
        pd.DataFrame({
            'name': pd.Series(dtype=str),
            'biosample_name': pd.Series(dtype=str),
        }),
        pd.DataFrame(
            {'name': ['t1', 't2'], 'biosample_name': ['Erythroid', 'B cell']}
        ),
        pd.DataFrame(
            {'name': ['t1', 't2'], 'biosample_name': ['Erythroid', None]}
        ),
        pd.DataFrame({'name': ['t1'], 'biosample_name': ['']}),
    ):
      with self.assertRaisesRegex(ValueError, 'biosample_name'):
        dataclasses.replace(grad_out, track_metadata=bad_track_meta)

  def test_input_gradients_output_to_track_data(self):
    scalar_score = float(np.log(2.5 + 1e-6))
    gradients = np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.75, 0.0],
        ],
        dtype=np.float32,
    )
    interval = genome.Interval('chr2', 100, 104)
    grad_out = dna_output.InputGradientsOutput(
        gradients=gradients,
        interval=interval,
        scalar_score=scalar_score,
        target_gene=dna_output.TargetGeneMetadata(
            gene_id='ENSG00000119866',
            interval=genome.Interval('chr2', 100, 200, '-'),
            gene_name='BCL11A',
        ),
        track_metadata=pd.DataFrame({
            'name': ['sc_rna_+', 'sc_rna_-'],
            'biosample_name': ['Erythroid', 'Erythroid'],
        }),
        output_type=dna_output.OutputType.SC_GENE_EXPRESSION,
    )

    unsigned_track = grad_out.to_track_data()
    self.assertEqual(unsigned_track.resolution, 1)
    self.assertEqual(unsigned_track.interval, interval)
    self.assertEqual(
        list(unsigned_track.metadata['name']),
        ['BCL11A input gradients (Erythroid)'],
    )
    self.assertEqual(
        list(unsigned_track.metadata['gene_id']), ['ENSG00000119866']
    )
    self.assertEqual(list(unsigned_track.metadata['gene_name']), ['BCL11A'])
    self.assertEqual(
        list(unsigned_track.metadata['biosample_name']), ['Erythroid']
    )
    np.testing.assert_allclose(
        unsigned_track.values,
        np.array([[0.5], [0.25], [0.0], [0.75]], dtype=np.float32),
    )

    grad_out_bcell = dataclasses.replace(
        grad_out,
        track_metadata=pd.DataFrame(
            {'name': ['sc_rna_+'], 'biosample_name': ['B cell']}
        ),
    )
    concat_tracks = track_data.concat(
        [unsigned_track, grad_out_bcell.to_track_data()]
    )
    self.assertEqual(concat_tracks.values.shape, (4, 2))
    self.assertEqual(
        list(concat_tracks.metadata['name']),
        [
            'BCL11A input gradients (Erythroid)',
            'BCL11A input gradients (B cell)',
        ],
    )

    signed_track = grad_out.to_track_data(signed=True, name='custom_grad')
    self.assertEqual(list(signed_track.metadata['name']), ['custom_grad'])
    self.assertEqual(
        list(signed_track.metadata['gene_id']), ['ENSG00000119866']
    )
    self.assertEqual(list(signed_track.metadata['gene_name']), ['BCL11A'])
    np.testing.assert_allclose(
        signed_track.values,
        np.array([[0.5], [-0.25], [0.0], [0.75]], dtype=np.float32),
    )

    grad_out_gene_id = dataclasses.replace(
        grad_out,
        gradients=gradients.astype(np.float64),
        target_gene=dna_output.TargetGeneMetadata(
            gene_id='ENSG00000119866',
            interval=genome.Interval('chr2', 100, 200, '-'),
        ),
    )
    self.assertIsNone(grad_out_gene_id.gene_name)
    td_gene_id = grad_out_gene_id.to_track_data()
    self.assertEqual(td_gene_id.values.dtype, np.float32)
    self.assertEqual(td_gene_id.values.shape, (4, 1))
    self.assertEqual(
        list(td_gene_id.metadata['name']),
        ['ENSG00000119866 input gradients (Erythroid)'],
    )
    self.assertEqual(list(td_gene_id.metadata['gene_id']), ['ENSG00000119866'])
    self.assertEqual(list(td_gene_id.metadata['biosample_name']), ['Erythroid'])
    self.assertNotIn('gene_name', td_gene_id.metadata.columns)


if __name__ == '__main__':
  absltest.main()
