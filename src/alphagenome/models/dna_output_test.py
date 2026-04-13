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

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from alphagenome.data import genome, junction_data, ontology, track_data
from alphagenome.models import dna_output

_EXAMPLE_ONTOLOGY_TERM = 'UBERON:0002037'

_ATAC_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar', 'baz', 'buz', 'fux'],
        strand=['.'] * 5,
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
        padding=[False] * 3,
        ontology_curie=[_EXAMPLE_ONTOLOGY_TERM] * 3,
        extra=[1] * 3,
    )
)
_DNASE_METADATA = pd.DataFrame(
    dict(name=['foo'], strand=['.'], padding=[False], extra=[1])
)
_RNA_SEQ_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'foo', 'baz', 'buz'],
        strand=['+', '-', '.', '.'],
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


def _create_splice_junctions():
  metadata = pd.DataFrame({
      'name': ['foo', 'bar'],
      'ontology_curie': ['UBERON:0000005', 'UBERON:0000006'],
  })
  junctions = np.array([
      genome.Interval('chr1', 10, 11, '+'),
      genome.Interval('chr1', 10, 11, '-'),
  ])
  return junction_data.JunctionData(
      junctions,
      np.zeros(
          (len(junctions), len(metadata)),
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
      splice_junctions=_create_splice_junctions(),
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

  # Build the 2 Output objects.
  output_1 = dna_output.Output(
      atac=atac_track_data_1,
      rna_seq=rna_seq_track_data_1,
  )
  output_2 = dna_output.Output(
      atac=atac_track_data_2,
      rna_seq=rna_seq_track_data_2,
  )
  output_3 = dna_output.Output(
      atac=atac_track_data_2,
      rna_seq=None,
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

  def test_map_track_data(self):

    def _fn(tdata, output_type):
      del output_type
      return tdata.select_tracks_by_index([0])

    filtered_output = self.output.map_track_data(_fn)

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

    self.assertIsNone(filtered_output.get(dna_output.OutputType.CONTACT_MAPS))

    splice_junction_output = filtered_output.get(
        dna_output.OutputType.SPLICE_JUNCTIONS
    )
    self.assertIsInstance(splice_junction_output, junction_data.JunctionData)
    np.testing.assert_array_equal(
        splice_junction_output.possible_strands, ['+']
    )

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

    self.assertIsNone(filtered_output.get(dna_output.OutputType.CONTACT_MAPS))

    splice_junction_output = filtered_output.get(
        dna_output.OutputType.SPLICE_JUNCTIONS
    )
    self.assertIsInstance(splice_junction_output, junction_data.JunctionData)
    np.testing.assert_array_equal(
        splice_junction_output.possible_strands, ['-']
    )

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

    splice_junction_output = filtered_output.get(
        dna_output.OutputType.SPLICE_JUNCTIONS
    )
    self.assertIsInstance(splice_junction_output, junction_data.JunctionData)
    np.testing.assert_array_equal(splice_junction_output.possible_strands, [])

    self.assertIsNone(filtered_output.get(dna_output.OutputType.CONTACT_MAPS))

  def test_filter_output_type(self):
    output = self.output.filter_output_type({dna_output.OutputType.ATAC})
    self.assertIsInstance(
        output.get(dna_output.OutputType.ATAC), track_data.TrackData
    )
    self.assertIsNone(output.get(dna_output.OutputType.CAGE))
    self.assertIsNone(output.get(dna_output.OutputType.DNASE))
    self.assertIsNone(output.get(dna_output.OutputType.CONTACT_MAPS))

  def test_filter_ontology_terms(self):
    filtered_output = self.output.filter_ontology_terms(
        {ontology.from_curie(_EXAMPLE_ONTOLOGY_TERM)}
    )
    dnase_output = filtered_output.get(dna_output.OutputType.DNASE)
    self.assertIsNotNone(dnase_output)
    self.assertLen(
        _DNASE_METADATA,
        dnase_output.num_tracks,
    )
    atac_output = filtered_output.get(dna_output.OutputType.ATAC)
    self.assertIsNotNone(atac_output)
    self.assertEqual(
        atac_output.num_tracks,
        np.sum(_ATAC_METADATA['ontology_curie'] == _EXAMPLE_ONTOLOGY_TERM),
    )

  def test_output_diff(self):
    output_diff = self.output1 - self.output2
    self.assertIsInstance(output_diff.rna_seq, track_data.TrackData)
    self.assertIsInstance(output_diff.atac, track_data.TrackData)

    # To fix failing build (since these fields can be None). Can't be list comp.
    assert self.output1.rna_seq is not None
    assert self.output2.rna_seq is not None
    assert output_diff.rna_seq is not None
    assert self.output1.atac is not None
    assert self.output2.atac is not None
    assert output_diff.atac is not None

    manual_diff_rnaseq = (
        self.output1.rna_seq.values - self.output2.rna_seq.values
    )
    np.testing.assert_array_equal(
        output_diff.rna_seq.values, manual_diff_rnaseq
    )

    manual_diff_atac = self.output1.atac.values - self.output2.atac.values
    np.testing.assert_array_equal(output_diff.atac.values, manual_diff_atac)

    # output3 has None for rna_seq. Where None is passed, just return None
    # rather than trying to do arithmetic.
    output_diff = self.output1 - self.output3
    self.assertIsNone(output_diff.rna_seq)
    self.assertIsNotNone(output_diff.atac)

    output_diff = self.output3 - self.output1
    self.assertIsNone(output_diff.rna_seq)
    self.assertIsNotNone(output_diff.atac)

  def test_output_add(self):
    output_sum = self.output1 + self.output2
    self.assertIsInstance(output_sum.rna_seq, track_data.TrackData)
    self.assertIsInstance(output_sum.atac, track_data.TrackData)

    # To fix failing build (since these fields can be None). Can't be list comp.
    assert self.output1.rna_seq is not None
    assert self.output2.rna_seq is not None
    assert output_sum.rna_seq is not None
    assert self.output1.atac is not None
    assert self.output2.atac is not None
    assert output_sum.atac is not None

    manual_sum_rnaseq = (
        self.output1.rna_seq.values + self.output2.rna_seq.values
    )
    np.testing.assert_array_equal(output_sum.rna_seq.values, manual_sum_rnaseq)

    manual_sum_atac = self.output1.atac.values + self.output2.atac.values
    np.testing.assert_array_equal(output_sum.atac.values, manual_sum_atac)

    # output3 has None for rna_seq. Where None is passed, just return None
    # rather than trying to do arithmetic.
    output_sum = self.output1 + self.output3
    self.assertIsNone(output_sum.rna_seq)
    self.assertIsNotNone(output_sum.atac)

    output_sum = self.output3 + self.output1
    self.assertIsNone(output_sum.rna_seq)
    self.assertIsNotNone(output_sum.atac)

  def test_output_metadata_from_outputs(self):
    output = _get_output()
    mapping = {
        dna_output.OutputType.ATAC: output.atac,
        dna_output.OutputType.CAGE: output.cage,
        dna_output.OutputType.DNASE: output.dnase,
        dna_output.OutputType.RNA_SEQ: output.rna_seq,
    }
    output_metadata = dna_output.OutputMetadata.from_outputs(mapping)
    for output_type in dna_output.OutputType:
      if output_type in mapping:
        self.assertIs(
            output_metadata.get(output_type),
            output.get(output_type).metadata,  # pytype: disable=attribute-error
        )
      else:
        self.assertIsNone(output_metadata.get(output_type))

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


if __name__ == '__main__':
  absltest.main()
