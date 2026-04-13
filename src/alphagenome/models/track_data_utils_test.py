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

import ml_dtypes
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from alphagenome import tensor_utils
from alphagenome.data import genome, track_data
from alphagenome.models import track_data_utils
from alphagenome.protos import dna_model_pb2


class TrackDataUtilsTest(parameterized.TestCase):

  @parameterized.product(
      include_ontologies=[True, False],
      bytes_per_chunk=[4, 8],
      dtypes=[np.float32, np.int32],
      resolution=[1, 128],
      include_interval=[True, False],
      include_biosample=[True, False],
      include_transcription_factor=[True, False],
      include_histone_mark=[True, False],
      include_gtex_tissue=[True, False],
      include_endedness=[True, False],
      include_genetically_modified=[True, False],
      include_data_source=[True, False],
      include_nonzero_mean=[True, False],
      override_interval=[True, False],
  )
  def test_proto_roundtrip(
      self,
      include_ontologies: bool,
      bytes_per_chunk: int,
      dtypes: np.dtype,
      resolution: int,
      include_interval: bool,
      include_biosample: bool,
      include_transcription_factor: bool,
      include_histone_mark: bool,
      include_gtex_tissue: bool,
      include_endedness: bool,
      include_genetically_modified: bool,
      include_data_source: bool,
      include_nonzero_mean: bool,
      override_interval: bool,
  ):
    df = pd.DataFrame({
        'name': [f'track_{i}' for i in range(7)],
        'strand': ['.', '+', '-', '.', '+', '-', '.'],
        'Assay title': ['foo', 'bar', 'baz', 'qux', 'quux', 'wuz', 'buz'],
    })
    if include_ontologies:
      df['ontology_curie'] = [
          'UBERON:0000005',
          'UBERON:0000007',
          'UBERON:0000007',
          'UBERON:0000007',
          None,
          None,
          None,
      ]
    if include_biosample:
      df['biosample_name'] = [f'biosample_{i}' for i in range(6)] + [None]
      df['biosample_type'] = [
          'primary_cell',
          'in_vitro_differentiated_cells',
          'cell_line',
          'tissue',
          'technical_sample',
          'organoid',
          None,
      ]
      df['biosample_life_stage'] = [
          'adult',
          'adult',
          None,
          'newborn',
          'child',
          'embryonic',
          None,
      ]
    if include_transcription_factor:
      df['transcription_factor'] = [
          'BCLAF1',
          'BRF2',
          'CBFA2T3',
          'CEBPB',
          'COPS2',
          'POLR2AphosphoS5',
          None,
      ]
    if include_histone_mark:
      df['histone_mark'] = [
          'H2AFZ',
          'H2AK5AC',
          'H2BK120AC',
          'H2BK12AC',
          'H2BK20AC',
          'H2BK5AC',
          None,
      ]
    if include_gtex_tissue:
      df['gtex_tissue'] = [
          'Liver',
          '',
          '',
          'Brain_Hippocampus',
          'Pancreas',
          'Artery_Aorta',
          None,
      ]
    if include_data_source:
      df['data_source'] = [
          'encode',
          'encode',
          'encode',
          'fantom',
          'fantom',
          'gtex',
          None,
      ]
    if include_endedness:
      df['endedness'] = [
          'paired',
          'single',
          'paired',
          'single',
          'paired',
          'single',
          None,
      ]
    if include_genetically_modified:
      df['genetically_modified'] = [
          True,
          False,
          True,
          False,
          True,
          False,
          None,
      ]
    if include_nonzero_mean:
      df['nonzero_mean'] = [1.1, None, 1.1, 1.1, 1.1, 0.6, None]

    values = np.array(
        [
            [0.0, 0.2, 0.3, 0.4, 0.5, 0.1, 0.1],
            [0.1, 0.3, 0.5, 0.6, 0.7, 0.1, 0.1],
            [0.2, 0.4, 0.7, 0.8, 0.9, 0.1, 0.1],
        ],
        dtype=dtypes,
    )
    expected_interval = (
        genome.Interval('chr1', 0, values.shape[0] * resolution)
        if include_interval or override_interval
        else None
    )

    metadata = track_data.TrackData(
        values=values,
        metadata=df,
        resolution=resolution,
        interval=genome.Interval('chr1', 0, values.shape[0] * resolution)
        if include_interval
        else None,
    )

    proto, chunks = track_data_utils.to_protos(
        metadata, bytes_per_chunk=bytes_per_chunk
    )
    roundtrip = track_data_utils.from_protos(
        proto,
        chunks,
        interval=genome.Interval('chr1', 0, values.shape[0] * resolution)
        if override_interval
        else None,
    )

    np.testing.assert_array_equal(roundtrip.values, values)
    pd.testing.assert_frame_equal(roundtrip.metadata, df)
    self.assertEqual(roundtrip.resolution, resolution)
    self.assertEqual(roundtrip.interval, expected_interval)

  @parameterized.product(
      bytes_per_chunk=[0, 4, 8], dtype=[ml_dtypes.bfloat16, np.float16]
  )
  def test_from_low_precision_proto(self, bytes_per_chunk, dtype):
    values = np.array(
        [
            [0.0, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.3, 0.5, 0.6, 0.7],
            [0.2, 0.4, 0.7, 0.8, 0.9],
        ],
        dtype=dtype,
    )
    metadata = pd.DataFrame({
        'name': ['track_0', 'track_1', 'track_2', 'track_3', 'track_4'],
        'strand': ['.'] * 5,
    })
    tensor, chunks = tensor_utils.pack_tensor(
        values, bytes_per_chunk=bytes_per_chunk
    )
    proto = dna_model_pb2.TrackData(
        values=tensor,
        metadata=track_data_utils.metadata_to_proto(metadata).metadata,
    )
    roundtrip = track_data_utils.from_protos(proto, chunks)
    np.testing.assert_array_equal(roundtrip.values, values.astype(np.float32))
    pd.testing.assert_frame_equal(roundtrip.metadata, metadata)


if __name__ == '__main__':
  absltest.main()
