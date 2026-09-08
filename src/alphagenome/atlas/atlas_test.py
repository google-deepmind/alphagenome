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

from collections.abc import Mapping
import struct
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.atlas import atlas
from alphagenome.data import genome
from alphagenome.protos import dna_model_pb2
from alphagenome.protos import atlas_service_pb2
import anndata
import grpc
import numpy as np
import pandas as pd


class AtlasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.addTypeEqualityFunc(
        atlas.ScorerMetadata, self.assertScorerMetadataEqual
    )

  def assertScorerMetadataEqual(self, a, b, msg=None):
    self.assertEqual(a.name, b.name, msg)
    self.assertEqual(a.is_signed, b.is_signed, msg)
    pd.testing.assert_frame_equal(a.track_metadata, b.track_metadata)

  @parameterized.named_parameters(
      ('Empty', [], {}, {}),
      dict(
          testcase_name='EmptyScores',
          protos=[
              atlas_service_pb2.DenseVariantScores(
                  variant=dna_model_pb2.Variant(
                      chromosome='chr1',
                      position=100,
                      reference_bases='A',
                      alternate_bases='T',
                  ),
                  scores=[
                      atlas_service_pb2.DenseVariantScore(
                          variant_scorer=atlas_service_pb2.VariantScorerInfo(
                              name='scorer1'
                          ),
                          shape=[0, 5],
                      )
                  ],
              ),
          ],
          metadata={
              'scorer1': pd.DataFrame(
                  {'name': [f'track{i}' for i in range(5)], 'strand': '+'},
                  index=[str(i) for i in range(5)],
              )
          },
          expected={
              'scorer1': anndata.AnnData(
                  X=np.zeros((0, 5)),
                  var=pd.DataFrame(
                      {
                          'name': [f'track{i}' for i in range(5)],
                          'strand': '+',
                      },
                      index=[str(i) for i in range(5)],
                  ),
              )
          },
      ),
      dict(
          testcase_name='EmptyTracksNoGenes',
          protos=[
              atlas_service_pb2.DenseVariantScores(
                  variant=dna_model_pb2.Variant(
                      chromosome='chr1',
                      position=101,
                      reference_bases='C',
                      alternate_bases='A',
                  ),
                  scores=[
                      atlas_service_pb2.DenseVariantScore(
                          variant_scorer=atlas_service_pb2.VariantScorerInfo(
                              name='scorer1'
                          ),
                          shape=[1, 0],
                      )
                  ],
              ),
          ],
          metadata={'scorer1': pd.DataFrame()},
          expected={
              'scorer1': anndata.AnnData(
                  X=np.zeros((1, 0)),
                  var=pd.DataFrame(),
                  obs=pd.DataFrame(
                      {'variant': [genome.Variant.from_str('chr1:101:C>A')]},
                      index=['0'],
                  ),
              )
          },
      ),
      dict(
          testcase_name='TrackScorer',
          protos=[
              atlas_service_pb2.DenseVariantScores(
                  variant=dna_model_pb2.Variant(
                      chromosome='chr1',
                      position=100,
                      reference_bases='A',
                      alternate_bases='T',
                  ),
                  scores=[
                      atlas_service_pb2.DenseVariantScore(
                          variant_scorer=atlas_service_pb2.VariantScorerInfo(
                              name='scorer1'
                          ),
                          shape=[1, 1],
                          scores=struct.pack('f', 0.0),
                      )
                  ],
              ),
          ],
          metadata={
              'scorer1': pd.DataFrame(
                  {'name': ['track1'], 'strand': ['+']}, index=['0']
              )
          },
          expected={
              'scorer1': anndata.AnnData(
                  X=np.zeros((1, 1), dtype=np.float32),
                  obs=pd.DataFrame(
                      {'variant': [genome.Variant.from_str('chr1:100:A>T')]},
                      index=['0'],
                  ),
                  var=pd.DataFrame(
                      {'name': ['track1'], 'strand': ['+']}, index=['0']
                  ),
              )
          },
      ),
      dict(
          testcase_name='GeneScorer',
          protos=[
              atlas_service_pb2.DenseVariantScores(
                  variant=dna_model_pb2.Variant(
                      chromosome='chr1',
                      position=100,
                      reference_bases='A',
                      alternate_bases='T',
                  ),
                  scores=[
                      atlas_service_pb2.DenseVariantScore(
                          variant_scorer=atlas_service_pb2.VariantScorerInfo(
                              name='scorer1'
                          ),
                          shape=[1, 1],
                          scores=struct.pack('f', 0.0),
                          metadata=[
                              atlas_service_pb2.Metadata(
                                  gene_scorers=atlas_service_pb2.GeneScorersMetadata(  # pylint: disable=line-too-long
                                      metadata=[
                                          dna_model_pb2.GeneScorerMetadata(
                                              gene_id='gene_id_1',
                                              name='gene_1',
                                              strand=(
                                                  dna_model_pb2.Strand.STRAND_POSITIVE  # pylint: disable=line-too-long
                                              ),
                                              junction_start=100,
                                              junction_end=200,
                                          )
                                      ]
                                  )
                              )
                          ],
                      )
                  ],
              ),
          ],
          metadata={
              'scorer1': pd.DataFrame({'name': ['track1'], 'strand': ['+']})
          },
          expected={
              'scorer1': anndata.AnnData(
                  X=np.array([[0]], dtype=np.float32),
                  obs=pd.DataFrame(
                      {
                          'gene_id': ['gene_id_1'],
                          'gene_name': ['gene_1'],
                          'strand': ['+'],
                          'junction_Start': [100],
                          'junction_End': [200],
                          'variant': [genome.Variant.from_str('chr1:100:A>T')],
                      },
                      index=['0'],
                  ),
                  var=pd.DataFrame(
                      {'name': ['track1'], 'strand': ['+']}, index=['0']
                  ),
              )
          },
      ),
      dict(
          testcase_name='CalibratedScores',
          protos=[
              atlas_service_pb2.DenseVariantScores(
                  variant=dna_model_pb2.Variant(
                      chromosome='chr1',
                      position=101,
                      reference_bases='G',
                      alternate_bases='C',
                  ),
                  scores=[
                      atlas_service_pb2.DenseVariantScore(
                          variant_scorer=atlas_service_pb2.VariantScorerInfo(
                              name='scorer1'
                          ),
                          shape=[1, 1],
                          scores=struct.pack('f', 0.0),
                          calibrated_scores=struct.pack('f', 0.1),
                      )
                  ],
              ),
          ],
          metadata={
              'scorer1': pd.DataFrame({'name': ['score'], 'strand': ['.']})
          },
          expected={
              'scorer1': anndata.AnnData(
                  X=np.array([[0]], dtype=np.float32),
                  obs=pd.DataFrame(
                      {'variant': [genome.Variant.from_str('chr1:101:G>C')]},
                      index=['0'],
                  ),
                  var=pd.DataFrame(
                      {'name': ['score'], 'strand': ['.']}, index=['0']
                  ),
                  layers={'quantiles': np.array([[0.1]], dtype=np.float32)},
              )
          },
      ),
  )
  def test_convert_scores_to_anndata(self, protos, metadata, expected):
    result = atlas.convert_variant_scores_to_anndata(protos, metadata)

    for (name, scores), (expected_name, expected_scores) in zip(
        result.items(), expected.items(), strict=True
    ):
      self.assertEqual(name, expected_name)
      np.testing.assert_array_equal(scores.X, expected_scores.X)
      pd.testing.assert_frame_equal(scores.obs, expected_scores.obs)
      pd.testing.assert_frame_equal(scores.var, expected_scores.var)
      expected_layers = expected_scores.layers
      assert isinstance(expected_layers, Mapping)
      layers = scores.layers
      assert isinstance(layers, Mapping)
      self.assertCountEqual(layers, expected_layers)
      for layer_name, layer in expected_layers.items():
        np.testing.assert_array_equal(layers[layer_name], layer)

  def test_scorer_metadata(self):
    response = atlas_service_pb2.ListVariantScoresMetadataResponse(
        variant_scorer_metadata=[
            atlas_service_pb2.VariantScorerMetadata(
                variant_scorer=atlas_service_pb2.VariantScorerInfo(
                    name='scorer1',
                    is_signed=True,
                ),
                metadata=[
                    atlas_service_pb2.Metadata(
                        tracks=dna_model_pb2.TracksMetadata(
                            metadata=[
                                dna_model_pb2.TrackMetadata(
                                    name='track1',
                                    strand=dna_model_pb2.Strand.STRAND_POSITIVE,
                                )
                            ],
                        ),
                    )
                ],
            ),
            atlas_service_pb2.VariantScorerMetadata(
                variant_scorer=atlas_service_pb2.VariantScorerInfo(
                    name='scorer2'
                ),
                metadata=[
                    atlas_service_pb2.Metadata(
                        tracks=dna_model_pb2.TracksMetadata(
                            metadata=[
                                dna_model_pb2.TrackMetadata(
                                    name='track2',
                                    strand=dna_model_pb2.Strand.STRAND_NEGATIVE,
                                )
                            ]
                        ),
                    )
                ],
            ),
        ]
    )

    mock_stub = mock.create_autospec(atlas.AtlasServiceStubType, instance=True)
    mock_stub.ListVariantScoresMetadata.return_value = response
    result = atlas.AtlasClient(stub=mock_stub).scorer_metadata()
    expected = {
        'scorer1': atlas.ScorerMetadata(
            name='scorer1',
            is_signed=True,
            track_metadata=pd.DataFrame(
                {'name': ['track1'], 'strand': ['+']}, index=['0']
            ),
        ),
        'scorer2': atlas.ScorerMetadata(
            name='scorer2',
            is_signed=False,
            track_metadata=pd.DataFrame(
                {'name': ['track2'], 'strand': ['-']}, index=['0']
            ),
        ),
    }
    self.assertCountEqual(result, expected)
    for name, metadata in result.items():
      self.assertEqual(metadata.name, expected[name].name)
      self.assertEqual(metadata.is_signed, expected[name].is_signed)
      pd.testing.assert_frame_equal(
          metadata.track_metadata, expected[name].track_metadata
      )

  def test_query_interval(self):
    mock_stub = mock.create_autospec(atlas.AtlasServiceStubType, instance=True)

    variant_scores = [
        atlas_service_pb2.DenseVariantScores(
            variant=dna_model_pb2.Variant(
                chromosome='chr1',
                position=100 + i,
                reference_bases='A',
                alternate_bases='T',
            ),
            scores=[
                atlas_service_pb2.DenseVariantScore(
                    variant_scorer=atlas_service_pb2.VariantScorerInfo(
                        name='scorer1'
                    ),
                    shape=[1, 1],
                    scores=struct.pack('f', 1.0 * i),
                    metadata=[
                        atlas_service_pb2.Metadata(
                            gene_scorers=atlas_service_pb2.GeneScorersMetadata(
                                metadata=[
                                    dna_model_pb2.GeneScorerMetadata(
                                        gene_id='gene_id_1',
                                        name='gene_1',
                                        strand=(
                                            dna_model_pb2.Strand.STRAND_POSITIVE
                                        ),
                                    )
                                ]
                            )
                        )
                    ],
                )
            ],
        )
        for i in range(3)
    ]
    mock_stub.ListDenseVariantScores.side_effect = [
        atlas_service_pb2.ListDenseVariantScoresResponse(
            variant_scores=[variant_scores[i]],
            next_page_token=f'page_token{i}' if i < 2 else '',
        )
        for i in range(len(variant_scores))
    ]
    response = atlas_service_pb2.ListVariantScoresMetadataResponse(
        variant_scorer_metadata=[
            atlas_service_pb2.VariantScorerMetadata(
                variant_scorer=atlas_service_pb2.VariantScorerInfo(
                    name='scorer1',
                    is_signed=True,
                ),
                metadata=[
                    atlas_service_pb2.Metadata(
                        tracks=dna_model_pb2.TracksMetadata(
                            metadata=[
                                dna_model_pb2.TrackMetadata(
                                    name='track1',
                                    strand=(
                                        dna_model_pb2.Strand.STRAND_UNSTRANDED
                                    ),
                                    ontology_term=dna_model_pb2.OntologyTerm(
                                        ontology_type=(
                                            dna_model_pb2.ONTOLOGY_TYPE_CL
                                        ),
                                        id=1,
                                    ),
                                ),
                                dna_model_pb2.TrackMetadata(
                                    name='track2',
                                    strand=(
                                        dna_model_pb2.Strand.STRAND_UNSTRANDED
                                    ),
                                    ontology_term=dna_model_pb2.OntologyTerm(
                                        ontology_type=(
                                            dna_model_pb2.ONTOLOGY_TYPE_CL
                                        ),
                                        id=2,
                                    ),
                                ),
                            ],
                        )
                    ),
                ],
            ),
        ]
    )
    mock_stub.ListVariantScoresMetadata.return_value = response

    interval = genome.Interval.from_str('chr1:100-120')
    result = atlas.AtlasClient(stub=mock_stub).query_interval(
        interval,
        requested_scorers=['scorer1'],
        ontology_terms=['CL:0000001'],
        max_workers=1,
    )

    self.assertEqual(mock_stub.ListDenseVariantScores.call_count, 3)
    mock_stub.ListVariantScoresMetadata.assert_called_once_with(
        atlas_service_pb2.ListVariantScoresMetadataRequest(
            organism=dna_model_pb2.Organism.ORGANISM_HOMO_SAPIENS
        ),
        metadata=(),
    )

    self.assertLen(result, 1)
    result = result['scorer1']
    expected_scores = anndata.AnnData(
        X=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                'gene_id': 'gene_id_1',
                'gene_name': 'gene_1',
                'strand': '+',
                'variant': [
                    genome.Variant.from_str('chr1:100:A>T'),
                    genome.Variant.from_str('chr1:101:A>T'),
                    genome.Variant.from_str('chr1:102:A>T'),
                ],
            },
            index=['0', '1', '2'],
        ),
        var=pd.DataFrame(
            {'name': 'track1', 'strand': '.', 'ontology_curie': 'CL:0000001'},
            index=['0'],
        ),
    )
    np.testing.assert_array_equal(result.X, expected_scores.X)
    pd.testing.assert_frame_equal(result.obs, expected_scores.obs)
    pd.testing.assert_frame_equal(result.var, expected_scores.var)

  def test_query_variant(self):
    mock_stub = mock.create_autospec(atlas.AtlasServiceStubType, instance=True)

    variant = genome.Variant.from_str('chr1:100:A>T')

    variant_score = atlas_service_pb2.DenseVariantScores(
        variant=variant.to_proto(),
        scores=[
            atlas_service_pb2.DenseVariantScore(
                variant_scorer=atlas_service_pb2.VariantScorerInfo(
                    name='scorer1'
                ),
                shape=[1, 2],
                scores=struct.pack('2f', *[1.0, 1.0]),
                metadata=[
                    atlas_service_pb2.Metadata(
                        gene_scorers=atlas_service_pb2.GeneScorersMetadata(
                            metadata=[
                                dna_model_pb2.GeneScorerMetadata(
                                    gene_id='gene_id_1',
                                    name='gene_1',
                                    strand=(
                                        dna_model_pb2.Strand.STRAND_POSITIVE
                                    ),
                                )
                            ]
                        )
                    ),
                    atlas_service_pb2.Metadata(
                        tracks=dna_model_pb2.TracksMetadata(
                            metadata=[
                                dna_model_pb2.TrackMetadata(
                                    name='track1',
                                    strand=(
                                        dna_model_pb2.Strand.STRAND_UNSTRANDED
                                    ),
                                    ontology_term=dna_model_pb2.OntologyTerm(
                                        ontology_type=(
                                            dna_model_pb2.ONTOLOGY_TYPE_CL
                                        ),
                                        id=1,
                                    ),
                                ),
                                dna_model_pb2.TrackMetadata(
                                    name='track2',
                                    strand=(
                                        dna_model_pb2.Strand.STRAND_UNSTRANDED
                                    ),
                                    ontology_term=dna_model_pb2.OntologyTerm(
                                        ontology_type=(
                                            dna_model_pb2.ONTOLOGY_TYPE_CL
                                        ),
                                        id=2,
                                    ),
                                ),
                            ],
                        ),
                    ),
                ],
            )
        ],
    )
    mock_stub.GetDenseVariantScores.return_value = variant_score
    result = atlas.AtlasClient(stub=mock_stub).query_variant(
        variant, requested_scorers=['scorer1']
    )

    mock_stub.GetDenseVariantScores.assert_called_once_with(
        atlas_service_pb2.GetDenseVariantScoresRequest(
            variant=variant.to_proto(),
            organism=dna_model_pb2.Organism.ORGANISM_HOMO_SAPIENS,
            filter='(scores.variant_scorer.name = "scorer1")',
        ),
        metadata=(),
    )

    self.assertLen(result, 1)
    actual = result['scorer1']
    expected_scores = anndata.AnnData(
        X=np.ones((1, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                'gene_id': 'gene_id_1',
                'gene_name': 'gene_1',
                'strand': '+',
                'variant': genome.Variant.from_str('chr1:100:A>T'),
            },
            index=['0'],
        ),
        var=pd.DataFrame(
            {
                'name': ['track1', 'track2'],
                'strand': '.',
                'ontology_curie': ['CL:0000001', 'CL:0000002'],
            },
            index=['0', '1'],
        ),
    )
    np.testing.assert_array_equal(actual.X, expected_scores.X)
    pd.testing.assert_frame_equal(actual.obs, expected_scores.obs)
    pd.testing.assert_frame_equal(actual.var, expected_scores.var)

  def test_query_variants(self):
    mock_stub = mock.create_autospec(atlas.AtlasServiceStubType, instance=True)

    variant = genome.Variant.from_str('chr1:100:A>T')

    variant_score = atlas_service_pb2.DenseVariantScores(
        variant=variant.to_proto(),
        scores=[
            atlas_service_pb2.DenseVariantScore(
                variant_scorer=atlas_service_pb2.VariantScorerInfo(
                    name='scorer1'
                ),
                shape=[1, 2],
                scores=struct.pack('2f', *[1.0, 1.0]),
                metadata=[
                    atlas_service_pb2.Metadata(
                        gene_scorers=atlas_service_pb2.GeneScorersMetadata(
                            metadata=[
                                dna_model_pb2.GeneScorerMetadata(
                                    gene_id='gene_id_1',
                                    name='gene_1',
                                    strand=(
                                        dna_model_pb2.Strand.STRAND_POSITIVE
                                    ),
                                )
                            ]
                        )
                    ),
                ],
            )
        ],
    )
    mock_stub.GetDenseVariantScores.return_value = variant_score
    metadata_response = atlas_service_pb2.ListVariantScoresMetadataResponse(
        variant_scorer_metadata=[
            atlas_service_pb2.VariantScorerMetadata(
                variant_scorer=atlas_service_pb2.VariantScorerInfo(
                    name='scorer1',
                    is_signed=True,
                ),
                metadata=[
                    atlas_service_pb2.Metadata(
                        tracks=dna_model_pb2.TracksMetadata(
                            metadata=[
                                dna_model_pb2.TrackMetadata(
                                    name='track1',
                                    strand=(
                                        dna_model_pb2.Strand.STRAND_UNSTRANDED
                                    ),
                                    ontology_term=dna_model_pb2.OntologyTerm(
                                        ontology_type=(
                                            dna_model_pb2.ONTOLOGY_TYPE_CL
                                        ),
                                        id=1,
                                    ),
                                ),
                                dna_model_pb2.TrackMetadata(
                                    name='track2',
                                    strand=(
                                        dna_model_pb2.Strand.STRAND_UNSTRANDED
                                    ),
                                    ontology_term=dna_model_pb2.OntologyTerm(
                                        ontology_type=(
                                            dna_model_pb2.ONTOLOGY_TYPE_CL
                                        ),
                                        id=2,
                                    ),
                                ),
                            ],
                        )
                    ),
                ],
            ),
        ]
    )
    mock_stub.ListVariantScoresMetadata.return_value = metadata_response

    result = atlas.AtlasClient(stub=mock_stub).query_variants(
        [variant],
        requested_scorers=['scorer1'],
    )

    mock_stub.ListVariantScoresMetadata.assert_called_once_with(
        atlas_service_pb2.ListVariantScoresMetadataRequest(
            organism=dna_model_pb2.Organism.ORGANISM_HOMO_SAPIENS
        ),
        metadata=(),
    )

    mock_stub.GetDenseVariantScores.assert_called_once_with(
        atlas_service_pb2.GetDenseVariantScoresRequest(
            variant=variant.to_proto(),
            organism=dna_model_pb2.Organism.ORGANISM_HOMO_SAPIENS,
            filter='(scores.variant_scorer.name = "scorer1")',
        ),
        metadata=mock.ANY,
    )

    self.assertLen(result, 1)
    actual = result['scorer1']
    expected_scores = anndata.AnnData(
        X=np.ones((1, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                'gene_id': 'gene_id_1',
                'gene_name': 'gene_1',
                'strand': '+',
                'variant': genome.Variant.from_str('chr1:100:A>T'),
            },
            index=['0'],
        ),
        var=pd.DataFrame(
            {
                'name': ['track1', 'track2'],
                'strand': '.',
                'ontology_curie': ['CL:0000001', 'CL:0000002'],
            },
            index=['0', '1'],
        ),
    )
    np.testing.assert_array_equal(actual.X, expected_scores.X)
    pd.testing.assert_frame_equal(actual.obs, expected_scores.obs)
    pd.testing.assert_frame_equal(actual.var, expected_scores.var)

  @parameterized.parameters(
      (grpc.StatusCode.INVALID_ARGUMENT, ValueError, 'foo'),
      (grpc.StatusCode.DEADLINE_EXCEEDED, TimeoutError, 'bar'),
      (grpc.StatusCode.UNAUTHENTICATED, PermissionError, 'baz'),
      (grpc.StatusCode.UNIMPLEMENTED, NotImplementedError, 'qux'),
      (grpc.StatusCode.OUT_OF_RANGE, IndexError, 'wux'),
      (grpc.StatusCode.UNKNOWN, grpc.RpcError, 'unknown error'),
  )
  def test_handle_rpc_error(self, code, expected_type, expected_message):

    class _RpcError(grpc.RpcError, grpc.Call):
      """Mock RPC error class."""

      def __init__(self, code, details):
        self._code = code
        self._details = details

      def code(self):
        return self._code

      def details(self):
        return self._details

      def initial_metadata(self):
        raise NotImplementedError()

      def trailing_metadata(self):
        raise NotImplementedError()

      def is_active(self):
        raise NotImplementedError()

      def add_callback(self, callback):
        raise NotImplementedError()

      def time_remaining(self):
        raise NotImplementedError()

      def cancel(self):
        raise NotImplementedError()

    with self.assertRaisesRegex(expected_type, expected_message):
      with atlas.handle_rpc_error():
        raise _RpcError(code, expected_message)

  def test_create(self):
    with (
        mock.patch.object(grpc, 'secure_channel') as mock_secure_channel,
        mock.patch.object(grpc, 'channel_ready_future'),
    ):
      client = atlas.create('foo')
      self.assertIsInstance(client, atlas.AtlasClient)
      mock_secure_channel.assert_called_once_with(
          'dns:///gdmscience.googleapis.com:443', mock.ANY, options=mock.ANY
      )


if __name__ == '__main__':
  absltest.main()
