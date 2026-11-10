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

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome import tensor_utils
from alphagenome.protos.v1 import dna_model_pb2 as dna_model_pb2_v1
from alphagenome.data import gene_expression_data
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.data import track_data
from alphagenome.interpretation import ism
from alphagenome.models.v2 import dna_client
from alphagenome.models.v2 import gene_expression_data_utils
from alphagenome.models.v2 import interval_scorers
from alphagenome.models.v2 import junction_data_utils
from alphagenome.models.v2 import model_condition
from alphagenome.models.v2 import track_data_utils
from alphagenome.models.v2 import variant_scorers
from alphagenome.protos.v2 import dna_model_pb2 as dna_model_pb2_v2
from alphagenome.protos.v2 import dna_model_service_pb2
import anndata
import grpc
import numpy as np
import pandas as pd


class _FakeRpcError(grpc.RpcError):
  """Fake RPC error for testing."""

  def __init__(self, code: grpc.StatusCode):
    self._code = code

  def code(self) -> grpc.StatusCode:
    return self._code


def _generate_prediction_proto(
    predictions: dna_client.Output,
    requested_outputs: Sequence[dna_model_pb2_v2.OutputType],
    response_proto_type,
):
  """Helper to generate proto responses for the provided predictions."""
  outputs = []
  for output_type in requested_outputs:
    prediction = predictions.get(dna_client.OutputType(output_type))
    if prediction is not None:
      if isinstance(prediction, track_data.TrackData):
        proto = track_data_utils.to_protos(prediction)
        outputs.append(
            dna_model_pb2_v2.Output(output_type=output_type, track_data=proto)
        )
      elif isinstance(prediction, junction_data.JunctionData):
        proto = junction_data_utils.to_protos(prediction)
        outputs.append(
            dna_model_pb2_v2.Output(
                output_type=output_type, junction_data=proto
            )
        )
      elif isinstance(prediction, gene_expression_data.GeneExpressionData):
        proto = gene_expression_data_utils.to_protos(prediction)
        outputs.append(
            dna_model_pb2_v2.Output(
                output_type=output_type, gene_expression_data=proto
            )
        )
      else:
        raise ValueError(f'Unsupported prediction type: {type(prediction)}')

  return response_proto_type(outputs=outputs)


def _generate_variant_proto(
    predictions: dna_client.VariantOutput,
    requested_outputs: Sequence[dna_model_pb2_v2.OutputType],
):
  """Helper to generate variant proto responses for the provided predictions."""
  reference_outputs = []
  alternate_outputs = []
  for output_type in requested_outputs:
    prediction = predictions.reference.get(dna_client.OutputType(output_type))
    if prediction is not None:
      if isinstance(prediction, track_data.TrackData):
        proto = track_data_utils.to_protos(prediction)
        reference_outputs.append(
            dna_model_pb2_v2.Output(output_type=output_type, track_data=proto)
        )
      elif isinstance(prediction, junction_data.JunctionData):
        proto = junction_data_utils.to_protos(prediction)
        reference_outputs.append(
            dna_model_pb2_v2.Output(
                output_type=output_type, junction_data=proto
            )
        )
      elif isinstance(prediction, gene_expression_data.GeneExpressionData):
        proto = gene_expression_data_utils.to_protos(prediction)
        reference_outputs.append(
            dna_model_pb2_v2.Output(
                output_type=output_type, gene_expression_data=proto
            )
        )
      else:
        raise ValueError(f'Unsupported prediction type: {type(prediction)}')

  for output_type in requested_outputs:
    prediction = predictions.alternate.get(dna_client.OutputType(output_type))
    if prediction is not None:
      if isinstance(prediction, track_data.TrackData):
        proto = track_data_utils.to_protos(prediction)
        alternate_outputs.append(
            dna_model_pb2_v2.Output(output_type=output_type, track_data=proto)
        )
      elif isinstance(prediction, junction_data.JunctionData):
        proto = junction_data_utils.to_protos(prediction)
        alternate_outputs.append(
            dna_model_pb2_v2.Output(
                output_type=output_type, junction_data=proto
            )
        )
      elif isinstance(prediction, gene_expression_data.GeneExpressionData):
        proto = gene_expression_data_utils.to_protos(prediction)
        alternate_outputs.append(
            dna_model_pb2_v2.Output(
                output_type=output_type, gene_expression_data=proto
            )
        )
      else:
        raise ValueError(f'Unsupported prediction type: {type(prediction)}')
  return dna_model_service_pb2.PredictVariantResponse(
      reference_outputs=reference_outputs, alternate_outputs=alternate_outputs
  )


def _generate_variant_scoring_proto(
    scores: list[anndata.AnnData],
    response_proto_type,
    variant: genome.Variant | None = None,
):
  """Helper to generate variant scoring responses for the provided scores."""
  outputs = []
  for score in scores:
    track_metadata_protos = track_data_utils.metadata_to_proto(
        score.var
    ).metadata
    gene_metadata_protos = []
    if score.obs is not None and 'gene_id' in score.obs:
      for _, row in score.obs.iterrows():
        if (strand := row.get('strand')) is not None:
          strand = genome.Strand.from_str(strand).to_proto()
        gene_metadata_protos.append(
            dna_model_pb2_v1.GeneScorerMetadata(
                gene_id=row['gene_id'],
                strand=strand,
                name=row.get('gene_name'),
                type=row.get('gene_type'),
                junction_start=row.get('junction_Start'),
                junction_end=row.get('junction_End'),
            )
        )

    if 'quantiles' in score.layers:  # pyrefly: ignore[not-iterable]
      quantiles_layer = score.layers['quantiles']  # pyrefly: ignore[bad-index]
      score_tensor = np.stack(  # pyrefly: ignore[no-matching-overload]
          [score.X, quantiles_layer]
      )
    else:
      # pyrefly: ignore[bad-index, unsupported-operation]
      score_tensor = score.X[np.newaxis]

    tensor_proto, _ = tensor_utils.pack_tensor(score_tensor)
    outputs.append(
        dna_model_pb2_v2.ScoreVariantOutput(
            variant_data=dna_model_pb2_v2.VariantData(
                values=tensor_proto,
                metadata=dna_model_pb2_v2.VariantMetadata(
                    variant=variant.to_proto() if variant else None,
                    track_metadata=track_metadata_protos,
                    gene_metadata=gene_metadata_protos,
                ),
            )
        )
    )
  return response_proto_type(outputs=outputs)


def _generate_interval_scoring_proto(
    scores: list[anndata.AnnData],
    response_proto_type,
    interval: genome.Interval,
):
  """Helper to generate interval scoring responses for the provided scores."""
  outputs = []
  for score in scores:
    track_metadata_protos = track_data_utils.metadata_to_proto(
        score.var
    ).metadata
    gene_metadata_protos = []
    if score.obs is not None and 'gene_id' in score.obs:
      for _, row in score.obs.iterrows():
        if (strand := row.get('strand')) is not None:
          strand = genome.Strand.from_str(strand).to_proto()
        gene_metadata_protos.append(
            dna_model_pb2_v1.GeneScorerMetadata(
                gene_id=row['gene_id'],
                strand=strand,
                name=row.get('gene_name'),
                type=row.get('gene_type'),
            )
        )

    # pyrefly: ignore[bad-index, unsupported-operation]
    tensor_proto, _ = tensor_utils.pack_tensor(score.X[np.newaxis])

    outputs.append(
        dna_model_pb2_v2.ScoreIntervalOutput(
            interval_data=dna_model_pb2_v2.IntervalData(
                values=tensor_proto,
                metadata=dna_model_pb2_v2.IntervalMetadata(
                    interval=interval.to_proto(),
                    track_metadata=track_metadata_protos,
                    gene_metadata=gene_metadata_protos,
                ),
            )
        )
    )
  return response_proto_type(outputs=outputs)


def _create_example_model_predictions(
    *,
    sequence_length: int = 0,
    value: float = 0.0,
    interval: genome.Interval | None = None,
) -> dna_client.Output:
  """Helper to generate example model predictions."""
  if sequence_length == 0:
    if interval is None:
      raise ValueError('Interval must be provided for zero length sequences.')
    sequence_length = interval.width

  if interval is None:
    # Gene expression is not supported for predict_sequence with no interval.
    sc_gene_expression_data = None
  else:
    sc_gene_expression_data = gene_expression_data.GeneExpressionData(
        values=np.full((16, 64), value, dtype=np.float32),
        gene_metadata=pd.DataFrame({
            'gene_id': [f'{i}' for i in range(16)],
            'Chromosome': ['chr1'] * 16,
            'Start': [i * 100 for i in range(16)],
            'End': [i * 100 + 100 for i in range(16)],
        }),
        track_metadata=pd.DataFrame({
            'name': [f'{i}' for i in range(64)],
            'strand': ['.'] * 64,
            'biosample_curie': ['UBERON:0000005'] * 64,
            'imputation_mode': ['imputed'] * 64,
            'biosample_type': ['tissue'] * 64,
            'biosample_name': ['sample'] * 64,
        }),
        interval=interval,
    )
  return dna_client.Output(
      atac=track_data.TrackData(
          values=np.full((sequence_length, 32), value, dtype=np.float32),
          metadata=pd.DataFrame({
              'name': [f'{i}' for i in range(32)],
              'strand': ['.'] * 32,
              'biosample_curie': ['UBERON:0000005'] * 32,
              'imputation_mode': ['imputed'] * 32,
              'biosample_type': ['tissue'] * 32,
              'biosample_name': ['sample'] * 32,
          }),
          interval=interval,
      ),
      rna_seq=track_data.TrackData(
          values=np.full((sequence_length, 64), value, dtype=np.float32),
          metadata=pd.DataFrame({
              'name': [f'{i}' for i in range(64)],
              'strand': ['.'] * 64,
              'biosample_curie': ['UBERON:0000005'] * 64,
              'imputation_mode': ['imputed'] * 64,
              'biosample_type': ['tissue'] * 64,
              'biosample_name': ['sample'] * 64,
          }),
          interval=interval,
      ),
      splice_junctions=junction_data.JunctionData(
          junctions=np.array([
              genome.Interval('chr1', 0, 1000, strand='+'),
              genome.Interval('chr1', 1500, 2000, strand='-'),
          ]),
          values=np.full([2, 3], value, dtype=np.float32),
          metadata=pd.DataFrame({
              'name': [f'{i}' for i in range(3)],
              'biosample_curie': ['UBERON:0000005'] * 3,
              'imputation_mode': ['imputed'] * 3,
              'biosample_type': ['tissue'] * 3,
              'biosample_name': ['sample'] * 3,
          }),
          interval=interval,
      ),
      sc_gene_expression=sc_gene_expression_data,
  )


def _create_mock_channel_and_stub(response_callable):
  """Creates a mock gRPC channel and stub for non-streaming (unary) endpoint."""
  mock_channel = mock.create_autospec(grpc.Channel)
  mock_stub = mock.create_autospec(
      grpc.UnaryUnaryMultiCallable, side_effect=response_callable
  )
  mock_channel.unary_unary.return_value = mock_stub
  return mock_channel, mock_stub


class ClientTest(parameterized.TestCase):

  def _assert_output_equal(
      self,
      actual: dna_client.Output,
      expected: dna_client.Output,
      msg,
  ) -> None:
    """Helper function to compare Output objects."""
    self.assertEqual(actual.atac, expected.atac, msg)
    self.assertEqual(actual.cage, expected.cage, msg)
    self.assertEqual(actual.dnase, expected.dnase, msg)
    self.assertEqual(actual.rna_seq, expected.rna_seq, msg)
    self.assertEqual(actual.chip_histone, expected.chip_histone, msg)
    self.assertEqual(actual.chip_tf, expected.chip_tf, msg)
    self.assertEqual(actual.splice_sites, expected.splice_sites, msg)
    self.assertEqual(actual.splice_site_usage, expected.splice_site_usage, msg)
    self.assertEqual(actual.splice_junctions, expected.splice_junctions)
    self.assertEqual(actual.contact_maps, expected.contact_maps, msg)
    self.assertEqual(
        actual.sc_gene_expression, expected.sc_gene_expression, msg
    )
    self.assertEqual(actual.sc_atac, expected.sc_atac, msg)

  def _assert_track_metadata_equal(
      self,
      actual: track_data.TrackMetadata | None,
      expected: track_data.TrackMetadata | None,
      msg,
  ) -> None:
    if actual is not None and expected is not None:
      pd.testing.assert_frame_equal(actual, expected)
    else:
      self.assertEqual(actual, expected, msg)

  def _assert_output_metadata_equal(
      self,
      actual: dna_client.OutputMetadata,
      expected: dna_client.OutputMetadata,
      msg,
  ) -> None:
    """Helper function to compare OutputMetadata objects."""
    self._assert_track_metadata_equal(actual.atac, expected.atac, msg)
    self._assert_track_metadata_equal(actual.cage, expected.cage, msg)
    self._assert_track_metadata_equal(actual.dnase, expected.dnase, msg)
    self._assert_track_metadata_equal(actual.rna_seq, expected.rna_seq, msg)
    self._assert_track_metadata_equal(
        actual.chip_histone, expected.chip_histone, msg
    )
    self._assert_track_metadata_equal(actual.chip_tf, expected.chip_tf, msg)
    self._assert_track_metadata_equal(
        actual.splice_sites, expected.splice_sites, msg
    )
    self._assert_track_metadata_equal(
        actual.splice_site_usage, expected.splice_site_usage, msg
    )
    self._assert_track_metadata_equal(
        actual.splice_junctions, expected.splice_junctions, msg
    )
    self._assert_track_metadata_equal(
        actual.sc_gene_expression, expected.sc_gene_expression, msg
    )
    self._assert_track_metadata_equal(actual.sc_atac, expected.sc_atac, msg)

  def _assert_track_data_equal(
      self,
      actual: track_data.TrackData,
      expected: track_data.TrackData,
      msg,
  ) -> None:
    """Helper function to compare TrackData objects."""
    pd.testing.assert_frame_equal(actual.metadata, expected.metadata)
    self.assertEqual(actual.interval, expected.interval, msg)
    np.testing.assert_array_equal(actual.values, expected.values)
    self.assertEqual(actual.uns, expected.uns, msg)

  def _assert_junction_data_equal(
      self,
      actual: junction_data.JunctionData,
      expected: junction_data.JunctionData,
      msg,
  ) -> None:
    pd.testing.assert_frame_equal(actual.metadata, expected.metadata)
    np.testing.assert_array_equal(actual.junctions, expected.junctions)
    np.testing.assert_array_equal(actual.values, expected.values)
    self.assertEqual(actual.interval, expected.interval, msg)
    self.assertEqual(actual.uns, expected.uns, msg)

  def _assert_gene_expression_data_equal(
      self,
      actual: gene_expression_data.GeneExpressionData,
      expected: gene_expression_data.GeneExpressionData,
      msg,
  ) -> None:
    pd.testing.assert_frame_equal(actual.gene_metadata, expected.gene_metadata)
    pd.testing.assert_frame_equal(
        actual.track_metadata, expected.track_metadata
    )
    self.assertEqual(actual.interval, expected.interval, msg)
    np.testing.assert_array_equal(actual.values, expected.values)

  def _assert_anndata_equal(
      self,
      actual: anndata.AnnData,
      expected: anndata.AnnData,
  ) -> None:
    """Helper function to compare AnnData objects."""
    pd.testing.assert_frame_equal(actual.obs, expected.obs, check_like=True)
    pd.testing.assert_frame_equal(actual.var, expected.var, check_like=True)
    self.assertEqual(actual.uns['interval'], expected.uns['interval'])
    if 'variant' in expected.uns:
      self.assertEqual(actual.uns['variant'], expected.uns['variant'])
    np.testing.assert_array_equal(actual.X, expected.X)
    if 'quantiles' in expected.layers:  # pyrefly: ignore[not-iterable]
      actual_q = actual.layers['quantiles']  # pyrefly: ignore[bad-index]
      expected_q = expected.layers['quantiles']  # pyrefly: ignore[bad-index]
      np.testing.assert_array_equal(actual_q, expected_q)
    else:
      self.assertNotIn(
          'quantiles', actual.layers  # pyrefly: ignore[bad-argument-type]
      )

  def setUp(self):
    super().setUp()
    self.addTypeEqualityFunc(dna_client.Output, self._assert_output_equal)
    self.addTypeEqualityFunc(
        track_data.TrackData, self._assert_track_data_equal
    )
    self.addTypeEqualityFunc(
        junction_data.JunctionData, self._assert_junction_data_equal
    )
    self.addTypeEqualityFunc(
        gene_expression_data.GeneExpressionData,
        self._assert_gene_expression_data_equal,
    )
    self.addTypeEqualityFunc(
        dna_client.OutputMetadata, self._assert_output_metadata_equal
    )

  @parameterized.product(
      (
          dict(
              requested_output_types=[*dna_client.OutputType],
              sequence_length=16_384,
              expected=dna_client.Output(
                  atac=track_data.TrackData(
                      values=np.zeros((16_384, 32), dtype=np.float32),
                      metadata=pd.DataFrame({
                          'name': [f'{i}' for i in range(32)],
                          'strand': ['.'] * 32,
                          'biosample_curie': ['UBERON:0000005'] * 32,
                          'imputation_mode': ['imputed'] * 32,
                          'biosample_type': ['tissue'] * 32,
                          'biosample_name': ['sample'] * 32,
                      }),
                  ),
                  rna_seq=track_data.TrackData(
                      values=np.zeros((16_384, 64), dtype=np.float32),
                      metadata=pd.DataFrame({
                          'name': [f'{i}' for i in range(64)],
                          'strand': ['.'] * 64,
                          'biosample_curie': ['UBERON:0000005'] * 64,
                          'imputation_mode': ['imputed'] * 64,
                          'biosample_type': ['tissue'] * 64,
                          'biosample_name': ['sample'] * 64,
                      }),
                  ),
                  splice_junctions=junction_data.JunctionData(
                      junctions=np.array([
                          genome.Interval('chr1', 0, 1000, strand='+'),
                          genome.Interval('chr1', 1500, 2000, strand='-'),
                      ]),
                      values=np.zeros([2, 3], dtype=np.float32),
                      metadata=pd.DataFrame({
                          'name': [f'{i}' for i in range(3)],
                          'biosample_curie': ['UBERON:0000005'] * 3,
                          'imputation_mode': ['imputed'] * 3,
                          'biosample_type': ['tissue'] * 3,
                          'biosample_name': ['sample'] * 3,
                      }),
                  ),
                  sc_gene_expression=None,
              ),
          ),
          dict(
              requested_output_types={dna_client.OutputType.RNA_SEQ},
              sequence_length=16_384,
              expected=dna_client.Output(
                  rna_seq=track_data.TrackData(
                      values=np.zeros((16_384, 64), dtype=np.float32),
                      metadata=pd.DataFrame({
                          'name': [f'{i}' for i in range(64)],
                          'strand': ['.'] * 64,
                          'biosample_curie': ['UBERON:0000005'] * 64,
                          'imputation_mode': ['imputed'] * 64,
                          'biosample_type': ['tissue'] * 64,
                          'biosample_name': ['sample'] * 64,
                      }),
                  ),
              ),
          ),
          dict(
              requested_output_types={dna_client.OutputType.CONTACT_MAPS},
              sequence_length=16_384,
              expected=dna_client.Output(),
          ),
      ),
      model_version=[None, dna_client.ModelVersion.FOLD_1],
      with_interval=[True, False],
  )
  def test_predict_sequence(
      self,
      requested_output_types: set[dna_client.OutputType] | None,
      sequence_length: int,
      expected: dna_client.Output,
      model_version: dna_client.ModelVersion | None,
      with_interval: bool,
  ):
    mock_predictions = _create_example_model_predictions(
        sequence_length=sequence_length
    )

    expected_request = dna_model_service_pb2.PredictSequenceRequest(
        sequence='A' * sequence_length,
        organism=dna_model_pb2_v1.ORGANISM_HOMO_SAPIENS,
        condition=dna_model_pb2_v2.Condition(uberon=4),
        requested_outputs=[
            o.to_proto()
            for o in requested_output_types  # pyrefly: ignore[not-iterable]
        ],
        model_version=model_version.name if model_version else None,
    )

    interval = None
    if with_interval:
      interval = genome.Interval('chr1', 0, sequence_length)
      expected_dict = {}
      for field in dataclasses.fields(expected):
        output_type = field.metadata['output_type']
        if output := expected.get(output_type):
          expected_dict[field.name] = dataclasses.replace(
              output, interval=interval
          )
      expected = dna_client.Output(**expected_dict)

    def _mock_generate(request, metadata):
      del metadata
      self.assertEqual(request, expected_request)
      return _generate_prediction_proto(
          mock_predictions,
          request.requested_outputs,
          dna_model_service_pb2.PredictSequenceResponse,
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(
        channel=mock_channel, model_version=model_version
    )

    outputs = model.predict_sequence(
        'A' * sequence_length,
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=requested_output_types,  # pyrefly: ignore[bad-argument-type]
        condition=model_condition.from_proto(
            dna_model_pb2_v2.Condition(uberon=4)
        ),
        interval=interval,
    )

    self.assertEqual(outputs, expected)
    mock_stub.assert_called_once()

  @parameterized.product(
      sequence_length=[dna_client.SEQUENCE_LENGTH_16KB],
      num_predictions=[1, 4],
      num_workers=[1, 4],
      with_interval=[True, False],
  )
  def test_predict_sequences(
      self,
      sequence_length: int,
      num_predictions: int,
      num_workers: int,
      with_interval: bool,
  ):
    interval = None
    if with_interval:
      interval = genome.Interval('chr1', 0, sequence_length)

    mock_predictions = _create_example_model_predictions(
        sequence_length=sequence_length, interval=interval
    )

    def _mock_generate(request, metadata):
      del metadata
      return _generate_prediction_proto(
          mock_predictions,
          request.requested_outputs,
          dna_model_service_pb2.PredictSequenceResponse,
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.predict_sequences(
        ['A' * sequence_length] * num_predictions,
        conditions=[
            model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
        ]
        * num_predictions,
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=[*dna_client.OutputType],
        progress_bar=False,
        max_workers=num_workers,
        intervals=(
            [interval] * num_predictions  # pyrefly: ignore[bad-argument-type]
            if with_interval
            else None
        ),
    )
    for output in outputs:
      self.assertEqual(output, mock_predictions)

    self.assertEqual(mock_stub.call_count, num_predictions)

  def test_predict_sequences_error_raises(self):
    mock_channel, mock_stub = _create_mock_channel_and_stub(
        _FakeRpcError(grpc.StatusCode.INTERNAL)
    )
    model = dna_client.DnaClient(channel=mock_channel)
    with self.assertRaises(grpc.RpcError):
      model.predict_sequences(
          ['A' * dna_client.SEQUENCE_LENGTH_16KB],
          conditions=[
              model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
          ],
          requested_outputs=[*dna_client.OutputType],
      )

    mock_stub.assert_called_once()

  def test_invalid_sequence_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError,
        'Invalid sequence, must only contain the characters "ACGTN". Found'
        ' invalid characters: "[wozers,]*"',
    ):
      model.predict_sequence(
          'wowzers',
          requested_outputs=[dna_client.OutputType.ATAC],
          condition=model_condition.from_proto(
              dna_model_pb2_v2.Condition(uberon=4)
          ),
      )

  @parameterized.product(
      (
          dict(
              requested_output_types={dna_client.OutputType.ATAC},
              interval=genome.Interval('chr1', 0, 16_384),
              expected=dna_client.Output(
                  atac=track_data.TrackData(
                      values=np.zeros((16_384, 32), dtype=np.float32),
                      metadata=pd.DataFrame({
                          'name': [f'{i}' for i in range(32)],
                          'strand': ['.'] * 32,
                          'biosample_curie': ['UBERON:0000005'] * 32,
                          'imputation_mode': ['imputed'] * 32,
                          'biosample_type': ['tissue'] * 32,
                          'biosample_name': ['sample'] * 32,
                      }),
                      interval=genome.Interval('chr1', 0, 16_384),
                  )
              ),
          ),
      ),
      organism=[
          dna_client.Organism.HOMO_SAPIENS,
          dna_client.Organism.MUS_MUSCULUS,
      ],
      model_version=[None, dna_client.ModelVersion.FOLD_1],
  )
  def test_predict_interval(
      self,
      requested_output_types,
      interval,
      organism,
      expected,
      model_version,
  ):
    mock_predictions = _create_example_model_predictions(interval=interval)

    def _mock_generate(request, metadata):
      del metadata
      return _generate_prediction_proto(
          mock_predictions,
          request.requested_outputs,
          dna_model_service_pb2.PredictIntervalResponse,
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(
        channel=mock_channel, model_version=model_version
    )

    outputs = model.predict_interval(
        interval=interval,
        organism=organism,
        requested_outputs=requested_output_types,
        condition=model_condition.from_proto(
            dna_model_pb2_v2.Condition(uberon=4)
        ),
    )
    self.assertEqual(expected, outputs)

    mock_stub.assert_called_once()

  @parameterized.product(
      intervals=[
          [genome.Interval('chr1', 0, 16_384)],
          [
              genome.Interval('chr1', 0, 16_384),
              genome.Interval('chr1', 16_384, 32_768),
          ],
      ],
      num_workers=[1, 4],
  )
  def test_predict_intervals(self, intervals, num_workers):
    examples = {
        (interval.start, interval.end): _create_example_model_predictions(
            interval=interval, value=interval.start
        )
        for interval in intervals
    }

    def _mock_generate(request, metadata):
      del metadata
      return _generate_prediction_proto(
          examples[(request.interval.start, request.interval.end)],
          request.requested_outputs,
          dna_model_service_pb2.PredictIntervalResponse,
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.predict_intervals(
        intervals=intervals,
        conditions=[
            model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
        ]
        * len(intervals),
        requested_outputs=[*dna_client.OutputType],
        max_workers=num_workers,
        progress_bar=False,
    )
    for interval, output in zip(intervals, outputs):
      self.assertEqual(output, examples[(interval.start, interval.end)])

    self.assertLen(intervals, mock_stub.call_count)

  def test_predict_intervals_error_raises(self):
    mock_channel, mock_stub = _create_mock_channel_and_stub(
        _FakeRpcError(grpc.StatusCode.INTERNAL)
    )
    model = dna_client.DnaClient(channel=mock_channel)
    with self.assertRaises(grpc.RpcError):
      model.predict_intervals(
          [genome.Interval('chr1', 0, 16_384)],
          conditions=[
              model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
          ],
          requested_outputs=[*dna_client.OutputType],
      )

    mock_stub.assert_called_once()

  @parameterized.product(
      (
          dict(
              requested_output_types={dna_client.OutputType.ATAC},
              interval=genome.Interval('chr1', 0, 16_384),
              variant=genome.Variant('chr1', 1000, 'A', 'C'),
              expected=(
                  dna_client.Output(
                      atac=track_data.TrackData(
                          values=np.zeros((16_384, 32), dtype=np.float32),
                          metadata=pd.DataFrame({
                              'name': [f'{i}' for i in range(32)],
                              'strand': ['.'] * 32,
                              'biosample_curie': ['UBERON:0000005'] * 32,
                              'imputation_mode': ['imputed'] * 32,
                              'biosample_type': ['tissue'] * 32,
                              'biosample_name': ['sample'] * 32,
                          }),
                          interval=genome.Interval('chr1', 0, 16_384),
                      )
                  ),
                  dna_client.Output(
                      atac=track_data.TrackData(
                          values=np.ones((16_384, 32), dtype=np.float32),
                          metadata=pd.DataFrame({
                              'name': [f'{i}' for i in range(32)],
                              'strand': ['.'] * 32,
                              'biosample_curie': ['UBERON:0000005'] * 32,
                              'imputation_mode': ['imputed'] * 32,
                              'biosample_type': ['tissue'] * 32,
                              'biosample_name': ['sample'] * 32,
                          }),
                          interval=genome.Interval('chr1', 0, 16_384),
                      )
                  ),
              ),
          ),
      ),
      organism=[
          dna_client.Organism.HOMO_SAPIENS,
          dna_client.Organism.MUS_MUSCULUS,
      ],
      model_version=[None, dna_client.ModelVersion.FOLD_1],
  )
  def test_predict_variant(
      self,
      requested_output_types,
      interval,
      variant,
      organism,
      expected,
      model_version,
  ):
    mock_predictions = dna_client.VariantOutput(
        reference=_create_example_model_predictions(interval=interval, value=0),
        alternate=_create_example_model_predictions(interval=interval, value=1),
    )
    expected = dna_client.VariantOutput(
        reference=expected[0],
        alternate=expected[1],
    )

    def _mock_generate(request, metadata):
      del metadata
      return _generate_variant_proto(
          mock_predictions, request.requested_outputs
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(
        channel=mock_channel, model_version=model_version
    )

    outputs = model.predict_variant(
        interval=interval,
        variant=variant,
        organism=organism,
        requested_outputs=requested_output_types,
        condition=model_condition.from_proto(
            dna_model_pb2_v2.Condition(uberon=4)
        ),
    )

    self.assertEqual(expected.reference, outputs.reference)
    self.assertEqual(expected.alternate, outputs.alternate)

    mock_stub.assert_called_once()

  @parameterized.product(
      intervals=[
          genome.Interval('chr1', 0, 16_384),
          [
              genome.Interval('chr1', 0, 16_384),
              genome.Interval('chr1', 16_384, 32_768),
          ],
      ],
      variants=[
          [
              genome.Variant('chr1', 1000, 'A', 'C'),
              genome.Variant('chr1', 2000, 'A', 'C'),
          ],
      ],
      num_workers=[1, 4],
  )
  def test_predict_variants(self, intervals, variants, num_workers):
    if isinstance(intervals, genome.Interval):
      intervals = [intervals] * len(variants)
    examples = {
        (interval.start, interval.end): _create_example_model_predictions(
            interval=interval, value=interval.start
        )
        for interval in intervals
    }

    def _mock_generate(request, metadata):
      del metadata
      key = (request.interval.start, request.interval.end)
      return _generate_variant_proto(
          dna_client.VariantOutput(
              reference=examples[key], alternate=examples[key]
          ),
          request.requested_outputs,
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.predict_variants(
        intervals=intervals,
        variants=variants,
        conditions=[
            model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
        ]
        * len(intervals),
        requested_outputs=[*dna_client.OutputType],
        max_workers=num_workers,
        progress_bar=False,
    )
    for interval, output in zip(intervals, outputs, strict=True):
      expected = examples[(interval.start, interval.end)]
      self.assertEqual(output.reference, expected)
      self.assertEqual(output.alternate, expected)

    self.assertLen(intervals, mock_stub.call_count)

  def test_predict_variants_error_raises(self):
    mock_channel, mock_stub = _create_mock_channel_and_stub(
        _FakeRpcError(grpc.StatusCode.INTERNAL)
    )
    model = dna_client.DnaClient(channel=mock_channel)
    with self.assertRaises(grpc.RpcError):
      model.predict_variants(
          [genome.Interval('chr1', 0, 16_384)],
          [genome.Variant('chr1', 1000, 'A', 'C')],
          conditions=[
              model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
          ],
          requested_outputs=[*dna_client.OutputType],
      )

    mock_stub.assert_called_once()

  def test_predict_variants_mismatch_lengths_raises(self):
    mock_channel, mock_stub = _create_mock_channel_and_stub(None)
    model = dna_client.DnaClient(channel=mock_channel)
    with self.assertRaisesRegex(
        ValueError, 'Intervals and variants must have the same length'
    ):
      model.predict_variants(
          [
              genome.Interval('chr1', 0, 16_384),
              genome.Interval('chr1', 0, 16_384),
          ],
          [genome.Variant('chr1', 1000, 'A', 'C')],
          conditions=[
              model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
          ],
          requested_outputs=[*dna_client.OutputType],
      )
    mock_stub.assert_not_called()

  def test_observed_metadata(self):

    def _mock_generate(requests, metadata):
      del metadata
      del requests
      return dna_model_service_pb2.ObservedMetadataResponse(
          observed_metadata=[
              dna_model_pb2_v2.ObservedMetadata(
                  output_type=dna_model_pb2_v2.OUTPUT_TYPE_ATAC,
                  tracks=dna_model_pb2_v2.TracksMetadata(
                      metadata=[
                          dna_model_pb2_v2.TrackMetadata(
                              name=f'track_{i}',
                              strand=dna_model_pb2_v1.STRAND_UNSTRANDED,
                              biosample=dna_model_pb2_v1.Biosample(
                                  name='sample',
                                  type=dna_model_pb2_v1.BiosampleType.Value(
                                      'BIOSAMPLE_TYPE_TISSUE'
                                  ),
                              ),
                              condition=dna_model_pb2_v2.Condition(uberon=5),
                              imputation_mode='imputed',
                          )
                          for i in range(32)
                      ]
                  ),
              ),
              dna_model_pb2_v2.ObservedMetadata(
                  output_type=dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_JUNCTIONS,
                  junctions=dna_model_pb2_v2.JunctionsMetadata(
                      metadata=[
                          dna_model_pb2_v2.JunctionMetadata(
                              name=f'junction_{i}',
                              biosample=dna_model_pb2_v1.Biosample(
                                  name='sample',
                                  type=dna_model_pb2_v1.BiosampleType.Value(
                                      'BIOSAMPLE_TYPE_TISSUE'
                                  ),
                              ),
                              condition=dna_model_pb2_v2.Condition(uberon=5),
                              imputation_mode='imputed',
                          )
                          for i in range(16)
                      ]
                  ),
              ),
          ]
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.observed_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
    expected = dna_client.OutputMetadata(
        atac=pd.DataFrame({
            'name': [f'track_{i}' for i in range(32)],
            'strand': ['.'] * 32,
            'biosample_curie': ['UBERON:0000005'] * 32,
            'imputation_mode': ['imputed'] * 32,
            'biosample_type': ['tissue'] * 32,
            'biosample_name': ['sample'] * 32,
        }),
        splice_junctions=pd.DataFrame({
            'name': [f'junction_{i}' for i in range(16)],
            'biosample_curie': ['UBERON:0000005'] * 16,
            'imputation_mode': ['imputed'] * 16,
            'biosample_type': ['tissue'] * 16,
            'biosample_name': ['sample'] * 16,
        }),
    )
    self.assertEqual(expected, outputs)
    mock_stub.assert_called_once()

  @parameterized.product(organism=list(dna_client.Organism))
  def test_supported_conditions(self, organism):
    def _mock_generate(request, metadata):
      del metadata
      del request
      return dna_model_service_pb2.SupportedConditionsResponse(
          supported_conditions=[
              dna_model_pb2_v2.SupportedCondition(
                  condition=dna_model_pb2_v2.Condition(uberon=5),
                  biosample=dna_model_pb2_v1.Biosample(
                      name='sample',
                      type=dna_model_pb2_v1.BiosampleType.Value(
                          'BIOSAMPLE_TYPE_TISSUE'
                      ),
                  ),
                  output_type_imputation_modes=[
                      dna_model_pb2_v2.OutputTypeImputationMode(
                          output_type=dna_model_pb2_v2.OUTPUT_TYPE_ATAC,
                          imputation_mode='imputed',
                      ),
                      dna_model_pb2_v2.OutputTypeImputationMode(
                          output_type=dna_model_pb2_v2.OUTPUT_TYPE_RNA_SEQ,
                          imputation_mode='observed',
                      ),
                      dna_model_pb2_v2.OutputTypeImputationMode(
                          output_type=dna_model_pb2_v2.OUTPUT_TYPE_DNASE,
                          imputation_mode='imputed',
                      ),
                  ],
              )
          ]
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.supported_conditions(organism=organism)
    expected = pd.DataFrame([{
        'condition': 'UBERON:0000005',
        'biosample_type': 'tissue',
        'biosample_name': 'sample',
        'atac': 'imputed',
        'rna_seq': 'observed',
        'dnase': 'imputed',
    }])
    pd.testing.assert_frame_equal(expected, outputs)
    mock_stub.assert_called_once()

  def test_output_type_to_proto(self):
    expected = [
        dna_model_pb2_v2.OUTPUT_TYPE_ATAC,
        dna_model_pb2_v2.OUTPUT_TYPE_CAGE,
        dna_model_pb2_v2.OUTPUT_TYPE_DNASE,
        dna_model_pb2_v2.OUTPUT_TYPE_RNA_SEQ,
        dna_model_pb2_v2.OUTPUT_TYPE_CHIP_HISTONE,
        dna_model_pb2_v2.OUTPUT_TYPE_CHIP_TF,
        dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_SITES,
        dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_SITE_USAGE,
        dna_model_pb2_v2.OUTPUT_TYPE_SPLICE_JUNCTIONS,
        dna_model_pb2_v2.OUTPUT_TYPE_CONTACT_MAPS,
        dna_model_pb2_v2.OUTPUT_TYPE_SC_GENE_EXPRESSION,
        dna_model_pb2_v2.OUTPUT_TYPE_SC_ATAC,
        dna_model_pb2_v2.OUTPUT_TYPE_SC_GENE_EXPRESSION_DENOISED,
        dna_model_pb2_v2.OUTPUT_TYPE_CELL_EMBEDDINGS,
        dna_model_pb2_v2.OUTPUT_TYPE_METHYLATION,
    ]

    for output_type, expected_proto in zip(
        dna_client.OutputType, expected, strict=True
    ):
      self.assertEqual(output_type.to_proto(), expected_proto)

  def test_organism_to_proto(self):
    expected = [
        dna_model_pb2_v1.ORGANISM_HOMO_SAPIENS,
        dna_model_pb2_v1.ORGANISM_MUS_MUSCULUS,
    ]

    for organism, expected_proto in zip(
        dna_client.Organism, expected, strict=True
    ):
      self.assertEqual(organism.to_proto(), expected_proto)

  @parameterized.product(
      (
          dict(
              interval=genome.Interval('chr1', 0, 16_384),
              scorers=[
                  interval_scorers.GeneMaskScorer(
                      requested_output=dna_client.OutputType.ATAC,
                      width=2_001,
                      aggregation_type=(
                          interval_scorers.IntervalAggregationType.MEAN
                      ),
                  ),
                  interval_scorers.GeneMaskScorer(
                      requested_output=dna_client.OutputType.DNASE,
                      width=501,
                      aggregation_type=(
                          interval_scorers.IntervalAggregationType.SUM
                      ),
                  ),
              ],
          ),
      ),
      expected=[
          (
              anndata.AnnData(
                  X=np.zeros((5, 10)),
                  var=pd.DataFrame({
                      'name': [f'{i}' for i in range(10)],
                      'strand': ['.'] * 10,
                      'biosample_curie': ['UBERON:0000005'] * 10,
                      'imputation_mode': ['imputed'] * 10,
                      'biosample_type': ['tissue'] * 10,
                      'biosample_name': ['sample'] * 10,
                  }),
                  uns={
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'interval_scorer': interval_scorers.GeneMaskScorer(
                          requested_output=dna_client.OutputType.ATAC,
                          width=2_001,
                          aggregation_type=(
                              interval_scorers.IntervalAggregationType.MEAN
                          ),
                      ),
                  },
              ),
              anndata.AnnData(
                  X=np.ones((5, 10)),
                  obs=pd.DataFrame({
                      'gene_id': [f'{i}' for i in range(5)],
                      'strand': ['-'] * 5,
                      'gene_name': [f'gene_{i}' for i in range(5)],
                      'gene_type': [f'gene_type_{i}' for i in range(5)],
                  }),
                  var=pd.DataFrame({
                      'name': [f'{i}' for i in range(10)],
                      'strand': ['.'] * 10,
                      'biosample_curie': ['UBERON:0000005'] * 10,
                      'imputation_mode': ['imputed'] * 10,
                      'biosample_type': ['tissue'] * 10,
                      'biosample_name': ['sample'] * 10,
                  }),
                  uns={
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'interval_scorer': interval_scorers.GeneMaskScorer(
                          requested_output=dna_client.OutputType.DNASE,
                          width=501,
                          aggregation_type=(
                              interval_scorers.IntervalAggregationType.SUM
                          ),
                      ),
                  },
              ),
          ),
      ],
      organism=[
          dna_client.Organism.HOMO_SAPIENS,
          dna_client.Organism.MUS_MUSCULUS,
      ],
  )
  def test_score_interval(
      self,
      interval,
      scorers,
      expected,
      organism,
  ):

    def _mock_generate(request, metadata):
      del metadata
      return _generate_interval_scoring_proto(
          scores=expected,
          response_proto_type=dna_model_service_pb2.ScoreIntervalResponse,
          interval=genome.Interval.from_proto(request.interval),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    outputs = model.score_interval(
        interval=interval,
        condition=model_condition.from_proto(
            dna_model_pb2_v2.Condition(uberon=4)
        ),
        interval_scorers=scorers,
        organism=organism,
    )
    for expected_anndata, output_anndata in zip(expected, outputs):
      self._assert_anndata_equal(expected_anndata, output_anndata)

    mock_stub.assert_called_once()

  @parameterized.product(organism=list(dna_client.Organism))
  def test_recommended_interval_scorers(self, organism):
    def _mock_generate(request, metadata):
      del metadata
      return _generate_interval_scoring_proto(
          scores=[
              anndata.AnnData(
                  np.zeros((0, 0)),
                  var=pd.DataFrame(
                      columns=[
                          'name',
                          'strand',
                          'biosample_curie',
                          'imputation_mode',
                          'biosample_type',
                          'biosample_name',
                      ]
                  ),
              )
              for _ in range(len(request.interval_scorers))
          ],
          response_proto_type=dna_model_service_pb2.ScoreIntervalResponse,
          interval=genome.Interval.from_proto(request.interval),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    scores = model.score_interval(
        interval=genome.Interval('chr1', 0, 2**20),
        condition=model_condition.from_proto(
            dna_model_pb2_v2.Condition(uberon=4)
        ),
        organism=organism,
    )

    self.assertSameElements(
        [score.uns['interval_scorer'] for score in scores],
        interval_scorers.RECOMMENDED_INTERVAL_SCORERS.values(),
    )
    mock_stub.assert_called_once()

  def test_duplicate_interval_scorers_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError, 'Duplicate interval scorers requested'
    ):
      model.score_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          condition=model_condition.from_proto(
              dna_model_pb2_v2.Condition(uberon=4)
          ),
          interval_scorers=[
              interval_scorers.GeneMaskScorer(
                  requested_output=dna_client.OutputType.DNASE,
                  width=501,
                  aggregation_type=(
                      interval_scorers.IntervalAggregationType.SUM
                  ),
              )
          ]
          * 2,
      )

  def test_invalid_width_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(ValueError, 'must have widths <= interval'):
      model.score_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          condition=model_condition.from_proto(
              dna_model_pb2_v2.Condition(uberon=4)
          ),
          interval_scorers=[
              interval_scorers.GeneMaskScorer(
                  requested_output=dna_client.OutputType.DNASE,
                  width=100_001,
                  aggregation_type=(
                      interval_scorers.IntervalAggregationType.SUM
                  ),
              )
          ],
      )

  def test_exceed_max_interval_scorers_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError, 'Too many interval scorers requested'
    ):
      scorers = []
      for output_type in interval_scorers.SUPPORTED_OUTPUT_TYPES[
          interval_scorers.BaseIntervalScorer.GENE_MASK
      ]:
        for width in interval_scorers.SUPPORTED_WIDTHS[
            interval_scorers.BaseIntervalScorer.GENE_MASK
        ]:
          scorers.append(
              interval_scorers.GeneMaskScorer(
                  requested_output=output_type,
                  width=width,
                  aggregation_type=(
                      interval_scorers.IntervalAggregationType.SUM
                  ),
              )
          )
      model.score_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          condition=model_condition.from_proto(
              dna_model_pb2_v2.Condition(uberon=4)
          ),
          interval_scorers=scorers,
      )

  @parameterized.product(
      (
          dict(
              interval=genome.Interval('chr1', 0, 16_384),
              scorers=[
                  interval_scorers.GeneMaskScorer(
                      requested_output=dna_client.OutputType.ATAC,
                      width=2_001,
                      aggregation_type=(
                          interval_scorers.IntervalAggregationType.MEAN
                      ),
                  ),
                  interval_scorers.GeneMaskScorer(
                      requested_output=dna_client.OutputType.DNASE,
                      width=501,
                      aggregation_type=(
                          interval_scorers.IntervalAggregationType.SUM
                      ),
                  ),
              ],
          ),
      ),
      expected=[
          (
              anndata.AnnData(
                  X=np.zeros((5, 10)),
                  var=pd.DataFrame({
                      'name': [f'{i}' for i in range(10)],
                      'strand': ['.'] * 10,
                      'biosample_curie': ['UBERON:0000005'] * 10,
                      'imputation_mode': ['imputed'] * 10,
                      'biosample_type': ['tissue'] * 10,
                      'biosample_name': ['sample'] * 10,
                  }),
                  uns={
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'interval_scorer': interval_scorers.GeneMaskScorer(
                          requested_output=dna_client.OutputType.ATAC,
                          width=2_001,
                          aggregation_type=(
                              interval_scorers.IntervalAggregationType.MEAN
                          ),
                      ),
                  },
              ),
              anndata.AnnData(
                  X=np.ones((5, 10)),
                  obs=pd.DataFrame({
                      'gene_id': [f'{i}' for i in range(5)],
                      'strand': ['-'] * 5,
                      'gene_name': [f'gene_{i}' for i in range(5)],
                      'gene_type': [f'gene_type_{i}' for i in range(5)],
                  }),
                  var=pd.DataFrame({
                      'name': [f'{i}' for i in range(10)],
                      'strand': ['.'] * 10,
                      'biosample_curie': ['UBERON:0000005'] * 10,
                      'imputation_mode': ['imputed'] * 10,
                      'biosample_type': ['tissue'] * 10,
                      'biosample_name': ['sample'] * 10,
                  }),
                  uns={
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'interval_scorer': interval_scorers.GeneMaskScorer(
                          requested_output=dna_client.OutputType.DNASE,
                          width=501,
                          aggregation_type=(
                              interval_scorers.IntervalAggregationType.SUM
                          ),
                      ),
                  },
              ),
          ),
      ],
      organism=[
          dna_client.Organism.HOMO_SAPIENS,
          dna_client.Organism.MUS_MUSCULUS,
      ],
  )
  def test_score_intervals(
      self,
      interval,
      scorers,
      expected,
      organism,
  ):
    def _mock_generate(request, metadata):
      del metadata
      return _generate_interval_scoring_proto(
          scores=expected,
          response_proto_type=dna_model_service_pb2.ScoreIntervalResponse,
          interval=genome.Interval.from_proto(request.interval),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    all_outputs = model.score_intervals(
        intervals=[interval, interval],
        conditions=[
            model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
        ]
        * 2,
        organism=organism,
        interval_scorers=scorers,
    )

    for outputs in all_outputs:
      for expected_anndata, output_anndata in zip(expected, outputs):
        self._assert_anndata_equal(expected_anndata, output_anndata)

    self.assertEqual(mock_stub.call_count, 2)

  @parameterized.product(
      (
          dict(
              interval=genome.Interval('chr1', 0, 16_384),
              variant=genome.Variant('chr1', 1000, 'A', 'C'),
              scorers=[
                  variant_scorers.CenterMaskScorer(
                      requested_output=dna_client.OutputType.ATAC,
                      width=10_001,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
                  variant_scorers.GeneMaskLFCScorer(
                      requested_output=dna_client.OutputType.RNA_SEQ,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
              ],
          ),
      ),
      expected=[
          (
              anndata.AnnData(
                  X=np.zeros((5, 10)),
                  var=pd.DataFrame({
                      'name': [f'{i}' for i in range(10)],
                      'strand': ['.'] * 10,
                      'biosample_curie': ['UBERON:0000005'] * 10,
                      'imputation_mode': ['imputed'] * 10,
                      'biosample_type': ['tissue'] * 10,
                      'biosample_name': ['sample'] * 10,
                  }),
                  uns={
                      'variant': genome.Variant('chr1', 1000, 'A', 'C'),
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'variant_scorer': variant_scorers.CenterMaskScorer(
                          requested_output=dna_client.OutputType.ATAC,
                          width=10_001,
                          aggregation_type=(
                              variant_scorers.AggregationType.DIFF_MEAN
                          ),
                      ),
                  },
              ),
              anndata.AnnData(
                  X=np.ones((5, 10)),
                  obs=pd.DataFrame({
                      'gene_id': [f'{i}' for i in range(5)],
                      'strand': ['-'] * 5,
                      'gene_name': [f'gene_{i}' for i in range(5)],
                      'gene_type': [f'gene_type_{i}' for i in range(5)],
                  }),
                  var=pd.DataFrame({
                      'name': [f'{i}' for i in range(10)],
                      'strand': ['.'] * 10,
                      'biosample_curie': ['UBERON:0000005'] * 10,
                      'imputation_mode': ['imputed'] * 10,
                      'biosample_type': ['tissue'] * 10,
                      'biosample_name': ['sample'] * 10,
                  }),
                  uns={
                      'variant': genome.Variant('chr1', 1000, 'A', 'C'),
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'variant_scorer': variant_scorers.GeneMaskLFCScorer(
                          requested_output=dna_client.OutputType.RNA_SEQ,
                          aggregation_type=(
                              variant_scorers.AggregationType.DIFF_MEAN
                          ),
                      ),
                  },
                  layers={'quantiles': np.zeros((5, 10))},
              ),
          ),
          (
              anndata.AnnData(
                  X=np.zeros((5, 10)),
                  var=pd.DataFrame({
                      'name': [f'{i}' for i in range(10)],
                      'strand': ['.'] * 10,
                      'biosample_curie': ['UBERON:0000005'] * 10,
                      'imputation_mode': ['imputed'] * 10,
                      'biosample_type': ['tissue'] * 10,
                      'biosample_name': ['sample'] * 10,
                  }),
                  uns={
                      'variant': genome.Variant('chr1', 1000, 'A', 'C'),
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'variant_scorer': variant_scorers.CenterMaskScorer(
                          requested_output=dna_client.OutputType.ATAC,
                          width=10_001,
                          aggregation_type=(
                              variant_scorers.AggregationType.DIFF_MEAN
                          ),
                      ),
                  },
              ),
              anndata.AnnData(
                  X=np.ones((5, 10)),
                  var=pd.DataFrame({
                      'name': [f'{i}' for i in range(10)],
                      'strand': ['.'] * 10,
                      'biosample_curie': ['UBERON:0000005'] * 10,
                      'imputation_mode': ['imputed'] * 10,
                      'biosample_type': ['tissue'] * 10,
                      'biosample_name': ['sample'] * 10,
                  }),
                  uns={
                      'variant': genome.Variant('chr1', 1000, 'A', 'C'),
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'variant_scorer': variant_scorers.GeneMaskLFCScorer(
                          requested_output=dna_client.OutputType.RNA_SEQ,
                          aggregation_type=(
                              variant_scorers.AggregationType.DIFF_MEAN
                          ),
                      ),
                  },
              ),
          ),
      ],
      organism=[
          dna_client.Organism.HOMO_SAPIENS,
          dna_client.Organism.MUS_MUSCULUS,
      ],
  )
  def test_score_variant(
      self,
      interval,
      variant,
      scorers,
      expected,
      organism,
  ):

    def _mock_generate(request, metadata):
      del metadata
      return _generate_variant_scoring_proto(
          scores=expected,
          response_proto_type=dna_model_service_pb2.ScoreVariantResponse,
          variant=genome.Variant.from_proto(request.variant),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    outputs = model.score_variant(
        interval=interval,
        variant=variant,
        condition=model_condition.from_proto(
            dna_model_pb2_v2.Condition(uberon=4)
        ),
        organism=organism,
        variant_scorers=scorers,
    )

    for expected_anndata, output_anndata in zip(expected, outputs):
      self._assert_anndata_equal(expected_anndata, output_anndata)

    mock_stub.assert_called_once()

  def test_duplicate_variant_scorers_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError, 'Duplicate variant scorers requested'
    ):
      model.score_variant(
          interval=genome.Interval('chr1', 0, 16_384),
          variant=genome.Variant('chr1', 1000, 'A', 'C'),
          condition=model_condition.from_proto(
              dna_model_pb2_v2.Condition(uberon=4)
          ),
          variant_scorers=[
              variant_scorers.GeneMaskLFCScorer(
                  requested_output=dna_client.OutputType.RNA_SEQ,
                  aggregation_type=(variant_scorers.AggregationType.DIFF_MEAN),
              ),
              variant_scorers.GeneMaskLFCScorer(
                  requested_output=dna_client.OutputType.RNA_SEQ,
                  aggregation_type=(variant_scorers.AggregationType.DIFF_MEAN),
              ),
          ],
      )

  def test_invalid_organism_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Unsupported organism: MUS_MUSCULUS for scorer PolyadenylationScorer().'
        " Supported organisms: ['HOMO_SAPIENS']",
    ):
      model.score_variant(
          interval=genome.Interval('chr1', 0, 16_384),
          variant=genome.Variant('chr1', 1000, 'A', 'C'),
          condition=model_condition.from_proto(
              dna_model_pb2_v2.Condition(uberon=4)
          ),
          variant_scorers=[variant_scorers.PolyadenylationScorer()],
          # The PaQTL scorer doesn't currently support mouse.
          organism=dna_client.Organism.MUS_MUSCULUS,
      )

  def test_exceed_max_variant_scorers_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError, 'Too many variant scorers requested'
    ):
      variant_scorers_to_use = []
      for output_type in variant_scorers.SUPPORTED_OUTPUT_TYPES[
          variant_scorers.BaseVariantScorer.CENTER_MASK
      ]:
        for width in variant_scorers.SUPPORTED_WIDTHS[
            variant_scorers.BaseVariantScorer.CENTER_MASK
        ]:
          variant_scorers_to_use.append(
              variant_scorers.CenterMaskScorer(
                  requested_output=output_type,
                  width=width,
                  aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
              )
          )
      model.score_variant(
          interval=genome.Interval('chr1', 0, 16_384),
          variant=genome.Variant('chr1', 1000, 'A', 'C'),
          condition=model_condition.from_proto(
              dna_model_pb2_v2.Condition(uberon=4)
          ),
          variant_scorers=variant_scorers_to_use,
      )

  @parameterized.product(organism=list(dna_client.Organism))
  def test_recommended_variant_scorers(self, organism):
    def _mock_generate(request, metadata):
      del metadata
      return _generate_variant_scoring_proto(
          scores=[
              anndata.AnnData(
                  np.zeros((0, 0)),
                  var=pd.DataFrame(
                      columns=[
                          'name',
                          'strand',
                          'biosample_curie',
                          'imputation_mode',
                          'biosample_type',
                          'biosample_name',
                      ]
                  ),
              )
              for _ in range(len(request.variant_scorers))
          ],
          response_proto_type=dna_model_service_pb2.ScoreVariantResponse,
          variant=genome.Variant.from_proto(request.variant),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    scores = model.score_variant(
        interval=genome.Interval('chr1', 0, 16_384),
        variant=genome.Variant('chr1', 1000, 'A', 'C'),
        condition=model_condition.from_proto(
            dna_model_pb2_v2.Condition(uberon=4)
        ),
        organism=organism,
    )

    # The server rejects SC_GENE_EXPRESSION_DENOISED scorers without
    # cell_conditioning, and the PolyadenylationScorer is human-only.
    excluded = {'SC_GENE_EXPRESSION_DENOISED'}
    if organism != dna_client.Organism.HOMO_SAPIENS:
      excluded.add('POLYADENYLATION')
    expected = [
        scorer
        for name, scorer in variant_scorers.RECOMMENDED_VARIANT_SCORERS.items()
        if name not in excluded
    ]
    self.assertSameElements(
        [score.uns['variant_scorer'] for score in scores], expected
    )
    mock_stub.assert_called_once()

  def test_recommended_variant_scorers_cell_conditioned(self):
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )

    def _mock_generate(request, metadata):
      del metadata
      return _generate_variant_scoring_proto(
          scores=[
              anndata.AnnData(
                  np.zeros((0, 0)),
                  var=pd.DataFrame(
                      columns=[
                          'name',
                          'strand',
                          'biosample_curie',
                          'imputation_mode',
                          'biosample_type',
                          'biosample_name',
                      ]
                  ),
              )
              for _ in range(len(request.variant_scorers))
          ],
          response_proto_type=dna_model_service_pb2.ScoreVariantResponse,
          variant=genome.Variant.from_proto(request.variant),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    scores = model.score_variant(
        interval=genome.Interval('chr1', 0, 16_384),
        variant=genome.Variant('chr1', 1000, 'A', 'C'),
        cell_conditioning=cell_conditioning_adata,
        cell_condition_name='cell_0',
        organism=dna_client.Organism.HOMO_SAPIENS,
    )

    self.assertSameElements(
        [score.uns['variant_scorer'] for score in scores],
        list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.values()),
    )
    mock_stub.assert_called_once()

  @parameterized.product(
      (
          dict(
              interval=genome.Interval('chr1', 0, 16_384),
              scorers=[
                  variant_scorers.CenterMaskScorer(
                      requested_output=dna_client.OutputType.ATAC,
                      width=10_001,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
                  variant_scorers.GeneMaskLFCScorer(
                      requested_output=dna_client.OutputType.RNA_SEQ,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
                  variant_scorers.SpliceJunctionScorer(),
              ],
          ),
      ),
      expected=[(
          anndata.AnnData(
              X=np.ones((5, 10)),
              obs=pd.DataFrame({'gene_id': [f'{i}' for i in range(5)]}),
              var=pd.DataFrame({
                  'name': [f'{i}' for i in range(10)],
                  'strand': ['.'] * 10,
                  'biosample_curie': ['UBERON:0000005'] * 10,
                  'imputation_mode': ['imputed'] * 10,
                  'biosample_type': ['tissue'] * 10,
                  'biosample_name': ['sample'] * 10,
              }),
              uns={
                  'interval': genome.Interval('chr1', 0, 16_384),
                  'variant_scorer': variant_scorers.CenterMaskScorer(
                      requested_output=dna_client.OutputType.ATAC,
                      width=10_001,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
              },
          ),
          anndata.AnnData(
              X=np.ones((5, 10)),
              obs=pd.DataFrame({'gene_id': [f'{i}' for i in range(5)]}),
              var=pd.DataFrame({
                  'name': [f'{i}' for i in range(10)],
                  'strand': ['.'] * 10,
                  'biosample_curie': ['UBERON:0000005'] * 10,
                  'imputation_mode': ['imputed'] * 10,
                  'biosample_type': ['tissue'] * 10,
                  'biosample_name': ['sample'] * 10,
              }),
              uns={
                  'interval': genome.Interval('chr1', 0, 16_384),
                  'variant_scorer': variant_scorers.GeneMaskLFCScorer(
                      requested_output=dna_client.OutputType.RNA_SEQ,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
              },
          ),
          anndata.AnnData(
              X=np.ones((5, 10)),
              obs=pd.DataFrame({
                  'gene_id': [f'{i}' for i in range(5)],
                  'junction_Start': list(range(5)),
                  'junction_End': list(range(1, 6)),
              }),
              var=pd.DataFrame({
                  'name': [f'{i}' for i in range(10)],
                  'strand': ['.'] * 10,
                  'biosample_curie': ['UBERON:0000005'] * 10,
                  'imputation_mode': ['imputed'] * 10,
                  'biosample_type': ['tissue'] * 10,
                  'biosample_name': ['sample'] * 10,
              }),
              uns={
                  'interval': genome.Interval('chr1', 0, 16_384),
                  'variant_scorer': variant_scorers.SpliceJunctionScorer(),
              },
          ),
      )],
      variants=[
          [
              genome.Variant('chr1', 100, 'G', 'T'),
          ],
          [
              genome.Variant('chr1', 100, 'G', 'T'),
              genome.Variant('chr1', 1000, 'A', 'C'),
              genome.Variant('chr1', 1001, 'A', 'C'),
              genome.Variant('chr1', 1002, 'A', 'C'),
          ],
      ],
      organism=[
          dna_client.Organism.HOMO_SAPIENS,
          dna_client.Organism.MUS_MUSCULUS,
      ],
  )
  def test_score_variants(
      self,
      interval: genome.Interval,
      variants: list[genome.Variant],
      scorers: list[
          dna_client.VariantScorerTypes  # pyrefly: ignore[missing-attribute]
      ],
      expected: list[anndata.AnnData],
      organism: dna_client.Organism,
  ):
    position_to_expected_scores = {}
    for variant in variants:
      scores = []
      for score in expected:
        score = score.copy()
        score.uns['variant'] = variant
        score.X *= variant.position  # pyrefly: ignore[unsupported-operation]
        scores.append(score)
      position_to_expected_scores[variant.position] = scores

    def _mock_generate(request, metadata):
      del metadata
      return _generate_variant_scoring_proto(
          scores=position_to_expected_scores[request.variant.position],
          response_proto_type=dna_model_service_pb2.ScoreVariantResponse,
          variant=genome.Variant.from_proto(request.variant),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    all_scores = model.score_variants(
        intervals=[interval] * len(variants),
        conditions=[
            model_condition.from_proto(dna_model_pb2_v2.Condition(uberon=4))
        ]
        * len(variants),
        variants=variants,
        organism=organism,
        variant_scorers=scorers,
    )

    for scores in all_scores:
      self.assertLen(scores, len(expected))
      expected = position_to_expected_scores[scores[0].uns['variant'].position]
      for expected_anndata, output_anndata in zip(expected, scores):
        self._assert_anndata_equal(output_anndata, expected_anndata)

    self.assertLen(variants, mock_stub.call_count)

  @parameterized.product(
      (
          dict(
              interval=genome.Interval('chr1', 0, 16_384),
              ism_interval=genome.Interval('chr1', 10, 11),
              interval_variant=None,
              scorers=[
                  variant_scorers.CenterMaskScorer(
                      requested_output=dna_client.OutputType.ATAC,
                      width=501,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
                  variant_scorers.PolyadenylationScorer(),
              ],
              example_scores=[
                  anndata.AnnData(
                      X=np.zeros((5, 10)),
                      obs=pd.DataFrame({'gene_id': [f'{i}' for i in range(5)]}),
                      var=pd.DataFrame({
                          'name': [f'{i}' for i in range(10)],
                          'strand': ['.'] * 10,
                          'biosample_curie': ['UBERON:0000005'] * 10,
                          'imputation_mode': ['imputed'] * 10,
                          'biosample_type': ['tissue'] * 10,
                          'biosample_name': ['sample'] * 10,
                      }),
                      uns={
                          'interval': genome.Interval('chr1', 0, 16_384),
                          'variant_scorer': variant_scorers.CenterMaskScorer(
                              requested_output=dna_client.OutputType.ATAC,
                              width=10_001,
                              aggregation_type=(
                                  variant_scorers.AggregationType.DIFF_MEAN
                              ),
                          ),
                      },
                  ),
              ],
              expected_variants=[
                  genome.Variant('chr1', 11, 'A', 'C'),
                  genome.Variant('chr1', 11, 'A', 'G'),
                  genome.Variant('chr1', 11, 'A', 'T'),
              ],
          ),
          dict(
              interval=genome.Interval('chr1', 0, 16_384),
              ism_interval=genome.Interval('chr1', 10, 11),
              interval_variant=genome.Variant('chr1', 11, 'A', 'C'),
              scorers=[
                  variant_scorers.CenterMaskScorer(
                      requested_output=dna_client.OutputType.ATAC,
                      width=501,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
              ],
              example_scores=[
                  anndata.AnnData(
                      X=np.zeros((5, 10)),
                      obs=pd.DataFrame({'gene_id': [f'{i}' for i in range(5)]}),
                      var=pd.DataFrame({
                          'name': [f'{i}' for i in range(10)],
                          'strand': ['.'] * 10,
                          'biosample_curie': ['UBERON:0000005'] * 10,
                          'imputation_mode': ['imputed'] * 10,
                          'biosample_type': ['tissue'] * 10,
                          'biosample_name': ['sample'] * 10,
                      }),
                      uns={
                          'interval': genome.Interval('chr1', 0, 16_384),
                          'variant_scorer': variant_scorers.CenterMaskScorer(
                              requested_output=dna_client.OutputType.ATAC,
                              width=10_001,
                              aggregation_type=(
                                  variant_scorers.AggregationType.DIFF_MEAN
                              ),
                          ),
                      },
                  ),
              ],
              expected_variants=[
                  genome.Variant('chr1', 11, 'C', 'A'),
                  genome.Variant('chr1', 11, 'C', 'G'),
                  genome.Variant('chr1', 11, 'C', 'T'),
              ],
          ),
          dict(
              interval=genome.Interval('chr1', 0, 16_384),
              ism_interval=genome.Interval('chr1', 10, 21),
              interval_variant=None,
              scorers=[
                  variant_scorers.CenterMaskScorer(
                      requested_output=dna_client.OutputType.ATAC,
                      width=501,
                      aggregation_type=(
                          variant_scorers.AggregationType.DIFF_MEAN
                      ),
                  ),
                  variant_scorers.PolyadenylationScorer(),
              ],
              example_scores=[
                  anndata.AnnData(
                      X=np.zeros((5, 10)),
                      obs=pd.DataFrame({'gene_id': [f'{i}' for i in range(5)]}),
                      var=pd.DataFrame({
                          'name': [f'{i}' for i in range(10)],
                          'strand': ['.'] * 10,
                          'biosample_curie': ['UBERON:0000005'] * 10,
                          'imputation_mode': ['imputed'] * 10,
                          'biosample_type': ['tissue'] * 10,
                          'biosample_name': ['sample'] * 10,
                      }),
                      uns={
                          'interval': genome.Interval('chr1', 0, 16_384),
                          'variant_scorer': variant_scorers.CenterMaskScorer(
                              requested_output=dna_client.OutputType.ATAC,
                              width=10_001,
                              aggregation_type=(
                                  variant_scorers.AggregationType.DIFF_MEAN
                              ),
                          ),
                      },
                  ),
                  anndata.AnnData(
                      X=np.ones((5, 10)),
                      obs=pd.DataFrame({'gene_id': [f'{i}' for i in range(5)]}),
                      var=pd.DataFrame({
                          'name': [f'{i}' for i in range(10)],
                          'strand': ['.'] * 10,
                          'biosample_curie': ['UBERON:0000005'] * 10,
                          'imputation_mode': ['imputed'] * 10,
                          'biosample_type': ['tissue'] * 10,
                          'biosample_name': ['sample'] * 10,
                      }),
                      uns={
                          'interval': genome.Interval('chr1', 0, 16_384),
                          'variant_scorer': (
                              variant_scorers.PolyadenylationScorer()
                          ),
                      },
                      layers={'quantiles': np.zeros((5, 10))},
                  ),
              ],
              expected_variants=[
                  genome.Variant('chr1', 11, 'A', 'C'),
                  genome.Variant('chr1', 11, 'A', 'G'),
                  genome.Variant('chr1', 11, 'A', 'T'),
                  genome.Variant('chr1', 12, 'A', 'C'),
                  genome.Variant('chr1', 12, 'A', 'G'),
                  genome.Variant('chr1', 12, 'A', 'T'),
                  genome.Variant('chr1', 13, 'A', 'C'),
                  genome.Variant('chr1', 13, 'A', 'G'),
                  genome.Variant('chr1', 13, 'A', 'T'),
                  genome.Variant('chr1', 14, 'A', 'C'),
                  genome.Variant('chr1', 14, 'A', 'G'),
                  genome.Variant('chr1', 14, 'A', 'T'),
                  genome.Variant('chr1', 15, 'A', 'C'),
                  genome.Variant('chr1', 15, 'A', 'G'),
                  genome.Variant('chr1', 15, 'A', 'T'),
                  genome.Variant('chr1', 16, 'A', 'C'),
                  genome.Variant('chr1', 16, 'A', 'G'),
                  genome.Variant('chr1', 16, 'A', 'T'),
                  genome.Variant('chr1', 17, 'A', 'C'),
                  genome.Variant('chr1', 17, 'A', 'G'),
                  genome.Variant('chr1', 17, 'A', 'T'),
                  genome.Variant('chr1', 18, 'A', 'C'),
                  genome.Variant('chr1', 18, 'A', 'G'),
                  genome.Variant('chr1', 18, 'A', 'T'),
                  genome.Variant('chr1', 19, 'A', 'C'),
                  genome.Variant('chr1', 19, 'A', 'G'),
                  genome.Variant('chr1', 19, 'A', 'T'),
                  genome.Variant('chr1', 20, 'A', 'C'),
                  genome.Variant('chr1', 20, 'A', 'G'),
                  genome.Variant('chr1', 20, 'A', 'T'),
                  genome.Variant('chr1', 21, 'A', 'C'),
                  genome.Variant('chr1', 21, 'A', 'G'),
                  genome.Variant('chr1', 21, 'A', 'T'),
              ],
          ),
      ),
  )
  def test_score_ism_variant(
      self,
      interval,
      ism_interval,
      scorers,
      example_scores,
      expected_variants,
      interval_variant: genome.Variant | None,
  ):
    received_requests = []

    def _mock_generate(request, metadata):
      del metadata
      received_requests.append(request)
      ism_interval = genome.Interval.from_proto(request.ism_interval)
      ism_sequence = 'A' * ism_interval.width
      if request.HasField('interval_variant'):
        ref_variant = genome.Variant.from_proto(request.interval_variant)
        self.assertTrue(ref_variant.is_snv)  # Only SNVs supported in test.
        if ref_variant.reference_overlaps(ism_interval):
          pos = ref_variant.start - ism_interval.start
          ism_sequence = (
              ism_sequence[:pos]
              + ref_variant.alternate_bases
              + ism_sequence[pos + 1 :]
          )
      variants = ism.ism_variants(ism_interval, ism_sequence)

      all_outputs = []
      for variant in variants:
        response = _generate_variant_scoring_proto(
            scores=example_scores,
            response_proto_type=dna_model_service_pb2.ScoreIsmVariantResponse,
            variant=variant,
        )
        all_outputs.extend(response.outputs)

      return dna_model_service_pb2.ScoreIsmVariantResponse(outputs=all_outputs)

    mock_channel, _ = _create_mock_channel_and_stub(_mock_generate)

    model = dna_client.DnaClient(channel=mock_channel)
    ism_scores = model.score_ism_variants(
        interval=interval,
        ism_interval=ism_interval,
        condition=model_condition.from_proto(
            dna_model_pb2_v2.Condition(uberon=4)
        ),
        variant_scorers=scorers,
        interval_variant=interval_variant,
    )
    self.assertLen(ism_scores, len(expected_variants))
    ism_scores = sorted(ism_scores, key=lambda x: str(x[0].uns['variant']))
    for expected_variant, outputs in zip(
        expected_variants, ism_scores, strict=True
    ):
      for output, example_output in zip(outputs, example_scores, strict=True):
        expected = example_output.copy()
        expected.uns['variant'] = expected_variant
        self._assert_anndata_equal(expected, output)

    expected_requests = []
    for position in range(
        0, ism_interval.width, dna_client.MAX_ISM_INTERVAL_WIDTH
    ):
      expected_requests.append(
          dna_model_service_pb2.ScoreIsmVariantRequest(
              interval=interval.to_proto(),
              ism_interval=genome.Interval(
                  ism_interval.chromosome,
                  ism_interval.start + position,
                  min(
                      ism_interval.start
                      + position
                      + dna_client.MAX_ISM_INTERVAL_WIDTH,
                      ism_interval.end,
                  ),
              ).to_proto(),
              condition=dna_model_pb2_v2.Condition(uberon=4),
              variant_scorers=[scorer.to_proto() for scorer in scorers],
              organism=dna_client.Organism.HOMO_SAPIENS.to_proto(),
              interval_variant=interval_variant.to_proto()
              if interval_variant
              else None,
          )
      )

    self.assertCountEqual(expected_requests, received_requests)

  def test_score_negative_strand_ism_interval_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError, 'ISM interval must be on the positive strand.'
    ):
      model.score_ism_variants(
          interval=genome.Interval('chr1', 0, 16_384),
          ism_interval=genome.Interval(
              'chr1', 10, 11, strand=genome.STRAND_NEGATIVE
          ),
          condition=model_condition.from_proto(
              dna_model_pb2_v2.Condition(uberon=4)
          ),
          variant_scorers=[],
      )

  def test_create(self):
    with (
        mock.patch.object(grpc, 'secure_channel') as mock_secure_channel,
        mock.patch.object(grpc, 'channel_ready_future'),
    ):
      _ = dna_client.create('foo')
      mock_secure_channel.assert_called_once_with(
          'dns:///gdmscience.googleapis.com:443',
          mock.ANY,
          options=mock.ANY,
      )

  def test_predict_interval_with_cell_conditioning(self):
    # Construct an AnnData input
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    # Expected converted proto for assertion
    expected_cell_conditioning = dna_model_pb2_v2.CellConditioningInput(
        gene_ids=['ENSG00000000003'],
    )
    cell_proto = expected_cell_conditioning.cells.add()
    cell_proto.gene_indices.append(0)
    cell_proto.gene_counts.append(5)

    mock_predictions = _create_example_model_predictions(
        interval=genome.Interval('chr1', 0, 16_384)
    )

    def _mock_generate(request, metadata):
      del metadata
      self.assertEqual(request.cell_conditioning, expected_cell_conditioning)
      self.assertFalse(request.HasField('condition'))
      return _generate_prediction_proto(
          mock_predictions,
          request.requested_outputs,
          dna_model_service_pb2.PredictIntervalResponse,
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    output = model.predict_interval(
        interval=genome.Interval('chr1', 0, 16_384),
        requested_outputs=[dna_client.OutputType.ATAC],
        cell_conditioning=cell_conditioning_adata,
        cell_condition_name='cell_0',
    )
    mock_stub.assert_called_once()
    assert output.atac is not None
    for name in output.atac.names:
      self.assertTrue(name.endswith('_cell_0'))

  def test_predict_mismatch_conditioning_and_condition(
      self,
  ):
    cell_conditioning = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    model = dna_client.DnaClient(channel=mock.Mock())

    with self.assertRaisesRegex(
        ValueError, 'Cannot specify both `condition` and `cell_conditioning`.'
    ):
      model.predict_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          requested_outputs=[dna_client.OutputType.ATAC],
          condition='foo',
          cell_conditioning=cell_conditioning,
          cell_condition_name='cell_0',
      )

    with self.assertRaisesRegex(
        ValueError, 'Must specify either `condition` or `cell_conditioning`.'
    ):
      model.predict_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          requested_outputs=[dna_client.OutputType.ATAC],
      )

  def test_predict_sequence_with_cell_conditioning(self):
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    expected_cell_conditioning = dna_model_pb2_v2.CellConditioningInput(
        gene_ids=['ENSG00000000003'],
    )
    cell_proto = expected_cell_conditioning.cells.add()
    cell_proto.gene_indices.append(0)
    cell_proto.gene_counts.append(5)

    mock_predictions = _create_example_model_predictions(sequence_length=16_384)

    def _mock_generate(request, metadata):
      del metadata
      self.assertEqual(request.cell_conditioning, expected_cell_conditioning)
      self.assertFalse(request.HasField('condition'))
      return _generate_prediction_proto(
          mock_predictions,
          request.requested_outputs,
          dna_model_service_pb2.PredictSequenceResponse,
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    output = model.predict_sequence(
        'A' * 16_384,
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs={dna_client.OutputType.ATAC},
        cell_conditioning=cell_conditioning_adata,
        cell_condition_name='cell_0',
    )
    mock_stub.assert_called_once()
    assert output.atac is not None
    self.assertIn('biosample_name', output.atac.metadata.columns)
    self.assertTrue((output.atac.metadata['biosample_name'] == 'cell_0').all())
    for name in output.atac.names:
      self.assertTrue(name.endswith('_cell_0'))

  def test_predict_variant_with_cell_conditioning(self):
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    expected_cell_conditioning = dna_model_pb2_v2.CellConditioningInput(
        gene_ids=['ENSG00000000003'],
    )
    cell_proto = expected_cell_conditioning.cells.add()
    cell_proto.gene_indices.append(0)
    cell_proto.gene_counts.append(5)

    interval = genome.Interval('chr1', 0, 16_384)
    mock_predictions = dna_client.VariantOutput(
        reference=_create_example_model_predictions(interval=interval, value=0),
        alternate=_create_example_model_predictions(interval=interval, value=1),
    )

    def _mock_generate(request, metadata):
      del metadata
      self.assertEqual(request.cell_conditioning, expected_cell_conditioning)
      self.assertFalse(request.HasField('condition'))
      return _generate_variant_proto(
          mock_predictions, request.requested_outputs
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    variant_output = model.predict_variant(
        interval=interval,
        variant=genome.Variant('chr1', 1000, 'A', 'C'),
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs={dna_client.OutputType.ATAC},
        cell_conditioning=cell_conditioning_adata,
        cell_condition_name='cell_0',
    )
    mock_stub.assert_called_once()
    assert variant_output.reference.atac is not None
    assert variant_output.alternate.atac is not None
    self.assertIn(
        'biosample_name', variant_output.reference.atac.metadata.columns
    )
    self.assertTrue(
        (
            variant_output.reference.atac.metadata['biosample_name'] == 'cell_0'
        ).all()
    )
    self.assertIn(
        'biosample_name', variant_output.alternate.atac.metadata.columns
    )
    self.assertTrue(
        (
            variant_output.alternate.atac.metadata['biosample_name'] == 'cell_0'
        ).all()
    )
    for name in variant_output.reference.atac.names:
      self.assertTrue(name.endswith('_cell_0'))
    for name in variant_output.alternate.atac.names:
      self.assertTrue(name.endswith('_cell_0'))

  def test_score_interval_with_cell_conditioning(self):
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    expected_cell_conditioning = dna_model_pb2_v2.CellConditioningInput(
        gene_ids=['ENSG00000000003'],
    )
    cell_proto = expected_cell_conditioning.cells.add()
    cell_proto.gene_indices.append(0)
    cell_proto.gene_counts.append(5)

    interval = genome.Interval('chr1', 0, 16_384)
    expected_scores = [
        anndata.AnnData(
            X=np.zeros((5, 10)),
            var=pd.DataFrame({
                'name': [f'{i}' for i in range(10)],
                'strand': ['.'] * 10,
                'biosample_curie': ['UBERON:0000005'] * 10,
                'imputation_mode': ['imputed'] * 10,
                'biosample_type': ['tissue'] * 10,
                'biosample_name': ['sample'] * 10,
            }),
            uns={
                'interval': interval,
                'interval_scorer': interval_scorers.GeneMaskScorer(
                    requested_output=dna_client.OutputType.ATAC,
                    width=501,
                    aggregation_type=(
                        interval_scorers.IntervalAggregationType.MEAN
                    ),
                ),
            },
        )
    ]

    def _mock_generate(request, metadata):
      del metadata
      self.assertEqual(request.cell_conditioning, expected_cell_conditioning)
      self.assertFalse(request.HasField('condition'))
      return _generate_interval_scoring_proto(
          scores=expected_scores,
          response_proto_type=dna_model_service_pb2.ScoreIntervalResponse,
          interval=genome.Interval.from_proto(request.interval),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.score_interval(
        interval=interval,
        cell_conditioning=cell_conditioning_adata,
        cell_condition_name='cell_0',
        interval_scorers=[
            interval_scorers.GeneMaskScorer(
                requested_output=dna_client.OutputType.ATAC,
                width=501,
                aggregation_type=interval_scorers.IntervalAggregationType.MEAN,
            )
        ],
        organism=dna_client.Organism.HOMO_SAPIENS,
    )
    mock_stub.assert_called_once()
    self.assertLen(outputs, 1)
    self.assertIn('biosample_name', outputs[0].var.columns)
    self.assertTrue((outputs[0].var['biosample_name'] == 'cell_0').all())
    for name in outputs[0].var['name']:
      self.assertTrue(name.endswith('_cell_0'))

  def test_score_variant_with_cell_conditioning(self):
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    expected_cell_conditioning = dna_model_pb2_v2.CellConditioningInput(
        gene_ids=['ENSG00000000003'],
    )
    cell_proto = expected_cell_conditioning.cells.add()
    cell_proto.gene_indices.append(0)
    cell_proto.gene_counts.append(5)

    interval = genome.Interval('chr1', 0, 16_384)
    variant = genome.Variant('chr1', 1000, 'A', 'C')
    expected_scores = [
        anndata.AnnData(
            X=np.zeros((5, 10)),
            var=pd.DataFrame({
                'name': [f'{i}' for i in range(10)],
                'strand': ['.'] * 10,
                'biosample_curie': ['UBERON:0000005'] * 10,
                'imputation_mode': ['imputed'] * 10,
                'biosample_type': ['tissue'] * 10,
                'biosample_name': ['sample'] * 10,
            }),
            uns={
                'variant': variant,
                'interval': interval,
                'variant_scorer': variant_scorers.CenterMaskScorer(
                    requested_output=dna_client.OutputType.ATAC,
                    width=None,
                    aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
                ),
            },
        )
    ]

    def _mock_generate(request, metadata):
      del metadata
      self.assertEqual(request.cell_conditioning, expected_cell_conditioning)
      self.assertFalse(request.HasField('condition'))
      return _generate_variant_scoring_proto(
          scores=expected_scores,
          response_proto_type=dna_model_service_pb2.ScoreVariantResponse,
          variant=genome.Variant.from_proto(request.variant),
      )

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.score_variant(
        interval=interval,
        variant=variant,
        cell_conditioning=cell_conditioning_adata,
        cell_condition_name='cell_0',
        organism=dna_client.Organism.HOMO_SAPIENS,
        variant_scorers=[
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.ATAC,
                width=None,
                aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
            )
        ],
    )
    mock_stub.assert_called_once()
    self.assertLen(outputs, 1)
    self.assertIn('biosample_name', outputs[0].var.columns)
    self.assertTrue((outputs[0].var['biosample_name'] == 'cell_0').all())
    for name in outputs[0].var['name']:
      self.assertTrue(name.endswith('_cell_0'))

  def test_score_ism_variant_with_cell_conditioning(self):
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    expected_cell_conditioning = dna_model_pb2_v2.CellConditioningInput(
        gene_ids=['ENSG00000000003'],
    )
    cell_proto = expected_cell_conditioning.cells.add()
    cell_proto.gene_indices.append(0)
    cell_proto.gene_counts.append(5)

    interval = genome.Interval('chr1', 0, 16_384)
    ism_interval = genome.Interval('chr1', 10, 20)
    variant = genome.Variant('chr1', 10, 'A', 'T')
    example_scores = [
        anndata.AnnData(
            X=np.zeros((5, 10)),
            var=pd.DataFrame({
                'name': [f'{i}' for i in range(10)],
                'strand': ['.'] * 10,
                'biosample_curie': ['UBERON:0000005'] * 10,
                'imputation_mode': ['imputed'] * 10,
                'biosample_type': ['tissue'] * 10,
                'biosample_name': ['sample'] * 10,
            }),
            uns={
                'variant': variant,
                'interval': interval,
                'variant_scorer': variant_scorers.CenterMaskScorer(
                    requested_output=dna_client.OutputType.ATAC,
                    width=None,
                    aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
                ),
            },
        )
    ]

    def _mock_generate(request, metadata):
      del metadata
      self.assertEqual(request.cell_conditioning, expected_cell_conditioning)
      self.assertFalse(request.HasField('condition'))
      # mock response for a single variant in the ism interval
      response = _generate_variant_scoring_proto(
          scores=example_scores,
          response_proto_type=dna_model_service_pb2.ScoreIsmVariantResponse,
          variant=variant,
      )
      return response

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs_list = model.score_ism_variants(
        interval=interval,
        ism_interval=ism_interval,
        cell_conditioning=cell_conditioning_adata,
        cell_condition_name='cell_0',
        variant_scorers=[
            variant_scorers.CenterMaskScorer(
                requested_output=dna_client.OutputType.ATAC,
                width=None,
                aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
            )
        ],
    )
    mock_stub.assert_called_once()
    self.assertLen(outputs_list, 1)
    outputs = outputs_list[0]
    self.assertLen(outputs, 1)
    self.assertIn('biosample_name', outputs[0].var.columns)
    self.assertTrue((outputs[0].var['biosample_name'] == 'cell_0').all())
    for name in outputs[0].var['name']:
      self.assertTrue(name.endswith('_cell_0'))

  def test_encode_cells(self):
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    # Expected converted proto for assertion
    expected_cell_conditioning = dna_model_pb2_v2.CellConditioningInput(
        gene_ids=['ENSG00000000003'],
    )
    cell_proto = expected_cell_conditioning.cells.add()
    cell_proto.gene_indices.append(0)
    cell_proto.gene_counts.append(5)

    expected_embeddings = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
    )
    response_proto = dna_model_service_pb2.EncodeCellsResponse()
    tensor_proto, _ = tensor_utils.pack_tensor(expected_embeddings)
    response_proto.cell_embeddings.values.CopyFrom(tensor_proto)

    def _mock_encode(request, metadata):
      del metadata
      self.assertEqual(request.cell_conditioning, expected_cell_conditioning)
      return response_proto

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_encode)
    model = dna_client.DnaClient(channel=mock_channel)

    embeddings = model.encode_cells(cells=cell_conditioning_adata)
    np.testing.assert_array_equal(embeddings, expected_embeddings)
    mock_stub.assert_called_once()

  def test_encode_cells_splits_into_chunks(self):
    num_cells = 5
    cell_conditioning_adata = anndata.AnnData(
        X=np.arange(1, num_cells + 1, dtype=np.int32).reshape(-1, 1),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=[f'cell_{i}' for i in range(num_cells)]),
    )
    received_num_cells = []

    def _mock_encode(request, metadata):
      del metadata
      cells = request.cell_conditioning.cells
      received_num_cells.append(len(cells))
      # Embed each cell as its gene count so the output order can be checked.
      embeddings = np.array(
          [[cell.gene_counts[0]] for cell in cells], dtype=np.float32
      )
      response = dna_model_service_pb2.EncodeCellsResponse()
      tensor_proto, _ = tensor_utils.pack_tensor(embeddings)
      response.cell_embeddings.values.CopyFrom(tensor_proto)
      return response

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_encode)
    model = dna_client.DnaClient(channel=mock_channel)

    with mock.patch.object(dna_client, 'MAX_ENCODE_CELLS_PER_REQUEST', 2):
      embeddings = model.encode_cells(
          cells=cell_conditioning_adata, progress_bar=False
      )

    np.testing.assert_array_equal(
        embeddings,
        np.arange(1, num_cells + 1, dtype=np.float32).reshape(-1, 1),
    )
    self.assertEqual(mock_stub.call_count, 3)
    self.assertCountEqual(received_num_cells, [2, 2, 1])

  def test_encode_cells_no_cells_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(ValueError, 'must have at least one cell'):
      model.encode_cells(
          cells=anndata.AnnData(
              X=np.zeros((0, 1), dtype=np.int32),
              var=pd.DataFrame(index=['ENSG00000000003']),
          )
      )

  def test_predict_interval_gradients(self):
    gene_id = 'ENSG00000130203'
    gradients = np.ones((dna_client.SEQUENCE_LENGTH_100KB, 4), np.float32)
    interval = genome.Interval('chr19', 44_905_796, 44_905_796).resize(
        dna_client.SEQUENCE_LENGTH_100KB
    )
    response = dna_model_service_pb2.PredictIntervalGradientsResponse(
        output=dna_model_pb2_v2.InputGradientsOutput(
            gradients=tensor_utils.pack_tensor(gradients)[0],
            output_type=dna_model_pb2_v2.OUTPUT_TYPE_SC_GENE_EXPRESSION,
            interval=interval.to_proto(),
            scalar_score=1.25,
            gene_metadata=dna_model_pb2_v2.GeneExpressionMetadata(
                gene_id=gene_id,
                chromosome='chr19',
                start=44_905_796,
                end=44_909_393,
                strand=dna_model_pb2_v1.STRAND_POSITIVE,
                name='APOE',
                type='protein_coding',
            ),
            track_metadata=[
                dna_model_pb2_v2.TrackMetadata(
                    name='track_0',
                    strand=dna_model_pb2_v1.STRAND_POSITIVE,
                    biosample=dna_model_pb2_v1.Biosample(name='brain'),
                )
            ],
        )
    )

    def _mock_predict(request, metadata):
      del metadata
      self.assertEqual(request.target.gene_id, gene_id)
      self.assertEqual(
          request.sequence_length, dna_client.SEQUENCE_LENGTH_100KB
      )
      return response

    mock_channel, mock_stub = _create_mock_channel_and_stub(_mock_predict)
    model = dna_client.DnaClient(channel=mock_channel)

    output = model.predict_interval_gradients(
        gene_id,
        requested_output=dna_client.OutputType.SC_GENE_EXPRESSION,
        sequence_length=dna_client.SEQUENCE_LENGTH_100KB,
        condition=model_condition.from_curies(['UBERON:0000955']),
    )
    np.testing.assert_array_equal(output.gradients, gradients)
    self.assertEqual(output.interval, interval)
    self.assertAlmostEqual(output.scalar_score, 1.25)
    self.assertEqual(
        output.output_type, dna_client.OutputType.SC_GENE_EXPRESSION
    )
    self.assertEqual(
        output.target_gene,
        dna_client.TargetGeneMetadata(
            gene_id=gene_id,
            interval=genome.Interval('chr19', 44_905_796, 44_909_393, '+'),
            gene_name='APOE',
            gene_type='protein_coding',
        ),
    )
    self.assertEqual(output.gene_id, gene_id)
    self.assertEqual(output.gene_name, 'APOE')
    self.assertEqual(output.biosample_name, 'brain')
    self.assertEqual(list(output.track_metadata['name']), ['track_0'])

    cell_output = model.predict_interval_gradients(
        gene_id,
        requested_output=dna_client.OutputType.SC_GENE_EXPRESSION_DENOISED,
        sequence_length=dna_client.SEQUENCE_LENGTH_100KB,
        cell_conditioning=anndata.AnnData(
            X=np.array([[5]], dtype=np.int32),
            var=pd.DataFrame(index=['ENSG00000000003']),
            obs=pd.DataFrame(index=['cell_0']),
        ),
        cell_condition_name='my_cell',
    )
    self.assertEqual(
        list(cell_output.track_metadata['name']), ['track_0_my_cell']
    )
    self.assertEqual(
        list(cell_output.track_metadata['biosample_name']), ['my_cell']
    )
    self.assertEqual(mock_stub.call_count, 2)

    with self.assertRaisesRegex(ValueError, 'sequence_length must be'):
      model.predict_interval_gradients(
          gene_id,
          requested_output=dna_client.OutputType.SC_GENE_EXPRESSION,
          sequence_length=dna_client.SEQUENCE_LENGTH_16KB,
          condition=model_condition.from_curies(['UBERON:0000955']),
      )
    with self.assertRaisesRegex(
        ValueError, 'Cannot specify both `condition` and `cell_conditioning`.'
    ):
      model.predict_interval_gradients_batch(
          gene_ids=gene_id,
          requested_output=dna_client.OutputType.SC_GENE_EXPRESSION_DENOISED,
          sequence_length=dna_client.SEQUENCE_LENGTH_1MB,
          conditions=['UBERON:0000955'],
          cell_conditionings=[
              anndata.AnnData(
                  X=np.array([[5]], dtype=np.int32),
                  var=pd.DataFrame(index=['ENSG00000000003']),
                  obs=pd.DataFrame(index=['cell_0']),
              )
          ],
          cell_condition_names=['my_cell'],
          progress_bar=False,
      )
    with self.assertRaisesRegex(
        ValueError, 'Must specify either `condition` or `cell_conditioning`.'
    ):
      model.predict_interval_gradients_batch(
          gene_ids=gene_id,
          requested_output=dna_client.OutputType.RNA_SEQ,
          sequence_length=dna_client.SEQUENCE_LENGTH_1MB,
          progress_bar=False,
      )

  def test_cell_conditioning_missing_name_raises(self):
    model = dna_client.DnaClient(channel=mock.Mock())
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[5]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    with self.assertRaisesRegex(
        ValueError,
        'cell_condition_name must be provided when cell_conditioning is used.',
    ):
      model.predict_sequence(
          'A' * 16_384,
          requested_outputs=[dna_client.OutputType.ATAC],
          cell_conditioning=cell_conditioning_adata,
      )
    with self.assertRaisesRegex(
        ValueError,
        'cell_condition_name must be provided when cell_conditioning is used.',
    ):
      model.predict_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          requested_outputs=[dna_client.OutputType.ATAC],
          cell_conditioning=cell_conditioning_adata,
      )
    with self.assertRaisesRegex(
        ValueError,
        'cell_condition_name must be provided when cell_conditioning is used.',
    ):
      model.predict_variant(
          interval=genome.Interval('chr1', 0, 16_384),
          variant=genome.Variant('chr1', 1000, 'A', 'T'),
          requested_outputs=[dna_client.OutputType.ATAC],
          cell_conditioning=cell_conditioning_adata,
      )
    with self.assertRaisesRegex(
        ValueError,
        'cell_condition_name must be provided when cell_conditioning is used.',
    ):
      model.score_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          cell_conditioning=cell_conditioning_adata,
      )
    with self.assertRaisesRegex(
        ValueError,
        'cell_condition_name must be provided when cell_conditioning is used.',
    ):
      model.score_variant(
          interval=genome.Interval('chr1', 0, 16_384),
          variant=genome.Variant('chr1', 1000, 'A', 'T'),
          cell_conditioning=cell_conditioning_adata,
      )
    with self.assertRaisesRegex(
        ValueError,
        'cell_condition_name must be provided when cell_conditioning is used.',
    ):
      model.score_ism_variants(
          interval=genome.Interval('chr1', 0, 16_384),
          ism_interval=genome.Interval('chr1', 1000, 1010),
          cell_conditioning=cell_conditioning_adata,
      )
    with self.assertRaisesRegex(
        ValueError,
        'cell_condition_name must be provided when cell_conditioning is used.',
    ):
      model.predict_interval_gradients(
          'ENSG00000130203',
          requested_output=dna_client.OutputType.SC_GENE_EXPRESSION_DENOISED,
          cell_conditioning=cell_conditioning_adata,
      )

  def test_cell_conditioning_too_many_cells_raises(self):
    model = dna_client.DnaClient(channel=mock.Mock())
    num_cells = dna_client.MAX_CONDITIONING_CELLS + 1
    cell_conditioning_adata = anndata.AnnData(
        X=np.ones((num_cells, 1), dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003']),
        obs=pd.DataFrame(index=[f'cell_{i}' for i in range(num_cells)]),
    )
    expected_msg = (
        f'Too many cells in cell_conditioning \\({num_cells}\\), maximum'
        f' allowed: {dna_client.MAX_CONDITIONING_CELLS}\\.'
    )
    with self.assertRaisesRegex(ValueError, expected_msg):
      model.predict_sequence(
          'A' * 16_384,
          requested_outputs=[dna_client.OutputType.ATAC],
          cell_conditioning=cell_conditioning_adata,
          cell_condition_name='test_cell',
      )
    with self.assertRaisesRegex(ValueError, expected_msg):
      model.predict_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          requested_outputs=[dna_client.OutputType.ATAC],
          cell_conditioning=cell_conditioning_adata,
          cell_condition_name='test_cell',
      )
    with self.assertRaisesRegex(ValueError, expected_msg):
      model.predict_variant(
          interval=genome.Interval('chr1', 0, 16_384),
          variant=genome.Variant('chr1', 1000, 'A', 'T'),
          requested_outputs=[dna_client.OutputType.ATAC],
          cell_conditioning=cell_conditioning_adata,
          cell_condition_name='test_cell',
      )
    with self.assertRaisesRegex(ValueError, expected_msg):
      model.score_interval(
          interval=genome.Interval('chr1', 0, 16_384),
          cell_conditioning=cell_conditioning_adata,
          cell_condition_name='test_cell',
      )
    with self.assertRaisesRegex(ValueError, expected_msg):
      model.score_variant(
          interval=genome.Interval('chr1', 0, 16_384),
          variant=genome.Variant('chr1', 1000, 'A', 'T'),
          cell_conditioning=cell_conditioning_adata,
          cell_condition_name='test_cell',
      )
    with self.assertRaisesRegex(ValueError, expected_msg):
      model.score_ism_variants(
          interval=genome.Interval('chr1', 0, 16_384),
          ism_interval=genome.Interval('chr1', 1000, 1010),
          cell_conditioning=cell_conditioning_adata,
          cell_condition_name='test_cell',
      )

  def test_cell_conditioning_rejected_for_non_human_organism(self):
    cell_conditioning_adata = anndata.AnnData(
        X=np.array([[1, 2]], dtype=np.int32),
        var=pd.DataFrame(index=['ENSG00000000003', 'ENSG00000139618']),
        obs=pd.DataFrame(index=['cell_0']),
    )
    mock_channel = mock.create_autospec(grpc.Channel, instance=True)
    model = dna_client.DnaClient(channel=mock_channel)

    with self.assertRaisesRegex(
        ValueError,
        'Cell conditioning is only supported for Homo sapiens',
    ):
      model.predict_sequence(
          sequence='A' * 524_288,
          requested_outputs=[dna_client.OutputType.RNA_SEQ],
          organism=dna_client.Organism.MUS_MUSCULUS,
          cell_conditioning=cell_conditioning_adata,
          cell_condition_name='test_cell',
      )


if __name__ == '__main__':
  absltest.main()
