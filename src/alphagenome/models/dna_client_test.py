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

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Sequence
from unittest import mock

import anndata
import grpc
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from alphagenome import tensor_utils
from alphagenome.data import genome, junction_data, track_data
from alphagenome.interpretation import ism
from alphagenome.models import (
    dna_client,
    interval_scorers,
    junction_data_utils,
    track_data_utils,
    variant_scorers,
)
from alphagenome.protos import (
    dna_model_pb2,
    dna_model_service_pb2,
    dna_model_service_pb2_grpc,
)


class _FakeRpcError(grpc.RpcError):
  """Fake RPC error for testing."""

  def __init__(self, code: grpc.StatusCode):
    self._code = code

  def code(self) -> grpc.StatusCode:
    return self._code


def _generate_prediction_protos(
    predictions: dna_client.Output,
    requested_outputs: Sequence[dna_model_pb2.OutputType],
    bytes_per_chunk: int,
    response_proto_type: ...,
):
  """Helper to generate proto responses for the provided predictions."""
  for output_type in requested_outputs:
    prediction = predictions.get(dna_client.OutputType(output_type))
    if prediction is not None:
      if isinstance(prediction, track_data.TrackData):
        proto, chunks = track_data_utils.to_protos(
            prediction, bytes_per_chunk=bytes_per_chunk
        )
        yield response_proto_type(
            output=dna_model_pb2.Output(
                output_type=output_type, track_data=proto
            )
        )
      elif isinstance(prediction, junction_data.JunctionData):
        proto, chunks = junction_data_utils.to_protos(
            prediction, bytes_per_chunk=bytes_per_chunk
        )
        yield response_proto_type(
            output=dna_model_pb2.Output(
                output_type=output_type, junction_data=proto
            )
        )
      else:
        raise ValueError(f'Unsupported prediction type: {type(prediction)}')

      for chunk in chunks:
        yield response_proto_type(tensor_chunk=chunk)


def _generate_variant_protos(
    predictions: dna_client.VariantOutput,
    requested_outputs: Sequence[dna_model_pb2.OutputType],
    bytes_per_chunk: int,
):
  """Helper to generate variant proto responses for the provided predictions."""
  for output_type in requested_outputs:
    prediction = predictions.reference.get(dna_client.OutputType(output_type))
    if prediction is not None:
      if isinstance(prediction, track_data.TrackData):
        proto, chunks = track_data_utils.to_protos(
            prediction, bytes_per_chunk=bytes_per_chunk
        )
        yield dna_model_service_pb2.PredictVariantResponse(
            reference_output=dna_model_pb2.Output(
                output_type=output_type, track_data=proto
            )
        )
      elif isinstance(prediction, junction_data.JunctionData):
        proto, chunks = junction_data_utils.to_protos(
            prediction, bytes_per_chunk=bytes_per_chunk
        )
        yield dna_model_service_pb2.PredictVariantResponse(
            reference_output=dna_model_pb2.Output(
                output_type=output_type, junction_data=proto
            )
        )
      else:
        raise ValueError(f'Unsupported prediction type: {type(prediction)}')

      for chunk in chunks:
        yield dna_model_service_pb2.PredictVariantResponse(tensor_chunk=chunk)

  for output_type in requested_outputs:
    prediction = predictions.alternate.get(dna_client.OutputType(output_type))
    if prediction is not None:
      if isinstance(prediction, track_data.TrackData):
        proto, chunks = track_data_utils.to_protos(
            prediction, bytes_per_chunk=bytes_per_chunk
        )
        yield dna_model_service_pb2.PredictVariantResponse(
            alternate_output=dna_model_pb2.Output(
                output_type=output_type, track_data=proto
            )
        )
      elif isinstance(prediction, junction_data.JunctionData):
        proto, chunks = junction_data_utils.to_protos(
            prediction, bytes_per_chunk=bytes_per_chunk
        )
        yield dna_model_service_pb2.PredictVariantResponse(
            alternate_output=dna_model_pb2.Output(
                output_type=output_type, junction_data=proto
            )
        )
      else:
        raise ValueError(f'Unsupported prediction type: {type(prediction)}')

      for chunk in chunks:
        yield dna_model_service_pb2.PredictVariantResponse(tensor_chunk=chunk)


def _generate_variant_scoring_protos(
    scores: list[anndata.AnnData],
    bytes_per_chunk: int,
    response_proto_type: ...,
    variant: genome.Variant | None = None,
):
  """Helper to generate variant scoring responses for the provided scores."""
  for score in scores:
    track_metadata_protos = [
        dna_model_pb2.TrackMetadata(
            name=name,
            strand=genome.Strand.from_str(strand).to_proto(),
        )
        for name, strand in score.var[['name', 'strand']].values
    ]
    gene_metadata_protos = []
    if score.obs is not None and 'gene_id' in score.obs:
      for _, row in score.obs.iterrows():
        if (strand := row.get('strand')) is not None:
          strand = genome.Strand.from_str(strand).to_proto()
        gene_metadata_protos.append(
            dna_model_pb2.GeneScorerMetadata(
                gene_id=row['gene_id'],
                strand=strand,
                name=row.get('gene_name'),
                type=row.get('gene_type'),
                junction_start=row.get('junction_Start'),
                junction_end=row.get('junction_End'),
            )
        )

    if 'quantiles' in score.layers:
      score_tensor = np.stack([score.X, score.layers['quantiles']])
    else:
      score_tensor = score.X[np.newaxis]

    tensor_proto, chunks = tensor_utils.pack_tensor(
        score_tensor, bytes_per_chunk=bytes_per_chunk
    )
    yield response_proto_type(
        output=dna_model_pb2.ScoreVariantOutput(
            variant_data=dna_model_pb2.VariantData(
                values=tensor_proto,
                metadata=dna_model_pb2.VariantMetadata(
                    variant=variant.to_proto() if variant else None,
                    track_metadata=track_metadata_protos,
                    gene_metadata=gene_metadata_protos,
                ),
            )
        )
    )
    for chunk in chunks:
      yield response_proto_type(tensor_chunk=chunk)


def _generate_interval_scoring_protos(
    scores: list[anndata.AnnData],
    bytes_per_chunk: int,
    response_proto_type: ...,
    interval: genome.Interval,
):
  """Helper to generate interval scoring responses for the provided scores."""
  for score in scores:
    track_metadata_protos = [
        dna_model_pb2.TrackMetadata(
            name=name,
            strand=genome.Strand.from_str(strand).to_proto(),
        )
        for name, strand in score.var[['name', 'strand']].values
    ]
    gene_metadata_protos = []
    if score.obs is not None and 'gene_id' in score.obs:
      for _, row in score.obs.iterrows():
        if (strand := row.get('strand')) is not None:
          strand = genome.Strand.from_str(strand).to_proto()
        gene_metadata_protos.append(
            dna_model_pb2.GeneScorerMetadata(
                gene_id=row['gene_id'],
                strand=strand,
                name=row.get('gene_name'),
                type=row.get('gene_type'),
            )
        )

    tensor_proto, chunks = tensor_utils.pack_tensor(
        score.X[np.newaxis], bytes_per_chunk=bytes_per_chunk
    )
    yield response_proto_type(
        output=dna_model_pb2.ScoreIntervalOutput(
            interval_data=dna_model_pb2.IntervalData(
                values=tensor_proto,
                metadata=dna_model_pb2.IntervalMetadata(
                    interval=interval.to_proto(),
                    track_metadata=track_metadata_protos,
                    gene_metadata=gene_metadata_protos,
                ),
            )
        )
    )
    for chunk in chunks:
      yield response_proto_type(tensor_chunk=chunk)


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

  return dna_client.Output(
      atac=track_data.TrackData(
          values=np.full((sequence_length, 32), value, dtype=np.float32),
          metadata=pd.DataFrame({
              'name': [f'{i}' for i in range(32)],
              'strand': ['.'] * 32,
          }),
          interval=interval,
      ),
      rna_seq=track_data.TrackData(
          values=np.full((sequence_length, 64), value, dtype=np.float32),
          metadata=pd.DataFrame({
              'name': [f'{i}' for i in range(64)],
              'strand': ['.'] * 64,
          }),
          interval=interval,
      ),
      splice_junctions=junction_data.JunctionData(
          junctions=np.array([
              genome.Interval('chr1', 0, 1000, strand='+'),
              genome.Interval('chr1', 1500, 2000, strand='-'),
          ]),
          values=np.full([2, 3], value, dtype=np.float32),
          metadata=pd.DataFrame({'name': [f'{i}' for i in range(3)]}),
          interval=interval,
      ),
  )


def _create_mock_channel_and_stream(
    response_callable, *, bidi_stream: bool = True
) -> ...:
  mock_stream = mock.create_autospec(
      grpc.StreamStreamMultiCallable, side_effect=response_callable
  )
  mock_channel = mock.create_autospec(grpc.Channel)
  if bidi_stream:
    mock_channel.stream_stream.return_value = mock_stream
  else:
    mock_channel.unary_stream.return_value = mock_stream
  return mock_channel, mock_stream


class ClientTest(parameterized.TestCase):

  def _assert_output_equal(
      self,
      actual: dna_client.Output,
      expected: dna_client.Output,
      msg: ...,
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

  def _assert_track_metadata_equal(
      self,
      actual: track_data.TrackMetadata,
      expected: track_data.TrackMetadata,
      msg: ...,
  ) -> None:
    if actual is not None and expected is not None:
      pd.testing.assert_frame_equal(actual, expected)
    else:
      self.assertEqual(actual, expected, msg)

  def _assert_output_metadata_equal(
      self,
      actual: dna_client.OutputMetadata,
      expected: dna_client.OutputMetadata,
      msg: ...,
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

  def _assert_track_data_equal(
      self,
      actual: track_data.TrackData,
      expected: track_data.TrackData,
      msg: ...,
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
      msg: ...,
  ) -> None:
    pd.testing.assert_frame_equal(actual.metadata, expected.metadata)
    np.testing.assert_array_equal(actual.junctions, expected.junctions)
    np.testing.assert_array_equal(actual.values, expected.values)
    self.assertEqual(actual.interval, expected.interval, msg)
    self.assertEqual(actual.uns, expected.uns, msg)

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
    if 'quantiles' in expected.layers:
      np.testing.assert_array_equal(
          actual.layers['quantiles'], expected.layers['quantiles']
      )
    else:
      self.assertNotIn('quantiles', actual.layers)

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
                      }),
                  ),
                  rna_seq=track_data.TrackData(
                      values=np.zeros((16_384, 64), dtype=np.float32),
                      metadata=pd.DataFrame({
                          'name': [f'{i}' for i in range(64)],
                          'strand': ['.'] * 64,
                      }),
                  ),
                  splice_junctions=junction_data.JunctionData(
                      junctions=np.array([
                          genome.Interval('chr1', 0, 1000, strand='+'),
                          genome.Interval('chr1', 1500, 2000, strand='-'),
                      ]),
                      values=np.zeros([2, 3], dtype=np.float32),
                      metadata=pd.DataFrame(
                          {'name': [f'{i}' for i in range(3)]}
                      ),
                  ),
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
      bytes_per_chunk=[0, 1024],
      model_version=[None, dna_client.ModelVersion.FOLD_0],
      with_interval=[True, False],
  )
  def test_predict_sequence(
      self,
      requested_output_types: set[dna_client.OutputType] | None,
      sequence_length: int,
      expected: dna_client.Output,
      bytes_per_chunk: int,
      model_version: dna_client.ModelVersion | None,
      with_interval: bool,
  ):
    mock_predictions = _create_example_model_predictions(
        sequence_length=sequence_length
    )

    expected_request = dna_model_service_pb2.PredictSequenceRequest(
        sequence='A' * sequence_length,
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        requested_outputs=[o.to_proto() for o in requested_output_types],
        ontology_terms=[
            dna_model_pb2.OntologyTerm(
                ontology_type=dna_model_pb2.ONTOLOGY_TYPE_UBERON, id=4
            )
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

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        self.assertEqual(request, expected_request)
        yield from _generate_prediction_protos(
            mock_predictions,
            request.requested_outputs,
            bytes_per_chunk,
            dna_model_service_pb2.PredictSequenceResponse,
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(
        channel=mock_channel, model_version=model_version
    )

    outputs = model.predict_sequence(
        'A' * sequence_length,
        organism=dna_client.Organism.HOMO_SAPIENS,
        ontology_terms=['UBERON:00004'],
        requested_outputs=requested_output_types,
        interval=interval,
    )

    self.assertEqual(outputs, expected)
    mock_stream.assert_called_once()

  @parameterized.product(
      sequence_length=[dna_client.SEQUENCE_LENGTH_16KB],
      num_predictions=[1, 4],
      num_workers=[1, 4],
      bytes_per_chunk=[0, 1024],
      with_interval=[True, False],
  )
  def test_predict_sequences(
      self,
      sequence_length: int,
      num_predictions: int,
      num_workers: int,
      bytes_per_chunk: int,
      with_interval: bool,
  ):
    interval = None
    if with_interval:
      interval = genome.Interval('chr1', 0, sequence_length)

    mock_predictions = _create_example_model_predictions(
        sequence_length=sequence_length, interval=interval
    )

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_prediction_protos(
            mock_predictions,
            request.requested_outputs,
            bytes_per_chunk,
            dna_model_service_pb2.PredictSequenceResponse,
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.predict_sequences(
        ['A' * sequence_length] * num_predictions,
        organism=dna_client.Organism.HOMO_SAPIENS,
        ontology_terms=None,
        requested_outputs=[*dna_client.OutputType],
        progress_bar=False,
        max_workers=num_workers,
        intervals=[interval] * num_predictions if with_interval else None,
    )
    for output in outputs:
      self.assertEqual(output, mock_predictions)

    self.assertEqual(mock_stream.call_count, num_predictions)

  def test_predict_sequences_error_raises(self):
    mock_channel, mock_stream = _create_mock_channel_and_stream(
        _FakeRpcError(grpc.StatusCode.INTERNAL)
    )
    model = dna_client.DnaClient(channel=mock_channel)
    with self.assertRaises(grpc.RpcError):
      model.predict_sequences(
          ['A' * dna_client.SEQUENCE_LENGTH_16KB],
          ontology_terms=None,
          requested_outputs=[*dna_client.OutputType],
      )

    mock_stream.assert_called_once()

  def test_invalid_sequence_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError,
        'Invalid sequence, must only contain the characters "ACGTN". Found'
        ' invalid characters: "[wozers,]*"',
    ):
      model.predict_sequence(
          'wowzers',
          ontology_terms=None,
          requested_outputs=[dna_client.OutputType.ATAC],
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
      bytes_per_chunk=[0, 1024],
      model_version=[None, dna_client.ModelVersion.FOLD_0],
  )
  def test_predict_interval(
      self,
      requested_output_types,
      interval,
      organism,
      expected,
      bytes_per_chunk,
      model_version,
  ):
    mock_predictions = _create_example_model_predictions(interval=interval)

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_prediction_protos(
            mock_predictions,
            request.requested_outputs,
            bytes_per_chunk,
            dna_model_service_pb2.PredictIntervalResponse,
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(
        channel=mock_channel, model_version=model_version
    )

    outputs = model.predict_interval(
        interval=interval,
        organism=organism,
        ontology_terms=['UBERON:00004'],
        requested_outputs=requested_output_types,
    )
    self.assertEqual(expected, outputs)

    mock_stream.assert_called_once()

  @parameterized.product(
      intervals=[
          [genome.Interval('chr1', 0, 16_384)],
          [
              genome.Interval('chr1', 0, 16_384),
              genome.Interval('chr1', 16_384, 32_768),
          ],
      ],
      num_workers=[1, 4],
      bytes_per_chunk=[0, 1024],
  )
  def test_predict_intervals(self, intervals, num_workers, bytes_per_chunk):
    examples = {
        (interval.start, interval.end): _create_example_model_predictions(
            interval=interval, value=interval.start
        )
        for interval in intervals
    }

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_prediction_protos(
            examples[(request.interval.start, request.interval.end)],
            request.requested_outputs,
            bytes_per_chunk,
            dna_model_service_pb2.PredictIntervalResponse,
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.predict_intervals(
        intervals=intervals,
        ontology_terms=None,
        requested_outputs=[*dna_client.OutputType],
        max_workers=num_workers,
        progress_bar=False,
    )
    for interval, output in zip(intervals, outputs):
      self.assertEqual(output, examples[(interval.start, interval.end)])

    self.assertLen(intervals, mock_stream.call_count)

  def test_predict_intervals_error_raises(self):
    mock_channel, mock_stream = _create_mock_channel_and_stream(
        _FakeRpcError(grpc.StatusCode.INTERNAL)
    )
    model = dna_client.DnaClient(channel=mock_channel)
    with self.assertRaises(grpc.RpcError):
      model.predict_intervals(
          [genome.Interval('chr1', 0, 16_384)],
          ontology_terms=None,
          requested_outputs=[*dna_client.OutputType],
      )

    mock_stream.assert_called_once()

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
      bytes_per_chunk=[0, 1024],
      model_version=[None, dna_client.ModelVersion.FOLD_1],
  )
  def test_predict_variant(
      self,
      requested_output_types,
      interval,
      variant,
      organism,
      expected,
      bytes_per_chunk,
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

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_variant_protos(
            mock_predictions, request.requested_outputs, bytes_per_chunk
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(
        channel=mock_channel, model_version=model_version
    )

    outputs = model.predict_variant(
        interval=interval,
        variant=variant,
        organism=organism,
        ontology_terms=['UBERON:00004'],
        requested_outputs=requested_output_types,
    )

    self.assertEqual(expected.reference, outputs.reference)
    self.assertEqual(expected.alternate, outputs.alternate)

    mock_stream.assert_called_once()

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
      bytes_per_chunk=[0, 2048],
  )
  def test_predict_variants(
      self, intervals, variants, num_workers, bytes_per_chunk
  ):
    if isinstance(intervals, genome.Interval):
      intervals = [intervals] * len(variants)
    examples = {
        (interval.start, interval.end): _create_example_model_predictions(
            interval=interval, value=interval.start
        )
        for interval in intervals
    }

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        key = (request.interval.start, request.interval.end)
        yield from _generate_variant_protos(
            dna_client.VariantOutput(
                reference=examples[key], alternate=examples[key]
            ),
            request.requested_outputs,
            bytes_per_chunk,
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.predict_variants(
        intervals=intervals,
        variants=variants,
        ontology_terms=None,
        requested_outputs=[*dna_client.OutputType],
        max_workers=num_workers,
        progress_bar=False,
    )
    for interval, output in zip(intervals, outputs, strict=True):
      expected = examples[(interval.start, interval.end)]
      self.assertEqual(output.reference, expected)
      self.assertEqual(output.alternate, expected)

    self.assertLen(intervals, mock_stream.call_count)

  def test_predict_variants_error_raises(self):
    mock_channel, mock_stream = _create_mock_channel_and_stream(
        _FakeRpcError(grpc.StatusCode.INTERNAL)
    )
    model = dna_client.DnaClient(channel=mock_channel)
    with self.assertRaises(grpc.RpcError):
      model.predict_variants(
          [genome.Interval('chr1', 0, 16_384)],
          [genome.Variant('chr1', 1000, 'A', 'C')],
          ontology_terms=None,
          requested_outputs=[*dna_client.OutputType],
      )

    mock_stream.assert_called_once()

  def test_predict_variants_mismatch_lengths_raises(self):
    mock_channel, mock_stream = _create_mock_channel_and_stream(None)
    model = dna_client.DnaClient(channel=mock_channel)
    with self.assertRaisesRegex(
        ValueError, 'Intervals and variants must have the same length'
    ):
      model.predict_variants(
          [
              genome.Interval('chr1', 0, 16_384),
              genome.Interval('chr1', 16_384, 32_768),
          ],
          [genome.Variant('chr1', 1000, 'A', 'C')],
          ontology_terms=None,
          requested_outputs=[*dna_client.OutputType],
      )
    mock_stream.assert_not_called()

  def test_output_metadata(self):

    def _mock_generate(requests, metadata):
      del metadata
      del requests
      yield dna_model_service_pb2.MetadataResponse(
          output_metadata=[
              dna_model_pb2.OutputMetadata(
                  output_type=dna_model_pb2.OUTPUT_TYPE_ATAC,
                  tracks=dna_model_pb2.TracksMetadata(
                      metadata=[
                          dna_model_pb2.TrackMetadata(
                              name=f'track_{i}',
                              strand=dna_model_pb2.STRAND_UNSTRANDED,
                          )
                          for i in range(32)
                      ]
                  ),
              ),
              dna_model_pb2.OutputMetadata(
                  output_type=dna_model_pb2.OUTPUT_TYPE_SPLICE_JUNCTIONS,
                  junctions=dna_model_pb2.JunctionsMetadata(
                      metadata=[
                          dna_model_pb2.JunctionMetadata(name=f'junction_{i}')
                          for i in range(16)
                      ]
                  ),
              ),
          ]
      )

    mock_channel, mock_stream = _create_mock_channel_and_stream(
        _mock_generate, bidi_stream=False
    )
    model = dna_client.DnaClient(channel=mock_channel)

    outputs = model.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
    expected = dna_client.OutputMetadata(
        atac=track_data.TrackMetadata(
            {'name': [f'track_{i}' for i in range(32)], 'strand': ['.'] * 32}
        ),
        splice_junctions=junction_data.JunctionMetadata(
            {'name': [f'junction_{i}' for i in range(16)]}
        ),
    )
    self.assertEqual(expected, outputs)
    mock_stream.assert_called_once()

  def test_output_type_to_proto(self):
    expected = [
        dna_model_pb2.OUTPUT_TYPE_ATAC,
        dna_model_pb2.OUTPUT_TYPE_CAGE,
        dna_model_pb2.OUTPUT_TYPE_DNASE,
        dna_model_pb2.OUTPUT_TYPE_RNA_SEQ,
        dna_model_pb2.OUTPUT_TYPE_CHIP_HISTONE,
        dna_model_pb2.OUTPUT_TYPE_CHIP_TF,
        dna_model_pb2.OUTPUT_TYPE_SPLICE_SITES,
        dna_model_pb2.OUTPUT_TYPE_SPLICE_SITE_USAGE,
        dna_model_pb2.OUTPUT_TYPE_SPLICE_JUNCTIONS,
        dna_model_pb2.OUTPUT_TYPE_CONTACT_MAPS,
        dna_model_pb2.OUTPUT_TYPE_PROCAP,
    ]

    for output_type, expected_proto in zip(
        dna_client.OutputType, expected, strict=True
    ):
      self.assertEqual(output_type.to_proto(), expected_proto)

  def test_organism_to_proto(self):
    expected = [
        dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        dna_model_pb2.ORGANISM_MUS_MUSCULUS,
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
      bytes_per_chunk=[0, 1024],
  )
  def test_score_interval(
      self,
      interval,
      scorers,
      expected,
      organism,
      bytes_per_chunk,
  ):

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_interval_scoring_protos(
            scores=expected,
            bytes_per_chunk=bytes_per_chunk,
            response_proto_type=dna_model_service_pb2.ScoreIntervalResponse,
            interval=genome.Interval.from_proto(request.interval),
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    outputs = model.score_interval(
        interval=interval,
        organism=organism,
        interval_scorers=scorers,
    )

    for expected_anndata, output_anndata in zip(expected, outputs):
      self._assert_anndata_equal(expected_anndata, output_anndata)

    mock_stream.assert_called_once()

  @parameterized.product(organism=list(dna_client.Organism))
  def test_recommended_interval_scorers(self, organism):
    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_interval_scoring_protos(
            scores=[
                anndata.AnnData(
                    np.zeros((0, 0)),
                    var=pd.DataFrame(columns=['name', 'strand']),
                )
                for _ in range(len(request.interval_scorers))
            ],
            bytes_per_chunk=1024,
            response_proto_type=dna_model_service_pb2.ScoreIntervalResponse,
            interval=genome.Interval.from_proto(request.interval),
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    scores = model.score_interval(
        interval=genome.Interval('chr1', 0, 2**20), organism=organism
    )

    self.assertSameElements(
        [score.uns['interval_scorer'] for score in scores],
        interval_scorers.RECOMMENDED_INTERVAL_SCORERS.values(),
    )
    mock_stream.assert_called_once()

  def test_duplicate_interval_scorers_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError, 'Duplicate interval scorers requested'
    ):
      model.score_interval(
          interval=genome.Interval('chr1', 0, 16_384),
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
          interval=genome.Interval('chr1', 0, 16_384), interval_scorers=scorers
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
      bytes_per_chunk=[0, 1024],
  )
  def test_score_intervals(
      self,
      interval,
      scorers,
      expected,
      organism,
      bytes_per_chunk,
  ):
    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_interval_scoring_protos(
            scores=expected,
            bytes_per_chunk=bytes_per_chunk,
            response_proto_type=dna_model_service_pb2.ScoreIntervalResponse,
            interval=genome.Interval.from_proto(request.interval),
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    all_outputs = model.score_intervals(
        intervals=[interval, interval],
        organism=organism,
        interval_scorers=scorers,
    )

    for outputs in all_outputs:
      for expected_anndata, output_anndata in zip(expected, outputs):
        self._assert_anndata_equal(expected_anndata, output_anndata)

    self.assertEqual(mock_stream.call_count, 2)

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
                  }),
                  uns={
                      'variant': genome.Variant('chr1', 1000, 'A', 'C'),
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'variant_scorer': variant_scorers.GeneMaskLFCScorer(
                          requested_output=dna_client.OutputType.RNA_SEQ,
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
                  }),
                  uns={
                      'variant': genome.Variant('chr1', 1000, 'A', 'C'),
                      'interval': genome.Interval('chr1', 0, 16_384),
                      'variant_scorer': variant_scorers.GeneMaskLFCScorer(
                          requested_output=dna_client.OutputType.RNA_SEQ,
                      ),
                  },
              ),
          ),
      ],
      organism=[
          dna_client.Organism.HOMO_SAPIENS,
          dna_client.Organism.MUS_MUSCULUS,
      ],
      bytes_per_chunk=[0, 1024],
  )
  def test_score_variant(
      self,
      interval,
      variant,
      scorers,
      expected,
      organism,
      bytes_per_chunk,
  ):

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_variant_scoring_protos(
            scores=expected,
            bytes_per_chunk=bytes_per_chunk,
            response_proto_type=dna_model_service_pb2.ScoreVariantResponse,
            variant=genome.Variant.from_proto(request.variant),
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    outputs = model.score_variant(
        interval=interval,
        variant=variant,
        organism=organism,
        variant_scorers=scorers,
    )

    for expected_anndata, output_anndata in zip(expected, outputs):
      self._assert_anndata_equal(expected_anndata, output_anndata)

    mock_stream.assert_called_once()

  def test_duplicate_variant_scorers_raises_error(self):
    model = dna_client.DnaClient(channel=mock.create_autospec(grpc.Channel))
    with self.assertRaisesRegex(
        ValueError, 'Duplicate variant scorers requested'
    ):
      model.score_variant(
          interval=genome.Interval('chr1', 0, 16_384),
          variant=genome.Variant('chr1', 1000, 'A', 'C'),
          variant_scorers=[
              variant_scorers.GeneMaskLFCScorer(
                  requested_output=dna_client.OutputType.RNA_SEQ,
              ),
              variant_scorers.GeneMaskLFCScorer(
                  requested_output=dna_client.OutputType.RNA_SEQ,
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
          variant_scorers=variant_scorers_to_use,
      )

  @parameterized.product(organism=list(dna_client.Organism))
  def test_recommended_variant_scorers(self, organism):
    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_variant_scoring_protos(
            scores=[
                anndata.AnnData(
                    np.zeros((0, 0)),
                    var=pd.DataFrame(columns=['name', 'strand']),
                )
                for _ in range(len(request.variant_scorers))
            ],
            bytes_per_chunk=1024,
            response_proto_type=dna_model_service_pb2.ScoreVariantResponse,
            variant=genome.Variant.from_proto(request.variant),
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    scores = model.score_variant(
        interval=genome.Interval('chr1', 0, 16_384),
        variant=genome.Variant('chr1', 1000, 'A', 'C'),
        organism=organism,
    )

    self.assertSameElements(
        [score.uns['variant_scorer'] for score in scores],
        variant_scorers.get_recommended_scorers(organism.to_proto()),
    )
    mock_stream.assert_called_once()

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
              }),
              uns={
                  'interval': genome.Interval('chr1', 0, 16_384),
                  'variant_scorer': variant_scorers.GeneMaskLFCScorer(
                      requested_output=dna_client.OutputType.RNA_SEQ,
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
      bytes_per_chunk=[0, 1024],
  )
  def test_score_variants(
      self,
      interval: genome.Interval,
      variants: list[genome.Variant],
      scorers: list[dna_client.VariantScorerTypes],
      expected: list[anndata.AnnData],
      organism: dna_client.Organism,
      bytes_per_chunk: int,
  ):
    position_to_expected_scores = {}
    for variant in variants:
      scores = []
      for score in expected:
        score = score.copy()
        score.uns['variant'] = variant
        score.X *= variant.position
        scores.append(score)
      position_to_expected_scores[variant.position] = scores

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        yield from _generate_variant_scoring_protos(
            scores=position_to_expected_scores[request.variant.position],
            bytes_per_chunk=bytes_per_chunk,
            response_proto_type=dna_model_service_pb2.ScoreVariantResponse,
            variant=genome.Variant.from_proto(request.variant),
        )

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)
    all_scores = model.score_variants(
        intervals=interval,
        variants=variants,
        organism=organism,
        variant_scorers=scorers,
    )

    for scores in all_scores:
      self.assertLen(scores, len(expected))
      expected = position_to_expected_scores[scores[0].uns['variant'].position]
      for expected_anndata, output_anndata in zip(expected, scores):
        self._assert_anndata_equal(output_anndata, expected_anndata)

    self.assertLen(variants, mock_stream.call_count)

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
                      }),
                      uns={
                          'interval': genome.Interval('chr1', 0, 16_384),
                          'variant_scorer': variant_scorers.CenterMaskScorer(
                              requested_output=dna_client.OutputType.ATAC,
                              width=501,
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
      bytes_per_chunk=[0, 1024],
  )
  def test_score_ism_variant(
      self,
      interval,
      ism_interval,
      scorers,
      bytes_per_chunk: int,
      example_scores,
      expected_variants,
      interval_variant: genome.Variant | None,
  ):

    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
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

        for variant in ism.ism_variants(ism_interval, ism_sequence):
          yield from _generate_variant_scoring_protos(
              scores=example_scores,
              bytes_per_chunk=bytes_per_chunk,
              response_proto_type=dna_model_service_pb2.ScoreIsmVariantResponse,
              variant=variant,
          )

    mock_channel, _ = _create_mock_channel_and_stream(_mock_generate)

    model = dna_client.DnaClient(channel=mock_channel)
    ism_scores = model.score_ism_variants(
        interval=interval,
        ism_interval=ism_interval,
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

  def test_missing_chunk_raises_error(self):
    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        for response in _generate_prediction_protos(
            _create_example_model_predictions(
                sequence_length=len(request.sequence)
            ),
            request.requested_outputs,
            bytes_per_chunk=128,
            response_proto_type=dna_model_service_pb2.PredictSequenceResponse,
        ):
          if response.WhichOneof('payload') != 'tensor_chunk':
            yield response

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    with self.assertRaisesRegex(
        ValueError, 'Expected tensor_chunk, got "output" payload'
    ):
      _ = model.predict_sequence(
          sequence='A' * 16_384,
          requested_outputs=[
              dna_client.OutputType.ATAC,
              dna_client.OutputType.RNA_SEQ,
          ],
          ontology_terms=None,
      )

    mock_stream.assert_called_once()

  def test_missing_end_of_stream_chunk_raises_error(self):
    def _mock_generate(requests, metadata):
      del metadata
      for request in requests:
        for response in _generate_prediction_protos(
            _create_example_model_predictions(
                sequence_length=len(request.sequence)
            ),
            request.requested_outputs,
            bytes_per_chunk=128,
            response_proto_type=dna_model_service_pb2.PredictSequenceResponse,
        ):
          if response.WhichOneof('payload') != 'tensor_chunk':
            yield response

    mock_channel, mock_stream = _create_mock_channel_and_stream(_mock_generate)
    model = dna_client.DnaClient(channel=mock_channel)

    with self.assertRaisesRegex(
        ValueError, 'Expected tensor_chunk, got end of stream'
    ):
      _ = model.predict_sequence(
          sequence='A' * 16_384,
          requested_outputs=[dna_client.OutputType.ATAC],
          ontology_terms=None,
      )

    mock_stream.assert_called_once()


class RpcRetryTest(parameterized.TestCase):

  def test_retry_rpc(self):
    mock_channel, _ = _create_mock_channel_and_stream([
        _FakeRpcError(grpc.StatusCode.UNAVAILABLE),
        'bar',
    ])

    @functools.partial(dna_client.retry_rpc, initial_backoff=0.01)
    def foo(channel):
      return channel.stream_stream('method')(iter([]))

    self.assertEqual(foo(mock_channel), 'bar')

  @parameterized.named_parameters(
      {
          'testcase_name': 'normal',
          'initial_backoff': 1.25,
          'backoff_multiplier': 1.5,
          'max_attempts': 5,
          'expected_backoff': [1.25, 1.875, 2.8125, 4.21875],
      },
      {
          'testcase_name': 'no_backoff_multiplier',
          'initial_backoff': 1,
          'backoff_multiplier': 1.0,
          'max_attempts': 2,
          'expected_backoff': [1.0],
      },
      {
          'testcase_name': 'no_retries',
          'initial_backoff': 1,
          'backoff_multiplier': 1.0,
          'max_attempts': 1,
          'expected_backoff': [],
      },
  )
  @mock.patch.object(time, 'sleep', autospec=True)
  def test_retry_fails_after_max_attempts(
      self,
      mock_sleep: mock.Mock,
      initial_backoff: float,
      backoff_multiplier: float,
      max_attempts: int,
      expected_backoff: Sequence[float],
  ):
    mock_channel, _ = _create_mock_channel_and_stream(
        _FakeRpcError(grpc.StatusCode.UNAVAILABLE)
    )

    @functools.partial(
        dna_client.retry_rpc,
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
        backoff_multiplier=backoff_multiplier,
        jitter=0.0,
    )
    def function(channel: grpc.Channel):
      return dna_model_service_pb2_grpc.DnaModelServiceStub(
          channel
      ).PredictSequence(iter([dna_model_service_pb2.PredictSequenceRequest()]))

    with self.assertRaises(grpc.RpcError):
      _ = function(mock_channel)

    mock_sleep.assert_has_calls(
        [mock.call(backoff) for backoff in expected_backoff]
    )

  def test_invalid_backoff_multiplier_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, 'Backoff multiplier must be >= 1.0.'
    ):

      @functools.partial(dna_client.retry_rpc, backoff_multiplier=0.5)
      def foo():
        return 'bar'

      _ = foo()


if __name__ == '__main__':
  absltest.main()
