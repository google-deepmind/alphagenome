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

from collections.abc import Callable, Iterable, Sequence
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome
from alphagenome.models.v2 import dna_model
from alphagenome.models.v2 import dna_output
from alphagenome.models.v2 import interval_scorers as interval_scorers_lib
from alphagenome.models.v2 import model_condition
from alphagenome.models.v2 import variant_scorers as variant_scorers_lib
import anndata
import numpy as np

_SEQUENCE_LENGTH = 2**20


class _MockDnaModel(dna_model.DnaModel):
  """Mock DNA model for testing."""

  def __init__(
      self,
      *,
      predict_sequence: Callable[..., dna_output.Output] | None = None,
      predict_interval: Callable[..., dna_output.Output] | None = None,
      predict_variant: Callable[..., dna_output.VariantOutput] | None = None,
      score_interval: Callable[..., list[anndata.AnnData]] | None = None,
      score_variant: Callable[..., list[anndata.AnnData]] | None = None,
      score_ism_variants: (
          Callable[..., list[list[anndata.AnnData]]] | None
      ) = None,
      observed_metadata: Callable[..., dna_output.OutputMetadata] | None = None,
      encode_cells: Callable[..., np.ndarray] | None = None,
      predict_interval_gradients: (
          Callable[..., dna_output.InputGradientsOutput] | None
      ) = None,
  ):
    self._predict_sequence_callable = predict_sequence
    self._predict_interval_callable = predict_interval
    self._predict_variant_callable = predict_variant
    self._score_interval_callable = score_interval
    self._score_variant_callable = score_variant
    self._observed_metadata_callable = observed_metadata
    self._score_ism_variants_callable = score_ism_variants
    self._encode_cells_callable = encode_cells
    self._predict_interval_gradients_callable = predict_interval_gradients

  def predict_sequence(
      self,
      sequence: str,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      interval: genome.Interval | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> dna_output.Output:
    if self._predict_sequence_callable is None:
      raise NotImplementedError('predict_sequence is not implemented.')
    return self._predict_sequence_callable(
        sequence,
        organism=organism,
        requested_outputs=requested_outputs,
        condition=condition,
        interval=interval,
        cell_conditioning=cell_conditioning,
        cell_condition_name=cell_condition_name,
    )

  def predict_interval(
      self,
      interval: genome.Interval,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> dna_output.Output:
    if self._predict_interval_callable is None:
      raise NotImplementedError('predict_interval is not implemented.')
    return self._predict_interval_callable(
        interval,
        organism=organism,
        requested_outputs=requested_outputs,
        condition=condition,
        cell_conditioning=cell_conditioning,
        cell_condition_name=cell_condition_name,
    )

  def predict_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> dna_output.VariantOutput:
    if self._predict_variant_callable is None:
      raise NotImplementedError('predict_variant is not implemented.')
    return self._predict_variant_callable(
        interval,
        variant,
        organism=organism,
        requested_outputs=requested_outputs,
        condition=condition,
        cell_conditioning=cell_conditioning,
        cell_condition_name=cell_condition_name,
    )

  def score_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      condition: model_condition.ModelCondition | str | None = None,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> list[anndata.AnnData]:
    if self._score_variant_callable is None:
      raise NotImplementedError('score_variant is not implemented.')
    return self._score_variant_callable(
        interval,
        variant,
        condition=condition,
        variant_scorers=variant_scorers,
        organism=organism,
        cell_conditioning=cell_conditioning,
        cell_condition_name=cell_condition_name,
    )

  def score_interval(
      self,
      interval: genome.Interval,
      condition: model_condition.ModelCondition | str | None = None,
      interval_scorers: Sequence[interval_scorers_lib.IntervalScorerTypes] = (),
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> list[anndata.AnnData]:
    if self._score_interval_callable is None:
      raise NotImplementedError('score_interval is not implemented.')
    return self._score_interval_callable(
        interval,
        condition=condition,
        interval_scorers=interval_scorers,
        organism=organism,
        cell_conditioning=cell_conditioning,
        cell_condition_name=cell_condition_name,
    )

  def observed_metadata(
      self, organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS
  ) -> dna_output.OutputMetadata:
    if self._observed_metadata_callable is None:
      raise NotImplementedError('observed_metadata is not implemented.')
    return self._observed_metadata_callable(organism)

  def score_ism_variants(
      self,
      interval: genome.Interval,
      ism_interval: genome.Interval,
      condition: model_condition.ModelCondition | str | None = None,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      interval_variant: genome.Variant | None = None,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> list[list[anndata.AnnData]]:
    if self._score_ism_variants_callable is None:
      raise NotImplementedError('score_ism_variants is not implemented.')
    return self._score_ism_variants_callable(
        interval,
        ism_interval,
        condition=condition,
        variant_scorers=variant_scorers,
        interval_variant=interval_variant,
        organism=organism,
        cell_conditioning=cell_conditioning,
        cell_condition_name=cell_condition_name,
    )

  def encode_cells(
      self,
      cells: anndata.AnnData,
  ) -> np.ndarray:
    if self._encode_cells_callable is None:
      raise NotImplementedError('encode_cells is not implemented.')
    return self._encode_cells_callable(cells)

  def predict_interval_gradients(
      self,
      gene_id: str,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_output: dna_output.OutputType,
      sequence_length: int,
      condition: model_condition.ModelCondition | str | None = None,
      cell_conditioning: anndata.AnnData | None = None,
      cell_condition_name: str | None = None,
  ) -> dna_output.InputGradientsOutput:
    if self._predict_interval_gradients_callable is None:
      raise NotImplementedError(
          'predict_interval_gradients is not implemented.'
      )
    return self._predict_interval_gradients_callable(
        gene_id,
        organism=organism,
        requested_output=requested_output,
        sequence_length=sequence_length,
        condition=condition,
        cell_conditioning=cell_conditioning,
        cell_condition_name=cell_condition_name,
    )


class DnaModelTest(parameterized.TestCase):

  def test_predict_sequences(self):
    mock_predict_sequence = mock.Mock(return_value=dna_output.Output())
    model = _MockDnaModel(predict_sequence=mock_predict_sequence)

    expected_sequences = [letter * 10 for letter in 'ACGT']

    model.predict_sequences(
        sequences=expected_sequences,
        requested_outputs=[dna_output.OutputType.RNA_SEQ],
        conditions=['UBERON:0000005_CL:0000001'] * 4,
        progress_bar=False,
    )
    for sequence in expected_sequences:
      mock_predict_sequence.assert_any_call(
          sequence,
          organism=dna_model.Organism.HOMO_SAPIENS,
          requested_outputs=[dna_output.OutputType.RNA_SEQ],
          condition='UBERON:0000005_CL:0000001',
          interval=None,
          cell_conditioning=None,
          cell_condition_name=None,
      )

  def test_predict_intervals(self):
    mock_predict_interval = mock.Mock(return_value=dna_output.Output())
    model = _MockDnaModel(predict_interval=mock_predict_interval)

    expected_intervals = [
        genome.Interval(chromosome, 0, 1000)
        for chromosome in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6']
    ]

    expected_conditions = [
        'UBERON:0000005_CL:0000001',
        'UBERON:0000005',
    ] * 3

    model.predict_intervals(
        intervals=expected_intervals,
        requested_outputs=[dna_output.OutputType.ATAC],
        conditions=expected_conditions,
        organism=dna_model.Organism.MUS_MUSCULUS,
        progress_bar=False,
    )
    for interval, condition in zip(
        expected_intervals, expected_conditions, strict=True
    ):
      mock_predict_interval.assert_any_call(
          interval,
          organism=dna_model.Organism.MUS_MUSCULUS,
          requested_outputs=[dna_output.OutputType.ATAC],
          condition=condition,
          cell_conditioning=None,
          cell_condition_name=None,
      )

  @parameterized.product(single_interval=[True, False])
  def test_predict_variants(self, single_interval: bool):
    mock_predict_variant = mock.Mock(
        return_value=dna_output.VariantOutput(
            reference=dna_output.Output(), alternate=dna_output.Output()
        )
    )
    model = _MockDnaModel(predict_variant=mock_predict_variant)

    expected_variants = [
        genome.Variant('chr1', i * 2048, 'A', 'C') for i in range(1, 10)
    ]
    expected_conditions = [
        'UBERON:0000005_CL:0000001',
        'UBERON:0000005',
        'UBERON:0000005_CL:0000002',
    ] * 3

    if single_interval:
      expected_intervals = [
          expected_variants[0].reference_interval.resize(2048)
      ] * len(expected_variants)
    else:
      expected_intervals = [
          variant.reference_interval.resize(2048)
          for variant in expected_variants
      ]

    model.predict_variants(
        intervals=expected_intervals[0]
        if single_interval
        else expected_intervals,
        variants=expected_variants,
        conditions=expected_conditions,
        requested_outputs=[dna_output.OutputType.CAGE],
        progress_bar=False,
    )
    for variant, interval, condition in zip(
        expected_variants, expected_intervals, expected_conditions
    ):
      mock_predict_variant.assert_any_call(
          interval,
          variant,
          organism=dna_model.Organism.HOMO_SAPIENS,
          requested_outputs=[dna_output.OutputType.CAGE],
          condition=condition,
          cell_conditioning=None,
          cell_condition_name=None,
      )

  def test_score_intervals(self):
    mock_score_interval = mock.Mock(return_value=[anndata.AnnData()])
    model = _MockDnaModel(score_interval=mock_score_interval)

    expected_intervals = [
        genome.Interval(chromosome, 0, 1000)
        for chromosome in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6']
    ]
    expected_conditions = [
        'UBERON:0000005_CL:0000001',
        'UBERON:0000005',
    ] * 3

    model.score_intervals(
        intervals=expected_intervals,
        conditions=expected_conditions,
        interval_scorers=list(
            interval_scorers_lib.RECOMMENDED_INTERVAL_SCORERS.values()
        ),
        progress_bar=False,
    )
    for interval, condition in zip(
        expected_intervals, expected_conditions, strict=True
    ):
      mock_score_interval.assert_any_call(
          interval,
          condition=condition,
          interval_scorers=list(
              interval_scorers_lib.RECOMMENDED_INTERVAL_SCORERS.values()
          ),
          organism=dna_model.Organism.HOMO_SAPIENS,
          cell_conditioning=None,
          cell_condition_name=None,
      )

  @parameterized.product(single_interval=[True, False])
  def test_score_variants(self, single_interval: bool):
    mock_score_variant = mock.Mock(return_value=[anndata.AnnData()])
    model = _MockDnaModel(score_variant=mock_score_variant)

    expected_variants = [
        genome.Variant('chr1', i * 2048, 'A', 'C') for i in range(1, 10)
    ]
    expected_conditions = [
        'UBERON:0000005_CL:0000001',
        'UBERON:0000005',
        'UBERON:0000005_CL:0000002',
    ] * 3

    if single_interval:
      expected_intervals = [
          expected_variants[0].reference_interval.resize(2048)
      ] * len(expected_variants)
    else:
      expected_intervals = [
          variant.reference_interval.resize(2048)
          for variant in expected_variants
      ]

    model.score_variants(
        intervals=expected_intervals[0]
        if single_interval
        else expected_intervals,
        conditions=expected_conditions,
        variants=expected_variants,
        variant_scorers=list(
            variant_scorers_lib.RECOMMENDED_VARIANT_SCORERS.values()
        ),
        progress_bar=False,
    )
    for variant, interval, condition in zip(
        expected_variants, expected_intervals, expected_conditions
    ):
      mock_score_variant.assert_any_call(
          interval,
          variant,
          condition=condition,
          variant_scorers=list(
              variant_scorers_lib.RECOMMENDED_VARIANT_SCORERS.values()
          ),
          organism=dna_model.Organism.HOMO_SAPIENS,
          cell_conditioning=None,
          cell_condition_name=None,
      )

  def test_batch_methods_cell_conditioning(self):
    cell_a = anndata.AnnData(np.ones((2, 3), dtype=np.float32))
    cell_b = anndata.AnnData(np.ones((2, 3), dtype=np.float32) * 2)
    cell_conds = [cell_a, cell_b]
    cell_names = ['cell_a', 'cell_b']
    intervals = [
        genome.Interval('chr1', 0, 2048),
        genome.Interval('chr2', 0, 2048),
    ]
    variants = [
        genome.Variant('chr1', 500, 'A', 'C'),
        genome.Variant('chr2', 500, 'G', 'T'),
    ]

    mock_seq = mock.Mock(return_value=dna_output.Output())
    mock_int = mock.Mock(return_value=dna_output.Output())
    mock_var = mock.Mock(
        return_value=dna_output.VariantOutput(
            reference=dna_output.Output(), alternate=dna_output.Output()
        )
    )
    mock_score_int = mock.Mock(return_value=[anndata.AnnData()])
    mock_score_var = mock.Mock(return_value=[anndata.AnnData()])
    mock_grads = mock.Mock(
        return_value=mock.MagicMock(spec=dna_output.InputGradientsOutput)
    )
    model = _MockDnaModel(
        predict_sequence=mock_seq,
        predict_interval=mock_int,
        predict_variant=mock_var,
        score_interval=mock_score_int,
        score_variant=mock_score_var,
        predict_interval_gradients=mock_grads,
    )

    model.predict_sequences(
        sequences=['A' * 16, 'C' * 16],
        requested_outputs=[dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED],
        cell_conditionings=cell_conds,
        cell_condition_names=cell_names,
        progress_bar=False,
    )
    model.predict_intervals(
        intervals=intervals,
        requested_outputs=[dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED],
        cell_conditionings=cell_conds,
        cell_condition_names=cell_names,
        progress_bar=False,
    )
    model.predict_variants(
        intervals=intervals[0],
        variants=variants,
        requested_outputs=[dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED],
        cell_conditionings=cell_conds,
        cell_condition_names=cell_names,
        progress_bar=False,
    )
    model.score_intervals(
        intervals=intervals,
        cell_conditionings=cell_conds,
        cell_condition_names=cell_names,
        progress_bar=False,
    )
    model.score_variants(
        intervals=intervals[0],
        variants=variants,
        cell_conditionings=cell_conds,
        cell_condition_names=cell_names,
        progress_bar=False,
    )
    model.predict_interval_gradients_batch(
        gene_ids='ENSG00000119866',
        requested_output=dna_output.OutputType.SC_GENE_EXPRESSION_DENOISED,
        sequence_length=_SEQUENCE_LENGTH,
        cell_conditionings=cell_conds,
        cell_condition_names=cell_names,
        progress_bar=False,
    )

    for mock_fn in (
        mock_seq,
        mock_int,
        mock_var,
        mock_score_int,
        mock_score_var,
        mock_grads,
    ):
      self.assertEqual(mock_fn.call_count, 2)
      by_name = {
          c.kwargs['cell_condition_name']: c.kwargs
          for c in mock_fn.call_args_list
      }
      self.assertCountEqual(by_name.keys(), ['cell_a', 'cell_b'])
      self.assertIsNone(by_name['cell_a']['condition'])
      self.assertIs(by_name['cell_a']['cell_conditioning'], cell_a)
      self.assertIsNone(by_name['cell_b']['condition'])
      self.assertIs(by_name['cell_b']['cell_conditioning'], cell_b)

  def test_batch_methods_reject_bare_str(self):
    model = _MockDnaModel()
    interval = genome.Interval('chr1', 0, 2048)
    variant = genome.Variant('chr1', 500, 'A', 'C')
    outputs = [dna_output.OutputType.RNA_SEQ]

    with self.assertRaisesRegex(
        TypeError, '`sequences` must be a sequence, not a single str'
    ):
      model.predict_sequences(
          sequences='ACGT',
          conditions=['CL:0000765'] * 4,
          requested_outputs=outputs,
          progress_bar=False,
      )

    for call in (
        lambda **kw: model.predict_sequences(
            sequences=['ACGT'], requested_outputs=outputs, **kw
        ),
        lambda **kw: model.predict_intervals(
            intervals=[interval], requested_outputs=outputs, **kw
        ),
        lambda **kw: model.predict_variants(
            intervals=[interval],
            variants=[variant],
            requested_outputs=outputs,
            **kw,
        ),
        lambda **kw: model.score_intervals(intervals=[interval], **kw),
        lambda **kw: model.score_variants(
            intervals=[interval], variants=[variant], **kw
        ),
        lambda **kw: model.predict_interval_gradients_batch(
            gene_ids='ENSG1',
            requested_output=dna_output.OutputType.RNA_SEQ,
            sequence_length=_SEQUENCE_LENGTH,
            **kw,
        ),
    ):
      with self.assertRaisesRegex(
          TypeError, '`conditions` must be a sequence, not a single str'
      ):
        call(conditions='CL:0000765', progress_bar=False)
      with self.assertRaisesRegex(
          TypeError,
          '`cell_condition_names` must be a sequence, not a single str',
      ):
        call(cell_condition_names='cell_a', progress_bar=False)

  def test_batch_methods_mismatch_lengths(self):
    model = _MockDnaModel()
    intervals = [
        genome.Interval('chr1', 0, 2048),
        genome.Interval('chr2', 0, 2048),
    ]
    variants = [
        genome.Variant('chr1', 500, 'A', 'C'),
        genome.Variant('chr2', 500, 'G', 'T'),
    ]
    cell_a = anndata.AnnData(np.ones((2, 3), dtype=np.float32))
    outputs = [dna_output.OutputType.RNA_SEQ]

    with self.assertRaisesRegex(
        ValueError, 'Sequences and intervals must have the same length'
    ):
      model.predict_sequences(
          sequences=['AAAA', 'CCCC'],
          conditions=['CL:0000765', 'CL:0000236'],
          intervals=[intervals[0]],
          requested_outputs=outputs,
          progress_bar=False,
      )
    with self.assertRaisesRegex(
        ValueError, 'Intervals and variants must have the same length'
    ):
      model.score_variants(
          intervals=intervals,
          variants=[variants[0]],
          conditions=['CL:0000765', 'CL:0000236'],
          progress_bar=False,
      )

    for prefix, call in (
        (
            'Sequences',
            lambda **kw: model.predict_sequences(
                sequences=['AAAA', 'CCCC'], requested_outputs=outputs, **kw
            ),
        ),
        (
            'Intervals',
            lambda **kw: model.predict_intervals(
                intervals=intervals, requested_outputs=outputs, **kw
            ),
        ),
        (
            'Intervals',
            lambda **kw: model.predict_variants(
                intervals=intervals,
                variants=variants,
                requested_outputs=outputs,
                **kw,
            ),
        ),
        (
            'Intervals',
            lambda **kw: model.score_intervals(intervals=intervals, **kw),
        ),
        (
            'Intervals',
            lambda **kw: model.score_variants(
                intervals=intervals, variants=variants, **kw
            ),
        ),
        (
            'gene_ids',
            lambda **kw: model.predict_interval_gradients_batch(
                gene_ids=['ENSG1', 'ENSG2'],
                requested_output=dna_output.OutputType.SC_GENE_EXPRESSION,
                sequence_length=_SEQUENCE_LENGTH,
                **kw,
            ),
        ),
    ):
      with self.assertRaisesRegex(
          ValueError, f'{prefix} and conditions must have the same length'
      ):
        call(conditions=['CL:0000765'], progress_bar=False)
      with self.assertRaisesRegex(
          ValueError,
          f'{prefix} and cell_conditionings must have the same length',
      ):
        call(
            cell_conditionings=[cell_a],
            cell_condition_names=['cell_a', 'cell_b'],
            progress_bar=False,
        )
      with self.assertRaisesRegex(
          ValueError,
          f'{prefix} and cell_condition_names must have the same length',
      ):
        call(
            cell_conditionings=[cell_a, cell_a],
            cell_condition_names=['cell_a'],
            progress_bar=False,
        )

  @parameterized.product(single_gene=[True, False])
  def test_predict_interval_gradients_batch(self, single_gene: bool):
    conditions = ['CL:0000765', 'CL:0000236', 'CL:0000084']
    gene_ids: str | list[str] = (
        'ENSG00000119866'
        if single_gene
        else ['ENSG00000119866', 'ENSG00000167978', 'ENSG00000119866']
    )
    expected_genes = (
        ['ENSG00000119866'] * 3
        if single_gene
        else ['ENSG00000119866', 'ENSG00000167978', 'ENSG00000119866']
    )
    outputs_by_call = {
        (gid, cond): mock.MagicMock(spec=dna_output.InputGradientsOutput)
        for gid, cond in zip(expected_genes, conditions, strict=True)
    }

    def _side_effect(gene_id: str, **kwargs):
      return outputs_by_call[(gene_id, kwargs['condition'])]

    mock_predict_grads = mock.Mock(side_effect=_side_effect)
    model = _MockDnaModel(predict_interval_gradients=mock_predict_grads)

    outputs = model.predict_interval_gradients_batch(
        gene_ids=gene_ids,
        requested_output=dna_output.OutputType.SC_GENE_EXPRESSION,
        sequence_length=_SEQUENCE_LENGTH,
        conditions=conditions,
        progress_bar=False,
    )
    expected_outputs = [
        outputs_by_call[(gid, cond)]
        for gid, cond in zip(expected_genes, conditions, strict=True)
    ]
    self.assertEqual(outputs, expected_outputs)
    for gid, cond in zip(expected_genes, conditions, strict=True):
      mock_predict_grads.assert_any_call(
          gid,
          organism=dna_model.Organism.HOMO_SAPIENS,
          requested_output=dna_output.OutputType.SC_GENE_EXPRESSION,
          sequence_length=_SEQUENCE_LENGTH,
          condition=cond,
          cell_conditioning=None,
          cell_condition_name=None,
      )


if __name__ == '__main__':
  absltest.main()
