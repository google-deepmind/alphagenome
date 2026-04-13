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

import anndata
from absl.testing import absltest, parameterized

from alphagenome.data import genome, ontology
from alphagenome.models import dna_model, dna_output
from alphagenome.models import interval_scorers as interval_scorers_lib
from alphagenome.models import variant_scorers as variant_scorers_lib


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
      output_metadata: Callable[..., dna_output.OutputMetadata] | None = None,
  ):
    self._predict_sequence_callable = predict_sequence
    self._predict_interval_callable = predict_interval
    self._predict_variant_callable = predict_variant
    self._score_interval_callable = score_interval
    self._score_variant_callable = score_variant
    self._output_metadata_callable = output_metadata
    self._score_ism_variants_callable = score_ism_variants

  def predict_sequence(
      self,
      sequence: str,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
      interval: genome.Interval | None = None,
  ) -> dna_output.Output:
    if self._predict_sequence_callable is None:
      raise NotImplementedError('predict_sequence is not implemented.')
    return self._predict_sequence_callable(
        sequence,
        organism=organism,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
        interval=interval,
    )

  def predict_interval(
      self,
      interval: genome.Interval,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
  ) -> dna_output.Output:
    if self._predict_interval_callable is None:
      raise NotImplementedError('predict_interval is not implemented.')
    return self._predict_interval_callable(
        interval,
        organism=organism,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
    )

  def predict_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
  ) -> dna_output.VariantOutput:
    if self._predict_variant_callable is None:
      raise NotImplementedError('predict_variant is not implemented.')
    return self._predict_variant_callable(
        interval,
        variant,
        organism=organism,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
    )

  def score_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
  ) -> list[anndata.AnnData]:
    if self._score_variant_callable is None:
      raise NotImplementedError('score_variant is not implemented.')
    return self._score_variant_callable(
        interval,
        variant,
        variant_scorers=variant_scorers,
        organism=organism,
    )

  def score_interval(
      self,
      interval: genome.Interval,
      interval_scorers: Sequence[interval_scorers_lib.IntervalScorerTypes] = (),
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
  ) -> list[anndata.AnnData]:
    if self._score_interval_callable is None:
      raise NotImplementedError('score_interval is not implemented.')
    return self._score_interval_callable(
        interval,
        interval_scorers=interval_scorers,
        organism=organism,
    )

  def output_metadata(
      self, organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS
  ) -> dna_output.OutputMetadata:
    if self._output_metadata_callable is None:
      raise NotImplementedError('output_metadata is not implemented.')
    return self._output_metadata_callable(organism)

  def score_ism_variants(
      self,
      interval: genome.Interval,
      ism_interval: genome.Interval,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      interval_variant: genome.Variant | None = None,
  ) -> list[list[anndata.AnnData]]:
    if self._score_ism_variants_callable is None:
      raise NotImplementedError('score_ism_variants is not implemented.')
    return self._score_ism_variants_callable(
        interval,
        ism_interval,
        variant_scorers=variant_scorers,
        organism=organism,
        interval_variant=interval_variant,
    )


class DnaModelTest(parameterized.TestCase):

  def test_predict_sequences(self):
    mock_predict_sequence = mock.Mock(return_value=dna_output.Output())
    model = _MockDnaModel(predict_sequence=mock_predict_sequence)

    expected_sequences = [letter * 10 for letter in 'ACGT']

    model.predict_sequences(
        sequences=expected_sequences,
        requested_outputs=[dna_output.OutputType.RNA_SEQ],
        ontology_terms=None,
        progress_bar=False,
    )
    for sequence in expected_sequences:
      mock_predict_sequence.assert_any_call(
          sequence,
          organism=dna_model.Organism.HOMO_SAPIENS,
          requested_outputs=[dna_output.OutputType.RNA_SEQ],
          ontology_terms=None,
          interval=None,
      )

  def test_predict_intervals(self):
    mock_predict_interval = mock.Mock(return_value=dna_output.Output())
    model = _MockDnaModel(predict_interval=mock_predict_interval)

    expected_intervals = [
        genome.Interval(chromosome, 0, 1000)
        for chromosome in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6']
    ]

    model.predict_intervals(
        intervals=expected_intervals,
        requested_outputs=[dna_output.OutputType.ATAC],
        ontology_terms=None,
        organism=dna_model.Organism.MUS_MUSCULUS,
        progress_bar=False,
    )
    for interval in expected_intervals:
      mock_predict_interval.assert_any_call(
          interval,
          organism=dna_model.Organism.MUS_MUSCULUS,
          requested_outputs=[dna_output.OutputType.ATAC],
          ontology_terms=None,
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
        requested_outputs=[dna_output.OutputType.CAGE],
        ontology_terms=[ontology.from_curie('UBERON:00001')],
        progress_bar=False,
    )
    for variant, interval in zip(expected_variants, expected_intervals):
      mock_predict_variant.assert_any_call(
          interval,
          variant,
          organism=dna_model.Organism.HOMO_SAPIENS,
          requested_outputs=[dna_output.OutputType.CAGE],
          ontology_terms=[ontology.from_curie('UBERON:00001')],
      )

  def test_score_intervals(self):
    mock_score_interval = mock.Mock(return_value=[anndata.AnnData()])
    model = _MockDnaModel(score_interval=mock_score_interval)

    expected_intervals = [
        genome.Interval(chromosome, 0, 1000)
        for chromosome in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6']
    ]

    model.score_intervals(
        intervals=expected_intervals,
        interval_scorers=list(
            interval_scorers_lib.RECOMMENDED_INTERVAL_SCORERS.values()
        ),
        progress_bar=False,
    )
    for interval in expected_intervals:
      mock_score_interval.assert_any_call(
          interval,
          interval_scorers=list(
              interval_scorers_lib.RECOMMENDED_INTERVAL_SCORERS.values()
          ),
          organism=dna_model.Organism.HOMO_SAPIENS,
      )

  @parameterized.product(single_interval=[True, False])
  def test_score_variants(self, single_interval: bool):
    mock_score_variant = mock.Mock(return_value=[anndata.AnnData()])
    model = _MockDnaModel(score_variant=mock_score_variant)

    expected_variants = [
        genome.Variant('chr1', i * 2048, 'A', 'C') for i in range(1, 10)
    ]

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
        variants=expected_variants,
        variant_scorers=list(
            variant_scorers_lib.RECOMMENDED_VARIANT_SCORERS.values()
        ),
        progress_bar=False,
    )
    for variant, interval in zip(expected_variants, expected_intervals):
      mock_score_variant.assert_any_call(
          interval,
          variant,
          variant_scorers=list(
              variant_scorers_lib.RECOMMENDED_VARIANT_SCORERS.values()
          ),
          organism=dna_model.Organism.HOMO_SAPIENS,
      )


if __name__ == '__main__':
  absltest.main()
