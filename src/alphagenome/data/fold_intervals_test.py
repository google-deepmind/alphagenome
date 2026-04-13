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

import pandas as pd
from absl.testing import absltest, parameterized

from alphagenome.data import fold_intervals
from alphagenome.models import dna_client

_dummy_intervals = pd.DataFrame({
    'chromosome': ['chr' + str(i) for i in range(8)],
    'start': [i for i in range(0, 8000, 1000)],
    'end': [i + 500 for i in range(0, 8000, 1000)],
    'fold': ['fold' + str(i) for i in range(8)],
})


class FoldIntervalsTest(parameterized.TestCase):
  """Tests for the fold intervals."""

  @parameterized.named_parameters(
      dict(
          testcase_name='train_fold_0',
          model_version=dna_client.ModelVersion.FOLD_0,
          subset=fold_intervals.Subset.TRAIN,
          expected_intervals=_dummy_intervals[
              # Folds 0 and 1 are held out
              _dummy_intervals.fold.isin(
                  ['fold' + str(i) for i in [2, 3, 4, 5, 6, 7]]
              )
          ],
      ),
      dict(
          testcase_name='train_fold_1',
          model_version=dna_client.ModelVersion.FOLD_1,
          subset=fold_intervals.Subset.TRAIN,
          expected_intervals=_dummy_intervals[
              # Folds 3 and 4 are held out (aligns with Borzoi fold 1)
              _dummy_intervals.fold.isin(
                  ['fold' + str(i) for i in [0, 1, 2, 5, 6, 7]]
              )
          ],
      ),
      dict(
          testcase_name='train_all_folds',
          model_version=dna_client.ModelVersion.ALL_FOLDS,
          subset=fold_intervals.Subset.TRAIN,
          expected_intervals=_dummy_intervals,
      ),
      dict(
          testcase_name='valid_fold_0',
          model_version=dna_client.ModelVersion.FOLD_0,
          subset=fold_intervals.Subset.VALID,
          expected_intervals=_dummy_intervals[
              # Model fold 0 uses data fold 0 for validation.
              _dummy_intervals.fold
              == 'fold0'
          ],
      ),
      dict(
          testcase_name='valid_fold_3',
          model_version=dna_client.ModelVersion.FOLD_3,
          subset=fold_intervals.Subset.VALID,
          expected_intervals=_dummy_intervals[
              # Model fold 3 uses data fold 6 for validation.
              _dummy_intervals.fold
              == 'fold6'
          ],
      ),
      dict(
          testcase_name='valid_all_folds',
          model_version=dna_client.ModelVersion.ALL_FOLDS,
          subset=fold_intervals.Subset.VALID,
          expected_intervals=_dummy_intervals[
              # All folds model uses data fold 0 for validation.
              _dummy_intervals.fold
              == 'fold0'
          ],
      ),
      dict(
          testcase_name='test_fold_1',
          model_version=dna_client.ModelVersion.FOLD_1,
          subset=fold_intervals.Subset.TEST,
          expected_intervals=_dummy_intervals[
              # Model fold 1 uses data fold 4 for testing.
              _dummy_intervals.fold
              == 'fold4'
          ],
      ),
      dict(
          testcase_name='test_fold_2',
          model_version=dna_client.ModelVersion.FOLD_2,
          subset=fold_intervals.Subset.TEST,
          expected_intervals=_dummy_intervals[
              # Model fold 4 uses data fold 5 for testing.
              _dummy_intervals.fold
              == 'fold5'
          ],
      ),
      dict(
          testcase_name='test_all_folds',
          model_version=dna_client.ModelVersion.ALL_FOLDS,
          subset=fold_intervals.Subset.TEST,
          expected_intervals=_dummy_intervals[
              # All folds model uses data fold 0 for testing.
              _dummy_intervals.fold
              == 'fold1'
          ],
      ),
  )
  def test_get_fold_intervals(
      self,
      model_version: dna_client.ModelVersion,
      subset: fold_intervals.Subset,
      expected_intervals: pd.DataFrame,
  ):
    path = self.create_tempfile(
        content=_dummy_intervals.to_csv(sep='\t', header=False)
    ).full_path
    intervals = fold_intervals.get_fold_intervals(
        model_version=model_version,
        organism=dna_client.Organism.HOMO_SAPIENS,
        subset=subset,
        example_regions_path=path,
    )
    pd.testing.assert_frame_equal(
        intervals,
        expected_intervals,
    )


if __name__ == '__main__':
  absltest.main()
