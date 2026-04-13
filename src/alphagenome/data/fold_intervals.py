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

"""Genomics intervals used for training model folds."""

import enum

import immutabledict
import pandas as pd

from alphagenome.models import dna_client

_DEFAULT_EXAMPLE_REGIONS = immutabledict.immutabledict({
    dna_client.Organism.HOMO_SAPIENS: (
        'https://github.com/calico/borzoi/raw/'
        '5c9358222b5026abb733ed5fb84f3f6c77239b37/data/sequences_human.bed.gz'
    ),
    dna_client.Organism.MUS_MUSCULUS: (
        'https://github.com/calico/borzoi/raw/'
        '5c9358222b5026abb733ed5fb84f3f6c77239b37/data/sequences_mouse.bed.gz'
    ),
})


class Subset(enum.Enum):
  """Subset of the data."""

  TRAIN = 0
  VALID = 1
  TEST = 2


# Fold ONE is aligned with all trained Borzoi checkpoints: 3 and 4 are held out.
_VALID_FOLD = immutabledict.immutabledict({
    0: 'fold0',
    1: 'fold3',
    2: 'fold2',
    3: 'fold6',
    -1: 'fold0',
})

_TEST_FOLD = immutabledict.immutabledict({
    0: 'fold1',
    1: 'fold4',
    2: 'fold5',
    3: 'fold7',
    -1: 'fold1',
})

_MODEL_VERSION_TO_FOLD = immutabledict.immutabledict({
    dna_client.ModelVersion.FOLD_0: 0,
    dna_client.ModelVersion.FOLD_1: 1,
    dna_client.ModelVersion.FOLD_2: 2,
    dna_client.ModelVersion.FOLD_3: 3,
    dna_client.ModelVersion.ALL_FOLDS: -1,
})


def get_all_folds() -> list[str]:
  """Returns the names of all data folds."""
  return [f'fold{i}' for i in range(8)]


def get_fold_names(
    model_version: dna_client.ModelVersion, subset: Subset
) -> list[str]:
  """Returns the names of the folds for a given model version and subset."""
  match subset:
    case Subset.VALID:
      return [_VALID_FOLD[_MODEL_VERSION_TO_FOLD[model_version]]]
    case Subset.TEST:
      return [_TEST_FOLD[_MODEL_VERSION_TO_FOLD[model_version]]]
    case Subset.TRAIN:
      all_folds = get_all_folds()
      if _MODEL_VERSION_TO_FOLD[model_version] == -1:
        return all_folds
      remove_folds = get_fold_names(
          model_version, Subset.VALID
      ) + get_fold_names(model_version, Subset.TEST)
      for fold in remove_folds:
        all_folds.remove(fold)
      return all_folds
    case _:
      raise ValueError(f'Unknown {subset=}')


def get_fold_intervals(
    model_version: dna_client.ModelVersion,
    organism: dna_client.Organism,
    subset: Subset,
    example_regions_path: str | None = None,
) -> pd.DataFrame:
  """Returns the intervals for a given model version and subset."""
  if example_regions_path is None:
    example_regions_path = _DEFAULT_EXAMPLE_REGIONS[organism]

  example_regions = pd.read_csv(
      example_regions_path,
      sep='\t',
      names=['chromosome', 'start', 'end', 'fold'],
  )
  return example_regions[
      example_regions.fold.isin(get_fold_names(model_version, subset))
  ]
