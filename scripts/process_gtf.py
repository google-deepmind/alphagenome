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

"""Script to process GTF into feather file."""

import os
import tempfile
from urllib import parse, request

import pandas as pd
import pyranges
from absl import app, flags, logging

from alphagenome.data import transcript as transcript_utils

_GTF_PATH = flags.DEFINE_string(
    'gtf_path', None, 'Path to GTF file.', required=True
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Path to GTF output feather file.', required=True
)
_SPLICE_SITES_OUTPUT_PATH = flags.DEFINE_string(
    'splice_sites_output_path',
    None,
    'Path to splice_sites output feather file.',
)


def generate_splice_sites(
    gtf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Generates tissue-agnostic splice start/end sites from GTF."""
  transcripts = {
      k: transcript_utils.Transcript.from_gtf_df(df)
      for k, df in gtf.groupby('transcript_id')
  }
  junctions = []
  for _, transcript in transcripts.items():
    junctions.extend(
        [x.chromosome, x.start, x.end, x.strand] for x in transcript.introns
    )
  df = pd.DataFrame(
      junctions, columns=['Chromosome', 'Start', 'End', 'Strand']
  ).sort_values(by=['Chromosome', 'Start'])
  df = df.drop_duplicates().reset_index(drop=True)
  return (
      df[['Chromosome', 'Start', 'Strand']],
      df[['Chromosome', 'End', 'Strand']],
  )


def generate_gtf(path: str) -> pd.DataFrame:
  """Generates GTF from a GTF path."""
  url = parse.urlparse(path)
  if all([url.scheme, url.netloc]):
    with tempfile.TemporaryDirectory() as d:
      path, _ = request.urlretrieve(
          path,
          filename=os.path.join(d, os.path.basename(path)),
      )
      logging.info('Downloaded GTF to %s', path)
      gtf = pyranges.read_gtf(path, as_df=True, duplicate_attr=True)
  else:
    gtf = pyranges.read_gtf(path, as_df=True, duplicate_attr=True)

  gtf['gene_id_nopatch'] = gtf['gene_id'].str.split('.', expand=True)[0]
  return gtf


def main(_) -> None:
  if not _OUTPUT_PATH.value.endswith('.feather'):
    raise ValueError('Output path must end with .feather')

  logging.info('Reading GTF from %s', _GTF_PATH.value)
  gtf = generate_gtf(_GTF_PATH.value)

  logging.info('Writing GTF to %s', _OUTPUT_PATH.value)
  gtf.to_feather(_OUTPUT_PATH.value)

  if _SPLICE_SITES_OUTPUT_PATH.value is not None:
    logging.info('Generating splice sites from GTF')
    splice_sites_starts, splice_sites_ends = generate_splice_sites(gtf)

    splice_sites_start_path = (
        _SPLICE_SITES_OUTPUT_PATH.value.removesuffix('.feather')
        + '_starts.feather'
    )
    splice_sites_end_path = (
        _SPLICE_SITES_OUTPUT_PATH.value.removesuffix('.feather')
        + '_ends.feather'
    )
    logging.info('Writing start splice sites to %s', splice_sites_start_path)
    splice_sites_starts.to_feather(splice_sites_start_path)
    logging.info('Writing end splice sites to %s', splice_sites_end_path)
    splice_sites_ends.to_feather(splice_sites_end_path)


if __name__ == '__main__':
  app.run(main)
