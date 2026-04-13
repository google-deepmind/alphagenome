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

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from . import process_gtf

_EXAMPLE_GTF = r"""
chr1	FOO	gene	12100	21316	.	+	.	gene_id "GENE00000000001.1";
chr1	FOO	transcript	12100	18244	.	+	.	gene_id "GENE00000000001.1"; transcript_id "TRANS00000000001.2";
chr1	FOO	exon	12100	14148	.	+	.	gene_id "GENE00000000001.1"; transcript_id "TRANS00000000001.2";
chr1	FOO	exon	16196	17220	.	+	.	gene_id "GENE00000000001.1"; transcript_id "TRANS00000000001.2";
"""

_PROCESSED_GTF = pd.DataFrame({
    'Chromosome': 'chr1',
    'Start': [12099, 12099, 12099, 16195],
    'End': [21316, 18244, 14148, 17220],
    'Strand': ['+', '+', '+', '+'],
    'gene_id': 'GENE00000000001.1',
    'gene_id_nopatch': 'GENE00000000001',
    'transcript_id': [
        np.nan,
        'TRANS00000000001.2',
        'TRANS00000000001.2',
        'TRANS00000000001.2',
    ],
    'Feature': ['gene', 'transcript', 'exon', 'exon'],
})


class ProcessGtfTest(parameterized.TestCase):

  def test_process_gtf(self):
    temp_file = self.create_tempfile()
    temp_file.write_text(_EXAMPLE_GTF)
    gtf = process_gtf.generate_gtf(temp_file.full_path)[_PROCESSED_GTF.columns]
    pd.testing.assert_frame_equal(
        gtf, _PROCESSED_GTF, check_categorical=False, check_dtype=False
    )

  def test_generate_splice_sites(self):
    starts, ends = process_gtf.generate_splice_sites(_PROCESSED_GTF)
    expected = pd.DataFrame({
        'Chromosome': 'chr1',
        'Start': [14148],
        'End': [16195],
        'Strand': ['+'],
    })
    pd.testing.assert_frame_equal(
        starts, expected[['Chromosome', 'Start', 'Strand']]
    )
    pd.testing.assert_frame_equal(
        ends, expected[['Chromosome', 'End', 'Strand']]
    )


if __name__ == '__main__':
  absltest.main()
