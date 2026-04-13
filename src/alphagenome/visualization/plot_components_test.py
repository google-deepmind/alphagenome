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

import matplotlib
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from alphagenome.data import genome, junction_data, track_data
from alphagenome.data import transcript as transcript_utils
from alphagenome.visualization import plot_components

_ATAC_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar', 'baz', 'buz', 'fux'],
        strand=['.'] * 5,
        padding=[False] * 5,
    )
)

_CONTACT_MAP_METADATA = pd.DataFrame(
    dict(
        name=['H1', 'HFF'],
        strand=['.'] * 2,
        padding=[False] * 2,
    )
)

_RNA_SEQ_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'boo', 'bar', 'far'],
        strand=['.'] * 4,
        padding=[False] * 4,
    )
)

_SPLICING_METADATA = pd.DataFrame(
    dict(
        name=['foo', 'bar'],
        ontology_curie=['ontology_curie_1', 'ontology_curie_2'],
        padding=[False] * 2,
    )
)


class PlotComponentsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(subinterval=genome.Interval('chr1', 100, 4196), logo_ylim=[0, 1]),
      dict(subinterval=genome.Interval('chr1', 0, 2**20), logo_ylim=None),
  ])
  def test_track_component(self, subinterval, logo_ylim):
    interval = genome.Interval('chr1', 0, 2**20)

    # Mock 1D track data.
    atac_values = np.stack(
        [np.arange(interval.width) * 0.0001] * len(_ATAC_METADATA), axis=1
    ).astype(np.float32)
    atac_tdata = track_data.TrackData(
        values=atac_values,
        metadata=_ATAC_METADATA,
        interval=interval,
        resolution=1,
    )

    rna_seq_values_ref = np.stack(
        [np.arange(interval.width) * 0.0001] * len(_RNA_SEQ_METADATA), axis=1
    ).astype(np.float32)
    rna_seq_tdata_ref = track_data.TrackData(
        values=rna_seq_values_ref,
        metadata=_RNA_SEQ_METADATA,
        interval=interval,
        resolution=1,
    )
    # Mimic the case where the ALT allele increases expression by 50%.
    rna_seq_values_alt = rna_seq_values_ref * 1.5
    rna_seq_tdata_alt = track_data.TrackData(
        values=rna_seq_values_alt,
        metadata=_RNA_SEQ_METADATA,
        interval=interval,
        resolution=1,
    )

    # Mock contact maps.
    contact_maps_shape = (
        interval.width // 2048,
        interval.width // 2048,
        len(_CONTACT_MAP_METADATA),
    )

    contact_maps_1 = track_data.TrackData(
        values=np.random.normal(0, 1, contact_maps_shape).astype(np.float32),
        metadata=_CONTACT_MAP_METADATA,
        interval=interval,
        resolution=2048,
    )

    contact_maps_2 = track_data.TrackData(
        values=contact_maps_1.values * 0.5,
        metadata=_CONTACT_MAP_METADATA,
        interval=interval,
        resolution=2048,
    )
    contact_maps_diff = track_data.TrackData(
        values=contact_maps_1.values - contact_maps_2.values,
        metadata=_CONTACT_MAP_METADATA,
        interval=interval,
        resolution=2048,
    )

    # Mock contribution scores.
    contrib = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        * (interval.width // 4)
    ).astype(np.float32)

    # Get actual transcripts.
    transcripts = [
        transcript_utils.Transcript(
            exons=[genome.Interval('chr1', 0, 100)], info=dict(gene_name='gene')
        )
    ]

    # Mock junctions.
    junctions = np.array([
        genome.Junction('chr1', 102, 108, '+'),
        genome.Junction('chr1', 80, 100, '+'),
        genome.Junction('chr1', 104, 108, '+'),
        genome.Junction('chr1', 110, 120, '+'),
    ])
    junctions = junction_data.JunctionData(
        junctions,
        values=np.zeros((4, 2), dtype=np.float32),
        metadata=_SPLICING_METADATA,
    )

    # Mock variants within subinterval to annotate.
    annotated_variants = [
        genome.Variant(
            chromosome=subinterval.chromosome,
            position=pos,
            reference_bases='A',
            alternate_bases='G',
        )
        for pos in subinterval.start + subinterval.width / np.array(
            [2, 3, 10], dtype=np.float32
        )
    ]

    # Mock intervals  within subinterval to annotate.
    annotated_intervals = [
        interval.resize(int(interval.width // 10)),
        interval.resize(int(interval.width // 100)).shift(
            -int(interval.width // 10)
        ),
    ]
    annotated_intervals_labels = [
        f'my_interval_{i+1}' for i in range(len(annotated_intervals))
    ]

    fig = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(transcripts, fig_height=3),
            plot_components.TranscriptAnnotation(
                transcripts, adaptive_fig_height=False
            ),
            plot_components.TranscriptAnnotation(
                transcripts, adaptive_fig_height=True, fig_height=1
            ),
            plot_components.Sashimi(junctions),
            plot_components.Tracks(atac_tdata, filled=False, linestyle='--'),
            plot_components.Tracks(atac_tdata, filled=True),
            plot_components.Tracks(atac_tdata, shared_y_scale=True),
            plot_components.Tracks(atac_tdata, shared_y_scale=False),
            plot_components.Tracks(
                rna_seq_tdata_ref, cmap='viridis', truncate_cmap=True
            ),
            plot_components.Tracks(
                rna_seq_tdata_ref, cmap='Set2', truncate_cmap=False
            ),
            plot_components.OverlaidTracks(
                {'REF': rna_seq_tdata_ref, 'ALT': rna_seq_tdata_alt}
            ),
            plot_components.OverlaidTracks(
                {'REF': rna_seq_tdata_ref, 'ALT': rna_seq_tdata_alt},
                colors={'REF': 'blue', 'ALT': 'red'},
            ),
            plot_components.OverlaidTracks(
                {'REF': rna_seq_tdata_ref, 'ALT': rna_seq_tdata_alt},
                cmap='Set2',
                ylabel_template='{name}',
            ),
            plot_components.ContactMaps(
                tdata=contact_maps_1, cmap=matplotlib.pyplot.get_cmap('autumn')
            ),
            plot_components.ContactMapsDiff(
                tdata=contact_maps_diff,
            ),
            plot_components.SeqLogo(
                contrib,
                subinterval,
                ylabel='contribution scores',
                ylim=logo_ylim,
            ),
        ],
        interval=subinterval,
        annotations=[
            plot_components.IntervalAnnotation(
                annotated_intervals,
                labels=annotated_intervals_labels,
                colors='darkgray',
            ),
            plot_components.VariantAnnotation(
                annotated_variants, use_default_labels=False, colors='purple'
            ),
            plot_components.VariantAnnotation(
                annotated_variants, use_default_labels=True, colors='orange'
            ),
        ],
    )
    self.assertIsInstance(fig, matplotlib.figure.Figure)


if __name__ == '__main__':
  absltest.main()
