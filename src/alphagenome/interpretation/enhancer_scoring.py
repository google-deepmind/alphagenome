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
"""Enhancer-gene scoring and calling from input gradients."""

from collections.abc import Mapping, Sequence
import types
from typing import Literal

from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.models.v2 import dna_output
import numpy as np
import pandas as pd

# Recommended defaults.
_DEFAULT_WINDOW_SIZE = 2400
_DEFAULT_SIGMA = 300.0
_DEFAULT_ATAC_SCORE_THRESHOLD = 1.25

#: Calibrated default per-tile `gradient_score` thresholds by `OutputType` on
#: 1 Mb genomic intervals.
DEFAULT_GRADIENT_SCORE_THRESHOLDS: Mapping[dna_output.OutputType, float] = (
    types.MappingProxyType({
        dna_output.OutputType.SC_GENE_EXPRESSION: 0.0012,
        dna_output.OutputType.RNA_SEQ: 0.0017,
    })
)

_ENHANCER_BASE_DTYPES = types.MappingProxyType({
    'chromosome': str,
    'start': int,
    'end': int,
    'name': str,
    'biosample_name': str,
    'gene_id': str,
    'gradient_score': float,
})
_SCORE_ENHANCERS_DTYPES = types.MappingProxyType({
    **_ENHANCER_BASE_DTYPES,
    'is_scored': bool,
})

_SUMMIT_GROUP_COLS = ('biosample_name', 'gene_id', 'chromosome')
_MERGE_GROUP_COLS = ('biosample_name', 'chromosome')

_ACTIVE_ENHANCER_BASE_DTYPES = types.MappingProxyType({
    **_ENHANCER_BASE_DTYPES,
    'tss': int,
    'tss_distance': float,
})
_ACTIVE_ENHANCER_COLS = (
    *_ACTIVE_ENHANCER_BASE_DTYPES,
    'gene_name',
    'atac_score',
)


def _validate_tiling_params(tile_size: int, step_size: int) -> None:
  if tile_size <= 0:
    raise ValueError(f'tile_size must be positive, got {tile_size}.')
  if step_size <= 0:
    raise ValueError(f'step_size must be positive, got {step_size}.')


def _validate_window_params(window_size: int, sigma: float) -> None:
  if window_size < 0:
    raise ValueError(f'window_size must be non-negative, got {window_size}.')
  if sigma <= 0:
    raise ValueError(f'sigma must be positive, got {sigma}.')


def _gaussian_window_score(
    per_bp_scores: np.ndarray,
    midpoint: int,
    *,
    window_size: int,
    sigma: float,
) -> float:
  """Gaussian-weighted mean of `per_bp_scores` centered on `midpoint`."""
  half_window = window_size // 2
  window_start = max(0, midpoint - half_window)
  window_end = min(per_bp_scores.shape[0], midpoint + half_window + 1)
  distances = np.abs(np.arange(window_start, window_end) - midpoint)
  weights = np.exp(-0.5 * (distances / sigma) ** 2)
  window = per_bp_scores[window_start:window_end]
  return float(np.sum(weights * window) / np.sum(weights))


def tile_interval(
    interval: genome.Interval,
    *,
    tile_size: int = 500,
    step_size: int = 250,
) -> list[genome.Interval]:
  """Partitions `interval` into overlapping sliding-window tiles.

  Args:
    interval: Genomic interval to tile.
    tile_size: Width of each tile in base pairs (default: 500 bp).
    step_size: Step size between adjacent tiles in base pairs (default: 250 bp).

  Returns:
    List of full-width `genome.Interval` tiles (each `tile_size` bp) that lie
    completely within `interval`. A trailing partial tile shorter than
    `tile_size` is omitted, so the list is empty if
    `interval.width < tile_size`.

  Raises:
    ValueError: If `tile_size` or `step_size` is not positive.
  """
  _validate_tiling_params(tile_size, step_size)
  return [
      genome.Interval(
          interval.chromosome,
          start,
          start + tile_size,
          name=f'{interval.chromosome}:{start}-{start + tile_size}',
      )
      for start in range(
          interval.start,
          interval.end - tile_size + 1,
          step_size,
      )
  ]


def score_enhancers(
    gradients_output: dna_output.InputGradientsOutput,
    enhancers: Sequence[genome.Interval],
    *,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    sigma: float = _DEFAULT_SIGMA,
) -> pd.DataFrame:
  """Scores candidate enhancers for the target gene of an input gradient.

  The per-position score is `|sum_b gradients[pos, b]|`, i.e. the gradients are
  summed across nucleotides before taking the absolute value (as in
  `InputGradientsOutput.to_track_data`). The score of an enhancer is the
  Gaussian-weighted mean of these per-position scores, centered on the enhancer
  midpoint (`(start + end) // 2`). Positions are restricted to
  `[midpoint - window_size // 2, midpoint + window_size // 2]` and clipped to
  the input interval.

  Enhancers whose midpoint lies outside the gradient interval are not scored:
  they get `is_scored=False` and a NaN `gradient_score`.

  Args:
    gradients_output: Output of `DnaClient.predict_interval_gradients`.
    enhancers: Candidate enhancer intervals.
    window_size: Width of the window around the enhancer midpoint (bp).
    sigma: Standard deviation of the Gaussian kernel (bp).

  Returns:
    DataFrame with one row per enhancer, in input order, with columns
    `chromosome`, `start`, `end`, `name` (of the enhancer interval),
    `biosample_name`, `gene_id` (target gene of the gradient),
    `gradient_score`, and `is_scored`, plus a trailing `gene_name` column
    when present in `gradients_output.target_gene`.

  Raises:
    ValueError: If `window_size` is negative or `sigma` is not positive.
  """
  _validate_window_params(window_size, sigma)

  interval = gradients_output.interval
  gradients = np.asarray(gradients_output.gradients)
  # Gradients are mean-centered and multiplied by the one-hot sequence, so the
  # sum over nucleotides is the per-position contribution.
  per_bp_scores = np.abs(gradients.sum(axis=-1, dtype=np.float64))
  biosample_name = gradients_output.biosample_name
  gene_id = gradients_output.gene_id

  if not enhancers:
    empty = {
        col: pd.Series(dtype=dtype)
        for col, dtype in _SCORE_ENHANCERS_DTYPES.items()
    }
    df = pd.DataFrame(empty)
  else:
    rows = []
    for enhancer in enhancers:
      midpoint = (enhancer.start + enhancer.end) // 2 - interval.start
      is_scored = (
          enhancer.chromosome == interval.chromosome
          and 0 <= midpoint < per_bp_scores.shape[0]
      )
      if is_scored:
        score = _gaussian_window_score(
            per_bp_scores, midpoint, window_size=window_size, sigma=sigma
        )
      else:
        score = np.nan
      rows.append((
          enhancer.chromosome,
          enhancer.start,
          enhancer.end,
          enhancer.name,
          biosample_name,
          gene_id,
          score,
          is_scored,
      ))
    df = pd.DataFrame(rows, columns=list(_SCORE_ENHANCERS_DTYPES)).astype({
        'gradient_score': np.float64,
        'is_scored': bool,
    })

  gene_name = gradients_output.gene_name
  if gene_name is not None:
    df['gene_name'] = pd.Series(dtype=str) if not enhancers else gene_name

  return df


def _overlap_cluster_ids(
    df: pd.DataFrame, group_cols: Sequence[str]
) -> pd.Series:
  """Assigns integer IDs to overlapping `[start, end)` interval clusters.

  Starts a new cluster whenever `group_cols` changes or `start` is at or past
  the running maximum `end` of preceding intervals in the same group.

  Args:
    df: DataFrame pre-sorted by `[*group_cols, 'start', 'end']`.
    group_cols: Columns partitioning independent coordinate groups.

  Returns:
    Integer Series of cluster IDs aligned with `df.index`.
  """
  group_list = list(group_cols)
  prev_max_end = (
      df.groupby(group_list, sort=False, observed=True)['end'].cummax().shift(1)
  )
  new_group = ~df.duplicated(subset=group_list)
  return (new_group | (df['start'] >= prev_max_end)).cumsum()


def _extract_summit_tiles(df: pd.DataFrame) -> pd.DataFrame:
  """Extracts local-maximum `gradient_score` tiles per `(biosample, gene)`.

  Drops unscored (`NaN`) rows, groups overlapping tiles per
  `(biosample_name, gene_id, chromosome)` into clusters, and keeps tiles whose
  `gradient_score` is strictly greater than their left neighbor and greater
  than or equal to their right neighbor in sorted coordinate order (so the
  leftmost tile wins flat score plateaus, and duplicate `(start, end)` tiles
  keep only the highest score).

  Args:
    df: Scored enhancer tile DataFrame.

  Returns:
    Subset of `df` containing only summit tiles, with a reset index.
  """
  df_sorted = (
      df.dropna(subset=['gradient_score'])
      .sort_values(
          [*_SUMMIT_GROUP_COLS, 'start', 'end', 'gradient_score'],
          ascending=[True, True, True, True, True, False],
      )
      .reset_index(drop=True)
  )
  score = df_sorted['gradient_score']
  by_cluster = score.groupby(
      _overlap_cluster_ids(df_sorted, _SUMMIT_GROUP_COLS)
  )
  is_summit = (score > by_cluster.shift(1, fill_value=-np.inf)) & (
      score >= by_cluster.shift(-1, fill_value=-np.inf)
  )
  return df_sorted.loc[is_summit].reset_index(drop=True)


def _merge_summits_across_genes(summits: pd.DataFrame) -> pd.DataFrame:
  """Merges overlapping per-gene summits into shared peaks per biosample.

  Clusters overlapping summit intervals across all genes within each
  `(biosample_name, chromosome)`. Within each cluster, keeps each gene's
  highest-`gradient_score` summit row and widens `start`, `end`, and `name`
  (`'{chromosome}:{start}-{end}'`) to the cluster's `[min(start), max(end)]`
  span, yielding at most one row per `(biosample_name, merged peak, gene_id)`.

  Args:
    summits: Per-gene summit DataFrame (for example from
      `_extract_summit_tiles`).

  Returns:
    Merged DataFrame sorted by `(biosample_name, chromosome, start, end,
    gene_id)` with updated `start`, `end`, and `name` columns.
  """
  order = [*_MERGE_GROUP_COLS, 'start', 'end', 'gene_id']
  summits_sorted = summits.sort_values(order).reset_index(drop=True)
  cluster_id = _overlap_cluster_ids(summits_sorted, _MERGE_GROUP_COLS)
  best = summits_sorted.groupby(
      [cluster_id, summits_sorted['gene_id']], observed=True
  )['gradient_score'].idxmax()
  by_cluster = summits_sorted.groupby(cluster_id)
  merged = (
      summits_sorted.loc[best]
      .assign(
          start=by_cluster['start'].transform('min').loc[best],
          end=by_cluster['end'].transform('max').loc[best],
      )
      .sort_values(order)
      .reset_index(drop=True)
  )
  span = merged['start'].astype(str) + '-' + merged['end'].astype(str)
  return merged.assign(name=merged['chromosome'].astype(str) + ':' + span)


def _mane_tss_by_gene(gtf: pd.DataFrame) -> dict[tuple[str, str], int]:
  """Maps `(chromosome, unversioned_gene_id)` to its `MANE_Select` TSS."""
  mane = gene_annotation.filter_to_mane_select_transcript(
      gtf[gtf['Feature'] == 'transcript']
  )
  tss = gene_annotation.extract_tss(mane)
  ids = tss['gene_id'].str.removesuffix('_PAR_Y').str.split('.', n=1).str[0]
  keys = zip(tss['Chromosome'].astype(str), ids, strict=True)
  return dict(zip(keys, tss['Start'].astype(int), strict=True))


def _compute_tile_atac_scores(
    track: track_data.TrackData,
    tiles: Sequence[genome.Interval],
    prefer_observed: bool = True,
) -> np.ndarray:
  """Computes per-tile accessibility scores from a 1-bp ATAC or scATAC track."""
  if track.interval is None:
    raise ValueError('Accessibility track must have an interval.')
  if track.num_tracks == 0 or not tiles:
    return np.full(len(tiles), np.nan, dtype=np.float64)

  if prefer_observed and 'imputation_mode' in track.metadata.columns:
    observed = (track.metadata['imputation_mode'] == 'observed').to_numpy()
    values = track.values[:, observed] if observed.any() else track.values
  else:
    values = track.values

  offset = track.interval.start
  scores = [
      values[tile.start - offset : tile.end - offset].max(axis=0).mean()
      for tile in tiles
  ]
  return np.array(scores, dtype=np.float64)


def _resolve_gradient_threshold(
    output_type: dna_output.OutputType, threshold: float | Literal['auto']
) -> float:
  """Resolves `'auto'` or numeric `gradient_score` threshold."""
  if threshold != 'auto':
    return float(threshold)
  resolved = DEFAULT_GRADIENT_SCORE_THRESHOLDS.get(output_type)
  if resolved is None:
    raise ValueError(
        "`gradient_score_threshold='auto'` is not calibrated for"
        f' {output_type!r}.'
    )
  return resolved


def _resolve_accessibility_tracks(
    gradients_outputs: Sequence[dna_output.InputGradientsOutput],
    accessibility_outputs: Sequence[dna_output.Output] | None,
    output_type: dna_output.OutputType,
    atac_score_threshold: float | None,
) -> Sequence[track_data.TrackData | None]:
  """Validates accessibility inputs and extracts aligned 1-bp tracks."""
  if accessibility_outputs is None:
    if atac_score_threshold is not None:
      raise ValueError(
          '`accessibility_outputs` must be provided when'
          ' `atac_score_threshold` is set.'
      )
    return [None] * len(gradients_outputs)
  if len(accessibility_outputs) != len(gradients_outputs):
    raise ValueError(
        '`gradients_outputs` and `accessibility_outputs` must have the same'
        ' length.'
    )
  if output_type == dna_output.OutputType.SC_ATAC:
    tracks = [acc.sc_atac for acc in accessibility_outputs]
  elif output_type == dna_output.OutputType.ATAC:
    tracks = [acc.atac for acc in accessibility_outputs]
  else:
    raise ValueError(f'Unsupported output type: {output_type}.')
  for grad_out, track in zip(gradients_outputs, tracks, strict=True):
    if track is None or track.resolution != 1:
      raise ValueError(f'Expected 1-bp {output_type.name} track.')
    if track.interval is None or not track.interval.contains(grad_out.interval):
      raise ValueError('Accessibility interval must contain gradient interval.')
    biosamples = track.metadata.get('biosample_name')
    if track.num_tracks > 0 and (
        biosamples is None
        or biosamples.isna().any()
        or not (biosamples == grad_out.biosample_name).all()
    ):
      raise ValueError(
          'Accessibility `biosample_name` must match'
          f' {grad_out.biosample_name!r}.'
      )
  return tracks


def _call_locus_summits(
    scored_df: pd.DataFrame,
    tiles: Sequence[genome.Interval],
    grad_thresh: float,
    atac_track: track_data.TrackData | None,
    atac_thresh: float | None,
    prefer_observed: bool,
) -> pd.DataFrame:
  """Calls local-maximum active tiles for one `(biosample, gene)` locus."""
  grad_df = scored_df.drop(columns=['is_scored'])
  grad_filtered = grad_df.loc[grad_df['gradient_score'] >= grad_thresh]
  if atac_track is None:
    return _extract_summit_tiles(grad_filtered)
  kept_tiles = [tiles[i] for i in grad_filtered.index]
  scores = _compute_tile_atac_scores(atac_track, kept_tiles, prefer_observed)
  with_atac = grad_filtered.assign(atac_score=scores)
  active_df = (
      with_atac.loc[with_atac['atac_score'] >= float(atac_thresh)]
      if atac_thresh is not None
      else with_atac
  )
  return _extract_summit_tiles(active_df)


def get_active_enhancers(
    gradients_outputs: Sequence[dna_output.InputGradientsOutput],
    *,
    gtf: pd.DataFrame,
    accessibility_outputs: Sequence[dna_output.Output] | None = None,
    accessibility_output_type: Literal[
        dna_output.OutputType.SC_ATAC, dna_output.OutputType.ATAC
    ] = dna_output.OutputType.SC_ATAC,
    tile_size: int = 500,
    step_size: int = 250,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    sigma: float = _DEFAULT_SIGMA,
    gradient_score_threshold: float | Literal['auto'] = 'auto',
    atac_score_threshold: float | None = _DEFAULT_ATAC_SCORE_THRESHOLD,
    prefer_observed_accessibility: bool = True,
) -> pd.DataFrame:
  """Calls and merges active enhancer-to-gene links from input gradients.

  For each `(biosample_name, gene_id)` locus, `tile_interval` windows are scored
  with `score_enhancers` and filtered by `gradient_score_threshold` and optional
  1-bp chromatin accessibility (`atac_score`, the per-tile maximum accessibility
  averaged across channels). Passing tiles are reduced per gene to local
  `gradient_score` summits within overlapping clusters, then overlapping summits
  across genes in the same `(biosample_name, chromosome)` are merged to their
  `[min(start), max(end)]` span while retaining each gene's top summit scores.

  Args:
    gradients_outputs: `InputGradientsOutput` sequence with unique
      `(biosample_name, gene_id)` pairs.
    gtf: GENCODE GTF DataFrame used to resolve `MANE_Select` transcript TSSs.
    accessibility_outputs: `dna_output.Output` sequence aligned with
      `gradients_outputs`, or `None` when `atac_score_threshold` is `None`.
    accessibility_output_type: Accessibility track to use (`SC_ATAC` or `ATAC`).
    tile_size: Candidate tile width in bp.
    step_size: Step between adjacent tiles in bp.
    window_size: Gaussian smoothing window width around each tile midpoint (bp).
    sigma: Gaussian kernel standard deviation (bp).
    gradient_score_threshold: Minimum `gradient_score`, or `'auto'` to use
      `DEFAULT_GRADIENT_SCORE_THRESHOLDS` (`0.0012` for `SC_GENE_EXPRESSION`,
      `0.0017` for `RNA_SEQ`).
    atac_score_threshold: Minimum `atac_score`, or `None` to skip accessibility
      filtering.
    prefer_observed_accessibility: If True, averages `atac_score` over
      `observed` channels when present, falling back to all channels otherwise.

  Returns:
    DataFrame with one row per `(biosample_name, merged peak, gene_id)` link and
    columns `chromosome`, `start`, `end`, `name`, `biosample_name`, `gene_id`,
    `gradient_score`, `tss`, and `tss_distance` (`|0.5 * (start + end) - tss|`),
    plus `gene_name` when present on any target gene and `atac_score` when
    `accessibility_outputs` is provided.

  Raises:
    ValueError: If tiling or window parameters are invalid, if `'auto'` lacks a
      threshold for `output_type`, if `accessibility_outputs` is missing,
      misaligned, not 1-bp `SC_ATAC`/`ATAC`, narrower than a gradient interval,
      or from a different `biosample_name`, if `(biosample_name, gene_id)` pairs
      repeat, or if a target gene has no `MANE_Select` TSS on its chromosome in
      `gtf`.
  """
  _validate_tiling_params(tile_size, step_size)
  _validate_window_params(window_size, sigma)
  atac_tracks = _resolve_accessibility_tracks(
      gradients_outputs,
      accessibility_outputs,
      accessibility_output_type,
      atac_score_threshold,
  )
  if not gradients_outputs:
    dtypes = (
        {**_ACTIVE_ENHANCER_BASE_DTYPES, 'atac_score': float}
        if accessibility_outputs is not None
        else _ACTIVE_ENHANCER_BASE_DTYPES
    )
    empty = {c: pd.Series(dtype=t) for c, t in dtypes.items()}
    return pd.DataFrame(empty)

  tss_by_gene = _mane_tss_by_gene(gtf)
  seen_pairs: set[tuple[str, str]] = set()
  summit_dfs: list[pd.DataFrame] = []
  for grad_out, atac_track in zip(gradients_outputs, atac_tracks, strict=True):
    grad_thresh = _resolve_gradient_threshold(
        grad_out.output_type, gradient_score_threshold
    )
    pair = (grad_out.biosample_name, grad_out.gene_id)
    if pair in seen_pairs:
      raise ValueError(f'Duplicate `(biosample_name, gene_id)` pair {pair!r}.')
    seen_pairs.add(pair)
    unversioned = grad_out.gene_id.removesuffix('_PAR_Y').split('.', 1)[0]
    tss = tss_by_gene.get((grad_out.interval.chromosome, unversioned))
    if tss is None:
      raise ValueError(
          f'No MANE_Select TSS for {grad_out.gene_id!r} on'
          f' {grad_out.interval.chromosome!r} in `gtf`.'
      )
    tiles = tile_interval(
        grad_out.interval, tile_size=tile_size, step_size=step_size
    )
    scored_df = score_enhancers(
        grad_out, tiles, window_size=window_size, sigma=sigma
    )
    summits = _call_locus_summits(
        scored_df,
        tiles,
        grad_thresh,
        atac_track,
        atac_score_threshold,
        prefer_observed_accessibility,
    )
    summit_dfs.append(summits.assign(tss=np.full(len(summits), tss, dtype=int)))

  merged = _merge_summits_across_genes(pd.concat(summit_dfs, ignore_index=True))
  centers = (merged['start'] + merged['end']) / 2.0
  enhancers = merged.assign(tss_distance=(centers - merged['tss']).abs())
  return enhancers[[c for c in _ACTIVE_ENHANCER_COLS if c in enhancers.columns]]


def assign_consensus_enhancers(
    enhancers_df: pd.DataFrame,
    *,
    min_reciprocal_overlap: float = 0.5,
) -> pd.DataFrame:
  """Assigns cross-biosample consensus peak bounds via reciprocal-overlap NMS.

  Calling active enhancers per biosample on a sliding tile grid can place
  summits of the same regulatory element on slightly shifted overlapping tiles;
  clustering them into shared consensus coordinates lets callers compare
  enhancer scores or count active biosamples across conditions.

  Within each `(gene_id, chromosome)` group, distinct `(start, end)` enhancer
  intervals are ranked by their row count in `enhancers_df` (breaking ties by
  ascending `(start, end)`) and clustered greedily: each candidate interval is
  assigned to the highest-overlapping already-accepted representative peak whose
  mutual overlap `overlap / max(len_a, len_b)` is at least
  `min_reciprocal_overlap`, or starts a new consensus peak otherwise.

  Args:
    enhancers_df: DataFrame of called enhancer-gene links (such as the output of
      `get_active_enhancers`) containing at least `gene_id`, `chromosome`,
      `start`, and `end` columns. Any existing `consensus_peak_start` and
      `consensus_peak_end` columns are replaced.
    min_reciprocal_overlap: Minimum mutual overlap fraction `overlap /
      max(len_a, len_b)` in `(0, 1]` required to merge an interval into an
      existing consensus representative.

  Returns:
    Copy of `enhancers_df` (preserving its index and row order) with integer
    `consensus_peak_start` and `consensus_peak_end` columns appended.

  Raises:
    ValueError: If `min_reciprocal_overlap` is not in `(0, 1]`, if
      `enhancers_df` is missing any of `gene_id`, `chromosome`, `start`, or
      `end`, if those columns contain null values, or if any interval does not
      satisfy `start < end`.
  """
  if not 0.0 < min_reciprocal_overlap <= 1.0:
    raise ValueError(
        'min_reciprocal_overlap must be in (0, 1], got'
        f' {min_reciprocal_overlap!r}.'
    )
  gene_cols = ['gene_id', 'chromosome', 'start', 'end']
  missing_cols = set(gene_cols) - set(enhancers_df.columns)
  if missing_cols:
    raise ValueError(
        f'enhancers_df is missing required columns: {sorted(missing_cols)}.'
    )
  base_df = enhancers_df.drop(
      columns=['consensus_peak_start', 'consensus_peak_end'], errors='ignore'
  )
  if base_df[gene_cols].isna().any().any():
    raise ValueError(
        f'enhancers_df columns {gene_cols} must not contain null values.'
    )
  if (base_df['end'] <= base_df['start']).any():
    raise ValueError('enhancers_df intervals must satisfy start < end.')
  if base_df.empty:
    return base_df.assign(
        consensus_peak_start=pd.Series(dtype=int, index=base_df.index),
        consensus_peak_end=pd.Series(dtype=int, index=base_df.index),
    )
  intervals = (
      base_df.groupby(gene_cols, as_index=False, observed=True)
      .agg(_count=('start', 'count'))
      .sort_values(
          ['gene_id', 'chromosome', '_count', 'start', 'end'],
          ascending=[True, True, False, True, True],
      )
      .reset_index(drop=True)
  )
  rep_rows: list[int] = []
  for _, grp in intervals.groupby(
      ['gene_id', 'chromosome'], sort=False, observed=True
  ):
    starts = grp['start'].to_numpy()
    ends = grp['end'].to_numpy()
    lengths = ends - starts
    reps = [0]
    assigned = [0]
    for i in range(1, len(starts)):
      ov = np.minimum(ends[i], ends[reps]) - np.maximum(starts[i], starts[reps])
      frac = ov / np.maximum(lengths[i], lengths[reps])
      best = int(np.argmax(frac))
      if frac[best] > 0 and frac[best] >= min_reciprocal_overlap - 1e-12:
        assigned.append(reps[best])
      else:
        reps.append(i)
        assigned.append(i)
    rep_rows.extend(grp.index[assigned])

  remap = intervals[gene_cols].assign(
      consensus_peak_start=intervals['start'].to_numpy(dtype=int)[rep_rows],
      consensus_peak_end=intervals['end'].to_numpy(dtype=int)[rep_rows],
  )
  return base_df.merge(remap, on=gene_cols, how='left').set_axis(base_df.index)
