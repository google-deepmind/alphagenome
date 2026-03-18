"""Utilities for aligning track data for visualization."""

import copy
from typing import Tuple

from alphagenome.data import genome
from alphagenome.data import track_data
import numpy as np
import scipy.interpolate


def align_ref_alt_tracks(
    ref_track: track_data.TrackData,
    alt_track: track_data.TrackData,
    variant: genome.Variant,
) -> Tuple[track_data.TrackData, track_data.TrackData]:
  """Aligns REF and ALT TrackData to a common REF-based coordinate grid.

  This is intended for visualization only. It keeps the REF track unchanged
  (except possibly for minor type conversions), and re-maps the ALT values
  onto the REF coordinate grid using interpolation.

  - For deletions: the REF interval contains coordinates that are missing in
    ALT. These are represented in the aligned ALT track as NaNs.
  - For insertions: ALT has additional coordinates locally, which get
    compressed when remapped to the REF grid via interpolation.

  The returned TrackData objects share:
    - The same interval (REF interval).
    - The same resolution.
    - The same shape for `values`.

  Args:
    ref_track: TrackData from the reference model output.
    alt_track: TrackData from the alternate model output.
    variant: The variant used to generate predictions.

  Returns:
    A tuple (ref_aligned, alt_aligned) both on the REF coordinate system.
  """
  # We use ref_track as the target coordinate system
  target_interval = ref_track.interval
  resolution = ref_track.resolution

  # Construct REF x-coordinates (target grid)
  # Corresponds to: interval.start + i * resolution
  ref_x = (
      np.arange(ref_track.values.shape[0]) * resolution + target_interval.start
  )
  ref_x_centers = ref_x + resolution / 2.0

  # Construct ALT x-coordinates (source grid) in its own "ALT" space
  assert alt_track.interval is not None
  alt_start = alt_track.interval.start
  alt_x_centers_native = (
      np.arange(alt_track.values.shape[0]) * alt_track.resolution
      + alt_start
      + alt_track.resolution / 2.0
  )

  # Now map ALT native coords to REF coords.
  # Logic:
  # Upstream of variant: x_ref = x_alt
  # Downstream of variant: x_ref = x_alt - len_alt + len_ref
  # Inside variant: Linear mapping

  var_start = variant.start
  len_ref = len(variant.reference_bases)
  len_alt = len(variant.alternate_bases)

  alt_x_centers_mapped = np.copy(alt_x_centers_native)

  # Mask for upstream, downstream
  mask_upstream = alt_x_centers_native < var_start
  mask_downstream = alt_x_centers_native >= (var_start + len_alt)
  mask_inside = (~mask_upstream) & (~mask_downstream)

  # Apply shifts
  # Upstream: No change.

  # Downstream: Add (len_ref - len_alt)
  shift = len_ref - len_alt
  alt_x_centers_mapped[mask_downstream] += shift

  # Inside: Stretch/Compress
  # We want to map [var_start, var_start + len_alt) to
  # [var_start, var_start + len_ref)
  if np.any(mask_inside) and len_alt > 0:
    if len_ref < len_alt:
      scale = len_ref / len_alt
    else:
      scale = 1.0
    alt_x_centers_mapped[mask_inside] = (
        var_start + (alt_x_centers_native[mask_inside] - var_start) * scale
    )

  # Make a copy of REF track for aligned output (to preserve metadata)
  ref_aligned = ref_track # It's already in REF coords.

  # Interpolate ALT values onto REF grid
  # values shape: (positional_bins, num_tracks)
  # We loop over tracks or use interp1d with axis.

  alt_values = alt_track.values
  # Cast to float if not already, because interpolation requires it.
  is_integer = np.issubdtype(alt_values.dtype, np.integer) or np.issubdtype(
      alt_values.dtype, np.bool_
  )

  if is_integer:
    # Nearest neighbor interpolation
    alt_values = alt_values.astype(np.float32)
    interp_kind = 'nearest'
  else:
    interp_kind = 'linear'

  interp_func = scipy.interpolate.interp1d(
      alt_x_centers_mapped,
      alt_values,
      kind=interp_kind,
      axis=0,
      bounds_error=False,
      fill_value=np.nan,
      assume_sorted=True,
  )

  new_alt_values = interp_func(ref_x_centers)

  # For Deletions: Explicitly NaN out the deleted region in REF coordinates.
  # The deleted region is [var_start + len_alt, var_start + len_ref).
  if len_ref > len_alt:  # Deletion
    gap_start = var_start + len_alt
    gap_end = var_start + len_ref
    mask_gap = (ref_x_centers >= gap_start) & (ref_x_centers < gap_end)
    new_alt_values[mask_gap] = np.nan

  # Create new TrackData for ALT
  alt_aligned = track_data.TrackData(
      values=new_alt_values,
      resolution=resolution,
      metadata=alt_track.metadata.copy(),
      interval=target_interval,  # Now matches REF
      uns=copy.deepcopy(alt_track.uns),
  )

  return ref_aligned, alt_aligned
