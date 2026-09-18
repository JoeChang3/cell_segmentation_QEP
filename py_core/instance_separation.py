"""
Instance separation: explicit, deterministic watershed marker generation.

Two modes, sharing every other setting:

  "legacy_none"  watershed(-dist, markers=None, mask=fg)
                 Reproduces the historical configuration used by
                 generate_gp_masks_test (Modified_Functions_RGasp.py:479) and by
                 round 1. With markers=None, skimage seeds a basin at EVERY
                 local minimum of the elevation, i.e. every local maximum of the
                 distance transform, so one irregular cell becomes several
                 labels. The legacy code comments note it is only an
                 approximation of EBImage::watershed, which performs its own
                 internal marker detection.

  "peak"         markers from skimage.feature.peak_local_max on the distance
                 transform, restricted to the foreground.

INSTALLED API (skimage 0.25.2, scipy 1.15.3) AND THE CHOICES MADE

  peak_local_max(image, min_distance=1, threshold_abs=None, threshold_rel=None,
                 exclude_border=True, num_peaks=inf, footprint=None,
                 labels=None, num_peaks_per_label=inf, p_norm=inf)

  min_distance   Units are PIXELS: the minimum allowed separation between two
                 retained peaks. Because the default p_norm=inf, the metric is
                 Chebyshev (chessboard), so min_distance=9 means peaks must
                 differ by >=9 px in row or column. Kept at p_norm=inf.

  exclude_border THE DEFAULT (True) EXCLUDES A BORDER OF WIDTH min_distance and
                 would silently discard every cell touching the image edge.
                 Set to False here so border cells keep their markers. This is
                 a deliberate deviation from the skimage default.

  threshold_abs  Set to None. A distance-transform peak is meaningful at any
                 height, and an absolute cutoff would be an implicit minimum
                 cell-radius filter applied inconsistently across datasets with
                 different cell sizes. Small objects are removed later by the
                 existing cleanup rule instead.

  threshold_rel  None, for the same reason: it would scale with the single
                 largest cell in the image.

  labels         Set to the binary foreground, so peaks are searched only
                 inside foreground and never in background.

  connectivity   watershed(connectivity=1) is the skimage default = 4-connectivity
                 in 2D. Left at the default, matching the legacy call. Foreground
                 component labelling for the no-marker policy uses the same
                 connectivity=1 for consistency.

  plateaux/ties  peak_local_max compares against a maximum filter, so an exactly
                 flat distance-transform plateau can yield several candidate
                 points; min_distance then prunes them, and the survivor depends
                 on array order. To make marker IDs reproducible regardless of
                 that order, the returned coordinates are sorted
                 lexicographically by (row, col) before IDs are assigned, and
                 each peak receives its OWN id rather than being passed through
                 ndi.label (which would fuse two peaks lying in adjacent pixels).

  empty
  components     A foreground connected component can end up with no peak (for
                 example a component only a few pixels across). Policy: every
                 such component receives exactly one marker at its
                 distance-transform argmax, tie-broken by the first position in
                 C raster order. Foreground is therefore never silently dropped,
                 and the number of rescued components is reported as
                 `n_components_rescued`.

  cleanup        eliminate_small_areas(labels, remove_size_threshold) is the
                 existing legacy rule and is applied identically in both modes.
                 No border-object removal is applied in either mode.

No ground-truth information (instance centres, counts, or boundaries) is used
anywhere in marker generation, so both modes are deployable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label as cc_label
from skimage.segmentation import watershed

from py_core.Modified_Functions_RGasp import eliminate_small_areas

MARKER_MODES = ("legacy_none", "peak")
WATERSHED_CONNECTIVITY = 1          # skimage default; 4-connectivity in 2D
PEAK_P_NORM = np.inf                # skimage default; Chebyshev metric
PEAK_EXCLUDE_BORDER = False         # deliberate deviation; keeps border cells


@dataclass
class SeparationResult:
    instance_mask: np.ndarray
    marker_image: np.ndarray            # int labels, 0 = no marker
    marker_coords: np.ndarray           # (n, 2) row, col; empty for legacy_none
    dist_map: np.ndarray
    n_instances: int
    n_markers: int
    n_fg_components: int
    n_components_rescued: int
    config: Dict = field(default_factory=dict)


def make_peak_markers(
    dist: np.ndarray,
    foreground: np.ndarray,
    *,
    min_distance: int,
    rescue_empty_components: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Deterministic integer marker image from distance-transform peaks.

    Returns (marker_image, coords_sorted, n_fg_components, n_rescued).
    """
    fg = foreground > 0
    coords = peak_local_max(
        dist,
        min_distance=int(min_distance),
        threshold_abs=None,
        threshold_rel=None,
        exclude_border=PEAK_EXCLUDE_BORDER,
        labels=fg,
        p_norm=PEAK_P_NORM,
    )
    # deterministic ordering, independent of skimage's internal traversal
    if len(coords):
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        coords = coords[order]

    comps, n_comp = ndi.label(fg, structure=ndi.generate_binary_structure(2, 1))

    marker_image = np.zeros(dist.shape, dtype=np.int32)
    # each peak gets its own id: never fuse two adjacent peaks via ndi.label
    for i, (r, c) in enumerate(coords, start=1):
        marker_image[r, c] = i

    n_rescued = 0
    if rescue_empty_components and n_comp:
        # which components already contain at least one marker
        have = np.zeros(n_comp + 1, dtype=bool)
        if len(coords):
            have[np.unique(comps[marker_image > 0])] = True
        have[0] = True                                   # background
        missing = np.flatnonzero(~have)
        if missing.size:
            nxt = int(marker_image.max())
            # find_objects gives each component's bounding slice; deterministic
            slices = ndi.find_objects(comps)
            for comp_id in missing:
                sl = slices[comp_id - 1]
                if sl is None:
                    continue
                sub_d = np.where(comps[sl] == comp_id, dist[sl], -1.0)
                # argmax over C raster order -> first maximum wins (documented tie-break)
                rr, cc = np.unravel_index(int(np.argmax(sub_d)), sub_d.shape)
                nxt += 1
                marker_image[sl[0].start + rr, sl[1].start + cc] = nxt
                n_rescued += 1
            if n_rescued:
                pts = np.argwhere(marker_image > 0)
                order = np.lexsort((pts[:, 1], pts[:, 0]))
                coords = pts[order]

    return marker_image, coords, int(n_comp), int(n_rescued)


def separate_instances(
    binary: np.ndarray,
    *,
    marker_mode: str = "legacy_none",
    min_distance: int = 9,
    remove_size_threshold: int = 50,
    rescue_empty_components: bool = True,
) -> SeparationResult:
    """Foreground -> instance labels. Only marker generation varies by mode.

    Distance transform, watershed elevation (-dist), mask, connectivity and the
    small-area cleanup are identical across modes, so a difference between modes
    is attributable to marker generation alone.
    """
    if marker_mode not in MARKER_MODES:
        raise ValueError(f"marker_mode must be one of {MARKER_MODES}")

    fg = binary > 0
    dist = ndi.distance_transform_edt(fg)

    if marker_mode == "legacy_none":
        markers_arg = None
        marker_image = np.zeros(fg.shape, dtype=np.int32)
        coords = np.empty((0, 2), dtype=int)
        n_comp = int(ndi.label(fg, structure=ndi.generate_binary_structure(2, 1))[1])
        n_rescued = 0
    else:
        marker_image, coords, n_comp, n_rescued = make_peak_markers(
            dist, fg, min_distance=min_distance,
            rescue_empty_components=rescue_empty_components)
        markers_arg = marker_image

    raw = watershed(-dist, markers=markers_arg, mask=fg,
                    connectivity=WATERSHED_CONNECTIVITY).astype(np.int32)
    inst = eliminate_small_areas(raw, remove_size_threshold)
    n_inst = int(len(np.unique(inst[inst > 0])))

    return SeparationResult(
        instance_mask=inst,
        marker_image=marker_image,
        marker_coords=np.asarray(coords, dtype=int),
        dist_map=dist,
        n_instances=n_inst,
        n_markers=int(len(coords)),
        n_fg_components=n_comp,
        n_components_rescued=n_rescued,
        config=dict(
            marker_mode=marker_mode,
            min_distance=(int(min_distance) if marker_mode == "peak" else None),
            remove_size_threshold=int(remove_size_threshold),
            watershed_connectivity=WATERSHED_CONNECTIVITY,
            peak_exclude_border=PEAK_EXCLUDE_BORDER,
            peak_p_norm=("inf" if PEAK_P_NORM == np.inf else PEAK_P_NORM),
            peak_threshold_abs=None, peak_threshold_rel=None,
            peak_labels="foreground",
            rescue_empty_components=bool(rescue_empty_components),
            elevation="-distance_transform_edt(foreground)",
            border_object_removal="none",
            uses_ground_truth=False,
        ),
    )


def partitions_equivalent(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two label images define the same partition up to label renaming.

    Used by the reproducibility check: identical segmentations may legitimately
    carry different integer ids.
    """
    if a.shape != b.shape:
        return False
    if not np.array_equal(a > 0, b > 0):
        return False
    pa, pb = a.ravel(), b.ravel()
    m = pa > 0
    if not m.any():
        return True
    # a bijection between label sets exists iff the pair (a,b) has as many
    # distinct combinations as each side has distinct labels
    pairs = np.unique(np.stack([pa[m], pb[m]], axis=1), axis=0)
    return (len(pairs) == len(np.unique(pa[m]))
            and len(pairs) == len(np.unique(pb[m])))
