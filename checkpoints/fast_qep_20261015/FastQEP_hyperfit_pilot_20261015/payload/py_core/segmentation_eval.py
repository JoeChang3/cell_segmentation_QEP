"""
Instance-segmentation evaluation for the real-data pipeline.

Keeps the legacy metric DEFINITIONS from py_core/metrics.py (so numbers remain
comparable to the stored results) while fixing two practical problems:

1. SPEED. `py_core.metrics.compute_ious` loops over every (true, pred) label
   pair and runs a full-image boolean AND: for nuclei_figure_1 that is
   330 x 393 = 129,690 passes over a 1.08 Mpixel array, several minutes per
   method. `compute_ious_fast` computes the identical IoU matrix from a single
   label-pair contingency table via np.bincount. Parity against the legacy
   implementation is asserted by `check_iou_parity`.

2. GROUND-TRUTH ORIENTATION. `py_core.metrics.process_image_mask` relabels the
   mask and then applies `mask.T[:, ::-1][:, ::-1]`, in which the two column
   reversals cancel, so the net effect is a plain TRANSPOSE (verified:
   np.array_equal(loaded, relabeled.T) is True). The raw ground-truth PNGs are
   already aligned with their images, so that transpose misaligns the mask
   against the image. Downstream, `align_mask_to_reference` transposes the
   PREDICTION to match, which restores consistency for IoU (IoU is invariant
   when both masks are transposed) but leaves the saved overlay figures drawing
   transposed masks on an un-transposed image.

   Two further hazards: for a SQUARE image `align_mask_to_reference` returns the
   prediction unchanged, so the ground truth would stay transposed and the
   comparison would silently be wrong; and whole_cell_figure_1 is 600x602, one
   pixel away from square.

   `load_instance_mask` therefore relabels WITHOUT transposing. IoU scores are
   unchanged (a consistent transpose of both masks preserves all overlaps), but
   overlays are now correct and the square-image hazard is gone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import tifffile as tiff

# Legacy definitions, reused for parity.
from py_core.metrics import compute_ap_from_ious, compute_ious

# Same grid as the legacy pipeline (R: seq(0.5, 0.80, by=0.05)), extended to
# 0.90 so AP@0.75 and AP@0.90 are both available. The legacy subset is a prefix.
AP_THRESHOLDS = np.round(np.arange(0.50, 0.90 + 1e-9, 0.05), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gray_image(path: str) -> np.ndarray:
    """Grayscale float image. First channel if RGB, matching the legacy path."""
    arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float64)


def load_instance_mask(path: str) -> np.ndarray:
    """Instance-labeled mask in NATIVE orientation (no transpose).

    Unique values are mapped to consecutive integers 0..K-1, with 0 assumed to
    be background (the smallest value). Uses np.unique + searchsorted rather
    than the legacy np.vectorize, which is far faster and gives the same map.
    """
    ext = os.path.splitext(path)[1].lower()
    arr = tiff.imread(path) if ext in (".tif", ".tiff") else imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    uniq = np.unique(arr)
    return np.searchsorted(uniq, arr).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Fast IoU
# ─────────────────────────────────────────────────────────────────────────────

def compute_ious_fast(true_mask: np.ndarray, pred_mask: np.ndarray) -> pd.DataFrame:
    """IoU matrix (rows = true labels > 0, cols = pred labels > 0).

    Identical values to py_core.metrics.compute_ious, computed in one pass.
    """
    if true_mask.shape != pred_mask.shape:
        raise ValueError(f"shape mismatch: true {true_mask.shape} vs "
                         f"pred {pred_mask.shape}")

    t_labels = np.unique(true_mask[true_mask > 0])
    p_labels = np.unique(pred_mask[pred_mask > 0])
    if t_labels.size == 0 or p_labels.size == 0:
        return pd.DataFrame(np.zeros((t_labels.size, p_labels.size)),
                            index=t_labels.astype(int),
                            columns=p_labels.astype(int))

    # compact 0..K indices; 0 reserved for background
    t_idx = np.zeros(int(true_mask.max()) + 1, dtype=np.int64)
    t_idx[t_labels] = np.arange(1, t_labels.size + 1)
    p_idx = np.zeros(int(pred_mask.max()) + 1, dtype=np.int64)
    p_idx[p_labels] = np.arange(1, p_labels.size + 1)

    ti = t_idx[true_mask.ravel()]
    pi = p_idx[pred_mask.ravel()]

    nt, npd = t_labels.size, p_labels.size
    hist = np.bincount(ti * (npd + 1) + pi,
                       minlength=(nt + 1) * (npd + 1)).reshape(nt + 1, npd + 1)

    inter = hist[1:, 1:].astype(np.float64)          # true>0 and pred>0
    t_area = hist[1:, :].sum(axis=1).astype(np.float64)[:, None]
    p_area = hist[:, 1:].sum(axis=0).astype(np.float64)[None, :]
    union = t_area + p_area - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = np.where(union > 0, inter / union, 0.0)

    return pd.DataFrame(ious, index=t_labels.astype(int),
                        columns=p_labels.astype(int))


def check_iou_parity(true_mask: np.ndarray, pred_mask: np.ndarray,
                     max_labels: int = 40) -> float:
    """Assert compute_ious_fast matches the legacy compute_ious on a subcrop.

    Returns the max absolute difference. Uses a crop so the slow legacy
    implementation stays affordable.
    """
    t_lab = np.unique(true_mask[true_mask > 0])[:max_labels]
    p_lab = np.unique(pred_mask[pred_mask > 0])[:max_labels]
    tm = np.where(np.isin(true_mask, t_lab), true_mask, 0)
    pm = np.where(np.isin(pred_mask, p_lab), pred_mask, 0)
    a = compute_ious(tm, pm).to_numpy()
    b = compute_ious_fast(tm, pm).to_numpy()
    if a.shape != b.shape:
        raise AssertionError(f"parity shape mismatch {a.shape} vs {b.shape}")
    return float(np.abs(a - b).max())


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    n_true: int
    n_pred: int
    mean_matched_iou: float
    ap_curve: pd.DataFrame
    per_threshold: Dict[float, Dict] = field(default_factory=dict)
    extras: Dict = field(default_factory=dict)

    def ap(self, thr: float) -> float:
        return float(self.per_threshold[thr]["precision"])


def evaluate_instances(true_mask: np.ndarray, pred_mask: np.ndarray,
                       thresholds: Optional[np.ndarray] = None) -> EvalResult:
    """TP/FP/FN and AP = TP/(TP+FP+FN) at each threshold (legacy definition)."""
    thresholds = AP_THRESHOLDS if thresholds is None else thresholds
    ious = compute_ious_fast(true_mask, pred_mask)

    per_thr: Dict[float, Dict] = {}
    rows = []
    for th in thresholds:
        r = compute_ap_from_ious(ious, threshold=float(th))
        per_thr[float(th)] = r
        rows.append(dict(Threshold=float(th), AP=r["precision"],
                         TP=r["tp"], FP=r["fp"], FN=r["fn"]))

    mat = ious.to_numpy()
    # mean of each true cell's best IoU: 0 for a true cell with no overlap
    mean_matched = float(mat.max(axis=1).mean()) if mat.size else 0.0

    n_true = int(ious.shape[0])
    n_pred = int(ious.shape[1])
    return EvalResult(
        n_true=n_true, n_pred=n_pred,
        mean_matched_iou=mean_matched,
        ap_curve=pd.DataFrame(rows),
        per_threshold=per_thr,
        extras=dict(iou_matrix_shape=(n_true, n_pred)),
    )


def classify_failure_modes(true_mask: np.ndarray, pred_mask: np.ndarray,
                           iou_match: float = 0.5,
                           overlap_frac: float = 0.25) -> Dict[str, int]:
    """Count concrete, countable segmentation failure modes.

    merged  : one predicted instance overlapping >=2 true cells by >=overlap_frac
              of each of those true cells -> touching cells fused
    split   : one true cell covered by >=2 predicted instances, each taking
              >=overlap_frac of it -> single cell fragmented
    missed  : true cells with best IoU < iou_match (weak/undetected cells)
    spurious: predicted instances whose best IoU with any true cell < iou_match
              and which cover <overlap_frac of every true cell -> false foreground
    """
    t_labels = np.unique(true_mask[true_mask > 0])
    p_labels = np.unique(pred_mask[pred_mask > 0])
    out = dict(merged=0, split=0, missed=0, spurious=0,
               n_true=int(t_labels.size), n_pred=int(p_labels.size))
    if t_labels.size == 0 or p_labels.size == 0:
        out["missed"] = int(t_labels.size)
        out["spurious"] = int(p_labels.size)
        return out

    t_idx = np.zeros(int(true_mask.max()) + 1, dtype=np.int64)
    t_idx[t_labels] = np.arange(1, t_labels.size + 1)
    p_idx = np.zeros(int(pred_mask.max()) + 1, dtype=np.int64)
    p_idx[p_labels] = np.arange(1, p_labels.size + 1)
    ti = t_idx[true_mask.ravel()]
    pi = p_idx[pred_mask.ravel()]
    nt, npd = t_labels.size, p_labels.size
    hist = np.bincount(ti * (npd + 1) + pi,
                       minlength=(nt + 1) * (npd + 1)).reshape(nt + 1, npd + 1)
    inter = hist[1:, 1:].astype(np.float64)
    t_area = hist[1:, :].sum(axis=1).astype(np.float64)[:, None]
    p_area = hist[:, 1:].sum(axis=0).astype(np.float64)[None, :]
    union = t_area + p_area - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = np.where(union > 0, inter / union, 0.0)

    frac_of_true = np.divide(inter, np.maximum(t_area, 1))     # nt x npd

    # merged: a pred instance substantially covering >= 2 true cells
    out["merged"] = int(((frac_of_true >= overlap_frac).sum(axis=0) >= 2).sum())
    # split: a true cell substantially covered by >= 2 pred instances
    out["split"] = int(((frac_of_true >= overlap_frac).sum(axis=1) >= 2).sum())
    # missed: true cell with no adequate match
    out["missed"] = int((ious.max(axis=1) < iou_match).sum())
    # spurious: pred with no adequate match and no substantial true overlap
    weak = ious.max(axis=0) < iou_match
    no_cover = (frac_of_true >= overlap_frac).sum(axis=0) == 0
    out["spurious"] = int((weak & no_cover).sum())
    return out
