"""
Edge/boundary-sensitive reconstruction metrics.

Separate from py_core/metrics.py, which holds the instance-segmentation
IoU/AP machinery used by the real-data pipeline. Nothing here modifies that.

Metric family follows Diff_QEP (relative L1 / L2 / Linf, as logged in
Diff_QEP/src/qEPsolver.py) plus edge-band and boundary metrics that are
specific to the cell-segmentation question.

Design rule: every threshold used here is fixed a priori from the synthetic
generator (never tuned per method), so no metric can be gamed by one arm.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, sobel
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Global reconstruction error
# ─────────────────────────────────────────────────────────────────────────────

def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def rel_l1(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L1 error: sum|e| / sum|truth|. Matches Diff_QEP 'RL1'."""
    return float(np.abs(pred - truth).sum() / np.abs(truth).sum())


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 error: ||e||_2 / ||truth||_2. Matches Diff_QEP 'RL2'."""
    return float(np.sqrt(((pred - truth) ** 2).sum() / (truth ** 2).sum()))


def rel_linf(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative Linf error: max|e| / max|truth|. Matches Diff_QEP 'RLI'."""
    return float(np.abs(pred - truth).max() / np.abs(truth).max())


# ─────────────────────────────────────────────────────────────────────────────
# Edge band
# ─────────────────────────────────────────────────────────────────────────────

def make_edge_band(labels: np.ndarray, band_width: int = 2) -> np.ndarray:
    """Binary mask of an edge band of half-width `band_width` px around every
    true object boundary (object/background AND object/object contacts).

    `labels` is the integer instance mask; find_boundaries(mode='thick') marks
    pixels adjacent to a label change, which includes touching-cell interfaces.
    """
    bnd = find_boundaries(labels, mode="thick")
    if band_width <= 0:
        return bnd
    return binary_dilation(bnd, iterations=int(band_width))


def edge_band_errors(pred: np.ndarray, truth: np.ndarray,
                     band: np.ndarray) -> Dict[str, float]:
    """RMSE inside the edge band vs outside it, and their ratio.

    The ratio is the quantity of interest: it says how much of a method's
    error is concentrated at boundaries rather than spread over flat regions.
    """
    e = pred - truth
    inside = e[band]
    outside = e[~band]
    edge = float(np.sqrt(np.mean(inside ** 2))) if inside.size else float("nan")
    interior = float(np.sqrt(np.mean(outside ** 2))) if outside.size else float("nan")
    return {
        "edge_band_rmse": edge,
        "interior_rmse": interior,
        "edge_interior_ratio": float(edge / interior) if interior > 0 else float("nan"),
        "edge_band_px": int(band.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gradient / sharpness
# ─────────────────────────────────────────────────────────────────────────────

def grad_magnitude(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=0, mode="reflect")
    gy = sobel(img, axis=1, mode="reflect")
    return np.sqrt(gx ** 2 + gy ** 2)


def sharpness_metrics(pred: np.ndarray, truth: np.ndarray,
                      band: np.ndarray) -> Dict[str, float]:
    """How well the reconstruction reproduces the true edge gradient.

    edge_sharpness_ratio = mean|grad pred| / mean|grad truth| inside the band.
    1.0 means edge steepness matched; < 1 means the edge was blurred out.
    Overshoot (> 1) is possible via ringing, so it is reported, not scored.
    """
    gp = grad_magnitude(pred)
    gt = grad_magnitude(truth)
    gp_b, gt_b = gp[band], gt[band]
    denom = float(gt_b.mean())
    out = {
        "edge_sharpness_ratio": float(gp_b.mean() / denom) if denom > 0 else float("nan"),
        "grad_rmse_band": float(np.sqrt(np.mean((gp_b - gt_b) ** 2))),
    }
    a, b = gp.ravel(), gt.ravel()
    if a.std() > 0 and b.std() > 0:
        out["grad_corr"] = float(np.corrcoef(a, b)[0, 1])
    else:
        out["grad_corr"] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Boundary F1 and foreground IoU
# ─────────────────────────────────────────────────────────────────────────────

def boundary_f1(pred: np.ndarray, truth_fg: np.ndarray, threshold: float,
                tol: int = 2) -> Dict[str, float]:
    """Boundary precision / recall / F1 with a `tol`-pixel matching tolerance.

    The reconstruction is binarized at `threshold` (fixed a priori, identical
    for every arm), its outer boundary is extracted, and boundary pixels are
    matched to the true foreground boundary within `tol` pixels using a
    distance transform.

    precision = fraction of predicted boundary px within tol of a true one
    recall    = fraction of true boundary px within tol of a predicted one
    """
    pred_fg = pred > threshold
    pb = find_boundaries(pred_fg, mode="outer")
    tb = find_boundaries(truth_fg, mode="outer")

    if pb.sum() == 0 or tb.sum() == 0:
        return {
            "boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0,
            "pred_boundary_px": int(pb.sum()), "true_boundary_px": int(tb.sum()),
        }

    # distance to nearest TRUE boundary pixel, evaluated at predicted ones
    dist_to_true = distance_transform_edt(~tb)
    precision = float((dist_to_true[pb] <= tol).mean())
    # distance to nearest PRED boundary pixel, evaluated at true ones
    dist_to_pred = distance_transform_edt(~pb)
    recall = float((dist_to_pred[tb] <= tol).mean())

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": float(f1),
        "pred_boundary_px": int(pb.sum()),
        "true_boundary_px": int(tb.sum()),
    }


def foreground_iou(pred: np.ndarray, truth_fg: np.ndarray, threshold: float) -> float:
    p = pred > threshold
    inter = np.logical_and(p, truth_fg).sum()
    union = np.logical_or(p, truth_fg).sum()
    return float(inter / union) if union > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_reconstruction(
    pred: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    truth_fg: np.ndarray,
    *,
    fixed_threshold: float,
    band_width: int = 2,
    boundary_tol: int = 2,
) -> Dict[str, float]:
    """All metrics for one reconstruction. Returns a flat dict for CSV export."""
    if not np.isfinite(pred).all():
        return {"invalid": 1.0}

    band = make_edge_band(labels, band_width=band_width)

    out: Dict[str, float] = {
        "rmse": rmse(pred, truth),
        "rel_l1": rel_l1(pred, truth),
        "rel_l2": rel_l2(pred, truth),
        "rel_linf": rel_linf(pred, truth),
    }
    out.update(edge_band_errors(pred, truth, band))
    out.update(sharpness_metrics(pred, truth, band))
    out.update(boundary_f1(pred, truth_fg, fixed_threshold, tol=boundary_tol))
    out["fg_iou_fixed_thr"] = foreground_iou(pred, truth_fg, fixed_threshold)

    # Secondary, threshold-free variant: Otsu chosen per reconstruction.
    # Reported alongside the fixed-threshold numbers so a method is not
    # penalized purely for an intensity offset.
    try:
        otsu = float(threshold_otsu(pred))
        b_otsu = boundary_f1(pred, truth_fg, otsu, tol=boundary_tol)
        out["boundary_f1_otsu"] = b_otsu["boundary_f1"]
        out["fg_iou_otsu"] = foreground_iou(pred, truth_fg, otsu)
        out["otsu_threshold"] = otsu
    except Exception:  # noqa: BLE001
        out["boundary_f1_otsu"] = float("nan")
        out["fg_iou_otsu"] = float("nan")
        out["otsu_threshold"] = float("nan")

    out["invalid"] = 0.0
    return out
