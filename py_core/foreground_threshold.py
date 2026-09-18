"""
Shared foreground-threshold interface for the real cell-segmentation pipeline.

Every rule has the same signature and returns a scalar threshold plus a binary
foreground mask, so the SAME algorithm can be applied to Raw / GP / QEP q=2 /
QEP q=1.5 reconstructions. The numerical threshold naturally differs between
those images because the images differ; the RULE does not.

WHY THIS MODULE EXISTS - WHAT criterion_1 ACTUALLY DOES
-------------------------------------------------------
`criterion_1` (py_core/Modified_Functions_RGasp.py:53) sweeps
p = 0.00, 0.01, ..., 1.00 and for each p counts pixels above
`threshold_image(mat, p)` = `p * nanmax(mat)`. It smooths |diff| of that count
curve with a Gaussian (sigma=2 in index units), takes its argmax, then walks
forward to the first index where consecutive smoothed diffs change by less than
`0.05 * std(diff_sm)`, and returns that p. Four consequences:

  1. ANCHORED TO THE MAXIMUM. The threshold is a fraction of `nanmax`, a
     single-pixel order statistic. One bright outlier rescales the whole sweep.

  2. ASSUMES BACKGROUND NEAR ZERO. The grid spans [0, max] only. For an image
     whose minimum is far above zero (nuclei_figure_1 has range [105, 255]),
     every p below min/max ~ 0.41 selects the entire image, so ~41% of the grid
     is degenerate and the usable resolution is compressed into the remainder.
     For a reconstruction that dips below zero the grid does not even cover the
     data range.

  3. KNEE HEURISTIC WITH A GLOBAL-SCALE TOLERANCE. The stability tolerance is
     0.05 * std of the whole diff curve, so whether the walk terminates depends
     on the global shape of that curve, not on local flatness.

  4. SILENT CATASTROPHIC DEFAULT. If the walk never satisfies the tolerance,
     `found_stable` stays False and the function returns p = 1.0 with an
     ALL-ZERO mask - the entire image is declared background. This is the
     observed failure: 3/16 tiles on a development image, and every tile of
     held-out nuclei_figure_2 under the legacy GP configuration (0 predictions).

Rules provided here are global unless stated. `quantile` and `robust_mad` take
one scalar hyperparameter chosen on development images only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from skimage.filters import (
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

from py_core.Modified_Functions_RGasp import criterion_1


@dataclass
class ThresholdResult:
    threshold: float                 # absolute intensity cut
    mask: np.ndarray                 # bool foreground
    rule: str
    params: Dict = field(default_factory=dict)
    failed: bool = False             # rule could not produce a usable cut
    note: str = ""


def _finish(img: np.ndarray, thr: float, rule: str, params: Dict,
            note: str = "") -> ThresholdResult:
    mask = img > thr
    return ThresholdResult(threshold=float(thr), mask=mask, rule=rule,
                           params=params, failed=False, note=note)


# ─────────────────────────────────────────────────────────────────────────────
# Rules
# ─────────────────────────────────────────────────────────────────────────────

def rule_criterion_1(img: np.ndarray, **_) -> ThresholdResult:
    """The existing rule, wrapped unchanged so it can be compared fairly."""
    c = criterion_1(img, delta=0.01, nugget=True)
    p = float(c.estimated_percentage)
    thr = p * float(np.nanmax(img))
    mask = c.thresholded_image > 0
    # p == 1.0 together with an empty mask is criterion_1's all-background default
    collapsed = bool(p >= 1.0 - 1e-12 and mask.sum() == 0)
    return ThresholdResult(
        threshold=thr, mask=mask, rule="criterion_1",
        params=dict(estimated_percentage=p, delta=0.01),
        failed=collapsed,
        note="all-background default (found_stable=False)" if collapsed else "")


def rule_otsu(img: np.ndarray, **_) -> ThresholdResult:
    return _finish(img, threshold_otsu(img), "otsu", {})


def rule_li(img: np.ndarray, **_) -> ThresholdResult:
    return _finish(img, threshold_li(img), "li", {})


def rule_yen(img: np.ndarray, **_) -> ThresholdResult:
    return _finish(img, threshold_yen(img), "yen", {})


def rule_triangle(img: np.ndarray, **_) -> ThresholdResult:
    return _finish(img, threshold_triangle(img), "triangle", {})


def rule_quantile(img: np.ndarray, p: float = 0.90, **_) -> ThresholdResult:
    """foreground = image > quantile(image, p). Invariant to any strictly
    increasing intensity transform, since it is defined on rank order alone."""
    return _finish(img, float(np.quantile(img, p)), "quantile", dict(p=p))


def rule_robust_mad(img: np.ndarray, c: float = 3.0, **_) -> ThresholdResult:
    """foreground = (I - median) / MAD >= c, with MAD scaled to be a consistent
    estimator of sigma for Gaussian data (factor 1.4826).

    Affine-equivariant: under I' = a*I + b (a>0) both median and MAD transform
    the same way, so the selected pixel set is unchanged.
    """
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med))) * 1.4826
    if mad <= 0:
        # degenerate (more than half the pixels identical): fall back to a
        # small positive offset above the median so the rule still returns
        # something deterministic rather than selecting everything
        sd = float(img.std())
        mad = sd if sd > 0 else 1.0
    return _finish(img, med + c * mad, "robust_mad", dict(c=c, median=med,
                                                          mad=mad))


RULES: Dict[str, Callable[..., ThresholdResult]] = {
    "criterion_1": rule_criterion_1,
    "otsu": rule_otsu,
    "li": rule_li,
    "yen": rule_yen,
    "triangle": rule_triangle,
    "quantile": rule_quantile,
    "robust_mad": rule_robust_mad,
}


def candidate_grid() -> List[Dict]:
    """The small, explicitly listed candidate set evaluated on development
    images. No large hyperparameter search is performed."""
    out: List[Dict] = [
        dict(name="criterion_1", rule="criterion_1", params={}),
        dict(name="otsu", rule="otsu", params={}),
        dict(name="li", rule="li", params={}),
        dict(name="yen", rule="yen", params={}),
        dict(name="triangle", rule="triangle", params={}),
    ]
    for p in (0.80, 0.85, 0.90, 0.95):
        out.append(dict(name=f"quantile_p{p:g}", rule="quantile",
                        params=dict(p=p)))
    for c in (1.5, 2.0, 3.0, 4.0):
        out.append(dict(name=f"robust_mad_c{c:g}", rule="robust_mad",
                        params=dict(c=c)))
    return out


def apply_rule(img: np.ndarray, rule: str, params: Optional[Dict] = None
               ) -> ThresholdResult:
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; have {sorted(RULES)}")
    return RULES[rule](img, **(params or {}))


# ─────────────────────────────────────────────────────────────────────────────
# Foreground quality
# ─────────────────────────────────────────────────────────────────────────────

def foreground_metrics(mask: np.ndarray, gt_fg: np.ndarray) -> Dict[str, float]:
    """Pixel-level foreground agreement. Dice and IoU are the headline numbers;
    precision and recall disambiguate over- from under-detection."""
    m, g = mask > 0, gt_fg > 0
    inter = float(np.logical_and(m, g).sum())
    union = float(np.logical_or(m, g).sum())
    return dict(
        fg_dice=float(2 * inter / (m.sum() + g.sum())) if (m.sum() + g.sum()) else float("nan"),
        fg_iou=float(inter / union) if union else float("nan"),
        fg_precision=float(inter / m.sum()) if m.sum() else 0.0,
        fg_recall=float(inter / g.sum()) if g.sum() else float("nan"),
        fg_fraction=float(m.mean()),
        gt_fg_fraction=float(g.mean()),
    )
