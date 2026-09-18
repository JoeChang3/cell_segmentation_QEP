"""
Edge-aware synthetic cell benchmark: controlled QEP-vs-Gaussian comparison.

QUESTION
--------
Does q < 2 preserve sharp cell boundaries better than q = 2, when the
comparison differs *primarily in q*?

WHY THIS BENCHMARK EXISTS
-------------------------
The linear-diffusion benchmark (experiments/simulated/linear_diffusion_*.py)
is retained untouched as a negative control. It cannot answer the question
above: its truth field is numerically rank-1 (singular values
[140.99, 3.57, 0.99, 0.60, 0, 0]) and edge-free apart from one degenerate
t=0 column that accounts for ~89% of the measured QEP MSE. A q<2 prior has
no sharp feature there to preserve.

This benchmark instead produces a piecewise-constant field of cell-like
objects: isolated cells, an overlapping/touching pair with differing
intensities (so the contact interface is a real intensity edge), a
near-touching pair separated by a ~2px background gap, a rotated ellipse,
and objects spanning radii of ~3-11 px. No PDE is involved.

CONTROLLED COMPARISON
---------------------
Every arm shares: image, noise realization (per seed), coordinate
representation, inducing-point initialization (deterministic grid), kernel
family and constraints, mean function, hyperparameter initialization,
optimizer, learning rate, iteration count, jitter, and float64 precision.
The only thing that varies across the QEP arms is `power`.

POWER SEMANTICS: verified empirically against the installed qpytorch, not
assumed. `power` is q directly; power=2.0 reproduces the Gaussian log_prob
and expected_log_prob exactly. See py_core/qep_variational_2d.py for the
derivation of the l^q residual penalty this induces.

An independent `gp_control` arm (gpytorch ApproximateGP + GaussianLikelihood)
is included to confirm that the q=2 QEP arm really does reduce to the
Gaussian case rather than merely being labelled as such.

ARCHITECTURE
------------
Joint 2D variational QEP on X=(x,y) coordinates with learnable inducing
points, adapted from Diff_QEP/src/qEPsolver.py and
Diff_QEP/demo/demo_QEP_diff2d_variational.py. The Diff_QEP GPyTorch fork is
NOT vendored; this runs against the installed `qpytorch`.

The previous row-only / row+column 1D factorization is deliberately not used
as a method here (see the diagnosis: it cannot represent 2D geometry and its
per-slice z-scoring is not a valid separable posterior).

USAGE
-----
    python experiments/simulated/cell_blobs_qep_benchmark.py
    python experiments/simulated/cell_blobs_qep_benchmark.py --seeds 1 2 3
    python experiments/simulated/cell_blobs_qep_benchmark.py --smoke

Outputs go to results/cell_blobs_qep/ . Nothing under
results/common_scale_linear_diffusion* is read or written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Repo root on the path so `py_core` imports work when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import torch
from skimage.segmentation import find_boundaries

from py_core.edge_metrics import (
    evaluate_reconstruction,
    grad_magnitude,
    make_edge_band,
)
from py_core.qep_variational_2d import fit_variational_2d, make_inducing_grid


OUT_DIR = os.path.join(_ROOT, "results", "cell_blobs_qep")


# ═════════════════════════════════════════════════════════════════════════════
# 1. Synthetic cell-like image
# ═════════════════════════════════════════════════════════════════════════════

BACKGROUND = 0.05

# Deterministic geometry: (cx, cy) centre and (a, b) semi-axes in normalized
# [0,1] coordinates, theta in degrees, I = object intensity.
# Coordinates are (row, col) to match the image and the model's X=(x,y).
# Sizes are chosen so that at the default 64x64 grid every object has a
# semi-axis of at least ~2.7 px, the touching pair overlaps by ~1.4 px, and the
# near-touching pair is separated by a ~1.9 px background gap. Radii in px are
# quoted for grid=64.
CELL_SPECS: List[Dict] = [
    # isolated, large (r ~ 6.7 px)
    dict(name="isolated_large",  cx=0.20, cy=0.22, a=0.105, b=0.105, theta=0.0,   I=1.00),
    # isolated, medium (r ~ 4.5 px)
    dict(name="isolated_medium", cx=0.78, cy=0.18, a=0.070, b=0.070, theta=0.0,   I=0.85),
    # touching pair: centres 0.138 apart, radii sum 0.160 -> ~1.4 px overlap.
    # Different intensities, so the contact interface is a genuine intensity edge.
    dict(name="touch_pair_A",    cx=0.44, cy=0.515, a=0.080, b=0.080, theta=0.0,  I=0.95),
    dict(name="touch_pair_B",    cx=0.578, cy=0.515, a=0.080, b=0.080, theta=0.0, I=0.72),
    # near-touching pair: centres 0.135 apart, radii sum 0.105 -> ~1.9 px gap
    dict(name="near_pair_A",     cx=0.17, cy=0.73, a=0.0525, b=0.0525, theta=0.0, I=0.90),
    dict(name="near_pair_B",     cx=0.305, cy=0.73, a=0.0525, b=0.0525, theta=0.0, I=0.78),
    # rotated ellipse (7.4 x 3.5 px)
    dict(name="rot_ellipse",     cx=0.80, cy=0.62, a=0.115, b=0.055, theta=35.0,  I=0.88),
    # small circle (r ~ 3.3 px)
    dict(name="small_circle",    cx=0.58, cy=0.87, a=0.052, b=0.052, theta=0.0,   I=1.00),
    # small rotated ellipse (2.7 x 3.7 px)
    dict(name="small_ellipse",   cx=0.90, cy=0.90, a=0.042, b=0.058, theta=-20.0, I=0.75),
]

# Region of interest for the zoomed touching-cell figure, in normalized coords
# (row_lo, row_hi, col_lo, col_hi) around touch_pair_A / touch_pair_B.
# NOTE on convention: in generate_cell_blobs, `cx` is the ROW coordinate and
# `cy` is the COLUMN coordinate. The touching pair therefore varies in row at a
# fixed column, so its contact interface is horizontal and must be crossed by a
# VERTICAL (along-row) intensity profile.
TOUCH_ROI = (0.32, 0.70, 0.38, 0.66)
# Column at which the profile across the touching-cell contact is taken.
TOUCH_PROFILE_COL = 0.515
# Row of the contact interface, midway between the two centres.
TOUCH_INTERFACE_ROW = 0.509


@dataclass
class CellBlobData:
    truth: np.ndarray        # (H, W) clean piecewise-constant intensity
    labels: np.ndarray       # (H, W) int instance labels, 0 = background
    fg: np.ndarray           # (H, W) bool foreground
    boundary: np.ndarray     # (H, W) bool thick boundaries incl. contacts
    edge_band: np.ndarray    # (H, W) bool dilated edge band
    fixed_threshold: float   # a-priori binarization threshold
    H: int
    W: int
    specs: List[Dict]


def generate_cell_blobs(H: int = 96, W: int = 96, band_width: int = 2) -> CellBlobData:
    """Piecewise-constant cell-like image. Fully deterministic (no RNG).

    Objects are drawn in list order; later objects overwrite earlier ones,
    which is what creates the sharp interface inside the touching pair.
    """
    rr = np.linspace(0.0, 1.0, H)[:, None]      # row coordinate, matches X[:,0]
    cc = np.linspace(0.0, 1.0, W)[None, :]      # col coordinate, matches X[:,1]

    truth = np.full((H, W), BACKGROUND, dtype=np.float64)
    labels = np.zeros((H, W), dtype=np.int32)

    for idx, s in enumerate(CELL_SPECS, start=1):
        th = np.deg2rad(s["theta"])
        dr = rr - s["cx"]
        dc = cc - s["cy"]
        # rotate into the ellipse frame
        u = dr * np.cos(th) + dc * np.sin(th)
        v = -dr * np.sin(th) + dc * np.cos(th)
        inside = (u / s["a"]) ** 2 + (v / s["b"]) ** 2 <= 1.0
        truth[inside] = s["I"]
        labels[inside] = idx

    fg = labels > 0
    boundary = find_boundaries(labels, mode="thick")
    edge_band = make_edge_band(labels, band_width=band_width)

    # Fixed threshold: midway between background and the dimmest object.
    min_obj_I = min(s["I"] for s in CELL_SPECS)
    fixed_threshold = float(0.5 * (BACKGROUND + min_obj_I))

    return CellBlobData(
        truth=truth, labels=labels, fg=fg, boundary=boundary,
        edge_band=edge_band, fixed_threshold=fixed_threshold,
        H=H, W=W, specs=CELL_SPECS,
    )


def add_noise(truth: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Additive i.i.d. Gaussian noise. Identical realization for every arm at
    a given (sigma, seed), so arms differ only in q."""
    rng = np.random.default_rng(seed)
    return truth + rng.normal(loc=0.0, scale=sigma, size=truth.shape)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Arms
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Arm:
    key: str
    label: str
    power: Optional[float]     # None -> independent Gaussian control
    is_probe: bool = False     # True -> outside the standard Q-EP range q in (0,2]


def build_arms(q_values: List[float], include_gp_control: bool = True) -> List[Arm]:
    """One arm per q, plus an optional independent Gaussian control.

    Any q > 2 is flagged as a mechanism probe, not a Q-EP model: the standard
    Q-EP range is q in (0, 2]. It is included because q enters qpytorch's
    likelihood as an l^q penalty on standardized residuals, so q > 2 gives a
    SUPER-quadratic penalty. If that direction sharpens edges while q < 2 blurs
    them, the l^q exponent is confirmed as the operative mechanism and its sign
    is opposite to the Besov/edge-preserving intuition.
    """
    arms = []
    for q in q_values:
        probe = q > 2.0
        arms.append(Arm(
            key=f"qep_q{q:.1f}".replace(".", "p"),
            label=f"q={q:.1f}" + (" (probe)" if probe else ""),
            power=q, is_probe=probe,
        ))
    if include_gp_control:
        arms.append(Arm(key="gp_control", label="GP control", power=None))
    return arms


# ═════════════════════════════════════════════════════════════════════════════
# 3. Figures
# ═════════════════════════════════════════════════════════════════════════════

def _roi_slice(data: CellBlobData, roi: Tuple[float, float, float, float]):
    r0 = int(roi[0] * (data.H - 1)); r1 = int(roi[1] * (data.H - 1))
    c0 = int(roi[2] * (data.W - 1)); c1 = int(roi[3] * (data.W - 1))
    return slice(r0, r1 + 1), slice(c0, c1 + 1), (r0, r1, c0, c1)


def fig_main_panels(data: CellBlobData, noisy: np.ndarray,
                    preds: Dict[str, np.ndarray], arms: List[Arm],
                    sigma: float, seed: int, out_png: str) -> None:
    """Truth, noisy, and one reconstruction per arm on a COMMON intensity scale."""
    vmin, vmax = float(data.truth.min()), float(data.truth.max())
    panels = [("Ground truth", data.truth), (f"Noisy  ($\\sigma$={sigma})", noisy)]
    panels += [(a.label, preds[a.key]) for a in arms if a.key in preds]

    n = len(panels)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow))
    axs = np.atleast_1d(axs).ravel()
    im = None
    for ax, (title, img) in zip(axs, panels):
        im = ax.imshow(img, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[n:]:
        ax.axis("off")
    fig.suptitle(f"Cell-blob benchmark: reconstructions (common scale [{vmin:.2f}, {vmax:.2f}])"
                 f"   seed={seed}", fontweight="bold", fontsize=13)
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    fig.colorbar(im, cax=cax)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_error_maps(data: CellBlobData, preds: Dict[str, np.ndarray],
                   arms: List[Arm], seed: int, out_png: str) -> None:
    """Signed error maps on a common symmetric scale, with true boundaries overlaid."""
    errs = {k: v - data.truth for k, v in preds.items()}
    amax = max(float(np.abs(e).max()) for e in errs.values()) if errs else 1.0
    keys = [a for a in arms if a.key in errs]
    ncol = min(4, len(keys)); nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow))
    axs = np.atleast_1d(axs).ravel()
    im = None
    for ax, a in zip(axs, keys):
        im = ax.imshow(errs[a.key], cmap="RdBu_r", origin="lower", vmin=-amax, vmax=amax)
        ax.contour(data.boundary.astype(float), levels=[0.5], colors="k",
                   linewidths=0.4, alpha=0.6)
        ax.set_title(f"{a.label}\nRMSE={np.sqrt(np.mean(errs[a.key]**2)):.4f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"Reconstruction error (pred - truth), common scale $\\pm${amax:.2f}"
                 f"   seed={seed}", fontweight="bold", fontsize=13)
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    fig.colorbar(im, cax=cax)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_zoom_touching(data: CellBlobData, noisy: np.ndarray,
                      preds: Dict[str, np.ndarray], arms: List[Arm],
                      seed: int, out_png: str) -> None:
    """Zoom on the touching-cell pair + an intensity profile across the contact."""
    rs, cs, (r0, r1, c0, c1) = _roi_slice(data, TOUCH_ROI)
    vmin, vmax = float(data.truth.min()), float(data.truth.max())

    keys = [a for a in arms if a.key in preds]
    ncol = 2 + len(keys)
    fig = plt.figure(figsize=(2.6 * ncol, 5.8))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.25, 1.0], hspace=0.32)

    for j, (title, img) in enumerate([("Ground truth", data.truth),
                                      ("Noisy", noisy)] +
                                     [(a.label, preds[a.key]) for a in keys]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img[rs, cs], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Vertical (along-row) profile through the touching pair. The pair varies in
    # row at fixed column, so this is the cut that actually crosses the contact.
    col = int(TOUCH_PROFILE_COL * (data.W - 1))
    iface = TOUCH_INTERFACE_ROW * (data.H - 1)
    ax = fig.add_subplot(gs[1, :])
    xs = np.arange(r0, r1 + 1)
    ax.plot(xs, data.truth[rs, col], "k-", lw=2.4, label="truth", zorder=5)
    ax.plot(xs, noisy[rs, col], color="0.6", lw=0.8, marker=".", ms=3,
            label="noisy", zorder=1)
    for a in keys:
        ax.plot(xs, preds[a.key][rs, col], lw=1.6, label=a.label, zorder=3)
    ax.axvline(iface, color="crimson", ls=":", lw=1.4, zorder=2,
               label="contact interface")
    ax.set_xlabel(f"row index (profile down column {col}, through the touching pair)")
    ax.set_ylabel("intensity")
    ax.set_title("Intensity profile across the touching-cell contact "
                 "(steeper transition = better edge preservation)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.suptitle(f"Touching-cell pair, zoomed   seed={seed}", fontweight="bold", fontsize=13)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_gradient_maps(data: CellBlobData, preds: Dict[str, np.ndarray],
                      arms: List[Arm], seed: int, out_png: str) -> None:
    """Sobel gradient magnitude: truth vs each arm, common scale."""
    gt = grad_magnitude(data.truth)
    gs_ = {k: grad_magnitude(v) for k, v in preds.items()}
    vmax = float(gt.max())
    keys = [a for a in arms if a.key in preds]
    panels = [("Ground truth", gt)] + [(a.label, gs_[a.key]) for a in keys]
    ncol = min(4, len(panels)); nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow))
    axs = np.atleast_1d(axs).ravel()
    im = None
    for ax, (title, img) in zip(axs, panels):
        im = ax.imshow(img, cmap="magma", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Gradient magnitude $|\\nabla u|$ (common scale)   seed={seed}",
                 fontweight="bold", fontsize=13)
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    fig.colorbar(im, cax=cax)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_vs_q(df: pd.DataFrame, out_png: str) -> None:
    """Metric-vs-q curves, averaged over seeds, with per-seed spread."""
    qdf = df[df["power"].notna()].copy()
    if qdf.empty:
        return
    metrics = [
        ("rmse", "global RMSE", "lower better"),
        ("edge_band_rmse", "edge-band RMSE", "lower better"),
        ("interior_rmse", "interior RMSE", "lower better"),
        ("rel_linf", "relative $L_\\infty$", "lower better"),
        ("edge_sharpness_ratio", "edge sharpness ratio\n($|\\nabla$pred$|/|\\nabla$truth$|$, 1=matched)", "higher better (<=1)"),
        ("boundary_f1", "boundary F1 (fixed thr)", "higher better"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 7.4))
    axs = axs.ravel()
    # select the reference arms by name: power is NaN for BOTH gp_control and
    # identity_noisy, so filtering on power alone would conflate them
    gp = df[df["arm"] == "gp_control"]
    ident = df[df["arm"] == "identity_noisy"]
    for ax, (col, name, note) in zip(axs, metrics):
        if col not in qdf:
            ax.axis("off"); continue
        qs = np.array(sorted(qdf["power"].unique()))
        mean = np.array([qdf.loc[qdf["power"] == q, col].mean() for q in qs])
        ax.plot(qs, mean, "o-", color="C0", lw=2, ms=6, label="QEP arms (seed mean)")
        m2 = qdf.loc[qdf["power"] == 2.0, col]
        if not m2.empty:
            ax.plot([2.0], [m2.mean()], "s", ms=12, mfc="none", mec="C3", mew=2.2,
                    label="q=2 (Gaussian)")
        for _, sub in qdf.groupby("seed"):
            sub = sub.sort_values("power")
            ax.plot(sub["power"], sub[col], "-", color="C0", alpha=0.22, lw=1)
        if not gp.empty and col in gp:
            ax.axhline(gp[col].mean(), color="C3", ls="--", lw=1.4,
                       label="independent GP control")
        if not ident.empty and col in ident:
            ax.axhline(ident[col].mean(), color="0.45", ls=":", lw=1.6,
                       label="no smoothing")
        ax.set_xlabel("q (POWER)")
        ax.set_title(f"{name}\n({note})", fontsize=10)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("Metrics vs q — cell-blob benchmark (q=2.0 is the Gaussian case)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dataset_overview(data: CellBlobData, noisy: np.ndarray, sigma: float,
                         inducing: np.ndarray, out_png: str) -> None:
    """Ground truth, labels, edge band, noisy, and inducing-point layout."""
    fig, axs = plt.subplots(1, 5, figsize=(19, 3.9))
    axs[0].imshow(data.truth, cmap="viridis", origin="lower")
    axs[0].set_title(f"Clean truth (bg={BACKGROUND}, {len(data.specs)} cells)")
    axs[1].imshow(data.labels, cmap="tab20", origin="lower")
    axs[1].set_title("Instance labels")
    axs[2].imshow(data.edge_band, cmap="gray", origin="lower")
    axs[2].set_title(f"Edge band ({int(data.edge_band.sum())} px)")
    axs[3].imshow(noisy, cmap="viridis", origin="lower")
    axs[3].set_title(f"Noisy obs ($\\sigma$={sigma})")
    axs[4].imshow(data.truth, cmap="Greys_r", origin="lower", alpha=0.85)
    axs[4].scatter(inducing[:, 1] * (data.W - 1), inducing[:, 0] * (data.H - 1),
                   s=5, c="red", marker="x", linewidths=0.6)
    axs[4].set_title(f"Inducing init ({inducing.shape[0]} pts)")
    rs, cs, (r0, r1, c0, c1) = _roi_slice(data, TOUCH_ROI)
    axs[0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                               edgecolor="red", lw=1.4, ls="--"))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Cell-blob benchmark dataset (red box = touching-pair ROI)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_loss_curves(fits: Dict[str, object], arms: List[Arm], seed: int,
                    out_png: str) -> None:
    """Convergence traces. NOTE: ELBO values are not comparable across q
    (the q-EP density adds q-dependent constants), so this is for convergence
    diagnosis only."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for a in arms:
        f = fits.get(a.key)
        if f is None:
            continue
        lh = f.loss_history
        it = np.arange(1, lh.size + 1)
        m = np.isfinite(lh)
        ax.plot(it[m], lh[m], lw=1.3, label=f"{a.label} (final={f.final_loss:+.4f})")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("-ELBO")
    ax.set_title("Convergence (NOT comparable across q: q-EP adds q-dependent constants)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"Optimization traces   seed={seed}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Driver
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, default=64, help="image side length")
    p.add_argument("--sigma", type=float, nargs="+", default=[0.10, 0.25],
                   help="Gaussian noise std(s). 0.10 is high-SNR (object "
                        "contrast/sigma ~ 6.7, where thresholding the raw image "
                        "already gives boundary F1 ~ 0.98, so boundary metrics "
                        "saturate); 0.25 is the discriminative regime.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--q", type=float, nargs="+",
                   default=[3.0, 2.0, 1.8, 1.5, 1.2, 1.0],
                   help="q (POWER) values; 2.0 is the Gaussian case. Values >2 are "
                        "outside the standard Q-EP range and are run only as a "
                        "mechanism probe (see build_arms).")
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--inference", choices=["variational", "exact"],
                   default="variational",
                   help="variational = inducing points + ELBO (primary); "
                        "exact = full marginal likelihood, no inducing "
                        "approximation (cross-check)")
    p.add_argument("--inducing-per-dim", type=int, default=32,
                   help="inducing grid side; 32 on a 64px grid = 2px spacing, "
                        "fine enough to represent a 1px edge")
    p.add_argument("--init-lengthscale", type=float, default=0.04)
    p.add_argument("--nu", type=float, default=2.5)
    p.add_argument("--band-width", type=int, default=2)
    p.add_argument("--boundary-tol", type=int, default=2)
    p.add_argument("--no-gp-control", action="store_true")
    p.add_argument("--fix-noise", action="store_true",
                   help="Freeze the observation noise at the true sigma instead "
                        "of learning it. Use this to remove the q<2 "
                        "interpolation singularity: for q<2 the q-EP density "
                        "diverges as the residual -> 0, so type-II MLE/ELBO is "
                        "ill-posed and drives noise -> 0. Freezing noise "
                        "isolates the effect of q itself.")
    p.add_argument("--out", type=str, default=OUT_DIR)
    p.add_argument("--tag", type=str, default="",
                   help="suffix for the output subdirectory")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to validate the pipeline end to end")
    args = p.parse_args()

    if args.smoke:
        args.grid, args.iters, args.seeds = 48, 60, [1]
        args.q, args.inducing_per_dim = [2.0, 1.2], 10
    if args.tag:
        args.out = f"{args.out}_{args.tag}"

    torch.set_default_dtype(torch.float64)
    os.makedirs(args.out, exist_ok=True)
    npz_dir = os.path.join(args.out, "npz")
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    data = generate_cell_blobs(H=args.grid, W=args.grid, band_width=args.band_width)
    arms = build_arms(args.q, include_gp_control=not args.no_gp_control)
    inducing_init = make_inducing_grid(args.inducing_per_dim).numpy()

    print("=" * 78)
    print("Cell-blob edge-aware benchmark: controlled QEP vs Gaussian")
    print("=" * 78)
    print(f"  grid          : {args.grid}x{args.grid}  ({args.grid**2} training points)")
    print(f"  cells         : {len(CELL_SPECS)}  (intensities "
          f"{min(s['I'] for s in CELL_SPECS):.2f}-{max(s['I'] for s in CELL_SPECS):.2f}, "
          f"bg={BACKGROUND})")
    print(f"  truth range   : [{data.truth.min():.3f}, {data.truth.max():.3f}]  "
          f"std={data.truth.std():.4f}")
    print(f"  edge band     : {int(data.edge_band.sum())} px "
          f"({100*data.edge_band.mean():.1f}% of image), half-width={args.band_width}")
    print(f"  fixed thr     : {data.fixed_threshold:.4f} (a priori, same for all arms)")
    min_contrast = min(s["I"] for s in CELL_SPECS) - BACKGROUND
    print(f"  noise sigma   : {args.sigma}  "
          f"(min object contrast {min_contrast:.2f} -> contrast/sigma "
          f"{[round(min_contrast/s, 1) for s in args.sigma]})")
    print(f"  seeds         : {args.seeds}")
    print(f"  arms          : {[a.label for a in arms]}")
    print(f"  inference     : {args.inference}")
    if args.inference == "variational":
        print(f"  inducing      : {args.inducing_per_dim}^2 = {inducing_init.shape[0]} "
              f"(deterministic grid, learnable), spacing "
              f"{args.grid/args.inducing_per_dim:.1f} px")
    else:
        print("  inducing      : none (exact marginal likelihood, no low-rank "
              "approximation)")
    print(f"  kernel        : ScaleKernel(Matern nu={args.nu}, ard_num_dims=2), "
          f"ls in (1e-3, 1.0)")
    print(f"  init          : ls={args.init_lengthscale}, os=1.0, noise=sigma")
    print(f"  optimizer     : Adam lr={args.lr}, {args.iters} iters, float64")
    print(f"  noise         : {'LEARNED' if not args.fix_noise else 'FROZEN at true sigma'}"
          + ("" if args.fix_noise else
             "  (note: for q<2 the q-EP marginal likelihood is unbounded as the\n                  fit approaches interpolation, so a learned noise can collapse\n                  toward 0; rerun with --fix-noise to isolate q)"))
    print("=" * 78, flush=True)

    # Truth-side sanity: this benchmark must NOT be near-rank-1 like linear diffusion.
    sv = np.linalg.svd(data.truth, compute_uv=False)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    rank999 = int(np.searchsorted(energy, 0.999) + 1)
    print(f"[truth structure] top-6 singular values: {np.round(sv[:6], 3)}")
    print(f"[truth structure] effective rank (99.9% energy) = {rank999}   "
          f"(linear diffusion was 1)")
    print("=" * 78, flush=True)

    rows: List[Dict] = []
    t_all = time.time()

    for sigma, seed in [(s, sd) for s in args.sigma for sd in args.seeds]:
        noisy = add_noise(data.truth, sigma, seed)
        print(f"\n### sigma={sigma}  seed={seed}   "
              f"noisy range [{noisy.min():.3f}, {noisy.max():.3f}]")

        # identity ("do nothing") reference, so every arm can be judged
        # against not smoothing at all
        ident = evaluate_reconstruction(
            noisy, data.truth, data.labels, data.fg,
            fixed_threshold=data.fixed_threshold,
            band_width=args.band_width, boundary_tol=args.boundary_tol)
        rows.append(dict(arm="identity_noisy", label="no smoothing",
                         is_probe=0, power=np.nan,
                         seed=seed, sigma=sigma, **ident,
                         runtime_s=0.0, final_loss=np.nan, n_iters=0,
                         lengthscale_0=np.nan, lengthscale_1=np.nan,
                         outputscale=np.nan, noise=np.nan,
                         n_nonfinite_loss=0, n_cholesky_retries=0,
                         failed=0, failure_reason="", inference=args.inference))
        print(f"    [identity] RMSE={ident['rmse']:.4f}  "
              f"edge_band_RMSE={ident['edge_band_rmse']:.4f}  "
              f"boundary_F1={ident['boundary_f1']:.4f}")

        preds: Dict[str, np.ndarray] = {}
        fits: Dict[str, object] = {}

        for arm in arms:
            print(f"\n  --- arm {arm.key}  ({arm.label})"
                  f"{'  [independent Gaussian control]' if arm.power is None else ''}")
            fit = fit_variational_2d(
                noisy,
                power=arm.power,
                inducing_per_dim=args.inducing_per_dim,
                nu=args.nu,
                init_lengthscale=args.init_lengthscale,
                init_outputscale=1.0,
                init_noise=sigma,
                lr=args.lr,
                train_iters=args.iters,
                inference=args.inference,
                learn_noise=not args.fix_noise,
                log_every=max(1, args.iters // 6),
                tag=arm.key,
            )
            fits[arm.key] = fit

            if fit.failed:
                print(f"    !! FAILED: {fit.failure_reason}")
                m = {"invalid": 1.0}
            else:
                preds[arm.key] = fit.pred
                m = evaluate_reconstruction(
                    fit.pred, data.truth, data.labels, data.fg,
                    fixed_threshold=data.fixed_threshold,
                    band_width=args.band_width, boundary_tol=args.boundary_tol)
                tif = fit.diagnostics.get("tail_improve_frac", float("nan"))
                print(f"    RMSE={m['rmse']:.4f}  edge_band={m['edge_band_rmse']:.4f}  "
                      f"interior={m['interior_rmse']:.4f}  relLinf={m['rel_linf']:.4f}  "
                      f"sharp={m['edge_sharpness_ratio']:.3f}  "
                      f"bF1={m['boundary_f1']:.4f}  ({fit.runtime_s:.0f}s)")
                print(f"    ls=({fit.lengthscale[0]:.4f},{fit.lengthscale[1]:.4f})  "
                      f"noise={fit.noise:.5f}  os={fit.outputscale:.4f}  "
                      f"tail_improve={tif:.4f}"
                      f"{'  [NOT CONVERGED]' if (np.isfinite(tif) and tif >= 0.01) else ''}")

            rows.append(dict(
                arm=arm.key, label=arm.label, is_probe=int(arm.is_probe),
                power=(arm.power if arm.power is not None else np.nan),
                seed=seed, sigma=sigma, **m,
                runtime_s=fit.runtime_s, final_loss=fit.final_loss,
                n_iters=fit.n_iters,
                lengthscale_0=float(fit.lengthscale[0]),
                lengthscale_1=float(fit.lengthscale[1]),
                outputscale=fit.outputscale, noise=fit.noise,
                n_nonfinite_loss=fit.n_nonfinite_loss,
                n_cholesky_retries=fit.n_cholesky_retries,
                failed=int(fit.failed), failure_reason=fit.failure_reason,
                inference=fit.diagnostics.get("inference"),
                num_inducing=fit.diagnostics.get("num_inducing"),
                init_lengthscale=fit.diagnostics.get("init_lengthscale"),
                init_noise=fit.diagnostics.get("init_noise"),
                lr=fit.diagnostics.get("lr"), nu=fit.diagnostics.get("nu"),
                dtype=fit.diagnostics.get("dtype"),
                learn_noise=fit.diagnostics.get("learn_noise"),
                n_train=fit.diagnostics.get("n_train"),
                loss_first=fit.diagnostics.get("loss_first"),
                loss_min=fit.diagnostics.get("loss_min"),
                tail_improve_frac=fit.diagnostics.get("tail_improve_frac"),
                converged=fit.diagnostics.get("converged"),
            ))

            # per-arm NPZ
            np.savez_compressed(
                os.path.join(npz_dir, f"{arm.key}_sigma{sigma:.2f}_seed{seed}.npz"),
                truth=data.truth, noisy=noisy, labels=data.labels,
                fg=data.fg, boundary=data.boundary, edge_band=data.edge_band,
                pred=fit.pred, var=fit.var, loss_history=fit.loss_history,
                inducing_init=inducing_init, inducing_final=fit.inducing_final,
                lengthscale=fit.lengthscale,
                outputscale=np.float64(fit.outputscale), noise=np.float64(fit.noise),
                power=np.float64(arm.power if arm.power is not None else np.nan),
                sigma=np.float64(sigma), seed=np.int64(seed),
                fixed_threshold=np.float64(data.fixed_threshold),
                n_iters=np.int64(fit.n_iters), runtime_s=np.float64(fit.runtime_s),
                failed=np.int64(int(fit.failed)),
                arm=arm.key, method="variational_qep_2d" if arm.power is not None
                else "variational_gp_2d",
            )

        # ---- figures for this seed ----
        tag = f"sigma{sigma:.2f}_seed{seed}"
        fig_dataset_overview(data, noisy, sigma, inducing_init,
                             os.path.join(fig_dir, f"dataset_overview_{tag}.png"))
        if preds:
            fig_main_panels(data, noisy, preds, arms, sigma, seed,
                            os.path.join(fig_dir, f"reconstructions_{tag}.png"))
            fig_error_maps(data, preds, arms, seed,
                           os.path.join(fig_dir, f"error_maps_{tag}.png"))
            fig_zoom_touching(data, noisy, preds, arms, seed,
                              os.path.join(fig_dir, f"zoom_touching_{tag}.png"))
            fig_gradient_maps(data, preds, arms, seed,
                              os.path.join(fig_dir, f"gradient_maps_{tag}.png"))
        fig_loss_curves(fits, arms, seed,
                        os.path.join(fig_dir, f"loss_curves_{tag}.png"))

    # ═════ tables ═════
    df = pd.DataFrame(rows)
    csv_runs = os.path.join(args.out, "runs.csv")
    df.to_csv(csv_runs, index=False)

    metric_cols = ["rmse", "rel_l1", "rel_l2", "rel_linf", "edge_band_rmse",
                   "interior_rmse", "edge_interior_ratio", "edge_sharpness_ratio",
                   "grad_corr", "boundary_f1", "boundary_precision",
                   "boundary_recall", "boundary_f1_otsu", "fg_iou_fixed_thr",
                   "lengthscale_0", "lengthscale_1", "noise", "outputscale",
                   "runtime_s"]
    metric_cols = [c for c in metric_cols if c in df]
    ok = df[df["failed"] == 0]
    summary = (ok.groupby(["sigma", "arm", "label", "power"], dropna=False)[metric_cols]
               .mean().reset_index()
               .sort_values(["sigma", "power"], ascending=[True, False],
                            na_position="last"))
    csv_summary = os.path.join(args.out, "summary_by_arm.csv")
    summary.to_csv(csv_summary, index=False)

    for sig in sorted(ok["sigma"].unique()):
        fig_metrics_vs_q(ok[ok["sigma"] == sig],
                         os.path.join(fig_dir, f"metrics_vs_q_sigma{sig:.2f}.png"))

    # ═════ config manifest ═════
    manifest = dict(
        experiment="cell_blobs_qep_benchmark",
        question="Does q<2 preserve sharp cell boundaries better than q=2?",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        args=vars(args),
        arms=[dict(key=a.key, label=a.label, power=a.power) for a in arms],
        cell_specs=CELL_SPECS,
        background=BACKGROUND,
        fixed_threshold=data.fixed_threshold,
        touch_roi=TOUCH_ROI,
        truth_singular_values=sv[:8].tolist(),
        truth_effective_rank_999=rank999,
        power_semantics=("power == q directly; verified: power=2.0 reproduces "
                         "gpytorch MultivariateNormal.log_prob and "
                         "GaussianLikelihood.expected_log_prob exactly"),
        elbo_comparability_warning=("ELBO/-loss values are NOT comparable across q: "
                                    "qpytorch's QExponentialLikelihood.expected_log_prob "
                                    "adds 0.5*(q/2-1)*log(r) + log(q/2)"),
        env=dict(
            python=sys.version.split()[0],
            platform=platform.platform(),
            torch=torch.__version__,
            numpy=np.__version__,
        ),
    )
    try:
        import gpytorch as _g, qpytorch as _q
        manifest["env"]["gpytorch"] = _g.__version__
        manifest["env"]["qpytorch"] = _q.__version__
    except Exception:  # noqa: BLE001
        pass
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # ═════ console report ═════
    n_seeds = len(args.seeds)
    for sig in sorted(summary["sigma"].unique()):
        sm = summary[summary["sigma"] == sig]
        print("\n" + "=" * 100)
        print(f"SUMMARY  sigma={sig}  (mean over {n_seeds} seed(s); "
              f"q=2.0 is the Gaussian case; inference={args.inference})")
        print("=" * 100)
        hdr = (f"{'arm':<16}{'q':>5}{'RMSE':>9}{'relL1':>8}{'relL2':>8}{'relLinf':>9}"
               f"{'edgeRMSE':>10}{'intRMSE':>9}{'e/i':>6}{'sharp':>7}{'bF1':>8}"
               f"{'lscale':>8}{'noise':>8}")
        print(hdr); print("-" * len(hdr))
        for _, r in sm.iterrows():
            q = f"{r['power']:.1f}" if np.isfinite(r["power"]) else "  -"
            ls = (f"{r['lengthscale_0']:.4f}" if np.isfinite(r["lengthscale_0"])
                  else "     -")
            nz = f"{r['noise']:.4f}" if np.isfinite(r["noise"]) else "     -"
            print(f"{r['arm']:<16}{q:>5}{r['rmse']:>9.4f}{r['rel_l1']:>8.4f}"
                  f"{r['rel_l2']:>8.4f}{r['rel_linf']:>9.4f}{r['edge_band_rmse']:>10.4f}"
                  f"{r['interior_rmse']:>9.4f}{r['edge_interior_ratio']:>6.2f}"
                  f"{r['edge_sharpness_ratio']:>7.3f}{r['boundary_f1']:>8.4f}"
                  f"{ls:>8}{nz:>8}")

        # explicit verdict, computed not asserted
        qonly = sm[sm["power"].notna()]
        if not qonly.empty and (qonly["power"] == 2.0).any():
            base = qonly[qonly["power"] == 2.0].iloc[0]
            sub = qonly[qonly["power"] < 2.0].sort_values("power", ascending=False)
            print("\n" + "-" * 100)
            print(f"VERDICT vs q=2.0 at sigma={sig}  "
                  f"(negative delta = q<2 better, for error metrics)")
            print("-" * 100)
            for _, r in sub.iterrows():
                print(f"  q={r['power']:.1f}:  dRMSE={r['rmse']-base['rmse']:+.5f}   "
                      f"dEdgeRMSE={r['edge_band_rmse']-base['edge_band_rmse']:+.5f}   "
                      f"dIntRMSE={r['interior_rmse']-base['interior_rmse']:+.5f}   "
                      f"dRelLinf={r['rel_linf']-base['rel_linf']:+.5f}   "
                      f"dSharp={r['edge_sharpness_ratio']-base['edge_sharpness_ratio']:+.4f}   "
                      f"dbF1={r['boundary_f1']-base['boundary_f1']:+.5f}")
            if not sub.empty:
                be = sub.loc[sub["edge_band_rmse"].idxmin()]
                better = be["edge_band_rmse"] < base["edge_band_rmse"]
                print(f"\n  Best q<2 on edge-band RMSE: q={be['power']:.1f} "
                      f"({be['edge_band_rmse']:.5f}) vs q=2.0 "
                      f"({base['edge_band_rmse']:.5f})  ->  "
                      f"{'q<2 WINS at boundaries' if better else 'q=2 IS AS GOOD OR BETTER'}")

        gpc = sm[sm["arm"] == "gp_control"]
        q2 = sm[sm["power"] == 2.0]
        if not gpc.empty and not q2.empty:
            d = abs(float(gpc.iloc[0]["rmse"]) - float(q2.iloc[0]["rmse"]))
            print(f"  Control check |RMSE(gp_control) - RMSE(q=2.0)| = {d:.2e}"
                  f"   (small => the q=2 arm does reduce to the Gaussian case)")

    nf = int(df["n_nonfinite_loss"].sum()); nr = int(df["n_cholesky_retries"].sum())
    nfail = int(df["failed"].sum())
    print(f"\n  Numerical: {nf} non-finite-loss iters, {nr} linalg retries, "
          f"{nfail} failed arms")
    print(f"  Total wall clock: {(time.time()-t_all)/60:.1f} min")
    print("\n  Wrote:")
    print(f"    {csv_runs}")
    print(f"    {csv_summary}")
    print(f"    {os.path.join(args.out, 'config.json')}")
    print(f"    {npz_dir}/*.npz")
    print(f"    {fig_dir}/*.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
