"""
Common-scale comparison figure for linear diffusion QEP variants.

Loads pre-saved .npz arrays from results/common_scale_linear_diffusion/ and
produces two publication figures:

  1. linear_diffusion_qep_common_scale.png / .pdf
       Truth | Noisy obs | Row-only QEP | Row+Col QEP | 2D SKI-QEP
       All panels share vmin=0, vmax=1 with a single colorbar.

  2. linear_diffusion_qep_error_maps.png
       Row-only error | Row+Col error | 2D SKI-QEP error
       Diverging colormap centered at zero, common symmetric range.

Run from the repo root:
    MPLBACKEND=Agg python experiments/simulated/plot_linear_diffusion_common_scale.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── paths ─────────────────────────────────────────────────────────────────────

CACHE_DIR = os.environ.get("COMMON_SCALE_DIR", "results/common_scale_linear_diffusion")
OUT_PNG   = os.path.join(CACHE_DIR, "linear_diffusion_qep_common_scale.png")
OUT_PDF   = os.path.join(CACHE_DIR, "linear_diffusion_qep_common_scale.pdf")
OUT_ERR   = os.path.join(CACHE_DIR, "linear_diffusion_qep_error_maps.png")

ROW_ONLY_NPZ = os.path.join(CACHE_DIR, "row_only_qep.npz")
ROW_COL_NPZ  = os.path.join(CACHE_DIR, "row_col_qep.npz")
SKI_2D_NPZ   = os.path.join(CACHE_DIR, "ski_qep_2d.npz")


# ── load ──────────────────────────────────────────────────────────────────────

missing = [p for p in (ROW_ONLY_NPZ, ROW_COL_NPZ, SKI_2D_NPZ) if not os.path.exists(p)]
if missing:
    print("ERROR: the following .npz files are missing:")
    for p in missing:
        print(f"  {p}")
    print(
        "\nPlease run the source scripts first:\n"
        "  PYTHONPATH=. python experiments/simulated/linear_diffusion_qep_ablation_test.py\n"
        "  PYTHONPATH=. python experiments/simulated/linear_diffusion_qep_2d.py"
    )
    sys.exit(1)

d_row  = np.load(ROW_ONLY_NPZ, allow_pickle=True)
d_rc   = np.load(ROW_COL_NPZ,  allow_pickle=True)
d_2d   = np.load(SKI_2D_NPZ,   allow_pickle=True)

truth       = d_row["truth"]
noisy       = d_row["noisy"]
pred_row    = d_row["pred"]
pred_rc     = d_rc["pred"]
pred_2d     = d_2d["pred"]

# Metadata for annotation
sigma0  = float(d_row["sigma0"])
seed    = int(d_row["seed"])
q_power = float(d_row["q_power"])

rmse_row = float(d_row["rmse"])
rmse_rc  = float(d_rc["rmse"])
rmse_2d  = float(d_2d["rmse"])

print(f"Loaded arrays — shape: {truth.shape}")
print(f"  truth:    min={truth.min():.4f}  max={truth.max():.4f}")
print(f"  noisy:    min={noisy.min():.4f}  max={noisy.max():.4f}")
print(f"  row-only: min={pred_row.min():.4f}  max={pred_row.max():.4f}  RMSE={rmse_row:.6f}")
print(f"  row+col:  min={pred_rc.min():.4f}  max={pred_rc.max():.4f}  RMSE={rmse_rc:.6f}")
print(f"  2D SKI:   min={pred_2d.min():.4f}  max={pred_2d.max():.4f}  RMSE={rmse_2d:.6f}")
print(f"  sigma0={sigma0}  seed={seed}  q_power={q_power}")

# Sanity: truth is in [0,1] (Dirichlet diffusion solution)
if truth.min() >= -0.01 and truth.max() <= 1.01:
    VMIN, VMAX = 0.0, 1.0
    print("  Using fixed vmin=0.0  vmax=1.0")
else:
    all_preds = [truth, pred_row, pred_rc, pred_2d]
    VMIN = float(min(a.min() for a in all_preds))
    VMAX = float(max(a.max() for a in all_preds))
    print(f"  Truth not in [0,1]; using common range vmin={VMIN:.4f}  vmax={VMAX:.4f}")


# ── Figure 1: common-scale 5-panel ────────────────────────────────────────────

images = [truth, noisy, pred_row, pred_rc, pred_2d]
titles = [
    "Truth",
    f"Noisy obs\n(σ₀={sigma0})",
    f"Row-only QEP\nRMSE={rmse_row:.4f}",
    f"Row+Col QEP\nRMSE={rmse_rc:.4f}",
    f"2D SKI-QEP\nRMSE={rmse_2d:.4f}",
]

fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), constrained_layout=True)
for ax, img, title in zip(axes, images, titles):
    im = ax.imshow(img, cmap="viridis", vmin=VMIN, vmax=VMAX,
                   origin="lower", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="signal value")
fig.suptitle(
    f"Linear diffusion — QEP variants (common scale, σ₀={sigma0}, seed={seed}, q={q_power})",
    fontweight="bold", fontsize=10
)

os.makedirs(CACHE_DIR, exist_ok=True)
fig.savefig(OUT_PNG, dpi=300)
fig.savefig(OUT_PDF, dpi=300)
plt.close(fig)
print(f"\n  Saved: {OUT_PNG}")
print(f"  Saved: {OUT_PDF}")


# ── Figure 2: error maps ───────────────────────────────────────────────────────

err_row = pred_row - truth
err_rc  = pred_rc  - truth
err_2d  = pred_2d  - truth

errors     = [err_row, err_rc, err_2d]
err_titles = [
    f"Row-only error\nRMSE={rmse_row:.4f}",
    f"Row+Col error\nRMSE={rmse_rc:.4f}",
    f"2D SKI-QEP error\nRMSE={rmse_2d:.4f}",
]
errmax = float(max(abs(e).max() for e in errors))

fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
for ax, err, title in zip(axes2, errors, err_titles):
    im2 = ax.imshow(err, cmap="coolwarm", vmin=-errmax, vmax=errmax,
                    origin="lower", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig2.colorbar(im2, ax=axes2, fraction=0.025, pad=0.02, label="prediction − truth")
fig2.suptitle(
    f"Linear diffusion — QEP error maps (σ₀={sigma0}, seed={seed}, q={q_power})",
    fontweight="bold", fontsize=10
)

fig2.savefig(OUT_ERR, dpi=300)
plt.close(fig2)
print(f"  Saved: {OUT_ERR}")

print("\nDone.")
