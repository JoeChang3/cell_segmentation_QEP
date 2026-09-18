"""
Task A4: RECONSTRUCTION-ONLY smoke test of the new `paper_fast_gp` arm.

Two development images only (one nuclei, one whole-cell). No thresholding, no
watershed, no AP -- this compares reconstructions and nothing else, as the brief
requires.

Arms:
  raw                        the image itself
  gp_isotropic_gpytorch_2025 the existing `separable_gp_smooth_gpytorch`
                             (isotropic Matern 2.5, float32, Adam, 6000-point
                             random subsample) -- the arm formerly labelled
                             "gp_legacy". Run tile-by-tile on the SAME tile grid
                             so the comparison is tiling-matched.
  paper_fast_gp              the new faithful port (single L-BFGS-B run from the
                             paper's param_ini; paper-faithful default)
  paper_fast_gp_ms8          same, with n_restarts=8 -- NOT the paper's
                             estimator, included to show the cost of the
                             non-convexity

Saves reconstructions and difference images as .npz + .png, a parameter table,
runtimes and summary statistics.

Usage:
  python audits/taskA4_paper_fast_gp_smoke.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import imageio.v2 as imageio
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py_core.Modified_Functions_RGasp import separable_gp_smooth_gpytorch
from py_core.paper_fast_gp import (
    estimate_shared_params,
    reconstruct_image,
    reconstruct_tile,
    tile_grid,
)

OUT = os.path.join(_ROOT, "results", "audit_paper_fast_gp_smoke_20260917")
DEV = [("nuclei", "data/nuclear_test_images/nuclei_figure_1/original_fig.png"),
       ("whole_cell", "data/whole_cell_test_images/whole_cell_figure_1/original_fig.jpg")]


def load_gray(p: str) -> np.ndarray:
    """Exactly our pipeline's loader (py_core/segmentation_no_gp.py:139)."""
    img = imageio.imread(p)
    return (img[..., 0] if img.ndim == 3 else img).astype(np.float64)


def iso_gp_tiled(image: np.ndarray, seed: int = 0) -> np.ndarray:
    """Current isotropic GPyTorch smoother, applied on the SAME tile grid.

    The legacy smoother subsamples with an unseeded np.random.choice
    (Modified_Functions_RGasp.py:234). We seed the global RNG here so this smoke
    test is at least reproducible; the arm itself is unchanged.
    """
    g = tile_grid(*image.shape)
    out = image.copy()
    np.random.seed(seed)
    for t in g["tiles"]:
        sl = (slice(t["y_offset"], t["y_offset"] + t["h"]),
              slice(t["x_offset"], t["x_offset"] + t["w"]))
        out[sl] = separable_gp_smooth_gpytorch(image[sl])
    return out


def stats(name: str, rec: np.ndarray, raw: np.ndarray) -> dict:
    d = rec - raw
    gy, gx = np.gradient(rec)
    return dict(arm=name, rec_min=float(rec.min()), rec_max=float(rec.max()),
                rec_mean=float(rec.mean()), rec_std=float(rec.std()),
                rmse_to_raw=float(np.sqrt((d ** 2).mean())),
                maxabs_to_raw=float(np.abs(d).max()),
                corr_to_raw=float(np.corrcoef(rec.ravel(), raw.ravel())[0, 1]),
                mean_grad_mag=float(np.sqrt(gx ** 2 + gy ** 2).mean()))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    srows, prows, drows = [], [], []

    for ds, rel in DEV:
        path = os.path.join(_ROOT, rel)
        raw = load_gray(path)
        H, W = raw.shape
        g = tile_grid(H, W)
        print("=" * 104)
        print(f"{ds}  {rel}  {H}x{W}  range [{raw.min():.1f}, {raw.max():.1f}]")
        print(f"  tiling: {g['num_pieces_y']}x{g['num_pieces_x']} tiles of "
              f"{g['crop_height']}x{g['crop_width']}  "
              f"covered {g['covered_rows']}x{g['covered_cols']}")
        print("=" * 104)

        recs = {"raw": raw}

        t0 = time.time()
        recs["gp_isotropic_gpytorch_2025"] = iso_gp_tiled(raw)
        t_iso = time.time() - t0
        print(f"  gp_isotropic_gpytorch_2025 : {t_iso:7.1f} s")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t0 = time.time()
            r1 = reconstruct_image(raw, verbose=True)
            t_pf = time.time() - t0
            degen_warn = [str(x.message) for x in w]
        recs["paper_fast_gp"] = r1["reconstruction"]
        sp = r1["shared_params"]
        er = sp.effective_ranges_px()
        print(f"  paper_fast_gp              : {t_pf:7.1f} s   "
              f"beta1={sp.beta1:.5f} beta2={sp.beta2:.5f} nu={sp.nugget:.6f}")
        print(f"      f={sp.neg_log_lik:.4f}  eff.range rows={er[0]:.4g}px "
              f"cols={er[1]:.4g}px  degenerate={sp.degenerate_axes}")
        if degen_warn:
            print(f"      WARNING RAISED: {degen_warn[0][:150]}")

        t0 = time.time()
        r2 = reconstruct_image(raw, n_restarts=8)
        t_ms = time.time() - t0
        recs["paper_fast_gp_ms8"] = r2["reconstruction"]
        sp2 = r2["shared_params"]
        er2 = sp2.effective_ranges_px()
        print(f"  paper_fast_gp_ms8          : {t_ms:7.1f} s   "
              f"beta1={sp2.beta1:.5f} beta2={sp2.beta2:.5f} nu={sp2.nugget:.6f}")
        print(f"      f={sp2.neg_log_lik:.4f}  eff.range rows={er2[0]:.4g}px "
              f"cols={er2[1]:.4g}px  degenerate={sp2.degenerate_axes}")

        for tag, r, rt in [("paper_fast_gp", r1, t_pf), ("paper_fast_gp_ms8", r2, t_ms)]:
            s = r["shared_params"]
            prows.append(dict(dataset=ds, arm=tag, beta1=s.beta1, beta2=s.beta2,
                              nugget=s.nugget, neg_log_lik=s.neg_log_lik,
                              eff_range_rows_px=s.effective_ranges_px()[0],
                              eff_range_cols_px=s.effective_ranges_px()[1],
                              degenerate_axes=",".join(s.degenerate_axes) or "none",
                              source_tile=s.source_tile, n_obj_evals=s.n_obj_evals,
                              optim_method=s.optim_method, runtime_sec=rt,
                              n_uncovered_px=r["n_uncovered_px"],
                              input_scale=r["input_scale"]))
            for pt in r["per_tile"]:
                drows.append(dict(dataset=ds, arm=tag, **pt))

        print(f"\n  {'arm':<28}{'RMSE-raw':>10}{'max-raw':>10}{'corr-raw':>10}"
              f"{'mean|grad|':>12}{'rec std':>10}")
        for k, v in recs.items():
            st = stats(k, v, raw)
            st.update(dataset=ds, runtime_sec={"raw": 0.0,
                      "gp_isotropic_gpytorch_2025": t_iso,
                      "paper_fast_gp": t_pf, "paper_fast_gp_ms8": t_ms}[k])
            srows.append(st)
            print(f"  {k:<28}{st['rmse_to_raw']:>10.4f}{st['maxabs_to_raw']:>10.4f}"
                  f"{st['corr_to_raw']:>10.6f}{st['mean_grad_mag']:>12.4f}"
                  f"{st['rec_std']:>10.4f}")

        # reconstructions ARE compared directly to each other, not only via
        # each one's RMSE to raw (round 2 showed that is uninformative)
        print(f"\n  pairwise RMSE between reconstructions:")
        keys = list(recs)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ka, kb = keys[a], keys[b]
                dd = recs[ka] - recs[kb]
                print(f"    {ka:<28} vs {kb:<28} RMSE={np.sqrt((dd**2).mean()):>9.4f}"
                      f"  max|diff|={np.abs(dd).max():>9.4f}")
                srows.append(dict(dataset=ds, arm=f"PAIR:{ka}|{kb}",
                                  rmse_to_raw=float(np.sqrt((dd ** 2).mean())),
                                  maxabs_to_raw=float(np.abs(dd).max())))

        np.savez_compressed(os.path.join(OUT, f"{ds}_reconstructions.npz"), **recs)

        # figures: reconstruction row + difference row
        n = len(recs)
        fig, ax = plt.subplots(2, n, figsize=(4.0 * n, 8.0))
        vmin, vmax = float(raw.min()), float(raw.max())
        for c, (k, v) in enumerate(recs.items()):
            ax[0, c].imshow(v, cmap="gray", vmin=vmin, vmax=vmax)
            ax[0, c].set_title(f"{k}", fontsize=9)
            ax[0, c].axis("off")
            d = v - raw
            m = max(1e-12, float(np.abs(d).max()))
            im = ax[1, c].imshow(d, cmap="RdBu_r", vmin=-m, vmax=m)
            ax[1, c].set_title(f"{k} - raw  (max|d|={m:.3g})", fontsize=8)
            ax[1, c].axis("off")
            fig.colorbar(im, ax=ax[1, c], fraction=0.046)
        fig.suptitle(f"{ds}: reconstruction only (no segmentation)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", f"{ds}_reconstruction_compare.png"),
                    dpi=130)
        plt.close(fig)
        print()

    pd.DataFrame(srows).to_csv(os.path.join(OUT, "reconstruction_summary.csv"),
                               index=False)
    pd.DataFrame(prows).to_csv(os.path.join(OUT, "shared_parameter_table.csv"),
                               index=False)
    pd.DataFrame(drows).to_csv(os.path.join(OUT, "per_tile_diagnostics.csv"),
                               index=False)
    print(f"wrote results into {OUT}")


if __name__ == "__main__":
    main()
