"""
Round-2 figures: marker before/after on development images, and held-out
failure overlays.

Reads only saved numerical arrays. Produces:
  figures/dev_markers_<dataset>.png        legacy vs peak markers, per method
  figures/heldout_overlay_<image>_<arm>.png predicted vs GT plus error map
  figures/heldout_ap_by_image.png          per-image AP@0.5, all methods
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from skimage.segmentation import find_boundaries

from py_core.segmentation_eval import load_gray_image, load_instance_mask

ROUND1 = os.path.join(_ROOT, "results", "real_cellseg_round1")
_rng = np.random.default_rng(0)
_cols = _rng.random((4096, 3)) * 0.75 + 0.25
_cols[0] = 0.0
LBL = ListedColormap(_cols)


def show_lbl(ax, m, title):
    ax.imshow(m % LBL.N, cmap=LBL, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def fig_dev_markers(out_dir: str, run_dir: str) -> None:
    """Legacy markers=None vs selected peak markers, development images."""
    md_sel = {"nuclei": 15, "whole_cell": 9}
    for ds in ("nuclei", "whole_cell"):
        arms = ["raw", "gp", "qep_q2", "qep_q1.5"]
        have = [(a, os.path.join(run_dir, "masks",
                                 f"{ds}_{a}_legacy_none.npz"),
                 os.path.join(run_dir, "masks",
                              f"{ds}_{a}_peak_md{md_sel[ds]}.npz"))
                for a in arms]
        have = [(a, l, p) for a, l, p in have
                if os.path.exists(l) and os.path.exists(p)]
        if not have:
            continue
        gt = np.load(os.path.join(ROUND1, "masks", f"{ds}_ground_truth.npy"))
        n = len(have)
        fig, axs = plt.subplots(3, n + 1, figsize=(3.0 * (n + 1), 9.4))
        for c, (a, lp, pp) in enumerate(have):
            dl, dp = np.load(lp), np.load(pp)
            axs[0, c].imshow(dl["binary"] > 0, cmap="gray")
            axs[0, c].set_title(f"{a}\nbinary (shared)", fontsize=8)
            show_lbl(axs[1, c], dl["instance_mask"],
                     f"legacy markers=None\nn={len(np.unique(dl['instance_mask']))-1}")
            show_lbl(axs[2, c], dp["instance_mask"],
                     f"peak md={md_sel[ds]}\nn={len(np.unique(dp['instance_mask']))-1}")
            for r in range(3):
                axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[0, -1].imshow(gt > 0, cmap="gray")
        axs[0, -1].set_title("GT foreground", fontsize=8)
        show_lbl(axs[1, -1], gt, f"GT instances n={len(np.unique(gt))-1}")
        axs[2, -1].axis("off")
        for r in range(2):
            axs[r, -1].set_xticks([]); axs[r, -1].set_yticks([])
        fig.suptitle(f"{ds} (DEVELOPMENT): only marker generation changes; "
                     f"foreground, elevation, connectivity and cleanup are fixed",
                     fontweight="bold", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = os.path.join(out_dir, f"dev_markers_{ds}.png")
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {os.path.relpath(p, _ROOT)}")


def fig_heldout_overlays(out_dir: str, run_dir: str, manifest: pd.DataFrame,
                         arms=("raw", "gp", "qep_q1.5")) -> None:
    for _, im in manifest.iterrows():
        ds, name = im["dataset"], im["image"]
        img = load_gray_image(os.path.join(_ROOT, im["path_image"]))
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        present = [a for a in arms
                   if os.path.exists(os.path.join(run_dir, "masks",
                                                  f"{ds}_{name}_{a}.npz"))]
        if not present:
            continue
        fig, axs = plt.subplots(2, len(present) + 1,
                                figsize=(3.4 * (len(present) + 1), 7.0))
        axs = np.atleast_2d(axs)
        vmin, vmax = float(img.min()), float(img.max())
        for c, a in enumerate(present):
            inst = np.load(os.path.join(run_dir, "masks",
                                        f"{ds}_{name}_{a}.npz"))["instance_mask"]
            axs[0, c].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ov = np.zeros(inst.shape + (4,))
            ov[find_boundaries(inst, mode="outer")] = (1, 0, 0, 1)
            ov[find_boundaries(gt, mode="outer")] = (0, 1, 0, 0.75)
            axs[0, c].imshow(ov)
            axs[0, c].set_title(f"{a}\npred red / GT green", fontsize=9)
            pf, tf = inst > 0, gt > 0
            err = np.zeros(inst.shape + (3,))
            err[pf & tf] = (0.55, 0.55, 0.55)
            err[pf & ~tf] = (1, 0, 1)
            err[~pf & tf] = (0, 1, 1)
            axs[1, c].imshow(err)
            axs[1, c].set_title("TP grey / FP magenta / FN cyan", fontsize=9)
        axs[0, -1].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axs[0, -1].set_title("image", fontsize=9)
        show_lbl(axs[1, -1], gt, f"GT n={len(np.unique(gt))-1}")
        for r in range(2):
            for c in range(len(present) + 1):
                axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        fig.suptitle(f"HELD-OUT {ds}/{name}", fontweight="bold", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = os.path.join(out_dir, f"heldout_overlay_{name}.png")
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {os.path.relpath(p, _ROOT)}")


def fig_heldout_ap(out_dir: str, run_dir: str) -> None:
    p = os.path.join(run_dir, "heldout_per_image_metrics.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d["ap50"].notna()]
    if d.empty:
        return
    labels = list(dict.fromkeys(d["label"]))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, ds in zip(axs, ("nuclei", "whole_cell")):
        s = d[d["dataset"] == ds]
        imgs = sorted(s["image"].unique())
        x = np.arange(len(imgs)); w = 0.8 / max(len(labels), 1)
        for i, lab in enumerate(labels):
            vals = [s[(s["image"] == im) & (s["label"] == lab)]["ap50"].mean()
                    for im in imgs]
            ax.bar(x + i * w, vals, w, label=lab)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([i.replace(f"{ds}_", "") for i in imgs],
                           rotation=20, fontsize=8)
        ax.set_ylabel("AP@0.5"); ax.set_title(f"{ds} (held-out)")
        ax.grid(alpha=0.3, axis="y")
    axs[0].legend(fontsize=8)
    fig.suptitle("Held-out AP@0.5 per image, frozen protocol "
                 "(no per-image method selection)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pth = os.path.join(out_dir, "heldout_ap_by_image.png")
    fig.savefig(pth, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {os.path.relpath(pth, _ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print("figures:")
    fig_dev_markers(fig_dir, args.out)
    mp = os.path.join(args.out, "image_manifest.csv")
    if os.path.exists(mp):
        man = pd.read_csv(mp)
        man = man[(man["status"] == "ok") & (man["role"] == "heldout_candidate")]
        fig_heldout_overlays(fig_dir, args.out, man)
    fig_heldout_ap(fig_dir, args.out)


if __name__ == "__main__":
    main()
