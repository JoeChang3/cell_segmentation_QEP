"""
Round 9 companion: (a) quantify WHY the q effect is small, (b) Step-9 figures.

(a) The gradient posterior turns out to be MEAN-DOMINATED: ||E[grad f]|| is large
    relative to the posterior gradient standard deviation, so ||grad f|| is almost
    deterministic and the q-dependent dispersion barely contributes. This script
    measures that ratio directly.

(b) Crops are selected by an EXPLICIT rule, not by eye:
      - largest merge-interface length
      - median merge-interface length
      - weakest local contrast among merged pairs
      - a representative correctly separated pair (median interface length)

Usage: python experiments/real_data/round9_figures.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qpytorch.kernels as QK
from skimage.segmentation import find_boundaries

from py_core.segmentation_eval import load_instance_mask

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "r9", os.path.join(_HERE, "round9_qep_boundary_signal.py"))
r9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r9)

R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
R5 = os.path.join(_ROOT, "results", "real_cellseg_round5_corrected_baseline_20260917")
CROP = 41


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = args.out
    cfg = json.load(open(os.path.join(out, "model_config.json")))
    inv = pd.read_csv(os.path.join(out, "adjacent_pair_inventory.csv"))
    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[(man["status"] == "ok") & (man["role"] == "development")]

    dom_rows: List[Dict] = []
    print("=" * 116)
    print("WHY IS THE q EFFECT SMALL?  mean-dominance of the gradient posterior")
    print("  ratio = posterior gradient SD / ||E[grad f]||.  A small ratio means")
    print("  ||grad f|| is nearly deterministic, so q-dependent dispersion cannot")
    print("  move it much.")
    print("=" * 116)
    print(f"  {'dataset':<12}{'region':<24}{'||E[grad]||':>13}{'grad post SD':>14}"
          f"{'SD/||E[grad]||':>16}")

    for _, im in man.iterrows():
        ds, name = im["dataset"], im["image"]
        raw = imageio.imread(os.path.join(_ROOT, im["path_image"]))
        raw = (raw[..., 0] if raw.ndim == 3 else raw).astype(np.float64)
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        pred = np.load(os.path.join(R5, "instance_masks",
                                    f"{ds}_{name}_raw.npz"))["instance_mask"]
        hp = [h for h in cfg["hyperparameters"] if h["dataset"] == ds][0]
        kern = QK.Matern52KernelGrad(ard_num_dims=2)
        with torch.no_grad():
            kern.lengthscale = torch.tensor([hp["lengthscale_px"]])
        rg, _ = r9.build_regions(gt, pred)
        H, W = raw.shape

        for cls in ("A_merged_interface", "D_interior"):
            sites = r9.pick_sites(rg["regions"][cls], 20)
            s1s, sds = [], []
            for (r, c, _sz) in sites:
                r0 = int(np.clip(r - r9.PATCH // 2, 0, H - r9.PATCH))
                c0 = int(np.clip(c - r9.PATCH // 2, 0, W - r9.PATCH))
                sub = raw[r0:r0 + r9.PATCH, c0:c0 + r9.PATCH]
                rr, cc = np.meshgrid(np.arange(r9.PATCH), np.arange(r9.PATCH),
                                     indexing="ij")
                Xtr = torch.tensor(np.stack([rr.ravel(), cc.ravel()], 1),
                                   dtype=torch.float64)
                Ytr = torch.tensor(sub.ravel(), dtype=torch.float64)
                o = (r9.PATCH - r9.QCORE) // 2
                qr, qc = np.meshgrid(np.arange(o, o + r9.QCORE),
                                     np.arange(o, o + r9.QCORE), indexing="ij")
                Xq = torch.tensor(np.stack([qr.ravel(), qc.ravel()], 1),
                                  dtype=torch.float64)
                m3, c3 = r9.grad_posterior(Xtr, Ytr, Xq, kern, hp["outputscale"],
                                           hp["noise"], hp["mean_const"])
                s1s.append(m3[:, 1:].pow(2).sum(-1).sqrt().numpy())
                sds.append(np.sqrt((c3[:, 1, 1] + c3[:, 2, 2]).numpy()))
            s1 = np.concatenate(s1s); sd = np.concatenate(sds)
            ratio = float(np.median(sd / np.maximum(s1, 1e-12)))
            dom_rows.append(dict(dataset=ds, region_class=cls,
                                 mean_norm_of_mean_grad=float(s1.mean()),
                                 mean_grad_posterior_sd=float(sd.mean()),
                                 median_sd_over_mean=ratio,
                                 n_query=int(len(s1))))
            print(f"  {ds:<12}{cls:<24}{s1.mean():>13.4f}{sd.mean():>14.4f}"
                  f"{ratio:>16.4f}")

        # ---- Step 9 figures, explicit selection rule ------------------
        sub_inv = inv[inv.dataset == ds]
        mg = sub_inv[sub_inv.merged].sort_values("interface_len_px")
        sp = sub_inv[~sub_inv.merged].sort_values("interface_len_px")
        picks = []
        if len(mg):
            picks.append(("largest-merge-interface", mg.iloc[-1]))
            picks.append(("median-merge-interface", mg.iloc[len(mg) // 2]))
            picks.append(("weakest-contrast-merged",
                          mg.sort_values("local_contrast").iloc[0]))
        if len(sp):
            picks.append(("representative-separated", sp.iloc[len(sp) // 2]))

        gy, gx = np.gradient(raw)
        grad_raw = np.sqrt(gx ** 2 + gy ** 2)

        for pi, (why, p) in enumerate(picks):
            r, c = int(round(p.centroid_r)), int(round(p.centroid_c))
            r0 = int(np.clip(r - CROP // 2, 0, H - CROP))
            c0 = int(np.clip(c - CROP // 2, 0, W - CROP))
            sl = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
            sub = raw[sl]
            rr, cc = np.meshgrid(np.arange(CROP), np.arange(CROP), indexing="ij")
            Xtr = torch.tensor(np.stack([rr.ravel(), cc.ravel()], 1),
                               dtype=torch.float64)
            Ytr = torch.tensor(sub.ravel(), dtype=torch.float64)
            m3, c3 = r9.grad_posterior(Xtr, Ytr, Xtr, kern, hp["outputscale"],
                                       hp["noise"], hp["mean_const"])
            S1 = m3[:, 1:].pow(2).sum(-1).sqrt().numpy().reshape(CROP, CROP)
            gn2 = r9.sample_grad_norm(m3, c3, 2.0, 4000, r9.SEED)
            gn15 = r9.sample_grad_norm(m3, c3, 1.5, 4000, r9.SEED)
            cval = float(np.percentile(gn2.reshape(-1).numpy(), 90))
            maps = {
                "S2 q=2": gn2.mean(0).numpy().reshape(CROP, CROP),
                "S2 q=1.5": gn15.mean(0).numpy().reshape(CROP, CROP),
                "S3 q=2": (gn2 > cval).double().mean(0).numpy().reshape(CROP, CROP),
                "S3 q=1.5": (gn15 > cval).double().mean(0).numpy().reshape(CROP, CROP),
            }
            panels = [("raw image", sub, "gray"),
                      ("GT instances", gt[sl], "nipy_spectral"),
                      ("current predicted", pred[sl], "nipy_spectral"),
                      ("S1 ||E[grad f]||", S1, "magma"),
                      ("S2 q=2", maps["S2 q=2"], "magma"),
                      ("S2 q=1.5", maps["S2 q=1.5"], "magma"),
                      (f"S3 q=2 (c=p90={cval:.2f})", maps["S3 q=2"], "viridis"),
                      ("S3 q=1.5 (same c)", maps["S3 q=1.5"], "viridis"),
                      ("Delta_S2 = q1.5 - q2",
                       maps["S2 q=1.5"] - maps["S2 q=2"], "RdBu_r"),
                      ("baseline |grad raw|", grad_raw[sl], "magma")]
            fig, ax = plt.subplots(2, 5, figsize=(22, 9.2))
            lo = min(maps["S2 q=2"].min(), maps["S2 q=1.5"].min())
            hi = max(maps["S2 q=2"].max(), maps["S2 q=1.5"].max())
            s3hi = max(1e-9, max(maps["S3 q=2"].max(), maps["S3 q=1.5"].max()))
            for k, (t, arr, cm) in enumerate(panels):
                a = ax[k // 5, k % 5]
                if t.startswith("Delta"):
                    v = max(1e-12, float(np.abs(arr).max()))
                    imh = a.imshow(arr, cmap=cm, vmin=-v, vmax=v)
                elif t.startswith("S2"):
                    imh = a.imshow(arr, cmap=cm, vmin=lo, vmax=hi)
                elif t.startswith("S3"):
                    imh = a.imshow(arr, cmap=cm, vmin=0, vmax=s3hi)
                else:
                    imh = a.imshow(arr, cmap=cm)
                if k >= 3:
                    fig.colorbar(imh, ax=a, fraction=0.046)
                a.contour(find_boundaries(gt[sl].astype(np.int32), mode="outer"),
                          levels=[0.5], colors="lime", linewidths=0.6)
                a.set_title(t, fontsize=9)
                a.axis("off")
            fig.suptitle(
                f"{ds}  [{why}]  GT {int(p.gt_id_1)}+{int(p.gt_id_2)}  "
                f"merged={bool(p.merged)}  iface_len={int(p.interface_len_px)}  "
                f"local_contrast={p.local_contrast:.2f}   "
                f"(the two S2 panels share one colour scale)", fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "figures", f"{ds}_crop{pi}_{why}.png"),
                        dpi=115)
            plt.close(fig)
            print(f"    figure: {ds} [{why}] pair {int(p.gt_id_1)}+"
                  f"{int(p.gt_id_2)} iface={int(p.interface_len_px)}")

    pd.DataFrame(dom_rows).to_csv(
        os.path.join(out, "tables", "gradient_mean_dominance.csv"), index=False)
    print(f"\nwrote figures and tables/gradient_mean_dominance.csv into {out}")


if __name__ == "__main__":
    main()
