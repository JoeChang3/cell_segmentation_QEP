"""Round 10 figures (Part 13): one consistent 8-panel per selected patch.

Panels: raw / GT instances / frozen current segmentation / Deep q=2 mean /
Deep q=1.5 mean / Delta_mu / |grad mu_q2| / |grad mu_q1.5|.

The two predictive-mean panels share ONE intensity scale and the two
mean-gradient panels share ONE scale, so a global shift cannot be mistaken for
structure. Delta_mu uses a symmetric scale. EVERY saved patch is plotted -- no
selection, no cherry-picking.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=73)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.out, "predictions",
                                          f"*_s{a.seed}.npz")))
    print(f"plotting {len(files)} patches (seed {a.seed}) - all of them, no selection")
    for f in files:
        z = np.load(f)
        raw, gt, pred = z["raw"], z["gt"], z["pred"]
        m2, m15, d = z["mu_q2"], z["mu_q15"], z["delta"]
        gy, gx = np.gradient(m2); g2 = np.sqrt(gx**2 + gy**2)
        gy, gx = np.gradient(m15); g15 = np.sqrt(gx**2 + gy**2)
        lo, hi = float(min(m2.min(), m15.min())), float(max(m2.max(), m15.max()))
        ghi = float(max(g2.max(), g15.max()))
        v = max(1e-12, float(np.abs(d).max()))
        base = os.path.basename(f).replace(".npz", "")
        merged = bool(z["merged"])
        panels = [("raw image", raw, "gray", None),
                  ("GT instances", gt, "nipy_spectral", None),
                  ("frozen current seg", pred, "nipy_spectral", None),
                  ("Deep q=2 mean", m2, "viridis", (lo, hi)),
                  ("Deep q=1.5 mean", m15, "viridis", (lo, hi)),
                  ("Delta_mu = q1.5 - q2", d, "RdBu_r", (-v, v)),
                  ("|grad mu| q=2", g2, "magma", (0, ghi)),
                  ("|grad mu| q=1.5", g15, "magma", (0, ghi))]
        fig, ax = plt.subplots(2, 4, figsize=(18, 8.6))
        for k, (t, arr, cm, lim) in enumerate(panels):
            axx = ax[k // 4, k % 4]
            kw = dict(cmap=cm)
            if lim: kw.update(vmin=lim[0], vmax=lim[1])
            imh = axx.imshow(arr, **kw)
            if k >= 3: fig.colorbar(imh, ax=axx, fraction=0.046)
            axx.contour(find_boundaries(gt.astype(np.int32), mode="outer"),
                        levels=[0.5], colors="lime", linewidths=0.7)
            axx.set_title(t, fontsize=9); axx.axis("off")
        fig.suptitle(f"{base}   merged={merged}   "
                     f"max|Delta_mu|={np.abs(d).max():.4f}   "
                     f"(mean panels share one scale; |grad| panels share one scale)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "figures", f"patch_{base}.png"), dpi=110)
        plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
