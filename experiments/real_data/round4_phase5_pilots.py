"""
Phase 5 pilots. Success criteria are written to pilot_configs/ BEFORE any pilot
score is computed.

TARGET FAILURE (from Phase 1/2): adjacent-cell MERGING at the instance
separation stage. Phase 1: merged accounts for 23-27% of all whole-cell GT
cells and 13-23% of nuclei cells, while missing_foreground is only 1.7-3.5%.
Phase 2: oracle MARKERS on the real foreground lift held-out AP@0.5 from 0.2248
to 0.4350 (whole-cell) and 0.7543 to 0.9001 (nuclei, merged 32.2 -> 8.0), which
locates the loss in seeding/elevation rather than in foreground extraction.

PILOT 1  intensity-aware watershed elevation (NON-QEP mechanism).
  The current elevation is -distance_transform_edt(binary). For two fused cells
  the distance transform has one broad maximum, so peak_local_max emits one
  seed and the pair merges. Between touching cells the IMAGE usually dips, so
  an intensity term should place a watershed ridge there.
      elevation = -( w * dist_norm + (1 - w) * img_norm )
  w = 1 reproduces the current rule exactly. Markers are still taken from the
  distance transform, so this changes the flooding surface only. The intensity
  term is taken from EACH ARM'S OWN reconstruction, so the pilot simultaneously
  asks whether a q<2 reconstruction supplies a better boundary surface than a
  Gaussian one. w is selected on DEVELOPMENT images only and then frozen.

PILOT 2  hyperparameter replay (separates q from parameter learning).
  q can only act through hyperparameter selection in this implementation
  (Phase 3, OBSERVED). So: take the hyperparameters that the q=1.5 arm fitted
  and run the GAUSSIAN model with them. If the resulting masks and scores match
  the q=1.5 arm, the effect is parameter learning, not q.

PRE-REGISTERED SUCCESS CRITERIA
  Pilot 1 counts as supporting the elevation mechanism if, on held-out images:
    (a) merged-cell count falls by >= 15% relative for at least one dataset, AND
    (b) AP@0.5 does not fall for that dataset, AND
    (c) split count does not rise by more than 50% relative.
  Pilot 1 counts as supporting a QEP-specific benefit only if a q=1.5 arm beats
  BOTH its Gaussian counterpart (q=2/GP, identical machinery) AND the Raw arm
  on AP@0.5 for the same dataset.
  Pilot 2 counts as showing "q matters beyond parameter learning" only if the
  Gaussian model with q=1.5's replayed hyperparameters differs materially from
  the q=1.5 arm (AP@0.5 gap > 0.01).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from py_core.Modified_Functions_RGasp import eliminate_small_areas
from py_core.segmentation_eval import (
    evaluate_instances,
    load_gray_image,
    load_instance_mask,
)

R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
R2 = os.path.join(_ROOT, "results", "real_cellseg_round2_20260915")
# reconstructions: development ones were regenerated into R3/masks, held-out
# ones remain the cached round-2 arrays in R2/masks
def recon_dir(role):
    return R3 if role == "development" else R2
ARMS = ["raw", "gp", "qep_q2", "qep_q1.5"]
MD = {"nuclei": 15, "whole_cell": 9}
RM = 50
W_GRID = [1.0, 0.8, 0.6, 0.4, 0.2]      # 1.0 == current frozen rule


def norm01(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Min-max normalize inside the mask; zeros elsewhere."""
    out = np.zeros_like(a, dtype=np.float64)
    if not m.any():
        return out
    v = a[m]
    lo, hi = float(v.min()), float(v.max())
    out[m] = (v - lo) / (hi - lo) if hi > lo else 0.0
    return out


def separate_intensity_aware(binary: np.ndarray, img: np.ndarray, *,
                             min_distance: int, w: float,
                             remove_size: int = RM):
    """Markers from the distance transform (unchanged); elevation blends the
    distance transform with the image so inter-cell dips become ridges."""
    fg = binary > 0
    dist = ndi.distance_transform_edt(fg)
    coords = peak_local_max(dist, min_distance=int(min_distance),
                            exclude_border=False, labels=fg, p_norm=np.inf)
    if len(coords):
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    mk = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        mk[r, c] = i
    # rescue components with no marker, same policy as the frozen rule
    comps, ncomp = ndi.label(fg, structure=ndi.generate_binary_structure(2, 1))
    if ncomp:
        have = np.zeros(ncomp + 1, dtype=bool)
        if len(coords):
            have[np.unique(comps[mk > 0])] = True
        have[0] = True
        nxt = int(mk.max())
        for cid in np.flatnonzero(~have):
            sl = ndi.find_objects(comps)[cid - 1]
            if sl is None:
                continue
            sub = np.where(comps[sl] == cid, dist[sl], -1.0)
            rr, cc = np.unravel_index(int(np.argmax(sub)), sub.shape)
            nxt += 1
            mk[sl[0].start + rr, sl[1].start + cc] = nxt

    elev = -(w * norm01(dist, fg) + (1.0 - w) * norm01(img, fg))
    lab = watershed(elev, markers=mk, mask=fg, connectivity=1).astype(np.int32)
    return eliminate_small_areas(lab, remove_size), int(mk.max())


def counts(gt: np.ndarray, inst: np.ndarray, share: float = 0.25):
    """merged / split counts with explicit overlap criteria."""
    tl = np.unique(gt[gt > 0]); pl = np.unique(inst[inst > 0])
    if tl.size == 0 or pl.size == 0:
        return 0, 0
    ti = np.zeros(int(gt.max()) + 1, dtype=np.int64); ti[tl] = np.arange(1, tl.size + 1)
    pi = np.zeros(int(inst.max()) + 1, dtype=np.int64); pi[pl] = np.arange(1, pl.size + 1)
    nt, npd = tl.size, pl.size
    hist = np.bincount(ti[gt.ravel()] * (npd + 1) + pi[inst.ravel()],
                       minlength=(nt + 1) * (npd + 1)).reshape(nt + 1, npd + 1)
    inter = hist[1:, 1:].astype(np.float64)
    t_area = hist[1:, :].sum(axis=1).astype(np.float64)
    cov = inter / np.maximum(t_area, 1)[:, None]
    merged = sum(1 for i in range(nt)
                 if any((cov[:, j] >= share).sum() >= 2
                        for j in np.flatnonzero(cov[i] >= share)))
    split = int(((cov >= share).sum(axis=1) >= 2).sum())
    return int(merged), split


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[man["status"] == "ok"]
    rows: List[Dict] = []

    # ═══ PILOT 1: select w on DEVELOPMENT only ═══
    print("=" * 120)
    print("PILOT 1  intensity-aware watershed elevation  (NON-QEP mechanism)")
    print("  elevation = -( w*dist_norm + (1-w)*img_norm );  w=1.0 is the frozen rule")
    print("  w selected on DEVELOPMENT images only, aggregated across all 4 arms")
    print("=" * 120)
    dev = man[man.role == "development"]
    for _, im in dev.iterrows():
        ds, name = im["dataset"], im["image"]
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        for arm in ARMS:
            bp = os.path.join(R3, "binary_masks", f"{ds}_{name}_{arm}.npz")
            rp = os.path.join(recon_dir("development"), "masks", f"{ds}_{name}_{arm}.npz")
            if not (os.path.exists(bp) and os.path.exists(rp)):
                continue
            binary = np.load(bp)["binary"] > 0
            recon = np.load(rp)["predmean"].astype(np.float64)
            for w in W_GRID:
                inst, nmk = separate_intensity_aware(binary, recon,
                                                     min_distance=MD[ds], w=w)
                ev = evaluate_instances(gt, inst)
                mg, sp = counts(gt, inst)
                rows.append(dict(pilot="P1_elevation", role="development",
                                 dataset=ds, image=name, arm=arm, w=w,
                                 ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                                 n_pred=ev.n_pred, n_gt=ev.n_true,
                                 merged=mg, split=sp, n_markers=nmk,
                                 mean_matched_iou=ev.mean_matched_iou))
    d = pd.DataFrame([r for r in rows if r["role"] == "development"])
    print(f"\n  {'dataset':<12}{'w':>5}{'AP@0.5':>9}{'merged':>8}{'split':>7}"
          f"{'#pred':>7}  (mean over 4 arms)")
    sel: Dict[str, float] = {}
    for ds in ["whole_cell", "nuclei"]:
        s = d[d.dataset == ds]
        if s.empty:
            continue
        g = s.groupby("w").agg(ap=("ap50", "mean"), mg=("merged", "mean"),
                               sp=("split", "mean"), np_=("n_pred", "mean")).reset_index()
        base = g[g.w == 1.0].iloc[0]
        for _, r in g.iterrows():
            print(f"  {ds:<12}{r.w:>5.1f}{r.ap:>9.4f}{r.mg:>8.1f}{r.sp:>7.1f}"
                  f"{r.np_:>7.0f}" + ("   <- frozen rule" if r.w == 1.0 else ""))
        # selection: lowest merged count subject to AP not falling below the
        # frozen rule; tie-break on AP
        elig = g[g.ap >= base.ap - 1e-9]
        pick = elig.sort_values(["mg", "ap"], ascending=[True, False]).iloc[0]
        sel[ds] = float(pick.w)
        print(f"  -> {ds}: selected w={pick.w:.1f} "
              f"(merged {base.mg:.1f} -> {pick.mg:.1f}, AP {base.ap:.4f} -> {pick.ap:.4f})\n")

    with open(os.path.join(args.out, "pilot_configs", "pilot1_selected_w.json"),
              "w") as fh:
        json.dump(dict(selected_w=sel, grid=W_GRID,
                       selection_rule=("lowest mean merged count on development "
                                       "images subject to AP@0.5 not falling "
                                       "below the w=1.0 frozen rule; aggregated "
                                       "over all 4 arms, never per arm"),
                       elevation="-(w*dist_norm + (1-w)*img_norm)",
                       markers="unchanged: peak_local_max on the distance transform",
                       intensity_source="each arm's own cached reconstruction"), fh,
                  indent=2)

    # ═══ PILOT 1 held-out ═══
    print("=" * 120)
    print("PILOT 1  HELD-OUT with the frozen w")
    print("=" * 120)
    ho = man[man.role == "heldout_candidate"]
    for _, im in ho.iterrows():
        ds, name = im["dataset"], im["image"]
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        for arm in ARMS:
            bp = os.path.join(R3, "binary_masks", f"{ds}_{name}_{arm}.npz")
            rp = os.path.join(recon_dir("heldout_candidate"), "masks",
                              f"{ds}_{name}_{arm}.npz")
            if not (os.path.exists(bp) and os.path.exists(rp)):
                print(f"  [missing recon] {ds}/{name}/{arm}")
                continue
            binary = np.load(bp)["binary"] > 0
            recon = np.load(rp)["predmean"].astype(np.float64)
            for w in sorted({1.0, sel.get(ds, 1.0)}):
                inst, nmk = separate_intensity_aware(binary, recon,
                                                     min_distance=MD[ds], w=w)
                ev = evaluate_instances(gt, inst)
                mg, sp = counts(gt, inst)
                rows.append(dict(pilot="P1_elevation", role="heldout_candidate",
                                 dataset=ds, image=name, arm=arm, w=w,
                                 ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                                 n_pred=ev.n_pred, n_gt=ev.n_true,
                                 merged=mg, split=sp, n_markers=nmk,
                                 mean_matched_iou=ev.mean_matched_iou))
    h = pd.DataFrame([r for r in rows if r["role"] == "heldout_candidate"])
    if h.empty:
        print("  !! no held-out rows produced; check reconstruction paths")
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "pilot_results.csv"),
                                  index=False)
        return
    print(f"  {'dataset':<12}{'arm':<10}{'w':>5}{'AP@0.5':>9}{'AP@0.75':>9}"
          f"{'merged':>8}{'split':>7}{'#pred':>7}")
    for ds in ["whole_cell", "nuclei"]:
        for arm in ARMS:
            for w in sorted({1.0, sel.get(ds, 1.0)}):
                s = h[(h.dataset == ds) & (h.arm == arm) & (h.w == w)]
                if s.empty:
                    continue
                tag = "  (frozen)" if w == 1.0 else "  (pilot)"
                print(f"  {ds:<12}{arm:<10}{w:>5.1f}{s.ap50.mean():>9.4f}"
                      f"{s.ap75.mean():>9.4f}{s.merged.mean():>8.1f}"
                      f"{s.split.mean():>7.1f}{s.n_pred.mean():>7.0f}{tag}")
        print()

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "pilot_results.csv"),
                              index=False)
    print(f"Wrote pilot_results.csv and pilot_configs/pilot1_selected_w.json")


if __name__ == "__main__":
    main()
