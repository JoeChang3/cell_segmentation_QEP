"""
Phase 1: instance-level failure inventory from cached round-3 outputs.

No model is retrained. Reads the round-3 binary masks, instance masks and
marker images together with the round-2/3 cached reconstructions.

CRITICAL DISTINCTION enforced here: pixel-level foreground recall is NOT
instance-level cell recall. A cell can contribute 90% of its pixels to the
foreground and still be unmatched at IoU 0.5 if its boundary is wrong or it is
fused with a neighbour.

Per ground-truth instance we record: best predicted IoU, the fraction of the
cell covered by predicted foreground, how many markers land inside it, how many
predicted instances substantially overlap it, area, whether it touches the image
border, local intensity contrast, and a failure category.

Failure categories (explicit criteria, evaluated in this order, using overlap
rules SEPARATE from the official one-to-one AP matching):
  matched            best IoU >= 0.5
  missing_foreground fg coverage < 0.30                       (never detected)
  boundary_under     fg coverage >= 0.30 and best IoU < 0.5 and exactly one
                     predicted instance covers >=25% of the cell and that
                     prediction is not shared with another cell
                     (detected but boundary badly wrong)
  merged             a predicted instance covers >=25% of this cell AND >=25%
                     of at least one other cell
  split              >=2 predicted instances each cover >=25% of this cell
  cleanup_lost       fg coverage >= 0.30 but no predicted instance covers >=10%
                     (survived thresholding, removed by watershed/cleanup)
  other_low_iou      remaining
Unmatched PREDICTED instances are tracked separately as spurious.
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

from py_core.segmentation_eval import (
    compute_ious_fast,
    load_gray_image,
    load_instance_mask,
)

R2 = os.path.join(_ROOT, "results", "real_cellseg_round2_20260915")
R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
ARMS = ["raw", "gp", "qep_q2", "qep_q1.5"]
COV_MAIN, COV_SHARE, COV_MIN, COV_MISS = 0.30, 0.25, 0.10, 0.30


def contingency(gt: np.ndarray, pred: np.ndarray):
    """inter[i,j], gt areas, pred areas for labels > 0."""
    tl = np.unique(gt[gt > 0]); pl = np.unique(pred[pred > 0])
    if tl.size == 0:
        return (np.zeros((0, pl.size)), np.zeros(0), np.zeros(pl.size), tl, pl)
    ti = np.zeros(int(gt.max()) + 1, dtype=np.int64); ti[tl] = np.arange(1, tl.size + 1)
    pmax = int(pred.max()) if pl.size else 0
    pi = np.zeros(pmax + 1, dtype=np.int64)
    if pl.size:
        pi[pl] = np.arange(1, pl.size + 1)
    a = ti[gt.ravel()]; b = pi[pred.ravel()] if pl.size else np.zeros(gt.size, dtype=np.int64)
    nt, npd = tl.size, pl.size
    hist = np.bincount(a * (npd + 1) + b,
                       minlength=(nt + 1) * (npd + 1)).reshape(nt + 1, npd + 1)
    inter = hist[1:, 1:].astype(np.float64)
    t_area = hist[1:, :].sum(axis=1).astype(np.float64)
    p_area = hist[:, 1:].sum(axis=0).astype(np.float64)
    return inter, t_area, p_area, tl, pl


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[man["status"] == "ok"]

    inst_rows: List[Dict] = []
    spur_rows: List[Dict] = []

    for _, im in man.iterrows():
        ds, name, role = im["dataset"], im["image"], im["role"]
        img = load_gray_image(os.path.join(_ROOT, im["path_image"]))
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        gt_fg = gt > 0
        # distance from each pixel to background, used for border contact
        H, W = gt.shape
        for arm in ARMS:
            bp = os.path.join(R3, "binary_masks", f"{ds}_{name}_{arm}.npz")
            ip = os.path.join(R3, "instance_masks", f"{ds}_{name}_{arm}.npz")
            if not (os.path.exists(bp) and os.path.exists(ip)):
                continue
            binary = np.load(bp)["binary"] > 0
            dz = np.load(ip)
            inst = dz["instance_mask"]
            markers = dz["marker_image"]

            inter, t_area, p_area, tl, pl = contingency(gt, inst)
            iou = np.zeros_like(inter)
            if inter.size:
                union = t_area[:, None] + p_area[None, :] - inter
                with np.errstate(divide="ignore", invalid="ignore"):
                    iou = np.where(union > 0, inter / union, 0.0)
            cov_of_gt = (inter / np.maximum(t_area, 1)[:, None]) if inter.size else inter

            # markers inside each GT cell
            mk_lab = gt[markers > 0] if markers.size else np.zeros(0, dtype=int)
            mk_count = pd.Series(mk_lab[mk_lab > 0]).value_counts().to_dict()

            # per-GT-instance records
            for i, lab in enumerate(tl):
                sel = gt == lab
                area = float(t_area[i])
                best_iou = float(iou[i].max()) if iou.size else 0.0
                fgcov = float((binary & sel).sum() / max(area, 1))
                covers = np.flatnonzero(cov_of_gt[i] >= COV_SHARE) if inter.size else np.zeros(0, int)
                any_overlap = np.flatnonzero(cov_of_gt[i] >= COV_MIN) if inter.size else np.zeros(0, int)
                # does any covering prediction also cover another cell?
                shared = False
                for j in covers:
                    if int((cov_of_gt[:, j] >= COV_SHARE).sum()) >= 2:
                        shared = True; break
                # border contact and local contrast
                ys, xs = np.nonzero(sel)
                touches_border = bool(ys.min() == 0 or xs.min() == 0 or
                                      ys.max() == H - 1 or xs.max() == W - 1)
                dil = ndi.binary_dilation(sel, iterations=3) & ~gt_fg
                inside_mean = float(img[sel].mean())
                ring_mean = float(img[dil].mean()) if dil.any() else float("nan")
                contrast = inside_mean - ring_mean

                if best_iou >= 0.5:
                    cat = "matched"
                elif fgcov < COV_MISS:
                    cat = "missing_foreground"
                elif len(any_overlap) == 0:
                    cat = "cleanup_lost"
                elif shared:
                    cat = "merged"
                elif len(covers) >= 2:
                    cat = "split"
                elif len(covers) == 1:
                    cat = "boundary_under"
                else:
                    cat = "other_low_iou"

                inst_rows.append(dict(
                    dataset=ds, image=name, role=role, arm=arm,
                    gt_instance=int(lab), area_px=area,
                    best_pred_iou=best_iou, fg_coverage=fgcov,
                    n_markers_inside=int(mk_count.get(int(lab), 0)),
                    n_pred_covering_25pct=int(len(covers)),
                    n_pred_overlap_10pct=int(len(any_overlap)),
                    shares_prediction_with_other_cell=bool(shared),
                    touches_border=touches_border,
                    inside_mean=inside_mean, ring_mean=ring_mean,
                    local_contrast=contrast, category=cat))

            # spurious predictions: best IoU with any GT < 0.5 and covering no
            # cell by >=25%
            if inter.size:
                best_per_pred = iou.max(axis=0)
                covers_any = (cov_of_gt >= COV_SHARE).sum(axis=0)
                for j, plab in enumerate(pl):
                    if best_per_pred[j] < 0.5 and covers_any[j] == 0:
                        spur_rows.append(dict(
                            dataset=ds, image=name, role=role, arm=arm,
                            pred_instance=int(plab), area_px=float(p_area[j]),
                            best_gt_iou=float(best_per_pred[j]),
                            frac_on_gt_fg=float(
                                (gt_fg & (inst == plab)).sum() / max(p_area[j], 1))))

    inst = pd.DataFrame(inst_rows)
    spur = pd.DataFrame(spur_rows)
    inst.to_csv(os.path.join(args.out, "instance_diagnostics.csv"), index=False)
    spur.to_csv(os.path.join(args.out, "tables", "spurious_predictions.csv"),
                index=False)

    # ── failure inventory, summarized BY IMAGE first ──
    cats = ["matched", "missing_foreground", "boundary_under", "merged", "split",
            "cleanup_lost", "other_low_iou"]
    inv: List[Dict] = []
    for (ds, name, role, arm), g in inst.groupby(["dataset", "image", "role", "arm"]):
        row = dict(dataset=ds, image=name, role=role, arm=arm, n_gt=len(g))
        for c in cats:
            row[c] = int((g["category"] == c).sum())
        row["instance_recall_iou50"] = float((g["best_pred_iou"] >= 0.5).mean())
        row["mean_fg_coverage"] = float(g["fg_coverage"].mean())
        row["n_spurious"] = int(len(spur[(spur.dataset == ds) & (spur.image == name)
                                         & (spur.arm == arm)]))
        inv.append(row)
    invdf = pd.DataFrame(inv)
    invdf.to_csv(os.path.join(args.out, "failure_inventory.csv"), index=False)

    print("=" * 128)
    print("PHASE 1  INSTANCE-LEVEL FAILURE INVENTORY  (from cached round-3 outputs; "
          "no model retrained)")
    print("  instance recall = fraction of GT cells with best predicted IoU >= 0.5 "
          "(NOT pixel foreground recall)")
    print("=" * 128)
    for ds in ["whole_cell", "nuclei"]:
        print(f"\n### {ds}")
        h = (f"  {'image':<10}{'arm':<10}{'#GT':>5}{'instRec':>8}{'fgCov':>7}"
             f"{'match':>6}{'missFG':>7}{'bndUnd':>7}{'merged':>7}{'split':>6}"
             f"{'clnLost':>8}{'other':>6}{'spur':>6}")
        print(h); print("  " + "-" * (len(h) - 2))
        s = invdf[invdf.dataset == ds]
        for _, r in s.sort_values(["image", "arm"]).iterrows():
            print(f"  {r['image'].split('_figure_')[-1]:<10}{r['arm']:<10}"
                  f"{int(r['n_gt']):>5}{r['instance_recall_iou50']:>8.3f}"
                  f"{r['mean_fg_coverage']:>7.3f}{int(r['matched']):>6}"
                  f"{int(r['missing_foreground']):>7}{int(r['boundary_under']):>7}"
                  f"{int(r['merged']):>7}{int(r['split']):>6}"
                  f"{int(r['cleanup_lost']):>8}{int(r['other_low_iou']):>6}"
                  f"{int(r['n_spurious']):>6}")
        # dataset-level category shares, per arm
        print(f"\n  {ds} category shares of all GT cells, by arm:")
        for arm in ARMS:
            g = inst[(inst.dataset == ds) & (inst.arm == arm)]
            if g.empty:
                continue
            tot = len(g)
            parts = "  ".join(f"{c}={100*(g.category==c).mean():.1f}%"
                              for c in cats if (g.category == c).any())
            print(f"    {arm:<10} n={tot:<5} {parts}")

    # pixel-vs-instance recall contrast, the key framing check
    print("\n" + "=" * 128)
    print("PIXEL vs INSTANCE RECALL  (the two are not the same quantity)")
    fg3 = pd.read_csv(os.path.join(R3, "heldout_foreground_metrics.csv"))
    print(f"  {'dataset':<12}{'arm':<10}{'pixel fg recall':>17}{'instance recall':>17}"
          f"{'gap':>8}")
    for ds in ["whole_cell", "nuclei"]:
        for arm in ARMS:
            a = fg3[(fg3.dataset == ds) & (fg3.arm == arm) &
                    (fg3.role == "heldout_candidate")]
            b = inst[(inst.dataset == ds) & (inst.arm == arm) &
                     (inst.role == "heldout_candidate")]
            if a.empty or b.empty:
                continue
            pr = a.fg_recall.mean(); ir = (b.best_pred_iou >= 0.5).mean()
            print(f"  {ds:<12}{arm:<10}{pr:>17.3f}{ir:>17.3f}{pr-ir:>8.3f}")

    print(f"\nWrote instance_diagnostics.csv ({len(inst)} GT instances), "
          f"failure_inventory.csv, tables/spurious_predictions.csv")


if __name__ == "__main__":
    main()
