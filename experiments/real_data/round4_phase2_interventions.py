"""
Phase 2: stage-replacement diagnostics.

(A) ORACLE-FOREGROUND DIAGNOSTIC. Replace the predicted foreground with the
    union of ground-truth instances, keep the marker ALGORITHM and all
    parameters frozen, and RECOMPUTE markers from the substituted foreground
    (marker coordinates from the original foreground are never reused). This is
    a DIAGNOSTIC intervention. It is not deployable and it is not a mathematical
    upper bound on achievable AP: a different marker rule could score higher.

(B) ORACLE-MARKER DIAGNOSTIC. Keep the real predicted foreground but seed one
    marker at the centroid of each ground-truth instance that lies inside the
    foreground. Separates "foreground is wrong" from "seeding is wrong".

(C) THRESHOLD SCOPE CONFOUND. Round 2 applied criterion_1 per tile; round 3
    applied Li globally. Those are two changes. This runs the full 2x2
    (criterion_1 / Li) x (global / per-tile) on the DEVELOPMENT images with all
    downstream rules frozen, so algorithm and scope are separated.

All ground-truth-derived variants are labelled ORACLE and are excluded from any
deployable comparison.
"""

from __future__ import annotations

import argparse
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

from skimage.segmentation import watershed

from py_core.Modified_Functions_RGasp import (
    criterion_1,
    eliminate_small_areas,
    get_proportion,
)
from py_core.foreground_threshold import apply_rule, foreground_metrics
from py_core.instance_separation import separate_instances
from py_core.segmentation_eval import (
    evaluate_instances,
    load_gray_image,
    load_instance_mask,
)

R2 = os.path.join(_ROOT, "results", "real_cellseg_round2_20260915")
R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
ARMS = ["raw", "gp", "qep_q2", "qep_q1.5"]
MD = {"nuclei": 15, "whole_cell": 9}
RM = 50


def merged_count(gt: np.ndarray, inst: np.ndarray, share: float = 0.25) -> int:
    tl = np.unique(gt[gt > 0]); pl = np.unique(inst[inst > 0])
    if tl.size == 0 or pl.size == 0:
        return 0
    ti = np.zeros(int(gt.max()) + 1, dtype=np.int64); ti[tl] = np.arange(1, tl.size + 1)
    pi = np.zeros(int(inst.max()) + 1, dtype=np.int64); pi[pl] = np.arange(1, pl.size + 1)
    nt, npd = tl.size, pl.size
    hist = np.bincount(ti[gt.ravel()] * (npd + 1) + pi[inst.ravel()],
                       minlength=(nt + 1) * (npd + 1)).reshape(nt + 1, npd + 1)
    inter = hist[1:, 1:].astype(np.float64)
    t_area = hist[1:, :].sum(axis=1).astype(np.float64)
    cov = inter / np.maximum(t_area, 1)[:, None]
    # a GT cell is 'merged' if some prediction covers it and another cell
    return int(sum(1 for i in range(nt)
                   if any((cov[:, j] >= share).sum() >= 2
                          for j in np.flatnonzero(cov[i] >= share))))


def per_tile_threshold(img: np.ndarray, rule: str, params: Dict) -> np.ndarray:
    """Apply a threshold rule tile by tile using the legacy tiling, then stitch.
    Mirrors how round 2 applied criterion_1."""
    H, W = img.shape
    rp, cp = get_proportion(H), get_proportion(W)
    cw, ch = int(W * cp), int(H * rp)
    nx, ny = max(1, W // cw), max(1, H // ch)
    cw, ch = W // nx, H // ny
    out = np.zeros((H, W), dtype=bool)
    for i in range(nx):
        for j in range(ny):
            xo, yo = i * cw, j * ch
            pw = W - xo if i == nx - 1 else cw
            ph = H - yo if j == ny - 1 else ch
            tile = img[yo:yo + ph, xo:xo + pw]
            if rule == "criterion_1":
                out[yo:yo + ph, xo:xo + pw] = criterion_1(tile, 0.01, True).thresholded_image > 0
            else:
                out[yo:yo + ph, xo:xo + pw] = apply_rule(tile, rule, params).mask
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[man["status"] == "ok"]
    rows: List[Dict] = []

    # ═══ A + B : oracle interventions ═══
    print("=" * 126)
    print("PHASE 2A/2B  ORACLE DIAGNOSTICS  (ground truth used to replace an "
          "intermediate; NOT deployable, NOT an upper bound)")
    print("=" * 126)
    h = (f"  {'dataset':<11}{'image':<8}{'arm':<10}{'variant':<26}"
         f"{'AP@0.5':>8}{'AP@0.75':>9}{'#pred':>7}{'#GT':>6}{'merged':>8}")
    print(h); print("  " + "-" * (len(h) - 2))
    for _, im in man.iterrows():
        ds, name, role = im["dataset"], im["image"], im["role"]
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        gt_fg = (gt > 0).astype(np.uint8)
        n_gt = int(len(np.unique(gt[gt > 0])))

        # A: oracle foreground, markers RECOMPUTED from it
        sepA = separate_instances(gt_fg, marker_mode="peak",
                                  min_distance=MD[ds], remove_size_threshold=RM)
        evA = evaluate_instances(gt, sepA.instance_mask)
        rows.append(dict(dataset=ds, image=name, role=role, arm="(none)",
                         variant="ORACLE_foreground", oracle=True,
                         ap50=evA.ap(0.5), ap75=evA.ap(0.75), n_pred=evA.n_pred,
                         n_gt=n_gt, n_markers=sepA.n_markers,
                         merged=merged_count(gt, sepA.instance_mask)))
        print(f"  {ds:<11}{name.split('_figure_')[-1]:<8}{'(none)':<10}"
              f"{'ORACLE_foreground':<26}{evA.ap(0.5):>8.4f}{evA.ap(0.75):>9.4f}"
              f"{evA.n_pred:>7}{n_gt:>6}"
              f"{merged_count(gt, sepA.instance_mask):>8}")

        for arm in ARMS:
            bp = os.path.join(R3, "binary_masks", f"{ds}_{name}_{arm}.npz")
            if not os.path.exists(bp):
                continue
            binary = np.load(bp)["binary"] > 0

            # real pipeline, for reference in the same table
            sep0 = separate_instances(binary.astype(np.uint8), marker_mode="peak",
                                      min_distance=MD[ds], remove_size_threshold=RM)
            ev0 = evaluate_instances(gt, sep0.instance_mask)
            rows.append(dict(dataset=ds, image=name, role=role, arm=arm,
                             variant="real_pipeline", oracle=False,
                             ap50=ev0.ap(0.5), ap75=ev0.ap(0.75),
                             n_pred=ev0.n_pred, n_gt=n_gt,
                             n_markers=sep0.n_markers,
                             merged=merged_count(gt, sep0.instance_mask)))

            # B: real foreground, ORACLE markers (one per GT centroid inside fg)
            cents = ndi.center_of_mass(gt > 0, gt,
                                       np.unique(gt[gt > 0]).tolist())
            mk = np.zeros(gt.shape, dtype=np.int32)
            k = 0
            for (cy, cx) in cents:
                yy, xx = int(round(cy)), int(round(cx))
                if 0 <= yy < gt.shape[0] and 0 <= xx < gt.shape[1] and binary[yy, xx]:
                    k += 1; mk[yy, xx] = k
            dist = ndi.distance_transform_edt(binary)
            labB = watershed(-dist, markers=mk, mask=binary,
                             connectivity=1).astype(np.int32)
            instB = eliminate_small_areas(labB, RM)
            evB = evaluate_instances(gt, instB)
            rows.append(dict(dataset=ds, image=name, role=role, arm=arm,
                             variant="ORACLE_markers_real_fg", oracle=True,
                             ap50=evB.ap(0.5), ap75=evB.ap(0.75),
                             n_pred=evB.n_pred, n_gt=n_gt, n_markers=k,
                             merged=merged_count(gt, instB)))
            if arm in ("raw", "gp", "qep_q1.5"):
                print(f"  {ds:<11}{name.split('_figure_')[-1]:<8}{arm:<10}"
                      f"{'real_pipeline':<26}{ev0.ap(0.5):>8.4f}{ev0.ap(0.75):>9.4f}"
                      f"{ev0.n_pred:>7}{n_gt:>6}"
                      f"{merged_count(gt, sep0.instance_mask):>8}")
                print(f"  {ds:<11}{name.split('_figure_')[-1]:<8}{arm:<10}"
                      f"{'ORACLE_markers_real_fg':<26}{evB.ap(0.5):>8.4f}"
                      f"{evB.ap(0.75):>9.4f}{evB.n_pred:>7}{n_gt:>6}"
                      f"{merged_count(gt, instB):>8}")

    df = pd.DataFrame(rows)

    print("\n  DATASET MEANS (held-out images only)")
    ho = df[df.role == "heldout_candidate"]
    for ds in ["whole_cell", "nuclei"]:
        print(f"    {ds}:")
        o = ho[(ho.dataset == ds) & (ho.variant == "ORACLE_foreground")]
        print(f"      ORACLE_foreground                AP@0.5={o.ap50.mean():.4f}  "
              f"merged={o.merged.mean():.1f}/{o.n_gt.mean():.0f}")
        for arm in ARMS:
            a = ho[(ho.dataset == ds) & (ho.arm == arm) &
                   (ho.variant == "real_pipeline")]
            b = ho[(ho.dataset == ds) & (ho.arm == arm) &
                   (ho.variant == "ORACLE_markers_real_fg")]
            if a.empty:
                continue
            print(f"      {arm:<12} real AP@0.5={a.ap50.mean():.4f} "
                  f"merged={a.merged.mean():>5.1f} | ORACLE-markers "
                  f"AP@0.5={b.ap50.mean():.4f} merged={b.merged.mean():>5.1f}")

    # ═══ C : threshold algorithm x scope, DEVELOPMENT only ═══
    print("\n" + "=" * 126)
    print("PHASE 2C  THRESHOLD ALGORITHM x SCOPE, DEVELOPMENT IMAGES ONLY "
          "(downstream frozen)")
    print("=" * 126)
    dev = man[man.role == "development"]
    crows: List[Dict] = []
    hh = (f"  {'dataset':<11}{'arm':<10}{'rule':<14}{'scope':<10}{'fgDice':>8}"
          f"{'AP@0.5':>8}{'#pred':>7}{'merged':>8}{'collapsed tiles':>16}")
    print(hh); print("  " + "-" * (len(hh) - 2))
    for _, im in dev.iterrows():
        ds, name = im["dataset"], im["image"]
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        gtfg = gt > 0
        for arm in ARMS:
            p = os.path.join(R3, "masks", f"{ds}_{name}_{arm}.npz")
            if not os.path.exists(p):
                continue
            img = np.load(p)["predmean"].astype(np.float64)
            for rule, params in [("criterion_1", {}), ("li", {})]:
                for scope in ["global", "per_tile"]:
                    if scope == "global":
                        b = apply_rule(img, rule, params).mask
                    else:
                        b = per_tile_threshold(img, rule, params)
                    sep = separate_instances(b.astype(np.uint8), marker_mode="peak",
                                             min_distance=MD[ds],
                                             remove_size_threshold=RM)
                    ev = evaluate_instances(gt, sep.instance_mask)
                    fm = foreground_metrics(b, gtfg)
                    mg = merged_count(gt, sep.instance_mask)
                    crows.append(dict(dataset=ds, image=name, arm=arm, rule=rule,
                                      scope=scope, fg_dice=fm["fg_dice"],
                                      fg_iou=fm["fg_iou"], ap50=ev.ap(0.5),
                                      ap75=ev.ap(0.75), n_pred=ev.n_pred,
                                      merged=mg))
                    if arm in ("raw", "gp"):
                        print(f"  {ds:<11}{arm:<10}{rule:<14}{scope:<10}"
                              f"{fm['fg_dice']:>8.4f}{ev.ap(0.5):>8.4f}"
                              f"{ev.n_pred:>7}{mg:>8}{'':>16}")
    cdf = pd.DataFrame(crows)
    print("\n  MEANS over development images x 4 arms (separates algorithm from scope)")
    for ds in ["whole_cell", "nuclei"]:
        s = cdf[cdf.dataset == ds]
        if s.empty:
            continue
        print(f"    {ds}:")
        for rule in ["criterion_1", "li"]:
            for scope in ["global", "per_tile"]:
                t = s[(s.rule == rule) & (s.scope == scope)]
                if t.empty:
                    continue
                print(f"      {rule:<12} {scope:<9} fgDice={t.fg_dice.mean():.4f}  "
                      f"AP@0.5={t.ap50.mean():.4f}  merged={t.merged.mean():.1f}")

    pd.concat([df, cdf.assign(variant="threshold_scope_2x2", oracle=False)],
              ignore_index=True).to_csv(
        os.path.join(args.out, "stage_interventions.csv"), index=False)
    cdf.to_csv(os.path.join(args.out, "tables", "threshold_scope_2x2.csv"),
               index=False)
    print(f"\nWrote stage_interventions.csv, tables/threshold_scope_2x2.csv")


if __name__ == "__main__":
    main()
