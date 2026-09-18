"""
Round 3, Step 5: did fixing thresholding change the conclusions?

Round 2 applied criterion_1 PER TILE; round 3 applies one global rule to the
stitched reconstruction. Those are two differences, so the comparison separates
them by also scoring the round-2 cached binary masks directly (which ARE the
per-tile criterion_1 foreground) with the identical round-3 metric code.

Questions answered with numbers, per the round-3 brief:
  1 catastrophic all/nearly-all-background cases eliminated?
  2 held-out foreground Dice/IoU improved?
  3 between-image variance in foreground quality reduced?
  4 ranking among Raw/GP/q=2/q=1.5 more stable?
  5 does reconstruction now consistently beat Raw?
  6 does q=1.5 show a repeatable advantage?
  7 does GP remain exactly aligned with q=2?
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

from py_core.foreground_threshold import foreground_metrics
from py_core.segmentation_eval import load_instance_mask

R2 = os.path.join(_ROOT, "results", "real_cellseg_round2_20260915")
ARMS = ["raw", "gp", "qep_q2", "qep_q1.5"]
LABEL = {"raw": "Raw", "gp": "GP", "qep_q2": "QEP q=2", "qep_q1.5": "QEP q=1.5"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.out, "image_manifest.csv"))
    man = man[(man["status"] == "ok") & (man["role"] == "heldout_candidate")]
    r2seg = pd.read_csv(os.path.join(R2, "heldout_per_image_metrics.csv"))
    r3seg = pd.read_csv(os.path.join(args.out, "heldout_segmentation_metrics.csv"))
    r3seg = r3seg[r3seg["role"] == "heldout_candidate"]
    r3fg = pd.read_csv(os.path.join(args.out, "heldout_foreground_metrics.csv"))
    r3fg = r3fg[r3fg["role"] == "heldout_candidate"]

    # round-2 foreground quality, recomputed from its cached binaries with the
    # SAME metric code, so the two rounds are measured identically
    rows: List[Dict] = []
    for _, im in man.iterrows():
        ds, name = im["dataset"], im["image"]
        gt_fg = load_instance_mask(os.path.join(_ROOT, im["path_gt"])) > 0
        for arm in ARMS:
            p = os.path.join(R2, "masks", f"{ds}_{name}_{arm}.npz")
            if not os.path.exists(p):
                continue
            b = np.load(p)["binary"] > 0
            fm = foreground_metrics(b, gt_fg)
            rows.append(dict(dataset=ds, image=name, arm=arm, round="round2",
                             threshold_rule="criterion_1 (per tile)",
                             n_components=int(ndi.label(
                                 b, structure=ndi.generate_binary_structure(2, 1))[1]),
                             **fm))
    r2fg = pd.DataFrame(rows)

    comp_rows: List[Dict] = []
    print("=" * 118)
    print("STEP 5  ROUND 2 vs ROUND 3")
    print("  round2 = criterion_1 applied PER TILE; round3 = 'li' applied GLOBALLY")
    print("  everything else (markers, cleanup, watershed, metrics, reconstructions) identical")
    print("=" * 118)

    print("\n[1] catastrophic foreground failures on held-out images")
    for rnd, df in (("round2", r2fg), ("round3", r3fg)):
        bad_lo = df[df["fg_fraction"] < 0.01]
        bad_hi = df[df["fg_fraction"] > 0.90]
        zero = df[df["fg_fraction"] == 0.0]
        print(f"  {rnd}: all-background={len(zero)}  <1% foreground={len(bad_lo)}  "
              f">90% foreground={len(bad_hi)}   of {len(df)} (image,arm) cases")
        comp_rows.append(dict(question="1_catastrophic", round=rnd,
                              n_all_background=len(zero), n_lt1pct=len(bad_lo),
                              n_gt90pct=len(bad_hi), n_cases=len(df)))

    print("\n[2] held-out foreground Dice / IoU, mean over the 4 images")
    hdr = (f"  {'dataset':<12}{'arm':<11}{'Dice r2':>9}{'Dice r3':>9}{'dDice':>9}"
           f"{'IoU r2':>9}{'IoU r3':>9}{'dIoU':>8}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for ds in sorted(r3fg["dataset"].unique()):
        for arm in ARMS:
            a = r2fg[(r2fg.dataset == ds) & (r2fg.arm == arm)]
            b = r3fg[(r3fg.dataset == ds) & (r3fg.arm == arm)]
            if a.empty or b.empty:
                continue
            print(f"  {ds:<12}{arm:<11}{a.fg_dice.mean():>9.4f}{b.fg_dice.mean():>9.4f}"
                  f"{b.fg_dice.mean()-a.fg_dice.mean():>+9.4f}"
                  f"{a.fg_iou.mean():>9.4f}{b.fg_iou.mean():>9.4f}"
                  f"{b.fg_iou.mean()-a.fg_iou.mean():>+8.4f}")
            comp_rows.append(dict(question="2_foreground", dataset=ds, arm=arm,
                                  dice_round2=a.fg_dice.mean(),
                                  dice_round3=b.fg_dice.mean(),
                                  iou_round2=a.fg_iou.mean(),
                                  iou_round3=b.fg_iou.mean()))

    print("\n[3] between-image variability of foreground Dice (sd over the 4 images)")
    for ds in sorted(r3fg["dataset"].unique()):
        for arm in ARMS:
            a = r2fg[(r2fg.dataset == ds) & (r2fg.arm == arm)]
            b = r3fg[(r3fg.dataset == ds) & (r3fg.arm == arm)]
            if a.empty or b.empty:
                continue
            print(f"  {ds:<12}{arm:<11} sd round2={a.fg_dice.std():.4f}  "
                  f"round3={b.fg_dice.std():.4f}  "
                  f"{'reduced' if b.fg_dice.std() < a.fg_dice.std() else 'increased'}")
            comp_rows.append(dict(question="3_variance", dataset=ds, arm=arm,
                                  dice_sd_round2=a.fg_dice.std(),
                                  dice_sd_round3=b.fg_dice.std()))

    print("\n[4,5] held-out AP@0.5 and whether reconstruction beats Raw")
    hdr2 = (f"  {'dataset':<12}{'arm':<11}{'AP r2':>9}{'AP r3':>9}{'dAP':>9}"
            f"{'sd r2':>8}{'sd r3':>8}{'vs Raw r3':>11}")
    print(hdr2); print("  " + "-" * (len(hdr2) - 2))
    for ds in sorted(r3seg["dataset"].unique()):
        raw3 = r3seg[(r3seg.dataset == ds) & (r3seg.method == "raw")]["ap50"].mean()
        for arm in ARMS:
            a = r2seg[(r2seg.dataset == ds) & (r2seg.label == LABEL[arm])]
            b = r3seg[(r3seg.dataset == ds) & (r3seg.method == arm)]
            if a.empty or b.empty:
                continue
            d = b.ap50.mean() - raw3
            print(f"  {ds:<12}{arm:<11}{a.ap50.mean():>9.4f}{b.ap50.mean():>9.4f}"
                  f"{b.ap50.mean()-a.ap50.mean():>+9.4f}{a.ap50.std():>8.4f}"
                  f"{b.ap50.std():>8.4f}{d:>+11.4f}")
            comp_rows.append(dict(question="4_5_segmentation", dataset=ds, arm=arm,
                                  ap50_round2=a.ap50.mean(), ap50_round3=b.ap50.mean(),
                                  ap50_sd_round2=a.ap50.std(),
                                  ap50_sd_round3=b.ap50.std(),
                                  ap50_minus_raw_round3=d))

    print("\n[6] q=1.5 vs q=2 per held-out image, round 3")
    for ds in sorted(r3seg["dataset"].unique()):
        diffs = []
        for im in sorted(r3seg[r3seg.dataset == ds]["image"].unique()):
            a = r3seg[(r3seg.dataset == ds) & (r3seg.image == im) &
                      (r3seg.method == "qep_q1.5")]
            b = r3seg[(r3seg.dataset == ds) & (r3seg.image == im) &
                      (r3seg.method == "qep_q2")]
            if a.empty or b.empty:
                continue
            dd = float(a.iloc[0].ap50 - b.iloc[0].ap50); diffs.append(dd)
            print(f"  {ds}/{im.split('_figure_')[-1]}: {dd:+.4f}")
        if diffs:
            print(f"  -> {ds}: q=1.5 better on {sum(1 for x in diffs if x>0)}/{len(diffs)} "
                  f"images, mean {np.mean(diffs):+.4f}")
            comp_rows.append(dict(question="6_q_effect", dataset=ds,
                                  n_images=len(diffs),
                                  n_q15_better=sum(1 for x in diffs if x > 0),
                                  mean_delta=float(np.mean(diffs))))

    print("\n[7] GP vs QEP q=2 alignment, round 3")
    n_same = 0; n_tot = 0
    for ds in sorted(r3seg["dataset"].unique()):
        for im in sorted(r3seg[r3seg.dataset == ds]["image"].unique()):
            a = r3seg[(r3seg.dataset == ds) & (r3seg.image == im) & (r3seg.method == "gp")]
            b = r3seg[(r3seg.dataset == ds) & (r3seg.image == im) & (r3seg.method == "qep_q2")]
            if a.empty or b.empty:
                continue
            n_tot += 1
            same = abs(float(a.iloc[0].ap50) - float(b.iloc[0].ap50)) < 1e-12
            n_same += int(same)
    print(f"  identical AP@0.5 in {n_same}/{n_tot} held-out images")
    comp_rows.append(dict(question="7_gp_q2", n_identical=n_same, n_images=n_tot))

    print("\n[8] dominant remaining failure mode, round 3 (counts summed over images)")
    for ds in sorted(r3seg["dataset"].unique()):
        for arm in ARMS:
            s = r3seg[(r3seg.dataset == ds) & (r3seg.method == arm)]
            if s.empty:
                continue
            print(f"  {ds:<12}{arm:<11} merged={int(s.merged.sum()):>4} "
                  f"split={int(s.split.sum()):>4} missed={int(s.missed.sum()):>5} "
                  f"spurious={int(s.spurious.sum()):>5} "
                  f"(FP={int(s.fp50.sum()):>5} FN={int(s.fn50.sum()):>5})")
            comp_rows.append(dict(question="8_failure_modes", dataset=ds, arm=arm,
                                  merged=int(s.merged.sum()), split=int(s.split.sum()),
                                  missed=int(s.missed.sum()),
                                  spurious=int(s.spurious.sum()),
                                  fp50=int(s.fp50.sum()), fn50=int(s.fn50.sum())))

    pd.concat([r2fg.assign(round="round2"), r3fg.assign(round="round3")],
              ignore_index=True).to_csv(
        os.path.join(args.out, "tables", "foreground_round2_vs_round3.csv"), index=False)
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(args.out, "round2_vs_round3_comparison.csv"), index=False)
    print(f"\nWrote round2_vs_round3_comparison.csv")


if __name__ == "__main__":
    main()
