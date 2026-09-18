"""
Round 3, Step 4: apply the frozen threshold rule to the held-out images.

Everything except the threshold rule is held at the round-2 frozen settings:
peak_local_max markers (min_distance nuclei=15, whole_cell=9),
eliminate_small_areas(50), watershed on -distance_transform_edt with
connectivity=1, and AP = TP/(TP+FP+FN).

Reconstructions are the cached round-2 `predmean` arrays. No GP or QEP model is
refitted, so the only thing that changed relative to round 2 is foreground
extraction.

Writes heldout_foreground_metrics.csv, heldout_segmentation_metrics.csv and
per-image mask arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.foreground_threshold import apply_rule, foreground_metrics
from py_core.instance_separation import separate_instances
from py_core.segmentation_eval import (
    classify_failure_modes,
    evaluate_instances,
    load_instance_mask,
)

R2 = os.path.join(_ROOT, "results", "real_cellseg_round2_20260915")
ARMS = ["raw", "gp", "qep_q2", "qep_q1.5"]
MIN_DISTANCE = {"nuclei": 15, "whole_cell": 9}
REMOVE_SIZE = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--roles", nargs="+", default=["heldout_candidate"])
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.out, "selected_threshold_config.json")))
    sel = cfg["selected"]
    man = pd.read_csv(os.path.join(args.out, "image_manifest.csv"))
    man = man[(man["status"] == "ok") & (man["role"].isin(args.roles))]

    bdir = os.path.join(args.out, "binary_masks")
    idir = os.path.join(args.out, "instance_masks")
    os.makedirs(bdir, exist_ok=True); os.makedirs(idir, exist_ok=True)

    print("=" * 122)
    print("ROUND 3 STEP 4  HELD-OUT with the frozen threshold rule")
    for ds, s in sel.items():
        print(f"  {ds:<12} rule={s['name']}  params={s['params']}")
    print(f"  frozen elsewhere: markers min_distance={MIN_DISTANCE}, "
          f"cleanup={REMOVE_SIZE}px, AP=TP/(TP+FP+FN)")
    print("=" * 122, flush=True)

    fg_rows: List[Dict] = []
    seg_rows: List[Dict] = []

    for _, im in man.iterrows():
        ds, name, role = im["dataset"], im["image"], im["role"]
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        gt_fg = gt > 0
        n_gt = int(len(np.unique(gt[gt > 0])))
        md = MIN_DISTANCE[ds]
        rule, params = sel[ds]["rule"], sel[ds]["params"]
        src_dir = args.dev_dir if role == "development" else R2
        print(f"\n### {ds}/{name}  GT={n_gt}  rule={sel[ds]['name']}", flush=True)

        # ImageJ comparator, unchanged by thresholding but reported for context
        if im.get("imagej_available") and im.get("imagej_shape_matches"):
            ij = load_instance_mask(os.path.join(_ROOT, im["path_imagej"]))
            ev = evaluate_instances(gt, ij)
            seg_rows.append(dict(dataset=ds, image=name, role=role,
                                 method="imagej", label="ImageJ (saved masks)",
                                 ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                                 tp50=ev.per_threshold[0.5]["tp"],
                                 fp50=ev.per_threshold[0.5]["fp"],
                                 fn50=ev.per_threshold[0.5]["fn"],
                                 n_pred=ev.n_pred, n_gt=ev.n_true,
                                 mean_matched_iou=ev.mean_matched_iou,
                                 threshold_rule="n/a (external masks)",
                                 runtime_s=0.0))
            print(f"  imagej      AP@0.5={ev.ap(0.5):.4f} AP@0.75={ev.ap(0.75):.4f}")

        for arm in ARMS:
            p = os.path.join(src_dir, "masks", f"{ds}_{name}_{arm}.npz")
            if not os.path.exists(p):
                print(f"  {arm:<11} [missing cached reconstruction] {p}")
                continue
            img = np.load(p)["predmean"].astype(np.float64)

            t0 = time.time()
            tr = apply_rule(img, rule, params)
            sep = separate_instances(tr.mask.astype(np.uint8), marker_mode="peak",
                                     min_distance=md,
                                     remove_size_threshold=REMOVE_SIZE)
            rt = time.time() - t0

            fm = foreground_metrics(tr.mask, gt_fg)
            ncomp = int(ndi.label(tr.mask,
                                  structure=ndi.generate_binary_structure(2, 1))[1])
            pctile = float((img < tr.threshold).mean() * 100.0)
            ev = evaluate_instances(gt, sep.instance_mask)
            fmode = classify_failure_modes(gt, sep.instance_mask)

            fg_rows.append(dict(dataset=ds, image=name, role=role, arm=arm,
                                threshold_rule=sel[ds]["name"],
                                threshold=tr.threshold,
                                threshold_percentile=pctile,
                                collapsed=bool(tr.failed),
                                n_components=ncomp, **fm))
            seg_rows.append(dict(
                dataset=ds, image=name, role=role, method=arm,
                label={"raw": "Raw", "gp": "GP", "qep_q2": "QEP q=2",
                       "qep_q1.5": "QEP q=1.5"}[arm],
                ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                tp50=ev.per_threshold[0.5]["tp"], fp50=ev.per_threshold[0.5]["fp"],
                fn50=ev.per_threshold[0.5]["fn"],
                tp75=ev.per_threshold[0.75]["tp"], fp75=ev.per_threshold[0.75]["fp"],
                fn75=ev.per_threshold[0.75]["fn"],
                n_pred=ev.n_pred, n_gt=ev.n_true,
                mean_matched_iou=ev.mean_matched_iou,
                threshold_rule=sel[ds]["name"], threshold=tr.threshold,
                fg_dice=fm["fg_dice"], fg_iou=fm["fg_iou"],
                n_markers=sep.n_markers, runtime_s=rt,
                merged=fmode["merged"], split=fmode["split"],
                missed=fmode["missed"], spurious=fmode["spurious"]))

            np.savez_compressed(os.path.join(bdir, f"{ds}_{name}_{arm}.npz"),
                                binary=tr.mask.astype(np.uint8),
                                threshold=np.float64(tr.threshold),
                                rule=sel[ds]["name"])
            np.savez_compressed(os.path.join(idir, f"{ds}_{name}_{arm}.npz"),
                                instance_mask=sep.instance_mask,
                                marker_image=sep.marker_image,
                                marker_coords=sep.marker_coords)

            print(f"  {arm:<11} thr={tr.threshold:>8.2f} (pctile {pctile:>5.1f}) "
                  f"fg={100*fm['fg_fraction']:>5.1f}% Dice={fm['fg_dice']:.4f} "
                  f"IoU={fm['fg_iou']:.4f} | AP@0.5={ev.ap(0.5):.4f} "
                  f"AP@0.75={ev.ap(0.75):.4f} n_pred={ev.n_pred}", flush=True)

    pd.DataFrame(fg_rows).to_csv(
        os.path.join(args.out, "heldout_foreground_metrics.csv"), index=False)
    pd.DataFrame(seg_rows).to_csv(
        os.path.join(args.out, "heldout_segmentation_metrics.csv"), index=False)

    fg = pd.DataFrame(fg_rows); sg = pd.DataFrame(seg_rows)
    ho = sg[(sg["role"] == "heldout_candidate") & sg["ap50"].notna()]
    print("\n" + "=" * 122)
    print("HELD-OUT SUMMARY (frozen threshold rule)")
    print("=" * 122)
    hh = (f"{'dataset':<12}{'method':<22}{'n':>3}{'Dice mean':>11}{'IoU mean':>10}"
          f"{'AP@0.5':>9}{'sd':>8}{'AP@0.75':>9}{'meanIoU':>9}")
    print(hh); print("-" * len(hh))
    for ds in sorted(ho["dataset"].unique()):
        for lab in ["Raw", "GP", "QEP q=2", "QEP q=1.5", "ImageJ (saved masks)"]:
            s = ho[(ho["dataset"] == ds) & (ho["label"] == lab)]
            if s.empty:
                continue
            f2 = fg[(fg["dataset"] == ds) &
                    (fg["arm"] == {"Raw": "raw", "GP": "gp", "QEP q=2": "qep_q2",
                                   "QEP q=1.5": "qep_q1.5"}.get(lab, "")) &
                    (fg["role"] == "heldout_candidate")]
            dm = f"{f2['fg_dice'].mean():.4f}" if not f2.empty else "   n/a"
            iu = f"{f2['fg_iou'].mean():.4f}" if not f2.empty else "   n/a"
            print(f"{ds:<12}{lab:<22}{len(s):>3}{dm:>11}{iu:>10}"
                  f"{s['ap50'].mean():>9.4f}{s['ap50'].std():>8.4f}"
                  f"{s['ap75'].mean():>9.4f}{s['mean_matched_iou'].mean():>9.4f}")
    print(f"\nWrote heldout_foreground_metrics.csv, heldout_segmentation_metrics.csv "
          f"into {args.out}")


if __name__ == "__main__":
    main()
