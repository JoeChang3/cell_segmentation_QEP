"""
Step 2C: is the AP/IoU implementation CORRECT, not merely legacy-compatible?

Convention preserved from the project:
    AP(tau) = TP(tau) / [TP(tau) + FP(tau) + FN(tau)]

Matching rule as implemented in py_core.metrics.compute_ap_from_ious: for each
ground-truth row, take its single best-IoU prediction; count TP if that IoU
>= tau. FN = ground-truth rows never matched. FP = prediction columns never
selected as some row's best match.

NOTE ON ONE-TO-ONE-NESS: this rule is greedy per GT row and does NOT forbid one
prediction from being the best match of two different GT rows. The legacy code
comments acknowledge this. For tau >= 0.5 it is geometrically impossible for one
prediction to reach IoU >= 0.5 with two disjoint GT objects (the two IoUs would
require overlapping majorities of the same predicted area), so at the thresholds
reported here the rule IS effectively one-to-one. Test `double_match_guard`
below verifies that no double-counting occurs on the real development masks.

"mean matched IoU" is defined as: for every GROUND-TRUTH instance, its maximum
IoU against any prediction, averaged over all ground-truth instances. The
denominator is the number of ground-truth instances, so an undetected GT cell
contributes 0. It is NOT averaged over matched pairs only.

Checks: identity, empty prediction, empty truth, pure split, pure merge,
label-permutation invariance, and the double-match guard.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.segmentation_eval import (
    compute_ious_fast,
    evaluate_instances,
)

ROUND1 = os.path.join(_ROOT, "results", "real_cellseg_round1")


def two_squares() -> np.ndarray:
    """Truth: two 10x10 squares, well separated."""
    t = np.zeros((40, 40), dtype=np.int32)
    t[5:15, 5:15] = 1
    t[5:15, 25:35] = 2
    return t


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows: List[Dict] = []

    def check(name, truth, pred, expect: Dict, note=""):
        ev = evaluate_instances(truth, pred)
        got = dict(ap50=round(ev.ap(0.5), 6), ap75=round(ev.ap(0.75), 6),
                   tp50=ev.per_threshold[0.5]["tp"],
                   fp50=ev.per_threshold[0.5]["fp"],
                   fn50=ev.per_threshold[0.5]["fn"],
                   n_pred=ev.n_pred, n_gt=ev.n_true,
                   mean_matched_iou=round(ev.mean_matched_iou, 6))
        ok = all(abs(got[k] - v) < 1e-6 if isinstance(v, float) else got[k] == v
                 for k, v in expect.items())
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        for k, v in expect.items():
            flag = "" if (abs(got[k]-v) < 1e-6 if isinstance(v, float)
                          else got[k] == v) else "   <-- MISMATCH"
            print(f"          {k:<18} expected={v!s:<10} got={got[k]!s:<10}{flag}")
        if note:
            print(f"          note: {note}")
        rows.append(dict(check=name, passed=bool(ok), note=note,
                         **{f"got_{k}": v for k, v in got.items()},
                         **{f"exp_{k}": v for k, v in expect.items()}))
        return ok

    print("=" * 96)
    print("STEP 2C  METRIC CORRECTNESS CHECKS")
    print("=" * 96)
    t = two_squares()

    all_ok = True
    # 1 identity
    all_ok &= check("identity_gt_vs_gt", t, t.copy(),
                    dict(ap50=1.0, ap75=1.0, tp50=2, fp50=0, fn50=0,
                         n_pred=2, n_gt=2, mean_matched_iou=1.0))
    # 2 empty prediction
    all_ok &= check("empty_prediction", t, np.zeros_like(t),
                    dict(ap50=0.0, tp50=0, fp50=0, fn50=2, n_pred=0, n_gt=2,
                         mean_matched_iou=0.0),
                    "AP=0 with FN=2; no prediction columns so FP=0")
    # 3 empty truth: 2 predictions where nothing exists -> both are FP, AP=0
    all_ok &= check("empty_truth", np.zeros_like(t), t,
                    dict(ap50=0.0, tp50=0, fp50=2, fn50=0, n_pred=2, n_gt=0,
                         mean_matched_iou=0.0),
                    "no GT, 2 predictions: correctly 2 FP and AP=0; n_pred=2 "
                    "because two prediction columns do exist")
    # 4a EXACT-BOUNDARY split: cell1 cut into equal halves.
    # Each half has IoU = 50/(100+50-50) = 0.5 exactly, and the rule is >= tau,
    # so BOTH halves qualify at tau=0.5. This documents the inclusive boundary.
    p = t.copy(); p[5:10, 5:15] = 3
    all_ok &= check("split_equal_halves_boundary", t, p,
                    dict(tp50=2, fn50=0, n_pred=3, n_gt=2),
                    "IoU is exactly 0.5 for each half; tau is inclusive (>=), so "
                    "the half chosen by argmax counts as TP. Boundary case.")
    # 4b UNEQUAL split, away from the boundary: 70/30 of cell1.
    # IoU(GT1, big) = 70/100 = 0.7 -> TP; IoU(GT1, small) = 30/100 = 0.3.
    # The small fragment is never any row's argmax -> it is an FP.
    p2 = t.copy(); p2[5:8, 5:15] = 3
    all_ok &= check("split_unequal_70_30", t, p2,
                    dict(tp50=2, fp50=1, fn50=0, n_pred=3, n_gt=2),
                    "big fragment IoU=0.7 gives TP for GT1, cell2 perfect; the "
                    "30% fragment is unmatched and counted as FP")
    # 5 pure merge: one prediction covering both cells
    m = np.zeros_like(t); m[5:15, 5:35] = 1
    all_ok &= check("pure_merge_two_cells", t, m,
                    dict(tp50=0, fp50=1, fn50=2, n_pred=1, n_gt=2, ap50=0.0),
                    "single blob spans both cells: IoU=100/300=0.333 with each, "
                    "below 0.5, so no TP")
    # 6 label permutation invariance
    perm = t.copy(); perm[t == 1] = 7; perm[t == 2] = 3
    all_ok &= check("label_permutation_invariant", t, perm,
                    dict(ap50=1.0, tp50=2, fp50=0, fn50=0, mean_matched_iou=1.0),
                    "integer ids renamed only")

    # 7 double-match guard on the REAL development masks
    print("\n  double_match_guard on real development masks "
          "(is one prediction ever the best match of 2+ GT rows at tau>=0.5?)")
    worst = 0
    for ds in ("nuclei", "whole_cell"):
        gt = np.load(os.path.join(ROUND1, "masks", f"{ds}_ground_truth.npy"))
        for arm in ("raw", "gp", "qep_q2", "qep_q1.5"):
            f = os.path.join(ROUND1, "masks", f"{ds}_{arm}.npz")
            if not os.path.exists(f):
                continue
            inst = np.load(f)["instance_mask"]
            iou = compute_ious_fast(gt, inst).to_numpy()
            if iou.size == 0:
                continue
            best = iou.argmax(axis=1)
            hits = best[iou.max(axis=1) >= 0.5]
            dup = len(hits) - len(np.unique(hits))
            worst = max(worst, dup)
            print(f"    {ds}/{arm:<10} GT rows matched={len(hits):>4}  "
                  f"duplicate prediction reuse={dup}")
    ok = (worst == 0)
    print(f"  {'PASS' if ok else 'FAIL'}  matching is effectively one-to-one at "
          f"tau=0.5 (max duplicate reuse = {worst})")
    rows.append(dict(check="double_match_guard", passed=bool(ok),
                     note=f"max duplicate prediction reuse across real masks = {worst}"))
    all_ok &= ok

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out, "metric_correctness_checks.csv"), index=False)
    print("\n" + "=" * 96)
    print(f"ALL METRIC CHECKS {'PASSED' if all_ok else 'HAD FAILURES'}")
    print("mean matched IoU denominator = number of GROUND-TRUTH instances "
          "(undetected GT contributes 0)")
    print(f"Wrote metric_correctness_checks.csv into {args.out}")
    print("=" * 96)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
