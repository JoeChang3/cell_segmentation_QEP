"""
Step 1: isolate the watershed-marker change on the saved round-1 arrays.

Reuses the round-1 numerical outputs (results/real_cellseg_round1/masks/*.npz).
No GP or QEP model is refitted: the foreground binary produced in round 1 is
held fixed and ONLY marker generation changes, so any difference is
attributable to markers alone. Screenshots/figures are never read.

Also performed:
  * a marker-parameter study over an explicitly listed min_distance grid,
    applied identically to every reconstruction method (no per-method tuning);
  * an oracle-FOREGROUND diagnostic, kept in a separate table and clearly not a
    deployable method and not a mathematical upper bound - it substitutes the
    ground-truth foreground while keeping the real separation stage;
  * a reproducibility check: the whole postprocessing is run twice on identical
    saved inputs and compared on markers, foreground, instance partition (up to
    label renaming) and metrics.

Outputs (into --out):
  development_marker_results.csv
  oracle_foreground_diagnostics.csv
  marker_reproducibility.csv
  masks/<image>_<method>_<mode>.npz
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.instance_separation import (
    partitions_equivalent,
    separate_instances,
)
from py_core.segmentation_eval import classify_failure_modes, evaluate_instances

ROUND1 = os.path.join(_ROOT, "results", "real_cellseg_round1")
ARMS = ["raw", "gp_legacy", "gp", "qep_q2", "qep_q1.5"]
MIN_DISTANCES = [3, 5, 7, 9, 12, 15]
REMOVE_SIZE = 50


def metrics_row(truth, inst, **extra) -> Dict:
    ev = evaluate_instances(truth, inst)
    fm = classify_failure_modes(truth, inst)
    r = dict(
        ap50=ev.ap(0.5), ap75=ev.ap(0.75),
        tp50=ev.per_threshold[0.5]["tp"], fp50=ev.per_threshold[0.5]["fp"],
        fn50=ev.per_threshold[0.5]["fn"],
        tp75=ev.per_threshold[0.75]["tp"], fp75=ev.per_threshold[0.75]["fp"],
        fn75=ev.per_threshold[0.75]["fn"],
        n_pred=ev.n_pred, n_gt=ev.n_true,
        mean_matched_iou=ev.mean_matched_iou,
        merged=fm["merged"], split=fm["split"],
        missed=fm["missed"], spurious=fm["spurious"],
    )
    r.update(extra)
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--min-distances", type=int, nargs="+", default=MIN_DISTANCES)
    args = p.parse_args()
    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    dev_rows: List[Dict] = []
    oracle_rows: List[Dict] = []
    repro_rows: List[Dict] = []

    for ds in ("nuclei", "whole_cell"):
        gt_p = os.path.join(ROUND1, "masks", f"{ds}_ground_truth.npy")
        if not os.path.exists(gt_p):
            print(f"[skip] {ds}: round-1 ground truth not saved")
            continue
        truth = np.load(gt_p)
        n_gt = int(len(np.unique(truth[truth > 0])))
        print("=" * 112)
        print(f"{ds}  (GT instances={n_gt})   reusing round-1 saved binaries, "
              f"no model refitting")
        print("=" * 112)

        for arm in ARMS:
            p_npz = os.path.join(ROUND1, "masks", f"{ds}_{arm}.npz")
            if not os.path.exists(p_npz):
                print(f"  [missing] {ds}/{arm}: no saved round-1 array")
                continue
            d = np.load(p_npz)
            binary = d["binary"]
            legacy_inst_saved = d["instance_mask"]

            print(f"\n  {arm}")
            for mode, md in [("legacy_none", None)] + [("peak", m) for m in
                                                       args.min_distances]:
                t0 = time.time()
                sep = separate_instances(
                    binary, marker_mode=mode,
                    min_distance=(md if md is not None else 9),
                    remove_size_threshold=REMOVE_SIZE)
                rt = time.time() - t0

                row = metrics_row(
                    truth, sep.instance_mask,
                    dataset=ds, method=arm, marker_mode=mode,
                    min_distance=(np.nan if md is None else md),
                    n_markers=sep.n_markers,
                    n_fg_components=sep.n_fg_components,
                    n_components_rescued=sep.n_components_rescued,
                    runtime_s=rt, source="round1_saved_binary",
                    remove_size_threshold=REMOVE_SIZE,
                )
                dev_rows.append(row)

                tag = "legacy(markers=None)" if md is None else f"peak md={md}"
                print(f"    {tag:<22} AP@0.5={row['ap50']:.4f} AP@0.75={row['ap75']:.4f} "
                      f"n_pred={row['n_pred']:>5} markers={sep.n_markers:>5} "
                      f"rescued={sep.n_components_rescued:>3} "
                      f"split={row['split']:>3} merged={row['merged']:>3} "
                      f"({rt:.1f}s)")

                if md is None:
                    # sanity: does re-running legacy separation on the saved
                    # binary reproduce the instance mask round 1 saved?
                    same = partitions_equivalent(sep.instance_mask,
                                                 legacy_inst_saved)
                    print(f"      reproduces round-1 saved instance_mask "
                          f"(up to label renaming): {'YES' if same else 'NO'}")
                    repro_rows.append(dict(
                        dataset=ds, method=arm, check="legacy_vs_round1_saved",
                        partitions_equivalent=bool(same),
                        n_pred_a=row["n_pred"],
                        n_pred_b=int(len(np.unique(
                            legacy_inst_saved[legacy_inst_saved > 0]))),
                        ap50_a=row["ap50"],
                        ap50_b=evaluate_instances(truth, legacy_inst_saved).ap(0.5)))

                np.savez_compressed(
                    os.path.join(mask_dir,
                                 f"{ds}_{arm}_{mode}"
                                 f"{'' if md is None else f'_md{md}'}.npz"),
                    binary=binary.astype(np.uint8),
                    marker_image=sep.marker_image,
                    marker_coords=sep.marker_coords,
                    instance_mask=sep.instance_mask,
                    config=json.dumps(sep.config))

        # ---- oracle-FOREGROUND diagnostic (separate table) ----
        print(f"\n  ORACLE-FOREGROUND DIAGNOSTIC ({ds}) - not a deployable method,")
        print( "  not an upper bound; substitutes GT foreground, real separation stage")
        fg_oracle = (truth > 0).astype(np.uint8)
        for mode, md in [("legacy_none", None)] + [("peak", m) for m in
                                                   args.min_distances]:
            sep = separate_instances(fg_oracle, marker_mode=mode,
                                     min_distance=(md if md is not None else 9),
                                     remove_size_threshold=REMOVE_SIZE)
            row = metrics_row(truth, sep.instance_mask,
                              dataset=ds, method="ORACLE_FOREGROUND",
                              marker_mode=mode,
                              min_distance=(np.nan if md is None else md),
                              n_markers=sep.n_markers,
                              n_fg_components=sep.n_fg_components,
                              n_components_rescued=sep.n_components_rescued,
                              source="ground_truth_foreground")
            oracle_rows.append(row)
            tag = "legacy(markers=None)" if md is None else f"peak md={md}"
            print(f"    {tag:<22} AP@0.5={row['ap50']:.4f} n_pred={row['n_pred']:>5} "
                  f"markers={sep.n_markers:>5}")

        # ---- reproducibility: run postprocessing twice ----
        print(f"\n  REPRODUCIBILITY ({ds}): postprocessing run twice on identical inputs")
        for arm in ["gp", "qep_q1.5"]:
            p_npz = os.path.join(ROUND1, "masks", f"{ds}_{arm}.npz")
            if not os.path.exists(p_npz):
                continue
            binary = np.load(p_npz)["binary"]
            a = separate_instances(binary, marker_mode="peak", min_distance=9,
                                   remove_size_threshold=REMOVE_SIZE)
            b = separate_instances(binary, marker_mode="peak", min_distance=9,
                                   remove_size_threshold=REMOVE_SIZE)
            same_mk = np.array_equal(a.marker_coords, b.marker_coords)
            same_fg = np.array_equal(a.instance_mask > 0, b.instance_mask > 0)
            same_part = partitions_equivalent(a.instance_mask, b.instance_mask)
            ap_a = evaluate_instances(truth, a.instance_mask).ap(0.5)
            ap_b = evaluate_instances(truth, b.instance_mask).ap(0.5)
            print(f"    {arm:<10} markers identical={same_mk}  "
                  f"foreground identical={same_fg}  "
                  f"partition equivalent={same_part}  "
                  f"|dAP@0.5|={abs(ap_a-ap_b):.2e}")
            repro_rows.append(dict(
                dataset=ds, method=arm, check="rerun_twice_peak_md9",
                markers_identical=bool(same_mk),
                foreground_identical=bool(same_fg),
                partitions_equivalent=bool(same_part),
                ap50_a=ap_a, ap50_b=ap_b, ap50_abs_delta=abs(ap_a - ap_b)))
        print()

    pd.DataFrame(dev_rows).to_csv(
        os.path.join(args.out, "development_marker_results.csv"), index=False)
    pd.DataFrame(oracle_rows).to_csv(
        os.path.join(args.out, "oracle_foreground_diagnostics.csv"), index=False)
    pd.DataFrame(repro_rows).to_csv(
        os.path.join(args.out, "marker_reproducibility.csv"), index=False)
    print("=" * 112)
    print(f"Wrote development_marker_results.csv, oracle_foreground_diagnostics.csv, "
          f"marker_reproducibility.csv into {args.out}")
    print("=" * 112)


if __name__ == "__main__":
    main()
