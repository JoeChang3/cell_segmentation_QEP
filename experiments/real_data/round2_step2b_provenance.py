"""
Step 2B: reconcile the historical whole-cell GP AP (~0.583864) with the new
shared-pipeline GP result (~0.1158).

Four distinct operations are often conflated. They are kept separate here:

  (1) recomputing metrics from HISTORICAL SAVED PREDICTIONS
  (2) reproducing the LITERAL OLD PYTHON pipeline
  (3) running the NEW SHARED PYTHON pipeline
  (4) reproducing the PUBLISHED R/EBImage workflow

This script performs (1) and (3), states precisely what is required for (2) and
(4), and reports any artifact that is missing rather than guessing.

Key structural fact to test: EBImage::watershed performs its own internal marker
detection, whereas the Python translation calls
watershed(..., markers=None, ...). If the historical IoU matrix implies a
predicted-instance count close to marker-based separation rather than to
markers=None, that is direct evidence about which separation produced it.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.instance_separation import separate_instances
from py_core.metrics import compute_ap_from_ious
from py_core.segmentation_eval import evaluate_instances, load_instance_mask

ROUND1 = os.path.join(_ROOT, "results", "real_cellseg_round1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows: List[Dict] = []

    print("=" * 112)
    print("A. WHERE DOES EVERY STORED 'GP AP' VALUE LIVE?")
    print("=" * 112)
    targets = ["combined_ap_table", "ap_table"]
    for pat in ["data/*.csv", "results/tables/*.csv"]:
        for f in sorted(glob.glob(os.path.join(_ROOT, pat))):
            if not any(t in os.path.basename(f) for t in targets):
                continue
            d = pd.read_csv(f)
            gp_cols = [c for c in d.columns if "GP" in c and "AP" in c]
            if not gp_cols:
                continue
            rel = os.path.relpath(f, _ROOT)
            for _, r in d[d.get("Threshold", pd.Series()).eq(0.50)].iterrows():
                for c in gp_cols:
                    print(f"  {rel}")
                    print(f"    row Threshold=0.50  Pair={r.get('Pair','?')}  "
                          f"{c}={r[c]:.6f}")
                    rows.append(dict(operation="stored_table_value", file=rel,
                                     column=c, pair=r.get("Pair", ""),
                                     threshold=0.50, ap50=float(r[c])))

    print("\n" + "=" * 112)
    print("B. OPERATION (1): RECOMPUTE FROM HISTORICAL SAVED IoU MATRICES")
    print("=" * 112)
    for ds, folder in [("whole_cell", "whole_cell_test_images/whole_cell_figure_1"),
                       ("nuclei", "nuclear_test_images/nuclei_figure_1")]:
        base = os.path.join(_ROOT, "data", folder)
        for tag in ("ious_gp", "ious_imagej"):
            p = os.path.join(base, f"{tag}.csv")
            if not os.path.exists(p):
                print(f"  [missing] {os.path.relpath(p, _ROOT)}")
                continue
            m = pd.read_csv(p, index_col=0)
            r50 = compute_ap_from_ious(m, threshold=0.5)
            r75 = compute_ap_from_ious(m, threshold=0.75)
            print(f"  {ds}/{tag}.csv  matrix {m.shape} (rows=GT, cols=pred)")
            print(f"    AP@0.5={r50['precision']:.6f} TP={r50['tp']} "
                  f"FP={r50['fp']} FN={r50['fn']}   AP@0.75={r75['precision']:.6f}")
            rows.append(dict(operation="1_recompute_from_saved_iou_matrix",
                             file=os.path.relpath(p, _ROOT), dataset=ds, tag=tag,
                             n_gt_rows=m.shape[0], n_pred_cols=m.shape[1],
                             ap50=r50["precision"], ap75=r75["precision"],
                             tp50=r50["tp"], fp50=r50["fp"], fn50=r50["fn"]))

    print("\n" + "=" * 112)
    print("C. WHICH SEPARATION PRODUCES THE HISTORICAL PREDICTED-INSTANCE COUNT?")
    print("=" * 112)
    hist = pd.read_csv(os.path.join(
        _ROOT, "data", "whole_cell_test_images", "whole_cell_figure_1",
        "ious_gp.csv"), index_col=0)
    n_hist_pred = hist.shape[1]
    print(f"  historical whole-cell GP IoU matrix implies n_pred={n_hist_pred}")
    print(f"  (the historical predicted MASK ARRAY itself is not saved in the "
          f"repo; only\n   this IoU matrix and a rendered GP_boundaries.png, "
          f"which is a figure and\n   cannot be used as numerical input)")
    b = np.load(os.path.join(ROUND1, "masks", "whole_cell_gp_legacy.npz"))["binary"]
    truth = np.load(os.path.join(ROUND1, "masks", "whole_cell_ground_truth.npy"))
    print(f"\n  new-pipeline gp_legacy foreground, varying ONLY separation:")
    for mode, md in [("legacy_none", None), ("peak", 5), ("peak", 7),
                     ("peak", 9), ("peak", 12), ("peak", 15)]:
        s = separate_instances(b, marker_mode=mode,
                               min_distance=(md or 9), remove_size_threshold=50)
        ev = evaluate_instances(truth, s.instance_mask)
        tag = "markers=None" if md is None else f"peak md={md}"
        print(f"    {tag:<16} n_pred={ev.n_pred:>5}  AP@0.5={ev.ap(0.5):.4f}  "
              f"|n_pred - {n_hist_pred}|={abs(ev.n_pred-n_hist_pred):>4}")
        rows.append(dict(operation="3_new_shared_pipeline", dataset="whole_cell",
                         tag=f"gp_legacy_{tag}", n_pred_cols=ev.n_pred,
                         ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                         historical_n_pred=n_hist_pred,
                         abs_npred_gap=abs(ev.n_pred - n_hist_pred)))

    print("\n" + "=" * 112)
    print("D. WHAT WOULD BE NEEDED FOR OPERATIONS (2) AND (4)")
    print("=" * 112)
    missing = []
    gp_mask_candidates = glob.glob(os.path.join(
        _ROOT, "data", "whole_cell_test_images", "whole_cell_figure_1", "*GP*"))
    print("  files in the historical whole-cell folder matching *GP*:")
    for f in gp_mask_candidates:
        print(f"    {os.path.basename(f)}  ({os.path.getsize(f)/1024:.0f} KB)")
    print("\n  (2) LITERAL OLD PYTHON PIPELINE: runnable, since "
          "generate_gp_masks_test is\n      untouched. NOT run here because it "
          "predicts on all ~68k tile pixels in\n      one shot and was "
          "OOM-killed at this tile size in round 1; it also draws its\n      "
          "training subsample from an UNSEEDED np.random.choice, so it does not\n"
          "      produce a single well-defined number to compare against.")
    missing.append("a seeded, memory-safe run of the literal generate_gp_masks_test")
    print("\n  (4) PUBLISHED R/EBImage WORKFLOW: NOT reproducible in this repo.")
    r_files = glob.glob(os.path.join(_ROOT, "r_reference", "**", "*.R"),
                        recursive=True)
    print(f"      {len(r_files)} R scripts are present under r_reference/, but no R\n"
          f"      interpreter or EBImage installation is available here, and the\n"
          f"      historical GP instance MASK produced by that workflow was never\n"
          f"      saved as an array.")
    missing.append("an R/EBImage runtime plus the historical GP instance mask array")
    print("\n  MISSING ARTIFACTS required to close the gap exactly:")
    for m in missing:
        print(f"    - {m}")

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out, "legacy_gp_provenance.csv"), index=False)
    print(f"\nWrote legacy_gp_provenance.csv into {args.out}")
    print("=" * 112)


if __name__ == "__main__":
    main()
