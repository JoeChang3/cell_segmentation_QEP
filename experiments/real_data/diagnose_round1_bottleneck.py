"""
Where is round-1 segmentation performance actually lost?

Reads the saved round-1 artifacts and decomposes the pipeline by substituting
ORACLE stages, so each stage's contribution is measured rather than guessed:

  stage_A  foreground quality   : Dice/IoU of the predicted binary vs GT foreground.
                                  Isolates reconstruction + threshold together.
  stage_B  oracle threshold     : replace the predicted binary with the GT
                                  foreground, keep the SAME watershed + cleanup.
                                  The resulting AP is the ceiling imposed by
                                  instance separation alone.
  stage_C  oracle separation    : GT instance mask scored against itself -> AP=1.
                                  Confirms the matcher is not the limit.
  stage_D  threshold sweep      : re-threshold each saved reconstruction across a
                                  range of percentages, keeping watershed +
                                  cleanup fixed, and report the best achievable
                                  AP. Gap between actual and best = how much the
                                  criterion_1 threshold choice costs.

Also quantifies the pipeline's own noise floor by comparing the `gp` and
`qep_q2` arms, which are the same model (power=2 reproduces the Gaussian
exactly), so any difference between them is pipeline noise rather than modeling.

Run after real_cellseg_round1.py:
    python experiments/real_data/diagnose_round1_bottleneck.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.Modified_Functions_RGasp import eliminate_small_areas, threshold_image
from py_core.segmentation_eval import evaluate_instances

RES = os.path.join(_ROOT, "results", "real_cellseg_round1")
OUT = os.path.join(RES, "tables")
ARMS = ["raw", "gp_legacy", "gp", "qep_q2", "qep_q1.5"]
REMOVE_SIZE = 50
SWEEP = np.round(np.arange(0.10, 0.96, 0.05), 2)


def instances_from_binary(binary: np.ndarray,
                          remove_size: int = REMOVE_SIZE) -> np.ndarray:
    """The exact shared post-threshold stage from the round-1 pipeline."""
    b = binary > 0
    dist = distance_transform_edt(b)
    lab = watershed(-dist, markers=None, mask=b).astype(np.int32)
    return eliminate_small_areas(lab, remove_size)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def fg_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows: List[Dict] = []
    sweep_rows: List[Dict] = []
    noise_rows: List[Dict] = []

    for ds in ("nuclei", "whole_cell"):
        gt_path = os.path.join(RES, "masks", f"{ds}_ground_truth.npy")
        if not os.path.exists(gt_path):
            print(f"[skip] {ds}: no saved ground truth")
            continue
        truth = np.load(gt_path)
        n_true = int(len(np.unique(truth[truth > 0])))
        print("=" * 96)
        print(f"{ds}: GT instances={n_true}, GT foreground="
              f"{100*(truth>0).mean():.1f}% of image")
        print("=" * 96)

        # ---- stage C: matcher sanity ----
        ev_self = evaluate_instances(truth, truth)
        print(f"  stage_C oracle separation (GT vs GT): AP@0.5={ev_self.ap(0.5):.4f}"
              f"  -> matcher {'OK' if ev_self.ap(0.5) > 0.999 else 'SUSPECT'}")

        # ---- stage B: oracle threshold, real watershed ----
        oracle_mask = instances_from_binary(truth > 0)
        ev_or = evaluate_instances(truth, oracle_mask)
        print(f"  stage_B oracle threshold + real watershed: "
              f"AP@0.5={ev_or.ap(0.5):.4f} AP@0.75={ev_or.ap(0.75):.4f} "
              f"n_pred={ev_or.n_pred} (GT {n_true})")
        print(f"          -> ceiling imposed by instance separation alone")
        rows.append(dict(dataset=ds, arm="ORACLE_threshold", ap50=ev_or.ap(0.5),
                         ap75=ev_or.ap(0.75), n_pred=ev_or.n_pred,
                         mean_iou=ev_or.mean_matched_iou,
                         fg_dice=1.0, fg_iou=1.0, note="GT foreground + real watershed"))
        rows.append(dict(dataset=ds, arm="ORACLE_separation", ap50=ev_self.ap(0.5),
                         ap75=ev_self.ap(0.75), n_pred=ev_self.n_pred,
                         mean_iou=ev_self.mean_matched_iou, fg_dice=1.0, fg_iou=1.0,
                         note="GT vs GT (matcher check)"))

        recons: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        print()
        for arm in ARMS:
            p = os.path.join(RES, "masks", f"{ds}_{arm}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            recon = d["predmean"].astype(np.float64)
            binary = d["binary"]
            inst = d["instance_mask"]
            recons[arm], masks[arm] = recon, inst

            ev = evaluate_instances(truth, inst)
            fgd, fgi = dice(binary, truth > 0), fg_iou(binary, truth > 0)
            thr = np.array(d["tile_thresholds"])
            n_collapse = int((thr >= 0.999).sum())
            print(f"  {arm:<10} AP@0.5={ev.ap(0.5):.4f}  fg_Dice={fgd:.4f}  "
                  f"fg_IoU={fgi:.4f}  n_pred={ev.n_pred:>4}  "
                  f"thr[{thr.min():.2f},{thr.max():.2f}] "
                  f"collapsed_tiles={n_collapse}/{thr.size}")
            rows.append(dict(dataset=ds, arm=arm, ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                             n_pred=ev.n_pred, mean_iou=ev.mean_matched_iou,
                             fg_dice=fgd, fg_iou=fgi,
                             thr_min=float(thr.min()), thr_max=float(thr.max()),
                             collapsed_tiles=n_collapse, n_tiles=int(thr.size),
                             note="actual"))

        # ---- stage D: threshold sweep on the saved reconstructions ----
        print(f"\n  stage_D threshold sweep (same watershed + cleanup, "
              f"only the threshold changes)")
        for arm, recon in recons.items():
            best = (-1.0, None)
            for pct in SWEEP:
                b = threshold_image(recon, float(pct), count=False)
                if b.sum() == 0 or b.sum() > 0.95 * b.size:
                    continue
                m = instances_from_binary(b)
                a = evaluate_instances(truth, m).ap(0.5)
                sweep_rows.append(dict(dataset=ds, arm=arm, pct=float(pct), ap50=a,
                                       fg_dice=dice(b, truth > 0)))
                if a > best[0]:
                    best = (a, float(pct))
            actual = next(r["ap50"] for r in rows
                          if r["dataset"] == ds and r["arm"] == arm
                          and r["note"] == "actual")
            print(f"    {arm:<10} actual AP@0.5={actual:.4f}  "
                  f"best-over-threshold={best[0]:.4f} @ pct={best[1]}  "
                  f"headroom=+{best[0]-actual:.4f}")

        # ---- pipeline noise floor: gp vs qep_q2 (same model) ----
        if "gp" in recons and "qep_q2" in recons:
            r1, r2 = recons["gp"], recons["qep_q2"]
            m1, m2 = masks["gp"], masks["qep_q2"]
            a1 = evaluate_instances(truth, m1).ap(0.5)
            a2 = evaluate_instances(truth, m2).ap(0.5)
            rec_rmse = float(np.sqrt(np.mean((r1 - r2) ** 2)))
            rec_max = float(np.abs(r1 - r2).max())
            print(f"\n  PIPELINE NOISE FLOOR (gp vs qep_q2, which are the SAME model)")
            print(f"    reconstruction difference: RMSE={rec_rmse:.4f}  "
                  f"max={rec_max:.4f}  (image range "
                  f"{r1.min():.0f}-{r1.max():.0f})")
            print(f"    binary agreement Dice={dice(masks['gp']>0, masks['qep_q2']>0):.4f}")
            print(f"    AP@0.5: gp={a1:.4f} vs qep_q2={a2:.4f} -> "
                  f"|delta|={abs(a1-a2):.4f}")
            print(f"    => any q effect smaller than {abs(a1-a2):.4f} AP is "
                  f"indistinguishable from pipeline noise")
            noise_rows.append(dict(dataset=ds, recon_rmse_gp_vs_qep2=rec_rmse,
                                   recon_max_gp_vs_qep2=rec_max,
                                   ap50_gp=a1, ap50_qep_q2=a2,
                                   ap50_abs_delta=abs(a1 - a2)))
        print()

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bottleneck_stages.csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(OUT, "threshold_sweep.csv"),
                                    index=False)
    pd.DataFrame(noise_rows).to_csv(os.path.join(OUT, "pipeline_noise_floor.csv"),
                                    index=False)
    print("=" * 96)
    print(f"Wrote {OUT}/bottleneck_stages.csv, threshold_sweep.csv, "
          f"pipeline_noise_floor.csv")
    print("=" * 96)


if __name__ == "__main__":
    main()
