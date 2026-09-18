"""
Round 5: the CORRECTED-BASELINE experiment.

Question this round exists to answer: the Rounds 1-4 "GP" arm was NOT the paper's
Fast GP (audits/PAPER_FAST_GP_WIRING_REPORT.md). So does the *actual* paper
Fast-GP reconstruction behave differently, under a downstream that is frozen and
byte-identical across every arm?

SCOPE (deliberately small)
  Two DEVELOPMENT images only: nuclei_figure_1, whole_cell_figure_1.
  Held-out images are not touched.

ARMS (reconstruction is the ONLY thing that differs)
  1 raw                   the image itself
  2 paper_fast_gp         paper-faithful fitting: ONE L-BFGS-B run from the
                          paper's param_ini = (-2,-2,-3)
  3 paper_fast_gp_robust  SAME statistical model, MODIFIED OPTIMIZER
                          (multi-start, n_restarts=8). This is NOT the published
                          algorithm and is labelled as such everywhere.
  4 gp_legacy             the previous isotropic GPyTorch GP arm, display label
                          gp_isotropic_gpytorch_2025
  5 qep_q2                controlled QEP at q=2
  6 qep_q1.5              controlled QEP at q=1.5

FROZEN DOWNSTREAM -- identical for all six arms, nothing retuned per arm.
Taken from results/real_cellseg_round3_thresholding_20260916/:
  threshold : the Round-3 SELECTED rule, read from selected_threshold_config.json
              (li, applied GLOBALLY to the stitched reconstruction)
  markers   : peak_local_max, min_distance nuclei=15 / whole_cell=9
  cleanup   : eliminate_small_areas(50)
  watershed : -distance_transform_edt, connectivity=1, mask=foreground
  evaluation: evaluate_instances, AP = TP/(TP+FP+FN)
The downstream functions are imported from the same modules Round 3 used, so the
code path is not re-implemented here.

CACHING
Arms 1, 4, 5, 6 reuse the Round-3 cached `predmean` arrays verbatim -- no GP or
QEP model is refitted. Only arms 2 and 3 are computed. Verified beforehand that
the cached arrays have the right shapes and that cached `raw` predmean is
bit-identical to the loaded image.

NOTE ON TERMINOLOGY
RMSE between a reconstruction and the raw image is reported as `rmse_to_raw` and
is NOT a measure of reconstruction accuracy -- there is no ground-truth clean
image. It only quantifies how far an arm moved from the input.

Usage:
  python experiments/real_data/round5_corrected_baseline.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.foreground_threshold import apply_rule, foreground_metrics
from py_core.instance_separation import separate_instances
from py_core.paper_fast_gp import reconstruct_image
from py_core.segmentation_eval import (
    classify_failure_modes,
    evaluate_instances,
    load_instance_mask,
)

R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
MIN_DISTANCE = {"nuclei": 15, "whole_cell": 9}
REMOVE_SIZE = 50
N_RESTARTS_ROBUST = 8

CACHED_ARMS = ["raw", "gp_legacy", "qep_q2", "qep_q1.5"]
NEW_ARMS = ["paper_fast_gp", "paper_fast_gp_robust"]
ARM_ORDER = ["raw", "paper_fast_gp", "paper_fast_gp_robust",
             "gp_legacy", "qep_q2", "qep_q1.5"]
LABELS = {
    "raw": "Raw",
    "paper_fast_gp": "paper_fast_gp (paper-faithful opt)",
    "paper_fast_gp_robust": "paper_fast_gp (MODIFIED opt: multi-start)",
    "gp_legacy": "gp_isotropic_gpytorch_2025",
    "qep_q2": "QEP q=2",
    "qep_q1.5": "QEP q=1.5",
}


def load_gray(p: str) -> np.ndarray:
    """Round-3 loader: first channel if RGB, else as-is; raw 0-255 units."""
    img = imageio.imread(p)
    return (img[..., 0] if img.ndim == 3 else img).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out
    for sub in ("masks", "binary_masks", "instance_masks", "tables", "figures",
                "paper_fast_gp_fits"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    cfg = json.load(open(os.path.join(R3, "selected_threshold_config.json")))
    sel = cfg["selected"]
    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[(man["status"] == "ok") & (man["role"] == "development")]

    print("=" * 126)
    print("ROUND 5  CORRECTED-BASELINE EXPERIMENT  (development images only)")
    for ds, s in sel.items():
        print(f"  threshold {ds:<12} rule={s['name']} params={s['params']} "
              f"(GLOBAL, on the stitched reconstruction)")
    print(f"  frozen: markers=peak min_distance={MIN_DISTANCE}, "
          f"cleanup=eliminate_small_areas({REMOVE_SIZE}), "
          f"watershed=-EDT conn=1, AP=TP/(TP+FP+FN)")
    print(f"  cached (not refitted): {CACHED_ARMS}")
    print(f"  computed this round  : {NEW_ARMS}")
    print("=" * 126, flush=True)

    fg_rows: List[Dict] = []
    seg_rows: List[Dict] = []
    fit_rows: List[Dict] = []
    tile_rows: List[Dict] = []
    pair_rows: List[Dict] = []

    for _, im in man.iterrows():
        ds, name = im["dataset"], im["image"]
        raw = load_gray(os.path.join(_ROOT, im["path_image"]))
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        gt_fg = gt > 0
        n_gt = int(len(np.unique(gt[gt > 0])))
        md = MIN_DISTANCE[ds]
        rule, params = sel[ds]["rule"], sel[ds]["params"]

        print(f"\n### {ds}/{name}  {raw.shape[0]}x{raw.shape[1]}  GT={n_gt}  "
              f"rule={sel[ds]['name']}", flush=True)

        recons: Dict[str, np.ndarray] = {}
        recon_runtime: Dict[str, float] = {}

        # ---- cached arms: reuse verbatim -----------------------------------
        for arm in CACHED_ARMS:
            p = os.path.join(R3, "masks", f"{ds}_{name}_{arm}.npz")
            if not os.path.exists(p):
                print(f"  {arm:<38} [missing cached reconstruction] {p}")
                continue
            d = np.load(p)
            recons[arm] = d["predmean"].astype(np.float64)
            recon_runtime[arm] = float(d["runtime_s"])
            assert recons[arm].shape == raw.shape, f"{arm} shape mismatch"
        # cached raw must equal the loaded image, else the cache is stale
        if "raw" in recons:
            assert np.abs(recons["raw"] - raw).max() == 0.0, "cached raw != image"

        # ---- new arms: paper_fast_gp, two optimizers ----------------------
        for arm, nrs in (("paper_fast_gp", 0),
                         ("paper_fast_gp_robust", N_RESTARTS_ROBUST)):
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                t0 = time.time()
                r = reconstruct_image(raw, n_restarts=nrs)
                dt = time.time() - t0
                degen_msgs = [str(x.message) for x in wlist
                              if "degenerate" in str(x.message)]
            recons[arm] = r["reconstruction"]
            recon_runtime[arm] = dt
            sp = r["shared_params"]
            er = sp.effective_ranges_px()
            fit_rows.append(dict(
                dataset=ds, image=name, arm=arm, label=LABELS[arm],
                paper_faithful_optimizer=(nrs == 0), n_restarts=nrs,
                beta1=sp.beta1, beta2=sp.beta2, nugget=sp.nugget,
                neg_log_lik=sp.neg_log_lik, optim_method=sp.optim_method,
                n_obj_evals=sp.n_obj_evals, source_tile=sp.source_tile,
                eff_range_rows_px=er[0], eff_range_cols_px=er[1],
                degenerate_axes=",".join(sp.degenerate_axes) or "none",
                degeneracy_warning=bool(degen_msgs),
                n_tiles=len(r["per_tile"]),
                crop_height=r["grid"]["crop_height"],
                crop_width=r["grid"]["crop_width"],
                n_uncovered_px=r["n_uncovered_px"], remainder=r["remainder"],
                input_scale=r["input_scale"],
                runtime_recon_s=dt))
            for pt in r["per_tile"]:
                tile_rows.append(dict(dataset=ds, image=name, arm=arm, **pt))
            np.savez_compressed(
                os.path.join(out, "paper_fast_gp_fits", f"{ds}_{name}_{arm}.npz"),
                reconstruction=recons[arm].astype(np.float32),
                beta1=np.float64(sp.beta1), beta2=np.float64(sp.beta2),
                nugget=np.float64(sp.nugget),
                neg_log_lik=np.float64(sp.neg_log_lik),
                theta_hat=np.array([t["theta_hat"] for t in r["per_tile"]]),
                s_2=np.array([t["s_2"] for t in r["per_tile"]]),
                sigma2_hat=np.array([t["sigma2_hat"] for t in r["per_tile"]]),
                tile_y=np.array([t["y_offset"] for t in r["per_tile"]]),
                tile_x=np.array([t["x_offset"] for t in r["per_tile"]]),
                runtime_s=np.float64(dt))
            print(f"  {LABELS[arm]:<40} recon {dt:7.1f}s  "
                  f"beta1={sp.beta1:.5f} beta2={sp.beta2:.5f} nu={sp.nugget:.6f} "
                  f"f={sp.neg_log_lik:.4f} degen={','.join(sp.degenerate_axes) or 'none'}",
                  flush=True)

        # ---- frozen downstream, identical for every arm -------------------
        print(f"\n  {'arm':<40}{'thr':>8}{'fg%':>7}{'Dice':>8}{'AP@.5':>8}"
              f"{'AP@.75':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'nPred':>6}"
              f"{'merge':>6}{'split':>6}")
        for arm in ARM_ORDER:
            if arm not in recons:
                continue
            img = recons[arm]
            t0 = time.time()
            tr = apply_rule(img, rule, params)
            sep = separate_instances(tr.mask.astype(np.uint8), marker_mode="peak",
                                     min_distance=md,
                                     remove_size_threshold=REMOVE_SIZE)
            rt_down = time.time() - t0

            fm = foreground_metrics(tr.mask, gt_fg)
            ncomp = int(ndi.label(tr.mask,
                                  structure=ndi.generate_binary_structure(2, 1))[1])
            pctile = float((img < tr.threshold).mean() * 100.0)
            ev = evaluate_instances(gt, sep.instance_mask)
            fmode = classify_failure_modes(gt, sep.instance_mask)

            fg_rows.append(dict(dataset=ds, image=name, arm=arm, label=LABELS[arm],
                                threshold_rule=sel[ds]["name"],
                                threshold=tr.threshold, threshold_percentile=pctile,
                                collapsed=bool(tr.failed), n_components=ncomp, **fm))
            seg_rows.append(dict(
                dataset=ds, image=name, role="development", method=arm,
                label=LABELS[arm], reconstruction_source=(
                    "round3 cache" if arm in CACHED_ARMS else "computed round5"),
                ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                tp50=ev.per_threshold[0.5]["tp"], fp50=ev.per_threshold[0.5]["fp"],
                fn50=ev.per_threshold[0.5]["fn"],
                tp75=ev.per_threshold[0.75]["tp"], fp75=ev.per_threshold[0.75]["fp"],
                fn75=ev.per_threshold[0.75]["fn"],
                n_pred=ev.n_pred, n_gt=ev.n_true,
                mean_matched_iou=ev.mean_matched_iou,
                threshold_rule=sel[ds]["name"], threshold=tr.threshold,
                fg_dice=fm["fg_dice"], fg_iou=fm["fg_iou"],
                n_markers=sep.n_markers,
                merged=fmode["merged"], split=fmode["split"],
                missed=fmode["missed"], spurious=fmode["spurious"],
                runtime_recon_s=recon_runtime.get(arm, np.nan),
                runtime_downstream_s=rt_down,
                runtime_total_s=recon_runtime.get(arm, np.nan) + rt_down))

            np.savez_compressed(os.path.join(out, "masks", f"{ds}_{name}_{arm}.npz"),
                                predmean=img.astype(np.float32),
                                runtime_s=np.float64(recon_runtime.get(arm, np.nan)),
                                rmse_to_raw=np.float64(
                                    np.sqrt(((img - raw) ** 2).mean())))
            np.savez_compressed(
                os.path.join(out, "binary_masks", f"{ds}_{name}_{arm}.npz"),
                binary=tr.mask.astype(np.uint8),
                threshold=np.float64(tr.threshold), rule=sel[ds]["name"])
            np.savez_compressed(
                os.path.join(out, "instance_masks", f"{ds}_{name}_{arm}.npz"),
                instance_mask=sep.instance_mask, marker_image=sep.marker_image,
                marker_coords=sep.marker_coords)

            print(f"  {LABELS[arm]:<40}{tr.threshold:>8.2f}"
                  f"{100*fm['fg_fraction']:>7.1f}{fm['fg_dice']:>8.4f}"
                  f"{ev.ap(0.5):>8.4f}{ev.ap(0.75):>8.4f}"
                  f"{ev.per_threshold[0.5]['tp']:>5}{ev.per_threshold[0.5]['fp']:>5}"
                  f"{ev.per_threshold[0.5]['fn']:>5}{ev.n_pred:>6}"
                  f"{fmode['merged']:>6}{fmode['split']:>6}", flush=True)

        # ---- reconstruction-to-reconstruction diagnostics -----------------
        pairs = [("raw", "paper_fast_gp"),
                 ("paper_fast_gp", "gp_legacy"),
                 ("paper_fast_gp", "paper_fast_gp_robust")]
        print(f"\n  reconstruction-to-reconstruction (NOT accuracy; no clean "
              f"ground-truth image exists)")
        print(f"  {'pair':<58}{'corr':>11}{'RMSE':>10}{'max|diff|':>11}")
        for a, b in pairs:
            if a not in recons or b not in recons:
                continue
            A, B = recons[a], recons[b]
            d = A - B
            corr = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
            rmse = float(np.sqrt((d ** 2).mean()))
            mx = float(np.abs(d).max())
            pair_rows.append(dict(dataset=ds, image=name, arm_a=a, arm_b=b,
                                  label_a=LABELS[a], label_b=LABELS[b],
                                  corr=corr, rmse_between=rmse, max_abs_diff=mx))
            print(f"  {a+' vs '+b:<58}{corr:>11.6f}{rmse:>10.4f}{mx:>11.4f}")

    # ---- write tables ------------------------------------------------------
    pd.DataFrame(fg_rows).to_csv(
        os.path.join(out, "development_foreground_metrics.csv"), index=False)
    pd.DataFrame(seg_rows).to_csv(
        os.path.join(out, "development_segmentation_metrics.csv"), index=False)
    pd.DataFrame(fit_rows).to_csv(
        os.path.join(out, "paper_fast_gp_shared_params.csv"), index=False)
    pd.DataFrame(tile_rows).to_csv(
        os.path.join(out, "paper_fast_gp_per_tile.csv"), index=False)
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(out, "reconstruction_pair_diagnostics.csv"), index=False)

    json.dump(dict(
        round="round5-corrected-baseline",
        scope="development images only; held-out images untouched",
        threshold_config_source=os.path.relpath(
            os.path.join(R3, "selected_threshold_config.json"), _ROOT),
        selected_threshold=sel,
        min_distance=MIN_DISTANCE, remove_size_threshold=REMOVE_SIZE,
        marker_mode="peak", watershed="-distance_transform_edt, connectivity=1",
        ap_definition="TP/(TP+FP+FN)",
        cached_arms=CACHED_ARMS, cached_from=os.path.relpath(R3, _ROOT),
        computed_arms=NEW_ARMS, n_restarts_robust=N_RESTARTS_ROBUST,
        retuned_per_arm="nothing; downstream is byte-identical across arms",
        environment=dict(python=platform.python_version(),
                         platform=platform.platform(),
                         numpy=np.__version__, pandas=pd.__version__),
    ), open(os.path.join(out, "round5_config.json"), "w"), indent=2)

    # ---- summary ----------------------------------------------------------
    sg = pd.DataFrame(seg_rows)
    print("\n" + "=" * 126)
    print("ROUND 5 SUMMARY  (two development images, frozen identical downstream)")
    print("=" * 126)
    hdr = (f"{'dataset':<12}{'arm':<40}{'Dice':>8}{'AP@.5':>8}{'AP@.75':>8}"
           f"{'TP':>5}{'FP':>5}{'FN':>5}{'nPred':>6}{'nGT':>5}"
           f"{'merge':>6}{'split':>6}{'recon s':>9}{'total s':>9}")
    print(hdr); print("-" * len(hdr))
    for ds in sorted(sg["dataset"].unique()):
        for arm in ARM_ORDER:
            s = sg[(sg.dataset == ds) & (sg.method == arm)]
            if s.empty:
                continue
            r = s.iloc[0]
            print(f"{ds:<12}{LABELS[arm]:<40}{r.fg_dice:>8.4f}{r.ap50:>8.4f}"
                  f"{r.ap75:>8.4f}{int(r.tp50):>5}{int(r.fp50):>5}{int(r.fn50):>5}"
                  f"{int(r.n_pred):>6}{int(r.n_gt):>5}{int(r.merged):>6}"
                  f"{int(r.split):>6}{r.runtime_recon_s:>9.1f}"
                  f"{r.runtime_total_s:>9.1f}")
        print()
    print(f"Wrote tables into {out}")


if __name__ == "__main__":
    main()
