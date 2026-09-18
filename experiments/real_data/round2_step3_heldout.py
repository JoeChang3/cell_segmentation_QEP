"""
Step 3: freeze the protocol on the development images, then run the held-out images.

The frozen configuration is written to frozen_config.json BEFORE any held-out
score is computed, and every rule in it was chosen using only the two
development images (nuclei_figure_1, whole_cell_figure_1).

min_distance was selected on DEVELOPMENT data by maximizing the MEAN AP@0.5
across the deployable arms (raw, gp, qep q=2, qep q=1.5) - never per method:
    nuclei      min_distance = 15
    whole_cell  min_distance = 9
This is a documented dataset-specific rule, applied identically to every method
within a dataset. q is NOT selected per image; both 2.0 and 1.5 are reported.

Fitting the reconstructor to each held-out image is part of this unsupervised
pipeline and uses no ground truth. No method setting is chosen using held-out
ground-truth scores.

Checkpointing: each (image, method) result is cached as a .npz under masks/.
Re-running skips completed, configuration-compatible outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import torch

from py_core.segmentation_eval import (
    AP_THRESHOLDS,
    classify_failure_modes,
    evaluate_instances,
    load_gray_image,
    load_instance_mask,
)
from py_core.segmentation_pipeline import run_segmentation

# ── frozen, development-selected settings ─────────────────────────────────────
MIN_DISTANCE = {"nuclei": 15, "whole_cell": 9}
FROZEN = dict(
    protocol_version="round2-v1",
    selected_on="development images only (nuclei_figure_1, whole_cell_figure_1)",
    preprocessing=dict(
        grayscale="first channel if RGB, else as-is",
        orientation="native; ground truth relabelled WITHOUT transpose",
        intensity_units="raw image units (8-bit 0-255 for these files)",
        per_tile_standardization="z-score per tile, inverted before thresholding",
    ),
    tiling="legacy get_proportion rule (py_core.Modified_Functions_RGasp)",
    coordinates="joint 2D X=(row,col) linearly mapped to [0,1]^2 per tile",
    training_pixels=dict(max_points=3000,
                         sampler="np.random.default_rng([seed, tile_index])",
                         seed=0),
    determinism=dict(torch_seed="seed*1000 + tile_index", exact_logdet=True,
                     reason=("gpytorch max_cholesky_size=800 would otherwise "
                             "estimate the log-determinant with 10 random probe "
                             "vectors; unseeded, that alone moved the fitted "
                             "lengthscale by ~1.8e-3 between identical reruns")),
    model=dict(mean="ConstantMean", kernel="ScaleKernel(MaternKernel(nu=2.5))",
               ard=False, init="library defaults (nothing overridden)",
               dtype="float64"),
    fitting=dict(optimizer="Adam", lr=0.1, iters=75,
                 objective="ExactMarginalLogLikelihood"),
    q_values=[2.0, 1.5],
    thresholding=dict(rule="criterion_1", delta=0.01, nugget=True,
                      scope="per tile, then outlier-tile re-threshold at the "
                            "non-outlier mean"),
    markers=dict(mode="peak", implementation="skimage.feature.peak_local_max",
                 min_distance=MIN_DISTANCE, min_distance_units="pixels, Chebyshev (p_norm=inf)",
                 exclude_border=False, threshold_abs=None, threshold_rel=None,
                 labels="binary foreground",
                 empty_component_policy="one marker at the component's "
                                        "distance-transform argmax (first in C "
                                        "raster order)",
                 uses_ground_truth=False),
    cleanup=dict(rule="eliminate_small_areas", remove_size_threshold=50,
                 border_object_removal="none"),
    watershed=dict(elevation="-distance_transform_edt(foreground)",
                   connectivity=1, mask="binary foreground"),
    evaluation=dict(ap="TP/(TP+FP+FN)", thresholds=AP_THRESHOLDS.tolist(),
                    matching="per-GT-row argmax IoU, tau inclusive (>=)",
                    mean_matched_iou="max IoU per GT instance, averaged over "
                                     "ALL GT instances (undetected contribute 0)"),
)

ARMS = [
    dict(key="raw", method="raw", q=None, label="Raw"),
    dict(key="gp", method="gp", q=2.0, label="GP"),
    dict(key="qep_q2", method="qep", q=2.0, label="QEP q=2"),
    dict(key="qep_q1.5", method="qep", q=1.5, label="QEP q=1.5"),
    dict(key="gp_legacy", method="gp_legacy", q=2.0, label="GP-legacy-config"),
]


def fm_only(fm: Dict) -> Dict:
    """classify_failure_modes also returns n_true/n_pred, which collide with the
    explicit evaluation columns; keep only the failure-mode counts."""
    return {k: v for k, v in fm.items()
            if k in ("merged", "split", "missed", "spurious")}


def cfg_hash(d: Dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--roles", nargs="+", default=["heldout_candidate"])
    ap.add_argument("--arms", nargs="+", default=[a["key"] for a in ARMS])
    args = ap.parse_args()

    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    manifest_p = args.manifest or os.path.join(args.out, "image_manifest.csv")
    man = pd.read_csv(manifest_p)
    man = man[(man["status"] == "ok") & (man["role"].isin(args.roles))]

    torch.set_default_dtype(torch.float64)
    frozen = dict(FROZEN)
    frozen["environment"] = dict(
        python=sys.version.split()[0], platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__)
    try:
        import gpytorch as _g, qpytorch as _q, skimage as _s
        frozen["environment"].update(gpytorch=_g.__version__,
                                     qpytorch=_q.__version__,
                                     skimage=_s.__version__)
    except Exception:  # noqa: BLE001
        pass
    frozen["config_hash"] = cfg_hash({k: v for k, v in FROZEN.items()})
    frozen["run_id"] = f"{frozen['protocol_version']}_{frozen['config_hash']}"
    with open(os.path.join(args.out, "frozen_config.json"), "w") as fh:
        json.dump(frozen, fh, indent=2, default=str)

    print("=" * 118)
    print(f"STEP 3  HELD-OUT RUN   run_id={frozen['run_id']}")
    print("=" * 118)
    print(f"  frozen config written BEFORE any held-out score was computed")
    print(f"  min_distance (development-selected): {MIN_DISTANCE}")
    print(f"  images ({len(man)}): "
          f"{', '.join(man['dataset'] + '/' + man['image'])}")
    print(f"  arms: {args.arms}")
    print("=" * 118, flush=True)

    rows: List[Dict] = []
    for _, im in man.iterrows():
        ds, name = im["dataset"], im["image"]
        md = MIN_DISTANCE[ds]
        img = load_gray_image(os.path.join(_ROOT, im["path_image"]))
        truth = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        if img.shape != truth.shape:
            print(f"!! {ds}/{name}: shape mismatch img{img.shape} gt{truth.shape}, "
                  f"skipped")
            continue
        n_gt = int(len(np.unique(truth[truth > 0])))
        print(f"\n### {ds}/{name}  {img.shape}  GT={n_gt}  min_distance={md}",
              flush=True)

        # ImageJ comparator, from saved masks, same metric and same image
        if im.get("imagej_available") and im.get("imagej_shape_matches"):
            ij = load_instance_mask(os.path.join(_ROOT, im["path_imagej"]))
            ev = evaluate_instances(truth, ij)
            fm = classify_failure_modes(truth, ij)
            rows.append(dict(dataset=ds, image=name, method="imagej",
                             label="ImageJ (saved masks)", q=None,
                             ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                             tp50=ev.per_threshold[0.5]["tp"],
                             fp50=ev.per_threshold[0.5]["fp"],
                             fn50=ev.per_threshold[0.5]["fn"],
                             tp75=ev.per_threshold[0.75]["tp"],
                             fp75=ev.per_threshold[0.75]["fp"],
                             fn75=ev.per_threshold[0.75]["fn"],
                             n_pred=ev.n_pred, n_gt=ev.n_true,
                             mean_matched_iou=ev.mean_matched_iou,
                             runtime_s=0.0, rmse_to_raw=np.nan,
                             min_distance=np.nan, marker_mode="n/a",
                             provenance="pre-existing original_ImageJ_masks.tif; "
                                        "produced outside this repo, any manual "
                                        "tuning unknown", **fm_only(fm)))
            print(f"  imagej            AP@0.5={ev.ap(0.5):.4f} "
                  f"AP@0.75={ev.ap(0.75):.4f} n_pred={ev.n_pred}")

        for arm in ARMS:
            if arm["key"] not in args.arms:
                continue
            ck = os.path.join(mask_dir, f"{ds}_{name}_{arm['key']}.npz")
            if os.path.exists(ck):
                d = np.load(ck, allow_pickle=True)
                if str(d.get("run_id", "")) == frozen["run_id"]:
                    inst = d["instance_mask"]
                    ev = evaluate_instances(truth, inst)
                    fm = classify_failure_modes(truth, inst)
                    rows.append(dict(
                        dataset=ds, image=name, method=arm["method"],
                        label=arm["label"], q=arm["q"],
                        ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                        tp50=ev.per_threshold[0.5]["tp"],
                        fp50=ev.per_threshold[0.5]["fp"],
                        fn50=ev.per_threshold[0.5]["fn"],
                        tp75=ev.per_threshold[0.75]["tp"],
                        fp75=ev.per_threshold[0.75]["fp"],
                        fn75=ev.per_threshold[0.75]["fn"],
                        n_pred=ev.n_pred, n_gt=ev.n_true,
                        mean_matched_iou=ev.mean_matched_iou,
                        runtime_s=float(d["runtime_s"]),
                        rmse_to_raw=float(d["rmse_to_raw"]),
                        min_distance=md, marker_mode="peak",
                        provenance="cached", **fm_only(fm)))
                    print(f"  {arm['key']:<16} [cached] AP@0.5={ev.ap(0.5):.4f}")
                    continue

            t0 = time.time()
            try:
                r = run_segmentation(
                    img, method=arm["method"],
                    q=(arm["q"] if arm["q"] is not None else 2.0),
                    seed=FROZEN["training_pixels"]["seed"],
                    max_points=FROZEN["training_pixels"]["max_points"],
                    train_iters=FROZEN["fitting"]["iters"],
                    lr=FROZEN["fitting"]["lr"], nu=2.5,
                    dtype=torch.float64, exact_logdet=True,
                    remove_size_threshold=FROZEN["cleanup"]["remove_size_threshold"],
                    marker_mode="peak", min_distance=md, verbose=False)
            except Exception as exc:  # noqa: BLE001
                with open(os.path.join(args.out, "logs",
                                       f"{ds}_{name}_{arm['key']}_ERROR.txt"),
                          "w") as fh:
                    fh.write(traceback.format_exc())
                print(f"  {arm['key']:<16} FAILED {type(exc).__name__}: {exc}")
                rows.append(dict(dataset=ds, image=name, method=arm["method"],
                                 label=arm["label"], q=arm["q"],
                                 provenance=f"FAILED: {type(exc).__name__}",
                                 n_gt=n_gt))
                continue
            rt = time.time() - t0
            rmse_to_raw = float(np.sqrt(np.mean((r.combined_predmean - img) ** 2)))
            ev = evaluate_instances(truth, r.instance_mask)
            fm = classify_failure_modes(truth, r.instance_mask)

            np.savez_compressed(
                ck, instance_mask=r.instance_mask,
                binary=r.combined_thresholded,
                predmean=r.combined_predmean.astype(np.float32),
                marker_image=r.marker_image, marker_coords=r.marker_coords,
                runtime_s=np.float64(rt), rmse_to_raw=np.float64(rmse_to_raw),
                run_id=frozen["run_id"],
                hypers=json.dumps(r.hypers, default=str),
                separation_config=json.dumps(r.separation_config, default=str))

            rows.append(dict(
                dataset=ds, image=name, method=arm["method"], label=arm["label"],
                q=arm["q"], ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                tp50=ev.per_threshold[0.5]["tp"], fp50=ev.per_threshold[0.5]["fp"],
                fn50=ev.per_threshold[0.5]["fn"],
                tp75=ev.per_threshold[0.75]["tp"], fp75=ev.per_threshold[0.75]["fp"],
                fn75=ev.per_threshold[0.75]["fn"],
                n_pred=ev.n_pred, n_gt=ev.n_true,
                mean_matched_iou=ev.mean_matched_iou,
                runtime_s=rt, rmse_to_raw=rmse_to_raw,
                min_distance=md, marker_mode="peak",
                lengthscale_mean=r.hypers.get("lengthscale_mean", np.nan),
                noise_mean=r.hypers.get("noise_mean", np.nan),
                threshold_pct_mean=r.hypers.get("threshold_pct_mean", np.nan),
                n_markers=r.hypers.get("n_markers", np.nan),
                n_components_rescued=r.hypers.get("n_components_rescued", np.nan),
                provenance="this run", **fm_only(fm)))
            print(f"  {arm['key']:<16} AP@0.5={ev.ap(0.5):.4f} "
                  f"AP@0.75={ev.ap(0.75):.4f} n_pred={ev.n_pred:>4} "
                  f"TP/FP/FN={ev.per_threshold[0.5]['tp']}/"
                  f"{ev.per_threshold[0.5]['fp']}/{ev.per_threshold[0.5]['fn']} "
                  f"({rt:.0f}s)", flush=True)

        pd.DataFrame(rows).to_csv(
            os.path.join(args.out, "heldout_per_image_metrics.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "heldout_per_image_metrics.csv"), index=False)
    ok = df[df["ap50"].notna()] if "ap50" in df else df
    if not ok.empty:
        summ = (ok.groupby(["dataset", "label"])
                .agg(n_images=("ap50", "size"), ap50_mean=("ap50", "mean"),
                     ap50_std=("ap50", "std"), ap50_min=("ap50", "min"),
                     ap50_max=("ap50", "max"), ap75_mean=("ap75", "mean"),
                     mean_iou=("mean_matched_iou", "mean"),
                     npred_mean=("n_pred", "mean"), ngt_mean=("n_gt", "mean"),
                     runtime_mean=("runtime_s", "mean")).reset_index())
        summ.to_csv(os.path.join(args.out, "heldout_summary.csv"), index=False)
        print("\n" + "=" * 118)
        print("HELD-OUT SUMMARY (mean over images; no per-image method selection)")
        print("=" * 118)
        h = (f"{'dataset':<12}{'method':<20}{'n':>3}{'AP@0.5 mean':>13}"
             f"{'sd':>8}{'min':>8}{'max':>8}{'AP@0.75':>9}{'meanIoU':>9}"
             f"{'#pred':>8}{'#GT':>7}")
        print(h); print("-" * len(h))
        for _, r in summ.iterrows():
            sd = "  n/a" if pd.isna(r["ap50_std"]) else f"{r['ap50_std']:.4f}"
            print(f"{r['dataset']:<12}{r['label']:<20}{int(r['n_images']):>3}"
                  f"{r['ap50_mean']:>13.4f}{sd:>8}{r['ap50_min']:>8.4f}"
                  f"{r['ap50_max']:>8.4f}{r['ap75_mean']:>9.4f}"
                  f"{r['mean_iou']:>9.4f}{r['npred_mean']:>8.0f}"
                  f"{r['ngt_mean']:>7.0f}")
    print(f"\nWrote heldout_per_image_metrics.csv, heldout_summary.csv, "
          f"frozen_config.json into {args.out}")
    print("=" * 118)


if __name__ == "__main__":
    main()
