"""
Round 1: end-to-end real cell segmentation, raw vs GP vs QEP(q=2) vs QEP(q=1.5).

GOAL
----
Make the REAL nuclei / whole-cell segmentation experiment correct, reproducible
and end-to-end. This is not an attempt to demonstrate a QEP advantage.

The legacy real-data path was GP-only: `generate_gp_masks_test` hardcodes the GP
smoother and the adjacent QEP hook calls a nonexistent qpytorch API, so QEP had
never reached IoU/AP on real images. Here every method runs through one shared
pipeline (py_core/segmentation_pipeline.py) in which only the reconstruction
stage differs; thresholding, outlier handling, watershed and cleanup are
identical and the chosen threshold is logged per tile.

DEVELOPMENT CASES ONLY
----------------------
One nuclei image and one whole-cell image (the first valid pair in each
dataset). These are for pipeline validation and failure diagnosis, NOT
test-set evidence.

Usage:
    python experiments/real_data/real_cellseg_round1.py
    python experiments/real_data/real_cellseg_round1.py --max-points 2000
    python experiments/real_data/real_cellseg_round1.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import torch
from skimage.segmentation import find_boundaries

from py_core.segmentation_eval import (
    AP_THRESHOLDS,
    check_iou_parity,
    classify_failure_modes,
    evaluate_instances,
    load_gray_image,
    load_instance_mask,
)
from py_core.segmentation_pipeline import run_segmentation

OUT_DIR = os.path.join(_ROOT, "results", "real_cellseg_round1")

DATASETS = [
    dict(name="nuclei", folder=os.path.join(_ROOT, "data", "nuclear_test_images",
                                            "nuclei_figure_1"),
         image="original_fig.png", truth="original_true_masks.png",
         imagej="original_ImageJ_masks.tif"),
    dict(name="whole_cell", folder=os.path.join(_ROOT, "data",
                                                "whole_cell_test_images",
                                                "whole_cell_figure_1"),
         image="original_fig.jpg", truth="original_true_masks.png",
         imagej="original_ImageJ_masks.tif"),
]


def method_arms(q_values: List[float], include_legacy: bool) -> List[Dict]:
    arms = [dict(key="raw", method="raw", q=None, label="Raw")]
    if include_legacy:
        arms.append(dict(key="gp_legacy", method="gp_legacy", q=2.0,
                         label="GP (legacy)"))
    arms.append(dict(key="gp", method="gp", q=2.0, label="GP"))
    for q in q_values:
        arms.append(dict(key=f"qep_q{q:g}", method="qep", q=q,
                         label=f"QEP q={q:g}"))
    return arms


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _rand_label_cmap(n: int = 4096, seed: int = 0) -> ListedColormap:
    rng = np.random.default_rng(seed)
    cols = rng.random((n, 3)) * 0.75 + 0.25
    cols[0] = 0.0
    return ListedColormap(cols)


LBL_CMAP = _rand_label_cmap()


def _show_labels(ax, mask: np.ndarray, title: str) -> None:
    ax.imshow(mask % LBL_CMAP.N, cmap=LBL_CMAP, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def fig_comparison_grid(img: np.ndarray, truth: np.ndarray,
                        results: Dict[str, object], arms: List[Dict],
                        dataset: str, out_png: str) -> None:
    """Rows = reconstruction / binary / instance mask. Columns = methods + GT."""
    keys = [a["key"] for a in arms if a["key"] in results]
    ncol = len(keys) + 1
    fig, axs = plt.subplots(3, ncol, figsize=(2.9 * ncol, 9.2))
    axs = np.atleast_2d(axs)

    vmin, vmax = float(img.min()), float(img.max())
    for c, k in enumerate(keys):
        r = results[k]
        lab = next(a["label"] for a in arms if a["key"] == k)
        axs[0, c].imshow(r.combined_predmean, cmap="gray", vmin=vmin, vmax=vmax)
        axs[0, c].set_title(f"{lab}\nreconstruction", fontsize=9)
        axs[1, c].imshow(r.combined_thresholded > 0, cmap="gray")
        axs[1, c].set_title(f"binary (thr mean="
                            f"{np.mean(r.tile_thresholds):.2f})", fontsize=9)
        _show_labels(axs[2, c], r.instance_mask, f"instances n={r.n_instances}")
        for row in range(3):
            axs[row, c].set_xticks([]); axs[row, c].set_yticks([])

    axs[0, -1].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    axs[0, -1].set_title("Ground truth\noriginal image", fontsize=9)
    axs[1, -1].imshow(truth > 0, cmap="gray")
    axs[1, -1].set_title("GT foreground", fontsize=9)
    _show_labels(axs[2, -1], truth,
                 f"GT instances n={len(np.unique(truth[truth>0]))}")
    for row in range(3):
        axs[row, -1].set_xticks([]); axs[row, -1].set_yticks([])

    fig.suptitle(f"{dataset}: shared pipeline, reconstruction varies by method",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_overlay(img: np.ndarray, pred: np.ndarray, truth: np.ndarray,
                label: str, out_png: str) -> None:
    """Predicted vs GT boundaries on the original image, plus an error map."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    vmin, vmax = float(img.min()), float(img.max())

    axs[0].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    pb = find_boundaries(pred, mode="outer")
    ov = np.zeros(pred.shape + (4,))
    ov[pb] = (1, 0, 0, 1)
    axs[0].imshow(ov)
    axs[0].set_title(f"{label}: predicted boundaries (red)", fontsize=10)

    axs[1].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    tb = find_boundaries(truth, mode="outer")
    ov2 = np.zeros(pred.shape + (4,))
    ov2[tb] = (0, 1, 0, 1)
    ov2[pb] = (1, 0, 0, 1)
    both = pb & tb
    ov2[both] = (1, 1, 0, 1)
    axs[1].imshow(ov2)
    axs[1].set_title("predicted (red) vs GT (green), agree = yellow", fontsize=10)

    # foreground error map: FP magenta, FN cyan, TP grey
    pf, tf = pred > 0, truth > 0
    err = np.zeros(pred.shape + (3,))
    err[pf & tf] = (0.55, 0.55, 0.55)
    err[pf & ~tf] = (1, 0, 1)
    err[~pf & tf] = (0, 1, 1)
    axs[2].imshow(err)
    axs[2].set_title("foreground: TP grey / FP magenta / FN cyan", fontsize=10)

    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_watershed_inputs(r, label: str, out_png: str) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    axs[0].imshow(r.combined_thresholded > 0, cmap="gray")
    axs[0].set_title(f"{label}: binary (watershed mask)", fontsize=10)
    im = axs[1].imshow(r.dist_map, cmap="magma")
    axs[1].set_title("distance transform (watershed elevation = -this)",
                     fontsize=10)
    plt.colorbar(im, ax=axs[1], fraction=0.046)
    _show_labels(axs[2], r.instance_mask, f"instances n={r.n_instances}")
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--q", type=float, nargs="+", default=[2.0, 1.5],
                   help="QEP q values; 2.0 is the same-architecture Gaussian control")
    p.add_argument("--max-points", type=int, default=3000,
                   help="training pixels subsampled per tile (same for all methods)")
    p.add_argument("--train-iters", type=int, default=75)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--nu", type=float, default=2.5)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="precision for the controlled gp/qep arms. float32 "
                        "matches the legacy GP smoother exactly and is ~3x "
                        "faster; float64 is more robust but was measured at "
                        "~195s/tile, making the full run infeasible here.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--remove-size-threshold", type=int, default=50)
    p.add_argument("--no-legacy", action="store_true",
                   help="skip the gp_legacy continuity arm")
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["nuclei", "whole_cell"])
    p.add_argument("--out", type=str, default=OUT_DIR)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.max_points, args.train_iters = 400, 8
        args.q = [1.5]

    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    # keep the global default consistent with the arm precision: gpytorch
    # allocates some internal tensors at the default dtype, and a mismatch
    # surfaces as "expected m1 and m2 to have the same dtype".
    torch.set_default_dtype(torch_dtype)
    for sub in ("figures", "masks", "logs", "tables"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    arms = method_arms(args.q, include_legacy=not args.no_legacy)
    datasets = [d for d in DATASETS if d["name"] in args.datasets]

    print("=" * 84)
    print("Real cell segmentation, round 1 (development cases only)")
    print("=" * 84)
    print(f"  arms        : {[a['label'] for a in arms]}")
    print(f"  max_points  : {args.max_points} training px/tile (identical per method)")
    print(f"  train_iters : {args.train_iters}   lr={args.lr}   Matern nu={args.nu}")
    print(f"  precision   : {args.dtype} (gp_legacy arm is always float32)")
    print(f"  seed        : {args.seed} (per-tile deterministic RNG)")
    print(f"  AP thresholds: {AP_THRESHOLDS[0]}..{AP_THRESHOLDS[-1]}")
    print("=" * 84, flush=True)

    metric_rows: List[Dict] = []
    hyper_rows: List[Dict] = []
    runtime_rows: List[Dict] = []
    failure_rows: List[Dict] = []
    ap_curve_rows: List[Dict] = []
    notes: List[str] = []

    for ds in datasets:
        name = ds["name"]
        img_path = os.path.join(ds["folder"], ds["image"])
        truth_path = os.path.join(ds["folder"], ds["truth"])
        if not (os.path.exists(img_path) and os.path.exists(truth_path)):
            notes.append(f"{name}: missing image or truth, skipped")
            print(f"!! {name}: missing files, skipped")
            continue

        img = load_gray_image(img_path)
        truth = load_instance_mask(truth_path)
        n_true = int(len(np.unique(truth[truth > 0])))

        print(f"\n{'#'*84}\n### {name}: image {img.shape} range "
              f"[{img.min():.0f},{img.max():.0f}], GT instances={n_true}\n{'#'*84}",
              flush=True)
        if img.shape != truth.shape:
            raise SystemExit(f"{name}: image {img.shape} vs truth {truth.shape} "
                             "mismatch after native-orientation load")
        if img.shape[0] == img.shape[1]:
            notes.append(f"{name}: image is square; legacy align_mask_to_reference "
                         "would have been a silent no-op here")

        np.save(os.path.join(args.out, "masks", f"{name}_ground_truth.npy"), truth)
        np.save(os.path.join(args.out, "masks", f"{name}_image.npy"), img)

        # ImageJ comparator, for context only
        ij_path = os.path.join(ds["folder"], ds["imagej"])
        if os.path.exists(ij_path):
            ij = load_instance_mask(ij_path)
            if ij.shape == truth.shape:
                e = evaluate_instances(truth, ij)
                metric_rows.append(dict(
                    dataset=name, method="imagej", q=None, label="ImageJ",
                    n_true=e.n_true, n_pred=e.n_pred,
                    ap50=e.ap(0.5), ap75=e.ap(0.75), ap90=e.ap(0.9),
                    mean_matched_iou=e.mean_matched_iou,
                    tp50=e.per_threshold[0.5]["tp"], fp50=e.per_threshold[0.5]["fp"],
                    fn50=e.per_threshold[0.5]["fn"],
                    runtime_total_s=0.0, runtime_recon_s=0.0,
                    recon_rmse_vs_raw=np.nan, threshold_pct_mean=np.nan))
                print(f"  [context] ImageJ: AP@0.5={e.ap(0.5):.4f} "
                      f"AP@0.75={e.ap(0.75):.4f} n_pred={e.n_pred}")

        results: Dict[str, object] = {}
        for arm in arms:
            key, method, q = arm["key"], arm["method"], arm["q"]
            print(f"\n  --- {name} / {arm['label']} ---", flush=True)
            t0 = time.time()
            try:
                r = run_segmentation(
                    img, method=method, q=(q if q is not None else 2.0),
                    seed=args.seed, max_points=args.max_points,
                    train_iters=args.train_iters, lr=args.lr, nu=args.nu,
                    dtype=torch_dtype,
                    remove_size_threshold=args.remove_size_threshold,
                    verbose=True)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                with open(os.path.join(args.out, "logs",
                                       f"{name}_{key}_ERROR.txt"), "w") as fh:
                    fh.write(tb)
                notes.append(f"{name}/{key}: FAILED {type(exc).__name__}: {exc}")
                print(f"  !! FAILED {type(exc).__name__}: {exc}")
                continue
            results[key] = r

            ev = evaluate_instances(truth, r.instance_mask)
            fm = classify_failure_modes(truth, r.instance_mask)
            # reconstruction fidelity relative to the raw image (no clean
            # reference intensity target exists for real microscopy)
            rec_rmse = float(np.sqrt(np.mean((r.combined_predmean - img) ** 2)))

            print(f"    n_pred={ev.n_pred}  AP@0.5={ev.ap(0.5):.4f}  "
                  f"AP@0.75={ev.ap(0.75):.4f}  meanIoU={ev.mean_matched_iou:.4f}  "
                  f"TP/FP/FN={ev.per_threshold[0.5]['tp']}/"
                  f"{ev.per_threshold[0.5]['fp']}/{ev.per_threshold[0.5]['fn']}")
            print(f"    failures: merged={fm['merged']} split={fm['split']} "
                  f"missed={fm['missed']} spurious={fm['spurious']}")
            print(f"    recon RMSE vs raw={rec_rmse:.3f}  thr_mean="
                  f"{r.hypers['threshold_pct_mean']:.3f}  "
                  f"runtime={r.runtime_total_s:.0f}s "
                  f"(recon {r.runtime_recon_s:.0f}s)")

            metric_rows.append(dict(
                dataset=name, method=method, q=q, label=arm["label"],
                n_true=ev.n_true, n_pred=ev.n_pred,
                ap50=ev.ap(0.5), ap75=ev.ap(0.75), ap90=ev.ap(0.9),
                mean_matched_iou=ev.mean_matched_iou,
                tp50=ev.per_threshold[0.5]["tp"], fp50=ev.per_threshold[0.5]["fp"],
                fn50=ev.per_threshold[0.5]["fn"],
                runtime_total_s=r.runtime_total_s,
                runtime_recon_s=r.runtime_recon_s,
                recon_rmse_vs_raw=rec_rmse,
                threshold_pct_mean=r.hypers["threshold_pct_mean"]))
            failure_rows.append(dict(dataset=name, method=method, q=q,
                                     label=arm["label"], **fm))
            hyper_rows.append(dict(dataset=name, method=method, q=q,
                                   label=arm["label"], **r.hypers))
            runtime_rows.append(dict(dataset=name, method=method, q=q,
                                     label=arm["label"],
                                     runtime_total_s=r.runtime_total_s,
                                     runtime_recon_s=r.runtime_recon_s,
                                     n_tiles=r.hypers["n_tiles"]))
            for _, row in ev.ap_curve.iterrows():
                ap_curve_rows.append(dict(dataset=name, method=method, q=q,
                                          label=arm["label"], **row.to_dict()))

            # artifacts
            np.savez_compressed(
                os.path.join(args.out, "masks", f"{name}_{key}.npz"),
                predmean=r.combined_predmean.astype(np.float32),
                binary=r.combined_thresholded,
                instance_mask=r.instance_mask,
                dist_map=r.dist_map.astype(np.float32),
                tile_thresholds=np.array(r.tile_thresholds))
            with open(os.path.join(args.out, "logs",
                                   f"{name}_{key}_tiles.json"), "w") as fh:
                json.dump(r.tile_diags, fh, indent=2, default=str)

            fig_overlay(img, r.instance_mask, truth, f"{name} / {arm['label']}",
                        os.path.join(args.out, "figures",
                                     f"overlay_{name}_{key}.png"))
            fig_watershed_inputs(r, f"{name} / {arm['label']}",
                                 os.path.join(args.out, "figures",
                                              f"watershed_{name}_{key}.png"))

        if results:
            fig_comparison_grid(img, truth, results, arms, name,
                                os.path.join(args.out, "figures",
                                             f"comparison_{name}.png"))
            # IoU parity check, once per dataset, on the GP arm if present
            ref_key = "gp" if "gp" in results else list(results)[0]
            try:
                d = check_iou_parity(truth, results[ref_key].instance_mask)
                notes.append(f"{name}: fast-vs-legacy IoU parity max|diff|={d:.2e}")
                print(f"\n  IoU parity (fast vs legacy) on {ref_key}: "
                      f"max|diff|={d:.2e}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{name}: IoU parity check failed: {exc}")

    # ── tables ──
    md = pd.DataFrame(metric_rows)
    md.to_csv(os.path.join(args.out, "per_image_metrics.csv"), index=False)
    pd.DataFrame(hyper_rows).to_csv(
        os.path.join(args.out, "per_method_hyperparameters.csv"), index=False)
    pd.DataFrame(runtime_rows).to_csv(
        os.path.join(args.out, "runtime.csv"), index=False)
    pd.DataFrame(failure_rows).to_csv(
        os.path.join(args.out, "failure_modes.csv"), index=False)
    pd.DataFrame(ap_curve_rows).to_csv(
        os.path.join(args.out, "tables", "ap_curves.csv"), index=False)

    manifest = dict(
        experiment="real_cellseg_round1",
        purpose=("make the real nuclei/whole-cell segmentation pipeline correct, "
                 "reproducible and end-to-end; development cases only"),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        args=vars(args),
        arms=arms,
        datasets=[{k: v for k, v in d.items()} for d in datasets],
        ap_thresholds=AP_THRESHOLDS.tolist(),
        shared_pipeline=("tile -> [method-dependent reconstruction] -> criterion_1 "
                         "adaptive threshold -> outlier tile re-threshold -> stitch "
                         "-> distance transform -> watershed(markers=None) -> "
                         "eliminate_small_areas"),
        deviations_from_legacy=[
            "training-pixel subsample is seeded per tile (legacy used an unseeded "
            "np.random.choice, which is why stored nuclei GP AP was not reproducible)",
            "ground truth loaded in native orientation (legacy process_image_mask "
            "applied a net transpose); IoU is transpose-invariant so scores are "
            "comparable, but overlays are now correctly aligned",
            "IoU computed by contingency table (verified identical to the legacy "
            "pairwise implementation)",
            "controlled gp/qep arms run float64 and standardize each tile; the "
            "gp_legacy arm keeps the original float32 unstandardized path",
        ],
        notes=notes,
        env=dict(python=sys.version.split()[0], platform=platform.platform(),
                 torch=torch.__version__, numpy=np.__version__),
    )
    try:
        import gpytorch as _g, qpytorch as _q
        manifest["env"]["gpytorch"] = _g.__version__
        manifest["env"]["qpytorch"] = _q.__version__
    except Exception:  # noqa: BLE001
        pass
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # ── console summary ──
    print("\n" + "=" * 110)
    print("RESULTS")
    print("=" * 110)
    if not md.empty:
        hdr = (f"{'dataset':<12}{'method':<14}{'AP@0.5':>9}{'AP@0.75':>9}"
               f"{'meanIoU':>9}{'TP':>6}{'FP':>7}{'FN':>6}{'#pred':>7}"
               f"{'#true':>7}{'runtime':>9}")
        print(hdr); print("-" * len(hdr))
        for _, r in md.iterrows():
            print(f"{r['dataset']:<12}{r['label']:<14}{r['ap50']:>9.4f}"
                  f"{r['ap75']:>9.4f}{r['mean_matched_iou']:>9.4f}"
                  f"{int(r['tp50']):>6}{int(r['fp50']):>7}{int(r['fn50']):>6}"
                  f"{int(r['n_pred']):>7}{int(r['n_true']):>7}"
                  f"{r['runtime_total_s']:>8.0f}s")
    if failure_rows:
        print("\nFAILURE MODES (merged = touching cells fused; split = one cell "
              "fragmented)")
        fdf = pd.DataFrame(failure_rows)
        h2 = (f"{'dataset':<12}{'method':<14}{'merged':>8}{'split':>7}"
              f"{'missed':>8}{'spurious':>10}")
        print(h2); print("-" * len(h2))
        for _, r in fdf.iterrows():
            print(f"{r['dataset']:<12}{r['label']:<14}{int(r['merged']):>8}"
                  f"{int(r['split']):>7}{int(r['missed']):>8}"
                  f"{int(r['spurious']):>10}")
    if notes:
        print("\nNOTES")
        for n in notes:
            print(f"  - {n}")
    print(f"\nWrote {args.out}/")
    print("=" * 110)


if __name__ == "__main__":
    main()
