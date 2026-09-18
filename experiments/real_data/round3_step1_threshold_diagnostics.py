"""
Round 3, Steps 1-3: characterize criterion_1, benchmark candidate threshold
rules, and select one rule per dataset using DEVELOPMENT images only.

Inputs are the cached reconstruction arrays (`predmean`) from round 2, so no
GP/QEP model is refitted. Watershed markers, cleanup and evaluation stay at the
frozen round-2 settings and are not touched here.

Step 1 writes threshold_diagnostics.csv: per image/method intensity statistics,
the criterion_1 threshold expressed both as a fraction of max and as a
percentile of the image, foreground fraction, component count and foreground
agreement with ground truth.

Steps 2-3 write development_threshold_comparison.csv and
selected_threshold_config.json. Selection uses foreground Dice aggregated
ACROSS ALL FOUR reconstruction arms, never per arm, and never per image.
"""

from __future__ import annotations

import argparse
import glob
import json
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

from py_core.foreground_threshold import (
    apply_rule,
    candidate_grid,
    foreground_metrics,
)
from py_core.segmentation_eval import load_instance_mask

R2 = os.path.join(_ROOT, "results", "real_cellseg_round2_20260915")
ARMS = ["raw", "gp", "qep_q2", "qep_q1.5"]
QS = [1, 5, 25, 50, 75, 95, 99]


def recon_path(round_dir: str, ds: str, image: str, arm: str) -> str:
    return os.path.join(round_dir, "masks", f"{ds}_{image}_{arm}.npz")


def collect(manifest: pd.DataFrame, dev_dir: str) -> List[Dict]:
    """(dataset, image, arm, predmean, gt) for every available cached array."""
    out = []
    for _, im in manifest.iterrows():
        ds, name, role = im["dataset"], im["image"], im["role"]
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        src = dev_dir if role == "development" else R2
        for arm in ARMS:
            p = recon_path(src, ds, name, arm)
            if not os.path.exists(p):
                continue
            out.append(dict(dataset=ds, image=name, role=role, arm=arm,
                            predmean=np.load(p)["predmean"].astype(np.float64),
                            gt=gt, src=os.path.relpath(p, _ROOT)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-dir", required=True,
                    help="round-3 dir holding regenerated development reconstructions")
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.out, "image_manifest.csv"))
    man = man[man["status"] == "ok"]
    data = collect(man, args.dev_dir)
    print(f"loaded {len(data)} cached reconstruction arrays "
          f"({len({(d['dataset'],d['image']) for d in data})} images)")

    # ── STEP 1: criterion_1 diagnostics ──
    rows: List[Dict] = []
    for d in data:
        img, gt = d["predmean"], d["gt"]
        gt_fg = gt > 0
        r = apply_rule(img, "criterion_1")
        qv = np.percentile(img, QS)
        # where does the chosen absolute threshold sit in the image's own
        # intensity distribution?
        pct_of_dist = float((img < r.threshold).mean() * 100.0)
        ncomp = int(ndi.label(r.mask, structure=ndi.generate_binary_structure(2, 1))[1])
        fm = foreground_metrics(r.mask, gt_fg)
        rows.append(dict(
            dataset=d["dataset"], image=d["image"], role=d["role"], arm=d["arm"],
            img_min=float(img.min()), img_max=float(img.max()),
            img_mean=float(img.mean()), img_std=float(img.std()),
            **{f"q{q:02d}": float(v) for q, v in zip(QS, qv)},
            c1_threshold=r.threshold,
            c1_fraction_of_max=float(r.params["estimated_percentage"]),
            c1_percentile_of_distribution=pct_of_dist,
            c1_collapsed=bool(r.failed),
            n_components=ncomp, **fm, src=d["src"]))
    diag = pd.DataFrame(rows)
    diag.to_csv(os.path.join(args.out, "threshold_diagnostics.csv"), index=False)

    print("\n" + "=" * 124)
    print("STEP 1  criterion_1 DIAGNOSTICS (cached reconstructions; criterion_1 unchanged)")
    print("=" * 124)
    h = (f"{'dataset':<11}{'image':<10}{'arm':<10}{'role':<6}{'range':>14}"
         f"{'thr':>9}{'frac_max':>9}{'pctile':>8}{'fg%':>7}{'GTfg%':>7}"
         f"{'Dice':>7}{'IoU':>7}{'prec':>7}{'rec':>7}{'#cc':>6}")
    print(h); print("-" * len(h))
    for _, r in diag.iterrows():
        flag = " <-- COLLAPSED" if r["c1_collapsed"] else ""
        print(f"{r['dataset']:<11}{r['image'].split('_figure_')[-1]:<10}{r['arm']:<10}"
              f"{r['role'][:4]:<6}{f'{r.img_min:.0f}..{r.img_max:.0f}':>14}"
              f"{r['c1_threshold']:>9.1f}{r['c1_fraction_of_max']:>9.2f}"
              f"{r['c1_percentile_of_distribution']:>8.1f}"
              f"{100*r['fg_fraction']:>6.1f}%{100*r['gt_fg_fraction']:>6.1f}%"
              f"{r['fg_dice']:>7.3f}{r['fg_iou']:>7.3f}{r['fg_precision']:>7.3f}"
              f"{r['fg_recall']:>7.3f}{int(r['n_components']):>6}{flag}")

    print("\n  FAILURE PATTERNS")
    coll = diag[diag["c1_collapsed"]]
    print(f"    all-background collapses      : {len(coll)}/{len(diag)}"
          + (f"  [{', '.join(coll['image']+'/'+coll['arm'])}]" if len(coll) else ""))
    nearly_all = diag[diag["fg_fraction"] > 0.90]
    print(f"    nearly-all-foreground (>90%)  : {len(nearly_all)}/{len(diag)}"
          + (f"  [{', '.join(nearly_all['image']+'/'+nearly_all['arm'])}]"
             if len(nearly_all) else ""))
    nearly_none = diag[(diag["fg_fraction"] < 0.01) & (~diag["c1_collapsed"])]
    print(f"    nearly-no-foreground (<1%)    : {len(nearly_none)}/{len(diag)}")
    # GP vs q=2 divergence despite identical images
    print("    GP vs q=2 threshold divergence on identical reconstructions:")
    for (ds, im), g in diag.groupby(["dataset", "image"]):
        a = g[g["arm"] == "gp"]; b = g[g["arm"] == "qep_q2"]
        if a.empty or b.empty:
            continue
        ia = np.load(os.path.join(_ROOT, a.iloc[0]["src"]))["predmean"].astype(np.float64)
        ib = np.load(os.path.join(_ROOT, b.iloc[0]["src"]))["predmean"].astype(np.float64)
        dimg = float(np.abs(ia - ib).max())
        dthr = abs(float(a.iloc[0]["c1_threshold"]) - float(b.iloc[0]["c1_threshold"]))
        ddice = abs(float(a.iloc[0]["fg_dice"]) - float(b.iloc[0]["fg_dice"]))
        if dimg < 1e-12 and (dthr > 1e-9 or ddice > 1e-9):
            print(f"      {ds}/{im}: images identical but thr differs by {dthr:.3g}"
                  f" / Dice by {ddice:.3g}  <-- would indicate nondeterminism")
    print("      (none listed above means identical images give identical thresholds)")
    print("    sensitivity to the absolute scale: criterion_1 cuts at "
          "fraction*max, so the usable part of its p-grid is only "
          "[min/max, 1]; per image that floor is:")
    for (ds, im), g in diag.groupby(["dataset", "image"]):
        r0 = g.iloc[0]
        print(f"      {ds}/{im}: min/max = {r0['img_min']/max(r0['img_max'],1e-9):.2f}"
              f"  -> {100*r0['img_min']/max(r0['img_max'],1e-9):.0f}% of the grid "
              f"selects the whole image")

    # ── STEPS 2-3: candidate comparison on DEVELOPMENT images only ──
    dev = [d for d in data if d["role"] == "development"]
    print("\n" + "=" * 124)
    print("STEPS 2-3  CANDIDATE THRESHOLD RULES, DEVELOPMENT IMAGES ONLY")
    print(f"  development arrays: {len(dev)} "
          f"({len({(d['dataset'],d['image']) for d in dev})} images x {len(ARMS)} arms)")
    print("=" * 124)
    crows: List[Dict] = []
    for cand in candidate_grid():
        for d in dev:
            img, gt_fg = d["predmean"], d["gt"] > 0
            try:
                r = apply_rule(img, cand["rule"], cand["params"])
            except Exception as exc:  # noqa: BLE001
                crows.append(dict(rule=cand["name"], dataset=d["dataset"],
                                  image=d["image"], arm=d["arm"],
                                  error=f"{type(exc).__name__}: {exc}"))
                continue
            fm = foreground_metrics(r.mask, gt_fg)
            ncomp = int(ndi.label(r.mask,
                                  structure=ndi.generate_binary_structure(2, 1))[1])
            crows.append(dict(rule=cand["name"], rule_family=cand["rule"],
                              params=json.dumps(cand["params"]),
                              dataset=d["dataset"], image=d["image"], arm=d["arm"],
                              threshold=r.threshold, collapsed=bool(r.failed),
                              n_components=ncomp, **fm))
    comp = pd.DataFrame(crows)
    comp.to_csv(os.path.join(args.out, "development_threshold_comparison.csv"),
                index=False)

    sel: Dict[str, Dict] = {}
    for ds in sorted(comp["dataset"].dropna().unique()):
        s = comp[(comp["dataset"] == ds) & comp["fg_dice"].notna()]
        agg = (s.groupby(["rule", "rule_family", "params"])
               .agg(dice_mean=("fg_dice", "mean"), dice_min=("fg_dice", "min"),
                    dice_max=("fg_dice", "max"), iou_mean=("fg_iou", "mean"),
                    prec_mean=("fg_precision", "mean"),
                    rec_mean=("fg_recall", "mean"),
                    fgfrac_mean=("fg_fraction", "mean"),
                    cc_mean=("n_components", "mean"),
                    n_collapsed=("collapsed", "sum"), n=("fg_dice", "size"))
               .reset_index())
        # selection: best mean Dice across ALL arms, excluding any rule that
        # collapses on any arm. dice_min guards against catastrophic failure.
        agg["eligible"] = agg["n_collapsed"] == 0
        agg = agg.sort_values(["eligible", "dice_mean"], ascending=[False, False])
        print(f"\n--- {ds} (mean over {len(ARMS)} arms x "
              f"{len(s['image'].unique())} development image) ---")
        hh = (f"  {'rule':<20}{'Dice mean':>10}{'Dice min':>10}{'IoU':>8}"
              f"{'prec':>8}{'rec':>8}{'fg%':>7}{'#cc':>8}{'collapse':>9}")
        print(hh); print("  " + "-" * (len(hh) - 2))
        for _, r in agg.iterrows():
            mark = ""
            print(f"  {r['rule']:<20}{r['dice_mean']:>10.4f}{r['dice_min']:>10.4f}"
                  f"{r['iou_mean']:>8.4f}{r['prec_mean']:>8.3f}{r['rec_mean']:>8.3f}"
                  f"{100*r['fgfrac_mean']:>6.1f}%{r['cc_mean']:>8.0f}"
                  f"{int(r['n_collapsed']):>9}{mark}")
        best = agg[agg["eligible"]].iloc[0]
        sel[ds] = dict(rule=best["rule_family"],
                       params=json.loads(best["params"]),
                       name=best["rule"],
                       dev_dice_mean=float(best["dice_mean"]),
                       dev_dice_min=float(best["dice_min"]),
                       dev_iou_mean=float(best["iou_mean"]))
        print(f"  -> SELECTED for {ds}: {best['rule']} "
              f"(dev mean Dice {best['dice_mean']:.4f}, worst arm "
              f"{best['dice_min']:.4f})")

    cfg = dict(
        selection_rule=("highest mean foreground Dice across ALL four "
                        "reconstruction arms on development images only; rules "
                        "that collapse on any arm are ineligible"),
        development_images=["nuclei_figure_1", "whole_cell_figure_1"],
        arms_aggregated=ARMS,
        selected=sel,
        frozen_elsewhere=dict(
            markers="peak_local_max, min_distance nuclei=15 whole_cell=9",
            cleanup="eliminate_small_areas(50)",
            watershed="-distance_transform_edt, connectivity=1",
            evaluation="AP = TP/(TP+FP+FN)"),
    )
    with open(os.path.join(args.out, "selected_threshold_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\nWrote threshold_diagnostics.csv, development_threshold_comparison.csv, "
          f"selected_threshold_config.json into {args.out}")


if __name__ == "__main__":
    main()
