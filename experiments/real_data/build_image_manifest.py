"""
Build an image-level manifest for the real cell-segmentation datasets.

Purpose: decide which images may legitimately serve as held-out evaluation
cases. Two images from the same underlying field, or overlapping crops, are not
independent, and tiles of one image are certainly not.

Checks performed per image:
  * file identity (md5 of raw bytes) -> exact duplicates
  * shape, dtype, intensity range
  * ground-truth instance count and foreground fraction
  * ImageJ comparator availability and label count

Cross-image checks within each dataset:
  * identical bytes
  * identical shape AND high correlation of a common-size downsample, which
    would indicate the same field or an overlapping crop

Development exposure is recorded explicitly: nuclei_figure_1 and
whole_cell_figure_1 were used in round 1 for parameter exploration, so they are
marked development and are NOT eligible as held-out.

Writes image_manifest.csv into the round directory given by --out.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.segmentation_eval import load_gray_image, load_instance_mask

DATASETS = {
    "nuclei": dict(dirname="nuclear_test_images", prefix="nuclei_figure",
                   image="original_fig.png"),
    "whole_cell": dict(dirname="whole_cell_test_images", prefix="whole_cell_figure",
                       image="original_fig.jpg"),
}
# Images used for parameter exploration in round 1. Permanently development.
DEV_IMAGES = {("nuclei", "nuclei_figure_1"), ("whole_cell", "whole_cell_figure_1")}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def downsample(a: np.ndarray, n: int = 64) -> np.ndarray:
    """Crude common-size downsample for cross-image correlation."""
    h, w = a.shape
    ys = (np.linspace(0, h - 1, n)).astype(int)
    xs = (np.linspace(0, w - 1, n)).astype(int)
    d = a[np.ix_(ys, xs)].astype(np.float64)
    return (d - d.mean()) / (d.std() + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows: List[Dict] = []
    thumbs: Dict[str, np.ndarray] = {}

    for ds, cfg in DATASETS.items():
        base = os.path.join(_ROOT, "data", cfg["dirname"])
        for i in range(1, 12):
            name = f"{cfg['prefix']}_{i}"
            folder = os.path.join(base, name)
            if not os.path.isdir(folder):
                continue
            img_p = os.path.join(folder, cfg["image"])
            gt_p = os.path.join(folder, "original_true_masks.png")
            ij_p = os.path.join(folder, "original_ImageJ_masks.tif")
            if not (os.path.exists(img_p) and os.path.exists(gt_p)):
                rows.append(dict(dataset=ds, image=name, status="incomplete",
                                 has_image=os.path.exists(img_p),
                                 has_gt=os.path.exists(gt_p)))
                continue

            img = load_gray_image(img_p)
            gt = load_instance_mask(gt_p)
            n_gt = int(len(np.unique(gt[gt > 0])))
            ij_ok, n_ij = False, np.nan
            if os.path.exists(ij_p):
                ij = load_instance_mask(ij_p)
                ij_ok = (ij.shape == img.shape)
                n_ij = int(len(np.unique(ij[ij > 0])))

            is_dev = (ds, name) in DEV_IMAGES
            rows.append(dict(
                dataset=ds, image=name, status="ok",
                path_image=os.path.relpath(img_p, _ROOT),
                path_gt=os.path.relpath(gt_p, _ROOT),
                path_imagej=(os.path.relpath(ij_p, _ROOT)
                             if os.path.exists(ij_p) else ""),
                height=img.shape[0], width=img.shape[1],
                square=bool(img.shape[0] == img.shape[1]),
                img_min=float(img.min()), img_max=float(img.max()),
                img_mean=float(img.mean()),
                gt_shape_matches_image=bool(gt.shape == img.shape),
                n_gt_instances=n_gt,
                gt_fg_fraction=float((gt > 0).mean()),
                imagej_available=bool(os.path.exists(ij_p)),
                imagej_shape_matches=ij_ok, n_imagej_instances=n_ij,
                md5_image=md5(img_p), md5_gt=md5(gt_p),
                role=("development" if is_dev else "heldout_candidate"),
                used_for_parameter_selection=bool(is_dev),
            ))
            thumbs[f"{ds}/{name}"] = downsample(img)

    df = pd.DataFrame(rows)

    # ---- cross-image independence checks, within dataset ----
    notes: List[Dict] = []
    for ds in df["dataset"].dropna().unique():
        sub = df[(df["dataset"] == ds) & (df["status"] == "ok")]
        for a, b in itertools.combinations(sub["image"].tolist(), 2):
            ra = sub[sub["image"] == a].iloc[0]
            rb = sub[sub["image"] == b].iloc[0]
            same_bytes = ra["md5_image"] == rb["md5_image"]
            same_shape = (ra["height"], ra["width"]) == (rb["height"], rb["width"])
            corr = float(np.mean(thumbs[f"{ds}/{a}"] * thumbs[f"{ds}/{b}"]))
            flag = ("EXACT_DUPLICATE" if same_bytes
                    else "SUSPECT_same_field" if corr > 0.90
                    else "ok_distinct")
            notes.append(dict(dataset=ds, image_a=a, image_b=b,
                              same_bytes=same_bytes, same_shape=same_shape,
                              thumb_corr=round(corr, 4), verdict=flag))
    nd = pd.DataFrame(notes)

    # demote any held-out candidate that duplicates a development image
    for _, r in nd[nd["verdict"] != "ok_distinct"].iterrows():
        for img in (r["image_a"], r["image_b"]):
            m = (df["dataset"] == r["dataset"]) & (df["image"] == img)
            if df.loc[m, "role"].eq("heldout_candidate").any():
                df.loc[m, "role"] = "excluded_not_independent"
                df.loc[m, "exclusion_reason"] = (
                    f"{r['verdict']} vs {r['image_a']}/{r['image_b']}")

    out_csv = os.path.join(args.out, "image_manifest.csv")
    df.to_csv(out_csv, index=False)
    nd.to_csv(os.path.join(args.out, "image_independence_checks.csv"), index=False)

    print("=" * 104)
    print("IMAGE MANIFEST")
    print("=" * 104)
    ok = df[df["status"] == "ok"]
    hdr = (f"{'dataset':<12}{'image':<22}{'HxW':>12}{'sq':>4}{'range':>12}"
           f"{'#GT':>6}{'fg%':>7}{'gt_al':>7}{'IJ':>4}{'#IJ':>6}  role")
    print(hdr); print("-" * len(hdr))
    for _, r in ok.iterrows():
        print(f"{r['dataset']:<12}{r['image']:<22}"
              f"{str(int(r['height']))+'x'+str(int(r['width'])):>12}"
              f"{'Y' if r['square'] else 'n':>4}"
              f"{f'{r.img_min:.0f}-{r.img_max:.0f}':>12}"
              f"{int(r['n_gt_instances']):>6}{100*r['gt_fg_fraction']:>6.1f}%"
              f"{'Y' if r['gt_shape_matches_image'] else 'N':>7}"
              f"{'Y' if r['imagej_available'] else 'n':>4}"
              f"{('' if np.isnan(r['n_imagej_instances']) else int(r['n_imagej_instances'])):>6}"
              f"  {r['role']}")
    bad = df[df["status"] != "ok"]
    if not bad.empty:
        print("\nincomplete folders:")
        for _, r in bad.iterrows():
            print(f"  {r['dataset']}/{r['image']}: image={r.get('has_image')} "
                  f"gt={r.get('has_gt')}")

    print("\nINDEPENDENCE CHECKS (within dataset)")
    susp = nd[nd["verdict"] != "ok_distinct"]
    if susp.empty:
        print("  no exact duplicates and no pair with downsample corr > 0.90")
        print(f"  max pairwise thumbnail correlation = {nd['thumb_corr'].max():.4f}")
    else:
        for _, r in susp.iterrows():
            print(f"  {r['verdict']}: {r['dataset']} {r['image_a']} vs "
                  f"{r['image_b']} (corr={r['thumb_corr']})")

    print("\nSPLIT")
    for role in ("development", "heldout_candidate", "excluded_not_independent"):
        n = ok[ok["role"] == role]
        if len(n):
            print(f"  {role:<28} n={len(n):>2}  "
                  f"{', '.join(n['dataset']+'/'+n['image'])}")
    print(f"\nWrote {out_csv}")
    print("=" * 104)


if __name__ == "__main__":
    main()
