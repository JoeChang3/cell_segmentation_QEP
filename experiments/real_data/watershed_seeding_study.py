"""
Is watershed marker seeding the real bottleneck in the cell-segmentation pipeline?

BACKGROUND
----------
Both the legacy pipeline (`generate_gp_masks_test`, Modified_Functions_RGasp.py:479)
and the round-1 shared pipeline separate instances with

    watershed(-distance_transform_edt(binary), markers=None, mask=binary)

With `markers=None`, skimage seeds a basin at EVERY local minimum of the
elevation, i.e. every local maximum of the distance transform. A single slightly
irregular nucleus has several such maxima, so one cell is split into several
labels. The legacy code comments acknowledge this is only an approximation of
EBImage::watershed, which does its own internal marker detection.

This script isolates that stage. It fixes everything else and varies only the
seeding, on:

  * the ORACLE foreground (the ground-truth foreground itself), which removes
    reconstruction and thresholding from the picture entirely, so whatever
    remains is attributable to instance separation alone;
  * each round-1 arm's actual predicted binary, which gives the achievable
    end-to-end score.

A large gap between `markers=None` and marker-based seeding on the ORACLE
foreground means the separation stage, not the model, is what caps the pipeline.

Run after real_cellseg_round1.py:
    python experiments/real_data/watershed_seeding_study.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.Modified_Functions_RGasp import eliminate_small_areas
from py_core.segmentation_eval import evaluate_instances

RES = os.path.join(_ROOT, "results", "real_cellseg_round1")
OUT = os.path.join(RES, "tables", "watershed_seeding_study.csv")
ARMS = ["raw", "gp_legacy", "gp", "qep_q2", "qep_q1.5"]
MIN_DISTANCES = [3, 5, 7, 9, 12, 15]
REMOVE_SIZE = 50

# For context: the ImageJ comparator's AP@0.5 on these same two images,
# recomputed from the stored IoU matrices.
IMAGEJ_AP50 = {"nuclei": 0.4586, "whole_cell": 0.3416}


def separate(binary: np.ndarray, min_distance: int | None) -> np.ndarray:
    """Instance separation. min_distance=None reproduces the legacy markers=None."""
    b = binary > 0
    dist = ndi.distance_transform_edt(b)
    if min_distance is None:
        lab = watershed(-dist, markers=None, mask=b)
    else:
        coords = peak_local_max(dist, min_distance=int(min_distance), labels=b)
        markers = np.zeros(dist.shape, dtype=np.int32)
        if len(coords):
            markers[tuple(coords.T)] = 1
        markers, _ = ndi.label(markers)
        lab = watershed(-dist, markers=markers, mask=b)
    return eliminate_small_areas(lab.astype(np.int32), REMOVE_SIZE)


def main() -> None:
    rows: List[Dict] = []
    for ds in ("nuclei", "whole_cell"):
        gt_path = os.path.join(RES, "masks", f"{ds}_ground_truth.npy")
        if not os.path.exists(gt_path):
            print(f"[skip] {ds}: run real_cellseg_round1.py first")
            continue
        gt = np.load(gt_path)
        n_true = int(len(np.unique(gt[gt > 0])))

        print("=" * 100)
        print(f"{ds}: GT instances={n_true}   (ImageJ comparator AP@0.5="
              f"{IMAGEJ_AP50.get(ds, float('nan')):.4f})")
        print("=" * 100)

        sources: Dict[str, np.ndarray] = {"ORACLE_foreground": (gt > 0).astype(np.uint8)}
        for arm in ARMS:
            p = os.path.join(RES, "masks", f"{ds}_{arm}.npz")
            if os.path.exists(p):
                sources[arm] = np.load(p)["binary"]

        for src_name, binary in sources.items():
            print(f"\n  {src_name}")
            base = None
            for md in [None] + MIN_DISTANCES:
                mask = separate(binary, md)
                ev = evaluate_instances(gt, mask)
                tag = "markers=None (legacy)" if md is None else f"peak min_distance={md}"
                if md is None:
                    base = ev.ap(0.5)
                gain = "" if md is None else f"  gain x{ev.ap(0.5)/max(base,1e-9):.1f}"
                print(f"    {tag:<26} AP@0.5={ev.ap(0.5):.4f}  AP@0.75={ev.ap(0.75):.4f}"
                      f"  n_pred={ev.n_pred:>5}  meanIoU={ev.mean_matched_iou:.4f}{gain}")
                rows.append(dict(dataset=ds, source=src_name,
                                 seeding=("markers_none" if md is None
                                          else f"peak_md{md}"),
                                 min_distance=(np.nan if md is None else md),
                                 ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                                 n_pred=ev.n_pred, n_true=n_true,
                                 mean_matched_iou=ev.mean_matched_iou,
                                 imagej_ap50=IMAGEJ_AP50.get(ds, np.nan)))
        print()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    for ds in df["dataset"].unique():
        d = df[df["dataset"] == ds]
        orc = d[d["source"] == "ORACLE_foreground"]
        if orc.empty:
            continue
        legacy = float(orc[orc["seeding"] == "markers_none"]["ap50"].iloc[0])
        best = orc.loc[orc["ap50"].idxmax()]
        arms = d[d["source"] != "ORACLE_foreground"]
        a_leg = arms[arms["seeding"] == "markers_none"]["ap50"].max()
        a_best = arms["ap50"].max()
        a_best_row = arms.loc[arms["ap50"].idxmax()]
        print(f"\n{ds}:")
        print(f"  separation-only ceiling: markers=None {legacy:.4f} -> "
              f"{best['ap50']:.4f} at {best['seeding']} "
              f"(x{best['ap50']/max(legacy,1e-9):.1f})")
        print(f"  best real arm end-to-end: {a_leg:.4f} -> {a_best:.4f} "
              f"({a_best_row['source']}, {a_best_row['seeding']}) "
              f"(x{a_best/max(a_leg,1e-9):.1f})")
        ij = IMAGEJ_AP50.get(ds, np.nan)
        print(f"  ImageJ comparator: {ij:.4f} -> best real arm "
              f"{'EXCEEDS' if a_best > ij else 'still below'} it")
    print(f"\nWrote {OUT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
