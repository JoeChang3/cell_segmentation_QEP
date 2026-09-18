"""
Round 3, Step 6: sanity controls on the threshold rules.

A. Scale invariance      I' = a*I,  a in {0.5, 2.0, 10.0}
B. Affine invariance     I' = a*I + b, a > 0
C. Determinism           same array twice -> same threshold, mask, segmentation

A rule is called INVARIANT when the resulting binary foreground is pixelwise
identical after the transform. The threshold value itself is expected to move
with the data; only the selected pixel SET must be stable.

Expectations to be confirmed or refuted, not assumed:
  criterion_1  cuts at fraction*max, so pure scaling should be harmless but an
               additive offset should not be, because min/max shift differently.
  quantile     depends only on rank order, so it should survive any strictly
               increasing transform.
  robust_mad   median and MAD are both affine-equivariant, so it should survive
               any positive affine map.
  otsu/li/yen/triangle  operate on the histogram and are generally equivariant
               under affine maps up to binning effects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.foreground_threshold import apply_rule, candidate_grid
from py_core.instance_separation import partitions_equivalent, separate_instances

R2 = os.path.join(_ROOT, "results", "real_cellseg_round2_20260915")
MIN_DISTANCE = {"nuclei": 15, "whole_cell": 9}
REPRESENTATIVE = [
    ("nuclei", "nuclei_figure_2", "gp"),
    ("nuclei", "nuclei_figure_4", "qep_q1.5"),
    ("whole_cell", "whole_cell_figure_3", "gp"),
    ("whole_cell", "whole_cell_figure_5", "raw"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.out, "selected_threshold_config.json")))

    # rules to test: the full candidate set, deduplicated by family+params
    cands = candidate_grid()
    rows: List[Dict] = []

    print("=" * 112)
    print("STEP 6A/B  SCALE AND AFFINE INVARIANCE")
    print("  invariant := binary foreground pixelwise identical after transform")
    print("=" * 112)
    transforms = [("scale_0.5", 0.5, 0.0), ("scale_2.0", 2.0, 0.0),
                  ("scale_10.0", 10.0, 0.0), ("affine_a2_b50", 2.0, 50.0),
                  ("affine_a1_b100", 1.0, 100.0)]
    for cand in cands:
        inv_counts = {t[0]: 0 for t in transforms}
        total = 0
        for ds, name, arm in REPRESENTATIVE:
            p = os.path.join(R2, "masks", f"{ds}_{name}_{arm}.npz")
            if not os.path.exists(p):
                continue
            img = np.load(p)["predmean"].astype(np.float64)
            try:
                base = apply_rule(img, cand["rule"], cand["params"])
            except Exception:  # noqa: BLE001
                continue
            total += 1
            for tname, a, b in transforms:
                try:
                    tr = apply_rule(a * img + b, cand["rule"], cand["params"])
                except Exception:  # noqa: BLE001
                    continue
                same = bool(np.array_equal(base.mask, tr.mask))
                inv_counts[tname] += int(same)
                rows.append(dict(rule=cand["name"], dataset=ds, image=name,
                                 arm=arm, transform=tname, a=a, b=b,
                                 mask_identical=same,
                                 base_threshold=base.threshold,
                                 transformed_threshold=tr.threshold,
                                 expected_threshold=(a * base.threshold + b),
                                 threshold_matches_expected=bool(
                                     abs(tr.threshold - (a * base.threshold + b))
                                     < 1e-6 * max(1.0, abs(a * base.threshold + b)))))
        if total:
            cells = "  ".join(f"{t[0]}:{inv_counts[t[0]]}/{total}"
                              for t in transforms)
            print(f"  {cand['name']:<20} {cells}")

    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(args.out, "tables", "threshold_invariance.csv"),
               index=False)
    print("\n  summary by rule (fraction of cases with identical foreground):")
    if not inv.empty:
        g = inv.groupby(["rule", "transform"])["mask_identical"].mean().unstack()
        for r in g.index:
            scale_ok = all(g.loc[r, c] == 1.0 for c in g.columns
                           if c.startswith("scale"))
            aff_ok = all(g.loc[r, c] == 1.0 for c in g.columns
                         if c.startswith("affine"))
            print(f"    {r:<20} scale-invariant={scale_ok!s:<5} "
                  f"affine-invariant={aff_ok}")

    # ── C determinism ──
    print("\n" + "=" * 112)
    print("STEP 6C  DETERMINISM: same array twice, threshold + mask + segmentation")
    print("=" * 112)
    drows: List[Dict] = []
    for ds, name, arm in REPRESENTATIVE:
        p = os.path.join(R2, "masks", f"{ds}_{name}_{arm}.npz")
        if not os.path.exists(p):
            continue
        img = np.load(p)["predmean"].astype(np.float64)
        rule, params = cfg["selected"][ds]["rule"], cfg["selected"][ds]["params"]
        a1 = apply_rule(img, rule, params); a2 = apply_rule(img, rule, params)
        s1 = separate_instances(a1.mask.astype(np.uint8), marker_mode="peak",
                                min_distance=MIN_DISTANCE[ds],
                                remove_size_threshold=50)
        s2 = separate_instances(a2.mask.astype(np.uint8), marker_mode="peak",
                                min_distance=MIN_DISTANCE[ds],
                                remove_size_threshold=50)
        same_thr = a1.threshold == a2.threshold
        same_mask = bool(np.array_equal(a1.mask, a2.mask))
        same_part = partitions_equivalent(s1.instance_mask, s2.instance_mask)
        print(f"  {ds}/{name}/{arm}: threshold identical={same_thr}  "
              f"mask identical={same_mask}  partition equivalent={same_part}")
        drows.append(dict(dataset=ds, image=name, arm=arm,
                          rule=cfg["selected"][ds]["name"],
                          threshold_identical=same_thr,
                          mask_identical=same_mask,
                          partition_equivalent=same_part))
    pd.DataFrame(drows).to_csv(
        os.path.join(args.out, "tables", "threshold_determinism.csv"), index=False)
    print(f"\nWrote tables/threshold_invariance.csv, tables/threshold_determinism.csv")


if __name__ == "__main__":
    main()
