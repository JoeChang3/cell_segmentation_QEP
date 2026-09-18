"""
Round 7: is Fast-GP's benefit under the PAPER downstream mediated by watershed
seed stabilization?

                          EBImage implicit seeds      Explicit peak markers
  Raw / no reconstruction           B                          B*
  Paper Fast-GP                     D                          D*

B, D are Round-6 results, reused and re-verified. B*, D* change ONLY seed
generation.

NOTHING IS REFIT. Every reconstruction, RobustGaSP threshold, binary foreground
and EBImage distance map is read from the Round-6 R dumps.

WATERSHED IMPLEMENTATION (Step 4) -- stated plainly, not hidden
  EBImage::watershed(x, tolerance=1, ext=1) has NO seed/marker argument, so it
  CANNOT accept external markers. EBImage::propagate(x, seeds, mask, lambda)
  does take seeds but is Voronoi-like propagation, not watershed flooding.
  Therefore B*/D* use skimage.segmentation.watershed with explicit markers on
  the SAME flooding surface and mask:
      elevation = -(EBImage distmap from R)    [EBImage floods maxima of the
                                                positive map; skimage floods
                                                minima, so the surface is negated
                                                -- same surface, same convention
                                                as our corrected pipeline]
      mask      = the R paper-threshold binary foreground
      cleanup   = the paper's eliminate_small_areas(., 50)
  This is a CONTROLLED SEEDING INTERVENTION, not a fully paper-faithful EBImage
  run. EBImage::propagate with the identical seeds is reported separately as an
  EBImage-native secondary (round7_propagate.R).

Usage: python experiments/real_data/round7_seed_interaction.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from py_core.Modified_Functions_RGasp import eliminate_small_areas
from py_core.foreground_threshold import foreground_metrics
from py_core.instance_separation import (
    PEAK_EXCLUDE_BORDER,
    PEAK_P_NORM,
    WATERSHED_CONNECTIVITY,
    make_peak_markers,
)
from py_core.segmentation_eval import (
    classify_failure_modes,
    evaluate_instances,
    load_instance_mask,
)

R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
R6 = os.path.join(_ROOT, "results", "real_cellseg_round6_paper_downstream_20260917")

# frozen dataset-level marker settings, from round-3 frozen_config.json
MIN_DISTANCE = {"nuclei": 15, "whole_cell": 9}
REMOVE_SIZE = 50                       # the paper's remove_size_threshold
R_TAG = {"B": "B_nogp_canonical", "D": "D_paper_fastgp"}
CELL_LABEL = {
    "B": "B   Raw + paper downstream, EBImage implicit seeds",
    "D": "D   FastGP + paper downstream, EBImage implicit seeds",
    "B*": "B*  Raw + paper downstream, EXPLICIT peak markers",
    "D*": "D*  FastGP + paper downstream, EXPLICIT peak markers",
}


def mat(p: str) -> np.ndarray:
    return np.loadtxt(p, delimiter=",")


def score(gt, labels, binary) -> Dict:
    labels = labels.astype(np.int32)
    ev = evaluate_instances(gt, labels)
    fm = classify_failure_modes(gt, labels)
    fg = foreground_metrics(binary > 0, gt > 0)
    return dict(ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                tp50=ev.per_threshold[0.5]["tp"], fp50=ev.per_threshold[0.5]["fp"],
                fn50=ev.per_threshold[0.5]["fn"],
                tp75=ev.per_threshold[0.75]["tp"], fp75=ev.per_threshold[0.75]["fp"],
                fn75=ev.per_threshold[0.75]["fn"],
                n_pred=ev.n_pred, n_gt=ev.n_true,
                mean_matched_iou=ev.mean_matched_iou,
                merged=fm["merged"], split=fm["split"], missed=fm["missed"],
                spurious=fm["spurious"], **fg)


def marker_diagnostics(ds: str, cell: str, gt: np.ndarray, fg: np.ndarray,
                       marker_img: np.ndarray, coords: np.ndarray,
                       n_comp: int, n_rescued: int) -> Dict:
    """Step 8: marker COUNT vs marker PLACEMENT.

    GT is used here ONLY to describe markers that were already generated without
    it. Nothing below feeds back into marker generation or parameter choice.
    """
    comps, _ = ndi.label(fg, structure=ndi.generate_binary_structure(2, 1))
    per_comp = np.zeros(n_comp + 1, dtype=int)
    if marker_img is not None and (marker_img > 0).any():
        cl = comps[marker_img > 0]
        for c in cl:
            per_comp[c] += 1
    pc = per_comp[1:]

    gt_ids = np.unique(gt[gt > 0])
    if marker_img is not None and (marker_img > 0).any():
        lbl_at_marker = gt[marker_img > 0]
    else:
        lbl_at_marker = np.empty(0, dtype=gt.dtype)
    cnt = pd.Series(lbl_at_marker[lbl_at_marker > 0]).value_counts()
    per_gt = np.array([int(cnt.get(g, 0)) for g in gt_ids])

    return dict(
        dataset=ds, cell=cell, label=CELL_LABEL[cell],
        n_markers=int(len(coords)) if coords is not None else 0,
        n_fg_components=int(n_comp), n_components_rescued=int(n_rescued),
        markers_per_component_mean=float(pc.mean()) if n_comp else float("nan"),
        markers_per_component_median=float(np.median(pc)) if n_comp else float("nan"),
        markers_per_component_max=int(pc.max()) if n_comp else 0,
        components_with_zero_markers=int((pc == 0).sum()),
        components_with_gt1_markers=int((pc > 1).sum()),
        n_gt=int(len(gt_ids)),
        frac_gt_with_ge1_marker=float((per_gt >= 1).mean()),
        mean_markers_per_gt_cell=float(per_gt.mean()),
        gt_cells_zero_markers=int((per_gt == 0).sum()),
        gt_cells_gt1_markers=int((per_gt > 1).sum()),
        markers_outside_gt=int(int(len(lbl_at_marker)) - int((lbl_at_marker > 0).sum())),
    )


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = args.out
    for s in ("figures", "markers", "instance_masks", "logs", "tables"):
        os.makedirs(os.path.join(out, s), exist_ok=True)

    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[(man["status"] == "ok") & (man["role"] == "development")]
    r6 = pd.read_csv(os.path.join(R6, "two_by_two_metrics.csv"))

    rows: List[Dict] = []
    diag_rows: List[Dict] = []
    basin_rows: List[Dict] = []

    print("=" * 128)
    print("ROUND 7  SEED-INTERACTION EXPERIMENT   (nothing refit; Round-6 arrays reused)")
    print("  EBImage::watershed has NO marker argument -> B*/D* use skimage watershed")
    print("  on the SAME surface: elevation = -(EBImage distmap), mask = paper binary,")
    print("  cleanup = paper eliminate_small_areas(.,50).  Labelled a seeding INTERVENTION.")
    print("=" * 128)

    for _, im in man.iterrows():
        ds, name = im["dataset"], im["image"]
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        rdir = os.path.join(R6, "r_outputs", ds)
        md = MIN_DISTANCE[ds]
        print(f"\n### {ds}/{name}   GT={int(len(np.unique(gt[gt>0])))}   "
              f"min_distance={md}")

        for cell in ("B", "D"):
            tag = R_TAG[cell]
            binary = mat(os.path.join(rdir, f"{tag}_combined_binary.csv")) > 0
            dist_r = mat(os.path.join(rdir, f"{tag}_distmap.csv"))
            pre = mat(os.path.join(rdir, f"{tag}_watershed_precleanup.csv"))
            fin = mat(os.path.join(rdir, f"{tag}_labels_final.csv"))
            assert binary.shape == gt.shape == dist_r.shape

            # ---- Step 1 verification: reproduce Round-6 -------------------
            m = score(gt, fin, binary)
            ref = r6[(r6.dataset == ds) & (r6.cell == cell)].iloc[0]
            assert abs(m["ap50"] - float(ref.ap50)) < 1e-12, (
                f"{cell} AP@0.5 {m['ap50']} != round6 {ref.ap50}")
            assert abs(m["ap75"] - float(ref.ap75)) < 1e-12
            assert abs(m["fg_dice"] - float(ref.fg_dice)) < 1e-12

            # ---- validate our python cleanup port against R's output -----
            py_clean = eliminate_small_areas(pre.astype(np.int32), REMOVE_SIZE)
            same_part = bool(np.array_equal(py_clean > 0, fin > 0))
            n_py = int(len(np.unique(py_clean[py_clean > 0])))
            n_r = int(len(np.unique(fin[fin > 0])))

            n_comp_b = int(ndi.label(binary,
                                     structure=ndi.generate_binary_structure(2, 1))[1])
            n_pre = int(len(np.unique(pre[pre > 0])))
            basin_rows.append(dict(
                dataset=ds, cell=cell, label=CELL_LABEL[cell],
                seeding="EBImage implicit (tolerance=1, ext=1)",
                n_fg_components=n_comp_b, n_markers=np.nan,
                n_basins_precleanup=n_pre, n_instances_final=n_r,
                python_cleanup_matches_R=same_part, n_labels_python_cleanup=n_py))
            print(f"  {CELL_LABEL[cell]:<52} AP@0.5={m['ap50']:.4f} "
                  f"[round-6 verified]  basins_pre={n_pre} final={n_r}  "
                  f"py-cleanup==R: {same_part}")

            rows.append(dict(dataset=ds, image=name, cell=cell,
                             label=CELL_LABEL[cell],
                             reconstruction=("raw" if cell == "B"
                                             else "paper_fast_gp"),
                             seeding="EBImage implicit (tolerance=1, ext=1)",
                             watershed_impl="EBImage::watershed (real R)",
                             source="round6 reused (verified identical)", **m))
            diag_rows.append(marker_diagnostics(
                ds, cell, gt, binary, None, np.empty((0, 2), int), n_comp_b, 0))

            # ---- Step 2 + 4 : explicit markers on the SAME R distmap -----
            star = cell + "*"
            t0 = time.time()
            marker_img, coords, n_comp, n_rescued = make_peak_markers(
                dist_r, binary, min_distance=md, rescue_empty_components=True)
            raw_lab = watershed(-dist_r, markers=marker_img, mask=binary,
                                connectivity=WATERSHED_CONNECTIVITY).astype(np.int32)
            fin_star = eliminate_small_areas(raw_lab, REMOVE_SIZE)
            dt = time.time() - t0

            ms = score(gt, fin_star, binary)
            # the binary foreground must be byte-identical between cell and cell*
            assert np.array_equal(binary, mat(os.path.join(
                rdir, f"{tag}_combined_binary.csv")) > 0)
            assert abs(ms["fg_dice"] - m["fg_dice"]) < 1e-12, (
                "foreground changed between implicit and explicit seeding")

            n_pre_s = int(len(np.unique(raw_lab[raw_lab > 0])))
            n_fin_s = int(len(np.unique(fin_star[fin_star > 0])))
            basin_rows.append(dict(
                dataset=ds, cell=star, label=CELL_LABEL[star],
                seeding=f"explicit peak_local_max min_distance={md}",
                n_fg_components=n_comp, n_markers=int(len(coords)),
                n_basins_precleanup=n_pre_s, n_instances_final=n_fin_s,
                python_cleanup_matches_R=np.nan,
                n_labels_python_cleanup=n_fin_s))
            rows.append(dict(dataset=ds, image=name, cell=star,
                             label=CELL_LABEL[star],
                             reconstruction=("raw" if cell == "B"
                                             else "paper_fast_gp"),
                             seeding=f"explicit peak_local_max min_distance={md}",
                             watershed_impl="skimage watershed on -(EBImage distmap) "
                                            "[controlled seeding intervention]",
                             source="computed round7", runtime_s=dt, **ms))
            diag_rows.append(marker_diagnostics(
                ds, star, gt, binary, marker_img, coords, n_comp, n_rescued))
            print(f"  {CELL_LABEL[star]:<52} AP@0.5={ms['ap50']:.4f}  "
                  f"markers={len(coords)} rescued={n_rescued}  "
                  f"basins_pre={n_pre_s} final={n_fin_s}  ({dt:.1f}s)")

            np.savez_compressed(
                os.path.join(out, "markers", f"{ds}_{name}_{cell}star.npz"),
                marker_image=marker_img, marker_coords=coords,
                dist_map_ebimage=dist_r.astype(np.float32),
                binary=binary.astype(np.uint8), n_markers=np.int64(len(coords)),
                n_fg_components=np.int64(n_comp),
                n_components_rescued=np.int64(n_rescued))
            np.savez_compressed(
                os.path.join(out, "instance_masks", f"{ds}_{name}_{cell}star.npz"),
                instance_mask=fin_star, labels_precleanup=raw_lab)
            np.savez_compressed(
                os.path.join(out, "instance_masks", f"{ds}_{name}_{cell}.npz"),
                instance_mask=fin.astype(np.int32),
                labels_precleanup=pre.astype(np.int32))
            # markers as CSV for the EBImage::propagate secondary
            np.savetxt(os.path.join(out, "markers", f"{ds}_{cell}star_markers.csv"),
                       marker_img, delimiter=",", fmt="%d")

        # ---- figures ---------------------------------------------------
        cells = ["B", "B*", "D", "D*"]
        fig, ax = plt.subplots(2, 4, figsize=(23, 11))
        for k, c in enumerate(cells):
            tag = R_TAG[c[0]]
            binary = mat(os.path.join(rdir, f"{tag}_combined_binary.csv")) > 0
            dist_r = mat(os.path.join(rdir, f"{tag}_distmap.csv"))
            if c.endswith("*"):
                z = np.load(os.path.join(out, "instance_masks",
                                         f"{ds}_{name}_{c[0]}star.npz"))
                mk = np.load(os.path.join(out, "markers",
                                          f"{ds}_{name}_{c[0]}star.npz"))
                lab, pre_l = z["instance_mask"], z["labels_precleanup"]
                coords = mk["marker_coords"]
            else:
                lab = mat(os.path.join(rdir, f"{tag}_labels_final.csv"))
                pre_l = mat(os.path.join(rdir, f"{tag}_watershed_precleanup.csv"))
                coords = np.empty((0, 2), int)
            a = ax[0, k]
            a.imshow(dist_r, cmap="magma")
            if len(coords):
                a.plot(coords[:, 1], coords[:, 0], "c.", ms=1.1)
            a.set_title(f"{c}: EBImage distmap"
                        + (f" + {len(coords)} markers" if len(coords) else
                           " (implicit seeds)"), fontsize=9)
            a.axis("off")
            a = ax[1, k]
            a.imshow(np.zeros_like(dist_r), cmap="gray")
            a.contour(gt > 0, levels=[0.5], colors="lime", linewidths=0.35)
            a.imshow(np.ma.masked_where(~find_boundaries(lab.astype(np.int32),
                                                         mode="outer"),
                                        np.ones_like(lab)),
                     cmap="autumn", alpha=0.95)
            r = [x for x in rows if x["dataset"] == ds and x["cell"] == c][0]
            a.set_title(f"{c}: AP@0.5={r['ap50']:.3f}  basins_pre="
                        f"{int(len(np.unique(pre_l[pre_l>0])))}  n={r['n_pred']}",
                        fontsize=9)
            a.axis("off")
        fig.suptitle(f"{ds}: top = flooding surface + seeds, bottom = green GT / "
                     f"red predicted boundaries", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"{ds}_seed_interaction.png"),
                    dpi=115)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "seed_interaction_metrics.csv"), index=False)
    pd.DataFrame(diag_rows).to_csv(os.path.join(out, "marker_diagnostics.csv"),
                                   index=False)
    pd.DataFrame(basin_rows).to_csv(
        os.path.join(out, "tables", "basin_counts.csv"), index=False)
    df[["dataset", "cell", "label", "merged", "split", "missed", "spurious",
        "tp50", "fp50", "fn50", "n_pred", "n_gt"]].to_csv(
        os.path.join(out, "instance_failure_counts.csv"), index=False)

    # ---- Step 7 : the mechanism test -------------------------------------
    dec: List[Dict] = []
    print("\n" + "=" * 128)
    print("STEP 7  MECHANISM TEST   Delta_implicit = D-B   Delta_explicit = D*-B*")
    print("        seed-mediated interaction = Delta_implicit - Delta_explicit")
    print("        (n=1 image per dataset; descriptive only)")
    print("=" * 128)
    for ds in sorted(df["dataset"].unique()):
        g = {c: df[(df.dataset == ds) & (df.cell == c)].iloc[0]
             for c in ("B", "D", "B*", "D*")}
        print(f"\n  {ds}")
        for met, hi_is_good in (("ap50", True), ("ap75", True), ("fg_dice", True),
                                ("merged", False), ("split", False)):
            B, D, Bs, Ds = (float(g[c][met]) for c in ("B", "D", "B*", "D*"))
            di, de = D - B, Ds - Bs
            frac = (1.0 - de / di) if abs(di) > 1e-12 else float("nan")
            dec.append(dict(dataset=ds, metric=met, B=B, D=D, B_star=Bs, D_star=Ds,
                            delta_implicit_DmB=di, delta_explicit_DsmBs=de,
                            seed_mediated_interaction=di - de,
                            frac_of_delta_removed_by_explicit_seeds=frac,
                            explicit_effect_raw_BsmB=Bs - B,
                            explicit_effect_fastgp_DsmD=Ds - D,
                            higher_is_better=hi_is_good))
            print(f"    {met:<8} B={B:>8.4f} D={D:>8.4f} B*={Bs:>8.4f} D*={Ds:>8.4f} | "
                  f"D-B={di:+.4f}  D*-B*={de:+.4f}  interaction={di-de:+.4f}"
                  + (f"  ({100*frac:.1f}% of D-B removed)"
                     if np.isfinite(frac) else ""))
    pd.DataFrame(dec).to_csv(
        os.path.join(out, "tables", "mechanism_decomposition.csv"), index=False)

    json.dump(dict(
        round="round7-seed-interaction",
        question="Is Fast-GP's benefit under the paper downstream mediated by "
                 "watershed seed stabilization?",
        nothing_refit=True,
        reused_from=os.path.relpath(R6, _ROOT),
        reused_artifacts=["*_combined_binary.csv (paper RobustGaSP threshold)",
                          "*_distmap.csv (EBImage distmap)",
                          "*_watershed_precleanup.csv", "*_labels_final.csv",
                          "GT_paper_orientation.csv"],
        held_fixed=dict(
            reconstruction="Round-6 arrays, byte-identical between cell and cell*",
            threshold="paper criterion_1 + real RobustGaSP, from Round 6; NOT "
                      "replaced by Li/Otsu/Round-3/anything",
            binary_foreground="identical between cell and cell*, asserted",
            distance_transform="EBImage::distmap output from R, reused verbatim",
            cleanup="paper eliminate_small_areas(., size_threshold=50)",
            orientation="Round-6 verified natural (height,width); no flips",
            evaluator="py_core/segmentation_eval.py, AP=TP/(TP+FP+FN)"),
        markers=dict(
            implementation="skimage.feature.peak_local_max via "
                           "py_core.instance_separation.make_peak_markers",
            source_surface="EBImage distmap from R (NOT recomputed)",
            min_distance=MIN_DISTANCE,
            min_distance_units=f"pixels, p_norm={PEAK_P_NORM} (Chebyshev)",
            threshold_abs=None, threshold_rel=None,
            exclude_border=PEAK_EXCLUDE_BORDER,
            mask_restriction="labels = paper binary foreground",
            fallback_for_markerless_components="one marker at the component's "
                "distance-transform argmax (first in C raster order)",
            connectivity=WATERSHED_CONNECTIVITY,
            label_construction="each peak gets its own integer id; peaks are "
                               "never fused by ndi.label",
            gt_used_for_markers=False,
            tuned_per_arm=False,
            settings_origin="round-3 frozen_config.json (dataset-level, frozen "
                            "before this experiment)"),
        watershed_step4=dict(
            ebimage_accepts_markers=False,
            ebimage_watershed_signature="watershed(x, tolerance=1, ext=1)",
            ebimage_propagate_signature="propagate(x, seeds, mask=NULL, lambda=1e-04)",
            decision="EBImage::watershed cannot take seeds; B*/D* use skimage "
                     "watershed on -(EBImage distmap) with the same mask. "
                     "CONTROLLED SEEDING INTERVENTION, not a paper-faithful "
                     "EBImage run. EBImage::propagate with identical seeds is "
                     "reported separately."),
        environment=dict(python=platform.python_version(),
                         platform=platform.platform(), numpy=np.__version__),
    ), open(os.path.join(out, "config.json"), "w"), indent=2)

    # ---- Step 8 print -----------------------------------------------------
    dd = pd.DataFrame(diag_rows)
    print("\n" + "=" * 128)
    print("STEP 8  MARKER COUNT vs MARKER PLACEMENT  (GT used only to DESCRIBE "
          "markers already generated without it)")
    print("=" * 128)
    cols = ["n_markers", "n_fg_components", "markers_per_component_mean",
            "components_with_gt1_markers", "frac_gt_with_ge1_marker",
            "mean_markers_per_gt_cell", "gt_cells_zero_markers",
            "gt_cells_gt1_markers"]
    print(f"  {'dataset':<12}{'cell':<5}{'markers':>9}{'fgComp':>8}{'mk/comp':>9}"
          f"{'comp>1mk':>10}{'%GT>=1mk':>10}{'mk/GTcell':>11}"
          f"{'GT 0mk':>8}{'GT>1mk':>8}")
    for _, r in dd.iterrows():
        print(f"  {r.dataset:<12}{r.cell:<5}"
              f"{('-' if not np.isfinite(r.n_markers) or r.n_markers==0 and not r.cell.endswith('*') else int(r.n_markers)):>9}"
              f"{int(r.n_fg_components):>8}{r.markers_per_component_mean:>9.3f}"
              f"{int(r.components_with_gt1_markers):>10}"
              f"{100*r.frac_gt_with_ge1_marker:>10.1f}"
              f"{r.mean_markers_per_gt_cell:>11.3f}"
              f"{int(r.gt_cells_zero_markers):>8}{int(r.gt_cells_gt1_markers):>8}")

    print("\n" + "=" * 128)
    print("BASIN COUNTS  (Step 5 verification, reproduced from saved Round-6 data)")
    print("=" * 128)
    bb = pd.DataFrame(basin_rows)
    print(f"  {'dataset':<12}{'cell':<5}{'fgComp':>8}{'markers':>9}"
          f"{'basins pre-cleanup':>20}{'final':>8}")
    for _, r in bb.iterrows():
        print(f"  {r.dataset:<12}{r.cell:<5}{int(r.n_fg_components):>8}"
              f"{('-' if not np.isfinite(r.n_markers) else int(r.n_markers)):>9}"
              f"{int(r.n_basins_precleanup):>20}{int(r.n_instances_final):>8}")
    print(f"\nWrote outputs into {out}")


if __name__ == "__main__":
    main()
