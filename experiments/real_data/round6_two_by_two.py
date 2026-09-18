"""
Round 6: the 2 x 2 reconstruction-x-downstream experiment.

                            Current corrected downstream   Paper downstream (real R)
    Raw / NoGP                        A                            B
    Paper Fast-GP                     C                            D

A, C  reused from Round 5 (identical config; nothing recomputed).
B, D  produced by audits/round6_paper_downstream.R using REAL RobustGaSP and
      REAL EBImage::watershed, then scored here with the SAME Python evaluator.

B is a CONSTRUCTED ABLATION: the paper downstream with the image Fast-GP removed
and nothing else changed. The paper's own published NoGP arm
(`generate_GP_Masks_test2`) differs from its GP arm in THREE ways at once, so it
cannot serve as the 2 x 2 control; it is scored separately as `B_literal`.

Steps 6 (orientation alignment + evaluator sanity checks), 7 (run), 9 (metrics),
10 (interaction decomposition).

Usage: python experiments/real_data/round6_two_by_two.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from py_core.foreground_threshold import foreground_metrics
from py_core.segmentation_eval import (
    classify_failure_modes,
    evaluate_instances,
    load_instance_mask,
)

R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
R5 = os.path.join(_ROOT, "results", "real_cellseg_round5_corrected_baseline_20260917")
R_REPO = "/Users/zchan/eclipse-workspace/cell_segmentation_original"
R_SHA = "44714c2e0be958fe796a8fd4bdbc220dae3c23dd"

DEV = {"nuclei": "nuclei_figure_1", "whole_cell": "whole_cell_figure_1"}
# Round-5 arm keys -> 2x2 cell
R5_MAP = {"A": "raw", "C": "paper_fast_gp"}
R_MAP = {"B": "B_nogp_canonical", "D": "D_paper_fastgp"}
CELL_LABEL = {
    "A": "A  Raw + current downstream",
    "B": "B  NoGP + paper downstream (constructed ablation)",
    "C": "C  paper_fast_gp + current downstream",
    "D": "D  paper_fast_gp + paper downstream",
    "B_literal": "B' published NoGP arm (confounded; secondary)",
}


def load_csv_mat(p: str) -> np.ndarray:
    return np.loadtxt(p, delimiter=",")


def evaluator_sanity_checks(gt: np.ndarray) -> List[Dict]:
    """Step 6: assert the evaluator behaves before trusting any real number."""
    print("=" * 124)
    print("STEP 6a  EVALUATOR SANITY CHECKS")
    print("=" * 124)
    rows = []

    ev = evaluate_instances(gt, gt.copy())
    ok = abs(ev.ap(0.5) - 1.0) < 1e-12 and abs(ev.ap(0.75) - 1.0) < 1e-12
    rows.append(dict(check="GT vs itself", expect="AP=1.0 at every tau",
                     got=f"AP@0.5={ev.ap(0.5):.12f} AP@0.75={ev.ap(0.75):.12f}",
                     passed=bool(ok)))
    print(f"  GT vs itself                 AP@0.5={ev.ap(0.5):.12f} "
          f"AP@0.75={ev.ap(0.75):.12f}  n_pred={ev.n_pred} n_gt={ev.n_true} "
          f"-> {'PASS' if ok else 'FAIL'}")

    # two disjoint squares, predicted exactly
    t = np.zeros((40, 40), int); t[2:10, 2:10] = 1; t[20:28, 20:28] = 2
    ev2 = evaluate_instances(t, t.copy())
    ok2 = abs(ev2.ap(0.5) - 1.0) < 1e-12
    rows.append(dict(check="2 exact squares", expect="AP=1.0",
                     got=f"AP@0.5={ev2.ap(0.5):.6f}", passed=bool(ok2)))
    print(f"  2 exact squares              AP@0.5={ev2.ap(0.5):.6f} "
          f"-> {'PASS' if ok2 else 'FAIL'}")

    # one square predicted, one missed  => TP=1 FP=0 FN=1 => AP = 1/2
    p = np.zeros((40, 40), int); p[2:10, 2:10] = 1
    ev3 = evaluate_instances(t, p)
    ok3 = abs(ev3.ap(0.5) - 0.5) < 1e-12
    rows.append(dict(check="1 of 2 detected", expect="AP=0.5 (TP1 FP0 FN1)",
                     got=f"AP@0.5={ev3.ap(0.5):.6f} tp={ev3.per_threshold[0.5]['tp']} "
                         f"fp={ev3.per_threshold[0.5]['fp']} fn={ev3.per_threshold[0.5]['fn']}",
                     passed=bool(ok3)))
    print(f"  1 of 2 detected              AP@0.5={ev3.ap(0.5):.6f} "
          f"(tp={ev3.per_threshold[0.5]['tp']} fp={ev3.per_threshold[0.5]['fp']} "
          f"fn={ev3.per_threshold[0.5]['fn']}) -> {'PASS' if ok3 else 'FAIL'}")

    # merged prediction: one blob covering both GT squares
    m = np.zeros((40, 40), int); m[2:28, 2:28] = 1
    fm = classify_failure_modes(t, m)
    ok4 = fm["merged"] >= 1
    rows.append(dict(check="merge detector", expect="merged>=1",
                     got=str({k: fm[k] for k in ('merged', 'split', 'missed', 'spurious')}),
                     passed=bool(ok4)))
    print(f"  merge detector               {dict((k, fm[k]) for k in ('merged','split','missed','spurious'))}"
          f" -> {'PASS' if ok4 else 'FAIL'}")

    fgm = foreground_metrics(gt > 0, gt > 0)
    ok5 = abs(fgm["fg_dice"] - 1.0) < 1e-12 and abs(fgm["fg_iou"] - 1.0) < 1e-12
    rows.append(dict(check="foreground self-Dice", expect="Dice=IoU=1",
                     got=f"Dice={fgm['fg_dice']:.12f} IoU={fgm['fg_iou']:.12f}",
                     passed=bool(ok5)))
    print(f"  foreground self-Dice         Dice={fgm['fg_dice']:.12f} "
          f"IoU={fgm['fg_iou']:.12f} -> {'PASS' if ok5 else 'FAIL'}")

    assert all(r["passed"] for r in rows), "evaluator sanity checks FAILED"
    return rows


def orientation_check(ds: str, gt: np.ndarray, r_gt: np.ndarray,
                      r_lab: np.ndarray) -> Dict:
    """Step 6: prove the R arrays are in the SAME frame as the Python GT.

    No mask is flipped because a picture 'looks right'. The R ground truth is
    regenerated by the driver's own code path and compared element-wise to the
    Python-loaded ground truth; then the candidate orientations of an R LABEL
    array are scored by foreground IoU against the Python GT and the winner must
    be the identity.
    """
    print(f"\n  [{ds}] orientation")
    print(f"    python GT {gt.shape}   R-driver GT {r_gt.shape}")
    assert r_gt.shape == gt.shape, f"R GT shape {r_gt.shape} != python {gt.shape}"

    # element-wise agreement of the two GT label images (labels may be permuted,
    # so compare the FOREGROUND masks exactly and the partition via IoU)
    same_fg = bool(np.array_equal(r_gt > 0, gt > 0))
    n_r = int(len(np.unique(r_gt[r_gt > 0])))
    n_p = int(len(np.unique(gt[gt > 0])))
    print(f"    GT foreground identical element-wise: {same_fg}   "
          f"#labels R={n_r} python={n_p}")

    cands = {"identity": r_lab,
             "flipud": np.flipud(r_lab),
             "fliplr": np.fliplr(r_lab),
             "rot180": np.flipud(np.fliplr(r_lab))}
    gfg = gt > 0
    scores = {}
    for k, v in cands.items():
        if v.shape != gt.shape:
            continue
        m = v > 0
        scores[k] = float(np.logical_and(m, gfg).sum() /
                          max(np.logical_or(m, gfg).sum(), 1))
    best = max(scores, key=scores.get)
    print(f"    foreground IoU of R labels vs python GT: "
          + "  ".join(f"{k}={s:.4f}" for k, s in scores.items()))
    print(f"    best = {best}  -> {'PASS (identity)' if best == 'identity' else 'MISMATCH'}")
    assert r_lab.shape == gt.shape, "R label shape mismatch"
    assert np.issubdtype(r_lab.dtype, np.floating) or \
        np.issubdtype(r_lab.dtype, np.integer), "R labels not numeric"
    assert best == "identity", (
        f"R labels are not in the python frame; best orientation was {best}")
    return dict(dataset=ds, gt_shape=str(gt.shape), r_gt_shape=str(r_gt.shape),
                gt_foreground_identical=same_fg, n_labels_r_gt=n_r,
                n_labels_python_gt=n_p, best_orientation=best, **
                {f"iou_{k}": v for k, v in scores.items()})


def score(gt: np.ndarray, labels: np.ndarray, binary: np.ndarray) -> Dict:
    labels = labels.astype(np.int32)
    ev = evaluate_instances(gt, labels)
    fm = classify_failure_modes(gt, labels)
    fg = foreground_metrics(binary > 0, gt > 0)
    return dict(
        ap50=ev.ap(0.5), ap75=ev.ap(0.75),
        tp50=ev.per_threshold[0.5]["tp"], fp50=ev.per_threshold[0.5]["fp"],
        fn50=ev.per_threshold[0.5]["fn"],
        tp75=ev.per_threshold[0.75]["tp"], fp75=ev.per_threshold[0.75]["fp"],
        fn75=ev.per_threshold[0.75]["fn"],
        n_pred=ev.n_pred, n_gt=ev.n_true, mean_matched_iou=ev.mean_matched_iou,
        merged=fm["merged"], split=fm["split"], missed=fm["missed"],
        spurious=fm["spurious"], **fg)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out
    for s in ("threshold_diagnostics", "binary_masks", "watershed_precleanup",
              "instance_masks", "overlays", "tables", "logs"):
        os.makedirs(os.path.join(out, s), exist_ok=True)

    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[(man["status"] == "ok") & (man["role"] == "development")]
    r5seg = pd.read_csv(os.path.join(R5, "development_segmentation_metrics.csv"))

    sanity = None
    orient_rows: List[Dict] = []
    rows: List[Dict] = []
    rt_rows: List[Dict] = []

    for _, im in man.iterrows():
        ds, name = im["dataset"], im["image"]
        assert name == DEV[ds], f"unexpected image {name} for {ds}"
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        raw = imageio.imread(os.path.join(_ROOT, im["path_image"]))
        raw = (raw[..., 0] if raw.ndim == 3 else raw).astype(np.float64)
        rdir = os.path.join(out, "r_outputs", ds)

        if sanity is None:
            sanity = evaluator_sanity_checks(gt)
            print("\n" + "=" * 124)
            print("STEP 6b  ORIENTATION ALIGNMENT OF R OUTPUTS")
            print("=" * 124)

        r_gt = load_csv_mat(os.path.join(rdir, "GT_paper_orientation.csv"))
        orient_rows.append(orientation_check(
            ds, gt, r_gt, load_csv_mat(os.path.join(rdir, "D_paper_fastgp_labels_final.csv"))))

        print(f"\n### {ds}/{name}  {raw.shape[0]}x{raw.shape[1]}  "
              f"GT={int(len(np.unique(gt[gt>0])))}")

        # ---- A, C : reuse Round 5 verbatim -----------------------------
        for cell, arm in R5_MAP.items():
            s = r5seg[(r5seg.dataset == ds) & (r5seg.method == arm)]
            assert len(s) == 1, f"round5 row missing for {ds}/{arm}"
            s = s.iloc[0]
            b = np.load(os.path.join(R5, "binary_masks",
                                     f"{ds}_{name}_{arm}.npz"))["binary"]
            lab = np.load(os.path.join(R5, "instance_masks",
                                       f"{ds}_{name}_{arm}.npz"))["instance_mask"]
            m = score(gt, lab, b)
            # reuse must reproduce round 5 exactly
            assert abs(m["ap50"] - float(s.ap50)) < 1e-12, (
                f"{cell} AP mismatch vs round5: {m['ap50']} vs {s.ap50}")
            rows.append(dict(dataset=ds, image=name, cell=cell,
                             label=CELL_LABEL[cell], reconstruction=(
                                 "raw" if cell == "A" else "paper_fast_gp"),
                             downstream="current corrected (global Li + peak markers)",
                             source="round5 (reused, verified identical)", **m))
            rt_rows.append(dict(dataset=ds, cell=cell, label=CELL_LABEL[cell],
                                t_recon_s=float(s.runtime_recon_s),
                                t_threshold_s=np.nan,
                                t_watershed_cleanup_s=np.nan,
                                t_downstream_s=float(s.runtime_downstream_s),
                                t_total_s=float(s.runtime_total_s)))
            print(f"  {CELL_LABEL[cell]:<52} AP@0.5={m['ap50']:.4f} "
                  f"AP@0.75={m['ap75']:.4f} Dice={m['fg_dice']:.4f} "
                  f"[round-5 reuse verified]")

        # ---- B, D (+ B_literal) : real R paper downstream ---------------
        for cell, tag in list(R_MAP.items()) + [("B_literal", "B_nogp_literal")]:
            lab = load_csv_mat(os.path.join(rdir, f"{tag}_labels_final.csv"))
            bin_ = load_csv_mat(os.path.join(rdir, f"{tag}_combined_binary.csv"))
            assert lab.shape == gt.shape and bin_.shape == gt.shape
            m = score(gt, lab, bin_)
            summ = pd.read_csv(os.path.join(rdir, f"{tag}_summary.csv")).iloc[0]
            rows.append(dict(
                dataset=ds, image=name, cell=cell, label=CELL_LABEL[cell],
                reconstruction=("paper_fast_gp (R, paper-faithful fit)"
                                if tag == "D_paper_fastgp" else "raw tiles"),
                downstream=("paper R: criterion_1+RobustGaSP, outlier tiles, "
                            "EBImage distmap+watershed, eliminate_small_areas(50)"
                            if tag != "B_nogp_literal" else
                            "paper R PUBLISHED NoGP: criterion_1_2 (NO rgasp), "
                            "EBImage, eliminate_small_areas2(mean*0.15/*0.05)"),
                source="real R this round", **m))
            rt_rows.append(dict(
                dataset=ds, cell=cell, label=CELL_LABEL[cell],
                t_recon_s=float(summ.get("t_recon_s", np.nan)),
                t_threshold_s=float(summ.get("t_threshold_s", np.nan)),
                t_watershed_cleanup_s=float(summ.get("t_watershed_cleanup_s", np.nan)),
                t_downstream_s=(float(summ.get("t_threshold_s", np.nan)) +
                                float(summ.get("t_watershed_cleanup_s", np.nan))),
                t_total_s=float(summ["t_total_s"])))
            print(f"  {CELL_LABEL[cell]:<52} AP@0.5={m['ap50']:.4f} "
                  f"AP@0.75={m['ap75']:.4f} Dice={m['fg_dice']:.4f}")

            # Step 8 artefacts
            np.savez_compressed(
                os.path.join(out, "binary_masks", f"{ds}_{name}_{cell}.npz"),
                binary=(bin_ > 0).astype(np.uint8))
            np.savez_compressed(
                os.path.join(out, "instance_masks", f"{ds}_{name}_{cell}.npz"),
                instance_mask=lab.astype(np.int32))
            pre = os.path.join(rdir, f"{tag}_watershed_precleanup.csv")
            if os.path.exists(pre):
                pl = load_csv_mat(pre)
                np.savez_compressed(
                    os.path.join(out, "watershed_precleanup",
                                 f"{ds}_{name}_{cell}.npz"),
                    labels_precleanup=pl.astype(np.int32),
                    n_labels_precleanup=np.int64(len(np.unique(pl)) - 1),
                    n_labels_final=np.int64(len(np.unique(lab)) - 1))

        # ---- overlays -------------------------------------------------
        cells = ["A", "B", "C", "D"]
        fig, ax = plt.subplots(1, 5, figsize=(26, 5.6))
        ax[0].imshow(raw, cmap="gray"); ax[0].set_title(f"{ds} raw", fontsize=9)
        ax[0].contour(gt > 0, levels=[0.5], colors="lime", linewidths=0.4)
        ax[0].axis("off")
        for k, cell in enumerate(cells):
            f = (os.path.join(R5, "instance_masks", f"{ds}_{name}_{R5_MAP[cell]}.npz")
                 if cell in R5_MAP else None)
            lab = (np.load(f)["instance_mask"] if f else
                   load_csv_mat(os.path.join(rdir, f"{R_MAP[cell]}_labels_final.csv")))
            a = ax[k + 1]
            a.imshow(raw, cmap="gray")
            a.contour(gt > 0, levels=[0.5], colors="lime", linewidths=0.4)
            a.imshow(np.ma.masked_where(~find_boundaries(lab.astype(np.int32),
                                                         mode="outer"),
                                        np.ones_like(lab)),
                     cmap="autumn", alpha=0.9)
            r = [x for x in rows if x["dataset"] == ds and x["cell"] == cell][0]
            a.set_title(f"{cell}  AP@0.5={r['ap50']:.3f}  n={r['n_pred']}",
                        fontsize=9)
            a.axis("off")
        fig.suptitle(f"{ds}: green=GT boundary, red=predicted boundary", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "overlays", f"{ds}_2x2_overlay.png"), dpi=115)
        plt.close(fig)

        # ---- threshold diagnostics figure -----------------------------
        for tag in ("D_paper_fastgp", "B_nogp_canonical"):
            p = os.path.join(rdir, f"threshdiag_{tag}_tile1.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            selp = os.path.join(rdir, f"threshdiag_{tag}_tile1_selected.csv")
            sel = pd.read_csv(selp).iloc[0] if os.path.exists(selp) else None
            d.to_csv(os.path.join(out, "threshold_diagnostics",
                                  f"{ds}_{tag}_tile1_curve.csv"), index=False)
            if sel is not None:
                pd.DataFrame([sel]).to_csv(
                    os.path.join(out, "threshold_diagnostics",
                                 f"{ds}_{tag}_tile1_selected.csv"), index=False)
            fig, a = plt.subplots(figsize=(7.5, 4.4))
            a.plot(d.percentage, d.raw_criterion_curve, ".", ms=3, alpha=.55,
                   label="raw criterion curve |diff(pixel counts)|")
            a.plot(d.percentage, d.rgasp_fitted_curve, "-", lw=1.7,
                   label="RobustGaSP rgasp() fitted mean")
            if sel is not None:
                a.axvline(sel.selected_percentage, color="crimson", ls="--", lw=1.3,
                          label=f"selected pct={sel.selected_percentage:.2f} "
                                f"(abs={sel.selected_abs_threshold:.4f})")
            a.set_xlabel("threshold as a PROPORTION of max(tile)")
            a.set_ylabel("|change in foreground pixel count|")
            a.set_title(f"{ds} {tag} tile 1: paper criterion_1 with real RobustGaSP",
                        fontsize=9)
            a.legend(fontsize=7); fig.tight_layout()
            fig.savefig(os.path.join(out, "threshold_diagnostics",
                                     f"{ds}_{tag}_tile1_curve.png"), dpi=125)
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "two_by_two_metrics.csv"), index=False)
    df[["dataset", "image", "cell", "label", "fg_dice", "fg_iou", "fg_precision",
        "fg_recall", "fg_fraction", "gt_fg_fraction"]].to_csv(
        os.path.join(out, "foreground_metrics.csv"), index=False)
    pd.DataFrame(rt_rows).to_csv(os.path.join(out, "runtime.csv"), index=False)
    pd.DataFrame(sanity).to_csv(
        os.path.join(out, "tables", "evaluator_sanity_checks.csv"), index=False)
    pd.DataFrame(orient_rows).to_csv(
        os.path.join(out, "tables", "orientation_checks.csv"), index=False)

    # ---- Step 10 : interaction decomposition --------------------------
    dec_rows: List[Dict] = []
    print("\n" + "=" * 124)
    print("STEP 10  INTERACTION DECOMPOSITION   (n=1 image per dataset; descriptive only)")
    print("=" * 124)
    for ds in sorted(df["dataset"].unique()):
        g = {c: df[(df.dataset == ds) & (df.cell == c)].iloc[0]
             for c in ("A", "B", "C", "D")}
        print(f"\n  {ds}")
        for met in ("ap50", "ap75", "fg_dice"):
            A, B, C, D = (float(g[c][met]) for c in "ABCD")
            dec = dict(dataset=ds, metric=met, A=A, B=B, C=C, D=D,
                       recon_effect_current_CmA=C - A,
                       recon_effect_paper_DmB=D - B,
                       downstream_effect_noGP_BmA=B - A,
                       downstream_effect_fastgp_DmC=D - C,
                       interaction=(D - B) - (C - A))
            dec_rows.append(dec)
            print(f"    {met:<8} A={A:.4f} B={B:.4f} C={C:.4f} D={D:.4f} | "
                  f"C-A={C-A:+.4f}  D-B={D-B:+.4f}  B-A={B-A:+.4f}  "
                  f"D-C={D-C:+.4f}  interaction={(D-B)-(C-A):+.4f}")
    pd.DataFrame(dec_rows).to_csv(
        os.path.join(out, "tables", "interaction_decomposition.csv"), index=False)

    json.dump(dict(
        round="round6-paper-downstream-2x2",
        cell_seg_qep_commit="a48381de87bf59490f4c8a9ec09670270a45caa3",
        r_reference_repo=R_REPO, r_reference_commit=R_SHA,
        development_images=DEV,
        A_C_source="results/real_cellseg_round5_corrected_baseline_20260917 (reused, AP verified identical)",
        B_D_source="real R via audits/round6_paper_downstream.R",
        paper_downstream=dict(
            threshold="criterion_1: candidates seq(0,1,by=0.01) as PROPORTIONS of max(tile), "
                      "per tile; diff_pixel_counts=abs(diff(counts)); smoothed by "
                      "RobustGaSP rgasp(nugget.est=TRUE) -> predict()$mean; walk forward "
                      "from which.max until |d[i]-d[i-1]| < 0.05*sd(d); "
                      "selected = percentages[stable_index+1]; fallback = all background, pct=1",
            outlier_tiles="|n_connected - mean| > 2*sd -> re-threshold at mean of "
                          "non-outlier pcts; >0.99 foreground -> revert to all background",
            distance_transform="EBImage::distmap(as.Image(binary)), metric='euclidean'",
            watershed="EBImage::watershed(dist_map) with defaults tolerance=1, ext=1; "
                      "POSITIVE distance map (not negated); EBImage does its own "
                      "tolerance-based seed detection",
            cleanup="eliminate_small_areas(mask, size_threshold=50): area<50 & !on_boundary "
                    "-> remove; on_boundary & area<10 -> remove",
            B_is="CONSTRUCTED ABLATION: predmean_mat <- img_matrix, downstream untouched",
            B_literal_is="PUBLISHED NoGP arm generate_GP_Masks_test2: differs from the GP "
                         "arm in THREE ways (no image GP, NO rgasp on the criterion curve, "
                         "eliminate_small_areas2 with mean*0.15/mean*0.05)"),
        current_downstream=dict(
            threshold="Round-3 selected rule: Li, GLOBAL on the stitched reconstruction",
            markers="peak_local_max, min_distance nuclei=15 whole_cell=9",
            watershed="-distance_transform_edt, connectivity=1",
            cleanup="eliminate_small_areas(50)"),
        evaluator="py_core/segmentation_eval.py, AP=TP/(TP+FP+FN), identical for all cells",
        environment=dict(python=platform.python_version(),
                         platform=platform.platform(), numpy=np.__version__),
    ), open(os.path.join(out, "paper_pipeline_config.json"), "w"), indent=2)

    # ---- final compact table ------------------------------------------
    print("\n" + "=" * 124)
    print("FINAL 2 x 2 TABLE   AP@0.5")
    print("=" * 124)
    print(f"{'Dataset':<12}{'A Raw+Current':>16}{'B NoGP+Paper':>15}"
          f"{'C FastGP+Current':>19}{'D FastGP+Paper':>17}")
    for ds in sorted(df["dataset"].unique()):
        g = {c: float(df[(df.dataset == ds) & (df.cell == c)].iloc[0].ap50)
             for c in "ABCD"}
        print(f"{ds:<12}{g['A']:>16.4f}{g['B']:>15.4f}{g['C']:>19.4f}{g['D']:>17.4f}")
    for ds in sorted(df["dataset"].unique()):
        g = {c: float(df[(df.dataset == ds) & (df.cell == c)].iloc[0].ap50)
             for c in "ABCD"}
        print(f"\n{ds}:")
        print(f"  C-A = {g['C']-g['A']:+.4f}")
        print(f"  D-B = {g['D']-g['B']:+.4f}")
        print(f"  interaction = {(g['D']-g['B'])-(g['C']-g['A']):+.4f}")
    print(f"\nWrote outputs into {out}")


if __name__ == "__main__":
    main()
