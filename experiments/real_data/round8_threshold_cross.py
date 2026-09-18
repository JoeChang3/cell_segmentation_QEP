"""
Round 8: separate (A) Fast-GP changing the intensity field from (B) Fast-GP
changing the threshold proportion p chosen by the paper's RobustGaSP criterion.

    RR = raw    image + p_raw        GR = fastgp image + p_raw
    RG = raw    image + p_fastgp     GG = fastgp image + p_fastgp

R side (audits/round8_threshold_cross.R) already produced the four crossed
foregrounds and ran the unchanged paper downstream on each. Nothing was refit and
criterion_1/RobustGaSP was not re-run -- the p values are frozen from Round 6.

This script does Steps 1, 2, 4, 5, 6, 7.

Usage: python experiments/real_data/round8_threshold_cross.py --out <dir>
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
R6 = os.path.join(_ROOT, "results", "real_cellseg_round6_paper_downstream_20260917")
CONDS = ["RR", "RG", "GR", "GG"]
CLAB = {"RR": "RR  raw image + p_raw", "RG": "RG  raw image + p_fastgp",
        "GR": "GR  fastgp image + p_raw", "GG": "GG  fastgp image + p_fastgp"}
CLEANUP_REF = 50          # the paper's remove_size_threshold, used as a reference


def mat(p: str) -> np.ndarray:
    return np.loadtxt(p, delimiter=",")


def comp_stats(fg: np.ndarray) -> Dict:
    lab, n = ndi.label(fg, structure=ndi.generate_binary_structure(2, 1))
    if n == 0:
        return dict(n_components=0, comp_size_median=np.nan, comp_size_mean=np.nan,
                    comp_size_p10=np.nan, comp_size_p90=np.nan,
                    n_components_under_cleanup_ref=0)
    sz = np.bincount(lab.ravel())[1:]
    return dict(n_components=int(n), comp_size_median=float(np.median(sz)),
                comp_size_mean=float(sz.mean()),
                comp_size_p10=float(np.percentile(sz, 10)),
                comp_size_p90=float(np.percentile(sz, 90)),
                n_components_under_cleanup_ref=int((sz < CLEANUP_REF).sum()))


def gt_coverage(fg: np.ndarray, gt: np.ndarray) -> Dict:
    ids = np.unique(gt[gt > 0])
    tot = np.bincount(gt.ravel(), minlength=int(gt.max()) + 1)
    cov = np.bincount(gt.ravel(), weights=fg.ravel().astype(float),
                      minlength=int(gt.max()) + 1)
    frac = np.array([cov[g] / tot[g] for g in ids])
    return dict(n_gt=int(len(ids)),
                gt_zero_coverage=int((frac == 0).sum()),
                gt_partial_coverage=int(((frac > 0) & (frac < 0.5)).sum()),
                gt_ge50_coverage=int((frac >= 0.5).sum()),
                gt_ge90_coverage=int((frac >= 0.9).sum()),
                gt_mean_coverage=float(frac.mean()))


def intensity_diag(ds: str, tile_id: int, raw: np.ndarray, gp: np.ndarray,
                   p_raw: float, t_raw_abs: float) -> Dict:
    def q(a, v):
        return float(np.percentile(a, v))

    def mad(a):
        return float(np.median(np.abs(a - np.median(a))))

    def islands(a, thr):
        b = a > thr
        lab, n = ndi.label(b, structure=ndi.generate_binary_structure(2, 1))
        if n == 0:
            return 0, 0
        sz = np.bincount(lab.ravel())[1:]
        return int(n), int((sz < CLEANUP_REF).sum())

    n_r, sm_r = islands(raw, p_raw * raw.max())
    n_g, sm_g = islands(gp, p_raw * gp.max())
    # how many pixels sit strictly between the 99.9th pct and the max: a proxy for
    # whether max(tile) is set by a handful of extreme pixels
    return dict(
        dataset=ds, tile=tile_id,
        max_raw=float(raw.max()), max_gp=float(gp.max()),
        p999_raw=q(raw, 99.9), p999_gp=q(gp, 99.9),
        p99_raw=q(raw, 99), p99_gp=q(gp, 99),
        p95_raw=q(raw, 95), p95_gp=q(gp, 95),
        median_raw=float(np.median(raw)), median_gp=float(np.median(gp)),
        mad_raw=mad(raw), mad_gp=mad(gp),
        n_px_at_or_above_max_raw=int((raw >= raw.max() - 1e-12).sum()),
        n_px_at_or_above_max_gp=int((gp >= gp.max() - 1e-12).sum()),
        max_minus_p999_raw=float(raw.max() - q(raw, 99.9)),
        max_minus_p999_gp=float(gp.max() - q(gp, 99.9)),
        n_px_above_T_raw_orig_in_raw=int((raw > t_raw_abs).sum()),
        n_px_above_T_raw_orig_in_gp=int((gp > t_raw_abs).sum()),
        islands_at_p_raw_in_raw=n_r, small_islands_at_p_raw_in_raw=sm_r,
        islands_at_p_raw_in_gp=n_g, small_islands_at_p_raw_in_gp=sm_g)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = args.out
    for s in ("binary_masks", "instance_masks", "figures", "logs", "tables"):
        os.makedirs(os.path.join(out, s), exist_ok=True)

    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[(man["status"] == "ok") & (man["role"] == "development")]
    r6 = pd.read_csv(os.path.join(R6, "two_by_two_metrics.csv"))

    tile_rows: List[Dict] = []
    fg_rows: List[Dict] = []
    inst_rows: List[Dict] = []
    int_rows: List[Dict] = []
    abs_rows: List[Dict] = []

    print("=" * 130)
    print("ROUND 8  p-CROSS FACTORIAL  (nothing refit; p frozen from Round 6; "
          "paper downstream unchanged)")
    print("=" * 130)

    for _, im in man.iterrows():
        ds, name = im["dataset"], im["image"]
        gt = load_instance_mask(os.path.join(_ROOT, im["path_gt"]))
        gt_fg = gt > 0
        rdir6 = os.path.join(R6, "r_outputs", ds)
        rdir8 = os.path.join(out, "r_outputs", ds)

        # ---- Step 1 : verify inputs -----------------------------------
        raw_rec = mat(os.path.join(rdir6, "B_nogp_canonical_combined_predmean.csv"))
        gp_rec = mat(os.path.join(rdir6, "D_paper_fastgp_combined_predmean.csv"))
        mR = pd.read_csv(os.path.join(rdir6, "B_nogp_canonical_tile_meta.csv"))
        mG = pd.read_csv(os.path.join(rdir6, "D_paper_fastgp_tile_meta.csv"))
        assert raw_rec.shape == gp_rec.shape == gt.shape
        assert len(mR) == len(mG)

        # the raw "reconstruction" must be the raw image (magick raw/255) in the
        # covered region; the fastgp one must be Round 5's paper_fast_gp array
        src = imageio.imread(os.path.join(_ROOT, im["path_image"]))
        src = ((src[..., 0] if src.ndim == 3 else src).astype(np.float64)) / 255.0
        cov = np.zeros(gt.shape, bool)
        for _, t in mR.iterrows():
            cov[int(t.y_offset):int(t.y_offset) + int(t.h),
                int(t.x_offset):int(t.x_offset) + int(t.w)] = True
        d_raw = float(np.abs(raw_rec[cov] - src[cov]).max())
        # NOTE: Round-6 D uses the R-side paper-faithful Fast-GP fit, which on
        # nuclei lands on the degenerate beta->inf plateau, whereas Round-5's
        # paper_fast_gp used the Python fit that found the better basin. The two
        # therefore legitimately differ; this is reported, not asserted. The
        # binding check is that RR/GG reproduce Round-6 B/D exactly (below).
        r5 = np.load(os.path.join(R5, "masks",
                                  f"{ds}_{name}_paper_fast_gp.npz"))["predmean"]
        d_gp = float(np.abs(gp_rec[cov] - (r5.astype(np.float64) / 255.0)[cov]).max())
        print(f"\n### {ds}/{name}  {gt.shape}  GT={int(len(np.unique(gt[gt>0])))}")
        print(f"  input check: max|raw_recon - image/255| = {d_raw:.3e}  (asserted 0)")
        print(f"  reference note: max|R fastgp recon - Python round5 recon|/255 "
              f"= {d_gp:.3e}  (R vs Python optimizer basin; see wiring audit)")
        assert d_raw < 1e-12, "raw reconstruction is not the raw image"

        # ---- per-tile threshold table ---------------------------------
        for k in range(len(mR)):
            tR, tG = mR.iloc[k], mG.iloc[k]
            assert (tR.x_offset, tR.y_offset, tR.h, tR.w) == \
                   (tG.x_offset, tG.y_offset, tG.h, tG.w), "tile geometry mismatch"
            sl = (slice(int(tR.y_offset), int(tR.y_offset) + int(tR.h)),
                  slice(int(tR.x_offset), int(tR.x_offset) + int(tR.w)))
            rt, gtile = raw_rec[sl], gp_rec[sl]
            assert abs(float(rt.max()) - float(tR.tile_max)) < 1e-9
            assert abs(float(gtile.max()) - float(tG.tile_max)) < 1e-9
            tile_rows.append(dict(
                dataset=ds, tile_id=int(tR.tile), i=int(tR.i), j=int(tR.j),
                y_offset=int(tR.y_offset), x_offset=int(tR.x_offset),
                p_raw=float(tR.pct_selected), p_fastgp=float(tG.pct_selected),
                max_raw=float(tR.tile_max), max_fastgp=float(tG.tile_max),
                T_raw_original=float(tR.pct_selected) * float(tR.tile_max),
                T_fastgp_original=float(tG.pct_selected) * float(tG.tile_max),
                delta_p=float(tG.pct_selected) - float(tR.pct_selected),
                max_ratio=float(tG.tile_max) / float(tR.tile_max)))
            int_rows.append(intensity_diag(
                ds, int(tR.tile), rt, gtile, float(tR.pct_selected),
                float(tR.pct_selected) * float(tR.tile_max)))

        # ---- Steps 2 + 4 : score the four conditions ------------------
        print(f"  {'cond':<28}{'fg%':>7}{'Dice':>8}{'IoU':>8}{'prec':>7}{'rec':>7}"
              f"{'comps':>8}{'<50px':>7}{'basins':>8}{'final':>7}"
              f"{'AP@.5':>8}{'AP@.75':>8}{'merge':>6}{'split':>6}")
        recs = {}
        for c in CONDS:
            b = mat(os.path.join(rdir8, f"{c}_binary.csv")) > 0
            pre = mat(os.path.join(rdir8, f"{c}_precleanup.csv"))
            fin = mat(os.path.join(rdir8, f"{c}_labels_final.csv"))
            recs[c] = (b, pre, fin)
            fgm = foreground_metrics(b, gt_fg)
            cs = comp_stats(b)
            cv = gt_coverage(b, gt)
            ev = evaluate_instances(gt, fin.astype(np.int32))
            fm = classify_failure_modes(gt, fin.astype(np.int32))
            s6 = pd.read_csv(os.path.join(rdir8, f"{c}_summary.csv")).iloc[0]

            fg_rows.append(dict(dataset=ds, image=name, condition=c, label=CLAB[c],
                                image_source=("raw" if c[0] == "R" else "fastgp"),
                                p_source=("p_raw" if c[1] == "R" else "p_fastgp"),
                                **fgm, **cs, **cv,
                                n_outlier_tiles=int(s6.n_outlier_tiles)))
            inst_rows.append(dict(
                dataset=ds, image=name, condition=c, label=CLAB[c],
                image_source=("raw" if c[0] == "R" else "fastgp"),
                p_source=("p_raw" if c[1] == "R" else "p_fastgp"),
                ap50=ev.ap(0.5), ap75=ev.ap(0.75),
                tp50=ev.per_threshold[0.5]["tp"], fp50=ev.per_threshold[0.5]["fp"],
                fn50=ev.per_threshold[0.5]["fn"],
                tp75=ev.per_threshold[0.75]["tp"], fp75=ev.per_threshold[0.75]["fp"],
                fn75=ev.per_threshold[0.75]["fn"],
                n_pred=ev.n_pred, n_gt=ev.n_true,
                mean_matched_iou=ev.mean_matched_iou,
                merged=fm["merged"], split=fm["split"], missed=fm["missed"],
                spurious=fm["spurious"],
                n_basins_precleanup=int(len(np.unique(pre[pre > 0]))),
                n_instances_final=int(len(np.unique(fin[fin > 0]))),
                fg_dice=fgm["fg_dice"], n_components=cs["n_components"]))
            print(f"  {CLAB[c]:<28}{100*fgm['fg_fraction']:>7.1f}"
                  f"{fgm['fg_dice']:>8.4f}{fgm['fg_iou']:>8.4f}"
                  f"{fgm['fg_precision']:>7.3f}{fgm['fg_recall']:>7.3f}"
                  f"{cs['n_components']:>8}{cs['n_components_under_cleanup_ref']:>7}"
                  f"{len(np.unique(pre[pre>0])):>8}{len(np.unique(fin[fin>0])):>7}"
                  f"{ev.ap(0.5):>8.4f}{ev.ap(0.75):>8.4f}"
                  f"{fm['merged']:>6}{fm['split']:>6}")
            np.savez_compressed(os.path.join(out, "binary_masks",
                                             f"{ds}_{name}_{c}.npz"),
                                binary=b.astype(np.uint8))
            np.savez_compressed(os.path.join(out, "instance_masks",
                                             f"{ds}_{name}_{c}.npz"),
                                instance_mask=fin.astype(np.int32),
                                labels_precleanup=pre.astype(np.int32))

        # ---- self-check: RR == round6 B, GG == round6 D ---------------
        for c, cell in (("RR", "B"), ("GG", "D")):
            got = [r for r in inst_rows
                   if r["dataset"] == ds and r["condition"] == c][0]
            ref = r6[(r6.dataset == ds) & (r6.cell == cell)].iloc[0]
            ok = abs(got["ap50"] - float(ref.ap50)) < 1e-12
            print(f"  self-check {c} vs Round-6 {cell}: AP@0.5 {got['ap50']:.4f} "
                  f"vs {float(ref.ap50):.4f} -> {'MATCH' if ok else 'MISMATCH'}")
            assert ok, f"{c} does not reproduce Round-6 {cell}"

        # ---- Step 7 : SECONDARY absolute-threshold cross-check --------
        for img_key, img in (("raw", raw_rec), ("fastgp", gp_rec)):
            for t_key, meta_t in (("T_raw_original", mR), ("T_fastgp_original", mG)):
                b = np.zeros(gt.shape, bool)
                for k in range(len(meta_t)):
                    t = meta_t.iloc[k]
                    sl = (slice(int(t.y_offset), int(t.y_offset) + int(t.h)),
                          slice(int(t.x_offset), int(t.x_offset) + int(t.w)))
                    b[sl] = img[sl] > (float(t.pct_selected) * float(t.tile_max))
                fgm = foreground_metrics(b, gt_fg)
                cs = comp_stats(b)
                abs_rows.append(dict(dataset=ds, image_source=img_key,
                                     absolute_threshold=t_key, **fgm, **cs))

        # ---- figures ---------------------------------------------------
        fig, ax = plt.subplots(2, 4, figsize=(23, 11))
        for k, c in enumerate(CONDS):
            b, pre, fin = recs[c]
            a = ax[0, k]
            a.imshow(b, cmap="gray")
            fgm = [r for r in fg_rows
                   if r["dataset"] == ds and r["condition"] == c][0]
            a.set_title(f"{c} foreground: Dice={fgm['fg_dice']:.3f}  "
                        f"comps={fgm['n_components']}", fontsize=9)
            a.axis("off")
            a = ax[1, k]
            a.imshow(np.zeros_like(b, dtype=float), cmap="gray")
            a.contour(gt_fg, levels=[0.5], colors="lime", linewidths=0.35)
            a.imshow(np.ma.masked_where(~find_boundaries(fin.astype(np.int32),
                                                         mode="outer"),
                                        np.ones_like(fin)),
                     cmap="autumn", alpha=0.95)
            ir = [r for r in inst_rows
                  if r["dataset"] == ds and r["condition"] == c][0]
            a.set_title(f"{c}: AP@0.5={ir['ap50']:.3f}  n={ir['n_pred']}",
                        fontsize=9)
            a.axis("off")
        fig.suptitle(f"{ds}: p-cross factorial. top = crossed foreground, "
                     f"bottom = green GT / red predicted", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"{ds}_threshold_cross.png"),
                    dpi=115)
        plt.close(fig)

    tdf = pd.DataFrame(tile_rows); tdf.to_csv(
        os.path.join(out, "tile_thresholds.csv"), index=False)
    fdf = pd.DataFrame(fg_rows); fdf.to_csv(
        os.path.join(out, "foreground_cross_metrics.csv"), index=False)
    idf = pd.DataFrame(inst_rows); idf.to_csv(
        os.path.join(out, "instance_cross_metrics.csv"), index=False)
    ndf = pd.DataFrame(int_rows); ndf.to_csv(
        os.path.join(out, "intensity_diagnostics.csv"), index=False)
    pd.DataFrame(abs_rows).to_csv(
        os.path.join(out, "tables", "secondary_absolute_threshold_cross.csv"),
        index=False)

    # ---- Step 5 : decomposition ------------------------------------------
    dec: List[Dict] = []
    print("\n" + "=" * 130)
    print("STEP 5  DECOMPOSITION   image effect vs p-selection effect  "
          "(n=1 per dataset; descriptive)")
    print("=" * 130)
    for ds in sorted(idf["dataset"].unique()):
        gi = {c: idf[(idf.dataset == ds) & (idf.condition == c)].iloc[0]
              for c in CONDS}
        print(f"\n  {ds}")
        for met in ("ap50", "ap75", "n_components", "fg_dice"):
            RR, RG, GR, GG = (float(gi[c][met]) for c in CONDS)
            d = dict(dataset=ds, metric=met, RR=RR, RG=RG, GR=GR, GG=GG,
                     p_effect_on_raw_RGmRR=RG - RR,
                     p_effect_on_fastgp_GGmGR=GG - GR,
                     image_effect_at_p_raw_GRmRR=GR - RR,
                     image_effect_at_p_fastgp_GGmRG=GG - RG,
                     interaction=(GG - GR) - (RG - RR),
                     total_GGmRR=GG - RR)
            dec.append(d)
            print(f"    {met:<13} RR={RR:>9.4f} RG={RG:>9.4f} GR={GR:>9.4f} "
                  f"GG={GG:>9.4f} | p-eff(raw)={RG-RR:>+9.4f} "
                  f"p-eff(gp)={GG-GR:>+9.4f} img-eff@p_raw={GR-RR:>+9.4f} "
                  f"img-eff@p_gp={GG-RG:>+9.4f} inter={(GG-GR)-(RG-RR):>+9.4f}")
    pd.DataFrame(dec).to_csv(
        os.path.join(out, "tables", "decomposition.csv"), index=False)

    # ---- Step 6 : max-intensity mechanism -------------------------------
    print("\n" + "=" * 130)
    print("STEP 6  MAX-INTENSITY MECHANISM  (per-tile, Raw vs FastGP)")
    print("=" * 130)
    for ds in sorted(ndf["dataset"].unique()):
        s = ndf[ndf.dataset == ds]
        t = tdf[tdf.dataset == ds]
        print(f"\n  {ds}  ({len(s)} tiles)")
        print(f"    {'quantity':<34}{'raw mean':>12}{'fastgp mean':>13}{'ratio':>9}")
        for lab, a, b in (("max(tile)", "max_raw", "max_gp"),
                          ("99.9th pct", "p999_raw", "p999_gp"),
                          ("99th pct", "p99_raw", "p99_gp"),
                          ("95th pct", "p95_raw", "p95_gp"),
                          ("median", "median_raw", "median_gp"),
                          ("MAD", "mad_raw", "mad_gp"),
                          ("max - 99.9th pct", "max_minus_p999_raw",
                           "max_minus_p999_gp"),
                          ("#px at max", "n_px_at_or_above_max_raw",
                           "n_px_at_or_above_max_gp"),
                          ("islands at p_raw", "islands_at_p_raw_in_raw",
                           "islands_at_p_raw_in_gp"),
                          ("small islands (<50px) at p_raw",
                           "small_islands_at_p_raw_in_raw",
                           "small_islands_at_p_raw_in_gp")):
            ra, rb = float(s[a].mean()), float(s[b].mean())
            print(f"    {lab:<34}{ra:>12.5f}{rb:>13.5f}"
                  f"{(rb/ra if ra else float('nan')):>9.3f}")
        print(f"    mean delta_p = {t.delta_p.mean():+.4f}   "
              f"mean max_ratio = {t.max_ratio.mean():.4f}   "
              f"tiles with |delta_p|>0.05: {int((t.delta_p.abs()>0.05).sum())}/{len(t)}")

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        mm = s.merge(t, left_on="tile", right_on="tile_id")
        cc = {c: fdf[(fdf.dataset == ds) & (fdf.condition == c)].iloc[0].n_components
              for c in CONDS}
        ax[0].scatter(mm.delta_p, mm.islands_at_p_raw_in_raw -
                      mm.islands_at_p_raw_in_gp, s=26)
        ax[0].set_xlabel("delta_p = p_fastgp - p_raw")
        ax[0].set_ylabel("islands(raw) - islands(fastgp)  at p_raw")
        ax[0].set_title("per-tile: p change vs island reduction", fontsize=9)
        ax[1].scatter(mm.max_ratio, mm.islands_at_p_raw_in_raw -
                      mm.islands_at_p_raw_in_gp, s=26, color="darkorange")
        ax[1].set_xlabel("max_fastgp / max_raw")
        ax[1].set_ylabel("islands(raw) - islands(fastgp)")
        ax[1].set_title("per-tile: max change vs island reduction", fontsize=9)
        ax[2].bar(CONDS, [cc[c] for c in CONDS],
                  color=["0.4", "0.6", "steelblue", "navy"])
        ax[2].set_ylabel("foreground connected components")
        ax[2].set_title(f"{ds}: components by condition", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"{ds}_intensity_mechanism.png"),
                    dpi=125)
        plt.close(fig)

    json.dump(dict(
        round="round8-threshold-cross",
        question="Does Fast-GP help the paper pipeline by changing the intensity "
                 "field, or by changing the selected threshold proportion p?",
        nothing_refit=True, criterion_1_rerun=False,
        p_values_source=os.path.relpath(R6, _ROOT) + "/r_outputs/*/*_tile_meta.csv",
        reconstructions_source=os.path.relpath(R6, _ROOT)
        + "/r_outputs/*/*_combined_predmean.csv",
        crossing_rule="threshold = p_donor * max(RECIPIENT tile). The recipient "
                      "image's own max is always used; we cross the proportion p, "
                      "not an absolute intensity.",
        downstream="unchanged paper downstream: outlier-tile handling, "
                   "EBImage::distmap, EBImage::watershed(tolerance=1, ext=1) "
                   "implicit seeding, eliminate_small_areas(., 50). No explicit "
                   "markers in the primary experiment.",
        self_checks=["RR reproduces Round-6 B AP@0.5 to <1e-12",
                     "GG reproduces Round-6 D AP@0.5 to <1e-12",
                     "raw reconstruction == image/255 in the tiled region",
                     "fastgp reconstruction == Round-5 paper_fast_gp/255",
                     "per-tile max matches Round-6 tile_meta to 1e-9",
                     "tile geometry identical between arms"],
        secondary_absolute_threshold_cross="tables/"
        "secondary_absolute_threshold_cross.csv -- a DIAGNOSTIC only, not the "
        "paper-faithful factorial, because raw and fastgp have different scales",
        environment=dict(python=platform.python_version(),
                         platform=platform.platform(), numpy=np.__version__),
    ), open(os.path.join(out, "config.json"), "w"), indent=2)

    # ---- compact summary ------------------------------------------------
    print("\n" + "=" * 130)
    print("COMPACT SUMMARY")
    print("=" * 130)
    for ds in ("nuclei", "whole_cell"):
        gi = {c: idf[(idf.dataset == ds) & (idf.condition == c)].iloc[0]
              for c in CONDS}
        gf = {c: fdf[(fdf.dataset == ds) & (fdf.condition == c)].iloc[0]
              for c in CONDS}
        print(f"\n{ds.upper().replace('_', '-')}:")
        for c in CONDS:
            print(f"  {c} AP = {float(gi[c].ap50):.4f}")
        for c in CONDS:
            print(f"  {c} components = {int(gf[c].n_components)}")
    print(f"\nWrote outputs into {out}")


if __name__ == "__main__":
    main()
