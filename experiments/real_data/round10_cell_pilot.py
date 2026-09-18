"""Round 10 cell pilot (Parts 6-10): is the DEEP q-dependent predictive-mean
difference spatially localized at difficult cell interfaces?

Stage 1 = 2 merged + 2 separated pairs per dataset, 1 seed.
Stage 2 = 4 + 4 pairs per dataset, 3 paired seeds (only if Stage 1 passes).

Patches are 25x25, reusing the Round-9 design so the spatial support and the
region bands match the shallow experiment being contrasted with. Pairs are
selected deterministically from the Round-9 inventory (longest and median
interface length) and NEVER using DeepQEP outputs. GT is used only to label
diagnostic regions after the model outputs exist.

Every arm is PAIRED: same seed, same inducing draw, same optimizer steps, same
MC sample streams. Only `power` differs.

No segmentation stage is touched and no AP is computed.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from typing import Dict, List
import imageio.v2 as imageio
import numpy as np, pandas as pd, torch
from scipy import ndimage as ndi
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
torch.set_default_dtype(torch.float64)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from py_core.segmentation_eval import load_instance_mask
from round10_deepqep_core import (env_info, predict_mean_batched, train_deep)
import importlib.util
_s = importlib.util.spec_from_file_location("r9", os.path.join(_HERE, "round9_qep_boundary_signal.py"))
r9 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r9)

R3 = os.path.join(_ROOT, "results", "real_cellseg_round3_thresholding_20260916")
R5 = os.path.join(_ROOT, "results", "real_cellseg_round5_corrected_baseline_20260917")
R9 = os.path.join(_ROOT, "results", "real_cellseg_round9_qep_boundary_signal_20260918")
PATCH, ITERS, NIND, HID = 25, 300, 48, 3
N_MC, N_BATCH = 48, 8
Q_PAIR = (2.0, 1.5)


def pick_pairs(inv: pd.DataFrame, ds: str, n_each: int) -> List[Dict]:
    """Deterministic: longest then median interface length, per merge status."""
    out = []
    for merged in (True, False):
        s = inv[(inv.dataset == ds) & (inv.merged == merged)].sort_values(
            ["interface_len_px", "gt_id_1", "gt_id_2"]).reset_index(drop=True)
        if s.empty: continue
        idx = [len(s) - 1, len(s) // 2] if n_each == 2 else \
              sorted(set(np.linspace(0, len(s) - 1, n_each).round().astype(int)))
        if n_each == 2: idx = sorted(set(idx))
        for k in idx[:n_each]:
            r = s.iloc[k]
            out.append(dict(dataset=ds, gt_id_1=int(r.gt_id_1), gt_id_2=int(r.gt_id_2),
                            merged=bool(r.merged), interface_len_px=int(r.interface_len_px),
                            local_contrast=float(r.local_contrast),
                            centroid_r=float(r.centroid_r), centroid_c=float(r.centroid_c),
                            rank_in_class=int(k), n_in_class=int(len(s))))
    return out


def run(out: str, stage: int):
    inv = pd.read_csv(os.path.join(R9, "adjacent_pair_inventory.csv"))
    man = pd.read_csv(os.path.join(R3, "image_manifest.csv"))
    man = man[(man.status == "ok") & (man.role == "development")]
    n_each = 2 if stage == 1 else 4
    seeds = [73] if stage == 1 else [73, 137, 211]

    prows, mrows, brows, srows, inv_rows = [], [], [], [], []
    print("=" * 124)
    print(f"ROUND 10 CELL PILOT  STAGE {stage}   patches={PATCH}x{PATCH}  "
          f"pairs/class/dataset={n_each}  seeds={seeds}")
    print("  PAIRED arms: identical seed/inducing/steps/MC stream; only power differs")
    print("=" * 124)

    for _, im in man.iterrows():
        ds, name = im.dataset, im.image
        raw = imageio.imread(os.path.join(_ROOT, im.path_image))
        raw = (raw[..., 0] if raw.ndim == 3 else raw).astype(np.float64)
        gt = load_instance_mask(os.path.join(_ROOT, im.path_gt))
        pred = np.load(os.path.join(R5, "instance_masks",
                                    f"{ds}_{name}_raw.npz"))["instance_mask"]
        fastgp = np.load(os.path.join(R5, "masks",
                                      f"{ds}_{name}_paper_fast_gp.npz"))["predmean"].astype(np.float64)
        rg, _ = r9.build_regions(gt, pred)
        H, W = raw.shape
        gy, gx = np.gradient(raw); grad_raw = np.sqrt(gx**2 + gy**2)
        gy, gx = np.gradient(fastgp); grad_fgp = np.sqrt(gx**2 + gy**2)
        sel = pick_pairs(inv, ds, n_each)
        print(f"\n### {ds}: {len(sel)} pairs selected "
              f"({sum(p['merged'] for p in sel)} merged)")

        for p in sel:
            inv_rows.append(dict(**p, patch=PATCH))
            r0 = int(np.clip(round(p["centroid_r"]) - PATCH // 2, 0, H - PATCH))
            c0 = int(np.clip(round(p["centroid_c"]) - PATCH // 2, 0, W - PATCH))
            sl = (slice(r0, r0 + PATCH), slice(c0, c0 + PATCH))
            sub = raw[sl]
            mu_c, sd_c = float(sub.mean()), float(sub.std() + 1e-12)
            Yn = torch.tensor((sub.ravel() - mu_c) / sd_c)
            rr, cc = np.meshgrid(np.arange(PATCH), np.arange(PATCH), indexing="ij")
            X = torch.tensor(np.stack([rr.ravel() / (PATCH - 1),
                                       cc.ravel() / (PATCH - 1)], 1))
            band = rg["iface_owner"].get((p["gt_id_1"], p["gt_id_2"]))
            bands = dict(
                interface=(band[sl] if band is not None else np.zeros((PATCH, PATCH), bool)),
                interior=rg["regions"]["D_interior"][sl],
                background=rg["regions"]["D_background"][sl])
            # edge ring, to test the "patch edge artifact" criterion
            ring = np.zeros((PATCH, PATCH), bool); ring[:2] = ring[-2:] = True
            ring[:, :2] = ring[:, -2:] = True
            bands["patch_edge"] = ring

            for seed in seeds:
                mus, ses, losses = {}, {}, {}
                for q in Q_PAIR:
                    model, ls = train_deep(X, Yn, q, seed=seed, iters=ITERS,
                                           hidden_dims=HID, num_inducing=NIND)
                    m, se = predict_mean_batched(model, X, n_batches=N_BATCH,
                                                 n_samples=N_MC, seed0=50_000)
                    mus[q] = (m * sd_c + mu_c).reshape(PATCH, PATCH)
                    ses[q] = (se * sd_c).reshape(PATCH, PATCH)
                    losses[q] = ls[-1]
                d = mus[1.5] - mus[2.0]
                se_d = np.sqrt(ses[1.5]**2 + ses[2.0]**2)
                row = dict(**{k: p[k] for k in ("dataset","gt_id_1","gt_id_2","merged",
                                                "interface_len_px","local_contrast")},
                           seed=seed,
                           loss_q2=losses[2.0], loss_q15=losses[1.5],
                           finite=bool(np.isfinite(d).all()),
                           max_abs_delta=float(np.abs(d).max()),
                           mean_abs_delta=float(np.abs(d).mean()),
                           rmse_between=float(np.sqrt((d**2).mean())),
                           corr_between=float(np.corrcoef(mus[1.5].ravel(),
                                                          mus[2.0].ravel())[0,1]),
                           mc_se_mean=float(se_d.mean()), mc_se_max=float(se_d.max()),
                           snr_mean=float(np.abs(d).mean()/max(se_d.mean(),1e-300)),
                           snr_max=float(np.abs(d).max()/max(se_d.max(),1e-300)),
                           nonuniform_sd_over_mean=float(np.abs(d).std()/
                                                         max(np.abs(d).mean(),1e-300)))
                for bn, bm in bands.items():
                    row[f"mean_absdelta_{bn}"] = (float(np.abs(d)[bm].mean())
                                                  if bm.any() else np.nan)
                    row[f"n_px_{bn}"] = int(bm.sum())
                row["localization_ratio"] = (row["mean_absdelta_interface"] /
                                             row["mean_absdelta_interior"]
                                             if row.get("n_px_interior",0) > 0 and
                                             np.isfinite(row["mean_absdelta_interior"]) and
                                             row["mean_absdelta_interior"] > 0 else np.nan)
                row["edge_over_interface"] = (row["mean_absdelta_patch_edge"] /
                                              row["mean_absdelta_interface"]
                                              if np.isfinite(row.get("mean_absdelta_interface", np.nan))
                                              and row["mean_absdelta_interface"] > 0 else np.nan)
                prows.append(row)
                mrows.append(dict(dataset=ds, gt_id_1=p["gt_id_1"], gt_id_2=p["gt_id_2"],
                                  seed=seed, n_mc=N_MC, n_batches=N_BATCH,
                                  mc_se_q2_mean=float(ses[2.0].mean()),
                                  mc_se_q15_mean=float(ses[1.5].mean()),
                                  mc_se_delta_mean=float(se_d.mean()),
                                  mc_se_delta_max=float(se_d.max())))
                # D/E: boundary quality of the MEAN
                brow = dict(dataset=ds, gt_id_1=p["gt_id_1"], gt_id_2=p["gt_id_2"],
                            merged=p["merged"], seed=seed,
                            interface_len_px=p["interface_len_px"])
                for q in Q_PAIR:
                    ggy, ggx = np.gradient(mus[q])
                    g = np.sqrt(ggx**2 + ggy**2)
                    tag = "q2" if q == 2.0 else "q15"
                    brow[f"gradmu_{tag}_interface"] = (float(g[bands["interface"]].mean())
                                                       if bands["interface"].any() else np.nan)
                    brow[f"gradmu_{tag}_interior"] = (float(g[bands["interior"]].mean())
                                                      if bands["interior"].any() else np.nan)
                    brow[f"gradmu_{tag}_contrast"] = (brow[f"gradmu_{tag}_interface"] -
                                                      brow[f"gradmu_{tag}_interior"])
                brow["gradmu_contrast_gain_q15_minus_q2"] = (
                    brow["gradmu_q15_contrast"] - brow["gradmu_q2_contrast"])
                for bl, arr in (("raw", grad_raw[sl]), ("fastgp", grad_fgp[sl])):
                    brow[f"grad_{bl}_interface"] = (float(arr[bands["interface"]].mean())
                                                    if bands["interface"].any() else np.nan)
                    brow[f"grad_{bl}_interior"] = (float(arr[bands["interior"]].mean())
                                                   if bands["interior"].any() else np.nan)
                    brow[f"grad_{bl}_contrast"] = (brow[f"grad_{bl}_interface"] -
                                                   brow[f"grad_{bl}_interior"])
                brows.append(brow)
                np.savez_compressed(
                    os.path.join(out, "predictions",
                                 f"{ds}_{p['gt_id_1']}_{p['gt_id_2']}_s{seed}.npz"),
                    mu_q2=mus[2.0], mu_q15=mus[1.5], delta=d, se_delta=se_d,
                    raw=sub, gt=gt[sl], pred=pred[sl],
                    band_interface=bands["interface"], band_interior=bands["interior"],
                    r0=r0, c0=c0, merged=p["merged"])
                print(f"  {ds} {p['gt_id_1']}+{p['gt_id_2']} "
                      f"{'MERGED ' if p['merged'] else 'separat'} seed={seed} "
                      f"iface={p['interface_len_px']:>3} | "
                      f"max|d|={row['max_abs_delta']:.4f} mean|d|={row['mean_abs_delta']:.4f} "
                      f"SNRmean={row['snr_mean']:.1f} loc_ratio="
                      f"{row['localization_ratio'] if np.isfinite(row['localization_ratio']) else float('nan'):.3f} "
                      f"gradcontrast q2={brow['gradmu_q2_contrast']:.4f} "
                      f"q15={brow['gradmu_q15_contrast']:.4f}")

    pd.DataFrame(inv_rows).to_csv(os.path.join(out, "patch_inventory.csv"), index=False)
    pd.DataFrame(prows).to_csv(os.path.join(out, "paired_run_metrics.csv"), index=False)
    pd.DataFrame(mrows).to_csv(os.path.join(out, "mc_error_metrics.csv"), index=False)
    pd.DataFrame(brows).to_csv(os.path.join(out, "boundary_mean_metrics.csv"), index=False)

    # seed variability (Stage 2 only)
    pr = pd.DataFrame(prows)
    if len(seeds) > 1:
        for (ds, a, b), g in pr.groupby(["dataset", "gt_id_1", "gt_id_2"]):
            srows.append(dict(dataset=ds, gt_id_1=a, gt_id_2=b, n_seeds=len(g),
                              q_effect_mean_abs_delta=float(g.mean_abs_delta.mean()),
                              q_effect_sd_across_seeds=float(g.mean_abs_delta.std(ddof=1))))
    pd.DataFrame(srows).to_csv(os.path.join(out, "seed_variability.csv"), index=False)
    json.dump(dict(stage=stage, patch=PATCH, iters=ITERS, num_inducing=NIND,
                   hidden_dims=HID, n_mc=N_MC, n_mc_batches=N_BATCH,
                   seeds=seeds, q_pair=list(Q_PAIR),
                   standardization="per-patch z-score, identical for both q; "
                                   "Delta_mu reported back in raw intensity units",
                   env=env_info()),
              open(os.path.join(out, "model_configs", f"stage{stage}.json"), "w"),
              indent=2)
    return pr, pd.DataFrame(brows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--stage", type=int, default=1)
    a = ap.parse_args()
    pr, br = run(a.out, a.stage)
    print("\n" + "=" * 124)
    print(f"STAGE {a.stage} AGGREGATE")
    print("=" * 124)
    for ds, g in pr.groupby("dataset"):
        gm = g[g.merged]; gs = g[~g.merged]
        print(f"\n  {ds}")
        print(f"    mean|Delta_mu| all={g.mean_abs_delta.mean():.5f}  "
              f"merged={gm.mean_abs_delta.mean():.5f}  sep={gs.mean_abs_delta.mean():.5f}")
        print(f"    MC SE(delta)  mean={g.mc_se_mean.mean():.6f}   "
              f"SNR_mean={g.snr_mean.mean():.2f}  SNR_max={g.snr_max.mean():.2f}")
        print(f"    spatial non-uniformity sd/mean = {g.nonuniform_sd_over_mean.mean():.3f}")
        print(f"    localization ratio (iface/interior) = "
              f"{g.localization_ratio.mean():.3f}   "
              f"edge/iface = {g.edge_over_interface.mean():.3f}")
        b = br[br.dataset == ds]; bm = b[b.merged]
        print(f"    MERGED gradmu contrast  q2={bm.gradmu_q2_contrast.mean():.5f}  "
              f"q15={bm.gradmu_q15_contrast.mean():.5f}  "
              f"gain={bm.gradmu_contrast_gain_q15_minus_q2.mean():+.5f}")
        print(f"    MERGED baselines        raw={bm.grad_raw_contrast.mean():.5f}  "
              f"fastgp={bm.grad_fastgp_contrast.mean():.5f}")
