"""Round 10 analysis: criterion-by-criterion evaluation of the preregistered rules.

The decisive control is criterion 4: the PAIRED q effect
  |mu_q15(seed s) - mu_q2(seed s)|
must be compared against seed-to-seed variation WITHIN the same q
  |mu_q2(seed s) - mu_q2(seed s')|
computed on the identical patch and query grid.
"""
from __future__ import annotations
import argparse, glob, itertools, json, math, os, re
import numpy as np, pandas as pd


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg): return float("nan")
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float)
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)-1)/2) / (len(pos)*len(neg)))


def cohend(a, b):
    a, b = np.asarray(a,float), np.asarray(b,float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<2 or len(b)<2: return float("nan")
    s=math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/s) if s>0 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = a.out
    pr = pd.read_csv(os.path.join(out, "paired_run_metrics.csv"))
    br = pd.read_csv(os.path.join(out, "boundary_mean_metrics.csv"))

    # ---- criterion 4: paired q effect vs same-q seed variation -----------
    rows = []
    files = glob.glob(os.path.join(out, "predictions", "*_s*.npz"))
    key = {}
    for f in files:
        m = re.match(r"(.+)_(\d+)_(\d+)_s(\d+)\.npz$", os.path.basename(f))
        if m: key.setdefault((m.group(1), m.group(2), m.group(3)), {})[int(m.group(4))] = f
    for (ds, a1, a2), sd in key.items():
        seeds = sorted(sd)
        if len(seeds) < 2: continue
        dat = {s: np.load(sd[s]) for s in seeds}
        qeff = [float(np.abs(dat[s]["mu_q15"] - dat[s]["mu_q2"]).mean()) for s in seeds]
        sv2 = [float(np.abs(dat[s]["mu_q2"] - dat[t]["mu_q2"]).mean())
               for s, t in itertools.combinations(seeds, 2)]
        sv15 = [float(np.abs(dat[s]["mu_q15"] - dat[t]["mu_q15"]).mean())
                for s, t in itertools.combinations(seeds, 2)]
        rows.append(dict(dataset=ds, gt_id_1=int(a1), gt_id_2=int(a2),
                         merged=bool(dat[seeds[0]]["merged"]), n_seeds=len(seeds),
                         paired_q_effect_mean=float(np.mean(qeff)),
                         seedvar_same_q2_mean=float(np.mean(sv2)),
                         seedvar_same_q15_mean=float(np.mean(sv15)),
                         q_over_seedvar=float(np.mean(qeff) /
                                              max(np.mean(sv2 + sv15), 1e-300))))
    sv = pd.DataFrame(rows)
    sv.to_csv(os.path.join(out, "seed_variability.csv"), index=False)

    print("=" * 124)
    print("CRITERION 4  paired q effect  vs  seed-to-seed variation within the SAME q")
    print("=" * 124)
    print(f"  {'dataset':<12}{'pairs':>6}{'paired q effect':>17}"
          f"{'same-q seed var':>17}{'ratio q/seed':>14}")
    for ds, g in sv.groupby("dataset"):
        r = g.paired_q_effect_mean.mean() / max(
            np.mean([g.seedvar_same_q2_mean.mean(), g.seedvar_same_q15_mean.mean()]), 1e-300)
        print(f"  {ds:<12}{len(g):>6}{g.paired_q_effect_mean.mean():>17.5f}"
              f"{np.mean([g.seedvar_same_q2_mean.mean(), g.seedvar_same_q15_mean.mean()]):>17.5f}"
              f"{r:>14.3f}")

    # ---- Part 8 D/E : mean-gradient discrimination ----------------------
    print("\n" + "=" * 124)
    print("PART 8 D/E  boundary quality of the PREDICTIVE MEAN  (pair-level)")
    print("=" * 124)
    res = []
    for ds, g in br.groupby("dataset"):
        gm = g[g.merged]
        print(f"\n  {ds}   merged pairs x seeds = {len(gm)}")
        print(f"    {'statistic':<26}{'iface':>10}{'interior':>10}{'contrast':>10}"
              f"{'cohen d':>9}{'AUC':>8}")
        for tag, lab in (("gradmu_q2","|grad mu| q=2"), ("gradmu_q15","|grad mu| q=1.5"),
                         ("grad_raw","baseline |grad raw|"),
                         ("grad_fastgp","baseline |grad fastgp|")):
            pos = gm[f"{tag}_interface"].values; neg = gm[f"{tag}_interior"].values
            c = float(np.nanmean(pos) - np.nanmean(neg))
            d = cohend(pos, neg); A = auc(pos, neg)
            res.append(dict(dataset=ds, statistic=lab, mean_interface=float(np.nanmean(pos)),
                            mean_interior=float(np.nanmean(neg)), contrast=c,
                            cohen_d=d, auc=A, n=len(gm)))
            print(f"    {lab:<26}{np.nanmean(pos):>10.4f}{np.nanmean(neg):>10.4f}"
                  f"{c:>10.4f}{d:>9.3f}{A:>8.4f}")
        a2 = [r for r in res if r["dataset"]==ds and r["statistic"]=="|grad mu| q=2"][0]
        a15= [r for r in res if r["dataset"]==ds and r["statistic"]=="|grad mu| q=1.5"][0]
        print(f"    q-SPECIFIC GAIN  AUC {a15['auc']-a2['auc']:+.4f}   "
              f"contrast {a15['contrast']-a2['contrast']:+.4f}   "
              f"cohen d {a15['cohen_d']-a2['cohen_d']:+.3f}")
    pd.DataFrame(res).to_csv(os.path.join(out, "tables_discrimination.csv"), index=False)

    # ---- summary json ---------------------------------------------------
    summ = {}
    for ds, g in pr.groupby("dataset"):
        b = br[br.dataset == ds]; bm = b[b.merged]
        s = sv[sv.dataset == ds]
        a2 = [r for r in res if r["dataset"]==ds and r["statistic"]=="|grad mu| q=2"][0]
        a15= [r for r in res if r["dataset"]==ds and r["statistic"]=="|grad mu| q=1.5"][0]
        summ[ds] = dict(
            mean_abs_delta=float(g.mean_abs_delta.mean()),
            mc_se=float(g.mc_se_mean.mean()), snr_mean=float(g.snr_mean.mean()),
            nonuniformity=float(g.nonuniform_sd_over_mean.mean()),
            localization_ratio=float(g.localization_ratio.mean()),
            edge_over_interface=float(g.edge_over_interface.mean()),
            seed_variation=float(np.mean([s.seedvar_same_q2_mean.mean(),
                                          s.seedvar_same_q15_mean.mean()])),
            q_over_seedvar=float(g.mean_abs_delta.mean() /
                max(np.mean([s.seedvar_same_q2_mean.mean(),
                             s.seedvar_same_q15_mean.mean()]),1e-300)),
            merged_auc_q2=a2["auc"], merged_auc_q15=a15["auc"],
            q_specific_auc_gain=a15["auc"]-a2["auc"],
            merged_contrast_q2=a2["contrast"], merged_contrast_q15=a15["contrast"],
            q_specific_contrast_gain=a15["contrast"]-a2["contrast"],
            baseline_raw_auc=[r for r in res if r["dataset"]==ds and
                              r["statistic"]=="baseline |grad raw|"][0]["auc"],
            baseline_fastgp_auc=[r for r in res if r["dataset"]==ds and
                                 r["statistic"]=="baseline |grad fastgp|"][0]["auc"])
    json.dump(summ, open(os.path.join(out, "final_summary.json"), "w"), indent=2)
    print(f"\nwrote seed_variability.csv, tables_discrimination.csv, final_summary.json")


if __name__ == "__main__":
    main()
