"""
Aggregate the cell-blob benchmark across inference modes and noise levels.

Reads the runs.csv produced by cell_blobs_qep_benchmark.py for each output
directory (one per inference mode) and produces:

  - a combined CSV
  - a markdown table per (inference, sigma) in the requested column order
  - a headline figure: edge-band RMSE and edge sharpness vs q, with the
    Gaussian case (q=2) marked, for every inference mode and sigma
  - an explicit, computed verdict on whether q<2 beats q=2 at boundaries

Run after the benchmark:
    python experiments/simulated/cell_blobs_aggregate_report.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = os.path.join(_ROOT, "results")
RUNS = [
    ("exact", os.path.join(RESULTS, "cell_blobs_qep_exact")),
    ("variational", os.path.join(RESULTS, "cell_blobs_qep_variational")),
]
OUT = os.path.join(RESULTS, "cell_blobs_qep_report")

TABLE_COLS = [
    ("power", "q"),
    ("rmse", "RMSE"),
    ("rel_l1", "rel L1"),
    ("rel_l2", "rel L2"),
    ("rel_linf", "rel Linf"),
    ("edge_band_rmse", "edge error"),
    ("boundary_f1", "boundary F1"),
]
EXTRA_COLS = [
    ("interior_rmse", "interior RMSE"),
    ("edge_interior_ratio", "edge/interior"),
    ("edge_sharpness_ratio", "sharpness"),
    ("lengthscale_0", "lengthscale"),
    ("noise", "learned noise"),
    # For EXACT inference the posterior mean is a function of the lengthscale
    # and the outputscale/noise ratio ONLY (it is provably q-invariant at fixed
    # hyperparameters -- see qep_power_semantics_check.py, claim 4). So these
    # two columns fully explain any apparent q effect in the exact arms.
    ("snr_ratio", "outputscale/noise"),
    ("tail_improve_frac", "tail improve"),
    ("runtime_s", "runtime s"),
]


def load() -> pd.DataFrame:
    frames = []
    for name, d in RUNS:
        p = os.path.join(d, "runs.csv")
        if not os.path.exists(p):
            print(f"  [skip] {p} not found")
            continue
        df = pd.read_csv(p)
        df["inference_mode"] = name
        frames.append(df)
        print(f"  [load] {name}: {len(df)} rows from {p}")
    if not frames:
        raise SystemExit("No runs.csv found. Run cell_blobs_qep_benchmark.py first.")
    return pd.concat(frames, ignore_index=True)


def fmt(v: float, nd: int = 4) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print("Loading runs...")
    df = load()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["snr_ratio"] = df["outputscale"] / df["noise"]
    df.to_csv(os.path.join(OUT, "all_runs_combined.csv"), index=False)

    ok = df[df["failed"] == 0].copy()
    key = ["inference_mode", "sigma", "arm", "label", "power"]
    # exclude the grouping keys from the aggregated columns, otherwise
    # reset_index() collides with the numeric keys (sigma, power)
    num = [c for c in ok.select_dtypes(include=[np.number]).columns
           if c not in key]
    agg = ok.groupby(key, dropna=False)[num].mean().reset_index()
    nseed = ok.groupby(key, dropna=False).size().rename("n_seeds").reset_index()
    agg = agg.merge(nseed, on=key)
    agg.to_csv(os.path.join(OUT, "summary_combined.csv"), index=False)

    lines: List[str] = []
    lines.append("# Cell-blob benchmark: does q<2 preserve sharp boundaries "
                 "better than q=2?\n")
    lines.append("`q=2.0` is the exact Gaussian case (verified: `power=2.0` "
                 "reproduces gpytorch's `log_prob` and `expected_log_prob` "
                 "bit-for-bit). `q=3.0` is a mechanism probe outside the "
                 "standard Q-EP range q in (0,2]. `identity_noisy` = no "
                 "smoothing at all.\n")
    lines.append("Lower is better for all error columns; higher is better for "
                 "boundary F1. `sharpness` is |grad pred| / |grad truth| inside "
                 "the edge band, where 1.0 = edge steepness matched and <1 = "
                 "blurred.\n")

    verdicts: List[Dict] = []

    for mode, _ in RUNS:
        sub_mode = agg[agg["inference_mode"] == mode]
        if sub_mode.empty:
            continue
        for sig in sorted(sub_mode["sigma"].unique()):
            s = (sub_mode[sub_mode["sigma"] == sig]
                 .sort_values("power", ascending=False, na_position="last"))
            nse = int(s["n_seeds"].max())
            lines.append(f"\n## inference = {mode}, sigma = {sig} "
                         f"(mean of {nse} seed(s))\n")
            hdr = "| arm | " + " | ".join(h for _, h in TABLE_COLS[1:]) + " | " \
                  + " | ".join(h for _, h in EXTRA_COLS) + " |"
            sep = "|---" * (1 + len(TABLE_COLS) - 1 + len(EXTRA_COLS)) + "|"
            lines.append(hdr)
            lines.append(sep)
            for _, r in s.iterrows():
                qlab = ("identity" if r["arm"] == "identity_noisy"
                        else "GP control" if not np.isfinite(r["power"])
                        else f"q={r['power']:.1f}"
                             + (" (probe)" if r["power"] > 2 else ""))
                cells = [fmt(r[c]) for c, _ in TABLE_COLS[1:]]
                cells += [fmt(r[c]) for c, _ in EXTRA_COLS]
                lines.append(f"| {qlab} | " + " | ".join(cells) + " |")

            q = s[s["power"].notna() & (s["arm"] != "identity_noisy")]
            if (q["power"] == 2.0).any():
                base = q[q["power"] == 2.0].iloc[0]
                lt = q[q["power"] < 2.0]
                if not lt.empty:
                    be = lt.loc[lt["edge_band_rmse"].idxmin()]
                    wins = bool(be["edge_band_rmse"] < base["edge_band_rmse"])
                    verdicts.append(dict(
                        inference=mode, sigma=sig,
                        best_q=float(be["power"]),
                        best_edge=float(be["edge_band_rmse"]),
                        base_edge=float(base["edge_band_rmse"]),
                        q_lt_2_wins=wins,
                        best_sharp=float(be["edge_sharpness_ratio"]),
                        base_sharp=float(base["edge_sharpness_ratio"]),
                    ))
                    lines.append(
                        f"\n**Verdict (sigma={sig}, {mode}):** best q<2 on "
                        f"edge-band RMSE is q={be['power']:.1f} at "
                        f"{be['edge_band_rmse']:.4f}, vs q=2.0 at "
                        f"{base['edge_band_rmse']:.4f} -> "
                        + ("**q<2 wins at boundaries**" if wins
                           else "**q=2 is as good or better**") + ".")

    if verdicts:
        vdf = pd.DataFrame(verdicts)
        vdf.to_csv(os.path.join(OUT, "verdicts.csv"), index=False)
        n_win = int(vdf["q_lt_2_wins"].sum())
        lines.append("\n## Overall\n")
        lines.append(f"q<2 beat q=2 on edge-band RMSE in **{n_win} of "
                     f"{len(vdf)}** (inference, sigma) conditions.\n")
        lines.append("| inference | sigma | best q<2 | edge err (best q<2) | "
                     "edge err (q=2) | q<2 wins? | sharpness q<2 | sharpness q=2 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in vdf.iterrows():
            lines.append(f"| {r['inference']} | {r['sigma']} | "
                         f"{r['best_q']:.1f} | {r['best_edge']:.4f} | "
                         f"{r['base_edge']:.4f} | "
                         f"{'YES' if r['q_lt_2_wins'] else 'NO'} | "
                         f"{r['best_sharp']:.3f} | {r['base_sharp']:.3f} |")

    md = os.path.join(OUT, "REPORT.md")
    with open(md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {md}")

    # ── headline figure ──
    modes = [m for m, _ in RUNS if not agg[agg["inference_mode"] == m].empty]
    sigmas = sorted(agg["sigma"].unique())
    if modes and sigmas:
        fig, axs = plt.subplots(2, len(modes) * len(sigmas),
                                figsize=(5.0 * len(modes) * len(sigmas), 8.0),
                                squeeze=False)
        col = 0
        for mode in modes:
            for sig in sigmas:
                s = agg[(agg["inference_mode"] == mode) & (agg["sigma"] == sig)]
                q = (s[s["power"].notna() & (s["arm"] != "identity_noisy")]
                     .sort_values("power"))
                ident = s[s["arm"] == "identity_noisy"]
                gpc = s[(s["arm"] == "gp_control")]
                if q.empty:
                    col += 1
                    continue
                for row, (metric, name, better) in enumerate([
                        ("edge_band_rmse", "edge-band RMSE", "lower better"),
                        ("edge_sharpness_ratio",
                         "edge sharpness  |grad pred|/|grad truth|", "1.0 = matched")]):
                    ax = axs[row][col]
                    ax.plot(q["power"], q[metric], "o-", lw=2, ms=7, color="C0",
                            label="QEP arms", zorder=3)
                    m2 = q[q["power"] == 2.0]
                    if not m2.empty:
                        ax.plot(m2["power"], m2[metric], "s", ms=13, mfc="none",
                                mec="C3", mew=2.4, label="q=2 (Gaussian)", zorder=4)
                    probe = q[q["power"] > 2.0]
                    if not probe.empty:
                        ax.plot(probe["power"], probe[metric], "D", ms=9,
                                color="C4", label="q>2 (probe)", zorder=4)
                    if not gpc.empty:
                        ax.axhline(float(gpc.iloc[0][metric]), color="C3", ls="--",
                                   lw=1.2, label="independent GP control")
                    if not ident.empty:
                        ax.axhline(float(ident.iloc[0][metric]), color="0.45",
                                   ls=":", lw=1.6, label="no smoothing")
                    if row == 1:
                        ax.axhline(1.0, color="green", ls="-", lw=0.9, alpha=0.5)
                    ax.set_xlabel("q (POWER)")
                    ax.set_title(f"{mode}, $\\sigma$={sig}\n{name} ({better})",
                                 fontsize=10)
                    ax.grid(alpha=0.3)
                    if col == 0 and row == 0:
                        ax.legend(fontsize=8)
                col += 1
        fig.suptitle("Does q<2 preserve sharp cell boundaries better than q=2?  "
                     "(cell-blob benchmark)", fontweight="bold", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        png = os.path.join(OUT, "headline_edge_metrics_vs_q.png")
        fig.savefig(png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png}")

    print(f"Wrote {os.path.join(OUT, 'all_runs_combined.csv')}")
    print(f"Wrote {os.path.join(OUT, 'summary_combined.csv')}")


if __name__ == "__main__":
    main()
