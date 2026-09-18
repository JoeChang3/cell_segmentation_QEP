"""
Is any q<2 "edge advantage" reproducible by a Gaussian with the same hyperparameters?

This is the capstone control for the cell-blob benchmark. It closes the loop
between the structural claim and the measured numbers.

The structural claim (qep_power_semantics_check.py, claim 4) is that an exact
QEP posterior mean is q-invariant at fixed hyperparameters, because an
elliptically-contoured process has the same LINEAR conditional mean as the
Gaussian with the same covariance:

    m = mu + K (K + s^2 I)^{-1} (y - mu)          <- contains no q

So whenever a q<2 arm appears to beat q=2 at boundaries, the difference must be
fully attributable to the (lengthscale, outputscale, noise) that its marginal
likelihood happened to select -- and a Gaussian handed those same three numbers
must return the identical image.

This script takes the hyperparameters each arm actually selected in the
completed exact benchmark run and verifies exactly that, per arm and per sigma.

If the max pixel difference is 0 for every arm, then q contributes nothing that
tuning the Gaussian case cannot already reach, and the apparent q<2 edge gain is
just a near-interpolating GP (noise driven toward 0 by the unbounded q<2
likelihood -- see claim 5).

Run after the exact benchmark:
    python experiments/simulated/cell_blobs_gp_equivalence_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

import gpytorch
import qpytorch
import qpytorch.kernels as QK
import qpytorch.means as QM
from qpytorch.distributions import MultivariateQExponential
from qpytorch.likelihoods import QExponentialLikelihood

from experiments.simulated.cell_blobs_qep_benchmark import (
    add_noise,
    generate_cell_blobs,
)
from py_core.edge_metrics import evaluate_reconstruction

RUNS_CSV = os.path.join(_ROOT, "results", "cell_blobs_qep_exact", "runs.csv")
OUT_CSV = os.path.join(_ROOT, "results", "cell_blobs_qep_report",
                       "gp_equivalence_check.csv")


class _ExactQEP(qpytorch.models.ExactQEP):
    def __init__(self, x, y, lik, power):
        super().__init__(x, y, lik)
        self.power = power
        self.mean_module = QM.ConstantMean()
        self.covar_module = QK.ScaleKernel(QK.MaternKernel(nu=2.5, ard_num_dims=2))

    def forward(self, x):
        return MultivariateQExponential(
            self.mean_module(x), self.covar_module(x), power=self.power)


def predict_with_fixed_hypers(X, Y, shape, q, ls, outputscale, noise):
    """Exact posterior mean with hyperparameters pinned to given values."""
    P = torch.tensor(float(q))
    lik = QExponentialLikelihood(power=P)
    lik.noise = torch.tensor(float(noise))
    m = _ExactQEP(X, Y, lik, P)
    m.covar_module.base_kernel.lengthscale = torch.tensor([[float(ls), float(ls)]])
    m.covar_module.outputscale = torch.tensor(float(outputscale))
    with torch.no_grad():
        m.mean_module.constant.fill_(float(Y.mean()))
    m.eval(); lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.debug(False):
        return m(X).mean.numpy().reshape(shape)


def main() -> None:
    if not os.path.exists(RUNS_CSV):
        raise SystemExit(f"{RUNS_CSV} not found. Run the exact benchmark first:\n"
                         "  python experiments/simulated/cell_blobs_qep_benchmark.py "
                         "--inference exact --tag exact")
    runs = pd.read_csv(RUNS_CSV)
    grid = int(runs["n_train"].dropna().iloc[0] ** 0.5)
    data = generate_cell_blobs(grid, grid)

    g = torch.linspace(0, 1, grid)
    rv, cv = torch.meshgrid(g, g, indexing="ij")
    X = torch.stack([rv.reshape(-1), cv.reshape(-1)], -1).contiguous()

    print("=" * 92)
    print("Does a Gaussian (q=2) with the SAME hyperparameters reproduce each "
          "q<2 arm's image?")
    print("=" * 92)
    print(f"grid {grid}x{grid}, exact inference, Matern nu=2.5, "
          f"hyperparameters taken from {os.path.relpath(RUNS_CSV, _ROOT)}\n")

    rows = []
    arms = runs[(runs["failed"] == 0) & runs["power"].notna()]
    for (sigma, seed), grp in arms.groupby(["sigma", "seed"]):
        y = add_noise(data.truth, float(sigma), int(seed))
        Y = torch.from_numpy(y.ravel())
        print(f"--- sigma={sigma}  seed={seed} " + "-" * 60)
        hdr = (f"  {'q':>5}{'lengthscale':>13}{'outputscale':>13}{'noise':>12}"
               f"{'snr':>10}{'RMSE(q)':>10}{'RMSE(q=2)':>11}"
               f"{'max|diff|':>12}  identical")
        print(hdr)
        for _, r in grp.sort_values("power", ascending=False).iterrows():
            q, ls = float(r["power"]), float(r["lengthscale_0"])
            os_, nz = float(r["outputscale"]), float(r["noise"])
            p_q = predict_with_fixed_hypers(X, Y, (grid, grid), q, ls, os_, nz)
            p_g = predict_with_fixed_hypers(X, Y, (grid, grid), 2.0, ls, os_, nz)
            dmax = float(np.abs(p_q - p_g).max())
            mq = evaluate_reconstruction(p_q, data.truth, data.labels, data.fg,
                                         fixed_threshold=data.fixed_threshold)
            mg = evaluate_reconstruction(p_g, data.truth, data.labels, data.fg,
                                         fixed_threshold=data.fixed_threshold)
            same = dmax < 1e-12
            print(f"  {q:>5.1f}{ls:>13.6f}{os_:>13.6g}{nz:>12.6g}"
                  f"{os_/nz:>10.1f}{mq['rmse']:>10.6f}{mg['rmse']:>11.6f}"
                  f"{dmax:>12.2e}  {'YES' if same else 'NO'}")
            rows.append(dict(
                sigma=sigma, seed=seed, q=q, lengthscale=ls, outputscale=os_,
                noise=nz, snr=os_ / nz,
                rmse_q=mq["rmse"], rmse_gaussian=mg["rmse"],
                edge_q=mq["edge_band_rmse"], edge_gaussian=mg["edge_band_rmse"],
                interior_q=mq["interior_rmse"],
                sharp_q=mq["edge_sharpness_ratio"],
                max_abs_diff=dmax, identical=bool(same)))
        print()

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n_same = int(out["identical"].sum())
    print("=" * 92)
    print(f"Pixel-identical to the Gaussian in {n_same} of {len(out)} arm/seed/sigma "
          f"combinations (max over all: {out['max_abs_diff'].max():.2e})")
    if n_same == len(out):
        print("\nCONCLUSION")
        print("Every q arm's reconstruction is reproduced EXACTLY by a Gaussian")
        print("process given the same (lengthscale, outputscale, noise). q therefore")
        print("contributes nothing to the reconstruction beyond selecting those")
        print("hyperparameters. Any q<2 'edge advantage' in this benchmark is a")
        print("near-interpolating Gaussian fit -- reachable directly by lowering the")
        print("GP noise, and obtained here only because the q<2 marginal likelihood")
        print("is unbounded as noise -> 0 (see qep_power_semantics_check.py claim 5).")
    print(f"\nWrote {OUT_CSV}")
    print("=" * 92)
    sys.exit(0 if n_same == len(out) else 1)


if __name__ == "__main__":
    main()
