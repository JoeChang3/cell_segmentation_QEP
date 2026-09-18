"""
Task A2: Python side of the REAL-TILE parity test against the original R Fast-GP.

Reads the tile that `audits/taskA2_real_tile_r_side.R` dumped, so both sides see
bit-identical pixels in an identical orientation. Compares:

  beta1, beta2, nugget nu, theta_hat (local mean), S_2 / sigma2_hat (local scale),
  predictive-mean RMSE, max abs predictive-mean difference, and the correlation
  between the two predictive means.

Two optimizers are reported because the original real-data function uses
L-BFGS-B (Modified_Functions_RGasp.R:106) while our ported `lattice_alg`
defaults to Nelder-Mead (the simulation function's choice). The difference is
material on real tiles and is one of the audit's findings.

Usage:  python audits/taskA2_real_tile_py_side.py [dataset ...]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.dim_2_lattice import lattice_alg
from py_core.paper_fast_gp import estimate_shared_params, reconstruct_tile

OUT = os.path.join(_HERE, "parity")


def compare(dataset: str) -> list:
    tag = os.path.join(OUT, f"real_{dataset}")
    tile = np.loadtxt(f"{tag}_r_tile.csv", delimiter=",")
    r_pm = np.loadtxt(f"{tag}_r_predmean.csv", delimiter=",")
    rp = pd.read_csv(f"{tag}_r_params.csv").iloc[0]
    n1, n2 = tile.shape

    print("=" * 100)
    print(f"REAL-TILE PARITY  dataset={dataset}  tile={n1}x{n2}  "
          f"intensity range [{tile.min():.6f}, {tile.max():.6f}]")
    print("=" * 100)
    print(f"  R: beta1={rp.beta1:.10f}  beta2={rp.beta2:.10f}  nu={rp.nugget_nu:.10f}")
    print(f"     theta_hat={rp.theta_hat:.10f}  S_2={rp.S_2:.6f}  "
          f"sigma2_hat={rp.sigma2_hat:.6e}")

    inp1 = np.linspace(0.0, 1.0, n1)
    inp2 = np.linspace(0.0, 1.0, n2)
    rows = []

    for method in ("L-BFGS-B", "Nelder-Mead"):
        t0 = time.time()
        out = lattice_alg(tile, inp1, inp2, kernel_type="matern",
                          param_ini=(-2.0, -2.0, -3.0), optim_method=method)
        dt = time.time() - t0
        b1, b2 = float(out["beta"][0]), float(out["beta"][1])
        nu = float(out["nu"])
        pm = np.asarray(out["pred_mean"], dtype=np.float64)
        d = pm - r_pm
        rows.append(dict(
            dataset=dataset, impl="lattice_alg", optim_method=method,
            n1=n1, n2=n2,
            r_beta1=rp.beta1, r_beta2=rp.beta2, r_nu=rp.nugget_nu,
            py_beta1=b1, py_beta2=b2, py_nu=nu,
            d_beta1=abs(b1 - rp.beta1), d_beta2=abs(b2 - rp.beta2),
            d_nu=abs(nu - rp.nugget_nu),
            rel_beta1=abs(b1 - rp.beta1) / abs(rp.beta1),
            rel_beta2=abs(b2 - rp.beta2) / abs(rp.beta2),
            predmean_maxabs=float(np.abs(d).max()),
            predmean_rmse=float(np.sqrt((d ** 2).mean())),
            predmean_corr=float(np.corrcoef(pm.ravel(), r_pm.ravel())[0, 1]),
            pct_of_range=100.0 * float(np.abs(d).max()) / float(r_pm.ptp()),
            runtime_sec=dt))
        print(f"\n  py lattice_alg / {method:<12} ({dt:.1f} s)")
        print(f"     beta1={b1:.10f}  beta2={b2:.10f}  nu={nu:.10f}")
        print(f"     |dbeta1|={abs(b1-rp.beta1):.3e} (rel {abs(b1-rp.beta1)/abs(rp.beta1):.3e})"
              f"  |dbeta2|={abs(b2-rp.beta2):.3e} (rel {abs(b2-rp.beta2)/abs(rp.beta2):.3e})"
              f"  |dnu|={abs(nu-rp.nugget_nu):.3e}")
        print(f"     predmean max|diff|={np.abs(d).max():.3e}  RMSE={np.sqrt((d**2).mean()):.3e}"
              f"  corr={np.corrcoef(pm.ravel(), r_pm.ravel())[0,1]:.12f}")

    # the actual paper_fast_gp path: split estimate / reconstruct, L-BFGS-B
    t0 = time.time()
    shared = estimate_shared_params(tile)
    t_est = time.time() - t0
    t0 = time.time()
    rec = reconstruct_tile(tile, shared)
    t_rec = time.time() - t0
    pm = rec.pred_mean
    d = pm - r_pm
    print(f"\n  py paper_fast_gp (estimate {t_est:.1f} s + reconstruct {t_rec:.1f} s)")
    print(f"     beta1={shared.beta1:.10f}  beta2={shared.beta2:.10f}  nu={shared.nugget:.10f}")
    print(f"     theta_hat={rec.theta_hat:.10f}  S_2={rec.s_2:.6f}  "
          f"sigma2_hat={rec.sigma2_hat:.6e}")
    print(f"     |dtheta_hat|={abs(rec.theta_hat-rp.theta_hat):.3e}  "
          f"|dS_2|={abs(rec.s_2-rp.S_2):.3e}")
    print(f"     predmean max|diff|={np.abs(d).max():.3e}  RMSE={np.sqrt((d**2).mean()):.3e}"
          f"  corr={np.corrcoef(pm.ravel(), r_pm.ravel())[0,1]:.12f}")
    rows.append(dict(
        dataset=dataset, impl="paper_fast_gp", optim_method="L-BFGS-B",
        n1=n1, n2=n2,
        r_beta1=rp.beta1, r_beta2=rp.beta2, r_nu=rp.nugget_nu,
        py_beta1=shared.beta1, py_beta2=shared.beta2, py_nu=shared.nugget,
        d_beta1=abs(shared.beta1 - rp.beta1), d_beta2=abs(shared.beta2 - rp.beta2),
        d_nu=abs(shared.nugget - rp.nugget_nu),
        rel_beta1=abs(shared.beta1 - rp.beta1) / abs(rp.beta1),
        rel_beta2=abs(shared.beta2 - rp.beta2) / abs(rp.beta2),
        r_theta_hat=rp.theta_hat, py_theta_hat=rec.theta_hat,
        d_theta_hat=abs(rec.theta_hat - rp.theta_hat),
        r_S_2=rp.S_2, py_S_2=rec.s_2, d_S_2=abs(rec.s_2 - rp.S_2),
        predmean_maxabs=float(np.abs(d).max()),
        predmean_rmse=float(np.sqrt((d ** 2).mean())),
        predmean_corr=float(np.corrcoef(pm.ravel(), r_pm.ravel())[0, 1]),
        pct_of_range=100.0 * float(np.abs(d).max()) / float(r_pm.ptp()),
        runtime_sec=t_est + t_rec))

    # decisive check: R's OWN params fed to our reconstruct => isolates the
    # optimizer from the linear algebra
    rec2 = reconstruct_tile(tile, shared.__class__(
        beta1=float(rp.beta1), beta2=float(rp.beta2), nugget=float(rp.nugget_nu),
        n1=n1, n2=n2, source_tile="R", neg_log_lik=float("nan"),
        optim_method="from_R", n_obj_evals=0, runtime_sec=0.0))
    d2 = rec2.pred_mean - r_pm
    print(f"\n  py reconstruct_tile GIVEN R's OWN params (isolates linear algebra "
          f"from the optimizer)")
    print(f"     |dtheta_hat|={abs(rec2.theta_hat-rp.theta_hat):.3e}  "
          f"|dS_2|={abs(rec2.s_2-rp.S_2):.3e}")
    print(f"     predmean max|diff|={np.abs(d2).max():.3e}  "
          f"RMSE={np.sqrt((d2**2).mean()):.3e}  "
          f"corr={np.corrcoef(rec2.pred_mean.ravel(), r_pm.ravel())[0,1]:.12f}")
    rows.append(dict(
        dataset=dataset, impl="reconstruct_given_R_params", optim_method="from_R",
        n1=n1, n2=n2, r_beta1=rp.beta1, r_beta2=rp.beta2, r_nu=rp.nugget_nu,
        py_beta1=rp.beta1, py_beta2=rp.beta2, py_nu=rp.nugget_nu,
        d_beta1=0.0, d_beta2=0.0, d_nu=0.0, rel_beta1=0.0, rel_beta2=0.0,
        r_theta_hat=rp.theta_hat, py_theta_hat=rec2.theta_hat,
        d_theta_hat=abs(rec2.theta_hat - rp.theta_hat),
        r_S_2=rp.S_2, py_S_2=rec2.s_2, d_S_2=abs(rec2.s_2 - rp.S_2),
        predmean_maxabs=float(np.abs(d2).max()),
        predmean_rmse=float(np.sqrt((d2 ** 2).mean())),
        predmean_corr=float(np.corrcoef(rec2.pred_mean.ravel(), r_pm.ravel())[0, 1]),
        pct_of_range=100.0 * float(np.abs(d2).max()) / float(r_pm.ptp())))
    return rows


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["nuclei"]
    allrows = []
    for ds in datasets:
        allrows += compare(ds)
    df = pd.DataFrame(allrows)
    p = os.path.join(OUT, "real_tile_parity_summary.csv")
    df.to_csv(p, index=False)
    print(f"\nwrote {p}")
