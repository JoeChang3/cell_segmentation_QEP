"""
Task A2 follow-up: why do R and Python disagree on the REAL tile?

The real-tile parity test showed:
  - given R's own (beta1, beta2, nu), our reconstruct_tile matches R's predictive
    mean to max|diff| = 5.3e-15, so the LINEAR ALGEBRA port is exact;
  - but the two optimizers land on different parameters, and R's point scores
    129310.4853 while Python's scores 116922.5219 under R's OWN objective code.

So the disagreement is an OPTIMIZATION issue on a non-convex objective, not a
translation error. This script characterizes the landscape:

  1. multi-start from many initializations, to count basins and find the best
     point actually reachable;
  2. a profile of the objective along log beta2 at the other two parameters
     fixed, to exhibit the beta2 -> infinity plateau directly;
  3. reconstruction quality (RMSE to raw, correlation) at each basin, so the
     practical consequence of picking the wrong basin is visible.

Writes audits/parity/real_tile_landscape_{multistart,profile}.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_core.dim_2_lattice import neg_log_lik_eigen_with_nugget
from py_core.paper_fast_gp import SharedParams, reconstruct_tile

OUT = os.path.join(_HERE, "parity")
PARAM_INI = np.array([-2.0, -2.0, -3.0])
R_POINT = np.array([np.log(26.0388376235), np.log(5679456.5183289703),
                    np.log(0.0822243107)])


def main(dataset: str = "nuclei") -> None:
    tile = np.loadtxt(os.path.join(OUT, f"real_{dataset}_r_tile.csv"), delimiter=",")
    r_pm = np.loadtxt(os.path.join(OUT, f"real_{dataset}_r_predmean.csv"), delimiter=",")
    n1, n2 = tile.shape
    N = n1 * n2
    i1, i2 = np.linspace(0, 1, n1), np.linspace(0, 1, n2)
    R01 = np.abs(np.subtract.outer(i1, i1))
    R02 = np.abs(np.subtract.outer(i2, i2))
    XL = [np.ones((n1, n2))]

    def f(p):
        try:
            v = neg_log_lik_eigen_with_nugget(np.asarray(p, float), "matern",
                                              R01, R02, N, 1, XL, tile)
        except Exception:
            return np.inf
        return float(v) if np.isfinite(v) else np.inf

    def quality(p):
        b = np.exp(np.asarray(p, float))
        sp = SharedParams(beta1=float(b[0]), beta2=float(b[1]), nugget=float(b[2]),
                          n1=n1, n2=n2, source_tile="probe", neg_log_lik=f(p),
                          optim_method="probe", n_obj_evals=0, runtime_sec=0.0)
        rec = reconstruct_tile(tile, sp)
        pm = rec.pred_mean
        return (float(np.sqrt(((pm - tile) ** 2).mean())),
                float(np.corrcoef(pm.ravel(), tile.ravel())[0, 1]),
                float(np.abs(pm - r_pm).max()))

    # ---- 1. multi-start ----------------------------------------------------
    print("=" * 108)
    print(f"1. MULTI-START  tile {n1}x{n2}   (objective = R's negative profiled log-lik)")
    print("=" * 108)
    starts = [("paper param_ini (-2,-2,-3)", PARAM_INI),
              ("R's converged point", R_POINT),
              ("(0,0,-3)", np.array([0., 0., -3.])),
              ("(1,1,-2)", np.array([1., 1., -2.])),
              ("(2,2,-2)", np.array([2., 2., -2.])),
              ("(3,3,-2)", np.array([3., 3., -2.])),
              ("(4,4,-1)", np.array([4., 4., -1.])),
              ("(3.26,3.16,-1.77) near PY opt", np.array([3.2596, 3.1646, -1.7686])),
              ("(-4,-4,-4)", np.array([-4., -4., -4.])),
              ("(5,5,0)", np.array([5., 5., 0.]))]
    rng = np.random.default_rng(7)
    for k in range(6):
        starts.append((f"random {k}", rng.uniform(-3, 5, 3)))

    rows = []
    print(f"  {'start':<32}{'f(start)':>13}{'f(opt)':>13}{'beta1':>11}"
          f"{'beta2':>14}{'nu':>10}{'RMSEraw':>10}{'corrRaw':>9}{'vs R pm':>10}")
    for name, x0 in starts:
        res = minimize(f, x0, method="L-BFGS-B")
        rmse, corr, vsr = quality(res.x)
        b = np.exp(res.x)
        rows.append(dict(dataset=dataset, start=name, f_start=f(x0), f_opt=float(res.fun),
                         beta1=b[0], beta2=b[1], nu=b[2], rmse_to_raw=rmse,
                         corr_to_raw=corr, maxabs_vs_r_predmean=vsr,
                         nfev=int(res.nfev), success=bool(res.success)))
        print(f"  {name:<32}{f(x0):>13.2f}{float(res.fun):>13.2f}{b[0]:>11.4f}"
              f"{b[1]:>14.4f}{b[2]:>10.5f}{rmse:>10.5f}{corr:>9.5f}{vsr:>10.2e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"real_tile_landscape_multistart_{dataset}.csv"),
              index=False)
    best = df.loc[df.f_opt.idxmin()]
    print(f"\n  best found: f={best.f_opt:.4f} at beta1={best.beta1:.5f} "
          f"beta2={best.beta2:.5f} nu={best.nu:.6f}  (start: {best.start})")
    plateau = df[df.beta2 > 1e4]
    print(f"  starts landing in the beta2->inf plateau: {len(plateau)}/{len(df)}"
          + (f"  (all at f ~ {plateau.f_opt.min():.4f})" if len(plateau) else ""))

    # ---- 2. profile along log beta2 ---------------------------------------
    print("\n" + "=" * 108)
    print("2. OBJECTIVE PROFILE along log(beta2), with beta1 and nu at R's values")
    print("=" * 108)
    prows = []
    print(f"  {'log beta2':>11}{'beta2':>14}{'f':>14}{'  note':<28}")
    for lb2 in [-2, -1, 0, 1, 2, 2.5, 3, 3.1646, 3.5, 4, 5, 6, 8, 10, 12,
                15.5524, 18, 21]:
        p = np.array([R_POINT[0], float(lb2), R_POINT[2]])
        v = f(p)
        note = ""
        if abs(lb2 - 15.5524) < 1e-3:
            note = "<- R's converged beta2"
        elif abs(lb2 - 3.1646) < 1e-3:
            note = "<- Python's beta2"
        prows.append(dict(dataset=dataset, log_beta2=float(lb2),
                          beta2=float(np.exp(lb2)), f=v, note=note))
        print(f"  {lb2:>11.4f}{np.exp(lb2):>14.4e}{v:>14.4f}  {note:<28}")
    pd.DataFrame(prows).to_csv(
        os.path.join(OUT, f"real_tile_landscape_profile_{dataset}.csv"), index=False)

    # ---- 3. the two competing fits, side by side --------------------------
    print("\n" + "=" * 108)
    print("3. CONSEQUENCE: reconstruction at R's basin vs the better basin")
    print("=" * 108)
    bp = np.log(np.array([best.beta1, best.beta2, best.nu]))
    for name, p in [("R's basin  (beta2 -> inf)", R_POINT), ("better basin", bp)]:
        rmse, corr, vsr = quality(p)
        b = np.exp(p)
        eff1 = (1.0 / b[0]) * (n1 - 1)
        eff2 = (1.0 / b[1]) * (n2 - 1)
        print(f"  {name:<28} f={f(p):>12.2f}  eff. range rows={eff1:>8.3f} px  "
              f"cols={eff2:>10.3e} px  RMSE_to_raw={rmse:.5f}  corr_to_raw={corr:.5f}")
    print("\n  An effective column range far below 1 pixel means the fitted process")
    print("  is uncorrelated across columns: the reconstruction smooths along rows")
    print("  only. That is a degenerate, axis-aligned smoother, not an isotropic one.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nuclei")
