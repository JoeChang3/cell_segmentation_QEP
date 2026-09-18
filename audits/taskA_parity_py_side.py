"""Task A4: numerical parity between the ORIGINAL R real-data Fast-GP
(separable_GP_param_est + separable_GP) and our Python py_core/dim_2_lattice.py
lattice_alg, on the identical tile the R script dumped.

The original real-data path uses L-BFGS-B and inputs seq(0,1); lattice_alg
defaults to Nelder-Mead. Both settings are tested so the optimizer's
contribution is separated from the model's.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from py_core.dim_2_lattice import lattice_alg

n = int(sys.argv[1]); d = sys.argv[2]
tile = np.loadtxt(f"{d}/r_tile_n{n}.csv", delimiter=",")
r_pm = np.loadtxt(f"{d}/r_predmean_n{n}.csv", delimiter=",")
import csv
with open(f"{d}/r_params_n{n}.csv") as fh:
    row = list(csv.DictReader(fh))[0]
rb1, rb2, rnu = float(row["beta1"]), float(row["beta2"]), float(row["nu"])

print(f"PY: tile {tile.shape} range [{tile.min():.6f}, {tile.max():.6f}]")
print(f"R : beta1={rb1:.10f} beta2={rb2:.10f} nu={rnu:.10f}\n")

# inputs exactly as the ORIGINAL separable_GP builds them
i1 = np.linspace(0.0, 1.0, tile.shape[0])
i2 = np.linspace(0.0, 1.0, tile.shape[1])

rows = []
for method in ["L-BFGS-B", "Nelder-Mead"]:
    t0 = time.time()
    res = lattice_alg(output_mat=tile, input1=i1, input2=i2,
                      kernel_type="matern", param_ini=(-2.0, -2.0, -3.0),
                      optim_method=method)
    dt = time.time() - t0
    b1, b2 = res["beta"]; nu = res["nu"]; pm = res["pred_mean"]
    dmax = float(np.abs(pm - r_pm).max())
    drmse = float(np.sqrt(np.mean((pm - r_pm) ** 2)))
    rng = float(r_pm.max() - r_pm.min())
    print(f"PY[{method}]  beta1={b1:.10f} beta2={b2:.10f} nu={nu:.10f}  ({dt:.1f}s)")
    print(f"    |dbeta1|={abs(b1-rb1):.3e}  |dbeta2|={abs(b2-rb2):.3e}  |dnu|={abs(nu-rnu):.3e}")
    print(f"    predmean: max|diff|={dmax:.3e}  RMSE={drmse:.3e}  "
          f"as % of R predmean range ({rng:.4f}): {100*dmax/rng:.4f}%")
    rows.append(dict(method=method, beta1=b1, beta2=b2, nu=nu,
                     d_beta1=abs(b1-rb1), d_beta2=abs(b2-rb2), d_nu=abs(nu-rnu),
                     predmean_max_abs_diff=dmax, predmean_rmse=drmse,
                     pct_of_range=100*dmax/rng, seconds=dt))

# also: predictive mean with R's EXACT fitted parameters plugged in, which
# isolates the prediction algebra from the optimizer entirely
res_fix = lattice_alg(output_mat=tile, input1=i1, input2=i2, kernel_type="matern",
                      param_ini=(np.log(rb1), np.log(rb2), np.log(rnu)),
                      optim_method="L-BFGS-B")
print(f"\nPY[R params as init, refit] beta1={res_fix['beta'][0]:.10f} "
      f"beta2={res_fix['beta'][1]:.10f} nu={res_fix['nu']:.10f}")

import pandas as pd
pd.DataFrame(rows).to_csv(f"{d}/parity_summary_n{n}.csv", index=False)
print(f"\nwrote {d}/parity_summary_n{n}.csv")
