"""
Task B6, claim 6: is the q<2 type-II marginal likelihood actually UNBOUNDED?

The earlier claim (experiments/simulated/qep_power_semantics_check.py, claim 5)
was that for q<2 the q-EP marginal log-likelihood DIVERGES to +inf as the fit
approaches interpolation, so type-II MLE is ill-posed with its optimum at
noise -> 0. The user's instruction is to distinguish

  (i)  a DENSITY SINGULARITY as the quadratic form r = (y-m)' C^-1 (y-m) -> 0
       (a statement about one ray through parameter space), from
  (ii) a proof that the actual hyperparameter marginal likelihood is GLOBALLY
       unbounded above.

Analytic setup. From qpytorch/distributions/multivariate_qexponential.py::log_prob

    L = -0.5*[ r**(q/2) + logdet(C) + d*log(2*pi) ]
        + 0.5*d*(q/2 - 1)*log(r) + log(q/2)          (the q != 2 branch)

Write C = s*C0 with s>0 a pure scale. Then r = r0/s and logdet = d*log s + ldet0,
so the s-dependence collects to

    L(s) = -0.5*r0**(q/2) * s**(-q/2) - (d*q/4)*log s + const,

whose derivative (q/(4s))*[ r0**(q/2) s**(-q/2) - d ] vanishes once, at

    s* = r0 / d**(2/q),

and L(s) -> -inf at both s->0 and s->inf. So along the SCALE direction the
q<2 objective has a finite interior maximum for every q>0 -- it does not run
away. That is a necessary condition for (ii) to fail; this script checks the
remaining directions numerically.

Three tests, all at fixed data:
  T1  noise -> 0 at fixed kernel hyperparameters (the literal "interpolation" ray)
  T2  noise -> 0 with the outputscale PROFILED at each noise (the honest version,
      since a shrinking nugget is compensated by a shrinking scale)
  T3  coarse global grid over (lengthscale, outputscale, noise), reporting the
      max and where it sits, to see whether the optimum is interior.

Fixed seeds. Nothing installed is modified.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

import gpytorch
import qpytorch
import qpytorch.kernels as QK
import qpytorch.means as QM
from qpytorch.distributions import MultivariateQExponential
from qpytorch.likelihoods import QExponentialLikelihood

OUT = os.path.join(_HERE, "parity")
os.makedirs(OUT, exist_ok=True)
Q_GRID = [1.0, 1.2, 1.5, 1.8, 2.0]
SEED = 20260917


def make_data(n: int = 40):
    torch.manual_seed(SEED)
    X = torch.linspace(0, 1, n).unsqueeze(-1)
    Y = torch.sin(6 * X.squeeze(-1)) + 0.1 * torch.randn(n)
    return X, Y


class ExQEP(qpytorch.models.ExactQEP):
    def __init__(self, x, y, lik, power):
        super().__init__(x, y, lik)
        self.power = power
        self.mean_module = QM.ZeroMean()
        self.covar_module = QK.ScaleKernel(QK.MaternKernel(nu=2.5))

    def forward(self, x):
        return MultivariateQExponential(self.mean_module(x),
                                        self.covar_module(x), power=self.power)


def mll_at(X, Y, q: float, ls: float, os_: float, noise: float) -> float:
    """Exact type-II marginal log-likelihood, per the installed package."""
    P = torch.tensor(q)
    # NOTE: QExponentialLikelihood defaults to noise_constraint=GreaterThan(1e-4),
    # so the installed package already FLOORS the nugget. Override it here so the
    # interpolation limit is actually reachable and the claim can be tested.
    lik = QExponentialLikelihood(
        power=P, noise_constraint=gpytorch.constraints.GreaterThan(1e-14))
    lik.noise = torch.tensor(float(noise))
    m = ExQEP(X, Y, lik, P)
    m.covar_module.base_kernel.lengthscale = torch.tensor([[float(ls)]])
    m.covar_module.outputscale = torch.tensor(float(os_))
    m.train(); lik.train()
    obj = qpytorch.mlls.ExactMarginalLogLikelihood(lik, m)
    with torch.no_grad(), gpytorch.settings.debug(False), \
            gpytorch.settings.max_cholesky_size(10_000):
        try:
            return float(obj(m(X), Y))
        except Exception:
            return float("nan")


def profile_scale(X, Y, q: float, ls: float, noise: float) -> Dict:
    """Maximize over outputscale at fixed (ls, noise) by 1-D golden-ish scan."""
    grid = np.geomspace(1e-6, 1e6, 121)
    vals = np.array([mll_at(X, Y, q, ls, float(s), noise) for s in grid])
    k = int(np.nanargmax(vals))
    return dict(best_outputscale=float(grid[k]), best_mll=float(vals[k]),
                at_grid_edge=bool(k in (0, len(grid) - 1)))


def t1_noise_ray(X, Y) -> List[Dict]:
    print("=" * 100)
    print("T1  noise -> 0 at FIXED kernel hyperparameters (lengthscale=0.15, outputscale=1)")
    print("=" * 100)
    noises = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
    rows = []
    print("  " + f"{'noise':>10}" + "".join(f"{('q=%g' % q):>13}" for q in Q_GRID))
    for nz in noises:
        line = f"  {nz:>10.0e}"
        for q in Q_GRID:
            v = mll_at(X, Y, q, 0.15, 1.0, nz)
            rows.append(dict(test="T1", q=q, lengthscale=0.15, outputscale=1.0,
                             noise=nz, mll=v))
            line += f"{v:>13.4f}"
        print(line)
    print("\n  If a q<2 column climbs without bound as noise falls, claim 6 holds")
    print("  along this ray; if it turns over or saturates, it does not.")
    return rows


def t2_noise_ray_profiled(X, Y) -> List[Dict]:
    print("\n" + "=" * 100)
    print("T2  noise -> 0 with the OUTPUTSCALE PROFILED at each noise (lengthscale=0.15)")
    print("    a shrinking nugget is otherwise silently compensated by the scale")
    print("=" * 100)
    noises = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    rows = []
    for q in Q_GRID:
        print(f"\n  q={q}")
        print(f"    {'noise':>10}{'profiled mll':>15}{'argmax outputscale':>21}{'at edge':>9}")
        for nz in noises:
            r = profile_scale(X, Y, q, 0.15, nz)
            rows.append(dict(test="T2", q=q, lengthscale=0.15, noise=nz, **r))
            print(f"    {nz:>10.0e}{r['best_mll']:>15.4f}"
                  f"{r['best_outputscale']:>21.3e}{str(r['at_grid_edge']):>9}")
    return rows


def t3_global_grid(X, Y) -> List[Dict]:
    print("\n" + "=" * 100)
    print("T3  COARSE GLOBAL GRID over (lengthscale, outputscale, noise)")
    print("=" * 100)
    ls_g = np.geomspace(1e-3, 1e2, 12)
    os_g = np.geomspace(1e-4, 1e4, 13)
    nz_g = np.geomspace(1e-10, 1e1, 12)
    rows = []
    print(f"  grid = {len(ls_g)} x {len(os_g)} x {len(nz_g)} = "
          f"{len(ls_g)*len(os_g)*len(nz_g)} evaluations per q")
    print(f"\n  {'q':>5}{'max mll':>13}{'ls*':>11}{'outscale*':>12}{'noise*':>11}"
          f"{'noise* at min':>15}{'any interior':>14}")
    for q in Q_GRID:
        best = (-np.inf, None)
        for ls in ls_g:
            for os_ in os_g:
                for nz in nz_g:
                    v = mll_at(X, Y, q, float(ls), float(os_), float(nz))
                    if np.isfinite(v):
                        rows.append(dict(test="T3", q=q, lengthscale=float(ls),
                                         outputscale=float(os_), noise=float(nz),
                                         mll=v))
                        if v > best[0]:
                            best = (v, (ls, os_, nz))
        v, (ls, os_, nz) = best
        at_min_noise = bool(abs(nz - nz_g[0]) < 1e-30)
        interior = bool(ls_g[0] < ls < ls_g[-1] and os_g[0] < os_ < os_g[-1]
                        and nz_g[0] < nz < nz_g[-1])
        print(f"  {q:>5}{v:>13.4f}{ls:>11.3e}{os_:>12.3e}{nz:>11.3e}"
              f"{str(at_min_noise):>15}{str(interior):>14}")
    return rows


if __name__ == "__main__":
    X, Y = make_data()
    r = t1_noise_ray(X, Y) + t2_noise_ray_profiled(X, Y) + t3_global_grid(X, Y)
    pd.DataFrame(r).to_csv(f"{OUT}/taskB_mll_boundedness.csv", index=False)
    print(f"\nwrote {OUT}/taskB_mll_boundedness.csv")
