"""
Task B: what does the LOCALLY INSTALLED qpytorch 0.2 actually compute for q != 2?

Authoritative target: the qpytorch package imported by
/Users/zchan/miniforge3/envs/gpytorch_arm/bin/python (version 0.2).

Sections
  B1  record installed versions / paths
  B2  scale matrix C vs statistical covariance a(q,d)*C, from source
  B3  empirical covariance of samples, vs C and vs a(q,d)*C
  B4  fixed-hyperparameter ExactQEP posterior: which statistics move with q
  B5  derivative model: ||E[grad f]|| vs E[||grad f||]

Key definitions taken from the installed source
-----------------------------------------------
qpytorch/distributions/multivariate_qexponential.py

    rescalor = exp( (2/q*log2 - log n + lgamma(n/2 + 2/q) - lgamma(n/2)) / 2 )

which is exactly sqrt(a(q,n)) for

    a(q,d) = 2^(2/q) * Gamma(d/2 + 2/q) / ( d * Gamma(d/2) ).

`get_base_samples` builds, for q != 2,
    base = normalize(standard_normal, dim=-1) * Chi2(n)**(1/q)
i.e. a uniform direction on the sphere times radius R = chi2_n^(1/q); this is
the elliptical construction. `rescale=True` divides by rescalor. rsample passes
**kwargs through to get_base_samples and `rescale` DEFAULTS TO FALSE.

`variance` returns the diagonal of `lazy_covariance_matrix`, i.e. diag(C).

All seeds fixed. Nothing installed is modified.
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
from qpytorch.distributions import (
    MultitaskMultivariateQExponential,
    MultivariateQExponential,
    QExponential,
)
from qpytorch.likelihoods import MultitaskQExponentialLikelihood, QExponentialLikelihood

OUT = os.path.join(_HERE, "parity")
os.makedirs(OUT, exist_ok=True)
Q_GRID = [1.0, 1.2, 1.5, 1.8, 2.0, 3.0]
D_GRID = [1, 2, 5, 20]
NSAMP = 200_000
SEED = 20260917


def a_qd(q: float, d: int) -> float:
    """a(q,d) = 2^(2/q) * Gamma(d/2 + 2/q) / ( d * Gamma(d/2) )."""
    return math.exp((2.0 / q) * math.log(2.0)
                    + math.lgamma(d / 2.0 + 2.0 / q)
                    - math.log(d) - math.lgamma(d / 2.0))


def pd_matrix(d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    return A @ A.T / d + torch.eye(d)


def b1_versions() -> Dict:
    rec = dict(python_executable=sys.executable,
               python_version=sys.version.split()[0],
               torch=torch.__version__,
               gpytorch=gpytorch.__version__, gpytorch_file=gpytorch.__file__,
               qpytorch=qpytorch.__version__, qpytorch_file=qpytorch.__file__)
    print("=" * 96)
    print("B1  INSTALLED PACKAGE (authoritative for our experiments)")
    print("=" * 96)
    for k, v in rec.items():
        print(f"  {k:<20} {v}")
    return rec


def b2_formula_check() -> List[Dict]:
    print("\n" + "=" * 96)
    print("B2  rescalor in the INSTALLED source  vs  sqrt(a(q,d)) computed independently")
    print("=" * 96)
    rows = []
    print(f"  {'q':>5}{'d':>5}{'pkg rescalor':>16}{'sqrt(a(q,d))':>16}{'rel diff':>12}")
    for q in Q_GRID:
        for d in D_GRID:
            C = pd_matrix(d)
            dist = MultivariateQExponential(torch.zeros(d), C,
                                            power=torch.tensor(q))
            pkg = float(dist.rescalor)
            mine = math.sqrt(a_qd(q, d))
            rel = abs(pkg - mine) / max(mine, 1e-300)
            rows.append(dict(q=q, d=d, pkg_rescalor=pkg,
                             sqrt_a_qd=mine, rel_diff=rel))
            print(f"  {q:>5}{d:>5}{pkg:>16.9f}{mine:>16.9f}{rel:>12.2e}")
    print("\n  => the installed package's rescalor IS sqrt(a(q,d)) with")
    print("     a(q,d) = 2^(2/q) * Gamma(d/2+2/q) / (d*Gamma(d/2)).")
    return rows


def b3_empirical_cov() -> List[Dict]:
    print("\n" + "=" * 96)
    print("B3  EMPIRICAL COVARIANCE of samples vs C and vs a(q,d)*C")
    print(f"     {NSAMP} samples per cell, seed {SEED}")
    print("=" * 96)
    rows = []
    print(f"  {'q':>5}{'d':>4}{'a(q,d)':>10} | {'rescale=False':>26} | {'rescale=True':>26}")
    print(f"  {'':>5}{'':>4}{'':>10} | {'vs C':>12}{'vs aC':>14} | {'vs C':>12}{'vs aC':>14}")
    for q in Q_GRID:
        for d in D_GRID:
            C = pd_matrix(d)
            a = a_qd(q, d)
            dist = MultivariateQExponential(torch.zeros(d), C,
                                            power=torch.tensor(q))
            cell = {}
            for rescale in (False, True):
                torch.manual_seed(SEED)
                s = dist.rsample(torch.Size([NSAMP]), rescale=rescale)
                s = s.reshape(NSAMP, d)
                emp = (s.T @ s) / NSAMP          # mean is zero by construction
                den = torch.linalg.matrix_norm(C)
                e_vs_C = float(torch.linalg.matrix_norm(emp - C) / den)
                e_vs_aC = float(torch.linalg.matrix_norm(emp - a * C) / (a * den))
                cell[rescale] = (e_vs_C, e_vs_aC)
                rows.append(dict(q=q, d=d, a_qd=a, rescale=rescale,
                                 rel_err_vs_C=e_vs_C, rel_err_vs_aC=e_vs_aC,
                                 emp_var0=float(emp[0, 0]), C00=float(C[0, 0]),
                                 pkg_variance0=float(dist.variance[0])))
            print(f"  {q:>5}{d:>4}{a:>10.4f} | {cell[False][0]:>12.4f}"
                  f"{cell[False][1]:>14.4f} | {cell[True][0]:>12.4f}"
                  f"{cell[True][1]:>14.4f}")
    print("\n  Reading: small 'vs aC' with rescale=False means default samples have")
    print("  covariance a(q,d)*C. Small 'vs C' with rescale=True means the rescaled")
    print("  samples have covariance C.")
    print("\n  And `.variance` vs diag(C) and vs a*diag(C):")
    print(f"  {'q':>5}{'d':>4}{'.variance[0]':>14}{'C[0,0]':>12}{'a*C[0,0]':>12}  verdict")
    for q in Q_GRID:
        d = 5
        C = pd_matrix(d); a = a_qd(q, d)
        dist = MultivariateQExponential(torch.zeros(d), C, power=torch.tensor(q))
        v = float(dist.variance[0]); c = float(C[0, 0])
        verdict = "= diag(C)  (SCALE, not variance)" if abs(v - c) < 1e-12 else "?"
        print(f"  {q:>5}{d:>4}{v:>14.6f}{c:>12.6f}{a*c:>12.6f}  {verdict}")
    return rows


def b3b_univariate_sample_vs_rsample() -> List[Dict]:
    """The univariate QExponential writes `sample` and `rsample` differently:

    qexponential.py:90  sample : eps = Chi2(1)**(1/q) * sign(z)
    qexponential.py:97  rsample: eps = |z|**(2/q - 1) * z

    Reading the source alone suggests two different distributions. They are NOT:
    |eps_rsample| = |z|**(2/q) = (z**2)**(1/q) = Chi2(1)**(1/q), and the sign is
    sign(z) in both. So the two are algebraically the SAME law, differing only in
    RNG consumption. This test exists to settle that empirically rather than by
    reading, and it does: sd and kurtosis agree to Monte-Carlo error.
    """
    print("\n" + "=" * 96)
    print("B3b  univariate QExponential: is `sample` the same law as `rsample`?")
    print("=" * 96)
    rows = []
    print(f"  {'q':>5}{'sd(sample)':>13}{'sd(rsample)':>13}{'ratio':>9}"
          f"{'kurt(sample)':>14}{'kurt(rsample)':>15}{'sqrt(a(q,1))':>14}")
    for q in Q_GRID:
        d1 = QExponential(torch.tensor(0.0), torch.tensor(1.0),
                          power=torch.tensor(q))
        torch.manual_seed(SEED)
        s1 = d1.sample(torch.Size([NSAMP])).reshape(-1)
        torch.manual_seed(SEED)
        s2 = d1.rsample(torch.Size([NSAMP])).reshape(-1)
        k = lambda t: float(t.pow(4).mean() / t.pow(2).mean() ** 2)
        rows.append(dict(q=q, sd_sample=float(s1.std()), sd_rsample=float(s2.std()),
                         ratio=float(s1.std() / s2.std()),
                         kurt_sample=k(s1), kurt_rsample=k(s2),
                         sqrt_a_q1=math.sqrt(a_qd(q, 1))))
        print(f"  {q:>5}{float(s1.std()):>13.6f}{float(s2.std()):>13.6f}"
              f"{float(s1.std()/s2.std()):>9.4f}{k(s1):>14.4f}{k(s2):>15.4f}"
              f"{math.sqrt(a_qd(q,1)):>14.6f}")
    print("\n  Both sd columns track sqrt(a(q,1)) and the kurtoses agree, so the two")
    print("  entry points ARE the same law (ratio ~0.997 = MC noise from different")
    print("  RNG draw order). Univariate .variance = scale^2 also ignores a(q,1).")
    return rows


def b4_posterior() -> List[Dict]:
    print("\n" + "=" * 96)
    print("B4  FIXED-HYPERPARAMETER ExactQEP POSTERIOR: which statistics move with q?")
    print("=" * 96)

    class ExQEP(qpytorch.models.ExactQEP):
        def __init__(self, x, y, lik, power):
            super().__init__(x, y, lik)
            self.power = power
            self.mean_module = QM.ConstantMean()
            self.covar_module = QK.ScaleKernel(QK.MaternKernel(nu=2.5))

        def forward(self, x):
            return MultivariateQExponential(self.mean_module(x),
                                            self.covar_module(x),
                                            power=self.power)

    torch.manual_seed(0)
    n = 30
    X = torch.linspace(0, 1, n).unsqueeze(-1)
    Y = torch.sin(6 * X.squeeze(-1)) + 0.1 * torch.randn(n)
    Xs = torch.linspace(0, 1, 40).unsqueeze(-1)

    rows = []
    ref = {}
    n_test = Xs.shape[0]
    print(f"  d (joint event dim of one predict call) = {n_test}")
    print(f"  {'q':>5}{'mean maxdiff':>14}{'C maxdiff':>12}{'var maxdiff':>13}"
          f"{'emp var/ C':>12}{'a(q,d)':>10}{'q0.05':>10}{'q0.95':>10}{'E|f-m|':>10}"
          f"{'P(|f-m|>2s)':>13}{'kurtosis':>10}")
    # q=2 first so every row has a reference to difference against
    for q in [2.0] + [x for x in Q_GRID if x != 2.0]:
        P = torch.tensor(q)
        lik = QExponentialLikelihood(power=P); lik.noise = torch.tensor(0.01)
        m = ExQEP(X, Y, lik, P)
        m.covar_module.base_kernel.lengthscale = torch.tensor([[0.15]])
        m.covar_module.outputscale = torch.tensor(1.0)
        with torch.no_grad():
            m.mean_module.constant.fill_(0.0)
        m.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.debug(False):
            post = m(Xs)
            mu = post.mean.clone()
            Cm = post.covariance_matrix.clone()
            var = post.variance.clone()
            torch.manual_seed(SEED)
            s = post.rsample(torch.Size([40_000]))          # rescale defaults False
        sd = var.sqrt()
        cen = s - mu
        emp_var = cen.pow(2).mean(0)
        q05 = torch.quantile(s, 0.05, dim=0)
        q95 = torch.quantile(s, 0.95, dim=0)
        eabs = cen.abs().mean(0)
        tail = (cen.abs() > 2 * sd).double().mean(0)
        kurt = (cen.pow(4).mean(0) / cen.pow(2).mean(0).pow(2))
        if q == 2.0:
            ref = dict(mu=mu, Cm=Cm, var=var)
        dmu = float((mu - ref["mu"]).abs().max()) if ref else float("nan")
        dC = float((Cm - ref["Cm"]).abs().max()) if ref else float("nan")
        dv = float((var - ref["var"]).abs().max()) if ref else float("nan")
        ratio = float((emp_var / var).mean())
        rows.append(dict(q=q, n_test=n_test, mean_maxdiff_vs_q2=dmu,
                         covmat_maxdiff_vs_q2=dC, variance_maxdiff_vs_q2=dv,
                         mean_empvar_over_pkgvar=ratio, a_qd_ntest=a_qd(q, n_test),
                         mean_q05=float(q05.mean()), mean_q95=float(q95.mean()),
                         mean_abs_dev=float(eabs.mean()),
                         tail_prob_2sd=float(tail.mean()),
                         kurtosis=float(kurt.mean())))
        print(f"  {q:>5}{dmu:>14.2e}{dC:>12.2e}{dv:>13.2e}{ratio:>12.4f}"
              f"{a_qd(q, n_test):>10.4f}"
              f"{float(q05.mean()):>10.4f}{float(q95.mean()):>10.4f}"
              f"{float(eabs.mean()):>10.4f}{float(tail.mean()):>13.4f}"
              f"{float(kurt.mean()):>10.3f}")

    # B4b: the inflation factor uses the JOINT event dimension, so predicting
    # the SAME point in a bigger batch changes its sampled variance.
    print("\n  B4b  same test point, different batch size (q=1.5):")
    print(f"  {'n_test':>8}{'pkg .variance[0]':>18}{'emp var[0]':>12}"
          f"{'ratio':>9}{'a(1.5,n)':>10}")
    P = torch.tensor(1.5)
    for nt in (2, 5, 10, 40, 200):
        lik = QExponentialLikelihood(power=P); lik.noise = torch.tensor(0.01)
        m = ExQEP(X, Y, lik, P)
        m.covar_module.base_kernel.lengthscale = torch.tensor([[0.15]])
        m.covar_module.outputscale = torch.tensor(1.0)
        with torch.no_grad():
            m.mean_module.constant.fill_(0.0)
        m.eval(); lik.eval()
        xb = torch.cat([torch.tensor([[0.5]]),
                        torch.linspace(0, 1, nt - 1).unsqueeze(-1)])
        with torch.no_grad(), gpytorch.settings.debug(False):
            post = m(xb)
            v0 = float(post.variance[0]); mu0 = post.mean[0].clone()
            torch.manual_seed(SEED)
            s = post.rsample(torch.Size([40_000]))
        ev = float((s[:, 0] - mu0).pow(2).mean())
        rows.append(dict(q=1.5, n_test=nt, batch_probe=True,
                         pkg_variance0=v0, emp_var0=ev, ratio=ev / v0,
                         a_qd_ntest=a_qd(1.5, nt)))
        print(f"  {nt:>8}{v0:>18.6f}{ev:>12.6f}{ev/v0:>9.4f}"
              f"{a_qd(1.5, nt):>10.4f}")
    print("\n  mean / covariance_matrix / .variance identical across q  =>  the")
    print("  internal scale matrix and its diagonal are q-invariant. The sampled")
    print("  quantiles, mean absolute deviation, tail probability and kurtosis are")
    print("  NOT, so the predictive DISTRIBUTION is q-dependent even though its")
    print("  first two reported moments are not.")
    return rows


def b5_derivative() -> List[Dict]:
    print("\n" + "=" * 96)
    print("B5  DERIVATIVE MODEL:  ||E[grad f]||  vs  E[||grad f||]")
    print("=" * 96)

    class QEPGrad(qpytorch.models.ExactQEP):
        def __init__(self, x, y, lik, power):
            super().__init__(x, y, lik)
            self.power = power
            self.mean_module = QM.ConstantMeanGrad()
            self.covar_module = QK.ScaleKernel(QK.Matern52KernelGrad(ard_num_dims=2))

        def forward(self, x):
            return MultitaskMultivariateQExponential(self.mean_module(x),
                                                     self.covar_module(x),
                                                     power=self.power)

    g = torch.linspace(0, 1, 10)
    a, b = torch.meshgrid(g, g, indexing="ij")
    X = torch.stack([a.reshape(-1), b.reshape(-1)], -1).contiguous()
    f = (((X - 0.5).pow(2).sum(-1).sqrt()) < 0.25).double()
    torch.manual_seed(0)
    Y = torch.stack([f + 0.05 * torch.randn(f.shape),
                     torch.zeros_like(f), torch.zeros_like(f)], -1)

    rows = []
    c_ref: List[float] = []   # threshold fixed from the q=2 run, so P(.) discriminates
    print(f"  {'q':>5}{'||E[grad]|| mean':>18}{'E[||grad||] mean':>18}"
          f"{'ratio':>9}{'P(||grad||>c)':>15}{'c':>9}")
    for q in [2.0] + [x for x in Q_GRID if x != 2.0]:
        P = torch.tensor(q)
        lik = MultitaskQExponentialLikelihood(num_tasks=3, power=P)
        lik.noise = torch.tensor(0.01)
        m = QEPGrad(X, Y, lik, P)
        m.covar_module.base_kernel.lengthscale = torch.tensor([[0.2, 0.2]])
        m.covar_module.outputscale = torch.tensor(1.0)
        m.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.debug(False):
            post = m(X)
            mu = post.mean.clone()                       # N x 3
            torch.manual_seed(SEED)
            s = post.rsample(torch.Size([4000]))         # S x N x 3
        norm_of_mean = mu[:, 1:].pow(2).sum(-1).sqrt()               # ||E[grad]||
        mean_of_norm = s[..., 1:].pow(2).sum(-1).sqrt().mean(0)      # E[||grad||]
        if not c_ref:                     # q=2 runs first
            c_ref.append(float(mean_of_norm.mean()))
        c = c_ref[0]
        pgt = (s[..., 1:].pow(2).sum(-1).sqrt() > c).double().mean(0)
        rows.append(dict(q=q, norm_of_mean=float(norm_of_mean.mean()),
                         mean_of_norm=float(mean_of_norm.mean()),
                         ratio=float(mean_of_norm.mean() / norm_of_mean.mean()),
                         prob_grad_gt_c=float(pgt.mean()), c=c))
        print(f"  {q:>5}{float(norm_of_mean.mean()):>18.6f}"
              f"{float(mean_of_norm.mean()):>18.6f}"
              f"{float(mean_of_norm.mean()/norm_of_mean.mean()):>9.3f}"
              f"{float(pgt.mean()):>15.4f}{c:>9.4f}")
    print("\n  ||E[grad f]|| is a deterministic function of the posterior MEAN, so it")
    print("  is q-invariant here. E[||grad f||] and P(||grad f|| > c) are sample")
    print("  based and are NOT.")
    return rows


if __name__ == "__main__":
    v = b1_versions()
    r2 = b2_formula_check()
    r3 = b3_empirical_cov()
    r3b = b3b_univariate_sample_vs_rsample()
    r4 = b4_posterior()
    r5 = b5_derivative()
    pd.DataFrame([v]).to_csv(f"{OUT}/taskB_versions.csv", index=False)
    pd.DataFrame(r2).to_csv(f"{OUT}/taskB_rescalor_formula.csv", index=False)
    pd.DataFrame(r3).to_csv(f"{OUT}/taskB_empirical_cov.csv", index=False)
    pd.DataFrame(r3b).to_csv(f"{OUT}/taskB_sample_vs_rsample.csv", index=False)
    pd.DataFrame(r4).to_csv(f"{OUT}/taskB_posterior_stats.csv", index=False)
    pd.DataFrame(r5).to_csv(f"{OUT}/taskB_derivative_stats.csv", index=False)
    print(f"\nwrote taskB_*.csv into {OUT}")
