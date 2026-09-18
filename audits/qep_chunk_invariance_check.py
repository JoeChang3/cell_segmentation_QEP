"""
Part B: is any QEP predictive uncertainty quantity usable as a SPATIAL feature?

A spatial feature must not change because the same query set was split into
different prediction chunks. The previous audit found that it does change, so
this script determines exactly why, and whether any scaling convention fixes it.

Sections
  B1  installed versions / paths
  B2  the three notions of scale, formalized and verified numerically
  B3  chunk-SIZE invariance at a single fixed query point x0
  B4  partition invariance over a fixed set of 200 query locations
  B5  derivative / boundary statistics under different partitions
  B6  what dimension actually enters the q scaling
  B7  the acceptance criterion, applied

THREE SCALING CONVENTIONS COMPARED THROUGHOUT
  A "default"  : post.rsample(...)                      (rescale defaults False)
  B "rescale"  : post.rsample(..., rescale=True)        divides by sqrt(a(q,d))
  C "marginal" : build the statistic from the UNIVARIATE q-exponential with
                 location mu(x0) and scale sqrt(C(x0,x0)), i.e. d = 1 always.
                 This is the pointwise marginal of the process, and it is
                 independent of how many other points share the batch BY
                 CONSTRUCTION. It is not a redefinition of the process; it is
                 the 1-D marginal evaluated consistently.

Fixed seeds throughout. Nothing in site-packages is modified.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

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

Q_GRID = [1.2, 1.5, 1.8, 2.0]
SEED = 20260917
X0 = 0.5                      # the fixed physical query location
LS, OS_, NOISE = 0.15, 1.0, 0.01
MAX_ELEM = 20_000_000         # cap on samples*dim per draw, keeps memory bounded


def a_qd(q: float, d: int) -> float:
    """Cov(X) = a(q,d) * C   for the package's default elliptical sampling."""
    return math.exp((2.0 / q) * math.log(2.0)
                    + math.lgamma(d / 2.0 + 2.0 / q)
                    - math.log(d) - math.lgamma(d / 2.0))


class ExQEP(qpytorch.models.ExactQEP):
    def __init__(self, x, y, lik, power):
        super().__init__(x, y, lik)
        self.power = power
        self.mean_module = QM.ConstantMean()
        self.covar_module = QK.ScaleKernel(QK.MaternKernel(nu=2.5))

    def forward(self, x):
        return MultivariateQExponential(self.mean_module(x),
                                        self.covar_module(x), power=self.power)


def make_model(q: float):
    """One tiny fixed-hyperparameter ExactQEP regression problem."""
    torch.manual_seed(0)
    n = 30
    X = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    Y = torch.sin(6.0 * X.squeeze(-1)) + 0.1 * torch.randn(n)
    P = torch.tensor(q)
    lik = QExponentialLikelihood(power=P)
    lik.noise = torch.tensor(NOISE)
    m = ExQEP(X, Y, lik, P)
    m.covar_module.base_kernel.lengthscale = torch.tensor([[LS]])
    m.covar_module.outputscale = torch.tensor(OS_)
    with torch.no_grad():
        m.mean_module.constant.fill_(0.0)
    m.eval()
    lik.eval()
    return m


def col0_stats(post, n_draws: int, rescale: bool, c_abs: float,
               seed: int = SEED) -> Dict:
    """Sample the posterior and keep statistics of COLUMN 0 only.

    Drawn in blocks so that a batch of 8192 points does not need a
    n_draws x 8192 tensor resident at once.
    """
    d = int(post.event_shape[0])
    block = max(1, min(n_draws, max(1, MAX_ELEM // max(d, 1))))
    mu0 = float(post.mean[0])
    got = []
    torch.manual_seed(seed)
    drawn = 0
    while drawn < n_draws:
        k = min(block, n_draws - drawn)
        with torch.no_grad(), gpytorch.settings.debug(False):
            s = post.rsample(torch.Size([k]), rescale=rescale)
        got.append(s[..., 0].reshape(-1).clone())
        drawn += k
        del s
    v = torch.cat(got)
    cen = v - mu0
    return dict(emp_var=float(cen.pow(2).mean()),
                emp_absdev=float(cen.abs().mean()),
                emp_q05=float(torch.quantile(v, 0.05)),
                emp_q95=float(torch.quantile(v, 0.95)),
                emp_tail=float((cen.abs() > c_abs).double().mean()),
                emp_kurt=float(cen.pow(4).mean() / cen.pow(2).mean() ** 2),
                n_draws=n_draws)


def marginal_stats(mu0: float, c00: float, q: float, c_abs: float,
                   n_draws: int = 400_000, seed: int = SEED) -> Dict:
    """Convention C: the UNIVARIATE q-exponential marginal, d = 1 always.

    Uses only mu(x0) and C(x0,x0), so it cannot depend on the batch.
    """
    torch.manual_seed(seed)
    dist = QExponential(torch.tensor(mu0), torch.tensor(math.sqrt(c00)),
                        power=torch.tensor(q))
    with torch.no_grad():
        v = dist.rsample(torch.Size([n_draws])).reshape(-1)
    cen = v - mu0
    return dict(emp_var=float(cen.pow(2).mean()),
                emp_absdev=float(cen.abs().mean()),
                emp_q05=float(torch.quantile(v, 0.05)),
                emp_q95=float(torch.quantile(v, 0.95)),
                emp_tail=float((cen.abs() > c_abs).double().mean()),
                emp_kurt=float(cen.pow(4).mean() / cen.pow(2).mean() ** 2),
                theory_var=a_qd(q, 1) * c00, n_draws=n_draws)


# ---------------------------------------------------------------- B1
def b1() -> Dict:
    rec = dict(python_executable=sys.executable,
               python_version=sys.version.split()[0], torch=torch.__version__,
               gpytorch=gpytorch.__version__, gpytorch_file=gpytorch.__file__,
               qpytorch=qpytorch.__version__, qpytorch_file=qpytorch.__file__)
    print("=" * 112)
    print("B1  INSTALLED ENVIRONMENT (authoritative)")
    print("=" * 112)
    for k, v in rec.items():
        print(f"  {k:<20} {v}")
    return rec


# ---------------------------------------------------------------- B2
def b2() -> List[Dict]:
    print("\n" + "=" * 112)
    print("B2  THREE NOTIONS OF SCALE")
    print("=" * 112)
    print("""  1. internal matrix C      = post.covariance_matrix = post.lazy_covariance_matrix
                               the kernel posterior 'scale matrix'.
  2. package `.variance`    = diag(C). A SCALE parameter, NOT a second moment
                               for q != 2. No a(q,d) applied.
  3. true covariance        = a(q,d) * C for the package's DEFAULT sampling,
                               with a(q,d) = 2^(2/q) Gamma(d/2+2/q)/(d Gamma(d/2))
                               and d = event_shape[0]. rescalor = sqrt(a(q,d)).
     -> `.variance` equals the true marginal variance ONLY at q = 2.""")
    rows = []
    print(f"\n  verification at x0={X0} in a batch of 40:")
    print(f"  {'q':>5}{'C[0,0]':>12}{'.variance[0]':>14}{'a(q,40)C00':>13}"
          f"{'emp var (default)':>19}{'emp var (rescale)':>19}")
    for q in Q_GRID:
        m = make_model(q)
        xb = torch.cat([torch.tensor([[X0]]), torch.linspace(0, 1, 39).unsqueeze(-1)])
        with torch.no_grad(), gpytorch.settings.debug(False):
            post = m(xb)
            c00 = float(post.covariance_matrix[0, 0])
            var0 = float(post.variance[0])
        c_abs = math.sqrt(c00)
        sa = col0_stats(post, 40_000, False, c_abs)
        sb = col0_stats(post, 40_000, True, c_abs)
        a = a_qd(q, 40)
        rows.append(dict(q=q, C00=c00, pkg_variance0=var0, a_qd_40=a,
                         a_times_C00=a * c00, emp_var_default=sa["emp_var"],
                         emp_var_rescale=sb["emp_var"]))
        print(f"  {q:>5}{c00:>12.6f}{var0:>14.6f}{a*c00:>13.6f}"
              f"{sa['emp_var']:>19.6f}{sb['emp_var']:>19.6f}")
    print("\n  '.variance' tracks C00 (q-invariant); 'emp var (default)' tracks")
    print("  a(q,40)*C00; 'emp var (rescale)' tracks C00. Confirms notion 3.")
    return rows


# ---------------------------------------------------------------- B3
def b3() -> List[Dict]:
    print("\n" + "=" * 112)
    print("B3  CHUNK-SIZE INVARIANCE at the single fixed point x0 = 0.5")
    print("    the SAME x0 is element 0 of every batch; only the batch SIZE changes")
    print("=" * 112)
    sizes = [1, 2, 10, 50, 100, 200, 1000, 8192]
    rows = []
    for q in Q_GRID:
        m = make_model(q)
        print(f"\n  q = {q}")
        print(f"    {'batch d':>8}{'mu(x0)':>11}{'C00':>11}{'.var':>11}"
              f"{'a(q,d)':>10} | {'A var':>10}{'A E|f-m|':>10}{'A tail':>8}{'A kurt':>8}"
              f" | {'B var':>10}{'B E|f-m|':>10}{'B tail':>8}{'B kurt':>8}")
        for d in sizes:
            others = torch.linspace(0.0, 1.0, max(d - 1, 1)).unsqueeze(-1)
            xb = torch.tensor([[X0]]) if d == 1 else torch.cat(
                [torch.tensor([[X0]]), others[: d - 1]])
            with torch.no_grad(), gpytorch.settings.debug(False):
                post = m(xb)
                mu0 = float(post.mean[0])
                c00 = float(post.covariance_matrix[0, 0])
                var0 = float(post.variance[0])
            c_abs = math.sqrt(c00)
            nd = 40_000 if d <= 1000 else 8_000
            A = col0_stats(post, nd, False, c_abs)
            B = col0_stats(post, nd, True, c_abs)
            rows.append(dict(q=q, batch_d=d, mu0=mu0, C00=c00, pkg_var0=var0,
                             a_qd=a_qd(q, d),
                             **{f"A_{k}": v for k, v in A.items()},
                             **{f"B_{k}": v for k, v in B.items()}))
            print(f"    {d:>8}{mu0:>11.6f}{c00:>11.6f}{var0:>11.6f}"
                  f"{a_qd(q,d):>10.4f} | {A['emp_var']:>10.6f}{A['emp_absdev']:>10.6f}"
                  f"{A['emp_tail']:>8.4f}{A['emp_kurt']:>8.3f}"
                  f" | {B['emp_var']:>10.6f}{B['emp_absdev']:>10.6f}"
                  f"{B['emp_tail']:>8.4f}{B['emp_kurt']:>8.3f}")
        # convention C, computed once: it cannot depend on d
        c_abs = math.sqrt(c00)
        C = marginal_stats(mu0, c00, q, c_abs)
        print(f"    {'C (d=1)':>8}{mu0:>11.6f}{c00:>11.6f}{'-':>11}"
              f"{a_qd(q,1):>10.4f} | {'':>38} | {C['emp_var']:>10.6f}"
              f"{C['emp_absdev']:>10.6f}{C['emp_tail']:>8.4f}{C['emp_kurt']:>8.3f}")
        rows.append(dict(q=q, batch_d=-1, mu0=mu0, C00=c00, pkg_var0=float("nan"),
                         a_qd=a_qd(q, 1),
                         **{f"C_{k}": v for k, v in C.items()}))
    print("\n  Read the A and B blocks DOWN each column: if a number drifts with")
    print("  'batch d', that statistic is chunk-dependent.")
    return rows


# ---------------------------------------------------------------- B4
def b4() -> List[Dict]:
    print("\n" + "=" * 112)
    print("B4  PARTITION INVARIANCE over a FIXED set of 200 query locations")
    print("    same 200 points every time; only the chunking changes")
    print("=" * 112)
    Xs = torch.linspace(0.0, 1.0, 200).unsqueeze(-1)
    parts = [("1 x 200", 200), ("2 x 100", 100), ("4 x 50", 50),
             ("20 x 10", 10), ("200 x 1", 1)]
    rows = []
    for q in Q_GRID:
        m = make_model(q)
        ref: Dict[str, np.ndarray] = {}
        print(f"\n  q = {q}")
        print(f"    {'partition':<12}{'conv':<10}{'stat':<10}"
              f"{'max|diff| vs 1x200':>20}{'rel diff':>12}{'verdict':>16}")
        for label, cs in parts:
            acc = {k: [] for k in ("mu", "C00", "A_var", "A_absdev", "A_tail",
                                   "B_var", "B_absdev", "B_tail",
                                   "C_var", "C_absdev", "C_tail")}
            for st in range(0, 200, cs):
                xb = Xs[st:st + cs]
                with torch.no_grad(), gpytorch.settings.debug(False):
                    post = m(xb)
                    mu = post.mean.clone()
                    cd = post.covariance_matrix.diagonal().clone()
                d = int(post.event_shape[0])
                nd = 20_000
                # statistics for EVERY point in this chunk
                blk = max(1, min(nd, max(1, MAX_ELEM // max(d, 1))))
                torch.manual_seed(SEED)
                for rescale, pre in ((False, "A"), (True, "B")):
                    got = []
                    drawn = 0
                    torch.manual_seed(SEED)
                    while drawn < nd:
                        k = min(blk, nd - drawn)
                        with torch.no_grad(), gpytorch.settings.debug(False):
                            s = post.rsample(torch.Size([k]), rescale=rescale)
                        got.append(s.reshape(k, d).clone())
                        drawn += k
                    v = torch.cat(got)
                    cen = v - mu
                    acc[f"{pre}_var"].append(cen.pow(2).mean(0))
                    acc[f"{pre}_absdev"].append(cen.abs().mean(0))
                    acc[f"{pre}_tail"].append(
                        (cen.abs() > cd.sqrt()).double().mean(0))
                # convention C: analytic/marginal, per point, d = 1
                a1 = a_qd(q, 1)
                cm, ca, ct = [], [], []
                for i in range(d):
                    s = marginal_stats(float(mu[i]), float(cd[i]), q,
                                       math.sqrt(float(cd[i])), n_draws=20_000)
                    cm.append(s["emp_var"]); ca.append(s["emp_absdev"])
                    ct.append(s["emp_tail"])
                acc["C_var"].append(torch.tensor(cm))
                acc["C_absdev"].append(torch.tensor(ca))
                acc["C_tail"].append(torch.tensor(ct))
                acc["mu"].append(mu); acc["C00"].append(cd)
            cat = {k: torch.cat(v).numpy() for k, v in acc.items()}
            if label == "1 x 200":
                ref = cat
            for key in ("mu", "C00", "A_var", "A_absdev", "A_tail",
                        "B_var", "B_absdev", "B_tail",
                        "C_var", "C_absdev", "C_tail"):
                dmax = float(np.abs(cat[key] - ref[key]).max())
                den = float(np.abs(ref[key]).mean()) or 1.0
                rel = dmax / den
                conv = {"m": "-", "C": "-"}.get(key[:1], key[:1]) if "_" in key else "-"
                verdict = ("INVARIANT" if rel < 1e-6 else
                           "mc-noise" if rel < 0.08 else "CHUNK-DEPENDENT")
                rows.append(dict(q=q, partition=label, statistic=key,
                                 max_abs_diff=dmax, rel_diff=rel, verdict=verdict))
                if label != "1 x 200":
                    print(f"    {label:<12}{conv:<10}{key.split('_',1)[-1]:<10}"
                          f"{dmax:>20.3e}{rel:>12.4f}{verdict:>16}")
    return rows


# ---------------------------------------------------------------- B5
def b5() -> List[Dict]:
    print("\n" + "=" * 112)
    print("B5  DERIVATIVE / BOUNDARY statistics under different partitions")
    print("=" * 112)

    class QEPGrad(qpytorch.models.ExactQEP):
        def __init__(self, x, y, lik, power):
            super().__init__(x, y, lik)
            self.power = power
            self.mean_module = QM.ConstantMeanGrad()
            self.covar_module = QK.ScaleKernel(QK.Matern52KernelGrad(ard_num_dims=2))

        def forward(self, x):
            return MultitaskMultivariateQExponential(
                self.mean_module(x), self.covar_module(x), power=self.power)

    g = torch.linspace(0, 1, 10)
    aa, bb = torch.meshgrid(g, g, indexing="ij")
    Xtr = torch.stack([aa.reshape(-1), bb.reshape(-1)], -1).contiguous()
    f = (((Xtr - 0.5).pow(2).sum(-1).sqrt()) < 0.25).double()
    torch.manual_seed(0)
    Ytr = torch.stack([f + 0.05 * torch.randn(f.shape),
                       torch.zeros_like(f), torch.zeros_like(f)], -1)

    parts = [("1 x 100", 100), ("2 x 50", 50), ("10 x 10", 10), ("100 x 1", 1)]
    rows = []
    for q in Q_GRID:
        P = torch.tensor(q)
        lik = MultitaskQExponentialLikelihood(num_tasks=3, power=P)
        lik.noise = torch.tensor(NOISE)
        m = QEPGrad(Xtr, Ytr, lik, P)
        m.covar_module.base_kernel.lengthscale = torch.tensor([[0.2, 0.2]])
        m.covar_module.outputscale = torch.tensor(OS_)
        m.eval(); lik.eval()

        ref = {}
        print(f"\n  q = {q}")
        print(f"    {'partition':<11}{'event_shape':>14}{'||E[grad]||':>13}"
              f"{'E[||grad||] A':>15}{'E[||grad||] B':>15}{'P(>c) A':>10}{'P(>c) B':>10}")
        for label, cs in parts:
            nm, mn_a, mn_b, p_a, p_b, esh = [], [], [], [], [], None
            for st in range(0, 100, cs):
                xb = Xtr[st:st + cs]
                with torch.no_grad(), gpytorch.settings.debug(False):
                    post = m(xb)
                    mu = post.mean.clone()
                    esh = tuple(post.event_shape)
                    torch.manual_seed(SEED)
                    sA = post.rsample(torch.Size([3000]))
                    torch.manual_seed(SEED)
                    sB = post.rsample(torch.Size([3000]), rescale=True)
                nm.append(mu[:, 1:].pow(2).sum(-1).sqrt())
                mn_a.append(sA[..., 1:].pow(2).sum(-1).sqrt().mean(0))
                mn_b.append(sB[..., 1:].pow(2).sum(-1).sqrt().mean(0))
                p_a.append((sA[..., 1:].pow(2).sum(-1).sqrt() > 1.0).double().mean(0))
                p_b.append((sB[..., 1:].pow(2).sum(-1).sqrt() > 1.0).double().mean(0))
            cat = dict(norm_of_mean=torch.cat(nm).numpy(),
                       mean_of_norm_A=torch.cat(mn_a).numpy(),
                       mean_of_norm_B=torch.cat(mn_b).numpy(),
                       prob_A=torch.cat(p_a).numpy(), prob_B=torch.cat(p_b).numpy())
            if label == "1 x 100":
                ref = cat
            for k, v in cat.items():
                dmax = float(np.abs(v - ref[k]).max())
                den = float(np.abs(ref[k]).mean()) or 1.0
                rows.append(dict(q=q, partition=label, statistic=k,
                                 mean_value=float(v.mean()), max_abs_diff=dmax,
                                 rel_diff=dmax / den, event_shape=str(esh)))
            print(f"    {label:<11}{str(esh):>14}{cat['norm_of_mean'].mean():>13.6f}"
                  f"{cat['mean_of_norm_A'].mean():>15.6f}"
                  f"{cat['mean_of_norm_B'].mean():>15.6f}"
                  f"{cat['prob_A'].mean():>10.4f}{cat['prob_B'].mean():>10.4f}")
    return rows


# ---------------------------------------------------------------- B6
def b6() -> List[Dict]:
    print("\n" + "=" * 112)
    print("B6  WHAT DIMENSION ENTERS THE q SCALING?")
    print("=" * 112)
    print("""  Source: qpytorch/distributions/multivariate_qexponential.py:131
      rescalor: n = self.event_shape[0]
  and get_base_samples builds the radius as Chi2(shape[-1])**(1/power) with
  shape = base_sample_shape.

  For MultivariateQExponential   event_shape = (n_points,)      -> n = n_points
  For MultitaskMultivariateQExp  event_shape = _output_shape[-2:]
                                             = (n_points, n_tasks)
                                -> rescalor uses event_shape[0] = n_points
                                -> radius   uses shape[-1]      = n_tasks
  so in the MULTITASK case the two disagree.""")
    rows = []
    print(f"\n  {'object':<34}{'event_shape':>16}{'rescalor n':>12}"
          f"{'radius dim':>12}{'flat cov dim':>14}{'consistent?':>13}")
    q = 1.5
    # plain MVQEP over n test points
    for n in (5, 40, 200):
        C = torch.eye(n) + 0.1
        dist = MultivariateQExponential(torch.zeros(n), C, power=torch.tensor(q))
        n_res = int(dist.event_shape[0])
        r_dim = int(dist.base_sample_shape[-1])
        ok = (n_res == r_dim == n)
        rows.append(dict(obj=f"MVQEP n={n}", event_shape=str(tuple(dist.event_shape)),
                         rescalor_n=n_res, radius_dim=r_dim, flat_cov_dim=n,
                         consistent=ok))
        print(f"  {'MVQEP over '+str(n)+' test points':<34}"
              f"{str(tuple(dist.event_shape)):>16}{n_res:>12}{r_dim:>12}{n:>14}"
              f"{str(ok):>13}")
    # multitask
    for n, t in ((5, 3), (40, 3), (100, 3)):
        mean = torch.zeros(n, t)
        C = torch.eye(n * t) + 0.1
        dist = MultitaskMultivariateQExponential(mean, C, power=torch.tensor(q))
        n_res = int(dist.event_shape[0])
        r_dim = int(dist.base_sample_shape[-1])
        ok = (n_res == r_dim == n * t)
        rows.append(dict(obj=f"MultitaskMVQEP n={n},t={t}",
                         event_shape=str(tuple(dist.event_shape)),
                         rescalor_n=n_res, radius_dim=r_dim, flat_cov_dim=n * t,
                         consistent=ok))
        print(f"  {'MultitaskMVQEP n='+str(n)+', t='+str(t):<34}"
              f"{str(tuple(dist.event_shape)):>16}{n_res:>12}{r_dim:>12}{n*t:>14}"
              f"{str(ok):>13}")
    print("\n  => the dimension is the NUMBER OF TEST QUERY POINTS IN THE CALL")
    print("     (times nothing else), not the process discretization, not the")
    print("     pixel count of the image, and for multitask the rescalor and the")
    print("     sampling radius use DIFFERENT dimensions.")
    return rows


if __name__ == "__main__":
    v = b1()
    r2, r3, r4, r5, r6 = b2(), b3(), b4(), b5(), b6()
    pd.DataFrame([v]).to_csv(f"{OUT}/chunk_versions.csv", index=False)
    pd.DataFrame(r2).to_csv(f"{OUT}/chunk_b2_scales.csv", index=False)
    pd.DataFrame(r3).to_csv(f"{OUT}/chunk_b3_batchsize.csv", index=False)
    pd.DataFrame(r4).to_csv(f"{OUT}/chunk_b4_partition.csv", index=False)
    pd.DataFrame(r5).to_csv(f"{OUT}/chunk_b5_derivative.csv", index=False)
    pd.DataFrame(r6).to_csv(f"{OUT}/chunk_b6_dimension.csv", index=False)
    print(f"\nwrote chunk_*.csv into {OUT}")
