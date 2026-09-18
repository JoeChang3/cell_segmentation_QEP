"""
What does `power` actually do in the installed qpytorch? (verification, not assumption)

This script exists because the whole cell-blob benchmark rests on three claims
about the installed qpytorch that must not be taken on faith:

  1. MAPPING      `power` is q itself, with no reparameterization.
  2. CONTROL      power=2.0 reproduces the Gaussian case exactly, so a q=2 QEP
                  arm is a legitimate Gaussian control rather than merely a
                  differently-labelled model.
  3. MECHANISM    q sets the growth exponent of the per-point residual penalty
                  in the ELBO's data-fit term, i.e. the loss is l^q in the
                  standardized residual.
  4. INVARIANCE   For EXACT inference, the QEP posterior mean is bit-for-bit
                  independent of q once the kernel hyperparameters are fixed.
  5. ILL-POSEDNESS For q < 2 the q-EP marginal log-likelihood DIVERGES to +inf
                  as the fit approaches interpolation, so type-II MLE for q < 2
                  is ill-posed and its optimum sits at noise -> 0.

Claim 3 is the one that matters scientifically. In qpytorch's
QExponentialLikelihood.expected_log_prob the per-point term is

    res_i = -0.5*[log s^2 + log 2pi + r_i**(q/2)]
            + 0.5*(q/2 - 1)*log(r_i) + log(q/2),
    r_i   = ((y_i - m_i)**2 + v_i) / s^2

Because r**(q/2) == |e/s|**q, the dominant penalty grows like |e|^q. So:

    q < 2  -> SUB-quadratic penalty: large residuals are TOLERATED (robust).
    q > 2  -> SUPER-quadratic penalty: large residuals are punished harder.

This has a direct and non-obvious consequence for images. At a sharp cell
boundary, any smooth process mean necessarily incurs large residuals. A
sub-quadratic penalty gives the optimizer LESS incentive to bend toward that
boundary, so q < 2 is expected to select a LONGER lengthscale and blur edges
MORE, not less. That is the opposite of the usual Besov/wavelet intuition, in
which a q < 2 prior on multiscale coefficients promotes sharp, blocky
reconstructions. The difference is where q sits: here it is in the pixel-space
likelihood, not in a prior over a sparsifying basis.

Run:
    python experiments/simulated/qep_power_semantics_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

import gpytorch
import qpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from qpytorch.distributions import MultivariateQExponential
from qpytorch.likelihoods import QExponentialLikelihood

Q_GRID = [3.0, 2.0, 1.8, 1.5, 1.2, 1.0]


def check_mapping_and_gaussian_control() -> bool:
    """Claims 1 and 2."""
    print("=" * 78)
    print("1+2. MAPPING and GAUSSIAN CONTROL")
    print("=" * 78)
    torch.manual_seed(0)
    d = 5
    A = torch.randn(d, d)
    C = A @ A.T + d * torch.eye(d)
    mu, x = torch.randn(d), torch.randn(d)

    mvn = float(MultivariateNormal(mu, C).log_prob(x))
    print(f"  gpytorch MultivariateNormal.log_prob            = {mvn:+.10f}")
    ok = True
    for q in Q_GRID:
        v = float(MultivariateQExponential(mu, C, power=torch.tensor(q)).log_prob(x))
        mark = ""
        if q == 2.0:
            same = abs(v - mvn) < 1e-10
            ok &= same
            mark = "  <-- MUST match MVN exactly: " + ("YES" if same else "NO")
        print(f"  MultivariateQExponential.log_prob(power={q:<4}) = {v:+.10f}{mark}")

    # same check at the likelihood level, which is what the ELBO actually uses
    n = 6
    mean = torch.zeros(n)
    cov = torch.diag(torch.full((n,), 0.01))
    tgt = torch.randn(n) * 0.5
    gl = GaussianLikelihood(); gl.noise = 0.04
    ql = QExponentialLikelihood(power=torch.tensor(2.0)); ql.noise = 0.04
    g = float(gl.expected_log_prob(tgt, MultivariateNormal(mean, cov)).sum())
    qq = float(ql.expected_log_prob(
        tgt, MultivariateQExponential(mean, cov, power=torch.tensor(2.0))).sum())
    same = abs(g - qq) < 1e-10
    ok &= same
    print(f"\n  GaussianLikelihood.expected_log_prob        = {g:+.10f}")
    print(f"  QExponentialLikelihood(power=2.0) same      = {qq:+.10f}")
    print(f"  identical -> q=2 arm is a true Gaussian control: "
          f"{'YES' if same else 'NO'}")
    return ok


def check_penalty_growth() -> None:
    """Claim 3: q sets the l^q growth exponent of the residual penalty."""
    print("\n" + "=" * 78)
    print("3. MECHANISM: how q reshapes the per-point residual penalty")
    print("=" * 78)
    sigma2, v = 0.01, 1e-6      # fixed noise and posterior variance
    res = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])

    def penalty(q: float, r: float) -> float:
        lik = QExponentialLikelihood(power=torch.tensor(q))
        lik.noise = sigma2
        dist = MultivariateQExponential(
            torch.zeros(1), torch.full((1, 1), v), power=torch.tensor(q))
        return -float(lik.expected_log_prob(torch.tensor([r]), dist).sum())

    print(f"  Per-point penalty (negative log-likelihood), noise^2={sigma2}")
    print("  residual |e| :" + "".join(f"{r:>10.2f}" for r in res))
    for q in Q_GRID:
        vals = [penalty(q, float(r)) for r in res]
        print(f"    q={q:<4}     :" + "".join(f"{x:>10.2f}" for x in vals))

    print("\n  Empirical growth exponent  d(log penalty)/d(log|e|)  on |e| in [0.4, 1.6]")
    print("    Predicted = q exactly (penalty ~ |e|^q). Values run slightly high")
    print("    because the range is finite and the +0.5*(q/2-1)*log(r) term")
    print("    contributes; the ordering and the crossing at q=2 are the point.")
    print(f"    {'q':>5}{'slope':>9}{'q (predicted)':>16}   interpretation")
    a, b = 0.4, 1.6
    for q in Q_GRID:
        slope = ((np.log(penalty(q, b)) - np.log(penalty(q, a)))
                 / (np.log(b) - np.log(a)))
        if q < 2.0:
            tag = "sub-quadratic -> TOLERATES edge residuals"
        elif q > 2.0:
            tag = "super-quadratic -> punishes edge residuals"
        else:
            tag = "quadratic (Gaussian reference)"
        print(f"    {q:>5}{slope:>9.3f}{q:>16}   {tag}")

    print("\n  CONSEQUENCE FOR IMAGES")
    print("  A sharp boundary forces large residuals on any smooth process mean.")
    print("  Sub-quadratic (q<2) penalties reduce the incentive to fit them, so")
    print("  q<2 is PREDICTED to select a longer lengthscale and blur edges MORE.")
    print("  This is opposite to the Besov intuition, where q<2 on multiscale")
    print("  coefficients promotes sharp, blocky reconstructions. Here q sits in")
    print("  the pixel-space LIKELIHOOD, not in a prior over a sparsifying basis.")


def check_posterior_mean_invariance() -> bool:
    """Claim 4, and the most consequential one.

    An elliptically-contoured process has the same LINEAR conditional mean as
    the Gaussian with the same covariance; q parameterizes only the radial
    (scale) part of the distribution. So for exact inference the posterior mean

        m = mu + K (K + s^2 I)^{-1} (y - mu)

    contains no q at all. Holding hyperparameters fixed, every q must return
    the identical reconstruction. Consequences:

      * "q<2 preserves sharp edges better" CANNOT hold for the exact-QEP
        posterior mean. It is ruled out by algebra, not by experiment.
      * q can influence an exact-QEP reconstruction ONLY indirectly, by making
        the marginal likelihood select different (lengthscale, outputscale,
        noise).
      * In the VARIATIONAL case the variational parameters are additionally
        fitted under a q-dependent ELBO, so some q dependence does survive --
        but it still acts through parameter selection, not through a different
        estimator.

    Edge-preserving behaviour in Besov/q-exponential theory comes from a MAP
    estimate under a sparsity-promoting prior on MULTISCALE COEFFICIENTS. A
    posterior-mean linear smoother in pixel space cannot reproduce it.
    """
    print("\n" + "=" * 78)
    print("4. POSTERIOR-MEAN INVARIANCE (exact inference, fixed hyperparameters)")
    print("=" * 78)
    n = 20
    g = torch.linspace(0, 1, n)
    a, b = torch.meshgrid(g, g, indexing="ij")
    X = torch.stack([a.reshape(-1), b.reshape(-1)], -1).contiguous()
    truth = (((X - 0.5).pow(2).sum(-1).sqrt()) < 0.25).double()   # sharp disc
    torch.manual_seed(0)
    y = truth + 0.1 * torch.randn(truth.shape)

    import qpytorch.kernels as QK
    import qpytorch.means as QM

    class ExQEP(qpytorch.models.ExactQEP):
        def __init__(self, x, yy, lik, p):
            super().__init__(x, yy, lik)
            self.power = p
            self.mean_module = QM.ConstantMean()
            self.covar_module = QK.ScaleKernel(
                QK.MaternKernel(nu=2.5, ard_num_dims=2))

        def forward(self, x):
            return MultivariateQExponential(
                self.mean_module(x), self.covar_module(x), power=self.power)

    print("  Fixed: lengthscale=0.08, outputscale=1.0, noise=0.01, mean=0")
    print("  Target: sharp-edged disc on a 20x20 grid, sigma=0.1 noise\n")
    means = {}
    for q in Q_GRID:
        P = torch.tensor(q)
        lik = QExponentialLikelihood(power=P); lik.noise = 0.01
        m = ExQEP(X, y, lik, P)
        m.covar_module.base_kernel.lengthscale = torch.tensor([[0.08, 0.08]])
        m.covar_module.outputscale = torch.tensor(1.0)
        with torch.no_grad():
            m.mean_module.constant.fill_(0.0)
        m.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.debug(False):
            means[q] = m(X).mean.numpy()
        print(f"    q={q:<4} RMSE vs truth = {np.sqrt(np.mean((means[q]-truth.numpy())**2)):.10f}")

    ref = means[2.0]
    print("\n  max |posterior_mean(q) - posterior_mean(q=2)|")
    allzero = True
    for q, v in means.items():
        dv = float(np.abs(v - ref).max())
        allzero &= dv < 1e-12
        print(f"    q={q:<4} -> {dv:.3e}")

    print(f"\n  q-INVARIANT posterior mean: {'CONFIRMED' if allzero else 'NO'}")
    if allzero:
        print("  => Under exact inference, no value of q can change the")
        print("     reconstruction. The 'q<2 sharpens edges' hypothesis is")
        print("     structurally impossible for the posterior mean; q acts only")
        print("     by shifting which hyperparameters the likelihood selects.")
    return allzero


def check_mle_illposedness() -> bool:
    """Claim 5.

    log_prob contains  + 0.5 * d * (q/2 - 1) * log(inv_quad).
    For q < 2 the coefficient is negative, so the term -> +inf as inv_quad -> 0.
    inv_quad -> 0 is exactly what a near-interpolating fit produces (noise -> 0).
    So the q-EP marginal likelihood is unbounded above for q < 2 and type-II MLE
    is ill-posed: the optimizer is rewarded without limit for interpolating.

    Practical consequence: any q < 2 result obtained by MLE/ELBO with a LEARNED
    noise is confounded. Near-interpolation happens to preserve sharp edges
    (it preserves everything, including the noise), so a naive reading looks
    like "q < 2 preserves edges" when it is really "q < 2 stopped denoising".
    Re-run with the noise frozen (--fix-noise) to isolate q.
    """
    print("\n" + "=" * 78)
    print("5. ILL-POSEDNESS of type-II MLE for q < 2")
    print("=" * 78)
    torch.manual_seed(0)
    d = 8
    A = torch.randn(d, d)
    R = A @ A.T / d + torch.eye(d)
    K = R + 1e-2 * torch.eye(d)
    u = torch.randn(d)
    u = u / u.norm()

    print("  Residual -> 0 at fixed covariance (what a near-interpolating fit does)")
    scales = [1.0, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8]
    print(f"    {'||resid||':>10}" + "".join(f"{'q='+str(q):>12}" for q in Q_GRID))
    traj = {q: [] for q in Q_GRID}
    for s in scales:
        row = f"    {s:>10.0e}"
        for q in Q_GRID:
            lp = float(MultivariateQExponential(
                torch.zeros(d), K, power=torch.tensor(q)).log_prob(u * s))
            row += f"{lp:>12.3f}"
            traj[q].append(lp)
        print(row)

    # A hard threshold on the final value is the wrong test: q=1.8 diverges too,
    # just more slowly. Test the trend instead.
    span = {q: traj[q][-1] - traj[q][0] for q in Q_GRID}
    bounded_at_2 = abs(span[2.0]) < 1.0
    up = all(span[q] > 5.0 for q in Q_GRID if q < 2.0)
    down = all(span[q] < -5.0 for q in Q_GRID if q > 2.0)

    print(f"\n  change in log-lik from ||resid||=1 to 1e-8:")
    for q in Q_GRID:
        direction = ("DIVERGES UP -> rewards interpolation" if span[q] > 5 else
                     "diverges DOWN -> penalizes interpolation" if span[q] < -5 else
                     "bounded (well-posed)")
        print(f"    q={q:<4} {span[q]:+10.2f}   {direction}")

    print(f"\n  q=2.0 bounded: {'YES' if bounded_at_2 else 'NO'}")
    print(f"  all q<2 diverge upward: {'YES' if up else 'NO'}")
    print(f"  all q>2 diverge downward: {'YES' if down else 'NO'}")
    if up:
        print("\n  => q<2: the marginal-likelihood optimum is at noise -> 0. A learned")
        print("     noise collapses to its lower bound, and the resulting 'edge")
        print("     preservation' is really a failure to denoise at all.")
    if down:
        print("  => q>2: the reverse. Perfect fits are penalized, so the noise is")
        print("     inflated and the fit is over-smoothed.")
    print("     Either way the noise estimate is biased by the SIGN of (q/2 - 1),")
    print("     not by any edge-related property. Freeze the noise (--fix-noise)")
    print("     to test q honestly.")
    return bounded_at_2 and up and down


if __name__ == "__main__":
    print(f"gpytorch {gpytorch.__version__}   qpytorch {qpytorch.__version__}   "
          f"torch {torch.__version__}\n")
    ok = check_mapping_and_gaussian_control()
    check_penalty_growth()
    inv = check_posterior_mean_invariance()
    ill = check_mle_illposedness()
    print("\n" + "=" * 78)
    print(f"RESULT: power == q and power=2.0 is an exact Gaussian control -> "
          f"{'VERIFIED' if ok else 'FAILED'}")
    print(f"RESULT: exact-QEP posterior mean is q-invariant at fixed "
          f"hyperparameters -> {'VERIFIED' if inv else 'NOT VERIFIED'}")
    print(f"RESULT: type-II MLE is ill-posed for q<2 (unbounded at "
          f"interpolation) -> {'VERIFIED' if ill else 'NOT VERIFIED'}")
    print("=" * 78)
    print("\nTAKEAWAY for the cell-segmentation question")
    print("-" * 78)
    print("q, as parameterized in qpytorch, cannot make a posterior-mean")
    print("reconstruction sharper: (4) the estimator is q-invariant at fixed")
    print("hyperparameters, and (5) the only route left -- hyperparameter")
    print("selection -- is corrupted for q<2 by an unbounded likelihood whose")
    print("optimum is interpolation. Edge preservation in Besov/q-exponential")
    print("theory comes from a NON-LINEAR MAP estimate under a sparsity-promoting")
    print("prior on MULTISCALE COEFFICIENTS, which is a different estimator, not")
    print("a different q in a pixel-space likelihood.")
    print("=" * 78)
    sys.exit(0 if (ok and inv and ill) else 1)
