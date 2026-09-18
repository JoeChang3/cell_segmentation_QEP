"""
Joint 2D variational Q-Exponential / Gaussian process on image coordinates.

Architecture is adapted from Diff_QEP/src/qEPsolver.py (QEPsolver) and
Diff_QEP/demo/demo_QEP_diff2d_variational.py:
  - ApproximateQEP + CholeskyVariationalDistribution(power=q)
  - VariationalStrategy with learnable inducing locations
  - VariationalELBO
  - float64, explicit lengthscale Interval constraints

Differences from Diff_QEP, and why:
  - Scalar output (image intensity), so plain VariationalStrategy replaces
    MultitaskVariationalStrategy. Diff_QEP needs multitask because it models
    derivative tasks (u, u_xx, u_yy) for a PDE operator; here there is no PDE.
  - No LinearMeanGradGrad / Matern52KernelGradGrad, for the same reason.
  - Implemented against the installed `qpytorch` package. The Diff_QEP GPyTorch
    fork is deliberately NOT vendored.

POWER semantics (verified empirically against this qpytorch install, not assumed):
  `power` is q itself, no reparameterization. power=2.0 reproduces the Gaussian
  log_prob and expected_log_prob exactly. In qpytorch's
  QExponentialLikelihood.expected_log_prob the data-fit term is
      -0.5 * r_i**(q/2) + 0.5*(q/2 - 1)*log(r_i) + log(q/2),
      r_i = ((y_i - m_i)**2 + v_i) / sigma^2
  i.e. an l^q penalty on standardized residuals (since r**(q/2) == |e/sigma|**q).
  That is the mechanism by which q < 2 could tolerate the few large residuals
  that a sharp edge produces instead of blurring them away.

NOTE: the additive log(q/2) and 0.5*(q/2-1)*log(r) terms make ELBO values
NOT comparable across different q. Loss is logged for convergence diagnosis
only, never for model selection across q.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import gpytorch
import qpytorch


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class VariationalQEP2D(qpytorch.models.ApproximateQEP):
    """Scalar-output variational QEP on 2D coordinates.

    Args:
        inducing_points: (M, 2) initial inducing locations.
        power:           q of the Q-Exponential process. q=2 -> Gaussian.
        nu:              Matern smoothness.
        ls_bounds:       (lo, hi) Interval constraint on lengthscale.
        learn_inducing:  whether inducing locations are optimized.
    """

    def __init__(self, inducing_points: torch.Tensor, power: torch.Tensor,
                 nu: float = 2.5, ls_bounds: Tuple[float, float] = (1e-3, 1.0),
                 learn_inducing: bool = True):
        self.power = power
        variational_distribution = qpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
            power=power,
        )
        variational_strategy = qpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing,
        )
        super().__init__(variational_strategy)

        self.mean_module = qpytorch.means.ConstantMean()
        base_kernel = qpytorch.kernels.MaternKernel(
            nu=nu,
            ard_num_dims=2,
            lengthscale_constraint=gpytorch.constraints.Interval(*ls_bounds),
        )
        self.covar_module = qpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return qpytorch.distributions.MultivariateQExponential(
            self.mean_module(x), self.covar_module(x), power=self.power
        )


class VariationalGP2D(gpytorch.models.ApproximateGP):
    """Gaussian twin of VariationalQEP2D. Used only as an independent control
    to confirm that the q=2 QEP arm really does reduce to the Gaussian case."""

    def __init__(self, inducing_points: torch.Tensor, nu: float = 2.5,
                 ls_bounds: Tuple[float, float] = (1e-3, 1.0),
                 learn_inducing: bool = True):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.MaternKernel(
            nu=nu,
            ard_num_dims=2,
            lengthscale_constraint=gpytorch.constraints.Interval(*ls_bounds),
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class ExactQEP2D(qpytorch.models.ExactQEP):
    """Exact (no inducing approximation) scalar QEP on 2D coordinates.

    Used as a cross-check arm: an inducing-point grid coarser than the edge
    width imposes a representational floor on sharpness that is shared by all
    arms and could mask any q effect. Exact inference removes that confound.
    Affordable only on small grids (O(N^3)).
    """

    def __init__(self, train_x, train_y, likelihood, power: torch.Tensor,
                 nu: float = 2.5, ls_bounds: Tuple[float, float] = (1e-3, 1.0)):
        super().__init__(train_x, train_y, likelihood)
        self.power = power
        self.mean_module = qpytorch.means.ConstantMean()
        base_kernel = qpytorch.kernels.MaternKernel(
            nu=nu, ard_num_dims=2,
            lengthscale_constraint=gpytorch.constraints.Interval(*ls_bounds),
        )
        self.covar_module = qpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return qpytorch.distributions.MultivariateQExponential(
            self.mean_module(x), self.covar_module(x), power=self.power
        )


class ExactGP2D(gpytorch.models.ExactGP):
    """Gaussian twin of ExactQEP2D."""

    def __init__(self, train_x, train_y, likelihood, nu: float = 2.5,
                 ls_bounds: Tuple[float, float] = (1e-3, 1.0)):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.MaternKernel(
            nu=nu, ard_num_dims=2,
            lengthscale_constraint=gpytorch.constraints.Interval(*ls_bounds),
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Inducing point initialization (identical across arms by construction)
# ─────────────────────────────────────────────────────────────────────────────

def make_inducing_grid(per_dim: int, lo: float = 0.02, hi: float = 0.98) -> torch.Tensor:
    """Deterministic square grid of inducing points in [lo, hi]^2.

    Deterministic on purpose: every arm of the comparison must start from
    byte-identical inducing locations, so a grid is preferred over a random
    subset of the training inputs.
    """
    g = torch.linspace(lo, hi, per_dim, dtype=torch.float64)
    a, b = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([a.reshape(-1), b.reshape(-1)], dim=-1).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Fit result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    pred: np.ndarray                    # (H, W) posterior mean
    var: np.ndarray                     # (H, W) posterior variance (latent)
    loss_history: np.ndarray            # (iters,) -ELBO per iteration
    final_loss: float
    lengthscale: np.ndarray             # (2,) final ARD lengthscales
    outputscale: float
    noise: float
    inducing_final: np.ndarray          # (M, 2) learned inducing locations
    n_iters: int
    runtime_s: float
    n_nonfinite_loss: int               # iterations with non-finite loss (step skipped)
    n_cholesky_retries: int             # jitter escalations / linalg retries
    failed: bool
    failure_reason: str = ""
    diagnostics: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_variational_2d(
    obs: np.ndarray,
    *,
    power: Optional[float],
    inducing_per_dim: int = 18,
    nu: float = 2.5,
    init_lengthscale: float = 0.05,
    init_outputscale: float = 1.0,
    init_noise: float = 0.10,
    ls_bounds: Tuple[float, float] = (1e-3, 1.0),
    noise_bounds: Tuple[float, float] = (1e-4, 1.0),
    lr: float = 0.02,
    train_iters: int = 1200,
    learn_inducing: bool = True,
    learn_noise: bool = True,
    jitter: float = 1e-6,
    inference: str = "variational",
    log_every: int = 100,
    verbose: bool = True,
    tag: str = "",
) -> FitResult:
    """Fit a joint 2D QEP (or GP if power is None) to an image.

    Everything except `power` is held fixed across arms by the caller, so any
    difference in outcome is attributable to q.

    Args:
        obs:       (H, W) noisy observed image.
        power:     q for the QEP. Pass None to run the independent Gaussian
                   control (ApproximateGP/ExactGP + GaussianLikelihood).
        inference: "variational" (inducing points + ELBO) or "exact"
                   (full O(N^3) marginal likelihood, no inducing approximation).
    """
    torch.set_default_dtype(torch.float64)
    H, W = obs.shape

    # Coordinates in [0,1]^2. indexing="ij" so x is the row axis, matching obs.
    ys = torch.linspace(0.0, 1.0, H, dtype=torch.float64)
    xs = torch.linspace(0.0, 1.0, W, dtype=torch.float64)
    rv, cv = torch.meshgrid(ys, xs, indexing="ij")
    train_x = torch.stack([rv.reshape(-1), cv.reshape(-1)], dim=-1).contiguous()
    train_y = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float64)).reshape(-1)

    if inference not in ("variational", "exact"):
        raise ValueError("inference must be 'variational' or 'exact'")

    inducing = make_inducing_grid(inducing_per_dim).clone()

    is_qep = power is not None
    POWER = torch.tensor(float(power), dtype=torch.float64) if is_qep else None

    if inference == "variational":
        if is_qep:
            model = VariationalQEP2D(inducing, POWER, nu=nu, ls_bounds=ls_bounds,
                                     learn_inducing=learn_inducing)
            likelihood = qpytorch.likelihoods.QExponentialLikelihood(
                power=POWER,
                noise_constraint=gpytorch.constraints.Interval(*noise_bounds),
            )
            mll = qpytorch.mlls.VariationalELBO(likelihood, model,
                                                num_data=train_y.numel())
        else:
            model = VariationalGP2D(inducing, nu=nu, ls_bounds=ls_bounds,
                                    learn_inducing=learn_inducing)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(*noise_bounds),
            )
            mll = gpytorch.mlls.VariationalELBO(likelihood, model,
                                                num_data=train_y.numel())
    else:  # exact
        if is_qep:
            likelihood = qpytorch.likelihoods.QExponentialLikelihood(
                power=POWER,
                noise_constraint=gpytorch.constraints.Interval(*noise_bounds),
            )
            model = ExactQEP2D(train_x, train_y, likelihood, POWER,
                               nu=nu, ls_bounds=ls_bounds)
            mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(*noise_bounds),
            )
            model = ExactGP2D(train_x, train_y, likelihood, nu=nu, ls_bounds=ls_bounds)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # ---- identical initialization for every arm ----
    model.covar_module.base_kernel.lengthscale = torch.tensor(
        [[init_lengthscale, init_lengthscale]], dtype=torch.float64)
    model.covar_module.outputscale = torch.tensor(init_outputscale, dtype=torch.float64)
    likelihood.noise = torch.tensor(init_noise, dtype=torch.float64)
    with torch.no_grad():
        model.mean_module.constant.fill_(float(train_y.mean()))

    model.train()
    likelihood.train()

    # Optionally freeze the observation noise.
    #
    # This matters for correctness, not just convenience. For q < 2 the q-EP
    # density carries a factor r**((d/2)(q/2-1)) that diverges as the residual
    # r -> 0, so the exact marginal likelihood (and the ELBO) is UNBOUNDED as
    # the fit approaches interpolation. Type-II MLE for q < 2 is therefore
    # ill-posed: its optimum sits at noise -> 0. Any apparent q < 2 "win"
    # obtained with a learned noise is confounded by the optimizer walking into
    # that singularity. Freezing the noise at the true sigma removes the
    # degenerate direction and isolates the effect of q itself.
    # Verified numerically in experiments/simulated/qep_power_semantics_check.py.
    params = list(model.parameters())
    if learn_noise:
        params += list(likelihood.parameters())
    else:
        for p in likelihood.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=lr)

    loss_history = np.full(train_iters, np.nan)
    n_nonfinite = 0
    n_retries = 0
    failed = False
    failure_reason = ""

    t0 = time.time()
    with gpytorch.settings.cholesky_jitter(double_value=jitter):
        for i in range(train_iters):
            optimizer.zero_grad(set_to_none=True)
            try:
                output = model(train_x)
                loss = -mll(output, train_y)
            except Exception as exc:  # noqa: BLE001 - want the reason recorded
                n_retries += 1
                if n_retries > 25:
                    failed = True
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    break
                continue

            if not torch.isfinite(loss):
                n_nonfinite += 1
                if n_nonfinite > max(50, train_iters // 10):
                    failed = True
                    failure_reason = "non-finite loss persisted"
                    break
                continue

            loss.backward()
            optimizer.step()
            loss_history[i] = float(loss.detach())

            if verbose and ((i + 1) % log_every == 0 or i == 0):
                ls = model.covar_module.base_kernel.lengthscale.detach().flatten()
                print(f"    [{tag}] iter {i+1:5d}/{train_iters}  loss={float(loss.detach()):+.5f}"
                      f"  ls=({ls[0]:.4f},{ls[1]:.4f})"
                      f"  os={float(model.covar_module.outputscale.detach()):.4f}"
                      f"  noise={float(likelihood.noise.detach()):.5f}", flush=True)

    runtime = time.time() - t0

    # ---- predict on the full grid ----
    model.eval()
    likelihood.eval()
    pred = np.full((H, W), np.nan)
    var = np.full((H, W), np.nan)
    if not failed:
        try:
            # fast_pred_var matters a lot here: without it, materializing the
            # posterior variance at N = H*W exact training points costs about as
            # much as the entire training loop.
            with torch.no_grad(), \
                 gpytorch.settings.cholesky_jitter(double_value=jitter), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.debug(False):
                out = model(train_x)
                pred = out.mean.detach().cpu().numpy().reshape(H, W)
                try:
                    var = out.variance.detach().cpu().numpy().reshape(H, W)
                except Exception:  # noqa: BLE001
                    # variance is diagnostic only; never fail a fit over it
                    var = np.full((H, W), np.nan)
        except Exception as exc:  # noqa: BLE001
            failed = True
            failure_reason = f"prediction failed: {type(exc).__name__}: {exc}"

    ls_final = model.covar_module.base_kernel.lengthscale.detach().flatten().cpu().numpy()
    finite_losses = loss_history[np.isfinite(loss_history)]

    if inference == "variational":
        inducing_final = model.variational_strategy.inducing_points.detach().cpu().numpy()
    else:
        # exact inference uses no inducing points; record an empty array so the
        # field stays present and the CSV/NPZ schema is uniform across arms
        inducing_final = np.zeros((0, 2))

    return FitResult(
        pred=pred,
        var=var,
        loss_history=loss_history,
        final_loss=float(finite_losses[-1]) if finite_losses.size else float("nan"),
        lengthscale=ls_final,
        outputscale=float(model.covar_module.outputscale.detach()),
        noise=float(likelihood.noise.detach()),
        inducing_final=inducing_final,
        n_iters=train_iters,
        runtime_s=runtime,
        n_nonfinite_loss=n_nonfinite,
        n_cholesky_retries=n_retries,
        failed=failed,
        failure_reason=failure_reason,
        diagnostics=dict(
            inference=inference,
            init_lengthscale=init_lengthscale,
            init_outputscale=init_outputscale,
            init_noise=init_noise,
            ls_bounds=ls_bounds,
            noise_bounds=noise_bounds,
            lr=lr,
            nu=nu,
            inducing_per_dim=(inducing_per_dim if inference == "variational" else 0),
            num_inducing=(int(inducing.size(0)) if inference == "variational" else 0),
            learn_inducing=(learn_inducing if inference == "variational" else False),
            learn_noise=learn_noise,
            jitter=jitter,
            dtype="float64",
            n_train=int(train_y.numel()),
            loss_first=float(finite_losses[0]) if finite_losses.size else float("nan"),
            loss_min=float(finite_losses.min()) if finite_losses.size else float("nan"),
            **_convergence_diagnostics(finite_losses),
        ),
    )


def _convergence_diagnostics(finite_losses: np.ndarray) -> Dict:
    """Quantify whether the optimizer had plateaued when it was stopped.

    Fairness matters here: if a q<2 arm simply converges more slowly, then
    comparing arms at an equal iteration count penalizes it for the wrong
    reason. `tail_improve_frac` is the fraction of the total loss reduction
    that was still being gained during the final 10% of iterations. Small
    values mean the fit had settled; large values mean it was still moving and
    the iteration budget should be raised for ALL arms before drawing a
    conclusion.
    """
    if finite_losses.size < 20:
        return dict(tail_improve_frac=float("nan"), converged=float("nan"))
    n = finite_losses.size
    tail = max(2, n // 10)
    total_drop = float(finite_losses[0] - finite_losses.min())
    tail_drop = float(finite_losses[-tail] - finite_losses[-1])
    frac = (tail_drop / total_drop) if abs(total_drop) > 1e-12 else float("nan")
    return dict(
        tail_improve_frac=float(frac),
        converged=float(1.0 if (np.isfinite(frac) and frac < 0.01) else 0.0),
    )
