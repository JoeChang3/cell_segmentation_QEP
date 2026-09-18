"""
True 2D QEP prototype for the linear diffusion experiment.

Architecture
------------
ExactQEP with GridInterpolationKernel (SKI / KISS-GP), scalar output.
The entire k×n field is treated as a single regression problem with 2D inputs
(space, time) in [0,1]×[0,1].

Template used
-------------
- branin_qep.py (ExactSKIQEPModel pattern, already proven with qpytorch)
- demo_QEP_diff2d.py (Diff_QEP repo) — for QEP architectural style

Why SKI (exact) rather than variational
----------------------------------------
qpytorch.models.ApproximateQEP may not be available in this install (the
Diff_QEP variational demo uses a custom gpytorch fork, not qpytorch).
SKI is already used and tested in branin_qep.py in this project and provides
exact inference on a structured inducing grid — sufficient for the prototype.

Key difference vs row/column sequential QEP
--------------------------------------------
Sequential 1D: z-scores each 1D slice independently → compresses 2D amplitude
  and destroys cross-slice coherence.
This 2D SKI-QEP: z-scores the entire field once globally, then fits a single
  2D QEP with ARD Matern kernel capturing both spatial and temporal correlations
  jointly in one model.

Diagnostic defaults (change at the bottom)
-------------------------------------------
    kernel    = "matern"  (MaternKernel nu=1.5, ard_num_dims=2)
    q_power   = 1.5
    sigma0    = 0.05
    seed      = 1
    k = n     = 200  → 40,000 training points
    grid_size = 64   → 64×64 = 4096 inducing points (SKI)
    iters     = 50
"""

import os
import numpy as np
import torch
import gpytorch
import qpytorch
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from linear_operator import settings as linop_settings


# ── helpers (verbatim from linear_diffusion_qep.py) ──────────────────────────

def _ensure_finite(arr, tag="", cap=1e6):
    if np.isfinite(arr).all():
        return arr
    bad_nan = np.isnan(arr).sum()
    bad_inf = np.isinf(arr).sum()
    print(f"[Non-finite detected] {tag}: NaN={bad_nan}, Inf={bad_inf}, shape={arr.shape}")
    idx = np.argwhere(~np.isfinite(arr))
    print("  first bad indices:", idx[:5].tolist())
    arr = np.nan_to_num(arr, nan=0.0, posinf=cap, neginf=-cap)
    return arr


def generate_linear_diffusion(k=200, n=200, L=1.0, T=0.2, D=1.0,
                               C_left=0.0, C_right=1.0,
                               clip_bounds=(0.0, 1.0)):
    """Crank-Nicolson 1D diffusion solver.  Returns reality: (k, n)."""
    dx = L / k
    dt = T / (n - 1)
    r = D * dt / (dx * dx)
    main = np.full(k, 2.0, dtype=np.float64)
    off  = np.full(k - 1, -1.0, dtype=np.float64)
    Lap  = sparse.diags([off, main, off], [-1, 0, 1], shape=(k, k), format="csr")
    I = sparse.eye(k, format="csr", dtype=np.float64)
    A = (I - 0.5 * r * Lap).tolil()
    B = (I + 0.5 * r * Lap).tocsr()
    for M in (A, B):
        M[0, :] = 0.0;  M[0, 0] = 1.0
        M[-1,:] = 0.0;  M[-1,-1] = 1.0
    A = A.tocsr()
    u = np.zeros(k, dtype=np.float64)
    u[0]  = C_left
    u[-1] = C_right
    reality = np.zeros((k, n), dtype=np.float64)
    reality[:, 0] = u
    for t in range(1, n):
        rhs = B @ reality[:, t-1]
        rhs[0]  = C_left
        rhs[-1] = C_right
        u_next = spsolve(A, rhs)
        if clip_bounds is not None:
            lo, hi = clip_bounds
            u_next = np.clip(u_next, lo, hi)
        u_next = np.nan_to_num(u_next, nan=0.0,
                               posinf=clip_bounds[1] if clip_bounds else 1e6,
                               neginf=clip_bounds[0] if clip_bounds else -1e6)
        reality[:, t] = u_next
    reality = np.nan_to_num(reality, nan=0.0, posinf=1.0, neginf=0.0)
    return reality


# ── 2D SKI-QEP model ─────────────────────────────────────────────────────────

class ExactSKIQEP2D(qpytorch.models.ExactQEP):
    """
    Scalar-output ExactQEP on 2D inputs using Structured Kernel Interpolation.

    Inputs:  (N, 2) tensor of (space, time) coordinates in [0, 1]^2
    Outputs: (N,)   scalar field values
    """
    def __init__(self, train_x, train_y, likelihood, power,
                 kernel_type="matern", grid_size=64, nu=1.5):
        super().__init__(train_x, train_y, likelihood)
        self.power = power
        self.mean_module = qpytorch.means.ConstantMean()
        if kernel_type == "matern":
            base_k = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=2)
        else:
            base_k = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                base_k, grid_size=grid_size, num_dims=2
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return qpytorch.distributions.MultivariateQExponential(
            mean_x, covar_x, power=self.power)


# ── fit function ─────────────────────────────────────────────────────────────

def fit_qep_2d_ski(obs,
                   kernel_type="matern", q_power=1.5, grid_size=64,
                   train_iters=50, lr=0.05, nu=1.5,
                   jitter=1e-3, cg_iters=1000, cg_tol=1e-2,
                   device="cpu"):
    """
    Fit a scalar-output 2D SKI-QEP to the full obs field.

    Args:
        obs       : (k, n) float64 numpy array — noisy observed field
        kernel_type: "matern" or "rbf"
        q_power   : QEP power parameter (0, 2]; q=2 → Gaussian
        grid_size : inducing grid side length; total inducing = grid_size^2
        train_iters: Adam iterations
        lr        : Adam learning rate
        nu        : Matern smoothness (used when kernel_type="matern")
        jitter    : Cholesky jitter
        cg_iters  : max CG iterations for SKI solve
        cg_tol    : CG tolerance
        device    : "cpu" or "cuda"

    Returns:
        pred : (k, n) float64 numpy array — QEP predictive mean
    """
    k, n = obs.shape

    # Build 2D grid coordinates in [0,1] × [0,1]
    xs = np.linspace(0.0, 1.0, k, dtype=np.float32)
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xv, tv = np.meshgrid(xs, ts, indexing="ij")  # each (k, n)
    train_x_np = np.column_stack([xv.ravel(), tv.ravel()])   # (k*n, 2)
    train_y_np = obs.ravel().astype(np.float32)

    # Global z-score normalization: one normalization for the whole 2D field.
    # This is the key structural difference from sequential 1D: each 1D row/col
    # fit z-scores its own slice, compressing relative amplitudes across slices.
    # Here we preserve the global amplitude scale.
    y_mean = float(train_y_np.mean())
    y_std  = float(train_y_np.std()) + 1e-12
    train_y_norm = (train_y_np - y_mean) / y_std

    train_x = torch.from_numpy(train_x_np).to(device=device)
    train_y = torch.from_numpy(train_y_norm).to(device=device)
    POWER   = torch.tensor(float(q_power), dtype=torch.float32, device=device)

    likelihood = qpytorch.likelihoods.QExponentialLikelihood(power=POWER).to(device)
    model = ExactSKIQEP2D(
        train_x, train_y, likelihood, power=POWER,
        kernel_type=kernel_type, grid_size=grid_size, nu=nu
    ).to(device)

    model.train()
    likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with linop_settings.max_cg_iterations(cg_iters), \
         linop_settings.cg_tolerance(cg_tol), \
         gpytorch.settings.cholesky_jitter(jitter), \
         gpytorch.settings.max_preconditioner_size(100):
        for i in range(train_iters):
            opt.zero_grad()
            out = model(train_x)
            loss = -mll(out, train_y)
            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss at iter {i+1}, skipping")
                continue
            loss.backward()
            opt.step()
            if (i + 1) % 10 == 0:
                ls = model.covar_module.base_kernel.base_kernel.lengthscale.squeeze()
                print(f"  iter {i+1:3d}/{train_iters}  loss={loss.item():.4f}"
                      f"  ls=[{ls[0].item():.3f}, {ls[1].item():.3f}]")

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_norm = likelihood(model(train_x)).mean.detach().cpu().numpy()

    pred = (pred_norm * y_std + y_mean).reshape(k, n).astype(np.float64)
    return pred


# ── experiment ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SIGMA0      = float(os.environ.get("SIGMA0", "0.05"))
    SEED        = 1
    Q_POWER     = 1.5
    KERNEL_TYPE = "matern"
    GRID_SIZE   = 64      # inducing: 64×64 = 4096 points
    TRAIN_ITERS = 50

    print("=" * 60)
    print("2D SKI-QEP prototype: linear diffusion")
    print(f"  sigma0={SIGMA0}  seed={SEED}  q_power={Q_POWER}")
    print(f"  kernel={KERNEL_TYPE}  grid_size={GRID_SIZE}  train_iters={TRAIN_ITERS}")
    print("=" * 60)

    reality = generate_linear_diffusion(
        k=200, n=200, L=1.0, T=0.2, D=1.0, C_left=0.0, C_right=1.0)
    reality = _ensure_finite(reality, tag="reality")
    assert np.isfinite(reality).all(), "reality has non-finite values"
    print(f"\nreality : min={reality.min():.4f}  max={reality.max():.4f}"
          f"  mean={reality.mean():.4f}  std={reality.std():.4f}")

    np.random.seed(SEED)
    y_obs = reality + np.random.normal(scale=SIGMA0, size=reality.shape)
    y_obs = _ensure_finite(y_obs, tag="y_obs")
    print(f"y_obs   : min={y_obs.min():.4f}  max={y_obs.max():.4f}"
          f"  mean={y_obs.mean():.4f}  std={y_obs.std():.4f}")

    N = 200 * 200
    print(f"\nFitting 2D SKI-QEP on {N} points "
          f"(inducing grid: {GRID_SIZE}^2 = {GRID_SIZE**2}) ...")

    pred = fit_qep_2d_ski(
        y_obs,
        kernel_type=KERNEL_TYPE, q_power=Q_POWER, grid_size=GRID_SIZE,
        train_iters=TRAIN_ITERS, lr=0.05, nu=1.5,
        jitter=1e-3, cg_iters=1000, cg_tol=1e-2, device="cpu"
    )
    pred = _ensure_finite(pred, tag="pred")
    rmse  = float(np.sqrt(np.mean((reality - pred) ** 2)))
    resid = reality - pred

    print(f"\nResults")
    print(f"  RMSE   : {rmse:.6f}  (reality.std={reality.std():.4f})")
    print(f"  pred   : min={pred.min():.4f}  max={pred.max():.4f}"
          f"  mean={pred.mean():.4f}  std={pred.std():.4f}")
    print(f"  resid  : mean={resid.mean():.4f}  std={resid.std():.4f}")

    os.makedirs("results", exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, mat, title in zip(
            axs,
            [reality, y_obs, pred],
            ["(A) Observation mean", "(B) Noisy observation",
             f"(C) 2D SKI-QEP (q={Q_POWER})"]):
        ax.imshow(mat, cmap="viridis", origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    out_png = "results/signal_obs_pred_linear_diffusion_qep_2d.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"  Saved: {out_png}")

    # ── Save arrays for common-scale comparison figure ────────────────────────
    _COMMON_SCALE_DIR = os.environ.get("COMMON_SCALE_DIR", "results/common_scale_linear_diffusion")
    os.makedirs(_COMMON_SCALE_DIR, exist_ok=True)
    np.savez(
        os.path.join(_COMMON_SCALE_DIR, "ski_qep_2d.npz"),
        truth=reality,
        noisy=y_obs,
        pred=pred,
        rmse=np.float64(rmse),
        sigma0=np.float64(SIGMA0),
        seed=np.int64(SEED),
        q_power=np.float64(Q_POWER),
        kernel=KERNEL_TYPE,
        grid_size=np.int64(GRID_SIZE),
        method="ski_qep_2d",
    )
    print(f"  Saved: {_COMMON_SCALE_DIR}/ski_qep_2d.npz")
    print("=" * 60)
