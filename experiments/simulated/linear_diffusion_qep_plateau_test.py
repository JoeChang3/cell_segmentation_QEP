"""
Plateau test: QEP-Matern on linear diffusion, train_iters=200 vs train_iters=500.

Purpose:
    Determine whether the prior-mean collapse seen in Version C (train_iters=200)
    is due to insufficient Adam iterations, or is a structural failure of the
    sequential 1D QEP approach.

Settings fixed for this test:
    - kernel_type = "matern"
    - q_power     = 1.5
    - sigma0      = 0.05  (low-noise regime, easiest for any denoiser)
    - repetitions = 1     (seed=1, same as main experiment rep 0)
    - k = n = 200         (same as main experiment)

Helper functions below are copied verbatim from linear_diffusion_qep.py.
The main experiment file has no __main__ guard so it cannot be imported safely;
this script is therefore intentionally standalone.
"""

import numpy as np
import torch
import gpytorch
import qpytorch
from scipy import sparse
from scipy.sparse.linalg import spsolve


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


def fit_qep_1d(y_1d,
               kernel_type="rbf", q_power=2.0,
               train_iters=50, lr=0.05,
               length_scale=0.2,
               nu=1.5,
               jitter=1e-3,
               device="cpu"):
    torch.set_default_dtype(torch.float64)
    T = len(y_1d)
    x = np.linspace(0.0, 1.0, T, dtype=np.float64)
    y = np.asarray(y_1d, dtype=np.float64)
    y_mean = y.mean()
    y_std  = y.std() + 1e-12
    y_targ = (y - y_mean) / y_std

    train_x = torch.from_numpy(x).unsqueeze(-1).to(
        device=device, dtype=torch.get_default_dtype())
    train_y = torch.from_numpy(y_targ).to(
        device=device, dtype=torch.get_default_dtype())
    POWER = float(q_power)

    class QEP1D(qpytorch.models.ExactQEP):
        def __init__(self, tx, ty, likelihood):
            super().__init__(tx, ty, likelihood)
            self.power = torch.tensor(POWER, dtype=tx.dtype, device=tx.device)
            self.mean_module = qpytorch.means.ConstantMean()
            if kernel_type.lower() in ("rbf", "exp"):
                base_k = qpytorch.kernels.RBFKernel(ard_num_dims=1)
            elif kernel_type.lower() == "matern":
                try:
                    base_k = qpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=1)
                except AttributeError:
                    base_k = qpytorch.kernels.RBFKernel(ard_num_dims=1)
            else:
                raise ValueError("kernel_type must be 'rbf/exp' or 'matern'")
            base_k.lengthscale = torch.tensor(
                [float(length_scale)], dtype=tx.dtype, device=tx.device)
            self.covar_module = gpytorch.kernels.ScaleKernel(base_k)
            self._jitter = float(jitter)

        def forward(self, x):
            m = self.mean_module(x)
            K = self.covar_module(x).add_jitter(self._jitter)
            return qpytorch.distributions.MultivariateQExponential(
                m, K, power=self.power)

    like  = qpytorch.likelihoods.QExponentialLikelihood(
        power=torch.tensor(POWER, dtype=train_x.dtype, device=train_x.device))
    model = QEP1D(train_x, train_y, like).to(device)

    model.train(); like.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = qpytorch.mlls.ExactMarginalLogLikelihood(like, model)

    with gpytorch.settings.cholesky_jitter(jitter), \
         gpytorch.settings.max_preconditioner_size(50), \
         gpytorch.settings.max_cg_iterations(500), \
         gpytorch.settings.cg_tolerance(1e-5):
        for _ in range(train_iters):
            opt.zero_grad()
            out = model(train_x)
            loss = -mll(out, train_y)
            if not torch.isfinite(loss):
                model._jitter = min(model._jitter * 10.0, 1e-1)
                continue
            loss.backward()
            opt.step()

    model.eval(); like.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = like(model(train_x)).mean.detach().cpu().numpy()

    pred = pred * y_std + y_mean
    return pred


def fit_qep_separable_2d(obs,
                         kernel_type="rbf", q_power=2.0,
                         train_iters_row=50, train_iters_col=50,
                         lr=0.05,
                         length_scale_row=0.2, length_scale_col=0.2,
                         nu=1.5,
                         jitter=1e-3,
                         device="cpu"):
    H, W = obs.shape
    after_rows = np.empty_like(obs, dtype=np.float64)
    for i in range(H):
        after_rows[i, :] = fit_qep_1d(
            obs[i, :],
            kernel_type=kernel_type, q_power=q_power,
            train_iters=train_iters_row, lr=lr,
            length_scale=length_scale_row, nu=nu,
            jitter=jitter, device=device
        )
    after_cols = np.empty_like(obs, dtype=np.float64)
    for j in range(W):
        after_cols[:, j] = fit_qep_1d(
            after_rows[:, j],
            kernel_type=kernel_type, q_power=q_power,
            train_iters=train_iters_col, lr=lr,
            length_scale=length_scale_col, nu=nu,
            jitter=jitter, device=device
        )
    return after_cols


# ── plateau test ─────────────────────────────────────────────────────────────

SIGMA0      = 0.05
SEED        = 1
Q_POWER     = 1.5
ITERS_LIST  = [200, 500]

print("=" * 60)
print("Plateau test: QEP-Matern, linear diffusion")
print(f"  sigma0={SIGMA0}  seed={SEED}  q_power={Q_POWER}")
print(f"  comparing train_iters: {ITERS_LIST}")
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

results = {}
for train_iters in ITERS_LIST:
    print(f"\n{'─'*60}")
    print(f"  train_iters = {train_iters}")
    print(f"{'─'*60}")
    pred = fit_qep_separable_2d(
        y_obs,
        kernel_type="matern", q_power=Q_POWER,
        train_iters_row=train_iters, train_iters_col=train_iters,
        lr=0.05,
        length_scale_row=0.2, length_scale_col=0.2,
        nu=1.5, device="cpu"
    )
    pred = _ensure_finite(pred, tag=f"pred (iters={train_iters})")
    rmse_val = float(np.sqrt(np.mean((reality - pred) ** 2)))
    resid    = reality - pred
    print(f"  RMSE   : {rmse_val:.6f}")
    print(f"  pred   : min={pred.min():.4f}  max={pred.max():.4f}"
          f"  mean={pred.mean():.4f}  std={pred.std():.4f}")
    print(f"  resid  : mean={resid.mean():.4f}  std={resid.std():.4f}")
    results[train_iters] = rmse_val

print(f"\n{'='*60}")
print("Summary")
print(f"{'─'*60}")
for iters, rmse_val in results.items():
    print(f"  train_iters={iters:3d}  RMSE={rmse_val:.6f}")
delta = results[ITERS_LIST[1]] - results[ITERS_LIST[0]]
print(f"  ΔRMSE (500-200) = {delta:+.6f}  "
      f"({'improvement' if delta < 0 else 'no improvement / worse'})")
print("=" * 60)
