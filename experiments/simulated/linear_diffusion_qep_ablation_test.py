"""
Ablation test: row-only QEP vs row-then-column sequential QEP on linear diffusion.

Purpose:
    Isolate whether the second (column) pass in fit_qep_separable_2d helps or hurts.
    If row_only and row_col give similar RMSE, the column pass adds nothing and the
    problem is structural (sequential 1D cannot capture 2D correlations).
    If row_col is meaningfully better than row_only, the column pass does help and
    the bottleneck is elsewhere.

Settings fixed for this test:
    - kernel_type    = "matern"
    - q_power        = 1.5
    - sigma0         = 0.05
    - repetitions    = 1  (seed=1)
    - train_iters    = 200 (row and col)
    - k = n          = 200

Helper functions are copied verbatim from linear_diffusion_qep.py.
The main experiment file has no __main__ guard so it cannot be imported safely;
this script is therefore intentionally self-contained.
"""

import os
import numpy as np
import torch
import gpytorch
import qpytorch
import matplotlib.pyplot as plt
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


def fit_qep_row_only(obs, kernel_type="matern", q_power=1.5,
                     train_iters_row=200, lr=0.05,
                     length_scale_row=0.2, nu=1.5,
                     jitter=1e-3, device="cpu"):
    """Row-wise 1D QEP smoothing only — no column pass."""
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
    return after_rows


def fit_qep_row_col(obs, kernel_type="matern", q_power=1.5,
                    train_iters_row=200, train_iters_col=200, lr=0.05,
                    length_scale_row=0.2, length_scale_col=0.2, nu=1.5,
                    jitter=1e-3, device="cpu"):
    """Row-then-column sequential 1D QEP (current behavior in main experiment)."""
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


# ── plotting (minimal, adapted from plot_triplet in linear_diffusion_qep.py) ──

def save_triplet(reality, y_obs, pred, title_right, out_png):
    cmap = "viridis"
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, mat, title in zip(
            axs,
            [reality, y_obs, pred],
            ["(A) Observation mean", "(B) Noisy observation", title_right]):
        ax.imshow(mat, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"  Saved: {out_png}")


# ── ablation test ─────────────────────────────────────────────────────────────

SIGMA0        = float(os.environ.get("SIGMA0", "0.05"))
SEED          = 1
Q_POWER       = 1.5
TRAIN_ITERS   = 200          # used for both row and col passes

print("=" * 60)
print("Ablation test: row_only vs row_col QEP-Matern, linear diffusion")
print(f"  sigma0={SIGMA0}  seed={SEED}  q_power={Q_POWER}  train_iters={TRAIN_ITERS}")
print("=" * 60)

# Ground truth
reality = generate_linear_diffusion(
    k=200, n=200, L=1.0, T=0.2, D=1.0, C_left=0.0, C_right=1.0)
reality = _ensure_finite(reality, tag="reality")
assert np.isfinite(reality).all(), "reality has non-finite values"
print(f"\nreality : min={reality.min():.4f}  max={reality.max():.4f}"
      f"  mean={reality.mean():.4f}  std={reality.std():.4f}")

# Noisy observation (same seed as main experiment rep it=0, so seed=1)
np.random.seed(SEED)
y_obs = reality + np.random.normal(scale=SIGMA0, size=reality.shape)
y_obs = _ensure_finite(y_obs, tag="y_obs")
print(f"y_obs   : min={y_obs.min():.4f}  max={y_obs.max():.4f}"
      f"  mean={y_obs.mean():.4f}  std={y_obs.std():.4f}")

os.makedirs("results", exist_ok=True)
results = {}

# ── Mode 1: row_only ──────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  Mode: row_only  (200 row fits, no column pass)")
print(f"{'─'*60}")

pred_row = fit_qep_row_only(
    y_obs,
    kernel_type="matern", q_power=Q_POWER,
    train_iters_row=TRAIN_ITERS, lr=0.05,
    length_scale_row=0.2, nu=1.5, device="cpu"
)
pred_row = _ensure_finite(pred_row, tag="pred_row")
rmse_row = float(np.sqrt(np.mean((reality - pred_row) ** 2)))
resid_row = reality - pred_row
print(f"  RMSE   : {rmse_row:.6f}")
print(f"  pred   : min={pred_row.min():.4f}  max={pred_row.max():.4f}"
      f"  mean={pred_row.mean():.4f}  std={pred_row.std():.4f}")
print(f"  resid  : mean={resid_row.mean():.4f}  std={resid_row.std():.4f}")
save_triplet(reality, y_obs, pred_row,
             title_right="(C) QEP-Matern (row only)",
             out_png="results/signal_obs_pred_linear_diffusion_qep_row_only.png")
results["row_only"] = rmse_row

# ── Mode 2: row_col ───────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  Mode: row_col  (200 row fits + 200 column fits)")
print(f"{'─'*60}")

pred_rc = fit_qep_row_col(
    y_obs,
    kernel_type="matern", q_power=Q_POWER,
    train_iters_row=TRAIN_ITERS, train_iters_col=TRAIN_ITERS, lr=0.05,
    length_scale_row=0.2, length_scale_col=0.2, nu=1.5, device="cpu"
)
pred_rc = _ensure_finite(pred_rc, tag="pred_row_col")
rmse_rc = float(np.sqrt(np.mean((reality - pred_rc) ** 2)))
resid_rc = reality - pred_rc
print(f"  RMSE   : {rmse_rc:.6f}")
print(f"  pred   : min={pred_rc.min():.4f}  max={pred_rc.max():.4f}"
      f"  mean={pred_rc.mean():.4f}  std={pred_rc.std():.4f}")
print(f"  resid  : mean={resid_rc.mean():.4f}  std={resid_rc.std():.4f}")
save_triplet(reality, y_obs, pred_rc,
             title_right="(C) QEP-Matern (row+col)",
             out_png="results/signal_obs_pred_linear_diffusion_qep_row_col.png")
results["row_col"] = rmse_rc

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Summary")
print(f"{'─'*60}")
for mode, rmse_val in results.items():
    print(f"  {mode:<10}  RMSE={rmse_val:.6f}")
delta = results["row_col"] - results["row_only"]
print(f"  Δ(row_col − row_only) = {delta:+.6f}  "
      f"({'column pass helps' if delta < -0.005 else 'column pass does not help'})")
print(f"\n  reality.std() = {reality.std():.4f}  "
      f"(RMSE at this level means near-constant prediction)")
print("=" * 60)

# ── Save arrays for common-scale comparison figure ───────────────────────────
_COMMON_SCALE_DIR = os.environ.get("COMMON_SCALE_DIR", "results/common_scale_linear_diffusion")
os.makedirs(_COMMON_SCALE_DIR, exist_ok=True)
np.savez(
    os.path.join(_COMMON_SCALE_DIR, "row_only_qep.npz"),
    truth=reality,
    noisy=y_obs,
    pred=pred_row,
    rmse=np.float64(rmse_row),
    sigma0=np.float64(SIGMA0),
    seed=np.int64(SEED),
    q_power=np.float64(Q_POWER),
    kernel="matern",
    method="row_only_qep",
)
np.savez(
    os.path.join(_COMMON_SCALE_DIR, "row_col_qep.npz"),
    truth=reality,
    noisy=y_obs,
    pred=pred_rc,
    rmse=np.float64(rmse_rc),
    sigma0=np.float64(SIGMA0),
    seed=np.int64(SEED),
    q_power=np.float64(Q_POWER),
    kernel="matern",
    method="row_col_qep",
)
print(f"  Saved: {_COMMON_SCALE_DIR}/row_only_qep.npz")
print(f"  Saved: {_COMMON_SCALE_DIR}/row_col_qep.npz")
