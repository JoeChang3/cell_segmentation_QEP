# ==== Linear diffusion experiment (R -> Python) ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import gpytorch
from gpytorch.kernels import GridInterpolationKernel, ScaleKernel, RBFKernel, MaternKernel
from gpytorch.constraints import GreaterThan
from pydmd import DMD

import torch, qpytorch
from py_core.fmou import cubic_solver, kf_rts_ar1, fmou_predictive_mean

def _ensure_finite(arr, tag="", cap=1e6):
    import numpy as np
    if np.isfinite(arr).all():
        return arr
    bad_nan = np.isnan(arr).sum()
    bad_inf = np.isinf(arr).sum()
    print(f"[Non-finite detected] {tag}: NaN={bad_nan}, Inf={bad_inf}, shape={arr.shape}")
    # 打印几个坏位置（可选）
    idx = np.argwhere(~np.isfinite(arr))
    print("  first bad indices:", idx[:5].tolist())
    # 兜底修复（继续跑得起来）
    arr = np.nan_to_num(arr, nan=0.0, posinf=cap, neginf=-cap)
    return arr


def choose_rank_via_criterion(output_mat):
    """
    back to randomized_svd when np.linalg.svd does not converge
    return est_d（1..ceil(k*2/3)）。
    """
    import numpy as np
    from math import ceil
    _has_scipy = False
    _has_rand  = False
    try:
        import scipy.linalg as sla
        _has_scipy = True
    except Exception:
        pass
    try:
        from sklearn.utils.extmath import randomized_svd
        _has_rand = True
    except Exception:
        pass

    A = np.asarray(output_mat)
    if A.ndim != 2:
        raise ValueError("output_mat must be 2D.")

    A = A.astype(np.float64, copy=True)
    # NaN → 0，Inf → bounded value
    A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
    # clip extreme value
    A = np.clip(A, -1e6, 1e6)
    # decentralize（col)
    A = A - A.mean(axis=0, keepdims=True)
    # make sure continuous memory
    A = np.ascontiguousarray(A)

    k, n = A.shape
    max_d = int(ceil(k * 2.0 / 3.0))
    if max_d < 1:
        return 1

    # 2) try stable SVD first
    U = s = Vt = None
    try:
        if _has_scipy:
            # gesvd is more stable than the default gesdd in dealing with some matrices
            U, s, Vt = sla.svd(A, full_matrices=False, lapack_driver="gesvd")
        else:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except Exception:
        # 3) back：randomized SVD（only fisrt max_d）
        if _has_rand:
            r = min(max_d, min(k, n) - 1) if min(k, n) > 1 else 1
            U, s, Vt = randomized_svd(A, n_components=r, n_iter=5, random_state=0)
            pass
        else:
            # try one more mini jitter
            jitter = 1e-8 * np.linalg.norm(A) / max(1.0, np.sqrt(A.size))
            A2 = A + jitter * np.random.randn(*A.shape)
            try:
                if _has_scipy:
                    U, s, Vt = sla.svd(A2, full_matrices=False, lapack_driver="gesvd")
                else:
                    U, s, Vt = np.linalg.svd(A2, full_matrices=False)
            except Exception as e2:
                raise np.linalg.LinAlgError(f"SVD still did not converge after stabilization: {e2}")

    # 4) compute the estimated rank
    k, n = A.shape
    # sometimes randomized_svd < max_d
    r_avail = s.shape[0]
    max_d_eff = min(max_d, r_avail)

    losses = []
    for d in range(1, max_d_eff + 1):
        Ud = U[:, :d]
        # reconstruct Ud @ Ud.T @ A
        recon = Ud @ (Ud.T @ A)
        resid = A - recon
        crit = np.log(np.mean(resid**2) + 1e-12) + d * (k + n) / (k * n) * np.log(k * n / (k + n))
        losses.append(crit)

    est_d = 1 + int(np.argmin(losses))
    return int(est_d)


def pca_reconstruct(obs, d):
    U, s, Vt = np.linalg.svd(obs, full_matrices=False)
    Ud = U[:, :d]
    Sd = np.diag(s[:d])
    Vtd = Vt[:d, :]
    return Ud @ Sd @ Vtd

def dmd_reconstruct(obs, r):
    """
    similar to the DMD_alg in r code: treat each column as a time step（match the r's cbind(output_mat[,1], in_sample_pred)）
    return the reconstruction with the same shape as obs
    """
    H, W = obs.shape
    # each col is a time step
    dmd = DMD(svd_rank=r, opt=True)
    dmd.fit(obs)
    recon = np.real(dmd.reconstructed_data)
    recon[:, 0] = obs[:, 0]
    return recon

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def make_violin_plot(all_rmse_arrays, method_labels, sigma_list, title):
    """
    all_rmse_arrays: list of arrays, each shape (R, len(sigmas))
    """
    records = []
    for m_idx, arr in enumerate(all_rmse_arrays):
        for s_idx, sigma in enumerate(sigma_list):
            vals = arr[:, s_idx]
            for v in vals:
                records.append({"RMSE": v, "Methods": method_labels[m_idx], "noise_level": f"sigma0={sigma}"})
    df = pd.DataFrame(records)
    uniq_noise = [f"sigma0={s}" for s in sigma_list]
    fig, axes = plt.subplots(1, len(uniq_noise), figsize=(12, 3), sharey=False)
    if len(uniq_noise) == 1:
        axes = [axes]
    for ax, noise in zip(axes, uniq_noise):
        sub = [df[(df["Methods"] == m) & (df["noise_level"] == noise)]["RMSE"].values for m in method_labels]
        parts = ax.violinplot(sub, showmeans=True, showextrema=True)
        ax.set_title(noise)
        ax.set_xticks(range(1, len(method_labels)+1))
        ax.set_xticklabels(method_labels, rotation=30)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_triplet(obs_mean, noisy_obs, pred_mean, title_left="(A) Observation mean",
                 title_mid="(B) Noisy observation", title_right="(C) Predictive mean",
                 suptitle=""):
    cmap = "viridis" 
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    im0 = axs[0].imshow(obs_mean, cmap=cmap, origin="lower")
    axs[0].set_title(title_left, fontsize=10)
    axs[0].axis("off")

    im1 = axs[1].imshow(noisy_obs, cmap=cmap, origin="lower")
    axs[1].set_title(title_mid, fontsize=10)
    axs[1].axis("off")

    im2 = axs[2].imshow(pred_mean, cmap=cmap, origin="lower")
    axs[2].set_title(title_right, fontsize=10)
    axs[2].axis("off")

    fig.suptitle(suptitle, fontweight="bold")
    plt.tight_layout()
    plt.show()
    

def fit_qep_1d(y_1d,
               kernel_type="rbf", q_power=2.0,
               train_iters=50, lr=0.05,
               length_scale=0.2,  # 1D input has been normalized to [0,1]
               nu=1.5,
               jitter=1e-3,
               device="cpu"):
    """
    y_1d: shape (T,)
    返回：posterior mean, shape (T,)
    """
    torch.set_default_dtype(torch.float64)
    T = len(y_1d)

    # normalization
    x = np.linspace(0.0, 1.0, T, dtype=np.float64)
    y = np.asarray(y_1d, dtype=np.float64)

    # standardization
    y_mean = y.mean()
    y_std  = y.std() + 1e-12
    y_targ = (y - y_mean) / y_std

    train_x = torch.from_numpy(x).unsqueeze(-1).to(device=device, dtype=torch.get_default_dtype())  # (T,1)
    train_y = torch.from_numpy(y_targ).to(device=device, dtype=torch.get_default_dtype())
    POWER   = float(q_power)

    class QEP1D(qpytorch.models.ExactQEP):
        def __init__(self, tx, ty, likelihood):
            super().__init__(tx, ty, likelihood)
            self.power = torch.tensor(POWER, dtype=tx.dtype, device=tx.device)
            self.mean_module = qpytorch.means.ConstantMean()
            # 1D kernel
            if kernel_type.lower() in ("rbf", "exp"):
                base_k = qpytorch.kernels.RBFKernel(ard_num_dims=1)
            elif kernel_type.lower() == "matern":
                try:
                    base_k = qpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=1)
                except AttributeError:
                    base_k = qpytorch.kernels.RBFKernel(ard_num_dims=1)
            else:
                raise ValueError("kernel_type must be 'rbf/exp' or 'matern'")
            base_k.lengthscale = torch.tensor([float(length_scale)], dtype=tx.dtype, device=tx.device)
            self.covar_module = gpytorch.kernels.ScaleKernel(base_k)
            self._jitter = float(jitter)

        def forward(self, x):
            m = self.mean_module(x)
            K = self.covar_module(x).add_jitter(self._jitter)
            return qpytorch.distributions.MultivariateQExponential(m, K, power=self.power)

    like  = qpytorch.likelihoods.QExponentialLikelihood(power=torch.tensor(POWER, dtype=train_x.dtype, device=train_x.device))
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

    # 反标准化
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
    """
    1D QEP by row first，then by col，return (H,W)
    avoid to constructing 4e4×4e4 kernel matrix
    """
    H, W = obs.shape
    # 1) smoothing by row
    after_rows = np.empty_like(obs, dtype=np.float64)
    for i in range(H):
        after_rows[i, :] = fit_qep_1d(
            obs[i, :],
            kernel_type=kernel_type, q_power=q_power,
            train_iters=train_iters_row, lr=lr,
            length_scale=length_scale_row, nu=nu,
            jitter=jitter, device=device
        )
    # 2) smoothing by column
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





def generate_linear_diffusion(k=200, n=200, L=1.0, T=0.2, D=1.0,
                              C_left=0.0, C_right=1.0,
                              clip_bounds=(0.0, 1.0)):
    """
    stable 1D diffusion: Crank–Nicolson ，Dirichlet bound.
    return reality: shape (k, n), one time step each column
    """
    dx = L / k
    dt = T / (n - 1)
    r = D * dt / (dx * dx)

    # Laplacian
    main = np.full(k, 2.0, dtype=np.float64)
    off  = np.full(k - 1, -1.0, dtype=np.float64)
    Lap  = sparse.diags([off, main, off], [-1, 0, 1], shape=(k, k), format="csr")

    # CN： (I - r/2 * L) u^{t+1} = (I + r/2 * L) u^{t}
    I = sparse.eye(k, format="csr", dtype=np.float64)
    A = (I - 0.5 * r * Lap).tolil()
    B = (I + 0.5 * r * Lap).tocsr()

    # Dirichlet bound condition
    for M in (A, B):
        M[0, :] = 0.0;  M[0, 0] = 1.0
        M[-1,:] = 0.0;  M[-1,-1] = 1.0
    A = A.tocsr()

    # initial condition
    u = np.zeros(k, dtype=np.float64)
    u[0]  = C_left
    u[-1] = C_right

    reality = np.zeros((k, n), dtype=np.float64)
    reality[:, 0] = u

    for t in range(1, n):
        rhs = B @ reality[:, t-1]
        # bounds
        rhs[0]  = C_left
        rhs[-1] = C_right
        u_next = spsolve(A, rhs)

        if clip_bounds is not None:
            lo, hi = clip_bounds
            u_next = np.clip(u_next, lo, hi)

        # secure clearing
        u_next = np.nan_to_num(u_next, nan=0.0, posinf=hi if clip_bounds else 1e6,
                               neginf=lo if clip_bounds else -1e6)

        reality[:, t] = u_next

    # global clearing before return
    reality = np.nan_to_num(reality, nan=0.0, posinf=1.0, neginf=0.0)
    return reality


# ===== hyperparameters =====
k = 200
n = 200
sigma0_list = [0.05, 0.1, 0.3]
num_repetition = 10

# “observation mean” reality（k x n）
reality = generate_linear_diffusion(k=k, n=n, L=1.0, T=0.2, D=1.0, C_left=0.0, C_right=1.0)
reality = _ensure_finite(reality, tag="reality")
assert np.isfinite(reality).all(), "reality has non-finite values"

# —— tool functions：
# fmou_predictive_mean(obs, d, ...)
# choose_rank_via_criterion(output_mat)
# pca_reconstruct(obs, d)
# dmd_reconstruct(obs, r)
# rmse(a,b)
# make_violin_plot(...); plot_triplet(...)

# result container
def _alloc(R, S): return np.full((R, S), np.nan)
rmse_qep_rbf     = _alloc(num_repetition, len(sigma0_list))   # lattice_exp in R
rmse_qep_matern  = _alloc(num_repetition, len(sigma0_list))   # lattice_matern in R
rmse_fmou        = _alloc(num_repetition, len(sigma0_list))
rmse_pca         = _alloc(num_repetition, len(sigma0_list))
rmse_dmd         = _alloc(num_repetition, len(sigma0_list))

y_record                     = [None]*len(sigma0_list)
pred_qep_rbf_record          = [None]*len(sigma0_list)
pred_qep_matern_record       = [None]*len(sigma0_list)
pred_fmou_record             = [None]*len(sigma0_list)
pred_pca_record              = [None]*len(sigma0_list)
pred_dmd_record              = [None]*len(sigma0_list)
est_d_record                 = np.full((num_repetition, len(sigma0_list)), np.nan, dtype=int)

# ===== main iteration =====
for j, sigma0 in enumerate(sigma0_list):
    for it in range(num_repetition):
        np.random.seed(it+1)
        y_obs = reality + np.random.normal(scale=sigma0, size=reality.shape)
        y_obs = _ensure_finite(y_obs, tag = f"y_obs (sigma={sigma0}, it = {it})")
        if it == 0:
            y_record[j] = y_obs   

        # QEP-RBF（ R: lattice_exp）
        # length_scale  0.1~0.5
        pred_rbf = fit_qep_separable_2d(
            y_obs, kernel_type="rbf", q_power=2.5,
            train_iters_row=50, train_iters_col=50, lr=0.05,
            length_scale_row=0.2, length_scale_col=0.2,
            device="cpu"
        )
        pred_rbf = _ensure_finite(pred_rbf, tag = f"pred_rbf (sigma={sigma0}, it={it})")

        # pred_rbf = fit_qep_on_grid_linear(y, kernel_type="rbf", q_power=2.0,
        #                           train_iters=200, lr=0.02,
        #                           length_scale=0.2, device="cpu",
        #                           inducing_per_dim=(64, 64))


        rmse_qep_rbf[it, j] = np.sqrt(np.mean((reality - pred_rbf)**2))
        if it == 0: pred_qep_rbf_record[j] = pred_rbf

        # QEP-Matern（R: lattice_matern）
        pred_mat = fit_qep_separable_2d(
            y_obs, kernel_type="matern", q_power=2.5,
            train_iters_row=50, train_iters_col=50, lr=0.05,
            length_scale_row=0.2, length_scale_col=0.2,
            nu=1.5, device="cpu"
        )
        pred_mat = _ensure_finite(pred_mat, tag=f"pred_mat (sigma={sigma0}, it={it})")
        rmse_qep_matern[it, j] = np.sqrt(np.mean((reality - pred_mat)**2))
        if it == 0: pred_qep_matern_record[j] = pred_mat
        
        assert np.isfinite(y_obs).all(), "y_obs contains NaN/Inf before SVD rank selection"
        # avoid extreme value
        # y_for_svd = np.clip(y_obs, -1e6, 1e6)
        y_for_svd = _ensure_finite(y_obs.copy(), tag=f"before SVD (sigma={sigma0}, it={it})")
        # estimate rank d
        d_hat = choose_rank_via_criterion(y_for_svd)
        est_d_record[it, j] = d_hat

        # FMOU
        fmou_out = fmou_predictive_mean(y_obs, d_hat, M=100, threshold=1e-6, est_U0=True, est_sigma0_2=True)
        pred_fmou = fmou_out['mean_obs']
        rmse_fmou[it, j] = np.sqrt(np.mean((reality - pred_fmou)**2))
        if it == 0: pred_fmou_record[j] = pred_fmou

        # PCA
        pred_pca = pca_reconstruct(y_obs, d_hat)
        rmse_pca[it, j] = np.sqrt(np.mean((reality - pred_pca)**2))
        if it == 0: pred_pca_record[j] = pred_pca

        # DMD
        pred_dmd = dmd_reconstruct(y_obs, r=d_hat)
        # DMD in R: cbind(y[,1], in_sample_pred)
        pred_dmd_with_first = pred_dmd.copy()
        pred_dmd_with_first[:, 0] = y_obs[:, 0]
        rmse_dmd[it, j] = np.sqrt(np.mean((pred_dmd_with_first - reality)**2))
        if it == 0: pred_dmd_record[j] = pred_dmd_with_first

# ======= RMSE Summary=======
rmse_summary = pd.DataFrame({
    0.05: [np.nanmean(rmse_qep_rbf[:,0]), np.nanmean(rmse_qep_matern[:,0]),
           np.nanmean(rmse_fmou[:,0]),    np.nanmean(rmse_pca[:,0]), np.nanmean(rmse_dmd[:,0])],
    0.10: [np.nanmean(rmse_qep_rbf[:,1]), np.nanmean(rmse_qep_matern[:,1]),
           np.nanmean(rmse_fmou[:,1]),    np.nanmean(rmse_pca[:,1]), np.nanmean(rmse_dmd[:,1])],
    0.30: [np.nanmean(rmse_qep_rbf[:,2]), np.nanmean(rmse_qep_matern[:,2]),
           np.nanmean(rmse_fmou[:,2]),    np.nanmean(rmse_pca[:,2]), np.nanmean(rmse_dmd[:,2])],
}, index=["qep_rbf","qep_matern","fmou","pca","dmd"])
print("\n=== RMSE summary (Linear diffusion) ===\n", rmse_summary)

# ======= Figure (B) violin plot =======
method_order = ["QEP-Mat","PCA","FMOU","DMD","QEP-RBF"]
def _reorder(arrs):
    # [rbf, mat, fmou, pca, dmd] -> ["QEP-Mat","PCA","FMOU","DMD","QEP-RBF"]
    mapping = {"QEP-Mat":1, "PCA":3, "FMOU":2, "DMD":4, "QEP-RBF":0}
    return [arrs[mapping[m]] for m in method_order]

make_violin_plot(_reorder([rmse_qep_rbf, rmse_qep_matern, rmse_fmou, rmse_pca, rmse_dmd]),
                 method_order, sigma0_list, title="(B) Linear diffusion")

# ======= Figure 6(B)：（observation mean、observation with noise、Matern estimation）=======
# sigma0=0.3, same with R code
plot_triplet(reality, y_record[2], pred_qep_matern_record[2],
             title_left="(D) Observation mean", title_mid="(E) Noisy observation",
             title_right="(F) Predictive mean", suptitle="Linear diffusion")
