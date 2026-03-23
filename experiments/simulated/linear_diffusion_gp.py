import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pydmd import DMD

from py_core.fmou import cubic_solver, kf_rts_ar1, fmou_predictive_mean
from py_core.dim_2_lattice import lattice_alg as _lattice_alg


def _ensure_finite(arr, tag="", cap=1e6):
    import numpy as np
    if np.isfinite(arr).all():
        return arr
    bad_nan = np.isnan(arr).sum()
    bad_inf = np.isinf(arr).sum()
    print(f"[Non-finite detected] {tag}: NaN={bad_nan}, Inf={bad_inf}, shape={arr.shape}")
    idx = np.argwhere(~np.isfinite(arr))
    print("  first bad indices:", idx[:5].tolist())
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


# ---------- 1) "observation mean" reality：same as R code :ReacTran + deSolve ----------
def generate_linear_diffusion(k=200, n=200, L=1.0, T=0.2, D=1.0, C_left=0.0, C_right=1.0):
    """
    1D diffusion: u_t = D * u_xx；implicited Euler
    bound：left:Dirichlet=C_left(=0)，right: Dirichlet=C_right(=1)
    return: reality shape (k, n)
    """
    dx = L / k
    dt = T / (n - 1)
    main = np.full(k, 2.0)
    off  = np.full(k - 1, -1.0)
    Lap = sparse.diags([off, main, off], [-1, 0, 1], shape=(k, k), format="csr") / (dx * dx)

    A = sparse.eye(k, format="csr") + dt * D * Lap  # implicit Euler: I + dt*D*Lap (Lap = -L_FD)
    A = A.tolil()
    # Dirichlet bounds
    A[0, :] = 0.0;  A[0, 0] = 1.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0
    A = A.tocsr()

    u = np.zeros(k)
    u[0]  = C_left
    u[-1] = C_right

    reality = np.zeros((k, n))
    reality[:, 0] = u.copy()
    for t in range(1, n):
        rhs = reality[:, t-1].copy()
        rhs[0]  = C_left
        rhs[-1] = C_right
        u_next = spsolve(A, rhs)
        reality[:, t] = u_next
    return reality

# ---------- 2) Gaussian Process on 2D grid ----------
def lattice_alg_gaussian(y, kernel_type="exp", length_scale=10.0, noise_var=1e-3):
    """
    R equivalent: lattice_alg(y, 1:k, 1:n, kernel_type=..., param_ini=c(-2,-2,-3))
    from 2dim_lattice_func.R. Uses separable Kronecker GP via eigendecomposition.
    y: (k, n) matrix; returns pred_mean of same shape.
    kernel_type: "exp" -> exp(-beta*d), "matern" -> Matern 5/2
    length_scale and noise_var are unused; hyperparameters are estimated by MLE.
    """
    H, W = y.shape
    input1 = np.arange(1, H + 1, dtype=float)  # R: 1:k
    input2 = np.arange(1, W + 1, dtype=float)  # R: 1:n
    result = _lattice_alg(
        output_mat=y,
        input1=input1,
        input2=input2,
        kernel_type=kernel_type,
        # param_ini=(-2.0, -2.0, -3.0) is the default, matching R's c(-2,-2,-3)
    )
    return result["pred_mean"]

# ---------- 3) hyperparameters ----------
k = 200
n = 200
num_repetition = 10
sigma0_list = [0.05, 0.1, 0.3]

# "observation" reality
reality = generate_linear_diffusion(k=k, n=n, L=1.0, T=0.2, D=1.0, C_left=0.0, C_right=1.0)

# ---------- 4) results ----------
def _alloc(R, S): return np.full((R, S), np.nan)
rmse_lattice_exp     = _alloc(num_repetition, len(sigma0_list))
rmse_lattice_matern  = _alloc(num_repetition, len(sigma0_list))
rmse_fmou            = _alloc(num_repetition, len(sigma0_list))
rmse_pca             = _alloc(num_repetition, len(sigma0_list))
rmse_dmd             = _alloc(num_repetition, len(sigma0_list))

y_record                       = [None]*len(sigma0_list)
pred_mean_lattice_exp_record   = [None]*len(sigma0_list)
pred_mean_lattice_matern_record= [None]*len(sigma0_list)
pred_mean_fmou_record          = [None]*len(sigma0_list)
pred_mean_pca_record           = [None]*len(sigma0_list)
pred_mean_dmd_record           = [None]*len(sigma0_list)

est_d_record                   = np.full((num_repetition, len(sigma0_list)), np.nan, dtype=int)

# ---------- 5) main iterations ----------
for j, sigma0 in enumerate(sigma0_list):
    for it in range(num_repetition):
        np.random.seed(it+1)
        # observation with noise
        y = reality + np.random.normal(scale=sigma0, size=reality.shape)
        if it == 0:
            y_record[j] = y

        # Lattice - exp kernel （Gaussian RBF）
        ls = max(5, min(k, n))
        pred_exp = lattice_alg_gaussian(y, kernel_type="exp",
                                        length_scale=ls, noise_var=sigma0**2)
        rmse_lattice_exp[it, j] = np.sqrt(np.mean((reality - pred_exp)**2))
        if it == 0: pred_mean_lattice_exp_record[j] = pred_exp

        # Lattice - matern kernel （Gaussian Matern ν=1.5）
        pred_mat = lattice_alg_gaussian(y, kernel_type="matern",
                                        length_scale=ls, noise_var=sigma0**2)
        rmse_lattice_matern[it, j] = np.sqrt(np.mean((reality - pred_mat)**2))
        if it == 0: pred_mean_lattice_matern_record[j] = pred_mat

        # ranks
        d_hat = choose_rank_via_criterion(y)
        est_d_record[it, j] = d_hat

        # FMOU
        fmou_out = fmou_predictive_mean(y, d_hat, M=100, threshold=1e-6,
                                        est_U0=True, est_sigma0_2=True)
        pred_fmou = fmou_out['mean_obs']
        rmse_fmou[it, j] = np.sqrt(np.mean((reality - pred_fmou)**2))
        if it == 0: pred_mean_fmou_record[j] = pred_fmou

        # PCA（U[:,1:d] U^T y）
        pred_pca = pca_reconstruct(y, d_hat)
        rmse_pca[it, j] = np.sqrt(np.mean((reality - pred_pca)**2))
        if it == 0: pred_mean_pca_record[j] = pred_pca

        # DMD（R: cbind(y[,1], in_sample_pred)）
        pred_dmd = dmd_reconstruct(y, r=d_hat)
        pred_dmd_with_first = pred_dmd.copy()
        pred_dmd_with_first[:, 0] = y[:, 0]
        rmse_dmd[it, j] = np.sqrt(np.mean((pred_dmd_with_first - reality)**2))
        if it == 0: pred_mean_dmd_record[j] = pred_dmd_with_first

# ---------- 6) RMSE ----------
rmse_summary = pd.DataFrame({
    0.05: [np.nanmean(rmse_lattice_exp[:,0]),
           np.nanmean(rmse_lattice_matern[:,0]),
           np.nanmean(rmse_fmou[:,0]),
           np.nanmean(rmse_pca[:,0]),
           np.nanmean(rmse_dmd[:,0])],
    0.10: [np.nanmean(rmse_lattice_exp[:,1]),
           np.nanmean(rmse_lattice_matern[:,1]),
           np.nanmean(rmse_fmou[:,1]),
           np.nanmean(rmse_pca[:,1]),
           np.nanmean(rmse_dmd[:,1])],
    0.30: [np.nanmean(rmse_lattice_exp[:,2]),
           np.nanmean(rmse_lattice_matern[:,2]),
           np.nanmean(rmse_fmou[:,2]),
           np.nanmean(rmse_pca[:,2]),
           np.nanmean(rmse_dmd[:,2])],
}, index=["lattice_exp","lattice_matern","fmou","pca","dmd"])
print("\n=== RMSE summary (Gaussian baseline, Linear diffusion) ===\n", rmse_summary)

# ---------- 7) (B) violin plot ----------
method_order = ["Fast-Mat","PCA","FMOU","DMD","Fast-Exp"]
def _reorder(arrs):
    # [exp, mat, fmou, pca, dmd] -> ["Fast-Mat","PCA","FMOU","DMD","Fast-Exp"]
    mapping = {"Fast-Mat":1, "PCA":3, "FMOU":2, "DMD":4, "Fast-Exp":0}
    return [arrs[mapping[m]] for m in method_order]

make_violin_plot(_reorder([rmse_lattice_exp, rmse_lattice_matern, rmse_fmou, rmse_pca, rmse_dmd]),
                 method_order, sigma0_list, title="(B) Linear diffusion (Gaussian)")

# ---------- 8) triplots ----------
plot_triplet(reality, y_record[2], pred_mean_lattice_matern_record[2],
             title_left="(D) Observation mean", title_mid="(E) Noisy observation",
             title_right="(F) Predictive mean", suptitle="Linear diffusion (Gaussian)")
