# ========= Linear diffusion (R -> Python, 原版 Gaussian 对照) =========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel


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


def cubic_solver(p):
    """Solve p3*x^3 + p2*x^2 + p1*x + p0 = 0, return a real root in (-1,1) if any, else -1."""
    p = list(p)
    if len(p) != 4 or abs(p[0]) < np.finfo(float).eps**0.95:
        raise ValueError("Bad cubic coefficients.")
    a0, a1, a2 = p[1]/p[0], p[2]/p[0], p[3]/p[0]            # normalize
    Q = (a0*a0 - 3*a1) / 9.0
    R = (2*a0*a0*a0 - 9*a0*a1 + 27*a2) / 54.0
    roots = []
    if R*R < Q*Q*Q:  # 3 real roots
        theta = np.arccos(R / np.sqrt(Q**3))
        for ang in (theta/3, (theta+2*np.pi)/3, (theta-2*np.pi)/3):
            roots.append(-2*np.sqrt(Q)*np.cos(ang) - a0/3.0)
    else:            # one real root
        A = -np.copysign(1.0, R) * (abs(R) + np.sqrt(R*R - Q*Q*Q))**(1.0/3.0)
        B = 0.0 if abs(A) < 1e-15 else Q / A
        roots = [
            (A + B) - a0/3.0,
            -0.5*(A+B) - a0/3.0 + np.sqrt(3.0)*1j*(A - B)/2.0,
            -0.5*(A+B) - a0/3.0 - np.sqrt(3.0)*1j*(A - B)/2.0
        ]
    for r in roots:
        if isinstance(r, complex):
            if abs(r.imag) > 1e-10: 
                continue
            r = r.real
        if -1.0 < r < 1.0:
            return float(r)
    return -1.0

def kf_rts_ar1(F, V, G, W, y):
    """Scalar AR(1) KF + RTS smoother. Returns dict with b_vec, s_t, f_t, Q_t, a_t, S_t, R_t, m_t, C_t."""
    y = np.asarray(y, float).ravel()
    n = y.size
    a = np.zeros(n); R = np.zeros(n); f = np.zeros(n); Q = np.zeros(n)
    m = np.zeros(n); C = np.zeros(n)
    # stationary prior
    a[0] = 0.0
    R[0] = W / (1.0 - G*G + 1e-12)
    f[0] = F * a[0]
    Q[0] = F * R[0] * F + V
    m[0] = a[0] + R[0]*F*(y[0]-f[0]) / Q[0]
    C[0] = R[0] - R[0]*F*F*R[0] / Q[0]
    for i in range(1, n):
        a[i] = G * m[i-1]
        R[i] = G * C[i-1] * G + W
        f[i] = F * a[i]
        Q[i] = F * R[i] * F + V
        m[i] = a[i] + R[i]*F*(y[i]-f[i]) / Q[i]
        C[i] = R[i] - R[i]*F*F*R[i] / Q[i]
    # RTS smoother
    s = np.zeros(n); S = np.zeros(n); b = np.zeros(n-1)
    s[-1] = m[-1]; S[-1] = C[-1]
    for i in range(n-2, -1, -1):
        inv_R_next = 1.0 / (R[i+1] + 1e-12)
        s[i] = m[i] + C[i]*G*inv_R_next*(s[i+1] - a[i+1])
        S[i] = C[i] - C[i]*G*inv_R_next*(R[i+1] - S[i+1])*inv_R_next*G*C[i]
        b[i] = C[i]*G*inv_R_next*S[i+1]
    return dict(b_vec=b, s_t=s, f_t=f, Q_t=Q, a_t=a, S_t=S, R_t=R, m_t=m, C_t=C)

def fmou_predictive_mean(obs, d, M=50, threshold=1e-4, 
                         est_U0=True, est_sigma0_2=True,
                         U0=None, U_init=None, rho_init=None, sigma2_init=None, sigma0_2=None):
    Y = np.asarray(obs, float)
    k, n = Y.shape
    tr_Y_Yt = np.sum(Y*Y)

    # --- Init U ---
    if est_U0:
        if U_init is not None:
            U = np.array(U_init, float)
        else:
            Uu, _, _ = np.linalg.svd(Y, full_matrices=False)
            U = Uu[:, :d]
    else:
        U = np.array(U0, float)

    # --- Init sigma0_2 ---
    if est_sigma0_2:
        sigma0_2_cur = float(np.log(1.5))
    else:
        sigma0_2_cur = float(sigma0_2)

    # --- Init rho, sigma2 ---
    if rho_init is None:
        rho = np.linspace(0.8, 0.99, d)
    else:
        rho = np.array(rho_init, float).reshape(-1)[:d]

    if sigma2_init is None:
        sigma2 = np.linspace(0.5, 1.0, d)
    else:
        sigma2 = np.array(sigma2_init, float).reshape(-1)[:d]

    # first posterior
    Y_tilde = U.T @ Y                    # (d,n)
    Z_hat = np.zeros_like(Y_tilde)       # (d,n)
    S_diag = np.zeros_like(Y_tilde)      # (d,n)
    B_off  = np.zeros((d, n-1))          # (d,n-1)
    for l in range(d):
        kf = kf_rts_ar1(1.0, sigma0_2_cur, rho[l], sigma2[l], Y_tilde[l, :])
        Z_hat[l, :] = kf['s_t']
        S_diag[l, :] = kf['S_t']
        B_off[l, :]  = kf['b_vec']

    pred_cur = U @ Z_hat
    pred_pre = pred_cur + 1.0            

    m = 1
    log_Q = np.zeros(d)
    while np.sqrt(np.sum((pred_cur - pred_pre)**2) / (k*n)) > threshold and m <= M:
        pred_pre = pred_cur.copy()

        # --- Update U (orthogonal Procrustes) ---
        if est_U0:
            ZYt = Z_hat @ Y.T            # (d,k)
            U1, _, Vt = np.linalg.svd(ZYt, full_matrices=False)
            U = Vt.T @ U1.T              # (k,d)
        else:
            U = np.array(U0, float)
        Y_tilde = U.T @ Y

        # --- Update sigma0_2 ---
        if est_sigma0_2:
            tr_ZYt = np.sum(Z_hat * Y_tilde)
            tr_ZtZ = np.sum(Z_hat*Z_hat)
            sigma0_2_cur = (tr_Y_Yt - 2*tr_ZYt + np.sum(S_diag) + tr_ZtZ) / (n*k)
            sigma0_2_cur = max(sigma0_2_cur, 1e-10)  
        else:
            sigma0_2_cur = float(sigma0_2)

        # --- Update rho, sigma2, and re-KF per latent dim ---
        for l in range(d):
            # cubic coefficients p3*rho^3 + p2*rho^2 + p1*rho + p0 = 0
            z_row = Z_hat[l, :]
            S_row = S_diag[l, :]
            b_row = B_off[l, :]

            # same as cpp code
            p3 = (n-1) * (np.sum(z_row[1:n-1]**2) + np.sum(S_row[1:n-1]))
            p2 = (2-n) * (np.dot(z_row[1:], z_row[:-1]) + np.sum(b_row))
            p1 = -np.sum(z_row**2) - np.sum(S_row) - n*(np.sum(z_row[1:n-1]**2) + np.sum(S_row[1:n-1]))
            p0 = n * (np.dot(z_row[1:], z_row[:-1]) + np.sum(b_row))

            rho_l = cubic_solver([p3, p2, p1, p0])
            if rho_l == -1.0:
                rho_l = rho[l]
            rho[l] = rho_l

            # sigma2 update
            term = (1 - rho_l*rho_l) * (z_row[0]**2)
            term += np.sum((z_row[1:] - rho_l*z_row[:-1])**2)
            term += np.sum(S_row)
            term += (rho_l*rho_l) * np.sum(S_row[1:n-1])
            term -= 2*rho_l * np.sum(b_row)
            sigma2_l = term / n
            sigma2[l] = max(sigma2_l, 1e-12)

            kf = kf_rts_ar1(1.0, sigma0_2_cur, rho_l, sigma2[l], Y_tilde[l, :])
            Z_hat[l, :] = kf['s_t']
            S_diag[l, :] = kf['S_t']
            B_off[l, :]  = kf['b_vec']
            log_Q[l] = np.sum(np.log(kf['Q_t'] + 1e-12))

        pred_cur = U @ Z_hat
        m += 1

    # predict mean/variance, diag(U * diag(S_t) * U^T)
    pred_mean_var = np.zeros_like(pred_cur)
    for t in range(n):
        # -wise sum of U * diag(S_diag[:,t]) * U^T diagonal
        pred_mean_var[:, t] = np.sum(U * (U * S_diag[:, t][None, :]), axis=1)

    out = {
        'mean_obs': pred_cur,
        'U': U,
        'sigma0_2': sigma0_2_cur,
        'rho': rho.copy(),
        'sigma2': sigma2.copy(),
        'post_z_mean': Z_hat,
        'post_z_var': S_diag,
        'post_z_cov': B_off,
        'mean_obs_95lb': pred_cur - 1.96*np.sqrt(np.maximum(pred_mean_var, 0.0)),
        'mean_obs_95ub': pred_cur + 1.96*np.sqrt(np.maximum(pred_mean_var, 0.0)),
        'num_iterations': m-1
    }
    # if track_neg_log_lik:
    #     pass
    return out

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


# ---------- 1) “observation mean” reality：same as R code :ReacTran + deSolve ----------
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

    A = sparse.eye(k, format="csr") - dt * D * Lap  # implicited Euler matirx
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
    equivalent: R code lattice_alg(...)：Gaussian process regression on (i, j) 2D grids。
    y: (k, n) matirx；return pred_mean, same shape with y
    kernel_type: "exp" -> RBF, "matern" -> Matern(nu=1.5)
    noise_var: observation noise variance（R : param_ini/optim , here we use sklearn）
    """
    H, W = y.shape
    xs = np.arange(H)
    ys = np.arange(W)
    X1, X2 = np.meshgrid(xs, ys, indexing="ij")
    X = np.column_stack([X1.ravel(), X2.ravel()])
    targ = y.ravel()

    if kernel_type == "exp":
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=(length_scale, length_scale),
                                           length_scale_bounds=(1e-3, 1e4)) \
                 + WhiteKernel(noise_level=max(noise_var, 1e-10),
                               noise_level_bounds=(1e-12, 1e1))
    elif kernel_type == "matern":
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=(length_scale, length_scale),
                                              length_scale_bounds=(1e-3, 1e4), nu=1.5) \
                 + WhiteKernel(noise_level=max(noise_var, 1e-10),
                               noise_level_bounds=(1e-12, 1e1))
    else:
        raise ValueError("kernel_type must be 'exp' or 'matern'")

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True,
                                   n_restarts_optimizer=1, copy_X_train=False, random_state=0)
    gpr.fit(X, targ)
    pred = gpr.predict(X)
    return pred.reshape(H, W)

# ---------- 3) hyperparameters ----------
k = 200
n = 200
num_repetition = 10
sigma0_list = [0.05, 0.1, 0.3]

# “observation” reality
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
