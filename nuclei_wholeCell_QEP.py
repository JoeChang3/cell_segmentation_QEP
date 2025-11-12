import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import qpytorch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel

from pydmd import DMD


PATH_NUCLEI = "/Users/zchan/Desktop/projects/Cell_Seg_GP/code/cell_segmentation-main/Image_Data/simulation_cells/nuclei_2.png"
PATH_WHOLE  = "/Users/zchan/Desktop/projects/Cell_Seg_GP/code/cell_segmentation-main/Image_Data/simulation_cells/whole_cell_2.jpg"


# tool functions about fmou
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

def load_image_as_array(path, to_gray=True, normalize=True):
    """
    read image -> numpy array
    to_gray=True 
    normalize=True 
    """
    img = imageio.imread(path)
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:  # H x W x C
        if to_gray:
            # R code get [,,1] on nuclei_figure
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=2)
    # normalize
    if normalize:
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
    return arr

def add_gaussian_noise(arr, sigma):
    return arr + np.random.normal(scale=sigma, size=arr.shape)

def choose_rank_via_criterion(output_mat):
    """
    corresponding to the residues and penalties in r code
    only to ceil(n_rows * 2 / 3)
    """
    U, s, Vt = np.linalg.svd(output_mat, full_matrices=False)
    k, n = output_mat.shape
    max_d = int(math.ceil(output_mat.shape[0] * 2.0 / 3.0))
    losses = []
    for d in range(1, max_d + 1):
        Ud = U[:, :d]
        recon = Ud @ (Ud.T @ output_mat)
        resid = output_mat - recon
        crit = np.log(np.mean(resid**2) + 1e-12) + d * (k + n) / (k * n) * np.log(k * n / (k + n))
        losses.append(crit)
    est_d = 1 + int(np.argmin(losses))
    return est_d
# ---- New: QEP on grid (QePyTorch) ----

def fit_qep_on_grid(obs, kernel_type="exp", q_power=1.0,  
                    train_iters=200, lr=0.1, length_scale=10.0, nu=1.5,
                    device="cpu"):
    """
    use QePyTorch to train Q-Exponential Process and predict on the same grid。
    obs: 2D numpy array (H, W)
    return: 2D numpy array (H, W) – predictive mean
    """
    torch.set_default_dtype(torch.float64)  
    H, W = obs.shape
    xs = np.arange(H, dtype=np.float64)
    ys = np.arange(W, dtype=np.float64)
    X1, X2 = np.meshgrid(xs, ys, indexing="ij")
    X = np.column_stack([X1.ravel(), X2.ravel()])           # (N, 2)
    y = obs.ravel()                                         # (N,)

    train_x = torch.from_numpy(X).to(device = device, dtype=torch.get_default_dtype())
    train_y = torch.from_numpy(y).to(device = device, dtype=torch.get_default_dtype())

    #  ExactQEP 
    POWER = float(q_power)  # q=2 → Gaussian
    class ExactQEPModel(qpytorch.models.ExactQEP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.power = torch.tensor(POWER, dtype=train_x.dtype, device=train_x.device)
            self.mean_module = qpytorch.means.ConstantMean()
            
            ls = float(length_scale)
            if kernel_type == "exp":
                base_k = qpytorch.kernels.RBFKernel(ard_num_dims=2)
                base_k.lengthscale = torch.tensor([ls, ls], 
                                                  dtype=train_x.dtype, device=train_x.device)
            
            elif kernel_type == "matern":
                base_k = qpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=2)
                base_k.lengthscale = torch.tensor([ls,ls], 
                                                  dtype=train_x.dtype, device=train_x.device)

            else:
                raise ValueError("kernel_type must be 'rbf/exp' or 'matern'")

            self.covar_module = qpytorch.kernels.ScaleKernel(base_k)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return qpytorch.distributions.MultivariateQExponential(
                mean_x, covar_x, power=self.power
            )

    likelihood = qpytorch.likelihoods.QExponentialLikelihood(power=torch.tensor(POWER, dtype=train_x.dtype, device=device))
    model = ExactQEPModel(train_x, train_y, likelihood).to(device)

    # 训练（Type-II MLE）：ExactMarginalLogLikelihood
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(train_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # 预测
    model.eval(); likelihood.eval()
    with torch.no_grad(), qpytorch.settings.fast_pred_var():  # 可加速方差计算
        test_x = train_x
        pred = likelihood(model(test_x))
        # 取预测均值；若想要置信区间，可用 pred.confidence_region(rescale=True)
        mean = pred.mean.detach().cpu().numpy().reshape(H, W)
    return mean

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

sigma0_list = [0.1, 0.3, 0.5]
num_repetition = 10


DOWNSAMPLE = 0.1

# read images
nuclei = load_image_as_array(PATH_NUCLEI, to_gray=True, normalize=True)
whole  = load_image_as_array(PATH_WHOLE, to_gray=True, normalize=True)

def maybe_downsample(arr, scale):
    if abs(scale - 1.0) < 1e-9:
        return arr
    H, W = arr.shape
    newH, newW = max(1, int(H*scale)), max(1, int(W*scale))
    return np.array(Image.fromarray(arr).resize((newW, newH), resample=Image.BILINEAR))

nuclei = maybe_downsample(nuclei, DOWNSAMPLE)
whole  = maybe_downsample(whole,  DOWNSAMPLE)



# recoding the results
def init_result_containers(sigma_list, R):
    shape = (R, len(sigma_list))
    return (np.full(shape, np.nan),  # lattice_exp
            np.full(shape, np.nan),  # lattice_matern
            np.full(shape, np.nan),  # fmou
            np.full(shape, np.nan),  # pca
            np.full(shape, np.nan))  # dmd

(nuc_rmse_exp, nuc_rmse_mat, nuc_rmse_fmou, nuc_rmse_pca, nuc_rmse_dmd) = init_result_containers(sigma0_list, num_repetition)
(whl_rmse_exp, whl_rmse_mat, whl_rmse_fmou, whl_rmse_pca, whl_rmse_dmd) = init_result_containers(sigma0_list, num_repetition)

# only in  iter=1 record the visual middle process
nuc_y_record = [None] * len(sigma0_list)
nuc_pred_exp_record = [None] * len(sigma0_list)
nuc_pred_mat_record = [None] * len(sigma0_list)
nuc_pred_fmou_record = [None] * len(sigma0_list)
nuc_pred_pca_record  = [None] * len(sigma0_list)
nuc_pred_dmd_record  = [None] * len(sigma0_list)

whl_y_record = [None] * len(sigma0_list)
whl_pred_exp_record = [None] * len(sigma0_list)
whl_pred_mat_record = [None] * len(sigma0_list)
whl_pred_fmou_record = [None] * len(sigma0_list)
whl_pred_pca_record  = [None] * len(sigma0_list)
whl_pred_dmd_record  = [None] * len(sigma0_list)

nuc_est_d = np.full((num_repetition, len(sigma0_list)), np.nan, dtype=int)
whl_est_d = np.full((num_repetition, len(sigma0_list)), np.nan, dtype=int)




# main iteration：nuclei
# -----------------------------
for j, sigma_0 in enumerate(sigma0_list):
    for it in range(num_repetition):
        #  set.seed(iter) in r
        np.random.seed(it + 1)

        obs = add_gaussian_noise(nuclei, sigma_0)
        if it == 0:
            nuc_y_record[j] = obs

        # Lattice - exp kernel (use GPR RBF)
        pred_exp = fit_qep_on_grid(obs, kernel_type="exp", q_power=1.0, train_iters=150, lr=0.1, length_scale=max(5,min(nuclei.shape)), device="cpu")
        print('calling GPR with N=', obs.size, flush=True)

        nuc_rmse_exp[it, j] = rmse(nuclei, pred_exp)
        if it == 0:
            nuc_pred_exp_record[j] = pred_exp

        # Lattice - matern kernel (use GPR Matern)
        pred_mat = fit_qep_on_grid(obs, kernel_type="matern", q_power=1.0, train_iters=150, lr=0.1, length_scale=max(5,min(nuclei.shape)), nu= 1.5, device="cpu")
        nuc_rmse_mat[it, j] = rmse(nuclei, pred_mat)
        if it == 0:
            nuc_pred_mat_record[j] = pred_mat

        # choose latent dimension d
        d_hat = choose_rank_via_criterion(obs)
        nuc_est_d[it, j] = d_hat

        # FMOU（Python)
        fmou_out = fmou_predictive_mean(obs, d_hat, M=50, threshold=1e-4,
                                        est_U0=True, est_sigma0_2=True)
        pred_fmou = fmou_out['mean_obs']
        nuc_rmse_fmou[it, j] = rmse(nuclei, pred_fmou)
        if it == 0:
            nuc_pred_fmou_record[j] = pred_fmou

        # PCA
        pred_pca = pca_reconstruct(obs, d_hat)
        nuc_rmse_pca[it, j] = rmse(nuclei, pred_pca)
        if it == 0:
            nuc_pred_pca_record[j] = pred_pca

        # DMD
        pred_dmd = dmd_reconstruct(obs, r=d_hat)
        nuc_rmse_dmd[it, j] = rmse(nuclei, pred_dmd)
        if it == 0:
            nuc_pred_dmd_record[j] = pred_dmd


# main iterations：whole cell
# -----------------------------
for j, sigma_0 in enumerate(sigma0_list):
    for it in range(num_repetition):
        np.random.seed(it + 1)

        obs = add_gaussian_noise(whole, sigma_0)
        if it == 0:
            whl_y_record[j] = obs

        pred_exp = fit_qep_on_grid(obs, kernel_type="exp", q_power=1.0, train_iters=150, lr=0.1, length_scale=max(5,min(nuclei.shape)), device="cpu")

        print('calling GPR with N=', obs.size, flush=True)
        whl_rmse_exp[it, j] = rmse(whole, pred_exp)
        if it == 0:
            whl_pred_exp_record[j] = pred_exp

        pred_mat = fit_qep_on_grid(obs, kernel_type="matern", q_power=1.0, train_iters=150, lr=0.1, length_scale=max(5,min(nuclei.shape)), nu= 1.5, device="cpu")
        whl_rmse_mat[it, j] = rmse(whole, pred_mat)
        if it == 0:
            whl_pred_mat_record[j] = pred_mat

        d_hat = choose_rank_via_criterion(obs)
        whl_est_d[it, j] = d_hat

        # FMOU（Python）
        fmou_out = fmou_predictive_mean(obs, d_hat, M=50, threshold=1e-4,
                                        est_U0=True, est_sigma0_2=True)
        pred_fmou = fmou_out['mean_obs']
        nuc_rmse_fmou[it, j] = rmse(whole, pred_fmou)
        if it == 0:
            nuc_pred_fmou_record[j] = pred_fmou

        pred_pca = pca_reconstruct(obs, d_hat)
        whl_rmse_pca[it, j] = rmse(whole, pred_pca)
        if it == 0:
            whl_pred_pca_record[j] = pred_pca

        pred_dmd = dmd_reconstruct(obs, r=d_hat)
        whl_rmse_dmd[it, j] = rmse(whole, pred_dmd)
        if it == 0:
            whl_pred_dmd_record[j] = pred_dmd


nuc_rmse_summary = pd.DataFrame({
    0.1: [np.nanmean(nuc_rmse_exp[:,0]),
          np.nanmean(nuc_rmse_mat[:,0]),
          np.nanmean(nuc_rmse_fmou[:,0]),
          np.nanmean(nuc_rmse_pca[:,0]),
          np.nanmean(nuc_rmse_dmd[:,0])],
    0.3: [np.nanmean(nuc_rmse_exp[:,1]),
          np.nanmean(nuc_rmse_mat[:,1]),
          np.nanmean(nuc_rmse_fmou[:,1]),
          np.nanmean(nuc_rmse_pca[:,1]),
          np.nanmean(nuc_rmse_dmd[:,1])],
    0.5: [np.nanmean(nuc_rmse_exp[:,2]),
          np.nanmean(nuc_rmse_mat[:,2]),
          np.nanmean(nuc_rmse_fmou[:,2]),
          np.nanmean(nuc_rmse_pca[:,2]),
          np.nanmean(nuc_rmse_dmd[:,2])]
}, index=["lattice_exp","lattice_matern","fmou(placeholder)","PCA","DMD"])

whl_rmse_summary = pd.DataFrame({
    0.1: [np.nanmean(whl_rmse_exp[:,0]),
          np.nanmean(whl_rmse_mat[:,0]),
          np.nanmean(whl_rmse_fmou[:,0]),
          np.nanmean(whl_rmse_pca[:,0]),
          np.nanmean(whl_rmse_dmd[:,0])],
    0.3: [np.nanmean(whl_rmse_exp[:,1]),
          np.nanmean(whl_rmse_mat[:,1]),
          np.nanmean(whl_rmse_fmou[:,1]),
          np.nanmean(whl_rmse_pca[:,1]),
          np.nanmean(whl_rmse_dmd[:,1])],
    0.5: [np.nanmean(whl_rmse_exp[:,2]),
          np.nanmean(whl_rmse_mat[:,2]),
          np.nanmean(whl_rmse_fmou[:,2]),
          np.nanmean(whl_rmse_pca[:,2]),
          np.nanmean(whl_rmse_dmd[:,2])]
}, index=["lattice_exp","lattice_matern","fmou(placeholder)","PCA","DMD"])

print("\n=== nuclei_rmse_summary ===\n", nuc_rmse_summary)
print("\n=== wholeCell_rmse_summary ===\n", whl_rmse_summary)

# -----------------------------
# violin plot（using Matplotlib）
# -----------------------------
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

method_order = ["Fast-Mat","PCA","FMOU","DMD","Fast-Exp"]
def reorder(arrs):
    # [exp, mat, fmou, pca, dmd]
    # ["Fast-Mat","PCA","FMOU","DMD","Fast-Exp"]
    mapping = {"Fast-Mat":1, "PCA":3, "FMOU":2, "DMD":4, "Fast-Exp":0}
    return [arrs[mapping[m]] for m in method_order]

make_violin_plot(reorder([nuc_rmse_exp, nuc_rmse_mat, nuc_rmse_fmou, nuc_rmse_pca, nuc_rmse_dmd]),
                 method_order, sigma0_list, title="(A) Cell nuclei")
make_violin_plot(reorder([whl_rmse_exp, whl_rmse_mat, whl_rmse_fmou, whl_rmse_pca, whl_rmse_dmd]),
                 method_order, sigma0_list, title="(B) Whole cell")


# thermo diagram（nuclei/whole）
# -----------------------------
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

# nuclei: use Matern predictive mean（sigma0_list[0] case）
plot_triplet(nuclei, nuc_y_record[0], nuc_pred_mat_record[0],
             title_left="(A) Observation mean", title_mid="(B) Noisy observation",
             title_right="(C) Predictive mean", suptitle="Cell nuclei")

# whole: use Matern predictive mean（sigma0_list[2] case）
plot_triplet(whole, whl_y_record[2], whl_pred_mat_record[2],
             title_left="(D) Observation mean", title_mid="(E) Noisy observation",
             title_right="(F) Predictive mean", suptitle="Whole cell")
