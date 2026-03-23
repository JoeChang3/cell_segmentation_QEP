import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from pydmd import DMD
from py_core.fmou import cubic_solver, kf_rts_ar1, fmou_predictive_mean
from py_core.dim_2_lattice import lattice_alg as _lattice_alg


PATH_NUCLEI = "/Users/zchan/Desktop/projects/Cell_Seg_GP/code/cell_segmentation-main/Image_Data/simulation_cells/nuclei_2.png"
PATH_WHOLE  = "/Users/zchan/Desktop/projects/Cell_Seg_GP/code/cell_segmentation-main/Image_Data/simulation_cells/whole_cell_2.jpg"


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

def fit_gpr_on_grid(obs, kernel_type="exp", noise_level=1e-3, length_scale=10.0, nu=1.5):
    """
    R equivalent: lattice_alg(output_mat, 1:nrow, 1:ncol, kernel_type=..., param_ini=c(-2,-2,2))
    from 2dim_lattice_func.R. Uses separable Kronecker GP via eigendecomposition.
    obs: (H, W) matrix; returns pred_mean of same shape.
    kernel_type: "exp" -> exp(-beta*d), "matern" -> Matern 5/2
    noise_level, length_scale, nu are unused; hyperparameters are estimated by MLE.
    """
    H, W = obs.shape
    input1 = np.arange(1, H + 1, dtype=float)  # R: 1:nrow
    input2 = np.arange(1, W + 1, dtype=float)  # R: 1:ncol
    result = _lattice_alg(
        output_mat=obs,
        input1=input1,
        input2=input2,
        kernel_type=kernel_type,
        param_ini=(-2.0, -2.0, 2.0),  # R: c(-2,-2,2) for nuclei_wholeCell
    )
    return result["pred_mean"]

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
        pred_exp = fit_gpr_on_grid(obs, kernel_type="exp", noise_level=(sigma_0**2), length_scale=max(5, min(nuclei.shape)))
        print('calling GPR with N=', obs.size, flush=True)

        nuc_rmse_exp[it, j] = rmse(nuclei, pred_exp)
        if it == 0:
            nuc_pred_exp_record[j] = pred_exp

        # Lattice - matern kernel (use GPR Matern)
        pred_mat = fit_gpr_on_grid(obs, kernel_type="matern", noise_level=(sigma_0**2), length_scale=max(5, min(nuclei.shape)), nu=1.5)
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

        pred_exp = fit_gpr_on_grid(obs, kernel_type="exp", noise_level=(sigma_0**2), length_scale=max(5, min(whole.shape)))
        print('calling GPR with N=', obs.size, flush=True)
        whl_rmse_exp[it, j] = rmse(whole, pred_exp)
        if it == 0:
            whl_pred_exp_record[j] = pred_exp

        pred_mat = fit_gpr_on_grid(obs, kernel_type="matern", noise_level=(sigma_0**2), length_scale=max(5, min(whole.shape)), nu=1.5)
        whl_rmse_mat[it, j] = rmse(whole, pred_mat)
        if it == 0:
            whl_pred_mat_record[j] = pred_mat

        d_hat = choose_rank_via_criterion(obs)
        whl_est_d[it, j] = d_hat

        # FMOU（Python）
        fmou_out = fmou_predictive_mean(obs, d_hat, M=50, threshold=1e-4,
                                        est_U0=True, est_sigma0_2=True)
        pred_fmou = fmou_out['mean_obs']
        whl_rmse_fmou[it, j] = rmse(whole, pred_fmou)
        if it == 0:
            whl_pred_fmou_record[j] = pred_fmou

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
