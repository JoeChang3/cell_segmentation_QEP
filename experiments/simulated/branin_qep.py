import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from py_core.DMD_alg import dmd_alg
from py_core.fmou import cubic_solver, kf_rts_ar1, fmou_predictive_mean
from linear_operator import settings as linop_settings
import qpytorch


# ============================================================
# 0) Branin function (same as R)
# ============================================================
def branin(xx, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    x1, x2 = xx
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s*(1-t)*np.cos(x1)
    return term1 + term2 + s


# ============================================================
# 1) Choose latent dimension d 
# ============================================================
def choose_d_via_criterion(output_mat: np.ndarray) -> int:
    U, svals, Vt = np.linalg.svd(output_mat, full_matrices=False)
    k, n = output_mat.shape

    loss_score = []
    max_d = int(np.ceil(k * 2 / 3))
    for d in range(1, max_d + 1):
        Ud = U[:, :d]
        proj = Ud @ (Ud.T @ output_mat)
        mse = np.mean((output_mat - proj) ** 2)
        crit = np.log(mse) + d * (k + n) / (k * n) * np.log((k * n) / (k + n))
        loss_score.append(crit)

    return int(np.argmin(loss_score) + 1)


def fmou_fit_mean_obs(output_mat: np.ndarray, d: int) -> np.ndarray:
    # branin uses a data-adaptive sigma0_2 init: 0.1 * Var(Y), floored at 1e-6
    fit = fmou_predictive_mean(output_mat, d=d, M=50, threshold=1e-4, est_U0=True, est_sigma0_2=True,
                               sigma0_2_init=max(0.1 * np.var(output_mat), 1e-6))
    return fit["mean_obs"]


# ============================================================
# 4) QEP model (SKI / KISS-QEP)
# ============================================================
class ExactSKIQEPModel(qpytorch.models.ExactQEP):
    def __init__(self, train_x, train_y, likelihood, power, base_kernel, grid_size=64):
        super().__init__(train_x, train_y, likelihood)
        self.power = power
        self.mean_module = qpytorch.means.ConstantMean()

        self.covar_module = qpytorch.kernels.ScaleKernel(
            qpytorch.kernels.GridInterpolationKernel(
                base_kernel,
                grid_size=grid_size,
                num_dims=train_x.size(-1),
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return qpytorch.distributions.MultivariateQExponential(mean_x, covar_x, power=self.power)

@dataclass
class QEPFitConfig:
    power: float = 1.5          # 2.0 -> Gaussian-like; <2 heavier tails
    grid_size: int = 40
    lr: float = 0.08
    training_iter: int = 25
    use_cuda: bool = False
    max_cg_iter: int = 120     # for SKI solve
    cg_tolerance: float = 5e-2
    jitter: float = 1e-3


def fit_qep_predictive_mean_on_grid(
    train_x_np: np.ndarray,
    train_y_np: np.ndarray,
    kernel_type: str,
    cfg: QEPFitConfig,
) -> np.ndarray:
    """
    Train a SKI Exact QEP on (train_x, train_y), return predictive mean at train_x.
    train_x_np: (N,2)
    train_y_np: (N,)
    """
    import torch
    import gpytorch
    from linear_operator import settings as linop_settings
    import qpytorch

    device = torch.device("cuda") if (cfg.use_cuda and torch.cuda.is_available()) else torch.device("cpu")
    dtype = torch.float32

    train_x = torch.tensor(train_x_np, dtype=dtype, device=device)
    train_y = torch.tensor(train_y_np, dtype=dtype, device=device)

    power = torch.tensor(cfg.power, dtype=dtype, device=device)

    likelihood = qpytorch.likelihoods.QExponentialLikelihood(power=power).to(device)

    if kernel_type.lower() == "rbf":
        base_kernel = qpytorch.kernels.RBFKernel().to(device)
    elif kernel_type.lower() == "matern":
        base_kernel = qpytorch.kernels.MaternKernel(nu=2.5).to(device)  # Matern 5/2 analogue
    else:
        raise ValueError("kernel_type must be 'rbf' or 'matern'")

    model = ExactSKIQEPModel(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        power=power,
        base_kernel=base_kernel,
        grid_size=cfg.grid_size,
    ).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with linop_settings.max_cg_iterations(cfg.max_cg_iter), \
         linop_settings.cg_tolerance(cfg.cg_tolerance), \
         gpytorch.settings.cholesky_jitter(cfg.jitter):

        # ---- Train ----
        for i in range(cfg.training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # ---- Predict (in-sample mean) ----
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            pred = likelihood(model(train_x))
            mean = pred.mean

    return mean.detach().cpu().numpy()

# ============================================================
# 5) Utilities: RMSE + plots
# ============================================================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plot_heatmap(mat: np.ndarray, title: str, ax):
    im = ax.imshow(mat.T, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# ============================================================
# 6) Main experiment (QEP replaces lattice_exp/lattice_matern)
# ============================================================
def run_experiment_qep(
    n1=100,
    n2=100,
    num_repetition=2,
    sigma0_list=(1, 5),
    qep_cfg: QEPFitConfig = QEPFitConfig(),
):
    print("Starting experiment...")
    N = n1 * n2

    LB = np.array([-5.0, 0.0])
    UB = np.array([10.0, 15.0])
    rng = UB - LB

    input1 = LB[0] + rng[0] * np.linspace(0, 1, n1)
    input2 = LB[1] + rng[1] * np.linspace(0, 1, n2)

    X1 = np.repeat(input1, n2)
    X2 = np.tile(input2, n1)

    # train_x in (N,2)
    train_x = np.column_stack([X1, X2])

    # IMPORTANT: scale features roughly to [-1, 1] for SKI
    # (ScaleToBounds exists in the model, but giving reasonable scale helps)
    train_x_scaled = (train_x - train_x.mean(axis=0)) / (train_x.std(axis=0) + 1e-12)
    train_x_scaled = np.clip(train_x_scaled / 3.0, -1.0, 1.0)

    f = np.array([branin((train_x[i, 0], train_x[i, 1])) for i in range(N)], dtype=float)
    f_mat = f.reshape(n1, n2, order="F")

    rmse_qep_rbf = np.full((num_repetition, len(sigma0_list)), np.nan)
    rmse_qep_matern = np.full((num_repetition, len(sigma0_list)), np.nan)
    rmse_fmou = np.full((num_repetition, len(sigma0_list)), np.nan)
    rmse_pca = np.full((num_repetition, len(sigma0_list)), np.nan)
    rmse_dmd = np.full((num_repetition, len(sigma0_list)), np.nan)

    y_record = [None] * len(sigma0_list)
    pred_mean_qep_matern_record = [None] * len(sigma0_list)

    est_d_record = np.full((num_repetition, len(sigma0_list)), np.nan)

    for j, sigma_0 in enumerate(sigma0_list):
        for it in range(num_repetition):
            np.random.seed(it + 1)

            y = f + np.random.normal(0, sigma_0, size=N)
            output_mat = y.reshape(n1, n2, order="F")

            if it == 0:
                y_record[j] = output_mat.copy()

            # -------- QEP-RBF --------
            pred_rbf_vec = fit_qep_predictive_mean_on_grid(
                train_x_np=train_x_scaled,
                train_y_np=y,
                kernel_type="rbf",
                cfg=qep_cfg
            )
            pred_rbf_mat = pred_rbf_vec.reshape(n1, n2, order="F")
            rmse_qep_rbf[it, j] = rmse(f_mat, pred_rbf_mat)

            # -------- QEP-Matern (nu=2.5 ~ Matern 5/2) --------
            pred_mat_vec = fit_qep_predictive_mean_on_grid(
                train_x_np=train_x_scaled,
                train_y_np=y,
                kernel_type="matern",
                cfg=qep_cfg
            )
            pred_mat = pred_mat_vec.reshape(n1, n2, order="F")
            rmse_qep_matern[it, j] = rmse(f_mat, pred_mat)
            if it == 0:
                pred_mean_qep_matern_record[j] = pred_mat.copy()

            # -------- choose d --------
            est_d = choose_d_via_criterion(output_mat)
            est_d_record[it, j] = est_d

            # -------- FMOU --------
            pred_fmou = fmou_fit_mean_obs(output_mat, d=est_d)
            rmse_fmou[it, j] = rmse(f_mat, pred_fmou)

            # -------- PCA --------
            U, svals, Vt = np.linalg.svd(output_mat, full_matrices=False)
            Ud = U[:, :est_d]
            pred_pca = Ud @ (Ud.T @ output_mat)
            rmse_pca[it, j] = rmse(f_mat, pred_pca)

            # -------- DMD (fix r = est_d) --------
            fit_dmd = dmd_alg(output_mat, r=est_d, fix_r=True)
            pred_dmd_full = np.column_stack([output_mat[:, 0], fit_dmd["in_sample_pred"]])
            rmse_dmd[it, j] = rmse(f_mat, pred_dmd_full)

    rmse_summary = np.vstack([
        np.nanmean(rmse_qep_rbf, axis=0),
        np.nanmean(rmse_qep_matern, axis=0),
        np.nanmean(rmse_fmou, axis=0),
        np.nanmean(rmse_pca, axis=0),
        np.nanmean(rmse_dmd, axis=0),
    ])

    methods = ["QEP-RBF", "QEP-Matern", "FMOU", "PCA", "DMD"]
    rmse_summary_df = pd.DataFrame(rmse_summary, index=methods, columns=list(sigma0_list))

    return {
        "f_mat": f_mat,
        "input1": input1,
        "input2": input2,
        "sigma0_list": sigma0_list,
        "rmse_summary": rmse_summary_df,
        "rmse_qep_rbf": rmse_qep_rbf,
        "rmse_qep_matern": rmse_qep_matern,
        "rmse_fmou": rmse_fmou,
        "rmse_pca": rmse_pca,
        "rmse_dmd": rmse_dmd,
        "y_record": y_record,
        "pred_mean_qep_matern_record": pred_mean_qep_matern_record,
    }


def plot_rmse_violin(res, out_png="rmse_violin_qep.png"):
    sigma0_list = list(res["sigma0_list"])
    num_rep = res["rmse_qep_rbf"].shape[0]

    def stack_rmse(arr, method_name):
        rows = []
        for j, sig in enumerate(sigma0_list):
            for i in range(num_rep):
                rows.append({"RMSE": arr[i, j], "Methods": method_name, "sigma0": sig})
        return rows

    rows = []
    rows += stack_rmse(res["rmse_qep_rbf"], "QEP-RBF")
    rows += stack_rmse(res["rmse_qep_matern"], "QEP-Matern")
    rows += stack_rmse(res["rmse_fmou"], "FMOU")
    rows += stack_rmse(res["rmse_pca"], "PCA")
    rows += stack_rmse(res["rmse_dmd"], "DMD")

    df = pd.DataFrame(rows).dropna()
    method_order = ["QEP-Matern", "PCA", "FMOU", "DMD", "QEP-RBF"]
    df["Methods"] = pd.Categorical(df["Methods"], categories=method_order, ordered=True)

    fig, axes = plt.subplots(1, len(sigma0_list), figsize=(12, 4), sharey=False)
    for ax, sig in zip(axes, sigma0_list):
        sub = df[df["sigma0"] == sig]
        data = [sub[sub["Methods"] == m]["RMSE"].values for m in method_order if m in sub["Methods"].unique()]
        ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        ax.set_title(rf"$\sigma_0={sig}$", fontsize=12)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([m for m in method_order if m in sub["Methods"].unique()],
                           rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("RMSE")
    fig.suptitle("(A) Branin (QEP)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_triplet(res, sigma_index=1, out_png="signal_obs_pred_branin_qep.png"):
    f_mat = res["f_mat"]
    y_mat = res["y_record"][sigma_index]
    pred_mat = res["pred_mean_qep_matern_record"][sigma_index]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plot_heatmap(f_mat, "(A) Observation mean", axes[0])
    plot_heatmap(y_mat, "(B) Noisy observation", axes[1])
    plot_heatmap(pred_mat, "(C) Predictive mean (QEP-Matern)", axes[2])
    fig.suptitle("Branin (QEP)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


if __name__ == "__main__":
    # You can tune POWER here:
    #   power=2.0 -> Gaussian-like (baseline)
    #   power<2.0 -> heavier tails (more robust)
    cfg = QEPFitConfig(
        power=1.2,
        grid_size=40,
        lr=0.08,
        training_iter=20,
        use_cuda=False,   # set True if you have CUDA torch
        max_cg_iter=1200,
        cg_tolerance=1e-3,
    )

    res = run_experiment_qep(qep_cfg=cfg)
    print(res["rmse_summary"])

    plot_rmse_violin(res, out_png="rmse_violin_qep.png")
    plot_triplet(res, sigma_index=1, out_png="signal_obs_pred_branin_qep.png")
    print("Saved: rmse_violin_qep.png, signal_obs_pred_branin_qep.png")