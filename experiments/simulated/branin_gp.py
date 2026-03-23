import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from py_core.dim_2_lattice import lattice_alg
from py_core.DMD_alg import dmd_alg
from py_core.fmou import cubic_solver, kf_rts_ar1, fmou_predictive_mean                 


# -----------------------------
# 1) Branin function (same as R)
# -----------------------------
def branin(xx, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    x1, x2 = xx
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s*(1-t)*np.cos(x1)
    return term1 + term2 + s


# -----------------------------
# 2) Choose latent dimension d (same criterion loop as R)
# -----------------------------
def choose_d_via_criterion(output_mat: np.ndarray) -> int:
    # R:
    # criteria_val = log(mean((Y - U_d U_d^T Y)^2)) + d*(k+n)/(k*n)*log(k*n/(k+n))
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

    est_d = int(np.argmin(loss_score) + 1)  # +1 because Python is 0-based
    return est_d


def fmou_fit_mean_obs(output_mat: np.ndarray, d: int):
    # branin uses a data-adaptive sigma0_2 init: 0.1 * Var(Y), floored at 1e-6
    fit = fmou_predictive_mean(output_mat, d=d, M=50, threshold=1e-4, est_U0=True, est_sigma0_2=True,
                               sigma0_2_init=max(0.1 * np.var(output_mat), 1e-6))
    return fit["mean_obs"]   # shape must be (n1, n2)


# -----------------------------
# 4) Utilities: RMSE + heatmap
# -----------------------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plot_heatmap(mat: np.ndarray, title: str, ax):
    im = ax.imshow(mat.T, origin="lower", aspect="auto")  # mimic image2D(t(mat))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# -----------------------------
# 5) Main experiment (matches your R loops)
# -----------------------------
def run_experiment(
    n1=100,
    n2=100,
    num_repetition=10,
    sigma0_list=(1, 5, 10),
    seed_base=1,
):
    N = n1 * n2

    # Domain (same as R)
    LB = np.array([-5.0, 0.0])
    UB = np.array([10.0, 15.0])
    rng = UB - LB

    input1 = LB[0] + rng[0] * np.linspace(0, 1, n1)  # row
    input2 = LB[1] + rng[1] * np.linspace(0, 1, n2)  # col

    # Build full input grid in same ordering as R:
    # input=cbind(rep(input1,n2), as.vector(t(matrix(input2,n2,n1))))
    # Equivalent: for each col in 1..n2, repeat input1; and input2 repeated in blocks
    X1 = np.repeat(input1, n2)
    X2 = np.tile(input2, n1)  # because t(matrix(input2,n2,n1)) pattern -> this matches
    # input = np.column_stack([X1, X2])

    # True function f
    f = np.array([branin((X1[i], X2[i])) for i in range(N)], dtype=float)
    f_mat = f.reshape(n1, n2, order="F")  # IMPORTANT: match R's matrix(f,n1,n2)

    # Storage like R
    rmse_lattice_exp = np.full((num_repetition, len(sigma0_list)), np.nan)
    rmse_lattice_matern = np.full((num_repetition, len(sigma0_list)), np.nan)
    rmse_fmou = np.full((num_repetition, len(sigma0_list)), np.nan)
    rmse_pca = np.full((num_repetition, len(sigma0_list)), np.nan)
    rmse_dmd = np.full((num_repetition, len(sigma0_list)), np.nan)

    y_record = [None] * len(sigma0_list)
    pred_mean_lattice_exp_record = [None] * len(sigma0_list)
    pred_mean_lattice_matern_record = [None] * len(sigma0_list)
    pred_mean_fmou_record = [None] * len(sigma0_list)
    pred_mean_pca_record = [None] * len(sigma0_list)
    pred_mean_dmd_record = [None] * len(sigma0_list)

    est_d_record = np.full((num_repetition, len(sigma0_list)), np.nan)

    # Loop over noise level and repetitions (same structure as R)
    for j, sigma_0 in enumerate(sigma0_list):
        for it in range(num_repetition):
            np.random.seed(it + 1)  

            # noisy observation
            output = f + np.random.normal(0, sigma_0, size=N)
            output_mat = output.reshape(n1, n2, order="F")

            if it == 0:
                y_record[j] = output_mat.copy()

            # -------- Lattice exp (Fast-Exp) --------
            fit_exp = lattice_alg(
                output_mat=output_mat,
                input1=input1,
                input2=input2,
                kernel_type="exp",
                testing_input1=input1,
                testing_input2=input2,
                param_ini=np.array([-2, -2, -3], dtype=float),
                optim_method="L-BFGS-B",
            )
            pred_exp = fit_exp["pred_mean"]
            rmse_lattice_exp[it, j] = rmse(f_mat, pred_exp)
            if it == 0:
                pred_mean_lattice_exp_record[j] = pred_exp.copy()

            # -------- Lattice matern (Fast-Mat) --------
            fit_mat = lattice_alg(
                output_mat=output_mat,
                input1=input1,
                input2=input2,
                kernel_type="matern",
                testing_input1=input1,
                testing_input2=input2,
                param_ini=np.array([-2, -2, -3], dtype=float),
                optim_method="Nelder-Mead",
            )
            pred_mat = fit_mat["pred_mean"]
            rmse_lattice_matern[it, j] = rmse(f_mat, pred_mat)
            if it == 0:
                pred_mean_lattice_matern_record[j] = pred_mat.copy()

            # -------- choose d --------
            est_d = choose_d_via_criterion(output_mat)
            est_d_record[it, j] = est_d

            # -------- FMOU --------
            pred_fmou = fmou_fit_mean_obs(output_mat, d=est_d)
            rmse_fmou[it, j] = rmse(f_mat, pred_fmou)
            if it == 0:
                pred_mean_fmou_record[j] = pred_fmou.copy()

            # -------- PCA --------
            U, svals, Vt = np.linalg.svd(output_mat, full_matrices=False)
            Ud = U[:, :est_d]
            pred_pca = Ud @ (Ud.T @ output_mat)
            rmse_pca[it, j] = rmse(f_mat, pred_pca)
            if it == 0:
                pred_mean_pca_record[j] = pred_pca.copy()

            # -------- DMD (fix r = est_d) --------
            fit_dmd = dmd_alg(output_mat, r=est_d, fix_r=True)
            # R compares cbind(output_mat[,1], in_sample_pred) against f_mat
            pred_dmd_full = np.column_stack([output_mat[:, 0], fit_dmd["in_sample_pred"]])
            rmse_dmd[it, j] = rmse(f_mat, pred_dmd_full)
            if it == 0:
                pred_mean_dmd_record[j] = pred_dmd_full.copy()

    # Summary table (same as R)
    rmse_summary = np.vstack([
        np.nanmean(rmse_lattice_exp, axis=0),
        np.nanmean(rmse_lattice_matern, axis=0),
        np.nanmean(rmse_fmou, axis=0),   # will be NaN until you plug FMOU
        np.nanmean(rmse_pca, axis=0),
        np.nanmean(rmse_dmd, axis=0),
    ])
    methods = ["lattice_exp", "lattice_matern", "fmou", "PCA", "DMD"]
    rmse_summary_df = pd.DataFrame(rmse_summary, index=methods, columns=list(sigma0_list))

    return {
        "f_mat": f_mat,
        "input1": input1,
        "input2": input2,
        "sigma0_list": sigma0_list,
        "rmse_summary": rmse_summary_df,
        "rmse_lattice_exp": rmse_lattice_exp,
        "rmse_lattice_matern": rmse_lattice_matern,
        "rmse_fmou": rmse_fmou,
        "rmse_pca": rmse_pca,
        "rmse_dmd": rmse_dmd,
        "y_record": y_record,
        "pred_mean_lattice_matern_record": pred_mean_lattice_matern_record,
    }


# -----------------------------
# 6) Plot: violin RMSE (Figure 5A analog)
# -----------------------------
def plot_rmse_violin(res, out_png="rmse_violin.png"):
    sigma0_list = list(res["sigma0_list"])
    num_rep = res["rmse_lattice_exp"].shape[0]

    # Build a long dataframe like R
    def stack_rmse(arr, method_name):
        # arr shape: (rep, len(sigma))
        rows = []
        for j, sig in enumerate(sigma0_list):
            for i in range(num_rep):
                rows.append({"RMSE": arr[i, j], "Methods": method_name, "sigma0": sig})
        return rows

    rows = []
    rows += stack_rmse(res["rmse_lattice_exp"], "Fast-Exp")
    rows += stack_rmse(res["rmse_lattice_matern"], "Fast-Mat")
    rows += stack_rmse(res["rmse_fmou"], "FMOU")
    rows += stack_rmse(res["rmse_pca"], "PCA")
    rows += stack_rmse(res["rmse_dmd"], "DMD")

    df = pd.DataFrame(rows).dropna()

    # Order like R: Fast-Mat, PCA, FMOU, DMD, Fast-Exp
    method_order = ["Fast-Mat", "PCA", "FMOU", "DMD", "Fast-Exp"]
    df["Methods"] = pd.Categorical(df["Methods"], categories=method_order, ordered=True)

    # Facet-like: one panel per sigma0
    fig, axes = plt.subplots(1, len(sigma0_list), figsize=(12, 4), sharey=False)

    for ax, sig in zip(axes, sigma0_list):
        sub = df[df["sigma0"] == sig]
        data = [sub[sub["Methods"] == m]["RMSE"].values for m in method_order if m in sub["Methods"].unique()]

        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        ax.set_title(rf"$\sigma_0={sig}$", fontsize=12)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([m for m in method_order if m in sub["Methods"].unique()], rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("RMSE")
    fig.suptitle("(A) Branin", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# -----------------------------
# 7) Plot: Figure 6(A) analog triplet
# -----------------------------
def plot_triplet(res, sigma_index=2, out_png="signal_obs_pred_branin.png"):
    # sigma_index=2 means sigma0_list[2] in R => the 3rd one (sigma0=10)
    f_mat = res["f_mat"]
    y_mat = res["y_record"][sigma_index]
    pred_mat = res["pred_mean_lattice_matern_record"][sigma_index]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plot_heatmap(f_mat, "(A) Observation mean", axes[0])
    plot_heatmap(y_mat, "(B) Noisy observation", axes[1])
    plot_heatmap(pred_mat, "(C) Predictive mean", axes[2])
    fig.suptitle("Branin", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


if __name__ == "__main__":
    res = run_experiment()
    print(res["rmse_summary"])

    # Violin plot (Figure 5A analog)
    plot_rmse_violin(res, out_png="rmse_violin.png")

    # Triplet plot (Figure 6A analog) - default uses sigma0_list[2]
    plot_triplet(res, sigma_index=2, out_png="signal_obs_pred_branin.png")

    print("Saved: rmse_violin.png, signal_obs_pred_branin.png")
