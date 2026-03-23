###### Running time of lattice method
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from py_core.dim_2_lattice import neg_log_lik_eigen_with_nugget, matern_5_2, exp_kernel


# -----------------------------------------------
# Parameters (R: param_here = c(-2,-2,-3))
# -----------------------------------------------
param_here = np.array([-2.0, -2.0, -3.0])
beta = np.exp(param_here[:2])   # R: beta = exp(param_here[1:2])
nu   = np.exp(param_here[2])    # R: nu   = exp(param_here[3])

sample_size = list(range(10, 81, 10))   # R: seq(10, 80, 10)
num_iter    = 10

fast_comp_time_record_matern   = np.full((num_iter, len(sample_size)), np.nan)
direct_comp_time_record_matern = np.full((num_iter, len(sample_size)), np.nan)
fast_comp_time_record_exp      = np.full((num_iter, len(sample_size)), np.nan)
direct_comp_time_record_exp    = np.full((num_iter, len(sample_size)), np.nan)

fast_comp_log_lik_record_matern   = np.full((num_iter, len(sample_size)), np.nan)
direct_comp_log_lik_record_matern = np.full((num_iter, len(sample_size)), np.nan)
fast_comp_log_lik_record_exp      = np.full((num_iter, len(sample_size)), np.nan)
direct_comp_log_lik_record_exp    = np.full((num_iter, len(sample_size)), np.nan)


# -----------------------------------------------
# Main timing loop
# -----------------------------------------------
for iter_idx in range(num_iter):
    print(iter_idx + 1)   # R: print(iter), 1-indexed
    for i, n in enumerate(sample_size):
        print(i + 1)      # R: print(i), 1-indexed
        n1 = n
        n2 = n
        N  = n1 * n2

        # R: set.seed(iter) where iter is 1-based; rnorm fills column-major into matrix(rnorm(N),n1,n2)
        # Approximation: np.random.seed mirrors R's set.seed for the seed value, but the
        # RNG sequences differ between R and Python — numerical output values will not match R.
        np.random.seed(iter_idx + 1)
        output     = np.random.randn(N).reshape(n1, n2, order="F")   # R: matrix(rnorm(N),n1,n2)
        output_vec = output.reshape(-1, order="F")                    # R: as.vector(output), column-major

        input1 = np.arange(1, n1 + 1, dtype=float)   # R: 1:n1
        input2 = np.arange(1, n2 + 1, dtype=float)   # R: 1:n2
        R01 = np.abs(np.subtract.outer(input1, input1))   # R: abs(outer(input1, input1, "-"))
        R02 = np.abs(np.subtract.outer(input2, input2))

        # R: X = matrix(1,N,1); X_list[[1]] = matrix(X[,1],n1,n2)
        X      = np.ones((N, 1), dtype=float)
        q_X    = 1
        X_list = [X[:, 0].reshape(n1, n2, order="F")]

        ### 1. Matern ------------------------------------------
        ####### 1.1 fast computation
        # Approximation: time.perf_counter() measures wall-clock time; R's system.time()[1]
        # measures user CPU time. Results are comparable on single-threaded runs but may differ.
        t0 = time.perf_counter()
        fast_comp_log_lik_record_matern[iter_idx, i] = neg_log_lik_eigen_with_nugget(
            param=param_here, kernel_type="matern",
            R01=R01, R02=R02, N=N, q_X=q_X, X_list=X_list, output_mat=output,
        )
        fast_comp_time_record_matern[iter_idx, i] = time.perf_counter() - t0

        ####### 1.2 direct computation
        t0 = time.perf_counter()
        R1 = matern_5_2(R01, beta=beta[0])
        R2 = matern_5_2(R02, beta=beta[1])
        R_tilde     = np.kron(R1, R2) + nu * np.eye(N)   # R: kronecker(R1,R2) + diag(nu,N,N)
        inv_R_tilde = np.linalg.inv(R_tilde)
        one_vec     = np.ones(N)
        # R calls solve(R_tilde) a second time inside mu_hat; we reuse inv_R_tilde (same result)
        mu_hat = (one_vec @ inv_R_tilde @ output_vec) / (one_vec @ inv_R_tilde @ one_vec)
        resid  = output_vec - mu_hat
        S2     = resid @ inv_R_tilde @ resid
        # R: determinant(R_tilde)$modulus[1] = log|det(R_tilde)|
        log_det = np.linalg.slogdet(R_tilde)[1]
        direct_comp_log_lik_record_matern[iter_idx, i] = log_det / 2.0 + (N / 2.0) * np.log(S2)
        direct_comp_time_record_matern[iter_idx, i] = time.perf_counter() - t0

        ### 2. Exp ------------------------------------------
        ####### 2.1 fast computation
        t0 = time.perf_counter()
        fast_comp_log_lik_record_exp[iter_idx, i] = neg_log_lik_eigen_with_nugget(
            param=param_here, kernel_type="exp",
            R01=R01, R02=R02, N=N, q_X=q_X, X_list=X_list, output_mat=output,
        )
        fast_comp_time_record_exp[iter_idx, i] = time.perf_counter() - t0

        ####### 2.2 direct computation
        t0 = time.perf_counter()
        R1 = exp_kernel(R01, beta=beta[0])
        R2 = exp_kernel(R02, beta=beta[1])
        R_tilde     = np.kron(R1, R2) + nu * np.eye(N)
        inv_R_tilde = np.linalg.inv(R_tilde)
        one_vec     = np.ones(N)
        mu_hat = (one_vec @ inv_R_tilde @ output_vec) / (one_vec @ inv_R_tilde @ one_vec)
        resid  = output_vec - mu_hat
        S2     = resid @ inv_R_tilde @ resid
        log_det = np.linalg.slogdet(R_tilde)[1]
        direct_comp_log_lik_record_exp[iter_idx, i] = log_det / 2.0 + (N / 2.0) * np.log(S2)
        direct_comp_time_record_exp[iter_idx, i] = time.perf_counter() - t0


# -----------------------------------------------
# Averages over first 4 iterations (R: colMeans(...[1:4,]))
# -----------------------------------------------
direct_comp_time_record_matern_avg = np.mean(direct_comp_time_record_matern[:4, :], axis=0)
direct_comp_time_record_exp_avg    = np.mean(direct_comp_time_record_exp[:4, :],    axis=0)
fast_comp_time_record_matern_avg   = np.mean(fast_comp_time_record_matern[:4, :],   axis=0)
fast_comp_time_record_exp_avg      = np.mean(fast_comp_time_record_exp[:4, :],      axis=0)


# -----------------------------------------------
# Build long-form dataframe (R: ggplot_df)
# -----------------------------------------------
N_labels = [f"{n}^2" for n in sample_size]   # ["10^2", "20^2", ..., "80^2"]

ggplot_df = pd.DataFrame({
    "time": np.concatenate([
        direct_comp_time_record_matern_avg,
        direct_comp_time_record_exp_avg,
        fast_comp_time_record_matern_avg,
        fast_comp_time_record_exp_avg,
    ]),
    "N": N_labels * 4,
    "Methods": (
        ["Direct-Matern"] * len(sample_size) +
        ["Direct-Exp"]    * len(sample_size) +
        ["Fast-Matern"]   * len(sample_size) +
        ["Fast-Exp"]      * len(sample_size)
    ),
})
ggplot_df["N"] = pd.Categorical(ggplot_df["N"], categories=N_labels, ordered=True)

color_map = {
    "Direct-Matern": "darkgreen",
    "Direct-Exp":    "orange",
    "Fast-Matern":   "#578FCA",
    "Fast-Exp":      "#FFC0CB",
}
marker_map = {
    "Direct-Matern": "o",
    "Direct-Exp":    "s",
    "Fast-Matern":   "^",
    "Fast-Exp":      "D",
}
methods_order = ["Direct-Matern", "Direct-Exp", "Fast-Matern", "Fast-Exp"]

# Approximation: x-axis tick labels use matplotlib LaTeX strings to match R's expression(n^{2})
x_tick_labels = [rf"${n}^2$" for n in sample_size]


# -----------------------------------------------
# f_1: linear time scale  |  f_2: log10 time scale
# R: gridExtra::grid.arrange(f_1, f_2, nrow=1)
# -----------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 2.5))

for ax, use_log in [(ax1, False), (ax2, True)]:
    for method in methods_order:
        sub = ggplot_df[ggplot_df["Methods"] == method].sort_values("N")
        y   = np.log10(sub["time"].values) if use_log else sub["time"].values
        ax.plot(
            range(len(sample_size)), y,
            color=color_map[method],
            marker=marker_map[method],
            markersize=1.5,
            linewidth=1.0,
            label=method,
        )
    ax.set_xticks(range(len(sample_size)))
    ax.set_xticklabels(x_tick_labels, fontsize=12)
    ax.set_ylabel("time (sec)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)

# f_1: legend at approx (0.25, 0.75) of axes (R: legend.position = c(0.25, 0.75))
ax1.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=12,
    frameon=True,
    facecolor="white",
    edgecolor="white",
    title=None,
)

# f_2: no legend (R: legend.position = "none"); custom y-axis labels
ax2.set_yticks([-2, 0, 2])
ax2.set_yticklabels([r"$10^{-2}$", r"$10^0$", r"$10^2$"], fontsize=12)

plt.tight_layout()
plt.savefig("compare_comp_time.pdf", bbox_inches="tight")
plt.close()
print("Saved: compare_comp_time.pdf")
