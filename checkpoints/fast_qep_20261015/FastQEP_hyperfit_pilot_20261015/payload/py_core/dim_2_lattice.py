import numpy as np
from numpy.linalg import eigh, solve
from scipy.optimize import minimize


# -----------------------------
# Kernels 
# -----------------------------
def matern_5_2(d, beta):
    """
    R:
      x = sqrt(5)*beta*d
      (1+x+x^2/3)*exp(-x)
    """
    x = np.sqrt(5.0) * beta * d
    return (1.0 + x + (x**2) / 3.0) * np.exp(-x)


def exp_kernel(d, beta):
    """
    R:
      x = beta*d
      exp(-x)
    """
    x = beta * d
    return np.exp(-x)


# -----------------------------
# Negative log-likelihood (eigen + nugget)
# -----------------------------
def neg_log_lik_eigen_with_nugget(param, kernel_type, R01, R02, N, q_X, X_list, output_mat):
    """
    param = (log(beta1), log(beta2), log(nu))
    kernel_type in {"matern", "exp"}
    output_mat is n1 x n2
    """

    n1 = R01.shape[0]
    n2 = R02.shape[0]

    beta = np.exp(param[:2])
    nu = np.exp(param[2])

    # Build R1, R2
    if kernel_type == "matern":
        R1 = matern_5_2(R01, beta=beta[0])
        R2 = matern_5_2(R02, beta=beta[1])
    elif kernel_type == "exp":
        R1 = exp_kernel(R01, beta=beta[0])
        R2 = exp_kernel(R02, beta=beta[1])
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    # eigen decomposition (R uses eigen; for symmetric PSD use eigh)
    evals1, evecs1 = eigh(R1)
    evals2, evecs2 = eigh(R2)

    U_x = np.empty((N, q_X), dtype=np.float64)
    for i_q in range(q_X):
        tmp = evecs1.T @ X_list[i_q] @ evecs2
        U_x[:, :] = tmp.reshape(-1, order="F")[:, None]  # replicate overwrite

    # Lambda_tilde_inv = 1/(kron(evals2, evals1) + nu)
    lam = np.outer(evals2, evals1).reshape(-1, order="C")
    Lambda_tilde_inv = 1.0 / (lam + nu)

    # Lambda_tilde_inv_U_x = Lambda_tilde_inv * U_x (broadcast rowwise)
    Lambda_tilde_inv_U_x = Lambda_tilde_inv[:, None] * U_x

    # X_R_tilde_inv_X_inv = solve(t(U_x) %*% (Lambda_tilde_inv_U_x))
    XtRX = U_x.T @ Lambda_tilde_inv_U_x
    X_R_tilde_inv_X_inv = solve(XtRX, np.eye(XtRX.shape[0]))

    # output_tilde = as.vector(t(evecs1) %*% output_mat %*% evecs2)
    output_tilde = (evecs1.T @ output_mat @ evecs2).reshape(-1, order="F")

    # theta_hat = X_R_tilde_inv_X_inv %*% (t(Lambda_tilde_inv_U_x) %*% output_tilde)
    theta_hat = X_R_tilde_inv_X_inv @ (Lambda_tilde_inv_U_x.T @ output_tilde)

    # output_mat_normalized = matrix(output - X%*%theta_hat, n1, n2)
    # X is all ones, shape (N,1)
    output_vec = output_mat.reshape(-1, order="F")
    output_mat_normalized = (output_vec - theta_hat[0]).reshape(n1, n2, order="F")

    # output_normalize_tilde = as.vector(t(evecs1)%*% output_mat_normalized %*% evecs2)
    output_normalize_tilde = (evecs1.T @ output_mat_normalized @ evecs2).reshape(-1, order="F")

    # S_2 = sum(output_normalize_tilde * Lambda_tilde_inv * output_normalize_tilde)
    S_2 = np.sum(output_normalize_tilde * Lambda_tilde_inv * output_normalize_tilde)

    # -(1/2*sum(log(Lambda_tilde_inv)) - N/2*log(S_2))
    # -> negative log marginal likelihood (up to constants) like R
    val = -(0.5 * np.sum(np.log(Lambda_tilde_inv)) - (N / 2.0) * np.log(S_2))
    return val


# -----------------------------
# Main lattice algorithm (fit + predictive mean)
# -----------------------------
def lattice_alg(
    output_mat,
    input1,
    input2,
    kernel_type="matern",
    testing_input1=None,
    testing_input2=None,
    param_ini=(-2.0, -2.0, -3.0),
    optim_method="Nelder-Mead",
):
    """
    Python version of R lattice_alg.
    output_mat: (n1,n2)
    input1: length n1
    input2: length n2
    testing_input1/testing_input2 default to training grid
    """
    output_mat = np.asarray(output_mat, dtype=np.float64)
    input1 = np.asarray(input1, dtype=np.float64)
    input2 = np.asarray(input2, dtype=np.float64)

    n1 = input1.size
    n2 = input2.size
    N = n1 * n2

    if testing_input1 is None:
        testing_input1 = input1
    if testing_input2 is None:
        testing_input2 = input2

    testing_input1 = np.asarray(testing_input1, dtype=np.float64)
    testing_input2 = np.asarray(testing_input2, dtype=np.float64)

    # Constant mean basis X=1
    X = np.ones((N, 1), dtype=np.float64)
    q_X = X.shape[1]
    X_list = [X[:, i].reshape(n1, n2, order="F") for i in range(q_X)]

    # Distances
    R01 = np.abs(input1[:, None] - input1[None, :])
    R02 = np.abs(input2[:, None] - input2[None, :])

    # ---- optimize parameters ----
    def obj(p):
        return neg_log_lik_eigen_with_nugget(p, kernel_type, R01, R02, N, q_X, X_list, output_mat)

    res = minimize(
        fun=obj,
        x0=np.array(param_ini, dtype=np.float64),
        method=optim_method,
        options={"maxiter": 2000, "disp": False},
    )

    # replicate R's retry logic if optimizer fails
    tries = 0
    while (not res.success) and tries < 10:
        tries += 1
        x0 = np.array(param_ini, dtype=np.float64) + np.random.rand(3)
        res = minimize(fun=obj, x0=x0, method=optim_method, options={"maxiter": 2000, "disp": False})

    if not res.success:
        raise RuntimeError(f"Optimization failed after retries: {res.message}")

    p_opt = res.x
    beta = np.exp(p_opt[:2])
    nu = np.exp(p_opt[2])

    # ---- build predictive mean ----
    if kernel_type == "matern":
        R1 = matern_5_2(R01, beta=beta[0])
        R2 = matern_5_2(R02, beta=beta[1])
    else:
        R1 = exp_kernel(R01, beta=beta[0])
        R2 = exp_kernel(R02, beta=beta[1])

    evals1, evecs1 = eigh(R1)
    evals2, evecs2 = eigh(R2)

    U_x = np.empty((N, q_X), dtype=np.float64)
    for i_q in range(q_X):
        tmp = evecs1.T @ X_list[i_q] @ evecs2
        U_x[:, :] = tmp.reshape(-1, order="F")[:, None]

    lam = np.outer(evals2, evals1).reshape(-1, order="C")
    Lambda_tilde_inv = 1.0 / (lam + nu)

    Lambda_tilde_inv_U_x = Lambda_tilde_inv[:, None] * U_x
    XtRX = U_x.T @ Lambda_tilde_inv_U_x
    X_R_tilde_inv_X_inv = solve(XtRX, np.eye(XtRX.shape[0]))

    output_tilde = (evecs1.T @ output_mat @ evecs2).reshape(-1, order="F")
    theta_hat = X_R_tilde_inv_X_inv @ (Lambda_tilde_inv_U_x.T @ output_tilde)

    output_vec = output_mat.reshape(-1, order="F")
    output_mat_normalized = (output_vec - theta_hat[0]).reshape(n1, n2, order="F")

    # test distances
    r01 = np.abs(input1[:, None] - testing_input1[None, :])
    r02 = np.abs(input2[:, None] - testing_input2[None, :])

    if kernel_type == "matern":
        r1 = matern_5_2(r01, beta=beta[0])
        r2 = matern_5_2(r02, beta=beta[1])
    else:
        r1 = exp_kernel(r01, beta=beta[0])
        r2 = exp_kernel(r02, beta=beta[1])

    output_normalize_tilde = (evecs1.T @ output_mat_normalized @ evecs2).reshape(-1, order="F")
    output_normalized_tilde_lambda_inv_mat = (Lambda_tilde_inv * output_normalize_tilde).reshape(n1, n2, order="F")

    R_tilde_inv_output_normalize_mat = evecs1 @ output_normalized_tilde_lambda_inv_mat @ evecs2.T

    # predmean = X_testing*theta_hat + as.vector(t(r1)%*%R_inv%*%r2)
    predmean = theta_hat[0] + (r1.T @ R_tilde_inv_output_normalize_mat @ r2)
    predmean_mat = np.asarray(predmean, dtype=np.float64)

    return {
        "beta": beta,
        "nu": nu,
        "pred_mean": predmean_mat,
        "opt_log_params": p_opt,
    }
