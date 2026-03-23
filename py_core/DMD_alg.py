import numpy as np
from scipy.linalg import pinv


def dmd_alg(
    output,
    method="ratio_sum",
    threshold=0.999,
    threshold_eigen_A_tilde=1e-8,
    s=1,
    d=1,
    fix_r=False,
    r=None,
):
    """
    Parameters
    ----------
    output : array-like, shape (k, n)
        Data matrix where columns are time snapshots.
    method : {"ratio_sum", "threshold"}
        How to choose rank when fix_r=False:
          - "ratio_sum": use cumulative sum ratio of singular values (matches R code)
          - otherwise: count singular values > threshold
    threshold : float
        For "ratio_sum": cumulative ratio threshold (e.g., 0.999)
        For "threshold": singular-value cutoff
    threshold_eigen_A_tilde : float
        Keep eigenvalues with abs(real_part) > this threshold.
    s : int
        Step size between snapshots in the reduced system (skip).
    d : int
        Number of timesteps stacked for HODMD. d=1 gives standard DMD.
    fix_r : bool
        If True, uses provided r.
    r : int or None
        Rank to use if fix_r=True.

    Returns
    -------
    res : dict
        Keys match R list:
          - A_hat
          - rank
          - in_sample_pred
          - rmse_in_sample
          - eigen_value
          - eigen_vector
          - SVD_X_d
          - Y
          - X
    """
    output = np.asarray(output, dtype=float)
    k, n = output.shape

    if d < 1:
        raise ValueError("d must be >= 1")
    if s < 1:
        raise ValueError("s must be >= 1")
    if n - d + 1 < 2:
        raise ValueError("Need n - d + 1 >= 2 to form X and Y.")

    # -----------------------------
    # 1) Build stacked output_tilde (d*k, n-d+1)
    # R:
    #   output_tilde[,i] = as.vector(output[,i:(i+d-1)])
    # Note: R uses column-major vectorization.
    # We'll replicate using order='F'.
    # -----------------------------
    ncols_tilde = n - d + 1
    output_tilde = np.empty((d * k, ncols_tilde), dtype=float)

    for i in range(ncols_tilde):
        block = output[:, i : i + d]          # shape (k, d)
        output_tilde[:, i] = block.reshape(-1, order="F")

    # -----------------------------
    # 2) Choose X and Y snapshots with skipping s
    # R:
    #   Y_index = seq(2, n-d+1, by=s)
    #   X = output_tilde[,(Y_index-1)]
    #   Y = output_tilde[, Y_index]
    # -----------------------------
    # Convert 1-based R indices to 0-based Python.
    # R Y_index starts at 2 => Python starts at 1.
    Y_index = np.arange(1, ncols_tilde, s)  # 0-based indices for Y
    X_index = Y_index - 1

    X = output_tilde[:, X_index]
    Y = output_tilde[:, Y_index]

    # -----------------------------
    # 3) SVD of X
    # R: SVD_X = svd(X)
    # -----------------------------
    U_full, svals, Vt_full = np.linalg.svd(X, full_matrices=False)
    # R's SVD_X$v is V (not Vt)
    V_full = Vt_full.T

    # -----------------------------
    # 4) Choose rank r
    # R:
    #   if(method=="ratio_sum"){
    #     sum_d=sum(SVD_X$d); cumsum_d=cumsum(SVD_X$d)
    #     r = sum(cumsum_d/sum_d <= threshold) + 1
    #   } else {
    #     r = sum(SVD_X$d > threshold)
    #   }
    # -----------------------------
    if fix_r:
        if r is None:
            raise ValueError("fix_r=True requires r to be provided.")
        r_use = int(r)
    else:
        if method == "ratio_sum":
            sum_d = np.sum(svals)
            cumsum_d = np.cumsum(svals)
            r_use = int(np.sum((cumsum_d / sum_d) <= threshold) + 1)
        else:
            r_use = int(np.sum(svals > threshold))

    # Clamp rank bounds
    r_use = max(1, min(r_use, svals.size))

    # -----------------------------
    # 5) Truncate U, D, V and compute A_tilde = U^T Y V D^{-1}
    # -----------------------------
    U = U_full[:, :r_use]
    D = np.diag(svals[:r_use])
    D_inv = np.diag(1.0 / svals[:r_use])
    V = V_full[:, :r_use]

    A_tilde = U.T @ Y @ V @ D_inv

    # -----------------------------
    # 6) Eigen-decomposition of A_tilde
    # R: eigen_A_tilde = eigen(A_tilde)
    # keep eigenvalues with abs(Re(val)) > threshold_eigen_A_tilde
    # -----------------------------
    eigvals, eigvecs = np.linalg.eig(A_tilde)
    keep = np.abs(np.real(eigvals)) > threshold_eigen_A_tilde
    non_zero_count = int(np.sum(keep))

    if non_zero_count == 0:
        # fallback: keep the largest magnitude eigenvalue
        idx = int(np.argmax(np.abs(eigvals)))
        keep = np.zeros_like(keep, dtype=bool)
        keep[idx] = True
        non_zero_count = 1

    eigvals_nz = eigvals[keep]
    eigvecs_nz = eigvecs[:, keep]

    # -----------------------------
    # 7) Compute phi (DMD modes)
    # R:
    # phi = Y V D_inv eigenvecs %*% diag(1/eigvals)
    #
    # NOTE: In R they do:
    #   ... %*% diag(1/eigen_A_tilde$values[...])
    # but implemented as: %*% diag(1/eigvals) with dimension non_zero_count
    # -----------------------------
    # Multiply by diag(1/eigvals): equivalent to column scaling
    inv_eigs = 1.0 / eigvals_nz
    phi = (Y @ V @ D_inv @ eigvecs_nz) * inv_eigs  # broadcast column-wise

    # Pseudoinverse (MASS::ginv)
    phi_inv = pinv(phi)

    # -----------------------------
    # 8) Build A_hat = phi diag(eigvals) phi_inv
    # and take real part if complex
    # -----------------------------
    A_hat = phi @ np.diag(eigvals_nz) @ phi_inv
    if np.iscomplexobj(A_hat):
        A_hat = np.real(A_hat)

    # -----------------------------
    # 9) In-sample prediction
    # R:
    # in_sample_pred[,i] = (A_hat %*% output_tilde[,i])[((d-1)*k+1):(d*k)]
    #
    # i = 1:(n-d)  (R)
    # Here output_tilde has (n-d+1) columns, we predict next (n-d) columns.
    # -----------------------------
    in_sample_pred = np.empty((k, n - d), dtype=float)

    start = (d - 1) * k
    end = d * k

    for i in range(n - d):
        vec = output_tilde[:, i]
        pred_full = A_hat @ vec
        in_sample_pred[:, i] = pred_full[start:end]

    # RMSE in R code is mean((pred - truth)^2) (actually MSE, not sqrt)
    truth = output[:, d:n]
    rmse_in_sample = float(np.mean((in_sample_pred - truth) ** 2))

    return {
        "A_hat": A_hat,
        "rank": r_use,
        "in_sample_pred": in_sample_pred,
        "rmse_in_sample": rmse_in_sample,
        "eigen_value": eigvals,
        "eigen_vector": phi,
        "SVD_X_d": svals,
        "Y": Y,
        "X": X,
    }
