import numpy as np


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


def _diag_check(arr, tag, raise_on_nonfinite=True):
    """Diagnostic helper: print shape/stats and optionally raise on non-finite."""
    bad = ~np.isfinite(arr)
    n_bad = bad.sum()
    if n_bad > 0:
        idx = np.argwhere(bad)
        vals = arr[bad]
        print(f"  [FMOU DIAG] {tag}: NONFINITE n={n_bad}/{arr.size}, "
              f"first_idx={idx[0].tolist()}, first_val={vals[0]:.6g}")
        if raise_on_nonfinite:
            raise ValueError(f"[FMOU] Non-finite values at stage '{tag}': "
                             f"{n_bad} of {arr.size} elements, first index {idx[0].tolist()}")
    return n_bad == 0


def fmou_predictive_mean(obs, d, M=50, threshold=1e-4,
                         est_U0=True, est_sigma0_2=True,
                         U0=None, U_init=None, rho_init=None, sigma2_init=None, sigma0_2=None,
                         sigma0_2_init=None):
    """
    FMOU EM algorithm.

    Parameters
    ----------
    sigma0_2_init : float or None
        Initial value for sigma0_2 when est_sigma0_2=True.
        Pass this explicitly to preserve experiment-specific initialization:
          - branin experiments: pass max(0.1 * np.var(obs), 1e-6)
          - all others: leave as None to use the default np.log(1.5)
    """
    # ---- DIAG 1: function entry ----
    Y = np.asarray(obs, float)
    k, n = Y.shape
    print(f"[FMOU DIAG] entry: shape=({k},{n}), dtype={Y.dtype}, "
          f"finite={np.isfinite(Y).all()}, "
          f"min={np.nanmin(Y):.4g}, max={np.nanmax(Y):.4g}")
    _diag_check(Y, "entry/Y")

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

    _diag_check(U, "init/U")

    # --- Init sigma0_2 ---
    # sigma0_2_init must be passed explicitly by branin callers to preserve
    # their data-adaptive initialization (0.1 * np.var(Y), floored at 1e-6).
    # All other experiments use the fixed default np.log(1.5).
    if est_sigma0_2:
        if sigma0_2_init is not None:
            sigma0_2_cur = float(sigma0_2_init)
        else:
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

    print(f"[FMOU DIAG] init params: d={d}, sigma0_2={sigma0_2_cur:.6g}, "
          f"rho=[{rho[0]:.4g}..{rho[-1]:.4g}], sigma2=[{sigma2[0]:.4g}..{sigma2[-1]:.4g}]")

    # ---- first posterior ----
    Y_tilde = U.T @ Y                    # (d,n)
    _diag_check(Y_tilde, "init/Y_tilde")

    Z_hat = np.zeros_like(Y_tilde)       # (d,n)
    S_diag = np.zeros_like(Y_tilde)      # (d,n)
    B_off  = np.zeros((d, n-1))          # (d,n-1)
    for l in range(d):
        kf = kf_rts_ar1(1.0, sigma0_2_cur, rho[l], sigma2[l], Y_tilde[l, :])
        # ---- DIAG 4: initial KF outputs ----
        for key in ('s_t', 'S_t', 'b_vec', 'm_t', 'C_t', 'f_t', 'Q_t'):
            arr = kf[key]
            if not np.isfinite(arr).all():
                _diag_check(arr, f"init_kf/l={l}/{key}")
        Z_hat[l, :] = kf['s_t']
        S_diag[l, :] = kf['S_t']
        B_off[l, :]  = kf['b_vec']

    _diag_check(Z_hat, "init/Z_hat")
    _diag_check(S_diag, "init/S_diag")

    pred_cur = U @ Z_hat
    pred_pre = np.full_like(pred_cur, np.inf)

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

        _diag_check(U,       f"iter{m}/U")
        _diag_check(Y_tilde, f"iter{m}/Y_tilde")

        # --- Update sigma0_2 ---
        if est_sigma0_2:
            tr_ZYt = np.sum(Z_hat * Y_tilde)
            tr_ZtZ = np.sum(Z_hat*Z_hat)
            sigma0_2_cur = (tr_Y_Yt - 2*tr_ZYt + np.sum(S_diag) + tr_ZtZ) / (n*k)
            sigma0_2_cur = max(sigma0_2_cur, 1e-10)
        else:
            sigma0_2_cur = float(sigma0_2)

        if not np.isfinite(sigma0_2_cur):
            raise ValueError(f"[FMOU] Non-finite sigma0_2 at iter {m}: {sigma0_2_cur}")

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

            # ---- DIAG 3: cubic_solver ----
            rho_l = cubic_solver([p3, p2, p1, p0])
            if not np.isfinite(rho_l):
                raise ValueError(f"[FMOU] cubic_solver returned non-finite rho={rho_l} "
                                 f"at iter {m}, l={l} "
                                 f"(coeffs p3={p3:.4g}, p2={p2:.4g}, p1={p1:.4g}, p0={p0:.4g})")
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
            # ---- DIAG 4: per-iteration KF outputs ----
            for key in ('s_t', 'S_t', 'b_vec', 'm_t', 'C_t', 'f_t', 'Q_t'):
                arr = kf[key]
                if not np.isfinite(arr).all():
                    _diag_check(arr, f"iter{m}_kf/l={l}/{key}")
            Z_hat[l, :] = kf['s_t']
            S_diag[l, :] = kf['S_t']
            B_off[l, :]  = kf['b_vec']
            log_Q[l] = np.sum(np.log(kf['Q_t'] + 1e-12))

        _diag_check(Z_hat,   f"iter{m}/Z_hat")
        _diag_check(S_diag,  f"iter{m}/S_diag")

        pred_cur = U @ Z_hat
        _diag_check(pred_cur, f"iter{m}/pred_cur")
        m += 1

    # predict mean/variance, diag(U * diag(S_t) * U^T)
    pred_mean_var = np.zeros_like(pred_cur)
    for t in range(n):
        # -wise sum of U * diag(S_diag[:,t]) * U^T diagonal
        pred_mean_var[:, t] = np.sum(U * (U * S_diag[:, t][None, :]), axis=1)

    # ---- DIAG 5: pre-return check ----
    if not np.isfinite(pred_cur).all():
        bad = ~np.isfinite(pred_cur)
        idx = np.argwhere(bad)
        vals = pred_cur[bad]
        print(f"[FMOU DIAG] pre-return mean_obs: {bad.sum()} non-finite values")
        print(f"  first 5 indices: {idx[:5].tolist()}")
        print(f"  first 5 values:  {vals[:5].tolist()}")
        raise ValueError(f"[FMOU] Non-finite mean_obs before return: "
                         f"{bad.sum()} of {pred_cur.size} elements")

    print(f"[FMOU DIAG] done: {m-1} iters, sigma0_2={sigma0_2_cur:.6g}, "
          f"rho=[{rho[0]:.4g}..{rho[-1]:.4g}]")

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
