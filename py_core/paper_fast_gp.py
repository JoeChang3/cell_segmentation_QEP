"""
`paper_fast_gp` -- a faithful port of the ORIGINAL paper's Fast-GP reconstruction.

Reference implementation (read-only):
  /Users/zchan/eclipse-workspace/cell_segmentation_original
  commit 44714c2e0be958fe796a8fd4bdbc220dae3c23dd
  src/Modified_Functions_RGasp.R

Paper: "Unsupervised Cellular Boundary Detection by Fast Gaussian Processes",
Baracaldo, King, Yan, Lin, Miolane, Gu (2025).

WHAT THIS MODULE IS
-------------------
The reconstruction stage ONLY. Thresholding, marker generation, watershed,
cleanup and metrics stay outside, exactly as they are today, so that
Raw / paper_fast_gp / QEP can be compared under identical postprocessing.

THE ORIGINAL REAL-DATA CALL GRAPH (traced, not inferred)
--------------------------------------------------------
`Nuclear_Real_Analysis/Nuclear_Data_Generate_IoU.R:205` and
`Whole_Cell_Real_Analysis/Whole_Cell_Data_Generate_IoU.R:204` both call
`generate_GP_Masks_test(file_path, nugget = T)` (L606), which does:

    for each tile (i, j):
        if (i == 1 && j == 1):
            parameters <- separable_GP_param_est(img_matrix)      # ONCE
        separable_GP_info <- separable_GP(img_matrix, parameters$param)

So `(beta1, beta2, nu)` are estimated on the FIRST tile and reused everywhere,
while `separable_GP` re-profiles the tile's own mean `theta_hat` (L245) from
that tile's data. That split is reproduced here as
`estimate_shared_params` + `reconstruct_tile`.

NOT the same function as the simulation code: `src/2dim_lattice_func.R::lattice_alg`
is used only by `Simulated_Experiments/*.R` and defaults to Nelder-Mead.
The real-data path uses L-BFGS-B (`Modified_Functions_RGasp.R:106`). Our
`py_core/dim_2_lattice.py` is a port of `lattice_alg`; this module reuses its
kernel and objective (one source of truth for the math) but reproduces the
real-data function's *structure* and *optimizer*.

PARAMETER DEFINITIONS (from the R source)
-----------------------------------------
  beta1, beta2  INVERSE range (not lengthscale) along axis 1 (rows) and axis 2
                (columns). Optimized as log beta, init log beta = -2.
  nu            Nugget, added to the Kronecker eigenvalues:
                    Lambda = kron(evals2, evals1) + nu
                Optimized as log nu, init log nu = -3. It is a RATIO to the
                signal variance, not an absolute noise level, because sigma^2 is
                profiled out.
  Matern 5/2    (1 + x + x^2/3) * exp(-x)  with  x = sqrt(5) * beta * d.
  inputs        input1 = seq(0, 1, 1/(n1-1)), input2 = seq(0, 1, 1/(n2-1)).
                NORMALIZED lattice coordinates -- so beta is dimensionless in
                units of the tile, and param_ini = (-2,-2,-3) is calibrated to
                this scale.
  mean          Constant basis X = 1. theta_hat is profiled analytically, PER
                TILE.
  variance      S_2 is profiled analytically and CANCELS OUT of the predictive
                mean, which is therefore scale-free. It is returned for
                diagnostics only. sigma2_hat = S_2 / N.
  objective     -(0.5*sum(log(Lambda_tilde_inv)) - N/2*log(S_2))
  no clipping   The original applies no clipping and no output transformation to
                predmean. z_lim at L271 feeds only a commented-out plot call.

ORIENTATION (verified empirically, not assumed)
-----------------------------------------------
`as.numeric(image_read(p)[[1]])` has dim (height, width, channels), so
`img_matrix <- as.numeric(cropped_img[[1]])[,,1]` is (rows, cols) -- natural
image orientation -- and n1 = crop_height, n2 = crop_width. The stitching at
L727-730 is therefore self-consistent. Intensities are magick's raw/255 on
channel 1 only.

The algorithm is covariant under transpose (transposing the input transposes the
output and swaps beta1/beta2), so feeding tiles in our native (row, col)
orientation matches the original.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh, solve
from scipy.optimize import minimize

from py_core.dim_2_lattice import matern_5_2, neg_log_lik_eigen_with_nugget
from py_core.Modified_Functions_RGasp import get_proportion

# The original real-data estimator: Modified_Functions_RGasp.R:100,106
PARAM_INI: Tuple[float, float, float] = (-2.0, -2.0, -3.0)
OPTIM_METHOD = "L-BFGS-B"
KERNEL_TYPE = "matern"          # Matern 5/2, the only branch the original uses
RNG_SEED = 1                    # R: set.seed(1) at Modified_Functions_RGasp.R:8


# A fitted beta whose effective range falls below this many pixels means the
# process is effectively uncorrelated along that axis -- the degenerate
# "beta -> infinity" plateau of the profiled likelihood. Observed on BOTH real
# development tiles when R's L-BFGS-B is used; see
# audits/PAPER_FAST_GP_WIRING_REPORT.md.
DEGENERATE_RANGE_PX = 0.5


@dataclass
class SharedParams:
    """(beta1, beta2, nu) estimated ONCE and reused across all tiles."""
    beta1: float
    beta2: float
    nugget: float
    n1: int
    n2: int
    source_tile: str
    neg_log_lik: float
    optim_method: str
    n_obj_evals: int
    runtime_sec: float

    def as_triple(self) -> np.ndarray:
        return np.array([self.beta1, self.beta2, self.nugget], dtype=np.float64)

    def effective_ranges_px(self) -> Tuple[float, float]:
        """Correlation range along each axis, in PIXELS of the source tile.

        beta is an inverse range on the normalized [0,1] lattice, so the range in
        pixels is (1/beta) * (n - 1).
        """
        r1 = (1.0 / self.beta1) * (self.n1 - 1) if self.beta1 > 0 else np.inf
        r2 = (1.0 / self.beta2) * (self.n2 - 1) if self.beta2 > 0 else np.inf
        return float(r1), float(r2)

    @property
    def degenerate_axes(self) -> List[str]:
        """Axes on which the fit collapsed to zero correlation length."""
        r1, r2 = self.effective_ranges_px()
        out = []
        if r1 < DEGENERATE_RANGE_PX:
            out.append("rows")
        if r2 < DEGENERATE_RANGE_PX:
            out.append("cols")
        return out


@dataclass
class TileReconstruction:
    """One tile's reconstruction plus the tile-specific profiled quantities."""
    pred_mean: np.ndarray
    theta_hat: float            # profiled constant mean, PER TILE
    s_2: float                  # profiled scale, cancels from pred_mean
    sigma2_hat: float           # S_2 / N
    n1: int
    n2: int
    runtime_sec: float
    diagnostics: Dict = field(default_factory=dict)


def _lattice_inputs(n1: int, n2: int) -> Tuple[np.ndarray, np.ndarray]:
    """R: seq(0, 1, 1/(n-1)). Normalized lattice coordinates."""
    return np.linspace(0.0, 1.0, n1), np.linspace(0.0, 1.0, n2)


def _distance_matrices(n1: int, n2: int) -> Tuple[np.ndarray, np.ndarray]:
    i1, i2 = _lattice_inputs(n1, n2)
    return np.abs(np.subtract.outer(i1, i1)), np.abs(np.subtract.outer(i2, i2))


def estimate_shared_params(
    tile: np.ndarray,
    param_ini: Tuple[float, float, float] = PARAM_INI,
    optim_method: str = OPTIM_METHOD,
    kernel_type: str = KERNEL_TYPE,
    source_tile: str = "tile(1,1)",
    max_retries: int = 5,
    n_restarts: int = 0,
) -> SharedParams:
    """Port of `separable_GP_param_est` (Modified_Functions_RGasp.R:3-122).

    Uses ALL pixels of the tile -- no subsampling, hence no RNG in the
    estimation itself. `set.seed(1)` in the original only matters for the
    `param_ini + runif(3)` retry path, which is mirrored here.

    `n_restarts = 0` (DEFAULT) is paper-faithful: a single L-BFGS-B run from
    `param_ini = (-2, -2, -3)`, exactly the original procedure.

    `n_restarts > 0` is a DELIBERATE DEVIATION, off by default. The profiled
    likelihood is non-convex with a flat `beta -> infinity` plateau, and on BOTH
    real development tiles the paper's single run ends at a degenerate,
    axis-aligned fit that is not the global optimum of its own objective
    (nuclei: f=129310.49 vs 116922.52 reachable; whole_cell: f=61119.24 vs
    41016.75). Set `n_restarts` to search for the better basin, and say so in
    any report -- it is a different estimator from the paper's.
    """
    tile = np.asarray(tile, dtype=np.float64)
    if tile.ndim != 2:
        raise ValueError(f"tile must be 2-D, got shape {tile.shape}")
    n1, n2 = tile.shape
    if n1 < 3 or n2 < 3:
        raise ValueError(f"tile too small for a lattice GP: {tile.shape}")
    N = n1 * n2

    R01, R02 = _distance_matrices(n1, n2)
    X_list = [np.ones((n1, n2), dtype=np.float64)]     # constant mean basis
    q_X = 1

    n_evals = [0]

    def obj(p: np.ndarray) -> float:
        n_evals[0] += 1
        try:
            v = neg_log_lik_eigen_with_nugget(
                p, kernel_type, R01, R02, N, q_X, X_list, tile)
        except Exception:
            return np.inf
        return float(v) if np.isfinite(v) else np.inf

    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    x0 = np.asarray(param_ini, dtype=np.float64)
    res = minimize(obj, x0, method=optim_method)
    tries = 0
    # R: while (!is.numeric(m_eigen[[1]])) retry at param_ini + runif(3)
    while (not np.all(np.isfinite(res.x))) and tries < max_retries:
        tries += 1
        res = minimize(obj, np.asarray(param_ini, dtype=np.float64) + rng.random(3),
                       method=optim_method)
    # opt-in, non-paper: search for a better basin of the non-convex objective
    for _ in range(max(0, n_restarts)):
        cand = minimize(obj, rng.uniform(-3.0, 5.0, 3), method=optim_method)
        if np.all(np.isfinite(cand.x)) and float(cand.fun) < float(res.fun):
            res = cand
    dt = time.time() - t0

    beta = np.exp(res.x[:2])
    nu = float(np.exp(res.x[2]))
    sp = SharedParams(
        beta1=float(beta[0]), beta2=float(beta[1]), nugget=nu,
        n1=n1, n2=n2, source_tile=source_tile,
        neg_log_lik=float(res.fun), optim_method=optim_method,
        n_obj_evals=n_evals[0], runtime_sec=dt)
    if sp.degenerate_axes:
        r1, r2 = sp.effective_ranges_px()
        warnings.warn(
            f"paper_fast_gp: degenerate fit on axis/axes {sp.degenerate_axes} "
            f"(effective range rows={r1:.3e} px, cols={r2:.3e} px). The profiled "
            f"likelihood has a flat 'beta -> infinity' plateau; the fit is "
            f"uncorrelated along that axis. This is the basin R's L-BFGS-B "
            f"reaches on both real development tiles.", RuntimeWarning,
            stacklevel=2)
    return sp


def reconstruct_tile(tile: np.ndarray, shared: SharedParams,
                     kernel_type: str = KERNEL_TYPE) -> TileReconstruction:
    """Port of `separable_GP` (Modified_Functions_RGasp.R:123-347).

    `beta1, beta2, nu` come from `shared`; `theta_hat` and `S_2` are re-profiled
    from THIS tile's data, matching the original (L245, L248).

    predmean = X %*% theta_hat + t(r1) %*% R_tilde_inv_output_normalize %*% r2

    with r1 = R1 and r2 = R2 (no nugget in the cross-covariance), so this is the
    shrinkage smoother R (R + nu I)^-1 (y - mean) + mean. sigma^2 cancels.

    The original also returns finite-difference gradients (grad1, grad2,
    grad_magnitude). `generate_GP_Masks_test` never reads them -- only
    `predmean_mat` -- so they are not computed here. Noted in diagnostics.
    """
    tile = np.asarray(tile, dtype=np.float64)
    n1, n2 = tile.shape
    N = n1 * n2
    beta1, beta2, nu = shared.beta1, shared.beta2, shared.nugget

    t0 = time.time()
    R01, R02 = _distance_matrices(n1, n2)
    if kernel_type != "matern":
        raise ValueError("the original real-data path uses Matern 5/2 only")
    R1 = matern_5_2(R01, beta=beta1)
    R2 = matern_5_2(R02, beta=beta2)

    evals1, evecs1 = eigh(R1)
    evals2, evecs2 = eigh(R2)

    # Lambda_tilde_inv = 1/(kron(evals2, evals1) + nu); column-major indexing
    lam = np.outer(evals2, evals1).reshape(-1, order="C")
    Lambda_tilde_inv = 1.0 / (lam + nu)

    X_mat = np.ones((n1, n2), dtype=np.float64)
    U_x = (evecs1.T @ X_mat @ evecs2).reshape(-1, order="F")
    Lam_U = Lambda_tilde_inv * U_x

    # theta_hat, profiled per tile
    XtRX = np.array([[float(U_x @ Lam_U)]])
    output_tilde = (evecs1.T @ tile @ evecs2).reshape(-1, order="F")
    theta_hat = float(solve(XtRX, np.array([Lam_U @ output_tilde]))[0])

    tile_norm = tile - theta_hat
    onl_tilde = (evecs1.T @ tile_norm @ evecs2).reshape(-1, order="F")
    s_2 = float(np.sum(onl_tilde * Lambda_tilde_inv * onl_tilde))

    onl_mat = (Lambda_tilde_inv * onl_tilde).reshape(n1, n2, order="F")
    R_inv_on = evecs1 @ onl_mat @ evecs2.T

    # testing_input == input, so r1 = R1 and r2 = R2
    pred_mean = theta_hat + (R1.T @ R_inv_on @ R2)
    dt = time.time() - t0

    return TileReconstruction(
        pred_mean=np.asarray(pred_mean, dtype=np.float64),
        theta_hat=theta_hat, s_2=s_2, sigma2_hat=s_2 / N,
        n1=n1, n2=n2, runtime_sec=dt,
        diagnostics=dict(
            beta1=beta1, beta2=beta2, nugget=nu,
            eff_range_rows_px=(1.0 / beta1) * (n1 - 1) if beta1 > 0 else np.inf,
            eff_range_cols_px=(1.0 / beta2) * (n2 - 1) if beta2 > 0 else np.inf,
            tile_min=float(tile.min()), tile_max=float(tile.max()),
            pred_min=float(pred_mean.min()), pred_max=float(pred_mean.max()),
            rmse_to_raw=float(np.sqrt(((pred_mean - tile) ** 2).mean())),
            gradients_computed=False,   # original computes but never consumes them
        ))


def tile_grid(height: int, width: int) -> Dict:
    """Replicate `generate_GP_Masks_test`'s tiling (L614-628) exactly.

    Note the original's remainder behavior: num_pieces_* = floor(size/crop), so
    up to crop-1 trailing rows/columns are never covered by any tile.
    """
    row_proportion = get_proportion(height)
    col_proportion = get_proportion(width)
    crop_width = int(width * col_proportion)
    crop_height = int(height * row_proportion)
    num_pieces_x = int(np.floor(width / crop_width))
    num_pieces_y = int(np.floor(height / crop_height))
    crop_width = width // num_pieces_x
    crop_height = height // num_pieces_y
    tiles = []
    for i in range(num_pieces_x):            # R loops i over x, then j over y
        for j in range(num_pieces_y):
            tiles.append(dict(i=i + 1, j=j + 1,
                              x_offset=i * crop_width, y_offset=j * crop_height,
                              h=crop_height, w=crop_width))
    return dict(row_proportion=row_proportion, col_proportion=col_proportion,
                crop_height=crop_height, crop_width=crop_width,
                num_pieces_x=num_pieces_x, num_pieces_y=num_pieces_y,
                tiles=tiles,
                covered_rows=num_pieces_y * crop_height,
                covered_cols=num_pieces_x * crop_width)


def reconstruct_image(
    image: np.ndarray,
    remainder: str = "raw",
    shared: Optional[SharedParams] = None,
    optim_method: str = OPTIM_METHOD,
    n_restarts: int = 0,
    verbose: bool = False,
) -> Dict:
    """Tiled paper-Fast-GP reconstruction of a whole image.

    Estimates (beta1, beta2, nu) on tile (1,1) and reuses them for every tile,
    re-profiling theta_hat per tile -- i.e. the original's behavior.

    INTENSITY SCALE. The original reads images through magick, so it works in
    [0,1] (raw/255); our Python loaders keep [0,255]. The estimator is exactly
    equivariant under y -> c*y: theta_hat -> c*theta_hat, S_2 -> c^2*S_2, and the
    objective shifts by the constant -N*log(c), so (beta1, beta2, nu) are
    INVARIANT and predmean scales by c. We therefore rescale to [0,1] internally
    to match magick bit-for-bit, and return the reconstruction in the SAME scale
    the caller passed in, so downstream code is unaffected either way.

    `remainder` controls the trailing rows/columns that the original's tiling
    never covers (it leaves them 0 in `combined_predmean`):
      "raw"  copy the raw pixels through  [default; avoids injecting a dark edge]
      "zero" leave them 0, bit-faithful to the original
    Either way the count is reported as `n_uncovered_px`.
    """
    raw = np.asarray(image, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {raw.shape}")
    # match magick's raw/255 convention; see the docstring for why this is exact
    scale = 255.0 if float(raw.max()) > 1.0 + 1e-9 else 1.0
    image = raw / scale

    H, W = image.shape
    grid = tile_grid(H, W)
    recon = image.copy() if remainder == "raw" else np.zeros_like(image)

    t0 = time.time()
    per_tile: List[Dict] = []
    for t in grid["tiles"]:
        sl = (slice(t["y_offset"], t["y_offset"] + t["h"]),
              slice(t["x_offset"], t["x_offset"] + t["w"]))
        sub = image[sl]
        if shared is None:                      # first tile in loop order
            shared = estimate_shared_params(
                sub, optim_method=optim_method, n_restarts=n_restarts,
                source_tile=f"tile(i={t['i']},j={t['j']})")
            if verbose:
                print(f"    shared params from {shared.source_tile}: "
                      f"beta1={shared.beta1:.6f} beta2={shared.beta2:.6f} "
                      f"nu={shared.nugget:.6f} ({shared.runtime_sec:.1f}s, "
                      f"{shared.n_obj_evals} evals)")
        rec = reconstruct_tile(sub, shared)
        recon[sl] = rec.pred_mean
        per_tile.append(dict(i=t["i"], j=t["j"],
                             y_offset=t["y_offset"], x_offset=t["x_offset"],
                             h=t["h"], w=t["w"], theta_hat=rec.theta_hat,
                             s_2=rec.s_2, sigma2_hat=rec.sigma2_hat,
                             runtime_sec=rec.runtime_sec, **rec.diagnostics))
    total = time.time() - t0

    n_unc = H * W - grid["covered_rows"] * grid["covered_cols"]
    recon_out = recon * scale          # back to the caller's intensity scale
    return dict(
        reconstruction=recon_out, shared_params=shared, per_tile=per_tile,
        grid=grid, runtime_sec=total, n_uncovered_px=int(n_unc),
        remainder=remainder, input_scale=scale,
        rmse_to_raw=float(np.sqrt(((recon_out - raw) ** 2).mean())),
    )
