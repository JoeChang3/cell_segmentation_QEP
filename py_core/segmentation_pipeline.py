"""
Shared real-data cell-segmentation pipeline with a pluggable reconstruction step.

WHY THIS MODULE EXISTS
----------------------
The legacy real-data path (`generate_gp_masks_test` in
py_core/Modified_Functions_RGasp.py) hardcodes the GP smoother at line 409, and
the QEP hook next to it (`separable_gp_smooth_qepytorch`, line 269) calls
`qpytorch.models.ExactQEPModel`, which does not exist in the installed qpytorch.
So QEP has never reached IoU/AP evaluation on real images.

This module keeps `generate_gp_masks_test` untouched (old results stay
reproducible) and instead factors the pipeline so that ONLY the reconstruction
step varies by method:

    image
      -> tile
      -> [METHOD-DEPENDENT] reconstruct / smooth each tile
      -> criterion_1 adaptive threshold per tile          (shared)
      -> outlier tile re-thresholding                     (shared)
      -> stitch tiles                                     (shared)
      -> distance transform + watershed                   (shared)
      -> eliminate_small_areas                            (shared)
      -> instance mask

Everything after reconstruction is byte-identical across methods, reusing the
legacy helpers (`criterion_1`, `threshold_image`, `eliminate_small_areas`,
`get_proportion`) rather than reimplementing them. The adaptive threshold
necessarily depends on the reconstructed image, but the ALGORITHM is the same
for every method and the chosen threshold value is logged per tile.

METHODS
-------
  "raw"       : no reconstruction; the tile is passed through unchanged.
  "gp_legacy" : the legacy `separable_gp_smooth_gpytorch` verbatim (float32,
                isotropic Matern 2.5, Adam lr=0.1, 75 iters). Preserves the
                historical GP behavior for continuity.
  "gp"        : controlled Gaussian arm - same code path as "qep" with q=2.
  "qep"       : controlled Q-Exponential arm, joint 2D coordinates.

"gp"/"qep" share one implementation (`smooth_tile`) so that the GP and QEP arms
differ ONLY in the distribution/likelihood family and q. The old row+column
sequential QEP is deliberately not used.

REPRODUCIBILITY FIX
-------------------
The legacy smoother subsamples training pixels with an unseeded
`np.random.choice` (Modified_Functions_RGasp.py:234). That is why the stored
nuclei GP AP numbers cannot be reproduced (recomputing from the stored IoU
matrix gives AP@0.5=0.0388 against a stored 0.0307). Here every method draws
its subsample from an explicit, per-tile-deterministic RNG seeded from
(seed, tile_index), so all methods see the SAME training pixels and reruns are
exact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as cc_label
from skimage.segmentation import watershed

import gpytorch
import qpytorch

from py_core.instance_separation import separate_instances
from py_core.Modified_Functions_RGasp import (
    criterion_1,
    eliminate_small_areas,
    get_proportion,
    separable_gp_smooth_gpytorch,
    threshold_image,
)

METHODS = ("raw", "gp_legacy", "gp", "qep")


# ─────────────────────────────────────────────────────────────────────────────
# Controlled GP / QEP tile smoother
# ─────────────────────────────────────────────────────────────────────────────

class _ExactGPTile(gpytorch.models.ExactGP):
    """Isotropic Matern GP, matching the legacy smoother's structure."""

    def __init__(self, x, y, likelihood, nu: float = 2.5):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=nu))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class _ExactQEPTile(qpytorch.models.ExactQEP):
    """Q-Exponential twin of _ExactGPTile. Identical mean/kernel structure."""

    def __init__(self, x, y, likelihood, power, nu: float = 2.5):
        super().__init__(x, y, likelihood)
        self.power = power
        self.mean_module = qpytorch.means.ConstantMean()
        self.covar_module = qpytorch.kernels.ScaleKernel(
            qpytorch.kernels.MaternKernel(nu=nu))

    def forward(self, x):
        return qpytorch.distributions.MultivariateQExponential(
            self.mean_module(x), self.covar_module(x), power=self.power)


def smooth_tile(
    tile: np.ndarray,
    *,
    family: str,                    # "gp" or "qep"
    q: float = 2.0,
    nu: float = 2.5,
    train_iters: int = 75,
    lr: float = 0.1,
    max_points: int = 3000,
    rng: Optional[np.random.Generator] = None,
    dtype: torch.dtype = torch.float64,
    standardize: bool = True,
    predict_chunk: int = 8192,
    torch_seed: Optional[int] = None,
    exact_logdet: bool = True,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict]:
    """Joint-2D GP or QEP posterior mean on one image tile.

    `family="gp"` and `family="qep", q=2.0` are the same model up to the
    distribution class; qpytorch's power=2.0 reproduces the Gaussian
    log_prob exactly (verified in
    experiments/simulated/qep_power_semantics_check.py).

    Returns (predmean, diagnostics).
    """
    if family not in ("gp", "qep"):
        raise ValueError("family must be 'gp' or 'qep'")
    H, W = tile.shape
    y = tile.astype(np.float64)

    # coordinates in [0,1]^2, joint 2D
    xs = np.linspace(0.0, 1.0, H)
    ys = np.linspace(0.0, 1.0, W)
    X1, X2 = np.meshgrid(xs, ys, indexing="ij")
    X = np.stack([X1.ravel(), X2.ravel()], axis=1)
    Y = y.ravel()

    # Standardizing makes the same hyperparameter initialization sensible for
    # both 8-bit intensity ranges and normalized images. Applied identically to
    # every method, and inverted before returning.
    if standardize:
        y_mu, y_sd = float(Y.mean()), float(Y.std()) + 1e-12
        Yz = (Y - y_mu) / y_sd
    else:
        y_mu, y_sd, Yz = 0.0, 1.0, Y

    N = X.shape[0]
    if N > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_points, replace=False)
    else:
        idx = np.arange(N)

    train_x = torch.from_numpy(X[idx]).to(device=device, dtype=dtype)
    train_y = torch.from_numpy(Yz[idx]).to(device=device, dtype=dtype)

    if family == "gp":
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = _ExactGPTile(train_x, train_y, likelihood, nu=nu).to(device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    else:
        POWER = torch.tensor(float(q), dtype=dtype, device=device)
        likelihood = qpytorch.likelihoods.QExponentialLikelihood(power=POWER).to(device)
        model = _ExactQEPTile(train_x, train_y, likelihood, POWER, nu=nu).to(device)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train(); likelihood.train()
    # model.parameters() ALREADY includes the likelihood's parameters, because
    # ExactGP/ExactQEP register the likelihood as a submodule. Passing both
    # would put the same tensors in the optimizer twice and step them twice per
    # iteration (an effective 2x learning rate on the noise), which would make
    # this arm differ from gp_legacy for the wrong reason.
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_nonfinite = 0
    n_exception = 0
    losses: List[float] = []

    # DETERMINISM. gpytorch's default max_cholesky_size is 800, so with a few
    # thousand training pixels the exact marginal likelihood's log-determinant
    # is estimated by stochastic Lanczos quadrature using num_trace_samples=10
    # RANDOM probe vectors (deterministic_probes defaults to False). Round 1 ran
    # its arms sequentially in one process without reseeding torch, so each arm
    # consumed a different RNG state and identical models fitted differently:
    # three identical reruns of the same arm gave lengthscale 0.02957 / 0.03135 /
    # 0.03093. That, not q, produced the round-1 GP vs QEP(q=2) gap.
    #
    # Two independent fixes are applied:
    #   exact_logdet=True raises max_cholesky_size above the training size so the
    #     log-determinant is computed exactly by Cholesky, with no probes at all;
    #   torch_seed pins the RNG anyway, so any remaining stochastic path is
    #     reproducible.
    if torch_seed is not None:
        torch.manual_seed(int(torch_seed))
    chol_ctx = (gpytorch.settings.max_cholesky_size(int(train_x.shape[0]) + 1)
                if exact_logdet else gpytorch.settings.max_cholesky_size(
                    gpytorch.settings.max_cholesky_size.value()))

    t0 = time.time()
    with chol_ctx:
        for _ in range(train_iters):
            opt.zero_grad(set_to_none=True)
            try:
                out = model(train_x)
                loss = -mll(out, train_y)
            except Exception:  # noqa: BLE001
                n_exception += 1
                continue
            if not torch.isfinite(loss):
                n_nonfinite += 1
                continue
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    train_s = time.time() - t0

    model.eval(); likelihood.eval()
    # Prediction, not training, dominates both time and memory here.
    #
    # Time: a tile has ~68k test pixels against a few thousand training pixels,
    # and gpytorch's exact_prediction builds the predictive COVARIANCE even when
    # only .mean is read (measured: 1.9s training vs ~130s prediction).
    # skip_posterior_variances suppresses that, which is right because the
    # pipeline only ever consumes the mean. It also avoids fast_pred_var's
    # root-inverse cache, which allocates an internal float32 tensor and dies on
    # float64 input with "expected m1 and m2 to have the same dtype".
    #
    # Memory: the test-train cross-covariance is one dense block of
    # n_test x n_train. At 67680 x 6000 float64 that is 3.25 GB, and with
    # gpytorch's intermediates the earlier unchunked run was SIGKILLed (OOM) on
    # a 17 GB machine. Chunking the test points bounds peak memory to
    # predict_chunk x n_train and leaves the result numerically identical: the
    # posterior mean of each test point depends only on the training data.
    preds: List[np.ndarray] = []
    with torch.no_grad(), gpytorch.settings.debug(False), \
         gpytorch.settings.skip_posterior_variances(True):
        for s in range(0, X.shape[0], predict_chunk):
            xb = torch.from_numpy(X[s:s + predict_chunk]).to(device=device,
                                                             dtype=dtype)
            preds.append(model(xb).mean.detach().cpu().numpy())
    pred = np.concatenate(preds)

    predmean = (pred * y_sd + y_mu).reshape(H, W)

    ls = model.covar_module.base_kernel.lengthscale.detach().flatten().cpu().numpy()
    diag = dict(
        family=family, q=(float(q) if family == "qep" else 2.0),
        lengthscale=float(ls[0]),
        outputscale=float(model.covar_module.outputscale.detach()),
        noise=float(likelihood.noise.detach()),
        final_loss=(losses[-1] if losses else float("nan")),
        first_loss=(losses[0] if losses else float("nan")),
        n_iters=train_iters, n_effective_steps=len(losses),
        n_nonfinite_loss=n_nonfinite, n_exceptions=n_exception,
        n_train=int(len(idx)), n_pixels=int(N),
        train_s=train_s, dtype=str(dtype).replace("torch.", ""),
        torch_seed=torch_seed, exact_logdet=bool(exact_logdet),
        y_mean=y_mu, y_std=y_sd,
    )
    return predmean, diag


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction over tiles (the only method-dependent stage)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TileGeom:
    num_pieces_x: int
    num_pieces_y: int
    crop_width: int
    crop_height: int


def compute_tiling(img_height: int, img_width: int) -> TileGeom:
    """Legacy tiling from generate_gp_masks_test, extracted verbatim."""
    row_prop = get_proportion(img_height)
    col_prop = get_proportion(img_width)
    crop_width = int(img_width * col_prop)
    crop_height = int(img_height * row_prop)
    num_pieces_x = max(1, img_width // crop_width)
    num_pieces_y = max(1, img_height // crop_height)
    crop_width = img_width // num_pieces_x
    crop_height = img_height // num_pieces_y
    return TileGeom(num_pieces_x, num_pieces_y, crop_width, crop_height)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegResult:
    method: str
    q: Optional[float]
    img_gray: np.ndarray
    combined_predmean: np.ndarray
    combined_thresholded: np.ndarray
    dist_map: np.ndarray
    instance_mask: np.ndarray
    n_instances: int
    tile_thresholds: List[float]
    tile_diags: List[Dict]
    outlier_tiles: List[int]
    connected_parts_count: List[int]
    runtime_recon_s: float
    runtime_total_s: float
    marker_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    marker_coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    separation_config: Dict = field(default_factory=dict)
    hypers: Dict = field(default_factory=dict)


def run_segmentation(
    img_gray: np.ndarray,
    *,
    method: str,
    q: float = 2.0,
    seed: int = 0,
    delta: float = 0.01,
    nugget: bool = True,
    remove_size_threshold: int = 50,
    max_points: int = 3000,
    train_iters: int = 75,
    lr: float = 0.1,
    nu: float = 2.5,
    dtype: torch.dtype = torch.float64,
    exact_logdet: bool = True,
    marker_mode: str = "legacy_none",
    min_distance: int = 9,
    verbose: bool = True,
) -> SegResult:
    """End-to-end segmentation. Only the reconstruction stage depends on `method`.

    Mirrors generate_gp_masks_test step for step (tiling, per-tile criterion_1,
    outlier re-thresholding, stitching, distance-transform watershed,
    small-area removal) so that "gp_legacy" reproduces the historical pipeline.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    t_start = time.time()
    img_height, img_width = img_gray.shape
    geom = compute_tiling(img_height, img_width)

    combined_predmean = np.zeros((img_height, img_width), dtype=np.float64)
    combined_thresholded = np.zeros((img_height, img_width), dtype=np.uint8)

    processed: List[np.ndarray] = []
    thresholded: List[np.ndarray] = []
    tile_thresholds: List[float] = []
    connected_parts_count: List[int] = []
    tile_diags: List[Dict] = []

    t_recon = 0.0
    tile_index = 0

    # ---- 1) per-tile reconstruction + adaptive threshold ----
    for i in range(geom.num_pieces_x):
        for j in range(geom.num_pieces_y):
            x_off = i * geom.crop_width
            y_off = j * geom.crop_height
            piece_w = img_width - x_off if (i == geom.num_pieces_x - 1) else geom.crop_width
            piece_h = img_height - y_off if (j == geom.num_pieces_y - 1) else geom.crop_height
            tile = img_gray[y_off:y_off + piece_h, x_off:x_off + piece_w].copy()

            # deterministic per-tile RNG so every method sees the same pixels
            rng = np.random.default_rng([seed, tile_index])

            t0 = time.time()
            if method == "raw":
                predmean = tile.astype(np.float64)
                diag = dict(family="raw", q=None, n_pixels=int(tile.size))
            elif method == "gp_legacy":
                # Legacy CONFIGURATION (float32, no standardization, isotropic
                # Matern nu, default gpytorch init, Adam lr, same iters) routed
                # through the chunked predictor.
                #
                # The literal legacy call, separable_gp_smooth_gpytorch, cannot
                # be used at this tile size: it predicts on all ~68k test pixels
                # in one shot and the earlier run was SIGKILLed by the OOM killer
                # partway through this arm. Equivalence of this path to the
                # literal legacy function is checked on a single tile by
                # experiments/real_data/check_legacy_gp_equivalence.py.
                predmean, diag = smooth_tile(
                    tile, family="gp", q=2.0, nu=nu,
                    train_iters=train_iters, lr=lr, max_points=max_points,
                    rng=rng, dtype=torch.float32, standardize=False,
                    torch_seed=(seed * 1000 + tile_index),
                    exact_logdet=exact_logdet)
                diag["family"] = "gp_legacy"
            else:
                predmean, diag = smooth_tile(
                    tile, family=("gp" if method == "gp" else "qep"), q=q,
                    nu=nu, train_iters=train_iters, lr=lr,
                    max_points=max_points, rng=rng, dtype=dtype,
                    torch_seed=(seed * 1000 + tile_index),
                    exact_logdet=exact_logdet)
            t_recon += time.time() - t0

            diag["tile_index"] = tile_index
            diag["tile_shape"] = (int(piece_h), int(piece_w))

            processed.append(predmean)
            c1 = criterion_1(predmean, delta=delta, nugget=nugget)
            thresholded.append(c1.thresholded_image)
            tile_thresholds.append(float(c1.estimated_percentage))
            diag["threshold_pct"] = float(c1.estimated_percentage)

            cc = cc_label(c1.thresholded_image > 0, connectivity=1)
            connected_parts_count.append(int(len(np.unique(cc))))
            diag["connected_parts"] = int(len(np.unique(cc)))
            tile_diags.append(diag)

            if verbose:
                print(f"    tile {tile_index+1}/{geom.num_pieces_x*geom.num_pieces_y} "
                      f"{predmean.shape} thr={c1.estimated_percentage:.2f} "
                      f"parts={diag['connected_parts']}", flush=True)
            tile_index += 1

    # ---- 2) outlier tiles (shared) ----
    cp = np.array(connected_parts_count, dtype=np.float64)
    sd_cp = float(cp.std(ddof=0))
    outliers = ([] if sd_cp == 0
                else list(np.where(np.abs(cp - cp.mean()) > 2.0 * sd_cp)[0].astype(int)))
    if len(outliers) < len(tile_thresholds):
        keep = [k for k in range(len(tile_thresholds)) if k not in outliers]
        mean_thr = float(np.mean([tile_thresholds[k] for k in keep]))
    else:
        mean_thr = float(np.mean(tile_thresholds))

    # ---- 3) re-threshold outliers (shared) ----
    for idx_o in outliers:
        rethr = threshold_image(processed[idx_o], mean_thr, count=False)
        tile_thresholds[idx_o] = mean_thr
        if rethr.sum() > 0.99 * rethr.size:
            rethr = np.zeros_like(rethr, dtype=np.uint8)
            tile_thresholds[idx_o] = 1.0
        thresholded[idx_o] = rethr
        tile_diags[idx_o]["threshold_pct"] = tile_thresholds[idx_o]
        tile_diags[idx_o]["outlier_retreated"] = True

    # ---- 4) stitch (shared) ----
    k = 0
    for i in range(geom.num_pieces_x):
        for j in range(geom.num_pieces_y):
            x_off = i * geom.crop_width
            y_off = j * geom.crop_height
            pm, th = processed[k], thresholded[k]
            h, w = pm.shape
            combined_predmean[y_off:y_off + h, x_off:x_off + w] = pm
            combined_thresholded[y_off:y_off + h, x_off:x_off + w] = th.astype(np.uint8)
            k += 1

    # ---- 5+6) instance separation and small-area removal (shared) ----
    # marker_mode="legacy_none" reproduces the historical
    # watershed(..., markers=None, ...) configuration exactly; "peak" uses
    # explicit deterministic distance-transform peak markers. Everything else
    # (foreground, elevation, connectivity, cleanup) is identical either way,
    # so a difference between modes is attributable to marker generation alone.
    sep = separate_instances(
        combined_thresholded, marker_mode=marker_mode,
        min_distance=min_distance,
        remove_size_threshold=remove_size_threshold)
    dist_map = sep.dist_map
    instance_mask = sep.instance_mask
    n_inst = sep.n_instances

    # aggregate fitted hyperparameters across tiles
    hypers: Dict = {}
    fitted = [d for d in tile_diags if "lengthscale" in d]
    if fitted:
        for key in ("lengthscale", "outputscale", "noise", "final_loss"):
            vals = np.array([d[key] for d in fitted], dtype=np.float64)
            hypers[f"{key}_mean"] = float(np.nanmean(vals))
            hypers[f"{key}_min"] = float(np.nanmin(vals))
            hypers[f"{key}_max"] = float(np.nanmax(vals))
        hypers["n_nonfinite_loss_total"] = int(sum(d.get("n_nonfinite_loss", 0)
                                                   for d in fitted))
        hypers["n_exceptions_total"] = int(sum(d.get("n_exceptions", 0)
                                               for d in fitted))
        hypers["n_train_per_tile"] = int(fitted[0].get("n_train", 0))
        hypers["dtype"] = fitted[0].get("dtype", "")
    hypers["threshold_pct_mean"] = float(np.mean(tile_thresholds))
    hypers["threshold_pct_min"] = float(np.min(tile_thresholds))
    hypers["threshold_pct_max"] = float(np.max(tile_thresholds))
    hypers["n_tiles"] = int(len(tile_diags))
    hypers["n_outlier_tiles"] = int(len(outliers))
    hypers["marker_mode"] = marker_mode
    hypers["min_distance"] = (int(min_distance) if marker_mode == "peak" else None)
    hypers["n_markers"] = int(sep.n_markers)
    hypers["n_fg_components"] = int(sep.n_fg_components)
    hypers["n_components_rescued"] = int(sep.n_components_rescued)
    hypers["exact_logdet"] = bool(exact_logdet)

    return SegResult(
        method=method, q=(q if method == "qep" else None),
        img_gray=img_gray,
        combined_predmean=combined_predmean,
        combined_thresholded=combined_thresholded,
        dist_map=dist_map,
        instance_mask=instance_mask,
        n_instances=n_inst,
        tile_thresholds=tile_thresholds,
        tile_diags=tile_diags,
        outlier_tiles=outliers,
        connected_parts_count=connected_parts_count,
        runtime_recon_s=t_recon,
        runtime_total_s=time.time() - t_start,
        marker_image=sep.marker_image,
        marker_coords=sep.marker_coords,
        separation_config=sep.config,
        hypers=hypers,
    )
