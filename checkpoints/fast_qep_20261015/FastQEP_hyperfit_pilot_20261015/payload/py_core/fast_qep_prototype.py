"""Fixed-grid latent QEP + independent Gaussian noise prototype.

This is not the old ExactQEP prediction rule, a full spatial-process definition,
or a segmentation pipeline. Hyperparameters are supplied, never fitted here.
The prior covariance is matched exactly across q. No jitter or eigenvalue
truncation is applied silently. Requires only NumPy and SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaln

from py_core.dim_2_lattice import matern_5_2


def covariance_factor(n: int, q: float) -> float:
    if n < 1 or not np.isfinite(q) or not (1 <= q <= 2):
        raise ValueError("Prototype supports n >= 1 and 1 <= q <= 2 only.")
    if q == 2:
        return 1.0
    return float(np.exp(2 / q * np.log(2) + gammaln(n / 2 + 2 / q)
                        - gammaln(n / 2) - np.log(n)))


@dataclass
class GaussianReference:
    mean: float
    signal_variance: float
    noise_variance: float
    nugget_ratio: float
    residual_spectral: np.ndarray
    prediction: np.ndarray
    sse: float
    profile_nll_without_constant: float


@dataclass
class KroneckerGrid:
    """Column-major pixel convention matches the original R/Python core."""

    shape: tuple[int, int]
    beta_rows: float
    beta_cols: float
    row_vectors: np.ndarray
    col_vectors: np.ndarray
    eigenvalues: np.ndarray

    @classmethod
    def matern52(cls, shape: tuple[int, int], beta_rows: float, beta_cols: float):
        if len(shape) != 2 or any(int(d) != d or d < 2 for d in shape):
            raise ValueError("A rectangular grid with both dimensions >= 2 is required.")
        if not np.all(np.isfinite([beta_rows, beta_cols])) or min(beta_rows, beta_cols) <= 0:
            raise ValueError("Inverse ranges must be finite and positive.")
        nrows, ncols = map(int, shape)
        xr, xc = np.linspace(0, 1, nrows), np.linspace(0, 1, ncols)
        er, ur = eigh(matern_5_2(abs(xr[:, None] - xr), beta_rows))
        ec, uc = eigh(matern_5_2(abs(xc[:, None] - xc), beta_cols))
        if np.any(er <= 0) or np.any(ec <= 0):
            raise ValueError("Nonpositive kernel eigenvalue; stop rather than silently alter the model.")
        lam = np.outer(er, ec).reshape(-1, order="F")
        return cls((nrows, ncols), beta_rows, beta_cols, ur, uc, lam)

    @property
    def n(self):
        return self.shape[0] * self.shape[1]

    def forward(self, tile):
        tile = np.asarray(tile, dtype=np.float64)
        if tile.shape != self.shape or not np.all(np.isfinite(tile)):
            raise ValueError("Expected a finite tile on this fixed grid.")
        return (self.row_vectors.T @ tile @ self.col_vectors).reshape(-1, order="F")

    def inverse(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (self.n,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Expected one finite spectral vector.")
        return self.row_vectors @ coefficients.reshape(self.shape, order="F") @ self.col_vectors.T

    def solve(self, tile, nugget_ratio=0.0):
        if not np.isfinite(nugget_ratio) or nugget_ratio < 0:
            raise ValueError("Nugget ratio must be finite and nonnegative.")
        return self.inverse(self.forward(tile) / (self.eigenvalues + nugget_ratio))

    def logdet(self, nugget_ratio=0.0):
        if not np.isfinite(nugget_ratio) or nugget_ratio < 0:
            raise ValueError("Nugget ratio must be finite and nonnegative.")
        return float(np.log(self.eigenvalues + nugget_ratio).sum())

    def gaussian_reference(self, tile, nugget_ratio):
        if not np.isfinite(nugget_ratio) or nugget_ratio <= 0:
            raise ValueError("Independent noise requires a positive nugget ratio.")
        yy, xx = self.forward(tile), self.forward(np.ones(self.shape))
        denom = self.eigenvalues + nugget_ratio
        mean = float(np.sum(xx * yy / denom) / np.sum(xx * xx / denom))
        residual = yy - mean * xx
        sse = float(np.sum(residual ** 2 / denom))
        if not np.isfinite(sse) or sse <= 0:
            raise ValueError("A positive residual scale is required; constant-tile handling is not implemented.")
        signal_var = sse / self.n
        pred = mean + self.inverse(self.eigenvalues / denom * residual)
        return GaussianReference(
            mean, signal_var, signal_var * nugget_ratio, nugget_ratio, residual,
            pred, sse, 0.5 * self.logdet(nugget_ratio) + self.n / 2 * np.log(sse))

    def posterior(self, reference: GaussianReference, q: float):
        return SpectralPosterior(
            self.eigenvalues, reference.residual_spectral, reference.signal_variance,
            reference.noise_variance, q)


@dataclass
class SpectralPosterior:
    """Finite-dimensional posterior in white Gaussian coordinates.

    eigenvalues describe K; prior Cov(v)=signal_variance*K for every q.
    All operations support a leading batch/chain dimension. Batches are
    independent full latent vectors; never split one vector into radial chunks.
    """

    eigenvalues: np.ndarray
    observations: np.ndarray
    signal_variance: float
    noise_variance: float
    q: float

    def __post_init__(self):
        self.eigenvalues = np.array(self.eigenvalues, dtype=np.float64, copy=True)
        self.observations = np.array(self.observations, dtype=np.float64, copy=True)
        if (self.eigenvalues.ndim != 1 or self.eigenvalues.size < 1
                or self.observations.shape != self.eigenvalues.shape
                or not np.all(np.isfinite(self.observations))
                or not np.all(np.isfinite(self.eigenvalues)) or np.any(self.eigenvalues <= 0)):
            raise ValueError("Expected positive eigenvalues and a matching finite observation vector.")
        if (not np.all(np.isfinite([self.signal_variance, self.noise_variance]))
                or min(self.signal_variance, self.noise_variance) <= 0):
            raise ValueError("Signal and noise variances must be positive.")
        self.n = self.eigenvalues.size
        self.a = covariance_factor(self.n, self.q)
        self.power = 2 / self.q - 1
        self.gaussian_root = np.sqrt(self.signal_variance * self.eigenvalues)
        self.root_scatter = self.gaussian_root / np.sqrt(self.a)
        # Fixed Gaussian reference preconditioner, not adapted to q outcomes.
        self.mass_diag = 1 + self.gaussian_root ** 2 / self.noise_variance
        self.score_coefficient = self.gaussian_root / self.mass_diag

    def gaussian_mean(self):
        variance = self.signal_variance * self.eigenvalues
        return variance / (variance + self.noise_variance) * self.observations

    def gaussian_white_mean(self):
        return self.gaussian_root / self.noise_variance / self.mass_diag * self.observations

    def latent(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.shape[-1] != self.n:
            raise ValueError("The last axis must contain the entire latent vector.")
        if self.q == 2:
            return z * self.gaussian_root
        radius = np.linalg.norm(z, axis=-1, keepdims=True)
        return self.root_scatter * z * radius ** self.power

    def potential_gradient(self, z):
        """Return -log posterior (up to constant) and its exact gradient."""
        z = np.asarray(z, dtype=np.float64)
        latent = self.latent(z)
        residual = latent - self.observations
        value = 0.5 * (np.sum(residual ** 2, axis=-1) / self.noise_variance
                       + np.sum(z ** 2, axis=-1))
        g = residual / self.noise_variance
        if self.q == 2:
            return value, z + self.gaussian_root * g
        radius = np.linalg.norm(z, axis=-1, keepdims=True)
        unit = np.divide(z, radius, out=np.zeros_like(z), where=radius > 0)
        dg = self.root_scatter * g
        # Equivalent to rho^(power-2)*z*z.T but finite at z=0.
        gradient = z + radius ** self.power * (
            dg + self.power * unit * np.sum(unit * dg, axis=-1, keepdims=True))
        return value, gradient

    def log_likelihood(self, z):
        return -0.5 * np.sum((self.latent(z) - self.observations) ** 2, axis=-1) / self.noise_variance

    def score_corrected_latent(self, z):
        """Fixed, zero-expectation score control variate for the mean only.

        E[grad(-log posterior)]=0 by integration by parts. The coefficient is
        fixed from the q=2 reference before sampling. These corrected values
        are NOT posterior draws and must not be used for posterior variance.
        """
        _, grad = self.potential_gradient(z)
        return self.latent(z) - self.score_coefficient * grad


@dataclass(frozen=True)
class HMCConfig:
    chains: int = 4
    warmup: int = 5000
    draws: int = 20000
    min_steps: int = 8
    max_steps: int = 12
    initial_step_size: float = 0.25
    target_accept: float = 0.85
    max_step_size: float = 1.5
    base_seed: int = 20261014


def leapfrog(target, z, momentum, step_sizes, steps):
    """Symplectic integration; callers perform the Metropolis correction."""
    z, momentum = np.array(z, copy=True), np.array(momentum, copy=True)
    eps = np.asarray(step_sizes).reshape(-1, 1)
    value, grad = target.potential_gradient(z)
    momentum -= eps * grad / 2
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(steps):
            z += eps * momentum / target.mass_diag
            value, grad = target.potential_gradient(z)
            momentum -= eps * grad * (0.5 if k == steps - 1 else 1.0)
    return z, momentum, value


def sample_hmc(target: SpectralPosterior, config: HMCConfig, fixture_id: int,
               progress: Callable | None = None):
    """Four independent chains; warmup-only dual averaging and fixed mass.

    A common independently generated trajectory length is used across chains
    each iteration; chain momentum and Metropolis random variables are separate.
    The sample array is retained for audit-sized fixtures, not full images.
    """
    if (config.chains != 4 or not (1 <= config.warmup <= 5000)
            or not (4 <= config.draws <= 20000)
            or not (1 <= config.min_steps <= config.max_steps)
            or not (0 < config.target_accept < 1)
            or not (0 < config.initial_step_size <= config.max_step_size)):
        raise ValueError("Configuration exceeds or violates the bounded prototype protocol.")
    rngs = [np.random.default_rng(np.random.SeedSequence([config.base_seed, fixture_id, i]))
            for i in range(config.chains)]
    length_rng = np.random.default_rng(np.random.SeedSequence([config.base_seed, fixture_id, 999]))
    center = target.gaussian_white_mean()
    z = np.stack([center + 3 * rng.normal(size=target.n) / np.sqrt(target.mass_diag) for rng in rngs])
    initial_positions = z.copy()
    value, _ = target.potential_gradient(z)
    logeps = np.full(config.chains, np.log(config.initial_step_size))
    logbar = logeps.copy()
    mu = np.log(10 * config.initial_step_size)
    hbar = np.zeros(config.chains)
    draws = np.empty((config.chains, config.draws, target.n), dtype=np.float64)
    accept_count = np.zeros(config.chains, dtype=int)
    divergent_count = np.zeros(config.chains, dtype=int)
    max_abs_delta = np.zeros(config.chains)
    started = time.perf_counter()
    for iteration in range(config.warmup + config.draws):
        momenta = np.stack([rng.normal(size=target.n) for rng in rngs]) * np.sqrt(target.mass_diag)
        steps = int(length_rng.integers(config.min_steps, config.max_steps + 1))
        proposed, p_final, v_final = leapfrog(target, z, momenta, np.exp(logeps), steps)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = v_final + 0.5 * np.sum(p_final ** 2 / target.mass_diag, axis=1)
            delta -= value + 0.5 * np.sum(momenta ** 2 / target.mass_diag, axis=1)
        finite = np.isfinite(delta) & np.all(np.isfinite(proposed), axis=1)
        log_accept = np.where(finite, np.minimum(0, -delta), -np.inf)
        accepted = np.array([np.log(rng.uniform()) for rng in rngs]) < log_accept
        z = np.where(accepted[:, None], proposed, z)
        value = np.where(accepted, v_final, value)
        if iteration < config.warmup:
            t = iteration + 1
            eta = 1 / (t + 10)
            hbar = (1 - eta) * hbar + eta * (config.target_accept - np.exp(log_accept))
            logeps = np.clip(mu - np.sqrt(t) / 0.05 * hbar, np.log(1e-5), np.log(config.max_step_size))
            weight = t ** -0.75
            logbar = weight * logeps + (1 - weight) * logbar
            if t == config.warmup:
                logeps = logbar.copy()
        else:
            j = iteration - config.warmup
            draws[:, j] = z
            accept_count += accepted
            divergent_count += (~finite) | (np.abs(delta) > 100)
            max_abs_delta = np.maximum(max_abs_delta, np.where(finite, abs(delta), np.inf))
        if progress is not None and (iteration + 1) % 5000 == 0:
            progress(iteration + 1, config.warmup + config.draws)
    return draws, {
        "runtime_sec": time.perf_counter() - started,
        "initial_positions": initial_positions.tolist(),
        "step_sizes": np.exp(logeps).tolist(),
        "acceptance_rate": (accept_count / config.draws).tolist(),
        "divergences_abs_delta_H_gt_100": divergent_count.tolist(),
        "max_abs_delta_H": max_abs_delta.tolist(),
        "seed_components": [[config.base_seed, fixture_id, i] for i in range(config.chains)],
        "trajectory_length_seed": [config.base_seed, fixture_id, 999],
        "mass_is_fixed_gaussian_reference": True,
        "adaptation_after_warmup": False,
    }
