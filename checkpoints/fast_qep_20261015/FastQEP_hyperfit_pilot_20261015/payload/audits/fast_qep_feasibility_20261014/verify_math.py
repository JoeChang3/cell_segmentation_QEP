"""Deterministic algebra checks; no images, fitting study, or segmentation runs.

Run from any directory with Python, NumPy, and SciPy. Results are printed as JSON.
These fixtures check formulas, not performance or generalization.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy
from scipy.integrate import quad_vec
from scipy.linalg import eigh, solve
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]


def covariance_factor(n, q):
    return float(np.exp((2 / q) * np.log(2) + gammaln(n / 2 + 2 / q)
                        - gammaln(n / 2) - np.log(n)))


def log_qed(r, logdet, n, q):
    return (np.log(q / 2) - n / 2 * np.log(2 * np.pi) - logdet / 2
            + n * (q - 2) / 4 * np.log(r) - r ** (q / 2) / 2)


def load_original_core():
    path = ROOT / "py_core/dim_2_lattice.py"
    spec = importlib.util.spec_from_file_location("audited_lattice", path)
    lattice = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lattice)
    # Execute unmodified definitions only, avoiding unrelated Torch/R dependencies.
    source = ROOT / "py_core/paper_fast_gp.py"
    tree = ast.parse(source.read_text())
    names = {"SharedParams", "TileReconstruction", "_lattice_inputs", "_distance_matrices", "reconstruct_tile"}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    assert len(nodes) == len(names)
    namespace = dict(globals(), matern_5_2=lattice.matern_5_2, KERNEL_TYPE="matern")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return lattice, namespace


def posterior_mean_fixture(q, angular_points):
    """Two-dimensional polar quadrature at fixed, matched prior covariance.

    R**q ~ chi-square(2); x = R**q / 2 has Exp(1) density.
    """
    cov = np.array([0.4, 1.6])
    obs = np.array([1.2, -0.8])
    noise_var = 0.35
    angles = 2 * np.pi * np.arange(angular_points) / angular_points
    directions = np.array([np.cos(angles), np.sin(angles)])
    sqrt_scatter = np.sqrt(cov / covariance_factor(2, q))

    def integrand(x):
        latent = sqrt_scatter[:, None] * (2 * x) ** (1 / q) * directions
        residual = latent - obs[:, None]
        weights = np.exp(-x - np.sum(residual ** 2, axis=0) / (2 * noise_var))
        return np.r_[np.mean(weights), np.mean(latent * weights, axis=1)]

    integral, error = quad_vec(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-11)
    return integral[1:] / integral[0], float(error), float(integral[0])


def main():
    lattice, core = load_original_core()
    rng = np.random.default_rng(713)
    tile = rng.normal(size=(5, 4)) + np.arange(5)[:, None] / 5
    y = tile.reshape(-1, order="F")
    n = y.size
    ones = np.ones(n)
    d1 = np.abs(np.linspace(0, 1, 5)[:, None] - np.linspace(0, 1, 5))
    d2 = np.abs(np.linspace(0, 1, 4)[:, None] - np.linspace(0, 1, 4))
    profile_rows = []
    core_errors = []
    for b1, b2, nu in [(1.7, 3.2, 0.08), (4.1, 0.9, 0.3), (0.8, 2.4, 0.7)]:
        k1 = lattice.matern_5_2(d1, b1)
        k2 = lattice.matern_5_2(d2, b2)
        kernel = np.kron(k2, k1)
        obs_shape = kernel + nu * np.eye(n)
        inv_one = solve(obs_shape, ones)
        mean = float(inv_one @ y / (ones @ inv_one))
        residual = y - mean
        sse = float(residual @ solve(obs_shape, residual))
        logdet = float(np.linalg.slogdet(obs_shape)[1])
        gp_profile = -logdet / 2 - n / 2 * np.log(sse)
        original_obj = lattice.neg_log_lik_eigen_with_nugget(
            np.log([b1, b2, nu]), "matern", d1, d2, n, 1, [np.ones_like(tile)], tile)
        shared = core["SharedParams"](beta1=b1, beta2=b2, nugget=nu, n1=5, n2=4,
                                      source_tile=(1, 1), neg_log_lik=float(original_obj),
                                      optim_method="none_algebra_fixture", n_obj_evals=0, runtime_sec=0.0)
        reconstructed = core["reconstruct_tile"](tile, shared).pred_mean.reshape(-1, order="F")
        dense_prediction = mean + kernel @ solve(obs_shape, residual)
        core_errors.append({"objective_abs_error": abs(float(original_obj) + gp_profile),
                            "mean_max_abs_error": float(np.max(abs(reconstructed - dense_prediction)))})
        for q in [1.5, 2.0]:
            scale = sse / n ** (2 / q)
            actual = log_qed(sse / scale, n * np.log(scale) + logdet, n, q)
            constant = np.log(q / 2) - n / 2 * np.log(2 * np.pi) + n / 2 * np.log(n) - n / 2
            error = abs(actual - gp_profile - constant)
            numeric = minimize_scalar(
                lambda logscale: -log_qed(sse / np.exp(logscale), n * logscale + logdet, n, q),
                bounds=(np.log(scale) - 3, np.log(scale) + 3), method="bounded",
                options={"xatol": 1e-12})
            profile_rows.append({"beta1": b1, "beta2": b2, "nu": nu, "q": q,
                                 "profile_identity_abs_error": float(error),
                                 "scale_numeric_relative_error": float(abs(np.exp(numeric.x) / scale - 1)),
                                 "scalar_optimizer_success": bool(numeric.success)})
    assert max(v["objective_abs_error"] for v in core_errors) < 1e-10
    assert max(v["mean_max_abs_error"] for v in core_errors) < 1e-11
    assert max(v["profile_identity_abs_error"] for v in profile_rows) < 1e-10
    assert all(v["scalar_optimizer_success"] and v["scale_numeric_relative_error"] < 1e-6 for v in profile_rows)

    # Change of variables: the QED density times the Jacobian equals Gaussian density.
    q = 1.5
    covariance = np.diag([0.4, 1.6, 0.9])
    n3 = 3
    a = covariance_factor(n3, q)
    scatter = covariance / a
    root = np.linalg.cholesky(scatter)
    z = np.array([0.8, -0.7, 1.1])
    rho = float(np.linalg.norm(z))
    power = 2 / q - 1
    f = root @ z * rho ** power
    jac = root @ (rho ** power * np.eye(n3) + power * rho ** (power - 2) * np.outer(z, z))
    r = float(f @ solve(scatter, f))
    change_error = abs(log_qed(r, np.linalg.slogdet(scatter)[1], n3, q)
                       + np.linalg.slogdet(jac)[1]
                       - (-n3 / 2 * np.log(2 * np.pi) - z @ z / 2))
    assert change_error < 1e-11

    # Orthogonal diagonalization moves identity-observation likelihood to O(n) arithmetic.
    eigenvalues, eigenvectors = eigh(covariance + 0.1 * np.ones((n3, n3)))
    root_eigen = eigenvectors @ np.diag(np.sqrt(eigenvalues / a))
    obs = np.array([1.0, -0.3, 0.5])
    latent = root_eigen @ z * rho ** power
    spectral_latent = np.sqrt(eigenvalues / a) * z * rho ** power
    likelihood_error = abs(np.sum((obs - latent) ** 2)
                           - np.sum((eigenvectors.T @ obs - spectral_latent) ** 2))
    assert likelihood_error < 1e-11

    # MAP pathology is in latent f, not maximization over a scale with fixed residual.
    map_values = []
    for logepsilon in [-2.0, -5.0, -10.0, -20.0]:
        epsilon = np.exp(logepsilon)
        latent = np.array([epsilon, 0.0])
        rr = epsilon ** 2
        nlp = np.sum((np.array([1.2, -0.8]) - latent) ** 2) / (2 * 0.35)
        nlp += 0.5 * rr ** (q / 2) + 2 * (2 - q) / 4 * np.log(rr)
        map_values.append({"log_epsilon": logepsilon, "negative_log_posterior_up_to_constant": float(nlp)})
    assert all(b["negative_log_posterior_up_to_constant"] < a0["negative_log_posterior_up_to_constant"]
               for a0, b in zip(map_values, map_values[1:]))

    quadrature = {}
    means = {}
    for q in [1.5, 2.0]:
        coarse, err_coarse, norm_coarse = posterior_mean_fixture(q, 128)
        fine, err_fine, norm_fine = posterior_mean_fixture(q, 256)
        angular_error = float(np.max(abs(fine - coarse)))
        assert angular_error < 1e-8
        assert err_fine / norm_fine < 1e-8
        means[q] = fine
        quadrature[str(q)] = {"posterior_mean": fine.tolist(), "angular_refinement_max_abs_change": angular_error,
                              "radial_integral_error_estimate": err_fine,
                              "normalizing_integral": norm_fine}
    analytic_gp = np.array([0.4, 1.6]) / (np.array([0.4, 1.6]) + 0.35) * np.array([1.2, -0.8])
    gaussian_error = float(np.max(abs(means[2.0] - analytic_gp)))
    q_difference = float(np.max(abs(means[1.5] - means[2.0])))
    assert gaussian_error < 1e-8
    assert q_difference > 1e-5

    paths = [ROOT / "py_core/dim_2_lattice.py", ROOT / "py_core/paper_fast_gp.py", Path(__file__).resolve()]
    result = {
        "scope": "deterministic mathematical fixtures only; no microscopy images or model-training experiment",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "original_core_dense_checks": core_errors,
        "profile_identity_checks": profile_rows,
        "whitening_density_jacobian_abs_error": float(change_error),
        "spectral_likelihood_abs_error": float(likelihood_error),
        "latent_map_center_approach": map_values,
        "fixed_covariance_2d_quadrature": quadrature,
        "q2_mean_vs_analytic_max_abs_error": gaussian_error,
        "q15_vs_q2_mean_max_abs_difference": q_difference,
        "q1_dimension_moments": [
            {"n": d, "unscaled_coordinate_variance": d + 2,
             "dimension_scaled_coordinate_variance": 1 + 2 / d,
             "coordinate_kurtosis_even_after_covariance_matching": 3 * (d + 4) * (d + 6) / (d + 2) ** 2}
            for d in [1, 2, 20]],
        "all_assertions_passed": True,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
