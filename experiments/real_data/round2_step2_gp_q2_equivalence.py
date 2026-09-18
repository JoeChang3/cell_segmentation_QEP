"""
Step 2A: why do the GP and QEP q=2 arms differ, when q=2 is the Gaussian case?

Round 1 reported nearly identical FITTED hyperparameters for the two arms
(nuclei: lengthscale 0.031983 vs 0.031996, noise 0.028230 vs 0.028165) yet
different reconstructions, different binary masks and an AP@0.5 gap of 0.0060.
That gap was previously described as a "pipeline noise floor". That label was
not earned: no repeatability study had been run, and similar RMSE-to-raw values
do not show that two reconstructions resemble EACH OTHER.

This script does the work properly, in three parts.

PART 1  environment and implementation record
        executables, versions, import paths, classes, dtype, device.

PART 2  FIXED-PARAMETER equivalence
        Identical training pixels, coordinates, preprocessing, mean, kernel and
        noise VALUES are forced into both a gpytorch ExactGP and a qpytorch
        ExactQEP(power=2), and the predictions are compared directly.
        If these differ beyond floating-point tolerance, the two arms are not
        the same estimator and no q-specific claim can be made.

PART 3  INDEPENDENT-TRAINING divergence
        Both arms are trained from their own library defaults exactly as
        round 1 did, on one representative development tile, logging the
        parameter vector at every iteration to locate where and why the
        trajectories separate. Separately optimized models are NOT required to
        be bitwise identical; the point is to explain the mechanism.

Tolerance: float64 Cholesky/CG solves on ~3000 training points accumulate
roughly O(n * eps) relative error; with eps=2.2e-16 that is ~1e-12 relative.
A max absolute prediction difference below 1e-8 in standardized units is
treated as numerically identical, and anything above 1e-4 as a genuine
implementation difference.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float64)

import gpytorch
import qpytorch

from py_core.segmentation_eval import load_gray_image
from py_core.segmentation_pipeline import (
    _ExactGPTile,
    _ExactQEPTile,
    compute_tiling,
)

TOL_IDENTICAL = 1e-8
TOL_IMPLEMENTATION = 1e-4


def env_record() -> Dict:
    return dict(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch.__version__, torch_path=os.path.dirname(torch.__file__),
        gpytorch=gpytorch.__version__,
        gpytorch_path=os.path.dirname(gpytorch.__file__),
        qpytorch=qpytorch.__version__,
        qpytorch_path=os.path.dirname(qpytorch.__file__),
        numpy=np.__version__,
        default_dtype=str(torch.get_default_dtype()),
        device="cpu",
        gp_model_class=f"{_ExactGPTile.__module__}.{_ExactGPTile.__name__}",
        qep_model_class=f"{_ExactQEPTile.__module__}.{_ExactQEPTile.__name__}",
        gp_likelihood="gpytorch.likelihoods.GaussianLikelihood",
        qep_likelihood="qpytorch.likelihoods.QExponentialLikelihood",
        gp_mll="gpytorch.mlls.ExactMarginalLogLikelihood",
        qep_mll="qpytorch.mlls.ExactMarginalLogLikelihood",
    )


def build(kind: str, train_x, train_y, nu=2.5):
    if kind == "gp":
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = _ExactGPTile(train_x, train_y, lik, nu=nu)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    else:
        P = torch.tensor(2.0)
        lik = qpytorch.likelihoods.QExponentialLikelihood(power=P)
        model = _ExactQEPTile(train_x, train_y, lik, P, nu=nu)
        mll = qpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    return model, lik, mll


def raw_params(model, lik) -> Dict[str, float]:
    return {
        "raw_lengthscale": float(model.covar_module.base_kernel.raw_lengthscale.detach().flatten()[0]),
        "lengthscale": float(model.covar_module.base_kernel.lengthscale.detach().flatten()[0]),
        "raw_outputscale": float(model.covar_module.raw_outputscale.detach()),
        "outputscale": float(model.covar_module.outputscale.detach()),
        "raw_noise": float(lik.raw_noise.detach().flatten()[0]),
        "noise": float(lik.noise.detach().flatten()[0]),
        "mean_constant": float(model.mean_module.constant.detach().flatten()[0]),
    }


def set_params(model, lik, ls: float, os_: float, noise: float, mean: float) -> None:
    model.covar_module.base_kernel.lengthscale = torch.tensor([[ls]])
    model.covar_module.outputscale = torch.tensor(os_)
    lik.noise = torch.tensor(noise)
    with torch.no_grad():
        model.mean_module.constant.fill_(mean)


def predict(model, lik, test_x, chunk=8192) -> np.ndarray:
    model.eval(); lik.eval()
    out = []
    with torch.no_grad(), gpytorch.settings.debug(False), \
         gpytorch.settings.skip_posterior_variances(True):
        for s in range(0, test_x.shape[0], chunk):
            out.append(model(test_x[s:s + chunk]).mean.detach().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=75)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    env = env_record()
    print("=" * 108)
    print("PART 1  ENVIRONMENT AND IMPLEMENTATION")
    print("=" * 108)
    for k, v in env.items():
        print(f"  {k:<22} {v}")

    # one representative development tile, exactly as round 1 cut it
    img = load_gray_image(os.path.join(
        _ROOT, "data", "nuclear_test_images", "nuclei_figure_1",
        "original_fig.png"))
    geom = compute_tiling(*img.shape)
    tile = img[0:geom.crop_height, 0:geom.crop_width].copy()
    H, W = tile.shape
    print(f"\n  representative tile: nuclei_figure_1 [0:{H}, 0:{W}]  "
          f"range [{tile.min():.0f}, {tile.max():.0f}]")

    # ---- identical preprocessing for both arms (round-1 controlled arms) ----
    xs = np.linspace(0.0, 1.0, H); ys = np.linspace(0.0, 1.0, W)
    X1, X2 = np.meshgrid(xs, ys, indexing="ij")
    X = np.stack([X1.ravel(), X2.ravel()], axis=1)
    Y = tile.astype(np.float64).ravel()
    y_mu, y_sd = float(Y.mean()), float(Y.std()) + 1e-12
    Yz = (Y - y_mu) / y_sd
    rng = np.random.default_rng([0, 0])          # round-1 rule: seed [seed, tile_idx]
    idx = rng.choice(X.shape[0], size=args.max_points, replace=False)
    train_x = torch.from_numpy(X[idx]); train_y = torch.from_numpy(Yz[idx])
    test_x = torch.from_numpy(X)

    rows: List[Dict] = []

    # ═════ PART 2: fixed-parameter equivalence ═════
    print("\n" + "=" * 108)
    print("PART 2  FIXED-PARAMETER EQUIVALENCE (identical data AND identical "
          "parameter values)")
    print("=" * 108)
    print(f"  identical training pixels: {len(idx)} indices from seed [0,0]; "
          f"standardized with mu={y_mu:.4f} sd={y_sd:.4f}")
    # round-1 fitted values for the nuclei gp arm, used as a realistic setting
    for ls, os_, nz, mean in [(0.031983, 1.727742, 0.028230, 0.0),
                              (0.10, 1.0, 0.01, 0.0),
                              (0.05, 2.0, 0.10, 0.25)]:
        g, gl, _ = build("gp", train_x, train_y)
        q, ql, _ = build("qep", train_x, train_y)
        set_params(g, gl, ls, os_, nz, mean)
        set_params(q, ql, ls, os_, nz, mean)
        pg, pq = predict(g, gl, test_x), predict(q, ql, test_x)
        dmax = float(np.abs(pg - pq).max())
        drms = float(np.sqrt(np.mean((pg - pq) ** 2)))
        verdict = ("numerically identical" if dmax < TOL_IDENTICAL else
                   "IMPLEMENTATION DIFFERENCE" if dmax > TOL_IMPLEMENTATION else
                   "borderline")
        print(f"\n  ls={ls:<9.6f} os={os_:<8.4f} noise={nz:<8.5f} mean={mean}")
        print(f"    max|GP - QEP(q=2)| = {dmax:.3e}   pairwise RMSE = {drms:.3e}"
              f"   -> {verdict}")
        rows.append(dict(check="fixed_parameter", lengthscale=ls, outputscale=os_,
                         noise=nz, mean_constant=mean, max_abs_diff=dmax,
                         pairwise_rmse=drms, tol_identical=TOL_IDENTICAL,
                         verdict=verdict))

    # ═════ PART 3: independent training ═════
    print("\n" + "=" * 108)
    print("PART 3  INDEPENDENT TRAINING FROM LIBRARY DEFAULTS (round-1 protocol)")
    print("=" * 108)
    hist: Dict[str, List[Dict]] = {}
    finals: Dict[str, Dict] = {}
    for kind in ("gp", "qep"):
        torch.manual_seed(0)
        model, lik, mll = build(kind, train_x, train_y)
        init = raw_params(model, lik)
        print(f"\n  {kind.upper()} initial parameters (library defaults, nothing set "
              f"by round 1):")
        for k in ("lengthscale", "outputscale", "noise", "mean_constant"):
            print(f"    {k:<16} {init[k]:.10f}")
        h = [dict(iter=0, loss=float("nan"), **init)]
        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        n_groups = sum(len(g["params"]) for g in opt.param_groups)
        print(f"    optimizer tensors in param_groups: {n_groups}")
        for i in range(args.iters):
            opt.zero_grad(set_to_none=True)
            loss = -mll(model(train_x), train_y)
            loss.backward(); opt.step()
            h.append(dict(iter=i + 1, loss=float(loss.detach()),
                          **raw_params(model, lik)))
        hist[kind] = h
        finals[kind] = raw_params(model, lik)
        finals[kind]["final_loss"] = h[-1]["loss"]
        finals[kind]["n_optimizer_tensors"] = n_groups
        print(f"    final: ls={finals[kind]['lengthscale']:.6f} "
              f"os={finals[kind]['outputscale']:.6f} "
              f"noise={finals[kind]['noise']:.6f} "
              f"loss={finals[kind]['final_loss']:.6f}")

    hdf = pd.DataFrame([dict(arm=k, **r) for k, v in hist.items() for r in v])
    hdf.to_csv(os.path.join(args.out, "gp_q2_training_traces.csv"), index=False)

    # where do the trajectories first separate?
    print("\n  first iteration at which the two trajectories differ, per parameter"
          " (tolerance 1e-12):")
    gh = pd.DataFrame(hist["gp"]); qh = pd.DataFrame(hist["qep"])
    n = min(len(gh), len(qh))
    for col in ("lengthscale", "outputscale", "noise", "mean_constant", "loss"):
        a = gh[col].to_numpy()[:n]; b = qh[col].to_numpy()[:n]
        diff = np.abs(a - b)
        ok = np.isfinite(diff)
        first = next((int(gh["iter"].to_numpy()[i]) for i in range(n)
                      if ok[i] and diff[i] > 1e-12), None)
        print(f"    {col:<16} first differing iter = "
              f"{'never' if first is None else first}"
              f"   final |diff| = {diff[ok][-1]:.3e}")
        rows.append(dict(check="trajectory_divergence", parameter=col,
                         first_differing_iter=(-1 if first is None else first),
                         final_abs_diff=float(diff[ok][-1])))

    # resulting predictions after independent training
    preds = {}
    for kind in ("gp", "qep"):
        torch.manual_seed(0)
        model, lik, mll = build(kind, train_x, train_y)
        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.iters):
            opt.zero_grad(set_to_none=True)
            loss = -mll(model(train_x), train_y)
            loss.backward(); opt.step()
        preds[kind] = predict(model, lik, test_x) * y_sd + y_mu
    dmax = float(np.abs(preds["gp"] - preds["qep"]).max())
    drms = float(np.sqrt(np.mean((preds["gp"] - preds["qep"]) ** 2)))
    print(f"\n  after independent training, on the ORIGINAL intensity scale:")
    print(f"    max|GP - QEP(q=2)| = {dmax:.4f}   pairwise RMSE = {drms:.4f}"
          f"   (tile range {tile.min():.0f}-{tile.max():.0f})")
    rows.append(dict(check="independent_training", max_abs_diff=dmax,
                     pairwise_rmse=drms,
                     gp_final_lengthscale=finals["gp"]["lengthscale"],
                     qep_final_lengthscale=finals["qep"]["lengthscale"],
                     gp_final_noise=finals["gp"]["noise"],
                     qep_final_noise=finals["qep"]["noise"],
                     gp_final_loss=finals["gp"]["final_loss"],
                     qep_final_loss=finals["qep"]["final_loss"]))

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out, "gp_q2_equivalence_checks.csv"), index=False)
    with open(os.path.join(args.out, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    print("\n" + "=" * 108)
    print(f"Wrote gp_q2_equivalence_checks.csv, gp_q2_training_traces.csv, "
          f"environment.json into {args.out}")
    print("=" * 108)


if __name__ == "__main__":
    main()
