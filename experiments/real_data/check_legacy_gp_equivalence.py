"""
Does the round-1 "gp_legacy" arm reproduce the literal legacy GP smoother?

The legacy smoother `separable_gp_smooth_gpytorch`
(py_core/Modified_Functions_RGasp.py:208) predicts on every pixel of a tile in
one shot. At the real nuclei tile size (282x240 = 67,680 test pixels against
6,000 training pixels) that is a 3.25 GB dense cross-covariance in float64, and
the first round-1 run was SIGKILLed by the OOM killer partway through that arm.

The round-1 pipeline therefore runs the legacy CONFIGURATION through
`smooth_tile(family="gp", standardize=False, dtype=float32)`, which chunks the
test points. Chunking cannot change the answer -- each test point's posterior
mean depends only on the training data -- but the two code paths differ in two
other ways worth checking rather than asserting:

  * the legacy path wraps prediction in `fast_pred_var()`, which affects the
    predictive VARIANCE only, not the mean;
  * the legacy path draws its training subsample from the global numpy RNG,
    while the new path uses an explicit Generator.

This script pins the subsample so both paths see identical training pixels, then
compares the returned posterior means on one tile.

Run:
    python experiments/real_data/check_legacy_gp_equivalence.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch.set_default_dtype(torch.float32)

from py_core.Modified_Functions_RGasp import separable_gp_smooth_gpytorch
from py_core.segmentation_eval import load_gray_image
from py_core.segmentation_pipeline import smooth_tile

# A tile small enough that the unchunked legacy call fits in memory, so the
# comparison is actually runnable.
TILE_H, TILE_W = 140, 120
MAX_POINTS = 2000
ITERS = 75
LR = 0.1
NU = 2.5
SEED = 0


def main() -> None:
    img = load_gray_image(os.path.join(
        _ROOT, "data", "nuclear_test_images", "nuclei_figure_1",
        "original_fig.png"))
    tile = img[:TILE_H, :TILE_W].copy()
    n = tile.size
    print("=" * 84)
    print("Legacy GP smoother vs round-1 gp_legacy arm (same configuration)")
    print("=" * 84)
    print(f"tile {tile.shape} = {n} px, max_points={MAX_POINTS}, "
          f"iters={ITERS}, lr={LR}, Matern nu={NU}\n")

    # Same training subsample for both paths.
    idx = np.random.default_rng(SEED).choice(n, size=MAX_POINTS, replace=False)

    class _FixedRng:
        """Hands back the pinned subsample so both paths train on the same px."""
        def choice(self, _n, size=None, replace=True):  # noqa: D102
            return idx

    torch.manual_seed(SEED)
    t0 = time.time()
    legacy_shim = np.random.choice

    def _fixed_choice(a, size=None, replace=True, p=None):
        return idx

    np.random.choice = _fixed_choice          # pin the legacy subsample
    try:
        legacy = separable_gp_smooth_gpytorch(
            tile, kernel_type="matern", nu=NU, train_iters=ITERS, lr=LR,
            max_points=MAX_POINTS)
    finally:
        np.random.choice = legacy_shim
    t_legacy = time.time() - t0

    torch.manual_seed(SEED)
    t0 = time.time()
    new, diag = smooth_tile(
        tile, family="gp", q=2.0, nu=NU, train_iters=ITERS, lr=LR,
        max_points=MAX_POINTS, rng=_FixedRng(), dtype=torch.float32,
        standardize=False)
    t_new = time.time() - t0

    diff = np.abs(legacy - new)
    rel = diff.max() / max(np.ptp(legacy), 1e-12)
    print(f"legacy  : {t_legacy:6.1f}s  range [{legacy.min():.2f}, {legacy.max():.2f}]")
    print(f"round-1 : {t_new:6.1f}s  range [{new.min():.2f}, {new.max():.2f}]  "
          f"(ls={diag['lengthscale']:.4f}, noise={diag['noise']:.5f})")
    print(f"\nmax |difference|      = {diff.max():.6g}")
    print(f"as fraction of range  = {rel:.3%}")
    print(f"RMSE between the two  = {np.sqrt((diff**2).mean()):.6g}")
    print(f"correlation           = {np.corrcoef(legacy.ravel(), new.ravel())[0,1]:.6f}")

    ok = rel < 0.02
    print(f"\nEquivalent to within 2% of intensity range: {'YES' if ok else 'NO'}")
    print("Any residual gap comes from optimizer nondeterminism in float32, not\n"
          "from chunking: chunking splits the test points only, and each test\n"
          "point's posterior mean depends solely on the training data.")
    print("=" * 84)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
