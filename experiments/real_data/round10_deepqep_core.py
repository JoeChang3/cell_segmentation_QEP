"""Round 10 core: Deep QEP model + paired q-comparison helpers.

REFERENCE IMPLEMENTATION
  https://github.com/lanzithinking/DeepQEP.git at commit
  aa843960a3e8692e06eb36522fdbb2043bf34f2e, cloned read-only to
  /Users/zchan/eclipse-workspace/DeepQEP  (a SIBLING, never vendored here).

WHICH CODE ACTUALLY RUNS HERE
  The reference repo ships a full vendored gpytorch fork
  (DeepQEP/gpytorch/, 6.6 MB) and its demos shadow the interpreter with
  `sys.path.insert(0, '../GPyTorch')`. We deliberately do NOT do that.
  Instead we use the INSTALLED, already-audited qpytorch 0.2, which is the
  successor package of that fork. Verified by diffing after normalising the
  package-root rename:
    models/deep_qeps/deep_qep.py      functionally identical; the installed
                                      version only adds
                                      `rsample(rescale=kwargs.pop('rescale', False))`
                                      whose default reproduces the fork exactly
    models/deep_qeps/__init__.py      byte-identical
    distributions/multivariate_qexponential.py   installed is a strict superset
                                      (adds `rescalor`, absent in the fork)
    likelihoods/qexponential_likelihood.py       cosmetic + an extra `reduction`
                                      kwarg
  So no fork is placed on sys.path and no package shadowing occurs.

WHERE q ENTERS (traced, not assumed)
  qpytorch/models/deep_qeps/deep_qep.py, DeepQEPLayer.__call__:
      inputs = QExponential(loc=inputs.mean,
                            scale=inputs.variance.sqrt(),
                            power=inputs.power).rsample(rescale=False)
  Between layers the latent is RESAMPLED from a univariate q-exponential whose
  rsample is  eps = |z|**(2/q - 1) * z  for q != 2, and  eps = z  for q == 2.
  So for q != 2 the latent handed to the next layer is a non-linear warp of the
  Gaussian latent, and the composed predictive MEAN becomes q-dependent. At
  q == 2 the branch is skipped, so the q=2 arm is the SAME architecture and the
  SAME code path -- this is the primary control, not a separate DeepGP codebase.
  Note also that this inter-layer resample is mean-field: it uses only
  `inputs.mean` and `inputs.variance` (the scale-matrix diagonal, per the
  Task-B audit), not the full latent covariance.

ARCHITECTURE  (matched to demo/demo_multi_Deep_QEP.py)
  2 stochastic layers: hidden (in -> H, linear mean) then last (H -> 1,
  constant mean); CholeskyVariationalDistribution carrying `power`;
  VariationalStrategy with learnable inducing locations;
  ScaleKernel(MaternKernel(nu=2.5, ard)); MultitaskQExponentialLikelihood;
  DeepApproximateMLL(VariationalELBO); Adam.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import Linear  # noqa: F401  (kept: reference demo imports it)

import gpytorch
import qpytorch
from qpytorch.distributions import (
    MultitaskMultivariateQExponential,
    MultivariateQExponential,
)
from qpytorch.likelihoods import MultitaskQExponentialLikelihood
from qpytorch.means import ConstantMean, LinearMean
from qpytorch.kernels import MaternKernel, ScaleKernel
from qpytorch.mlls import DeepApproximateMLL, VariationalELBO
from qpytorch.models.deep_qeps import DeepQEP, DeepQEPLayer
from qpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

REF_REPO = "/Users/zchan/eclipse-workspace/DeepQEP"
REF_COMMIT = "aa843960a3e8692e06eb36522fdbb2043bf34f2e"


class DQEPHiddenLayer(DeepQEPLayer):
    """Verbatim structure of the reference demo's DQEPHiddenLayer."""

    def __init__(self, input_dims, output_dims, power, num_inducing=64,
                 mean_type="constant"):
        self.power = power
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape,
            power=self.power)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True)
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = {"constant": ConstantMean(),
                            "linear": LinearMean(input_dims)}[mean_type]
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape,
                         ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x):
        return MultivariateQExponential(self.mean_module(x),
                                        self.covar_module(x), power=self.power)


class DeepQEPRegressor(DeepQEP):
    """2-layer Deep QEP for scalar regression. q=2 gives the Deep-GP control."""

    def __init__(self, input_dims: int, power: float, hidden_dims: int = 3,
                 num_inducing: int = 64):
        self.power_val = float(power)
        p = torch.tensor(float(power))
        hidden = DQEPHiddenLayer(input_dims, hidden_dims, p,
                                 num_inducing=num_inducing, mean_type="linear")
        last = DQEPHiddenLayer(hidden_dims, 1, p,
                               num_inducing=num_inducing, mean_type="constant")
        super().__init__()
        self.hidden_layer = hidden
        self.last_layer = last
        self.likelihood = MultitaskQExponentialLikelihood(num_tasks=1, power=p)

    def forward(self, inputs):
        return self.last_layer(self.hidden_layer(inputs))


def train_deep(X: torch.Tensor, Y: torch.Tensor, power: float, *,
               seed: int, iters: int = 300, lr: float = 0.01,
               hidden_dims: int = 3, num_inducing: int = 64,
               n_lik_samples: int = 8) -> Tuple[DeepQEPRegressor, list]:
    """Train one arm. Seeding is identical across q so the pair is matched.

    The seed is set immediately before construction AND before the loop, so the
    inducing-point draw, the variational init and the ELBO's MC stream are the
    same for q=2 and q=1.5.
    """
    torch.manual_seed(seed)
    model = DeepQEPRegressor(X.shape[-1], power, hidden_dims=hidden_dims,
                             num_inducing=num_inducing)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = DeepApproximateMLL(
        VariationalELBO(model.likelihood, model, num_data=Y.shape[0]))
    torch.manual_seed(seed)                     # common MC stream across q
    losses = []
    with gpytorch.settings.num_likelihood_samples(n_lik_samples):
        for _ in range(iters):
            opt.zero_grad()
            out = model(X)
            loss = -mll(out, Y.unsqueeze(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss))
    return model, losses


def predict_mean(model: DeepQEPRegressor, Xq: torch.Tensor, *,
                 n_samples: int, seed: int) -> np.ndarray:
    """MC predictive mean. `seed` fixes the sample stream (common random numbers)."""
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(n_samples):
        pred = model.likelihood(model(Xq)).to_data_independent_dist()
        m = pred.mean.mean(0)                   # average the MC sample dim
    return m.squeeze(-1).cpu().numpy()


def predict_mean_batched(model: DeepQEPRegressor, Xq: torch.Tensor, *,
                         n_batches: int, n_samples: int, seed0: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Repeat the MC predictive mean with different streams.

    Returns (pooled mean, per-point MC standard error of that pooled mean).
    """
    est = [predict_mean(model, Xq, n_samples=n_samples, seed=seed0 + k)
           for k in range(n_batches)]
    A = np.stack(est, 0)
    return A.mean(0), A.std(0, ddof=1) / math.sqrt(n_batches)


def env_info() -> Dict:
    return dict(python_torch=torch.__version__,
                gpytorch=gpytorch.__version__, gpytorch_file=gpytorch.__file__,
                qpytorch=qpytorch.__version__, qpytorch_file=qpytorch.__file__,
                reference_repo=REF_REPO, reference_commit=REF_COMMIT,
                fork_on_sys_path=False,
                deep_qep_module=qpytorch.models.deep_qeps.deep_qep.__file__)
