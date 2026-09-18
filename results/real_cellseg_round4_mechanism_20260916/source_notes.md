# Source notes

Status labels: OBSERVED (measured here), DERIVED UNDER ASSUMPTIONS, HYPOTHESIS, UNVERIFIED.

## Papers

### Bayesian Learning via Q-Exponential Process — Li, O'Connor, Lan, NeurIPS 2023
Fetched: https://papers.neurips.cc/paper_files/paper/2023/hash/e6bfdd58f1326ff821a1b92743963bdf-Abstract-Conference.html
Retrieved: abstract page only.
- STATED (A): the q-exponential distribution has density proportional to
  `exp(-1/2 |u|^q)`, generalized to a process by specifying consistent
  multivariate q-exponential distributions from elliptic contour distributions.
- STATED (A): q enters as an **L_q regularization of functions**, giving "a
  flexible prior on functions with sharper penalty (q<2)" versus GP (q=2).
- STATED (A): the paper advertises a "tractable prediction formula".
- UNVERIFIED: the explicit prediction equation and whether the predictive MEAN
  coincides with the GP mean. Only the abstract page was retrievable; the full
  PDF and the exact equation numbers were NOT obtained. I did not guess an
  arXiv identifier after one lookup returned an unrelated paper, so no equation
  is cited here.

### Deep Q-Exponential Processes — Chang, Obite, Zhou, Lan, PMLR v289
Fetched: https://proceedings.mlr.press/v289/chang25a.html
Retrieved: abstract page only.
- STATED (A): stacks multiple Q-EP layers as latent variable models; uses
  sparse approximation via inducing points and variational inference.
- STATED (A): q is an `L_q` relaxation control with "q = 2 corresponding to
  GP"; alternative q gives "more desirable regularization properties",
  motivated for "objects with inhomogeneous features like image edges".
- UNVERIFIED: layer-wise equations, the variational family details, and any
  theoretical statement about the deep predictive mean.

### Bayesian Regularization of Latent Representation (ICLR 2025)
URL supplied: https://proceedings.iclr.cc/paper_files/paper/2025/hash/2232e8fee69b150005ac420bfa83d705-Abstract-Conference.html
- NOT RETRIEVED in this session. Treated as UNVERIFIED; no claim in this report
  depends on it.

### Unsupervised Cell Segmentation by Fast Gaussian Processes
- NOT RETRIEVED. The repository contains a Python re-implementation of its
  pipeline (`py_core/Modified_Functions_RGasp.py`, `r_reference/*.R`), which is
  what this project actually measures. No paper equation is cited.

### Local search for sources
`find` over /Users/zchan/eclipse-workspace for *.pdf/*.tex/*.bib returned only
figures produced by this project (compare_comp_time.pdf,
gaussian_vs_qep_density.pdf, linear_diffusion_qep_common_scale.pdf). No paper
sources, no Overleaf checkout, in either repository.

## Implementation (what the code actually computes) — all OBSERVED

Installed: qpytorch 0.2, gpytorch 1.15.1, torch 2.10.0, skimage 0.25.2,
python 3.10 at /Users/zchan/miniforge3/envs/gpytorch_arm/bin/python.

| claim | evidence |
|---|---|
| `power` == q, no reparameterization; power=2 reproduces the Gaussian `log_prob` and `expected_log_prob` bit-for-bit | `experiments/simulated/qep_power_semantics_check.py` |
| q appears in the density via `inv_quad**(power/2)` plus `0.5*d*(q/2-1)*log(inv_quad)` and `log(q/2)` | `qpytorch/distributions/multivariate_qexponential.py::log_prob` |
| `qpytorch/models/exact_qep.py` is structurally identical to `gpytorch/models/exact_gp.py` (diff shows only names/docstrings) | `diff` performed this session |
| ExactQEP predictive MEAN is q-independent at fixed hyperparameters | max abs diff 0.000e+00 over q in {1.0,1.2,1.5,1.8,2.0,3.0} |
| ExactQEP predictive VARIANCE is q-independent AND data-independent | 0.000e+00 across q, and 0.000e+00 between two datasets differing 3x in scale on the same design |
| Derivative-aware QEP (`Matern52KernelGrad`, 3 tasks, Diff_QEP's architecture) has q-independent predictive mean AND variance at fixed hyperparameters | 0.000e+00 for q in {1.2,1.5,1.8} vs q=2 |
| **DeepQEP predictive mean IS q-dependent** at identical init and identical RNG stream | max abs diff 3.893e-04 (q=1.5), 6.134e-04 (q=1.2), exactly 0 for q=2 |
| the only q channel in the EXACT model is hyperparameter selection, and a GP given those hyperparameters reproduces the image exactly | `experiments/simulated/cell_blobs_gp_equivalence_check.py`, 36/36 pixel-identical |

## Consequence used in this report

DERIVED UNDER ASSUMPTIONS: because every predictive moment of the exact and
derivative-augmented QEP is q-independent at fixed hyperparameters, no
q-specific decision path into the final masks exists through those models; q can
only shift which hyperparameters are learned, and that shift is reproducible by
a Gaussian model given the same hyperparameters. This is a statement about the
installed implementation, NOT a refutation of Q-EP theory. Deep Q-EP is the one
audited architecture where the channel is non-zero.
