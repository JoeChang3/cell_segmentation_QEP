# Mechanism cards

Status labels: OBSERVED (measured in this round), DERIVED UNDER ASSUMPTIONS,
HYPOTHESIS, UNVERIFIED.

The target failure for every card is the one Phase 1/2 located:

> **Adjacent-cell MERGING at instance separation.** OBSERVED: merged accounts
> for 23.4-26.9% of all whole-cell GT cells and 13.4-23.2% of nuclei cells,
> against only 1.7-3.5% missing_foreground. Merged cells carry **0.693
> (whole-cell) / 0.526 (nuclei) markers each** versus **1.384 / 1.398** for
> matched cells, while their foreground coverage is equal or higher (0.720 vs
> 0.690; 0.873 vs 0.870). So the cells are present in the foreground and simply
> never receive their own watershed seed. Oracle markers on the real foreground
> lift held-out AP@0.5 from 0.2248 to 0.4350 (whole-cell) and 0.7543 to 0.9001
> (nuclei, merged 32.2 -> 8.0).

---

## Card 1 — Exact-QEP posterior mean / variance / derivative maps as a boundary signal

**REJECTED. No q-sensitive decision path exists.**

- Chain attempted: merging -> need an inter-cell boundary signal -> take it from
  a QEP posterior quantity -> use it as marker surface or watershed elevation.
- Where q enters: `qpytorch/distributions/multivariate_qexponential.py::log_prob`
  only, via `inv_quad**(q/2)`, `0.5*d*(q/2-1)*log(inv_quad)` and `log(q/2)`.
- What the implementation actually computes (B): `qpytorch/models/exact_qep.py`
  is structurally identical to `gpytorch/models/exact_gp.py` (verified by diff;
  only names and docstrings differ).
- OBSERVED, fixed hyperparameters, max absolute difference versus q=2:
  | quantity | result |
  |---|---|
  | ExactQEP predictive mean, q in {1.0,1.2,1.5,1.8,3.0} | **0.000e+00** |
  | ExactQEP predictive variance, same q grid | **0.000e+00** |
  | ExactQEP variance across two datasets differing 3x in scale | **0.000e+00** |
  | derivative-aware QEP (`Matern52KernelGrad`, 3 tasks) mean AND variance | **0.000e+00** |
- Consequence (DERIVED UNDER ASSUMPTIONS): every candidate boundary quantity is
  q-independent. `||E[grad f]||` is available but equals the Gaussian value;
  `E[||grad f||]` needs the derivative posterior covariance, which is also
  q-independent; boundary-event probabilities need the same covariance. The
  safeguard about `norm(E[grad])` vs `E[norm(grad)]` is therefore moot here:
  both are q-invariant in this implementation.
- Simple non-QEP explanation that fully accounts for anything observed: an
  ordinary GP with the same hyperparameters. Already demonstrated 36/36
  pixel-identical in `experiments/simulated/cell_blobs_gp_equivalence_check.py`.
- Verdict: reject. Not "q is useless", but "this implementation exposes no
  q-dependent output that could reach a segmentation decision".

---

## Card 2 — Intensity-informed separation surface (NON-QEP competing explanation)

**TESTED AND FAILED its pre-registered criteria.**

- Chain: merged cells share one seed -> the distance transform of a fused blob
  has a single broad maximum -> add image intensity, which dips between touching
  cells, to either the flooding surface (Pilot 1) or the marker surface
  (Pilot 2).
- Code changed: `separate_intensity_aware` in
  `experiments/real_data/round4_phase5_pilots.py`, and the Pilot 2 script.
  Markers/elevation/cleanup otherwise identical to the frozen round-3 rule;
  `w = 1` / `wm = 1` reproduces it exactly.
- Pre-registered criterion (a): merged count must fall >= 15% relative on at
  least one dataset. Written to `pilot_configs/` before any pilot score.
- OBSERVED held-out, whole-cell:
  | pilot | arm | AP@0.5 frozen -> pilot | merged frozen -> pilot |
  |---|---|---|---|
  | P1 elevation w=0.2 | raw | 0.2248 -> 0.2674 | 33.5 -> 33.8 (+0.9%) |
  | P1 elevation w=0.2 | gp/q=2 | 0.1821 -> 0.2185 | 34.8 -> 34.2 (-1.7%) |
  | P1 elevation w=0.2 | q=1.5 | 0.2107 -> 0.2260 | 37.2 -> 39.0 (+4.8%) |
  | P2 markers wm=0.5 | raw | 0.2248 -> 0.2393 | 33.5 -> 38.0 (+13.4%) |
  | P2 markers wm=0.5 | gp/q=2 | 0.1821 -> 0.1740 | 34.8 -> 35.5 (+2.0%) |
  | P2 markers wm=0.5 | q=1.5 | 0.2107 -> 0.1873 | 37.2 -> 36.8 (-1.1%) |
- Criterion (a) fails in both pilots; the best merged reduction is 1.7%. Pilot 1
  did raise AP (+0.043 raw) but by cutting SPLITS (5.8 -> 3.2), not merges.
  Pilot 2 selected wm=0.5 on development (merged 147.5 -> 144.5, AP 0.4671 ->
  0.4962) and that did NOT transfer: held-out merges got worse.
- QEP-specific criterion (q=1.5 must beat BOTH its Gaussian twin AND Raw): fails
  in both pilots. q=1.5 beats q=2 in P1 (0.2260 vs 0.2185) but loses to Raw
  (0.2674).
- Why it fails (HYPOTHESIS, consistent with the marker counts): raw already
  emits 1137 markers for ~155 whole-cell GT cells, a 7.3x over-seeding, and
  still merges 33.5 cells. The surplus seeds sit in noise specks removed by
  cleanup, while a fused pair still shares one seed. Blending intensity into a
  global surface does not put a seed inside each fused cell, because whole-cell
  intensity is not reliably peaked at cell centres.
- Verdict: the mechanism is real but insufficient. Retain Pilot 1's elevation as
  a small non-QEP AP gain candidate (+0.043 raw whole-cell) needing fresh
  validation; it is not a merge fix.

---

## Card 3 — Deep Q-EP latent representation

**ONLY LIVE QEP CANDIDATE. Not tested. Deprioritized for now, with a stated reason.**

- Source (A): *Deep Q-Exponential Processes*, Chang, Obite, Zhou, Lan, PMLR v289
  (fetched abstract). States: multiple Q-EP layers as latent variable models,
  sparse approximation via inducing points, variational inference; q is an `L_q`
  relaxation with q=2 recovering GP; motivated for "objects with inhomogeneous
  features like image edges". UNVERIFIED: layer equations, variational family
  details, any claim about the deep predictive mean (abstract page only).
- Why this card survives when Card 1 is rejected (OBSERVED): the invariance in
  Card 1 is a property of EXACT single-layer inference. In a deep model the
  intermediate layer is SAMPLED from a q-dependent distribution and passed
  non-linearly onward, so the mean is not a fixed linear smoother. Measured with
  identical parameter initialization and an identical RNG stream
  (`qpytorch.models.deep_qeps`, 2-layer, 2 hidden dims):
  | q | max abs difference of predictive mean vs q=2 |
  |---|---|
  | 2.0 | 0.000e+00 |
  | 1.5 | **3.893e-04** |
  | 1.2 | **6.134e-04** |
  The channel is non-zero, unlike every quantity in Card 1.
- Chain that would have to hold: q<2 latent regularization -> a latent
  representation that keeps inter-cell intensity dips instead of smoothing them
  -> a reconstruction or latent map whose minima lie between touching cells ->
  markers/elevation derived from it place one seed per cell -> merged count
  falls.
- What must change in code: a Deep Q-EP image model (2+ layers, inducing
  points, `DeepApproximateMLL` + `VariationalELBO`), producing a per-pixel map
  consumed by the existing frozen marker/watershed/cleanup stages.
- What must stay fixed: peak markers with min_distance 15/9, cleanup 50 px,
  AP = TP/(TP+FP+FN), the Raw baseline.
- Mandatory control: a **Deep GP with identical depth, inducing points,
  variational family and training budget**. The magnitudes above (~6e-4 at
  initialization) are small, so depth/architecture is at least as plausible an
  explanation as q, and the Gaussian counterpart must be given the same
  machinery.
- Simple non-QEP competing explanations: (i) a Deep GP does the same;
  (ii) any learned non-linear feature map does the same; (iii) a classical
  seeded-watershed improvement using cell-shape priors beats both.
- Cost: high. Round-2/3 timings were 360-650 s per image per arm for a
  single-layer exact model; a deep variational model on 8 images x 2 model
  families is a multi-hour job and needs its own convergence checks.
- Evidence link to the located failure: **weak**. Nothing measured here shows
  that a q<2 latent representation preserves inter-cell dips. That is the
  HYPOTHESIS, not a finding.
- Verdict: the one mechanism with a verified non-zero q-channel, but it does not
  yet earn a multi-hour experiment ahead of the cheaper marker question, because
  the located failure is a *seed-placement* problem and no evidence connects a
  latent q-prior to seed placement.

---

## Rejected without a card

- **PDE / Diff_QEP solver reuse.** The located failure is seed placement inside
  a fused binary blob. No PDE residual, boundary condition or forcing term is
  available for a microscopy image, and building one would be an unrelated model
  invented to use the solver. Explicitly out of scope per the brief.
- **MAP with a coordinate-wise sparse transform prior.** The radial term
  `(u' C^-1 u)^(q/2)` in the Q-EP density is NOT the same object as
  `sum_j |(T u)_j|^q`; treating them as equivalent would be a different model.
  A TV/wavelet MAP baseline is worth knowing about, but it is a non-QEP
  edge-preserving denoiser, and Pilot 1/2 show that a better boundary SURFACE is
  not what the merge failure needs.
- **Marginal-likelihood unboundedness at small q.** Not claimed here. The
  earlier statement was withdrawn in round 2; the objective and parameter path
  were not re-examined in this round, so the question stays open and unused.
