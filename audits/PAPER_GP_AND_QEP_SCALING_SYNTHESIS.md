# Synthesis — `paper_fast_gp` and QEP predictive scaling

2026-09-17. Draws only on `audits/PAPER_FAST_GP_WIRING_REPORT.md` and
`audits/QEP_CHUNK_INVARIANCE_AUDIT.md`. No Round 1–4 result was modified. No
full benchmark was run.

---

## 1. Is `paper_fast_gp` now a faithful reconstruction of the original paper's Fast-GP reconstruction stage?

**Yes for the model and the computation; no for the optimizer's output on one of
the two development images.** The distinction is real and worth keeping.

Faithful, **VERIFIED**:

- Separable Matérn-5/2 `K1 ⊗ K2` with distinct inverse-range `beta1`, `beta2`;
  nugget on the Kronecker eigenvalues; `log` parameterization;
  `param_ini = (-2,-2,-3)`; normalized `seq(0,1)` lattice inputs; all pixels;
  Kronecker eigendecomposition; constant mean and scale profiled analytically;
  shared `(beta1, beta2, nu)` from the first tile reused on all later tiles;
  `theta_hat` re-profiled per tile.
- Given R's own fitted parameters, our reconstruction matches R's to
  `max|diff| = 5.3e-15` (nuclei) and `5.3e-14` (whole_cell), with `theta_hat` to
  `2.9e-15` and `S_2` to `7.4e-13`. **The linear algebra is exact.**
- End-to-end on whole_cell (each side running its own optimizer):
  `max|diff| = 1.95e-06`, correlation `1.000000000000`.

Not faithful, **VERIFIED**:

- End-to-end on nuclei: `max|diff| = 4.26e-02`, correlation 0.993367, because
  R's L-BFGS-B stops at a point that scores **129310.4853** while ours scores
  **116922.5219** under R's *own* objective code — worse by 12388. R reports
  `convergence = 0` after 24 evaluations.
- The cause is the objective, not either implementation: it is non-convex with a
  flat `beta → ∞` plateau (identical to 4 dp from `beta2 = 2981` to `1.3e9`), on
  which `beta` is unidentifiable. R's fit is degenerate on `beta2` for nuclei and
  on `beta1` for whole_cell — an axis-aligned smoother with **zero** correlation
  across one axis (effective range 4.2e-05 px and 5.2e-06 px respectively).
  Multi-start finds 2 local optima on nuclei and 5 on whole_cell; the paper's own
  initialization reaches the best basin on nuclei but not on whole_cell
  (61119.2354 vs 41016.7491 reachable).

So: **the paper's estimator is not at the global optimum of its own objective on
either real development tile.** We default to reproducing the paper's *procedure*
(`n_restarts = 0`), and expose `n_restarts > 0` as an explicitly labelled
deviation that fits the model better. Both are cheap to run.

---

## 2. What remains different from the full original R/EBImage segmentation pipeline?

Reconstruction stage (minor, documented, mostly switchable):

| # | difference |
|---|---|
| 1 | Optimizer basin on nuclei (item above). Ours is better by 12388 in the paper's own objective. |
| 2 | `beta` on a degenerate axis is unidentifiable, so reported `beta` values are not comparable between implementations even when the reconstructions agree to 2e-6. |
| 3 | Uncovered trailing strip: the original leaves it 0; we default to `remainder="raw"`. `remainder="zero"` restores the original. |
| 4 | `run_segmentation`'s last tile absorbs the remainder; the original always crops a fixed size. Pre-existing, identical for every arm, so unbiased for comparisons. |
| 5 | We skip `separable_GP`'s finite-difference gradients — the original computes and never reads them. |

Downstream stages — **unchanged this session and still materially different**:

| # | stage | original | ours |
|---|---|---|---|
| 6 | foreground threshold | `criterion_1` smoothed with RobustGaSP `rgasp()` | `gaussian_filter1d(sigma=2)`, or Li (round-3 deliberate replacement) |
| 7 | outlier tiles | `outlier_threshold = 2`, re-threshold at the non-outlier mean, revert if > 0.99 foreground | **absent** |
| 8 | distance map | `EBImage::distmap` | `scipy.ndimage.distance_transform_edt` |
| 9 | watershed | `EBImage::watershed`, defaults `tolerance = 1, ext = 1`, own tolerance-based seeding | `skimage.watershed` with explicit `peak_local_max` markers, or `markers=None` |
| 10 | IoU / AP | computed in `*_Generate_IoU.R` | `py_core/segmentation_eval.py`; never cross-checked numerically |

**EBImage is not installed**, so items 8–10 could not be executed in R at all.
Any claim of reproducing the paper's *published AP numbers* remains
**UNVERIFIED**, and nothing in this session changes that.

---

## 3. Can we now fairly compare Raw / `paper_fast_gp` / QEP under the same downstream pipeline?

**Yes.** That is exactly what the wiring achieves, and it is the narrow claim
that is now supported.

- All three are dispatched inside the same `run_segmentation` loop, sharing
  identical tiling, thresholding, marker generation, watershed, cleanup and
  metrics. Only the reconstruction branch differs. **OBSERVED.**
- `paper_fast_gp` is deterministic — no RNG anywhere in estimation or
  prediction — unlike the old `gp_legacy` arm, whose unseeded
  `np.random.choice` made its results irreproducible. **OBSERVED.**
- Verified end-to-end on a 400x400 crop: shared parameters identical on all 4
  tiles, `theta_hat` re-profiled per tile (0.21307 / 0.13844 / 0.19300 /
  0.17800), scale round-tripped (`input_scale = 255.0`). **VERIFIED.**
- Cost is no longer an obstacle: 3.5 s vs 504 s (nuclei) and 14.5 s vs 273 s
  (whole_cell) against the isotropic arm — 19–144× faster. **VERIFIED.**

Three caveats to carry into the comparison:

1. It compares **reconstructions under our downstream**, not the paper's
   pipeline. It cannot settle whether the *published* method beats Raw.
2. Report the basin. On whole_cell the paper-faithful default is degenerate;
   run both `n_restarts = 0` and `n_restarts = 8`.
3. On whole_cell `paper_fast_gp` barely smooths (`corr-to-raw = 0.999662`,
   `nu = 0.00225`), so Raw and `paper_fast_gp` are near-identical *inputs* there
   and the comparison has little room to resolve anything. **VERIFIED** — this is
   an observation about the reconstructions, **not** an AP prediction.

The two GP arms are genuinely different images — pairwise RMSE 5.69 and 9.69
with `max|diff|` 56.2 and 110.4 grey levels — compared directly to each other,
not via each one's RMSE to raw. **VERIFIED.**

---

## 4. Which QEP predictive quantities are genuinely q-dependent?

At fixed hyperparameters, exactly two things are q-**invariant** (both
`0.00e+00` / float-epsilon): the predictive **mean** and the internal **scale
matrix C** (hence `.variance = diag(C)`). The mechanism is structural: `power`
appears in `models/exact_prediction_strategies.py` only as something propagated
onto the output distribution, never in any linear algebra.

Genuinely q-dependent, **VERIFIED**:

| quantity | q = 1.2 | q = 2.0 | note |
|---|---|---|---|
| true predictive covariance | `a(q,d)·C` | `C` | up to 406× at d = 8192 |
| kurtosis (d = 1 marginal) | 7.354 | 2.998 | shape, not just scale |
| E\|f − mu\| (d = 1 marginal) | 0.064736 | 0.057507 | |
| P(\|f − mu\| > sd) (d = 1) | 0.3174 | 0.3174 → see note | tail mass redistributes |
| E[‖∇f‖ \| y] (multitask) | 1.516611 | 1.000973 | |
| P(‖∇f‖ > c \| y) (multitask) | 0.6000 | 0.4566 | |

Critically, **`a(q,d)·C` does not capture everything** — kurtosis moves
independently of any rescaling. That is what makes a controlled q-test
meaningful: a Gaussian model with a rescaled kernel cannot imitate it.

And `‖E[∇f]‖ ≠ E[‖∇f‖]`: the first is q-invariant (0.038816 for every q), the
second is not. They differ by 21–49× in the tested configuration.

---

## 5. Which of those quantities are also chunk / partition invariant?

Of 11 candidate (statistic, convention) pairs tested, **2 pass** both
q-sensitivity and partition invariance; 9 fail.

**PASS**

| statistic | convention | q-sensitivity | partition invariance |
|---|---|---|---|
| E\|f−mu\|, quantiles, P(\|f−mu\|>c), kurtosis | **C — univariate d = 1 marginal at `(mu(x0), sqrt(C(x0,x0)))`** | kurt 7.354 → 2.998 | **≤ 1e-14; tail exactly 0.000e+00** |
| E[‖∇f‖], P(‖∇f‖>c) | **A — default sampling, MULTITASK** | 1.5166 → 1.0010 | 0.4% spread (Monte-Carlo scale) |

**FAIL — q-invariant (criterion 1)**: predictive mean; `C(x,x)` / `.variance`;
`‖E[∇f]‖`; scalar `rescale=True` variance (it equals `C00`, so correcting the
scale removes the q-signal).

**FAIL — chunk-dependent (criterion 2)**: every sampled statistic from a scalar
`MultivariateQExponential` at the default `rescale=False` (variance drifts 205×
with batch size; partition rel diff up to 1.52); scalar `rescale=True` *shape*
statistics (kurtosis collapses 7.310 → 2.874 toward Gaussian, tail +42%);
multitask `rescale=True` gradients (0.328 → 1.093).

Mechanism, **VERIFIED**: one prediction call builds one joint elliptical law over
all `d` requested points with a single shared radius `χ²_d^(1/q)`, whose second
moment `a(q,d)` grows with `d` — so sampled spread at `x0` is a property of the
batch. `rescalor = sqrt(a(q, event_shape[0]))` normalizes that correctly for
scalar models, but the 1-D marginal of a d-dimensional elliptical vector tends to
Gaussian as `d` grows, so normalizing the variance cannot restore the *shape*.
In the multitask case the radius instead uses `shape[-1] = n_tasks`, confirmed by
`Cov = a(q, n_tasks)·C` matching in all 15 tested configurations — which makes
default multitask sampling chunk-invariant and makes `rescale=True` wrong there.

---

## 6. Is a single-layer uncertainty/boundary QEP experiment justified?

**Yes — and now on stated evidence rather than plausibility, with one honest
demotion.**

Justified because all four gates the previous rounds set are met:

1. **The channel is measured, not assumed.** `E[‖∇f‖]` moves 1.52 → 1.00 and
   `P(‖∇f‖>c)` 0.600 → 0.457 across `q ∈ [1.2, 2.0]` at pinned hyperparameters.
2. **It survives the partition test**, which is the gate that killed the naive
   version: 0.4% spread across 1x100 / 2x50 / 10x10 / 100x1 partitions.
3. **The Gaussian control cannot imitate it by rescaling**, because kurtosis
   changes independently of scale.
4. **The architecture is shared** — `power = 2` skips the `χ²` branch entirely,
   so the q = 2 arm is the same machinery.

It also targets the measured dominant failure from Round 4 (adjacent-cell
**merging**, i.e. a boundary-evidence deficit), rather than a mechanism chosen
because it sounded plausible.

**The honest demotion.** Multitask convention A is chunk-invariant *because the
implementation gives each spatial point its own independent `χ²_{n_tasks}`
radius* — it has factorized the elliptical structure across space. So it is a
legitimate q-dependent boundary feature, but it is **not** the paper's joint
q-EP posterior, and must not be described as such. If joint-posterior semantics
are wanted, convention C (the d = 1 marginal) is the defensible choice, at the
cost of discarding spatial coupling in the uncertainty.

Mandatory preconditions, from B7: never use `rescale=True` for the gradient
statistic; never use default scalar sampling; fix and report the chunk
partition, and re-assert partition invariance at the actual image sizes (verified
only to d = 8192 in 1-D and 100 points in 2-D); distinguish `‖E[∇f]‖` from
`E[‖∇f‖]` in the write-up.

Still **not** justified, unchanged: PDE-informed segmentation; MAP /
transform-domain priors; treating derived gradient observations as independent
data; and any use of `.variance` as an uncertainty — B2 shows it is a scale
parameter, q-invariant, and therefore measures nothing q-dependent.

---

## 7. What ONE experiment should be run next?

**Score Raw vs `paper_fast_gp` vs QEP on the two development images under the
now-identical downstream pipeline — the corrected-baseline benchmark. Not the
gradient-tail experiment.**

Why this and not the QEP mechanism, despite the mechanism now being justified:
every QEP claim is a *delta from a baseline*, and until this session there was no
valid baseline. Answering "does q help?" against a baseline known to be wrong in
seven ways produces an uninterpretable number. This experiment is also far
cheaper — `paper_fast_gp` runs in 3.5 s/image versus 504 s for the old arm — and
it directly settles the standing question of whether "Raw > GP" was ever about
the paper's method.

Scope, in order, stopping at the first failure:

1. Both basins for `paper_fast_gp`: `n_restarts = 0` (paper-faithful) and
   `n_restarts = 8`, reported separately.
2. Arms: `raw`, `paper_fast_gp`, `paper_fast_gp_ms8`, `gp_isotropic_gpytorch_2025`
   (for continuity with Rounds 1–4), `gp` (q = 2 control), `qep` at q = 1.5.
3. Fixed downstream: the round-3 Li threshold, `marker_mode="peak"`,
   `min_distance` 15 (nuclei) / 9 (whole_cell), unchanged cleanup and metrics.
4. Two development images only; held-out images stay untouched.
5. Report AP@0.5, matched-IoU, and the Round-4 failure taxonomy, plus the
   reconstruction-level pairwise comparison already produced here.

Pre-registered readings: if `paper_fast_gp` still loses to Raw, that is a finding
about the paper's reconstruction *under our downstream*, and must be stated with
that qualifier. If it wins, Rounds 1–4's GP arm was the problem and their
comparative conclusions need restating. Either way the whole-cell result will be
weakly informative, because `paper_fast_gp ≈ raw` there (`corr-to-raw` 0.9997).

The gradient-tail QEP experiment is the one after that, under the B7 conditions.

**Not started. Awaiting approval.**

---

# FINAL REPORT

## Files created

| file | purpose |
|---|---|
`py_core/paper_fast_gp.py` | the faithful Fast-GP reconstruction port (new module)
`audits/taskA2_real_tile_r_side.R` | R side of the real-tile parity test
`audits/taskA2_real_tile_py_side.py` | Python side of the real-tile parity test
`audits/taskA2_objective_landscape.py` | multi-start + objective profile diagnosis
`audits/taskA4_paper_fast_gp_smoke.py` | reconstruction-only smoke test
`audits/qep_chunk_invariance_check.py` | Part B, sections B1–B6
`audits/PAPER_FAST_GP_WIRING_REPORT.md` | Part A deliverable
`audits/QEP_CHUNK_INVARIANCE_AUDIT.md` | Part B deliverable
`audits/PAPER_GP_AND_QEP_SCALING_SYNTHESIS.md` | this file
`audits/parity/real_{nuclei,whole_cell}_r_{params,tile,predmean,meta}.csv` | R reference dumps
`audits/parity/real_tile_parity_summary.csv` | parity deltas
`audits/parity/real_tile_landscape_{multistart,profile}_{nuclei,whole_cell}.csv` | landscape
`audits/parity/chunk_{versions,b2_scales,b3_batchsize,b4_partition,b5_derivative,b6_dimension}.csv` | Part B numerics
`audits/parity/chunk_run.log` | Part B raw log
`results/audit_paper_fast_gp_smoke_20260917/` | reconstructions (.npz), figures, `reconstruction_summary.csv`, `shared_parameter_table.csv`, `per_tile_diagnostics.csv`

## Files modified

**One file only:** `py_core/segmentation_pipeline.py` — module docstring
corrected; `METHODS` extended with `paper_fast_gp`; new `METHOD_LABELS` and
`method_label()`; `run_segmentation` gained a `paper_fast_gp_restarts` keyword, a
`shared_pfg` accumulator and a `paper_fast_gp` dispatch branch; new import from
`py_core.paper_fast_gp`.

Unchanged: thresholding, marker generation, watershed, cleanup, metrics, and
every existing arm's numerics. **No file under `results/` from Rounds 1–4 was
modified or deleted.** Stored method keys are unchanged, so old CSVs still
resolve.

## R reference repository commit

`https://github.com/UncertaintyQuantification/cell_segmentation.git` at
**`44714c2e0be958fe796a8fd4bdbc220dae3c23dd`** (short `44714c2`, 2025-09-04,
"Update on Sept 4"), cloned read-only at
`/Users/zchan/eclipse-workspace/cell_segmentation_original`, not in this
project's git index. R packages present: `magick 2.9.1`, `pracma 2.4.6`,
`RobustGaSP 0.6.8`, `plot3D 1.4.2`, `dplyr 1.1.4`. **EBImage MISSING.**

## Real-tile parity result

**Linear algebra: EXACT.** Given R's own parameters, `predmean max|diff|` =
**5.33e-15** (nuclei 282x240) and **5.29e-14** (whole_cell 200x200);
`theta_hat` 2.9e-15; `S_2` 7.4e-13. Inputs verified bit-identical between
`magick` and `imageio` (0 pixels differing above 1e-9 in both files).

**End-to-end with each side's own optimizer:** whole_cell agrees
(`max|diff| = 1.95e-06`, corr 1.000000000000); nuclei does not
(`max|diff| = 4.26e-02`, corr 0.993367) because R's L-BFGS-B stops 12388 worse in
its own objective (129310.4853 vs 116922.5219) while reporting
`convergence = 0`. The objective is non-convex with a flat `beta → ∞` plateau; R
returns a degenerate zero-correlation axis on **both** development tiles.

## `paper_fast_gp` smoke-test result

Reconstruction only; no AP computed.

| dataset | arm | runtime | RMSE-to-raw | corr-to-raw | beta1 | beta2 | nu | degenerate |
|---|---|---|---|---|---|---|---|---|
| nuclei | gp_isotropic_gpytorch_2025 | 504.0 s | 7.7356 | 0.921729 | — | — | — | — |
| nuclei | paper_fast_gp | **3.5 s** | 5.0520 | 0.966086 | 35.50395 | 23.68895 | 0.170597 | none |
| whole_cell | gp_isotropic_gpytorch_2025 | 273.3 s | 9.8702 | 0.973863 | — | — | — | — |
| whole_cell | paper_fast_gp | **14.5 s** | 1.0652 | 0.999662 | 25232.17 | 37.99262 | 0.002251 | **rows** |
| whole_cell | paper_fast_gp_ms8 | 55.7 s | 0.5899 | 0.999896 | 81.01078 | 62.33656 | 0.001954 | none |

The two GP arms differ substantially from each other: pairwise RMSE 5.69
(nuclei) and 9.69 (whole_cell), `max|diff|` 56.2 and 110.4 grey levels.
Wiring verified end-to-end on a 400x400 crop: first-tile estimation, cross-tile
reuse, per-tile `theta_hat` re-profiling, degeneracy warning, scale round-trip.

## Chunk-invariance verdict

The q scaling dimension is **`event_shape[0]` = the number of test query points
in the single prediction call** (scalar models); for multitask, `rescalor` uses
`n_points` while the sampling radius uses `shape[-1] = n_tasks`, which is an
**implementation inconsistency**. Default scalar sampling is catastrophically
chunk-dependent — 205× variance drift for the *same physical point* between
batch sizes 1 and 8192, partition relative differences to 1.52.
`rescale=True` fixes the scalar **variance** only (and thereby removes its
q-sensitivity); it does **not** fix shape statistics (kurtosis collapses
7.310 → 2.874 toward Gaussian), and it **breaks** the multitask gradient
statistics. `mu(x)` and `C(x,x)` are invariant to 4e-14 but q-invariant.
Overall classification: mathematically intended finite-dimensional behaviour plus
an inappropriate pointwise interpretation on our side, with one genuine
implementation inconsistency in the multitask `rescalor`.

## Safe q-dependent quantities

Two of eleven tested (statistic, convention) pairs pass both criteria:

1. **Convention C — the univariate d = 1 marginal** at `(mu(x0),
   sqrt(C(x0,x0)))`: `E|f−mu|`, quantiles, `P(|f−mu|>c)`, kurtosis.
   Partition-invariant to `≤ 1e-14` (tail probability identically 0.000e+00),
   q-sensitive (kurtosis 7.354 at q = 1.2 vs 2.998 at q = 2.0). Strongest option.
2. **Convention A default sampling, MULTITASK** gradient statistics:
   `E[‖∇f‖]` (1.5166 → 1.0010 across q; 0.4% across partitions) and
   `P(‖∇f‖>c)` (0.600 → 0.457). Usable, but its invariance follows from
   per-point independent radii, so it is **not** the paper's joint q-EP
   posterior and must be described accordingly.

Not safe: all default scalar sampled statistics, scalar `rescale=True` shape
statistics, multitask `rescale=True` gradients, `.variance` as an uncertainty,
and `confidence_region()` at its default.

## Proposed next experiment

**The corrected-baseline benchmark**: Raw vs `paper_fast_gp` (both basins) vs
`gp_isotropic_gpytorch_2025` vs `gp` (q=2) vs `qep` (q=1.5), on the two
development images, under the identical fixed downstream pipeline (Li threshold,
peak markers, unchanged cleanup and metrics). Cheap — 3.5 s/image for the new
arm. This settles whether "Raw > GP" was ever a statement about the paper's
method. The single-layer gradient-tail QEP experiment comes after, under the B7
conditions.

**Not started. Awaiting approval.**
