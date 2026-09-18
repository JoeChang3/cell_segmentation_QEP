# A + B Synthesis

2026-09-17. Answers to the seven synthesis questions, drawing only on
`audits/TASK_A_FASTGP_BASELINE_AUDIT.md` and
`audits/TASK_B_QPYTORCH_SEMANTICS_AUDIT.md`. No existing results were modified
and no large experiment was started.

---

## 1. Have we actually reproduced the Fast-GP method from the base paper yet?

**No.**

The GP *core* is reproduced; the *method* is not.

- **Reproduced (VERIFIED).** `py_core/dim_2_lattice.py` matches the original R
  `separable_GP_param_est` + `separable_GP` to `max|diff| = 1.03e-07` on the
  predictive mean at n=40 and `1.12e-07` at n=120, with hyperparameter agreement
  to `|Δbeta1| ≤ 4.2e-05` and `|Δnu| ≤ 5.0e-07`.
- **Not reproduced.** That code is **not wired into the real pipeline**. It is a
  port of `lattice_alg`, which the original uses only in
  `Simulated_Experiments/*.R`. The real-data path is
  `generate_GP_Masks_test`, and our stand-in for it — the `gp_legacy` arm —
  differs from the paper in seven identified ways (isotropic instead of separable
  kernel, random 6000-pixel subsample instead of all pixels, per-tile instead of
  first-tile-only hyperparameters, Adam instead of profiled L-BFGS-B, float32,
  a Gaussian-filter `criterion_1` instead of RobustGaSP, `markers=None` watershed
  instead of tolerance-merging watershed, and no outlier-tile handling).

So: the paper's GP is available to us and is correct, but has never been run as
the paper runs it.

---

## 2. Is the current Raw > GP result evidence against the published Fast-GP method, or only against our current Python GP arm?

**Only against our current Python GP arm.** It is not admissible evidence about
the published method.

Reasons, in descending strength:

1. The arm that produced the result is not the paper's method (Q1), and at least
   two of the differences have *measured* large effects: watershed seeding
   (improved 32/32 cases when corrected) and the thresholding rule (nuclei raw AP
   0.3383 → 0.7543 when replaced with Li). Those two are downstream of the GP
   entirely.
2. The `gp_legacy` arm's hyperparameters are **not reproducible** — an unseeded
   `np.random.choice` at `py_core/Modified_Functions_RGasp.py:234`. This is
   documented in-repo: recomputing AP from the stored IoU matrix gives 0.0388
   against a stored 0.0307. A non-reproducible arm cannot support a comparative
   claim.
3. The isotropy loss is real, not hypothetical: the fitted parameters are
   genuinely anisotropic (`beta1/beta2 = 3.754/2.611 = 1.44` at n=40;
   `15.50/11.48 = 1.35` at n=120, VERIFIED). An isotropic kernel must split the
   difference on both axes.

Honest counter-consideration: the *direction* Raw ≳ GP is at least plausible for
the paper's method too, because a smoother that blurs a 1-pixel boundary can only
hurt a boundary-driven distance transform. But "plausible" is not evidence, and
the claim as currently stated must be scoped to our arm.

**Required relabelling:** every result table saying "GP" for rounds 1–4 should
read `gp_isotropic_gpytorch_2025` (or `gp_project_legacy`), not "GP" and not
"Fast GP".

---

## 3. Which exact parts of the original Fast-GP method must be restored for a fair baseline?

In the GP itself (all five are needed; each is OBSERVED in the original source):

1. **Separable Matérn-5/2 `K1 ⊗ K2`** with two inverse-range parameters, solved
   by eigendecomposition — not an isotropic joint-2D kernel.
2. **Normalized lattice inputs** `input1 = input2 = linspace(0, 1, n)` — not pixel
   indices. `param_ini = c(-2,-2,-3)` is calibrated to this scale.
3. **All pixels** for hyperparameter estimation — no subsample, hence no RNG.
4. **Hyperparameters estimated once on tile (1,1) and reused for every tile**, not
   re-estimated per tile.
5. **Profiled-likelihood L-BFGS-B** (mean and variance analytically profiled out),
   not Adam on the full objective; intensities as magick floats in `[0,1]`.

In the surrounding pipeline (needed for a fair *method* comparison, not merely a
fair GP):

6. **Outlier-tile handling** — `outlier_threshold = 2`, re-threshold at the mean of
   non-outlier percentages, revert a tile to background if it exceeds 0.99
   foreground. Currently MISSING, and its absence explains our observed
   catastrophic all-background/all-foreground tiles.
7. **Tolerance-merging watershed.** EBImage's `watershed(dist_map)` defaults to
   `tolerance = 1, ext = 1` and does its own seed detection, merging basins
   shallower than the tolerance. `markers=None` in skimage merges nothing, so
   `marker_mode="legacy_none"` is *more fragmented than the paper*, not equal to it.
8. **`criterion_1` with a RobustGaSP-equivalent smoother** — or an explicit,
   labelled decision to deviate. Round 3 already made that deviation knowingly
   (Li), so this one is a documentation obligation rather than a code change.

Items 1–5 are cheap: `dim_2_lattice.py` already implements 1 and 5 correctly, so
this is wiring plus three configuration changes. Items 6–7 are new code.

Cannot be restored in this environment: **EBImage is not installed**, so the
original `distmap`/`watershed` cannot be executed for a reference comparison.
Any watershed fidelity claim stays UNVERIFIED until that is resolved.

---

## 4. At fixed kernel hyperparameters, which QEP posterior quantities genuinely depend on q?

**q-INVARIANT — exactly two things (VERIFIED, `0.00e+00`):**

- the predictive **mean**;
- the internal **scale matrix C** (`covariance_matrix`) and its diagonal
  (`.variance`).

The mechanism is structural: in `models/exact_prediction_strategies.py`, `power`
is only propagated onto the output distribution and never enters any linear
algebra.

**q-DEPENDENT (VERIFIED):**

| quantity | q=1 | q=2 | q=3 |
|---|---|---|---|
| true predictive covariance, as a multiple of C | 42.0× | 1.0× | 0.29× |
| 90% interval width | 1.551 | 0.240 | 0.130 |
| E\|f − mean\| | 0.3707 | 0.0583 | 0.0315 |
| P(\|f − mean\| > 2·sd(C)) | 0.7437 | 0.0455 | 0.0001 |
| kurtosis | 3.447 | 3.003 | 2.923 |
| E[‖∇f‖ \| y] | 1.913 | 1.001 | 0.837 |
| P(‖∇f‖ > c \| y) | 0.6449 | 0.4560 | 0.3225 |

Two points that matter for designing anything on top of this:

- **`.variance` is a scale parameter, not a variance.** The true covariance is
  `a(q,d)·C` with `a(q,d) = 2^(2/q)·Γ(d/2+2/q)/(d·Γ(d/2))`, which is exactly
  `rescalor²` in the installed package (agreement to 9.3e-16). The default
  `rescale=False` sampling path carries this factor; `.variance` does not.
- **`a(q,d)·C` does not capture everything.** Kurtosis changes independently of
  scaling. This is what makes a q-dependent rule non-imitable by a Gaussian model
  with a rescaled kernel — i.e. it is what makes a controlled test meaningful.
- **`‖E[∇f]‖ ≠ E[‖∇f‖]`.** The first is q-invariant (0.038816 for all q); the
  second is q-dependent. They differ by 21–49× in the tested configuration, and
  the gap itself moves with q.

**IMPLEMENTATION-DEPENDENT (VERIFIED):** the inflation uses the **predictive
batch dimension** (`rescalor` reads `event_shape[0]`). The same test point at
q=1.5 has sampled variance 1.50× the reported value in a batch of 2 and 5.78× in
a batch of 200, tracking `a(1.5, n_test)` throughout.

---

## 5. Does our CURRENT segmentation pipeline consume any of those q-dependent quantities?

**No. Not one of them.**

`py_core/segmentation_pipeline.py::smooth_tile` returns a reconstructed
posterior **mean** image; everything after it —
`foreground_threshold` → `distance_transform_edt` → `peak_local_max` markers →
`watershed` → `eliminate_small_areas` → `evaluate_instances` — is a deterministic
function of that mean. No sampling, no quantile, no interval, no tail
probability, and `skip_posterior_variances(True)` is set in the prediction path.

This is the precise reason the round-1–4 observation "GP and q=2 are bit-for-bit
identical, and q=1.5 shows no stable advantage" is **correct and remains
correct**. The pipeline routes around the only channel through which q could have
acted. It is not that q does nothing; it is that we never read the output q
affects.

---

## 6. Is there a scientifically defensible single-layer QEP mechanism still worth testing before Deep QEP?

**Yes — one.** Round 4 closed this door on a false premise (claim 4/5 in B6), so
it is legitimately reopened.

**The candidate:** a boundary score built from the posterior **distribution of
the gradient**, specifically `E[‖∇f‖ | y]` or `P(‖∇f‖ > c | y)`, from a
single-layer `Matern52KernelGrad` QEP, used to modulate the watershed cost
surface (or to gate marker acceptance).

Why this one passes the evidence gate the previous rounds established:

1. **The channel is measured, not assumed.** `E[‖∇f‖]` moves 2.3× and
   `P(‖∇f‖>c)` moves 2.0× across q ∈ [1,3] at pinned hyperparameters (VERIFIED).
2. **It is not reproducible by the Gaussian control.** Kurtosis changes
   independently of scale, so a Gaussian model with a rescaled kernel cannot
   imitate the q < 2 statistic. This was the fatal flaw in every previously
   rejected mechanism card.
3. **It attacks the measured dominant failure.** Round 4 established that the
   dominant FN mechanism is **adjacent-cell merging** (marker counts 0.693/0.526
   vs 1.384/1.398), which is exactly a *boundary-evidence* deficit — the
   watershed has no reason to place a ridge between two touching cells. A
   gradient-tail statistic is evidence about precisely that.
4. **The architecture gives the Gaussian arm the same machinery** (same kernel,
   same 3-task structure, q=2), satisfying the round-4 safeguard.

**Two hard preconditions that must be satisfied first, or the test is worthless:**

- Divide by `rescalor` (or pass `rescale=True`). Otherwise the "q effect" is
  partly the `a(q,d)` scale factor, which *is* imitable by rescaling and therefore
  proves nothing.
- Fix the predictive batch size. With `predict_chunk = 8192` the inflation is
  ~90× at q=1.5, and the trailing partial chunk gets a different factor — a pure
  artifact that would appear as spatial structure in the boundary score.

**Explicitly NOT recommended** (unchanged from round 4, and this audit gives no
new support for any of them): PDE-informed segmentation; MAP/transform-domain
priors; treating derived gradient observations as independent data; and
uncertainty-based segmentation via `.variance`, which B2 shows is not an
uncertainty at all.

**A cheap secondary option**, noted but not recommended as the next step:
`distributions/power.py::Power` is a `Module` with constraint and prior support,
so q can be *learned* rather than fixed. No round has tried this. It is lower
value than the above because a learned q still only reaches the masks through the
mean, which is q-invariant.

---

## 7. What ONE experiment should be run next after these audits?

**Build and score `paper_fast_gp` — the faithful original Fast-GP — on the two
dev images, as a corrected baseline. Not the QEP mechanism.**

Ordering rationale: Q2 says our central comparative result currently has no valid
baseline. Every QEP question is a question about a *delta from the baseline*, so
measuring a q-effect against a baseline known to be wrong in seven ways would
produce an uninterpretable number. The QEP gradient mechanism from Q6 is the
experiment *after* this one.

It is also by far the cheaper of the two: the GP core is already written and
already validated to 1e-07.

Scope, in ascending cost, stopping at the first failure:

1. Extend the existing parity harness to n = 256 and to a whole-cell tile;
   assert predmean `max|diff| < 1e-05`. Seconds.
2. Wire `dim_2_lattice.py` into the tiled pipeline as a new `paper_fast_gp`
   method with the five A4 corrections. Confirm tile (1,1) reproduces the R
   `parameters$param` and that every later tile consumes it unchanged.
3. Add the missing outlier-tile handling and a tolerance-merging watershed.
4. Compare `paper_fast_gp` reconstructions **directly against**
   `gp_isotropic_gpytorch_2025` reconstructions — not via each one's RMSE to raw,
   which round 2 already showed is uninformative.
5. Only then score AP@0.5 on the 2 dev images and re-ask: is Raw > Fast-GP?

Success criterion, fixed in advance: if `paper_fast_gp` still loses to Raw, then
Raw > Fast-GP is a finding about the *published method* and can be reported as
such. If it wins, rounds 1–4's GP arm was the problem and their comparative
conclusions need restating.

Also required before step 5, as pure bookkeeping: rename the `gp_legacy` arm to
`gp_isotropic_gpytorch_2025` and annotate
`results/real_cellseg_round4_mechanism_20260916/source_notes.md` with the B6
corrections. Neither touches any stored numbers.

**Not to be started without approval.**

---

# FINAL REPORT

## Files created this session

Under `audits/` (all new; nothing overwritten, reverted or deleted):

| file | purpose |
|---|---|
| `audits/TASK_A_FASTGP_BASELINE_AUDIT.md` | Task A deliverable (A1–A5) |
| `audits/TASK_B_QPYTORCH_SEMANTICS_AUDIT.md` | Task B deliverable (B1–B7) |
| `audits/A_B_SYNTHESIS.md` | this file |
| `audits/taskA_parity_r_side.R` | R side of the R↔Python parity test |
| `audits/taskA_parity_py_side.py` | Python side of the parity test |
| `audits/qep_local_semantics_check.py` | B2, B3, B3b, B4, B4b, B5 |
| `audits/qep_mll_boundedness_check.py` | B6 claim 6 boundedness test |
| `audits/parity/r_params_n{40,120}.csv` | R fitted hyperparameters |
| `audits/parity/r_tile_n{40,120}.csv` | R-dumped input tiles (shared input) |
| `audits/parity/r_predmean_n{40,120}.csv` | R predictive means |
| `audits/parity/parity_summary_n{40,120}.csv` | parity deltas |
| `audits/parity/taskB_versions.csv` | installed versions |
| `audits/parity/taskB_rescalor_formula.csv` | `rescalor` vs `sqrt(a(q,d))` |
| `audits/parity/taskB_empirical_cov.csv` | empirical covariance vs C and a·C |
| `audits/parity/taskB_sample_vs_rsample.csv` | `sample` vs `rsample` laws |
| `audits/parity/taskB_posterior_stats.csv` | fixed-hyperparameter posterior audit |
| `audits/parity/taskB_derivative_stats.csv` | ‖E[∇f]‖ vs E[‖∇f‖] |
| `audits/parity/taskB_mll_boundedness.csv` | T1/T2/T3 marginal-likelihood grids |

No file outside `audits/` was created or modified. No `results/` directory was
touched. No branch, commit, or push was made.

## External repository

| item | value |
|---|---|
| remote | `https://github.com/UncertaintyQuantification/cell_segmentation.git` |
| local path | `/Users/zchan/eclipse-workspace/cell_segmentation_original` (sibling; read-only) |
| commit SHA | `44714c2e0be958fe796a8fd4bdbc220dae3c23dd` (short `44714c2`) |
| date / subject | 2025-09-04, "Update on Sept 4" |
| in this project's git index | **No** — verified absent |

## Installed package versions and paths

| package | version | path |
|---|---|---|
| python | 3.10.19 | `/Users/zchan/miniforge3/envs/gpytorch_arm/bin/python` |
| torch | 2.10.0 | `…/site-packages/torch` |
| gpytorch | 1.15.1 | `…/site-packages/gpytorch/__init__.py` |
| qpytorch | **0.2** | `…/site-packages/qpytorch/__init__.py` |
| R | installed; used for the parity test | system R |
| **EBImage** | **NOT INSTALLED** | — original watershed/distmap could not be executed |

## Main Task A conclusion

We have never run the paper's Fast-GP method. Our `gp_legacy` arm differs from it
in seven identified ways, and two of those differences (watershed seeding,
thresholding rule) have *measured* large effects on AP, while a third (isotropic
vs separable kernel) discards anisotropy that the fitted parameters show is real
(`beta1/beta2 ≈ 1.44`). The good news: `py_core/dim_2_lattice.py` reproduces the
original separable GP core to `max|diff| ≈ 1.1e-07`, so a faithful
`paper_fast_gp` is cheap to build — it needs wiring plus five configuration
corrections, not a rewrite. The arm should be renamed
`gp_isotropic_gpytorch_2025`; the name `gp_legacy` has been misread as "the
paper's method" throughout this project.

## Main Task B conclusion

`.variance` and `covariance_matrix` in qpytorch 0.2 return the **scale matrix C**,
not the statistical covariance. The true covariance is `a(q,d)·C` with
`a(q,d) = 2^(2/q)·Γ(d/2+2/q)/(d·Γ(d/2))` — exactly the factor the package
implements as `rescalor²` (agreement 9.3e-16), with `rescale` defaulting to
`False` everywhere. At fixed hyperparameters, exactly two quantities are
q-invariant — the predictive mean and C — and the round-4 conclusion
over-generalized from that to "no q channel exists". It does exist: quantiles,
tail probabilities, mean absolute deviation, kurtosis, `E[‖∇f‖]` and
`P(‖∇f‖>c)` are all q-dependent, by factors of 2–42×. Of the six audited claims,
1 is CONFIRMED and 5 are INCORRECT, including the retracted claim that the q<2
type-II MLE is unbounded — profiling the scale gives a unique interior optimum
`s* = r₀/d^(2/q)`, and a 1872-point grid finds an interior maximum for every q
with `noise*` never at the floor. Rounds 1–4 are nonetheless unaffected, because
they consumed only the posterior mean.

## Proposed next experiment

**Build and score a faithful `paper_fast_gp` on the two dev images**, not the QEP
mechanism — because every QEP question is a delta from a baseline, and the
current baseline is invalid. Cheapest first: extend the parity harness to n=256
and a whole-cell tile (`max|diff| < 1e-05`), wire `dim_2_lattice.py` in with the
five A4 corrections, restore outlier-tile handling and a tolerance-merging
watershed, compare reconstructions directly against the old arm, then score
AP@0.5. The single-layer QEP gradient-tail mechanism (`E[‖∇f‖|y]`,
`P(‖∇f‖>c|y)`, with `rescalor` divided out and batch size fixed) is the
experiment after that.

**Not started. Awaiting approval.**
