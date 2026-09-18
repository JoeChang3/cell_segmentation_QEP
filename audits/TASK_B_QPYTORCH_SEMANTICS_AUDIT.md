# Task B — What q actually does in the locally installed qpytorch 0.2

Audit date 2026-09-17. The **locally installed package is treated as
authoritative**; where it disagrees with any paper, the installed behavior is
what our experiments measured and therefore what our past conclusions were
about.

Scripts written for this audit (new files):

- `audits/qep_local_semantics_check.py` — B2, B3, B3b, B4, B4b, B5
- `audits/qep_mll_boundedness_check.py` — B6 claim 6

CSV outputs in `audits/parity/taskB_*.csv`. All seeds fixed (`SEED = 20260917`).
Nothing in `site-packages` was modified.

---

## B1. Installed environment — **OBSERVED**

| item | value |
|---|---|
| python | 3.10.19, `/Users/zchan/miniforge3/envs/gpytorch_arm/bin/python` |
| torch | 2.10.0 |
| gpytorch | 1.15.1, `…/site-packages/gpytorch/__init__.py` |
| **qpytorch** | **0.2**, `…/site-packages/qpytorch/__init__.py` |

Files read: `distributions/multivariate_qexponential.py`,
`distributions/qexponential.py`, `models/exact_qep.py`,
`likelihoods/qexponential_likelihood.py`,
`mlls/exact_marginal_log_likelihood.py`, and gpytorch's
`exact_prediction_strategies.py` (qpytorch reuses it).

`models/exact_qep.py` is structurally identical to `gpytorch/models/exact_gp.py`
— `diff` shows only names and docstrings. **OBSERVED.**

---

## B2. Scale matrix C vs true covariance — the central distinction

### B2.1 The package's own inflation factor — **OBSERVED**

`distributions/multivariate_qexponential.py:131`:

```python
@property
def rescalor(self) -> Tensor:
    n = self.event_shape[0]
    return torch.exp((2./self.power*math.log(2) - math.log(n)
                      + torch.lgamma(n/2.+2./self.power) - math.lgamma(n/2.))/2.)
```

Algebraically this is exactly `sqrt(a(q,n))` for

```
a(q,d) = 2^(2/q) · Γ(d/2 + 2/q) / ( d · Γ(d/2) )
```

which is the factor hypothesized in the task brief. **The brief's formula is
correct and needs no correction.** Checked independently against a from-scratch
`lgamma` implementation over q ∈ {1.0,1.2,1.5,1.8,2.0,3.0} × d ∈ {1,2,5,20}:
max relative difference **9.33e-16**, i.e. floating-point exact. **VERIFIED**
(`taskB_rescalor_formula.csv`).

Sanity check of the definition: `a(2,d) = 2·Γ(d/2+1)/(d·Γ(d/2)) = 2·(d/2)/d = 1`,
so the Gaussian case is unscaled, as it must be. Observed: `rescalor = 1.000000000`
at q=2 for every d. **VERIFIED.**

The univariate `distributions/qexponential.py:63` uses the d=1 specialization
`sqrt(2^(2/q)·Γ(1/2+2/q)/√π)`, consistent since `Γ(1/2) = √π`. **OBSERVED.**

### B2.2 What `.variance` returns — the answer is A, **diag(C)**

`MultivariateQExponential.variance` returns
`self.lazy_covariance_matrix.diagonal(...)` (clamped to `settings.min_variance`),
and `covariance_matrix` returns `self._covar.to_dense()`. Neither applies
`rescalor`. **OBSERVED.**

Tested rather than inferred, at d=5 with a fixed PD C:

| q | `.variance[0]` | `C[0,0]` | `a(q,5)·C[0,0]` | verdict |
|---|---|---|---|---|
| 1.0 | 2.676634 | 2.676634 | 18.736438 | = diag(C) |
| 1.2 | 2.676634 | 2.676634 | 9.494176 | = diag(C) |
| 1.5 | 2.676634 | 2.676634 | 4.958222 | = diag(C) |
| 1.8 | 2.676634 | 2.676634 | 3.274497 | = diag(C) |
| 2.0 | 2.676634 | 2.676634 | 2.676634 | = diag(C) |
| 3.0 | 2.676634 | 2.676634 | 1.499080 | = diag(C) |

**Conclusion: `.variance` is option A, `diag(C)`. It is a SCALE parameter, not a
statistical variance, for every q ≠ 2.** **VERIFIED.**

The same holds for the univariate class: `QExponential.variance` returns
`self.stddev.pow(2)` = scale², ignoring `a(q,1)`. **OBSERVED.**

### B2.3 Where `rescale` appears, and its default — **OBSERVED**

```python
def get_base_samples(self, sample_shape=torch.Size(), rescale=False):
    base_samples = _standard_normal(shape, ...)
    if self.power != 2:
        base_samples = torch.nn.functional.normalize(base_samples, dim=-1) \
            * Chi2(shape[-1]).sample(shape[:-1]+torch.Size([1])).to(...)**(1./self.power)
    if rescale:
        base_samples /= self.rescalor
```

This is the elliptical construction: a uniform direction on the unit sphere times
radius `R = χ²_d^(1/q)`. `rsample` forwards `**kwargs` to `get_base_samples`, and
`sample` wraps `rsample`. **`rescale` defaults to `False` everywhere** —
including `confidence_region(self, rescale=False)`.

So by default the package's samples carry the `a(q,d)` inflation while its
reported `.variance` does not. **The two are inconsistent with each other by
construction, and we used the defaults.**

---

## B3. Empirical covariance test — **VERIFIED**

200,000 samples per cell, seed 20260917, zero mean, `C` a fixed PD matrix.
Reported values are relative Frobenius errors. Full table in
`taskB_empirical_cov.csv`.

| q | d | a(q,d) | rescale=False vs C | rescale=False vs a·C | rescale=True vs C | rescale=True vs a·C |
|---|---|---|---|---|---|---|
| 1.0 | 1 | 3.0000 | 1.9889 | **0.0037** | **0.0037** | 0.6679 |
| 1.0 | 2 | 4.0000 | 3.0272 | **0.0074** | **0.0074** | 0.7483 |
| 1.0 | 5 | 7.0000 | 6.0171 | **0.0098** | **0.0098** | 0.8568 |
| 1.0 | 20 | 22.0000 | 21.0317 | **0.0094** | **0.0094** | 0.9545 |
| 1.2 | 1 | 1.9387 | 0.9320 | **0.0035** | **0.0035** | 0.4860 |
| 1.2 | 20 | 7.7729 | 6.7821 | **0.0089** | **0.0089** | 0.8712 |
| 1.5 | 1 | 1.3373 | 0.3333 | **0.0030** | **0.0030** | 0.2545 |
| 1.5 | 5 | 1.8524 | 0.8547 | **0.0072** | **0.0072** | 0.4595 |
| 1.5 | 20 | 2.7737 | 1.7764 | **0.0085** | **0.0085** | 0.6392 |
| 1.8 | 20 | 1.4034 | 0.4046 | **0.0084** | **0.0084** | 0.2870 |
| 2.0 | 1 | 1.0000 | 0.0042 | 0.0042 | 0.0042 | 0.0042 |
| 2.0 | 20 | 1.0000 | 0.0084 | 0.0084 | 0.0084 | 0.0084 |
| 3.0 | 5 | 0.5601 | 0.4399 | **0.0051** | **0.0051** | 0.7859 |
| 3.0 | 20 | 0.3644 | 0.6355 | **0.0081** | **0.0081** | 1.7459 |

Reading:

- With the **default** `rescale=False`, the empirical covariance matches
  **a(q,d)·C** to 0.3–1.0% (pure Monte-Carlo error at this sample size), and
  disagrees with `C` by up to **2100%** (q=1, d=20).
- With `rescale=True` it matches **C**.
- At q=2 both columns coincide, as required.
- Note q=3 gives a(q,d) < 1: the inflation is a *deflation* for q > 2.

**So: `Cov(X) = a(q,d)·C` is confirmed for the installed package's default
sampling path, and the brief's formula for `a(q,d)` is exactly right.**

### B3b. Are `sample` and `rsample` the same law? — **VERIFIED, and this corrected a source-reading error of mine**

The univariate class writes them differently:

```python
qexponential.py:90   sample : eps = Chi2(1).sample(shape)**(1./power) * _standard_normal(...).sign()
qexponential.py:97   rsample: eps = _standard_normal(...);  if power != 2: eps = eps.abs()**(2./power-1) * eps
```

Reading the source, I expected two different distributions. **They are the same
law.** `|eps_rsample| = |z|^(2/q)` and `(z²)^(1/q) = χ²₁^(1/q)`, with `sign(z)` in
both. The test confirms it:

| q | sd(`sample`) | sd(`rsample`) | ratio | kurt(`sample`) | kurt(`rsample`) | sqrt(a(q,1)) |
|---|---|---|---|---|---|---|
| 1.0 | 1.733543 | 1.739656 | 0.9965 | 11.6169 | 11.5525 | 1.732051 |
| 1.5 | 1.156429 | 1.159803 | 0.9971 | 4.7129 | 4.6923 | 1.156417 |
| 2.0 | 0.999733 | 1.002103 | 0.9976 | 3.0084 | 3.0011 | 1.000000 |
| 3.0 | 0.911141 | 0.912661 | 0.9983 | 1.9404 | 1.9388 | 0.911516 |

Both track `sqrt(a(q,1))`; the ~0.3% gap is different RNG consumption order, not
a distributional difference. Recorded because it is exactly the kind of claim the
brief warned against inferring from reading.

---

## B4. Fixed-hyperparameter ExactQEP posterior: which statistics are q-dependent?

Setup: n=30 1-D training points, `ConstantMean` fixed at 0,
`ScaleKernel(MaternKernel(nu=2.5))` with lengthscale 0.15 and outputscale 1,
noise 0.01 — **all hyperparameters pinned identically across q**, so q is the
only thing that varies. 40 test points, 40,000 posterior draws, default
`rescale=False`. Differences are against q=2. **VERIFIED**
(`taskB_posterior_stats.csv`).

| q | mean max\|diff\| | covariance_matrix max\|diff\| | `.variance` max\|diff\| | emp.var / `.variance` | a(q,40) | mean q₀.₀₅ | mean q₀.₉₅ | E\|f−m\| | P(\|f−m\|>2·sd) | kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.9990 | 1.0000 | −0.1390 | 0.1015 | 0.0583 | 0.0455 | 3.003 |
| 1.0 | **0.00e+00** | **0.00e+00** | **0.00e+00** | **41.8941** | 42.0000 | −0.7947 | 0.7564 | 0.3707 | 0.7437 | 3.447 |
| 1.2 | **0.00e+00** | **0.00e+00** | **0.00e+00** | 11.9903 | 12.0192 | −0.4345 | 0.3968 | 0.1998 | 0.5513 | 3.261 |
| 1.5 | **0.00e+00** | **0.00e+00** | **0.00e+00** | 3.4500 | 3.4576 | −0.2420 | 0.2045 | 0.1078 | 0.2773 | 3.113 |
| 1.8 | **0.00e+00** | **0.00e+00** | **0.00e+00** | 1.5082 | 1.5112 | −0.1664 | 0.1290 | 0.0715 | 0.1034 | 3.035 |
| 3.0 | **0.00e+00** | **0.00e+00** | **0.00e+00** | 0.2903 | 0.2908 | −0.0835 | 0.0463 | 0.0315 | 0.0001 | 2.923 |

### B4 findings

**q-INVARIANT (bit-for-bit, `0.00e+00`):**

- predictive `mean` — the conditional mean of an elliptically-contoured
  distribution is the same linear smoother as the Gaussian one, so this is
  expected and is now confirmed for the installed code.
- `covariance_matrix`, i.e. the internal scale matrix **C**.
- `.variance`, i.e. `diag(C)`.

**q-DEPENDENT (from the samples):**

- empirical predictive covariance: ratio to `.variance` tracks **a(q,40)** to
  three digits (41.89 vs 42.00; 3.450 vs 3.458). So the *true* posterior
  covariance is `a(q,d)·C` and **is** q-dependent.
- posterior quantiles: the 90% interval width goes from 0.240 at q=2 to 1.551 at
  q=1 — a 6.5× change.
- E|f − mean|: 0.0583 → 0.3707.
- tail probability P(|f − mean| > 2·sd(C)): 0.0455 → 0.7437.
- kurtosis: 3.003 → 3.447 at q=1, 2.923 at q=3. Not merely a rescaling — the
  *shape* changes too, so `a(q,d)·C` does not capture everything.

### B4b. The inflation factor uses the TEST-BATCH dimension — **VERIFIED, and this is an implementation artifact with direct consequences for our pipeline**

`rescalor` reads `n = self.event_shape[0]`, so `d` is the number of points in the
**single joint predictive call**, not a property of the point. Probing the *same*
test point x=0.5 at q=1.5 inside batches of different size:

| n_test | `.variance[0]` | empirical var[0] | ratio | a(1.5, n_test) |
|---|---|---|---|---|
| 2 | 0.005188 | 0.007795 | 1.5026 | 1.5001 |
| 5 | 0.005188 | 0.009692 | 1.8682 | 1.8524 |
| 10 | 0.005188 | 0.011690 | 2.2534 | 2.2471 |
| 40 | 0.005188 | 0.017830 | 3.4370 | 3.4576 |
| 200 | 0.005188 | 0.030009 | 5.7846 | 5.8610 |

The reported `.variance` is constant; the sampled variance of the identical point
grows without bound with batch size, tracking `a(1.5, n_test)` throughout.

**Consequence for this project:** `py_core/segmentation_pipeline.py` predicts with
`predict_chunk = 8192`. Under q ≠ 2, any sample-based uncertainty quantity we
compute therefore depends on the chunk size — a purely computational parameter.
Since `a(q,d)` grows roughly like `d^(2/q − 1)` for large d, at q=1.5 and
d=8192 the inflation is ~90×, and the last (partial) chunk of an image would get
a *different* inflation than the full chunks. Any future uncertainty-driven
segmentation rule must either pass `rescale=True` or divide by
`rescalor` explicitly, and must fix the batch size. **This did not affect rounds
1–4, which used only the posterior mean.** **INFERRED** from the verified
batch-size scaling.

---

## B5. Derivative and boundary quantities

Model: `ConstantMeanGrad` + `ScaleKernel(Matern52KernelGrad(ard_num_dims=2))`,
`MultitaskMultivariateQExponential` with 3 tasks (value + 2 partials) — the
Diff_QEP architecture — on a 10×10 lattice with a disc target, hyperparameters
pinned across q, 4000 draws. Threshold `c` fixed at the q=2 value of E‖∇f‖ so the
probability column discriminates. **VERIFIED** (`taskB_derivative_stats.csv`).

| q | ‖E[∇f]‖ (mean over grid) | E[‖∇f‖] (mean over grid) | ratio | P(‖∇f‖ > c) | c |
|---|---|---|---|---|---|
| 2.0 | 0.038816 | 1.001079 | 25.791 | 0.4560 | 1.0011 |
| 1.0 | **0.038816** | 1.913213 | 49.290 | 0.6449 | 1.0011 |
| 1.2 | **0.038816** | 1.515827 | 39.052 | 0.5993 | 1.0011 |
| 1.5 | **0.038816** | 1.220826 | 31.452 | 0.5396 | 1.0011 |
| 1.8 | **0.038816** | 1.067233 | 27.495 | 0.4878 | 1.0011 |
| 3.0 | **0.038816** | 0.836843 | 21.559 | 0.3225 | 1.0011 |

Note also that ‖E[∇f]‖ and E[‖∇f‖] differ by a factor of 21–49 here. They are
not interchangeable quantities, and the gap is itself q-dependent.

### B5 classification

| quantity | classification | basis |
|---|---|---|
| ‖E[∇f \| y]‖ (norm of the posterior mean gradient) | **q-INVARIANT** | VERIFIED, `0.038816` identical for all six q. It is a deterministic function of the posterior mean, which B4 showed is q-invariant. |
| E[‖∇f‖ \| y] (posterior mean of the gradient norm) | **q-DEPENDENT** | VERIFIED, monotone decreasing in q, 1.913 → 0.837 |
| P(‖∇f‖ > c \| y) | **q-DEPENDENT** | VERIFIED, 0.6449 → 0.3225 |
| P(\|∂f/∂x\| > c \| y) | **q-DEPENDENT** | INFERRED — a marginal of the same sampled law as the row above; same mechanism, not separately tabulated |
| posterior edge probability (any sample-based thresholding of gradient magnitude) | **q-DEPENDENT** | INFERRED from the two VERIFIED rows above; this is the same functional family |
| sample-based boundary scores generally | **IMPLEMENTATION-DEPENDENT** | VERIFIED via B4b: they inherit the `a(q, n_test)` factor, so they depend on predictive batch size as well as on q |
| any decision rule that is a deterministic function of the posterior mean alone (including every mask produced in rounds 1–4) | **q-INVARIANT** | VERIFIED |

**Important qualification, as the brief requires:** the fact that `mean`,
`.variance` and `C` are all equal across q does **not** license "the posterior is
q-invariant". The posterior *distribution* is demonstrably q-dependent — its
covariance is `a(q,d)·C`, and its kurtosis changes independently of any scaling.
What is q-invariant is precisely (i) the conditional mean and (ii) the internal
scale matrix. Everything computed by sampling is not.

---

## B6. Re-audit of the six previous claims

The six claims are quoted verbatim from the audit brief. Sources audited:
`experiments/simulated/qep_power_semantics_check.py`,
`results/real_cellseg_round4_mechanism_20260916/mechanism_cards.md`,
`results/real_cellseg_round4_mechanism_20260916/source_notes.md`. **None of these
files was edited**, per the instruction not to modify existing results.

| # | previous claim (verbatim) | label | basis |
|---|---|---|---|
| 1 | "Exact QEP posterior mean is q-invariant at fixed hyperparameters." | **CONFIRMED** | `0.00e+00` max abs difference for every q ∈ {1.0,1.2,1.5,1.8,3.0} vs q=2, reproduced this session (B4). Mechanism now located: in `models/exact_prediction_strategies.py`, `power` appears **only** at lines 163, 237-238, 391, 430, 487-488, 659, 744, and at every one of them it is merely *propagated onto the output distribution* — it never enters any linear algebra. So the invariance is structural, not coincidental. **OBSERVED + VERIFIED.** |
| 2 | "Exact QEP posterior variance is q-invariant." | **INCORRECT** | The quantity previously measured was `.variance`, which B2.2 establishes is `diag(C)` — the **scale** matrix diagonal, not a variance. The true predictive covariance is `a(q,d)·C`; the empirical-to-reported ratio runs from 0.2903 (q=3) to **41.89** (q=1) at d=40, matching `a(q,40)` to three digits. The *scale matrix* is q-invariant; the *variance* is not. **VERIFIED.** |
| 3 | "q enters only `log_prob`." | **INCORRECT** | q has at least four independent entry points in the installed package: (a) `MultivariateQExponential.log_prob` via `inv_quad**(power/2)`; (b) `QExponentialLikelihood.expected_log_prob` via `r**(q/2)`; (c) the **`rescalor` property**, which is not part of any density evaluation; (d) **`get_base_samples`**, the *sampling* path, where `power` selects the `normalize(z)·χ²_d^(1/q)` radial construction. (d) is the decisive one: q changes what is **drawn**, not only what is **scored**. Additionally `distributions/power.py::Power` is a full `Module` with constraint and prior support, so q is itself an optimizable hyperparameter. The narrower statement "q does not enter the exact predictive linear algebra" **is** CONFIRMED (see claim 1). **OBSERVED.** |
| 4 | "No single-layer q-dependent boundary statistic exists." | **INCORRECT** | Two explicit counterexamples, measured on a single-layer `Matern52KernelGrad` model with hyperparameters pinned across q (B5): `E[‖∇f‖ \| y]` moves 1.913 → 0.837 (2.3×) and `P(‖∇f‖ > c \| y)` moves 0.6449 → 0.3225 (2.0×) over q ∈ [1,3]. Both are single-layer, both are boundary statistics, both are monotone in q. **VERIFIED.** |
| 5 | "Only Deep QEP provides a live q-dependent channel." | **INCORRECT** | Falls with claim 4. The exact single-layer model has a live q channel through the predictive *distribution* — any sampled, quantile-based or tail-based functional. What is true, and remains CONFIRMED, is the *weaker* statement that Deep QEP is the only audited architecture whose **posterior mean** is q-dependent (max abs diff 3.893e-04 at q=1.5, 6.134e-04 at q=1.2, exactly 0 at q=2). The error was equating "mean is q-invariant" with "no channel exists". |
| 6 | "Type-II MLE for q<2 is unbounded and necessarily drives noise to zero." | **INCORRECT** | See below. |

**Consequential correction to `mechanism_cards.md` / `source_notes.md`.** The
round-4 derived conclusion — *"because every predictive moment of the exact and
derivative-augmented QEP is q-independent at fixed hyperparameters, no
q-specific decision path into the final masks exists through those models"* —
is **UNSUPPORTED** as written. Its premise is false: not every predictive moment
is q-independent, only the first moment and the scale matrix. The *conclusion*
happens to hold for the masks we actually produced, but for a different and
narrower reason, given in "Net effect" below.

### B6, claim 6 in detail

The brief asked to distinguish (i) a density singularity as the residual → 0
from (ii) a proof that the hyperparameter marginal likelihood is globally
unbounded. Neither (i) nor (ii) survives.

**Analytic part.** From `log_prob`,
`L = −0.5[r^(q/2) + logdet C + d·log2π] + 0.5·d·(q/2−1)·log r + log(q/2)`.
Writing `C = s·C₀` gives `r = r₀/s`, `logdet = d·log s + ldet₀`, so the
s-dependence collects to

```
L(s) = −0.5·r₀^(q/2)·s^(−q/2) − (d·q/4)·log s + const
dL/ds = (q/4s)·[ r₀^(q/2)·s^(−q/2) − d ]  =  0   at   s* = r₀ / d^(2/q)
```

a single interior stationary point, with `L → −∞` as `s → 0` and as `s → ∞`, for
**every** q > 0. The objective does not run away along the scale direction.

**Numerical part** (`audits/qep_mll_boundedness_check.py`, **VERIFIED**):

*T1 — noise → 0 at fixed kernel hyperparameters (lengthscale 0.15, outputscale 1):*

| noise | q=1 | q=1.2 | q=1.5 | q=1.8 | q=2 |
|---|---|---|---|---|---|
| 1e-02 | −0.2518 | −0.1162 | 0.0720 | 0.2341 | 0.3210 |
| 1e-03 | −0.1338 | 0.0284 | **0.1925** | 0.1836 | 0.0054 |
| 1e-05 | −0.2172 | −0.1048 | −0.2454 | −1.2068 | −2.8217 |
| 1e-08 | −0.2243 | −0.1162 | −0.2785 | −1.3084 | −3.0308 |
| 1e-10 | −0.2243 | −0.1162 | −0.2785 | −1.3085 | −3.0310 |

Every q < 2 column **rises, turns over, then saturates** at a finite value. No
divergence.

*T2 — noise → 0 with the outputscale profiled at each noise* (the honest test,
since a shrinking nugget is otherwise silently compensated by a shrinking scale).
Same shape: q=1 peaks at 0.4101 (noise 1e-4) then falls to −0.1159 at noise 1e-8;
q=1.5 peaks at 0.3813 then falls to −0.1056. The profiled argmax outputscale is
never at a grid edge.

*T3 — coarse global grid, 12 × 13 × 12 = 1872 evaluations per q over
(lengthscale, outputscale, noise) spanning 1e-3…1e2, 1e-4…1e4, 1e-10…1e1:*

| q | max mll | ls* | outputscale* | noise* | noise* at grid floor? | optimum interior? |
|---|---|---|---|---|---|---|
| 1.0 | 0.4564 | 5.337e-01 | 4.642e-02 | 1.000e-04 | No | **Yes** |
| 1.2 | 0.4971 | 5.337e-01 | 2.154e-01 | 1.000e-03 | No | **Yes** |
| 1.5 | 0.3736 | 1.874e-01 | 4.642e-02 | 1.000e-03 | No | **Yes** |
| 1.8 | 0.4583 | 5.337e-01 | 1.000e+00 | 1.000e-02 | No | **Yes** |
| 2.0 | 0.4898 | 5.337e-01 | 4.642e+00 | 1.000e-02 | No | **Yes** |

For every q the maximum is **interior**, `noise*` is never at the grid floor, and
the attained maxima are comparable across q (0.37–0.50). The q < 2 optimum is not
at noise → 0 and the objective is not unbounded above on this grid.

**Verdict: claim 6 is INCORRECT** for the installed implementation on this
problem. It is retracted. Two honest qualifications:

- This is one dataset on a finite grid — it demonstrates the claim is false as
  *stated* (unboundedness is a universal assertion, so a single bounded interior
  optimum refutes it), but it is not a proof of boundedness for all data and all
  kernels.
- Incidental **OBSERVED** finding that reinforces the practical conclusion:
  `QExponentialLikelihood` defaults to `noise_constraint = GreaterThan(1.000E-04)`,
  so the installed package *floors the nugget at 1e-4 anyway*. The audit had to
  override this constraint to reach the interpolation limit at all. Any previously
  observed "noise → 0" behavior in our fits was bounded below by that floor, not
  by a divergence.

### Net effect of B6 on the round-4 conclusions

What **survives**: every mask produced in rounds 1–4 is a deterministic function
of the posterior **mean** alone, and the mean is VERIFIED q-invariant at fixed
hyperparameters. So for those specific experiments, the only q channel really was
hyperparameter selection, and a Gaussian model given those hyperparameters really
does reproduce the masks. The round-1–4 *numerical results and their
interpretation stand.*

What **does not survive**: the general claim that no q channel exists through the
exact single-layer model. One does — through the predictive distribution
(covariance `a(q,d)·C`, quantiles, tail probabilities, `E‖∇f‖`). Rounds 1–4 never
consumed it, which is why they were unaffected.

The distinction matters for what comes next: it reopens single-layer QEP as a
candidate mechanism, which round 4 had closed off. See
`audits/A_B_SYNTHESIS.md` question 6.

---

## B7. Task B conclusions

1. `a(q,d) = 2^(2/q)·Γ(d/2+2/q)/(d·Γ(d/2))` is **exactly** the factor the
   installed package implements as `rescalor²`. The brief's formula is correct.
   **VERIFIED.**
2. `.variance` and `covariance_matrix` return the **scale** matrix C and its
   diagonal, not the statistical covariance. `Cov = a(q,d)·C`. **VERIFIED.**
3. `rescale` defaults to `False` on `get_base_samples`, `rsample`, `sample` and
   `confidence_region`, so the default sampling path and the default `.variance`
   are mutually inconsistent for q ≠ 2. **OBSERVED.**
4. At fixed hyperparameters, exactly two things are q-invariant: the predictive
   mean and the scale matrix. Quantiles, mean absolute deviation, tail
   probabilities, kurtosis, empirical covariance, E‖∇f‖ and P(‖∇f‖>c) are all
   q-dependent. **VERIFIED.**
5. The inflation uses the **predictive batch dimension**, so sample-based
   uncertainty is a function of chunk size. Must be controlled before any
   uncertainty-driven rule is built. **VERIFIED.**
6. Of the six previously stated claims, **one is CONFIRMED** (claim 1, posterior
   mean q-invariance — and its structural mechanism is now located) and **five
   are INCORRECT** (claims 2, 3, 4, 5, 6). All five share a single root cause:
   treating `.variance` as a statistical variance and inferring "no q channel"
   from the invariance of the first two *reported* moments.
7. The round-1–4 experimental results are unaffected, because they consumed only
   the posterior mean. The *scope* of their conclusion must be narrowed.

### Implications specifically for our cell-segmentation pipeline

1. **No past result is invalidated.** Rounds 1–4 used only reconstructed
   posterior means, which are q-invariant, so the observed Raw ≈ GP ≈ QEP
   behavior is real and correctly explained.
2. **Single-layer QEP is back on the table** as a mechanism, but only through
   distributional functionals — never through the mean.
3. **Any such rule must control two nuisance factors before it means anything:**
   (a) pass `rescale=True` or divide by `rescalor`, since `.variance` is a scale;
   (b) fix the predictive batch size, since the inflation is `a(q, n_test)`
   (B4b). With `predict_chunk = 8192`, the inflation at q=1.5 is ~90× and the
   trailing partial chunk of an image would receive a *different* inflation from
   the full chunks — a pure artifact that would look like spatial structure.
4. **`a(q,d)·C` is not the whole story.** Kurtosis changes independently of any
   scaling (3.003 → 3.447), so a q-dependent rule cannot be reproduced by a
   Gaussian model with a rescaled kernel. That is what makes a genuine test
   possible: the Gaussian control cannot trivially imitate it.
5. **`q` is itself learnable** (`distributions/power.py::Power` is a `Module`
   with constraint and prior support), which was not exploited in any round.
