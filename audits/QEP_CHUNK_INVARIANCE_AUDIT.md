# Part B — QEP chunk-invariance / scaling audit

Date 2026-09-17. The **locally installed package is authoritative**. Nothing in
`site-packages` was modified.

Scripts: `audits/qep_chunk_invariance_check.py` (B1–B6). Raw log:
`audits/parity/chunk_run.log`. CSVs: `audits/parity/chunk_*.csv`.
All seeds fixed (`SEED = 20260917`).

---

## B1. Installed environment — **OBSERVED**

| item | value |
|---|---|
| python | 3.10.19, `/Users/zchan/miniforge3/envs/gpytorch_arm/bin/python` |
| torch | 2.10.0 |
| gpytorch | 1.15.1, `…/site-packages/gpytorch/__init__.py` |
| **qpytorch** | **0.2**, `…/site-packages/qpytorch/__init__.py` |

Inspected: `distributions/multivariate_qexponential.py`,
`distributions/multitask_multivariate_qexponential.py`,
`distributions/qexponential.py`, `distributions/power.py`,
`models/exact_qep.py`, `models/exact_prediction_strategies.py`,
`likelihoods/qexponential_likelihood.py`.

Key source facts, all **OBSERVED**:

```python
# multivariate_qexponential.py:131
@property
def rescalor(self):
    n = self.event_shape[0]
    return torch.exp((2./self.power*math.log(2) - math.log(n)
                      + torch.lgamma(n/2.+2./self.power) - math.lgamma(n/2.))/2.)

# multivariate_qexponential.py:216
def get_base_samples(self, sample_shape=torch.Size(), rescale=False):
    base_samples = _standard_normal(shape, ...)
    if self.power != 2:
        base_samples = torch.nn.functional.normalize(base_samples, dim=-1) \
            * Chi2(shape[-1]).sample(shape[:-1]+torch.Size([1]))**(1./self.power)
    if rescale:
        base_samples /= self.rescalor
```

`rsample` forwards `**kwargs` to `get_base_samples`; `sample` wraps `rsample`.
**`rescale` defaults to `False`** everywhere, including
`confidence_region(self, rescale=False)`. `variance` returns
`lazy_covariance_matrix.diagonal(...)` and `covariance_matrix` returns
`_covar.to_dense()` — neither applies `rescalor`.

In `models/exact_prediction_strategies.py`, `power` occurs only at lines 163,
237-238, 391, 430, 487-488, 659, 744, and at every one of them it is merely
**propagated onto the output distribution** — it never enters any linear
algebra. This is the structural reason the predictive mean and the scale matrix
are q-invariant.

`distributions/power.py::Power` is a full `Module` with constraint and prior
support, so `q` is itself an optimizable hyperparameter. **OBSERVED**, unused by
us so far.

---

## B2. Three notions of scale, formalized and verified

| # | object | what it is |
|---|---|---|
| 1 | **internal matrix C** | `post.covariance_matrix` = `post.lazy_covariance_matrix`. The kernel posterior *scale matrix*. |
| 2 | **package `.variance`** | `diag(C)`. A **scale parameter**, NOT a second moment for `q != 2`. No `a(q,d)` applied. |
| 3 | **true statistical covariance** | `a(q,d) · C` under the package's default sampling, with `a(q,d) = 2^(2/q) Γ(d/2+2/q) / (d Γ(d/2))` and `rescalor = sqrt(a(q,d))`. |

`a(2,d) = 2·Γ(d/2+1)/(d·Γ(d/2)) = 1`, so notions 2 and 3 coincide **only at
q = 2**.

Numerical verification at `x0 = 0.5` inside a batch of 40, hyperparameters
pinned (`lengthscale 0.15`, `outputscale 1`, `noise 0.01`) — **VERIFIED**:

| q | C[0,0] | `.variance[0]` | a(q,40)·C00 | emp var (default) | emp var (rescale=True) |
|---|---|---|---|---|---|
| 1.2 | 0.005188 | 0.005188 | 0.062352 | **0.061964** | **0.005155** |
| 1.5 | 0.005188 | 0.005188 | 0.017937 | **0.017830** | **0.005157** |
| 1.8 | 0.005188 | 0.005188 | 0.007840 | **0.007795** | **0.005158** |
| 2.0 | 0.005188 | 0.005188 | 0.005188 | 0.005166 | 0.005166 |

`.variance` tracks `C00` and is q-invariant; default samples track `a(q,40)·C00`;
`rescale=True` samples track `C00`. The package's naming does **not** equal the
mathematical second moment.

### Three conventions compared throughout this audit

- **A "default"** — `post.rsample(...)`, i.e. `rescale=False`.
- **B "rescale"** — `post.rsample(..., rescale=True)`, divides by `sqrt(a(q,d))`.
- **C "marginal"** — build the statistic from the **univariate** q-exponential
  with location `mu(x0)` and scale `sqrt(C(x0,x0))`, i.e. **d = 1 always**. This
  is the pointwise marginal of the process evaluated consistently; it is not a
  redefinition of the process, and by construction it cannot depend on which
  other points share the batch.

---

## B3. Chunk-SIZE invariance at a single fixed point

`x0 = 0.5` is element 0 of every batch; only the batch size changes.
Full table in `audits/parity/chunk_b3_batchsize.csv`. Excerpt at **q = 1.2**:

| batch d | mu(x0) | C00 | `.var` | a(q,d) | A var | A E\|f−m\| | A tail | A kurt | B var | B E\|f−m\| | B tail | B kurt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.095640 | 0.005188 | 0.005188 | 1.94 | 0.010182 | 0.064769 | 0.3153 | 7.310 | 0.005252 | 0.046518 | 0.2235 | 7.310 |
| 10 | 0.095640 | 0.005188 | 0.005188 | 5.15 | 0.026772 | 0.125784 | 0.6225 | 3.854 | 0.005202 | 0.055447 | 0.2875 | 3.854 |
| 100 | 0.095640 | 0.005188 | 0.005188 | 21.78 | 0.111603 | 0.265801 | 0.8269 | 3.074 | 0.005123 | 0.056950 | 0.3134 | 3.074 |
| 1000 | 0.095640 | 0.005188 | 0.005188 | 100.11 | 0.514133 | 0.571398 | 0.9192 | 3.022 | 0.005160 | 0.057370 | 0.3194 | 2.942 |
| **8192** | 0.095640 | 0.005188 | 0.005188 | **406.43** | **2.082842** | 1.160779 | 0.9616 | 2.870 | 0.005122 | 0.057549 | 0.3177 | **2.874** |
| **C (d=1)** | 0.095640 | 0.005188 | — | 1.94 | — | — | — | — | **0.010079** | **0.064736** | **0.3174** | **7.354** |

At q = 1.5 the same pattern holds with `A var` running 0.006994 → 0.103312 and
`B kurt` 4.710 → 2.870. At q = 2.0 everything is flat, as required.

**Findings — VERIFIED**

1. `mu(x0)` = 0.095640 and `C00` = `.variance[0]` = 0.005188 are **identical for
   every batch size and every q**. Both are chunk-invariant and q-invariant.
2. **Convention A is catastrophically chunk-dependent.** At q = 1.2 the sampled
   variance of the *same physical point* grows **205×** (0.010182 → 2.082842)
   purely from batch size, tracking `a(q,d)`.
3. **Convention B fixes the variance but not the shape.** `B var` is flat
   (0.00512–0.00525 across all d, ≈ C00). But `B E|f−m|` drifts +24%
   (0.046518 → 0.057549), `B tail` +42% (0.2235 → 0.3177), and **`B kurt`
   collapses 7.310 → 2.874**, i.e. toward the Gaussian value 3.
4. **Convention C reproduces the d = 1 values exactly** (0.010079 vs A's
   0.010182 at d=1; kurt 7.354 vs 7.310) and is constant by construction.

**Why B's shape drifts — the mechanism.** A 1-D marginal of a d-dimensional
elliptically-contoured vector with radius `R = χ²_d^(1/q)` converges to a
Gaussian as `d → ∞` (concentration of measure on the sphere). Normalizing the
variance cannot undo that: the *shape* of the marginal is d-dependent. So for
`q < 2` the q-signal in any shape statistic **vanishes as the chunk grows**.
`B kurt = 2.874` at d = 8192 and q = 1.2 is essentially Gaussian. **INFERRED
from the construction, VERIFIED by the kurtosis column.**

Direct practical consequence: with our `predict_chunk = 8192`, convention B
would have destroyed the q-signal even if we had used it.

---

## B4. Partition invariance over a fixed set of 200 locations

Same 200 query points every time; only the chunking changes; original ordering
restored before comparing pointwise. Reference = one call of 200.
`audits/parity/chunk_b4_partition.csv`. Excerpt at **q = 1.2**:

| partition | statistic | max\|diff\| vs 1x200 | rel diff | verdict |
|---|---|---|---|---|
| 2 x 100 | mu | 4.263e-14 | 0.0000 | INVARIANT |
| 2 x 100 | C00 | 4.663e-15 | 0.0000 | INVARIANT |
| 2 x 100 | A var | 1.037e-01 | 0.5744 | CHUNK-DEPENDENT |
| 2 x 100 | B var | 1.546e-04 | 0.0294 | mc-noise |
| 2 x 100 | C var | 8.913e-15 | 0.0000 | INVARIANT |
| 2 x 100 | C tail | 0.000e+00 | 0.0000 | INVARIANT |
| 20 x 10 | A var | 2.459e-01 | 1.3618 | CHUNK-DEPENDENT |
| 20 x 10 | A tail | 2.522e-01 | 0.2920 | CHUNK-DEPENDENT |
| 20 x 10 | B tail | 3.930e-02 | 0.1249 | CHUNK-DEPENDENT |
| 20 x 10 | C var | 6.793e-15 | 0.0000 | INVARIANT |
| **200 x 1** | A var | 2.737e-01 | **1.5154** | CHUNK-DEPENDENT |
| **200 x 1** | A tail | 5.529e-01 | **0.6400** | CHUNK-DEPENDENT |
| **200 x 1** | B absdev | 1.335e-02 | 0.2314 | CHUNK-DEPENDENT |
| **200 x 1** | B tail | 9.980e-02 | 0.3172 | CHUNK-DEPENDENT |
| **200 x 1** | C var / absdev / tail | ≤ 7.4e-15 / 2.2e-14 / **0.000e+00** | 0.0000 | INVARIANT |

(In the printed log the "conv" column shows `-` for the `mu`, `C00` and
convention-C rows because of a cosmetic label bug in the script's print
statement; the `statistic` column in the CSV is unambiguous. Rows are ordered
`mu, C00, A_*, B_*, C_*`.)

**Findings — VERIFIED**

- `mu` and `C00`: **partition-invariant to 4e-14 / 4e-15** (pure float noise).
- **Convention A: structurally chunk-dependent**, relative differences up to
  1.52. Not numerical — it is the `a(q, n_chunk)` factor.
- **Convention B: variance is invariant within Monte-Carlo error** (2.9–4.5%,
  consistent with 20 000 draws), but `absdev` and `tail` become genuinely
  chunk-dependent at fine partitions (23% and 32% at 200 x 1). Structural, same
  marginal-shape mechanism as B3.
- **Convention C: invariant to 1e-14, and `tail` identical to 0.000e+00.**
  Exactly as the construction requires.

Distinguishing numerical from structural: A's and B's deviations grow
**monotonically with the number of chunks** and match the predicted `a(q,d)` /
shape-convergence behaviour, so they are structural. C's deviations are at
float-epsilon and do not grow.

---

## B5. Derivative / boundary statistics under different partitions

Single-layer `ConstantMeanGrad` + `ScaleKernel(Matern52KernelGrad(ard_num_dims=2))`,
`MultitaskMultivariateQExponential` with 3 tasks (value + 2 partials), 10x10
lattice, hyperparameters pinned across q. `audits/parity/chunk_b5_derivative.csv`.

**q = 1.2**

| partition | event_shape | ‖E[∇f]‖ | E[‖∇f‖] A | E[‖∇f‖] B | P(>c) A | P(>c) B |
|---|---|---|---|---|---|---|
| 1 x 100 | (100, 3) | 0.038816 | 1.516611 | 0.328316 | 0.6000 | 0.0189 |
| 2 x 50 | (50, 3) | 0.038816 | 1.513965 | 0.409104 | 0.5991 | 0.0502 |
| 10 x 10 | (10, 3) | 0.038816 | 1.515146 | 0.669316 | 0.5969 | 0.2076 |
| 100 x 1 | (1, 3) | 0.038816 | 1.520962 | 1.092931 | 0.5984 | 0.4430 |

**q = 1.5 / 1.8 / 2.0** (E[‖∇f‖] A, first→last partition):
1.221474→1.229168 / 1.067771→1.076139 / 1.000973→1.001177.
`P(>c) A`: 0.5410→0.5460 / 0.4887→0.4944 / 0.4566→0.4547.

**This inverts the scalar case — VERIFIED.**

- `‖E[∇f]‖` = 0.038816 for **every q and every partition**: q-invariant AND
  chunk-invariant. It is a function of the posterior mean only.
- **Convention A is chunk-INVARIANT here** (1.5166 / 1.5140 / 1.5151 / 1.5210 —
  0.4% spread, Monte-Carlo scale) **and q-dependent** (1.5166 at q=1.2 →
  1.0010 at q=2.0). `P(>c) A` likewise: 0.600 → 0.457 across q, ~0.5% across
  partitions.
- **Convention B is chunk-DEPENDENT here** (0.3283 → 1.0929, a 3.3× drift):
  `rescale=True` *breaks* invariance in the multitask case.

**Why — the mechanism, now confirmed directly.** For multitask,
`base_sample_shape = event_shape = (n_points, n_tasks)`, so `get_base_samples`
normalizes along the **task** axis and draws the radius from
`Chi2(shape[-1]) = χ²_{n_tasks}`. Each spatial point therefore receives its
**own independent** `n_tasks`-dimensional elliptical radius. Consequently

```
Cov(multitask sample) = a(q, n_tasks) · C
```

with `n_tasks` fixed, independent of `n_points`. Verified against three
candidates over 15 configurations — **every** `q < 2` row matches `a(q, t)` and
none matches `a(q, n)` or `a(q, n·t)`:

| q | n | t | a(q,t) | a(q,n) | a(q,n·t) | empirical Cov/C | matches |
|---|---|---|---|---|---|---|---|
| 1.2 | 2 | 3 | 2.8003 | 2.3884 | 3.8921 | 2.7959 | **a(q,t)** |
| 1.2 | 50 | 3 | 2.8003 | 13.8724 | 28.4399 | 2.8013 | **a(q,t)** |
| 1.2 | 10 | 2 | 2.3884 | 5.1463 | 7.7729 | 2.3903 | **a(q,t)** |
| 1.2 | 10 | 5 | 3.5471 | 5.1463 | 13.8724 | 3.5476 | **a(q,t)** |
| 1.5 | 50 | 3 | 1.6345 | 3.7166 | 5.3290 | 1.6349 | **a(q,t)** |
| 1.5 | 10 | 5 | 1.8524 | 2.2471 | 3.7166 | 1.8540 | **a(q,t)** |

(At q=2 all three candidates equal 1, so that block is uninformative by design.)

Since `rescalor` uses `event_shape[0] = n_points`, it is the **correct**
normalizer for scalar `MultivariateQExponential` and the **wrong** one for
`MultitaskMultivariateQExponential`, where the right factor is
`sqrt(a(q, n_tasks))`. That single mismatch explains both halves of B5.

**Important caveat, stated plainly.** Convention A's chunk-invariance in the
multitask case comes from per-point *independent* radii. That is not the QEP
paper's process, which has a single global radius coupling all coordinates. So A
here is invariant, but it is invariant because the implementation has factorized
the elliptical structure across space. It is usable as a *feature*, but it should
not be described as "the q-EP joint posterior". **INFERRED** from the source plus
the covariance table.

---

## B6. What dimension enters the q scaling?

**Answer: the number of TEST QUERY POINTS IN THE CALL** for scalar models, and
**the number of TASKS** for the sampling radius in multitask models.
**OBSERVED + VERIFIED.**

| object | event_shape | `rescalor` n | radius dim | flat cov dim | consistent? |
|---|---|---|---|---|---|
| MVQEP, 5 test points | (5,) | 5 | 5 | 5 | **True** |
| MVQEP, 40 test points | (40,) | 40 | 40 | 40 | **True** |
| MVQEP, 200 test points | (200,) | 200 | 200 | 200 | **True** |
| MultitaskMVQEP n=5, t=3 | (5, 3) | 5 | **3** | 15 | **False** |
| MultitaskMVQEP n=40, t=3 | (40, 3) | 40 | **3** | 120 | **False** |
| MultitaskMVQEP n=100, t=3 | (100, 3) | 100 | **3** | 300 | **False** |

It is explicitly **not**: the process discretization dimension, the image pixel
count, or the flattened dimension of the covariance (except coincidentally in the
scalar case).

**Why the same physical `x0` changes when unrelated points are added.** For a
scalar `ExactQEP`, one prediction call constructs a single
`MultivariateQExponential` over all `d` requested points, and the default
sampler draws ONE global radius `χ²_d^(1/q)` shared by all of them. That radius's
second moment is `a(q,d)`, which grows with `d`. So `x0`'s sampled spread is a
property of the *joint object*, not of `x0`. Adding unrelated query points
enlarges `d`, enlarges the shared radius, and inflates `x0`. Conversely `mu(x0)`
and `C(x0,x0)` come from the prediction strategy's linear algebra, which never
sees `power` — hence they do not move (4e-14). **VERIFIED.**

**Comparison with the paper's process construction.** A q-EP defined as a
consistent stochastic process must satisfy Kolmogorov consistency: the
finite-dimensional marginal at a subset of points must not depend on which other
points were included. A family of d-dimensional elliptical laws with radius
`χ²_d^(1/q)` and a d-independent scale matrix does **not** satisfy this for
`q != 2`, because the 1-D marginal of such a vector drifts toward Gaussian as `d`
grows (see B3). The `rescalor` fixes the second moment but, as B3 shows, not the
shape. So the observed behaviour is a real tension between a finite-dimensional
elliptical construction and process consistency.

**Classification, as requested.** Two distinct things must be graded separately:

| item | classification | reasoning |
|---|---|---|
| Scalar MVQEP: `Cov = a(q, n_points) C` under default sampling, with `rescalor` correcting it | **mathematically intended finite-dimensional consistency**, combined with an **inappropriate pointwise interpretation** on our side | The d-dependence is intrinsic to a d-dimensional elliptical law with a shared radius; `rescalor` exists precisely to normalize it, and it does so correctly. The error was ours: reading a *joint* object's sampled spread as a *pointwise* uncertainty. |
| Multitask: `rescalor` uses `n_points` while the radius uses `n_tasks` | **implementation inconsistency** | The two dimensions are not the same object, and `rescale=True` consequently multiplies by a wrong factor, demonstrably breaking an otherwise chunk-invariant statistic (B5). |
| `rescale` defaulting to `False` while `.variance` omits `a(q,d)` | **API usage issue** | Defaults are mutually inconsistent, so naive use mixes notion 2 with notion 3. Documented behaviour, not a numerical error. |
| Whether a shape statistic can be made process-consistent for `q < 2` | **unresolved in general**; resolved for our purpose by convention C | Convention C is consistent by construction, but we have not proven it equals the paper's intended finite-dimensional marginal. |

Deliberately **not** labelled a bug overall: one component (multitask rescalor)
is a genuine inconsistency; the rest is correct code used incorrectly by us.

---

## B7. The acceptance criterion, applied

A statistic is usable for segmentation only if it satisfies **both**:

1. **q-sensitivity** — changes between `q < 2` and `q = 2` at fixed parameters.
2. **Partition invariance** — its value at a physical location is unchanged, to
   numerical tolerance, under different chunk partitions of the same query set.

| statistic | convention | q-sensitive? | partition-invariant? | **verdict** |
|---|---|---|---|---|
| predictive `mean` | — | **No** (0.00e+00 across q) | Yes (4e-14) | **FAIL (1)** |
| `C(x0,x0)` / `.variance` | — | **No** (q-invariant) | Yes (4e-15) | **FAIL (1)** |
| ‖E[∇f]‖ | — | **No** (0.038816 all q) | Yes (exact) | **FAIL (1)** |
| empirical variance | A default | Yes | **No** (205× over batch size) | **FAIL (2)** |
| E\|f−mu\|, tail, quantiles | A default | Yes | **No** (rel diff to 1.52) | **FAIL (2)** |
| empirical variance | B rescale | **No** (≈ C00 for all q) | Yes (within MC noise) | **FAIL (1)** |
| E\|f−mu\|, tail, kurtosis | B rescale | Yes at small d | **No** (kurt 7.31→2.87; tail +42%) | **FAIL (2)** |
| **E\|f−mu\|, tail, quantiles, kurtosis** | **C marginal (d=1)** | **Yes** (kurt 7.354 @ q=1.2 vs 2.998 @ q=2.0; E\|f−m\| 0.0647 vs 0.0575) | **Yes** (≤ 1e-14; tail exactly 0) | **PASS** |
| **E[‖∇f‖], P(‖∇f‖>c)** | **A default, MULTITASK** | **Yes** (1.5166 → 1.0010; 0.600 → 0.457) | **Yes** (0.4% spread, MC scale) | **PASS, with the B5 caveat** |
| E[‖∇f‖], P(‖∇f‖>c) | B rescale, multitask | Yes | **No** (0.328 → 1.093) | **FAIL (2)** |

**Two statistics pass.** So the audit does **not** terminate at "no statistic
qualifies", and equally it does not license the whole menu — 9 of 11 rows fail.

Ranking them:

1. **Convention C (d = 1 marginal) shape statistics** — strongest. Invariant to
   float epsilon by construction; depends only on `mu(x0)` and `C(x0,x0)`, both
   independently verified chunk-invariant; q-sensitivity is large (kurtosis
   7.35 → 3.00). Cost: it discards all spatial coupling in the uncertainty, so it
   is a *pointwise* feature computed from a *joint* posterior's diagonal.
2. **Multitask gradient statistics under convention A** — usable and directly
   boundary-relevant, but its invariance is a side effect of the implementation
   factorizing the radius per point, so it is not the paper's joint q-EP. Must
   be described accurately if used.

---

## B8. Required statements

**What dimension enters the q scaling.** `event_shape[0]`. For scalar
`MultivariateQExponential` that is the number of test query points in the single
prediction call; for `MultitaskMultivariateQExponential` it is the number of
points, while the *sampling radius* separately uses `shape[-1]` = the number of
tasks. Never the process discretization, the pixel count, or the flattened
covariance dimension.

**Why the previous chunk-size dependence occurred.** One prediction call builds
one joint elliptical distribution over all requested points with a single shared
radius `χ²_d^(1/q)`, whose second moment `a(q,d)` grows with `d`. Sampled spread
at `x0` is therefore a property of the batch, not of `x0`. Our `predict_chunk =
8192` made `d` a computational parameter, so any sampled uncertainty map would
have inherited it — including a different factor for an image's trailing partial
chunk. The reported `.variance` did not move because the prediction strategy's
linear algebra never touches `power`.

**Does `rescale=True` fix it?** **Partially, and it makes the multitask case
worse.** It fixes the *variance* in the scalar case (flat at ≈ C00 across d = 1
… 8192; partition rel diff 2.9–4.5%, i.e. Monte-Carlo only). It does **not** fix
shape statistics: kurtosis still collapses 7.310 → 2.874 and tail probability
drifts 42% across batch size, because the marginal of a d-dimensional elliptical
law tends to Gaussian regardless of normalization. And in the multitask case it
*introduces* chunk-dependence (E[‖∇f‖] 0.328 → 1.093) because `rescalor` uses the
wrong dimension there. Also note that once the variance is corrected to `C00`, it
is q-invariant, so `rescale=True` variance fails the q-sensitivity criterion
instead.

**Does paper-consistent scaling fix it?** **Yes — convention C does.** Building
the statistic from the univariate q-exponential marginal at `(mu(x0),
sqrt(C(x0,x0)))` gives partition invariance to `≤ 1e-14` (tail probability
identically 0.000e+00 across all five partitions) while retaining full
q-sensitivity. It is invariant because it depends on nothing but two
chunk-invariant quantities.

**Statistics SAFE for spatial use**

- `mu(x)` and `C(x,x)` — safe, but q-invariant, so useless for a q comparison.
- Convention C shape statistics: `E|f−mu|`, posterior quantiles,
  `P(|f−mu| > c)`, kurtosis — **safe and q-sensitive**.
- Multitask convention A: `E[‖∇f‖]`, `P(‖∇f‖ > c)` — **safe and q-sensitive**,
  with the caveat that the underlying construction has per-point independent
  radii.
- `‖E[∇f]‖` — safe, q-invariant.

**Statistics NOT safe for spatial use**

- Anything sampled from a scalar `MultivariateQExponential` with `rescale=False`:
  variance, `E|f−mu|`, quantiles, tail probabilities. Up to 205× batch-size
  artifacts.
- Scalar `rescale=True` *shape* statistics (tail, kurtosis, `E|f−mu|` at fine
  partitions).
- Multitask `rescale=True` gradient statistics.
- `.variance` interpreted as an uncertainty for `q != 2` — it is a scale
  parameter.
- `confidence_region()` at its default `rescale=False`.

**Is a gradient-tail segmentation experiment scientifically valid?**

**Yes, conditionally — and the conditions are now concrete rather than
aspirational.** It is valid if and only if:

1. the statistic is either convention C, or multitask convention A **without**
   `rescale=True`;
2. the predictive batch/chunk partition is fixed and reported, and partition
   invariance is re-asserted on the actual image sizes used (verified here only
   up to d = 8192 in 1-D and 100 points in 2-D);
3. the q = 2 control uses the identical architecture and machinery, differing
   only in `q` — satisfied, since `power = 2` skips the `χ²` branch entirely;
4. the write-up distinguishes `‖E[∇f]‖` (q-invariant, and therefore not the
   mechanism) from `E[‖∇f‖]` (q-dependent);
5. if multitask convention A is used, the report states that its chunk
   invariance follows from per-point independent radii and is therefore not the
   paper's joint q-EP posterior.

It would be **invalid** with default scalar sampling, with `rescale=True`
gradients, or with `.variance` used as an uncertainty — those are measuring chunk
size or measuring nothing q-dependent at all.
