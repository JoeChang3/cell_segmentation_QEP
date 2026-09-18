# Round 9 — q-dependent local gradient-posterior statistics at merge interfaces

2026-09-18. Two development images only. Rounds 1–8 preserved and unmodified.
No segmentation stage was touched; AP is not reported as an outcome.

## Terminology (held throughout, per the preregistration)

The tested object is **q-dependent local gradient-posterior statistics under the
current QePyTorch multitask construction**. It is **not** called "the joint QEP
posterior boundary probability". The Task-B audit showed the multitask sampler
draws a per-point radius from `Chi2(n_tasks)`, i.e. it **factorizes the
elliptical structure across space**, so this construction is *not established* to
be identical to the paper's full spatial joint QEP. Every claim below is about
the installed implementation's local construction.

Design was frozen in `preregistered_design.json` **before any result was
computed**, including the success criteria and the stop rule.

## Model (Step 2)

Exact regression on **value observations only**. The gradient posterior is
obtained analytically from the gradient-augmented kernel's cross-blocks
(`qpytorch.kernels.Matern52KernelGrad`; layout verified interleaved
`[f, ∂f/∂x₁, ∂f/∂x₂]`, gradient prior variance `5/(3l²)` = 18.5185 at l=0.3,
matching the analytic value). **No zero-valued gradient pseudo-observations are
fed as data** — the concern raised in earlier rounds is avoided outright.

Per query point: a 3-dimensional `MultivariateQExponential` over
`(f, ∂f/∂x, ∂f/∂y)` with the exact posterior mean and 3×3 covariance. Event
dimension `d = n_tasks = 3` is fixed. `rescale=True` is **not** used (the Task-B
audit showed it applies `sqrt(a(q, n_points))`, the wrong factor here, and breaks
partition invariance).

Hyperparameters fitted **once per dataset at q=2** on a fixed calibration patch
(48×48 at the image centre — no GT, no region labels), then **frozen** for both q:

| dataset | lengthscale (px) | outputscale | noise | mean |
|---|---|---|---|---|
| nuclei | [9.452, 10.356] | 316.855 | 31.365 | 182.951 |
| whole_cell | [4.631, 9.617] | 954.156 | 4.042 | 30.139 |

Matched across q: training pixels, kernel family, hyperparameters, mean, noise,
query coordinates, float64, RNG seed. **The primary comparison changes only q.**

## Step 1 — regions and adjacent-pair inventory

Frozen segmentation used for merge labelling: Round-5 `raw` + current corrected
downstream (best on nuclei, tied-best on whole-cell). GT is used **only to label
regions after all model outputs were computed**.

| dataset | adjacent GT pairs | merged | separated | A sites | B sites | C sites | D interior | D background |
|---|---|---|---|---|---|---|---|---|
| nuclei | 64 | **27** | 37 | 27 | 36 | 2 | 60 | 22 |
| whole_cell | 210 | **78** | 132 | 60 | 60 | 9 | 60 | 60 |

Class C (outer boundary ≥4 px from any interface) is nearly empty — 2 and 9 sites
— because in these crowded images almost every cell boundary is close to a
neighbour interface. Reported as-is; C is therefore not usable for contrast.

## Step 3 — partition invariance (criterion 1)

Two tests. The rigorous one is on the **distribution**, since `S2`/`S3` are Monte
Carlo and will always differ by sampling noise:

| test | result |
|---|---|
| **deterministic: per-point `mean3` and `cov3` across 1 / 2 / 5 / 40 chunks** | **INVARIANT — max abs diff 5.7e-12 (mean3), 1.0e-12 (cov3)** on both datasets |
| S1 (deterministic statistic) | **INVARIANT — max rel diff 9.9e-14** across all chunkings and both q |
| S2 (MC, 4000 samples) | within 3 MC standard errors in every partitioned case (max rel diff 1.06e-2, MC SE ≈ 3.2e-2 nuclei / 6.3e-2 whole-cell) |
| S3 (MC) | max rel diff 1.14e-2; per-point binomial SE at p≈0.1, n=4000 is 4.7e-3, so 3 SE ≈ 1.4e-2 — also within tolerance. My initial automatic verdict compared a per-point **max** against a **mean** SE and mislabelled 11 of 48 rows "NOT INVARIANT"; the corrected comparison and the deterministic test both pass. |

**Criterion 1: PASS.** The construction is partition-invariant; all sampled
differences are Monte-Carlo noise. Tolerance: `<1e-6` relative for deterministic
quantities, within 3 MC SE for sampled ones.

## Steps 4 + 5 + 7 — discrimination (primary unit = cell pair)

Thresholds `c` taken from the **q=2** pooled `||∇f||` distribution at the 90th,
95th and 99th percentiles, then the **same absolute c** applied to q=1.5
(nuclei: 5.3141 / 6.2329 / 8.0505; whole-cell: 30.9434 / 36.4519 / 49.0561).
`c` was never tuned per q.

### nuclei — 27 merged pairs, 33 separated pairs, 60 interior sites

| statistic | merged | separated | interior | contrast (merged−interior) | Cohen d | **AUC** |
|---|---|---|---|---|---|---|
| **S1 = ‖E[∇f]‖** (q-invariant) | 2.8961 | 2.2999 | 2.0005 | **0.8956** | **0.779** | **0.7438** |
| S2 q=2 | 3.0073 | 2.4447 | 2.1846 | 0.8227 | 0.752 | 0.7420 |
| S2 q=1.5 | 3.0734 | 2.5250 | 2.2819 | 0.7914 | 0.742 | **0.7414** |
| S3 p90 q=2 | 0.0949 | 0.0406 | 0.0557 | 0.0392 | 0.306 | 0.6327 |
| S3 p90 q=1.5 | 0.1044 | 0.0470 | 0.0597 | 0.0447 | 0.346 | 0.6679 |
| S3 p95 q=2 → q=1.5 | — | — | — | — | 0.135 → 0.168 | 0.6247 → 0.6444 |
| S3 p99 q=2 → q=1.5 | — | — | — | — | 0.030 → 0.037 | 0.6568 → **0.5778** |
| baseline \|∇ raw\| | 5.8881 | 5.7177 | 5.6019 | 0.2862 | 0.212 | 0.5531 |
| baseline \|∇ fastgp\| | 2.7401 | 2.1804 | 1.9779 | 0.7622 | 0.673 | 0.7154 |
| baseline \|∇ dist-transform\| | 0.8876 | 0.6407 | 0.8327 | 0.0549 | 0.374 | 0.6463 |

### whole_cell — 50 merged pairs, 56 separated pairs, 60 interior sites

| statistic | merged | separated | interior | contrast | Cohen d | AUC |
|---|---|---|---|---|---|---|
| S1 = ‖E[∇f]‖ | 14.7955 | 13.4445 | 20.4839 | **−5.6884** | −0.958 | 0.2407 |
| S2 q=2 | 14.8654 | 13.5373 | 20.5489 | −5.6834 | −0.962 | 0.2400 |
| S2 q=1.5 | 14.9098 | 13.5950 | 20.5874 | −5.6776 | −0.963 | **0.2397** |
| S3 p90 q=2 | 0.0478 | 0.0591 | 0.2032 | −0.1554 | −1.100 | 0.1853 |
| S3 p90 q=1.5 | 0.0485 | 0.0599 | 0.2038 | −0.1553 | −1.101 | 0.1867 |
| baseline \|∇ raw\| | 14.7644 | 13.4606 | 20.0774 | −5.3131 | −0.936 | 0.2427 |
| baseline \|∇ fastgp\| | 14.4738 | 13.2943 | 19.9743 | −5.5006 | −0.972 | 0.2363 |
| **baseline \|∇ dist-transform\|** | 0.8644 | 0.5997 | 0.8169 | **+0.0475** | **+0.405** | **0.6467** |

On whole-cell **every intensity-gradient statistic is anti-predictive** (AUC
0.19–0.24): merged interfaces have *lower* gradient than cell interiors, because
whole-cell interiors are strongly textured. The only positive signal there is the
distance-transform ridge — a purely geometric, non-QEP quantity.

## Global-rescaling test (criterion 4)

Over every query point of every site:

| dataset | n query points | **Spearman ρ(S2 q=1.5, S2 q=2)** | median ratio S2(1.5)/S2(2) | ratio IQR | `sqrt(a(1.5,3)/a(2,3))` predicted |
|---|---|---|---|---|---|
| nuclei | 11907 | **0.99998575** | 1.0374 | [1.0101, 1.1067] | 1.2785 |
| whole_cell | 20169 | **0.99999535** | 1.0026 | [1.0007, 1.0221] | 1.2785 |

ρ is 1.0 to five decimal places. **The q effect is an almost exactly
rank-preserving transformation**, so it cannot change which pixels look like
boundaries. **Criterion 4: FAIL.**

## Why the q effect is negligible — mechanism

| dataset | region | ‖E[∇f]‖ | posterior gradient SD | **SD / ‖E[∇f]‖** |
|---|---|---|---|---|
| nuclei | merged interfaces | 2.9658 | **0.8937** | 0.336 |
| nuclei | interiors | 1.8591 | **0.8937** | 0.688 |
| whole_cell | merged interfaces | 15.9452 | **1.6346** | 0.115 |
| whole_cell | interiors | 18.7549 | **1.6346** | 0.103 |

Two things are visible and they jointly explain the null result:

1. **The gradient posterior is mean-dominated.** The dispersion is only 10–34% of
   the mean gradient magnitude, so `‖∇f‖ ≈ ‖E[∇f]‖` and the q-dependent part of
   the distribution has little to act on. This is why the observed S2 ratio
   (1.003–1.037) is far below the 1.2785 a pure dispersion rescaling would give.
2. **The posterior gradient SD is identical across region classes** — 0.8937 at
   both merged interfaces and interiors on nuclei; 1.6346 at both on whole-cell.
   At fixed hyperparameters on a fixed lattice the posterior covariance depends
   only on the train/query **geometry**, not on the pixel values. (This matches
   the Round-4 audit finding that the exact predictive covariance is
   data-independent: 0.000e+00 between two datasets differing 3× in scale on the
   same design.) Since `q` enters **only** through that covariance, the
   q-dependent contribution is a spatially near-constant offset and carries **no
   spatial information whatsoever**.

Point 2 is the decisive one, and it is structural rather than a property of these
two images: on a regular pixel lattice with frozen hyperparameters, any statistic
whose q-dependence flows solely through the posterior covariance is
spatially constant up to boundary effects.

## Step 6 — architecture effect vs q effect

| comparison | nuclei AUC | whole_cell AUC |
|---|---|---|
| **architecture**: derivative-aware q=2 (S1) vs \|∇ raw\| | 0.7438 vs 0.5531 → **+0.19** | 0.2407 vs 0.2427 → −0.002 |
| **architecture**: derivative-aware q=2 (S1) vs \|∇ fastgp\| | 0.7438 vs 0.7154 → +0.028 | 0.2407 vs 0.2363 → +0.004 |
| **q effect**: S2 q=1.5 vs identical S2 q=2 | 0.7414 vs 0.7420 → **−0.0006** | 0.2397 vs 0.2400 → **−0.0003** |

The derivative-aware machinery **does** add something over a raw-image gradient
on nuclei (+0.19 AUC). But that gain is entirely an **architecture / smoothing**
effect available at q=2, and it is nearly matched by the cheap `|∇ fastgp|`
baseline (0.7154). The **q effect is −0.0006 and −0.0003** — negative and
negligible. Per Step 6's own rule, this must be reported as *"derivative-aware
modelling helps, but no QEP-specific benefit is demonstrated"*, and the gain must
not be attributed to q.

## Step 9 — figures

Crops selected by explicit rule, not by eye: largest merge interface, median
merge interface, weakest local contrast among merged pairs, and a representative
separated pair. Ten panels each (raw, GT, current prediction, S1, S2 q=2,
S2 q=1.5, S3 q=2, S3 q=1.5 at the same `c`, Δ_S2, `|∇ raw|`), with the two S2
panels sharing one colour scale so a global shift is not mistaken for structure.

| dataset | crop | pair | iface len |
|---|---|---|---|
| nuclei | largest-merge-interface | 133+137 | 92 |
| nuclei | median-merge-interface | 58+66 | 54 |
| nuclei | weakest-contrast-merged | 327+328 | 36 |
| nuclei | representative-separated | 306+310 | 32 |
| whole_cell | largest-merge-interface | 133+141 | 83 |
| whole_cell | median-merge-interface | 67+76 | 39 |
| whole_cell | weakest-contrast-merged | 130+379 | 4 |
| whole_cell | representative-separated | 117+129 | 20 |

## Scope limit

Two images, one per dataset. No significance is claimed and none is needed: the
decisive quantities (Spearman ρ = 0.99999, q-effect AUC change ≈ −0.0005,
region-invariant posterior SD) are not marginal effects requiring statistics.

---

# Answers

## 1. Which current segmentation failure is being targeted?

**Adjacent-cell merging** under an already-correct foreground — the residual
identified at the end of Round 8. Concretely: 27 merged GT pairs on nuclei and 78
on whole-cell under the frozen Round-5 `raw` + current corrected downstream,
which Rounds 4 and 7 both established as the dominant instance-level error once
fragmentation and seeding are fixed.

## 2. Does the derivative-aware q=2 model provide useful boundary information beyond ordinary image gradients?

**On nuclei yes, substantially; on whole-cell no.**

- nuclei: `S1 = ‖E[∇f]‖` gives AUC **0.7438** and Cohen d 0.779, against
  **0.5531** for `|∇ raw|` — a +0.19 AUC gain. But `|∇ fastgp|` already reaches
  0.7154, so most of that gain is *smoothing*, not the derivative machinery: the
  derivative-aware model adds only +0.028 AUC over a plain gradient of the
  Fast-GP reconstruction.
- whole-cell: AUC 0.2407 vs 0.2427 for `|∇ raw|` — no gain, and both are
  anti-predictive. The only useful boundary signal on whole-cell is the
  distance-transform ridge (AUC 0.6467), which involves no GP at all.

## 3. At fixed matched hyperparameters, does q=1.5 change S1, S2, S3?

| statistic | change |
|---|---|
| `‖E[∇f]‖` (S1) | **No — exactly invariant.** It is a function of the posterior mean, which is q-invariant. Confirmed to 1e-13. |
| `E[‖∇f‖]` (S2) | **Yes, but almost purely as a small monotone rescaling.** Median ratio 1.0374 (nuclei), 1.0026 (whole-cell); Spearman ρ vs q=2 = 0.99999. |
| `P(‖∇f‖ > c)` (S3) | **Yes, slightly.** nuclei p90: 0.0949 → 0.1044 on merged, 0.0557 → 0.0597 on interiors — both rise, so the contrast barely moves (0.0392 → 0.0447). whole-cell: 0.0478 → 0.0485 and 0.2032 → 0.2038 — essentially nothing. |

## 4. Which changes are merely scale changes and which alter spatial localization?

**All of them are scale changes. None alters spatial localization.**

- Spearman ρ = 0.99998575 and 0.99999535 — the q=1.5 map is a rank-preserving
  transform of the q=2 map, so no pixel changes its relative boundary-ness.
- The mechanism is structural: the posterior gradient SD is **identical at merged
  interfaces and interiors** (0.8937/0.8937; 1.6346/1.6346), because at frozen
  hyperparameters the posterior covariance depends only on the lattice geometry.
  Since q enters solely through that covariance, its contribution is a spatially
  constant offset.
- The observed ratios (1.003–1.037) are even smaller than the 1.2785 that a pure
  `a(q,3)` dispersion rescaling would predict, because the mean term — which is
  q-invariant — dominates (SD/mean = 0.10–0.34).

## 5. On MERGED interfaces specifically, does q=1.5 improve boundary-vs-interior discrimination relative to q=2?

**No.**

| dataset | statistic | q=2 | q=1.5 | change |
|---|---|---|---|---|
| nuclei | S2 AUC | 0.7420 | 0.7414 | **−0.0006** |
| nuclei | S2 Cohen d | 0.752 | 0.742 | −0.010 |
| nuclei | S2 contrast | 0.8227 | 0.7914 | −0.0313 |
| nuclei | S3 p90 AUC | 0.6327 | 0.6679 | +0.0352 |
| nuclei | S3 p99 AUC | 0.6568 | 0.5778 | **−0.0790** |
| whole_cell | S2 AUC | 0.2400 | 0.2397 | −0.0003 |
| whole_cell | S3 p90 AUC | 0.1853 | 0.1867 | +0.0014 |

The only positive movement is `S3` at the p90 threshold on nuclei (+0.035 AUC),
and it reverses sign at p99 (−0.079). That is threshold-dependent noise, not a
mechanism — and note `S1`, which is *q-invariant by construction*, outperforms
every q=1.5 statistic on nuclei (AUC 0.7438). **Criterion 3: FAIL.**

## 6. Does any q-specific gain survive comparison with simple gradient baselines?

**No.** On nuclei, `|∇ fastgp|` reaches AUC 0.7154 and Cohen d 0.673 at
essentially zero cost, against 0.7414 / 0.742 for S2 q=1.5 — and the q-invariant
S1 beats both at 0.7438. On whole-cell the trivial distance-transform ridge
(AUC 0.6467) is the only statistic with any positive discrimination, while every
QEP gradient statistic is anti-predictive. **Criterion 5: FAIL.**

## 7. Is the effect consistent across nuclei and whole-cell?

**No, and the inconsistency is qualitative rather than a matter of degree.**
On nuclei intensity-gradient statistics are predictive of merge interfaces
(AUC ≈ 0.74); on whole-cell they are *anti*-predictive (AUC ≈ 0.24) because
whole-cell interiors are textured, so interior gradient exceeds interface
gradient. There is an application-specific reason, but it does not rescue the
hypothesis — it means an intensity-gradient boundary statistic is not a
transferable mechanism across these two cell types. **Criterion 6: FAIL.**

## 8. Is the tested statistic partition invariant?

**Yes.** The underlying distribution is invariant to `5.7e-12` (`mean3`) and
`1.0e-12` (`cov3`) across 1 / 2 / 5 / 40 chunkings; the deterministic statistic
S1 to `9.9e-14`; the Monte-Carlo statistics S2 and S3 agree within 3 standard
errors. This is expected by construction, since the event dimension is the fixed
`n_tasks = 3` rather than the query-batch size. I initially auto-flagged 11 of 48
rows as "NOT INVARIANT" by comparing a per-point maximum against a mean standard
error; the corrected comparison and the deterministic test both pass.
**Criterion 1: PASS.**

## 9. Overall verdict

### QEP-SPECIFIC BOUNDARY SIGNAL: **NOT SUPPORTED**

Criteria scorecard (all six were required):

| # | criterion | verdict |
|---|---|---|
| 1 | partition / chunk invariant | **PASS** (5.7e-12) |
| 2 | q=1.5 differs from q=2 at fixed hyperparameters | **PASS**, but trivially — median ratio 1.003–1.037 |
| 3 | q=1.5 improves boundary-vs-interior discrimination on merge failures | **FAIL** (AUC −0.0006, −0.0003) |
| 4 | not merely a global rescaling | **FAIL** (Spearman ρ = 0.99999) |
| 5 | not reproduced by a simple Raw/FastGP gradient | **FAIL** (`\|∇ fastgp\|` AUC 0.7154; q-invariant S1 0.7438 beats all q=1.5) |
| 6 | consistent across both datasets | **FAIL** (predictive on nuclei, anti-predictive on whole-cell) |

Four of six fail. Per the preregistered stop rule, this mechanism is unsupported
and the search ends here — no other q, threshold or statistic will be tried.

## 10. If PROMISING — one downstream use to test next

Not applicable.

## 11. If WEAK or NOT SUPPORTED — what hypothesis was ruled out?

**Ruled out:** *that q-dependent local gradient-posterior statistics, under the
installed QePyTorch multitask construction and at fixed matched hyperparameters,
carry spatially-localized boundary evidence useful for separating adjacent cells
that the current pipeline merges.*

The reason is structural, not a tuning failure, and it explains every observation:

1. At fixed hyperparameters the posterior **mean** is q-invariant (established in
   Round 4), so `‖E[∇f]‖` cannot depend on q — verified to 1e-13.
2. `q` therefore enters **only** through the posterior covariance.
3. But at fixed hyperparameters on a regular pixel lattice, the posterior
   covariance depends only on the train/query **geometry**, not on the pixel
   values — measured here as an *identical* gradient SD at merged interfaces and
   at interiors (0.8937/0.8937 nuclei; 1.6346/1.6346 whole-cell).
4. Hence the q-dependent contribution is a spatially near-constant offset:
   Spearman ρ = 0.99999, no change in localization.
5. And it is small in absolute terms anyway, because the gradient posterior is
   mean-dominated (SD/mean = 0.10–0.34), so the realized ratio (1.003–1.037) is
   well below even the 1.2785 that a pure dispersion rescaling would give.

This closes the specific route that Rounds 7 and 8 had pointed to. It does **not**
refute Q-EP theory, and it does not speak to constructions where q reaches the
posterior mean — the Round-4 audit found Deep QEP to be the one architecture with
a q-dependent mean (max abs diff 3.9e-04 at q=1.5), which is outside this round's
scope and was explicitly excluded. What is ruled out is the exact mechanism that
the preceding rounds had identified as the most promising remaining candidate.

---

# COMPACT SUMMARY

```
NUCLEI:
best q=2 boundary contrast   = 0.8227  (S2 q=2; Cohen d 0.752, AUC 0.7420)
                               [note: q-INVARIANT S1 is better: 0.8956, d 0.779, AUC 0.7438]
best q=1.5 boundary contrast = 0.7914  (S2 q=1.5; Cohen d 0.742, AUC 0.7414)
q-specific gain              = -0.0313 contrast, -0.0006 AUC, -0.010 Cohen d   (NEGATIVE)
simple gradient baseline     = 0.7622 contrast, AUC 0.7154  (|grad fastgp|)
                               0.2862 contrast, AUC 0.5531  (|grad raw|)
partition invariant          = YES  (distribution invariant to 5.7e-12)

WHOLE-CELL:
best q=2 boundary contrast   = -5.6834 (S2 q=2; AUC 0.2400 -- ANTI-predictive)
best q=1.5 boundary contrast = -5.6776 (S2 q=1.5; AUC 0.2397 -- ANTI-predictive)
q-specific gain              = -0.0003 AUC   (NEGLIGIBLE)
simple gradient baseline     = +0.0475 contrast, AUC 0.6467  (|grad distance-transform|,
                               the ONLY positive signal on this dataset)
partition invariant          = YES  (distribution invariant to 1.0e-12)

VERDICT = NOT SUPPORTED
          (criteria 3, 4, 5, 6 fail; 1 and 2 pass. Spearman rho = 0.99999 between
           q maps; posterior gradient SD identical at interfaces and interiors, so
           q contributes a spatially constant offset carrying no localization.)

NEXT ACTION = STOP. Report only. Do not test another q, threshold or statistic
              (preregistered stop rule). The mean-only / fixed-hyperparameter QEP
              boundary route is closed; any future QEP work would have to use a
              construction in which q reaches the posterior MEAN (e.g. Deep QEP),
              which is outside this round's scope and was not tested here.
```
