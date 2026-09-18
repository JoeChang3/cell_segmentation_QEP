# Round 10 — Deep QEP spatial-mean pilot

2026-09-18. Frozen pre-pilot checkpoint `f312ff045625aac960bc6d1c97227a89f6a1833c`
verified as HEAD with no tracked modifications; Rounds 1–9 intact and untouched.
Design preregistered in `preregistered_design.json` **before any pilot output was
examined**. No segmentation stage was modified and no AP was computed.

## What was tested

Rounds 1–9 closed the shallow route: at fixed hyperparameters the shallow ExactQEP
predictive **mean** is q-invariant, and Round 9 found that shallow q-dependent
gradient *distributional* statistics produce rank-preserving rescalings rather than
localized boundary information (`NOT SUPPORTED`). Round 10 asks the one remaining
structural question: **does depth make the predictive mean itself q-dependent, and
is that difference useful at adjacent-cell interfaces the pipeline currently merges?**

## Reference implementation and shadowing decision

| item | value |
|---|---|
| repo | `https://github.com/lanzithinking/DeepQEP.git` |
| local path | `/Users/zchan/eclipse-workspace/DeepQEP` (**sibling**, never vendored) |
| branch / commit | `main` / **`aa843960a3e8692e06eb36522fdbb2043bf34f2e`** |
| git status | clean, 0 dirty files, before and after our work |

The repo ships a **full vendored gpytorch fork** (`DeepQEP/gpytorch/`, 6.6 MB) and
its demos shadow the interpreter via `sys.path.insert(0, '../GPyTorch')`. We did
**not** put that fork on our `sys.path`. Instead the pilot uses the **installed,
already-audited qpytorch 0.2** — the successor package — justified by a diff after
normalising the package-root rename:

| file | result |
|---|---|
| `models/deep_qeps/deep_qep.py` | **functionally identical**; installed only adds `rsample(rescale=kwargs.pop('rescale', False))`, whose default reproduces the fork exactly |
| `models/deep_qeps/__init__.py` | **byte-identical** |
| `distributions/multivariate_qexponential.py` | installed is a **strict superset** (adds `rescalor`, absent in the fork) |
| `likelihoods/qexponential_likelihood.py` | cosmetic; installed adds an optional `reduction` kwarg |

## Where q enters — traced, not assumed

`qpytorch/models/deep_qeps/deep_qep.py`, `DeepQEPLayer.__call__`:

```python
inputs = QExponential(loc=inputs.mean, scale=inputs.variance.sqrt(),
                      power=inputs.power).rsample(rescale=False)
```

The inter-layer latent is **resampled** from a univariate q-exponential, whose
`rsample` is `eps = |z|^(2/q − 1) · z` for q ≠ 2 and `eps = z` for q = 2. So q
non-linearly warps the latent handed to the next layer, and the composed predictive
mean becomes q-dependent. At q = 2 the branch is skipped, so **the q=2 arm is the
same architecture and the same code path** — the primary control, not a separate
DeepGP codebase.

Two honest caveats: this resample is **mean-field** (it uses only `inputs.mean` and
`inputs.variance`, the scale-matrix diagonal per the Task-B audit, not the full
latent covariance); and `rescale=False` means the latent carries the `a(q,d)` scale
factor documented in the chunk-invariance audit.

## Part 3 — reference demo reproduced first

`demo/demo_multi_Deep_QEP.py`, run in an **isolated subprocess** (fork on path only
inside that process), unmodified except the `POWER` value and dropping the plotting
tail:

| POWER | final ELBO loss | mean finite | var finite |
|---|---|---|---|
| 1.0 (as shipped) | −4.926163 | yes | yes |
| 2.0 (control) | −0.142754 | yes | yes |

Identical-seed rerun: `max|mean diff| = 0.0`, identical loss. **Compatibility work
required: installing `tqdm` (listed in the README, missing from the env). No code
patch, no model rewrite.**

## Stage 0 (Part 5) — does depth make the mean q-dependent at all?

Matched Deep q=2 vs q=1.5 on a kinked piecewise-smooth 1-D target; identical seed,
inducing draw, steps; common MC streams; MC SE from 12 independent streams on the
same trained model.

| quantity | value |
|---|---|
| max \|Δμ\| | **0.133734** |
| mean \|Δμ\| | 0.045233 |
| RMSE(μ_q1.5, μ_q2) | 0.058146 |
| corr(μ_q1.5, μ_q2) | 0.989912 |
| MC SE of Δ (mean / max) | 0.001889 / 0.004872 |
| **SNR_max / SNR_mean** | **27.45 / 23.95** |
| spatial non-uniformity (sd/mean of \|Δμ\|) | 0.808 |
| **shallow ExactQEP control, max \|Δμ\|** | **0.000e+00 exactly** |

**Gate PASSED.** Depth genuinely opens a q-dependent predictive-mean channel that
the shallow model does not have — the shallow control is exactly zero. This is a
real mechanistic finding and it is the one thing Round 10 establishes positively.

## Stage 1 — smoke pilot (25×25 patches, 1 seed)

2 merged + 2 separated pairs per dataset, selected deterministically from the
Round-9 inventory (longest and median interface length), never using DeepQEP output.

| dataset | mean \|Δμ\| | MC SE | SNR_mean | non-uniformity | localization ratio | edge/iface |
|---|---|---|---|---|---|---|
| nuclei | 0.5766 | 0.0760 | 7.56 | 0.813 | **1.453** | 0.834 |
| whole_cell | 2.8307 | 0.2823 | 9.49 | 0.880 | **0.856** | **1.071** |

Stage-1 stop conditions: (1) differs beyond MC error **PASS**; (2) spatially
non-uniform **PASS**; (3) concentrated near cell structure — **PASS on nuclei
(1.453), FAIL on whole-cell** (0.856, and patch edges carried more difference than
interfaces); (4) no numerical pathology **PASS**. Proceeded to Stage 2 on the literal
wording "at least some difference is concentrated near real cell structure", with the
whole-cell failure flagged at the time rather than after the fact.

## Stage 2 — confirmation (4 merged + 4 separated pairs per dataset, 3 paired seeds)

16 pairs × 3 seeds × 2 q = **96 paired fits**.

| dataset | mean \|Δμ\| | MC SE | SNR_mean | non-uniformity | localization ratio | edge/iface |
|---|---|---|---|---|---|---|
| nuclei | 0.77494 | 0.09342 | 8.50 | 0.799 | 1.282 | **1.120** |
| whole_cell | 3.11235 | 0.29137 | 10.79 | 0.929 | 1.645 | 0.669 |

### Criterion 4 — the decisive control

Paired q effect `|μ_q1.5(s) − μ_q2(s)|` versus seed-to-seed variation **within the
same q** `|μ_q2(s) − μ_q2(s′)|`, same patch and query grid:

| dataset | pairs | paired q effect | same-q seed variation | **ratio q / seed** |
|---|---|---|---|---|
| nuclei | 8 | 0.77494 | 0.66154 | **1.171** |
| whole_cell | 8 | 3.11235 | 3.27385 | **0.951** |

**The q effect on the predictive mean is the same size as simply changing the random
seed** — and on whole-cell it is smaller than seed noise.

### Parts 8 D/E — boundary quality of the predictive mean, MERGED pairs

| dataset | statistic | interface | interior | contrast | Cohen d | **AUC** |
|---|---|---|---|---|---|---|
| nuclei | \|∇μ\| q=2 | 3.6360 | 2.1615 | 1.4745 | 1.896 | **0.8750** |
| nuclei | \|∇μ\| q=1.5 | 3.5210 | 2.2634 | 1.2576 | 1.466 | **0.8750** |
| nuclei | baseline \|∇ raw\| | 6.9569 | 5.8609 | 1.0959 | 2.574 | **1.0000** |
| nuclei | baseline \|∇ fastgp\| | 3.6559 | 2.5841 | 1.0718 | 1.375 | 0.8750 |
| whole_cell | \|∇μ\| q=2 | 14.0726 | 13.4567 | 0.6159 | 0.164 | 0.5764 |
| whole_cell | \|∇μ\| q=1.5 | 14.1791 | 13.3890 | 0.7901 | 0.224 | 0.5972 |
| whole_cell | baseline \|∇ raw\| | 19.0323 | 12.5403 | 6.4920 | 2.372 | **0.9375** |
| whole_cell | baseline \|∇ fastgp\| | 18.7069 | 12.4262 | 6.2808 | 2.295 | **0.9375** |

**q-specific gain:** nuclei AUC **+0.0000**, contrast **−0.2169**, Cohen d **−0.430**.
whole_cell AUC **+0.0208**, contrast **+0.1743**, Cohen d **+0.059**.

## Figures

All 20 Stage-2 seed-73 patches plotted, **no selection and no cherry-picking**, in
`figures/patch_*.png`. Each is the prescribed 8-panel layout (raw / GT / frozen
segmentation / μ_q2 / μ_q1.5 / Δμ / |∇μ_q2| / |∇μ_q1.5|); the two mean panels share
one intensity scale, the two gradient panels share one scale, and Δμ uses a symmetric
scale. `figures/stage0_synthetic_gate.png` shows the Stage-0 gate.

## Scope limit

Two development images, 16 pairs, 3 seeds. No significance is claimed. The decisive
numbers (q/seed ratio ≈ 1, AUC gain +0.000/+0.021, baseline AUC 1.000/0.938) are not
marginal effects needing statistics.

---

# Answers

**1. Which exact DeepQEP reference implementation was used?**
`https://github.com/lanzithinking/DeepQEP.git`, branch `main`, commit
`aa843960a3e8692e06eb36522fdbb2043bf34f2e`, cloned read-only to
`/Users/zchan/eclipse-workspace/DeepQEP` as a sibling. The architecture follows
`demo/demo_multi_Deep_QEP.py`; the executed code is the installed qpytorch 0.2,
verified functionally identical for `deep_qeps` (see the diff table above).

**2. What compatibility changes were necessary?**
One dependency install: **`tqdm` 4.70.1**, which the DeepQEP README requires and the
environment lacked. **No code patch, no model rewrite, no adaptation layer.** For the
reference-demo run only, `POWER` and the plotting tail were altered **in memory**;
the repository files were never modified (0 dirty files). The vendored fork was
deliberately kept off `sys.path`.

**3. Is q=2 implemented by the same architecture/code path as q=1.5?**
**Yes.** Both arms are the same `DeepQEPRegressor` with the same layers, kernels,
means, variational distributions, likelihood and optimizer; `power` is the only
differing argument. At q=2 the `rsample` warp reduces to the identity (`eps = z`),
so the q=2 arm *is* the Deep-GP control inside the QEP code path. The repository's
separate `deep_gps` demo was not used as the primary control.

**4. Does depth make the predictive mean genuinely q-dependent?**
**Yes — this is the clean positive result.** Stage 0: max |Δμ| = 0.1337 with
SNR_max = 27.4, while the **shallow ExactQEP control at fixed hyperparameters gives
exactly 0.000e+00**. On cell patches, mean |Δμ| = 0.775 (nuclei) and 3.112
(whole-cell) in raw intensity units. Depth therefore does open the channel that
Rounds 4 and 9 showed is absent in the shallow model.

**5. Is that mean difference larger than Monte Carlo error?**
**Yes, comfortably.** SNR_mean = 8.50 (nuclei) and 10.79 (whole-cell); SNR_max
13.54 and 17.91; Stage-0 SNR_max 27.4. MC SE was estimated from repeated independent
sample streams applied to the same trained model.

**6. Is it larger than ordinary seed-to-seed variability?**
**No — this is the first decisive failure.** Ratio of paired q effect to same-q seed
variation is **1.171** (nuclei) and **0.951** (whole-cell). The q effect on the mean
is indistinguishable in magnitude from re-rolling the random seed, and on whole-cell
it is smaller. The *useful* quantity — the discrimination gain — is far below seed
noise (AUC +0.0000 / +0.0208).

**7. Is the q-dependent mean difference spatially localized?**
**Yes, but not preferentially at the interfaces that matter.** It is clearly
non-uniform (sd/mean 0.799 / 0.929) and the interface/interior localization ratio
exceeds 1 in Stage 2 (1.282 / 1.645). However on nuclei the **patch edge carries more
q-difference than the interface** (edge/iface = **1.120**), so part of the
localization is a support artifact rather than cell structure — criterion 5 is only
partially met.

**8. Does q=1.5 improve mean-based boundary discrimination at CURRENT merged interfaces?**
**No.** nuclei: AUC **0.8750 → 0.8750** (+0.0000), contrast 1.4745 → 1.2576
(**−0.2169**), Cohen d 1.896 → 1.466 (**−0.430**) — unchanged or worse. whole-cell:
AUC 0.5764 → 0.5972 (+0.0208) from a near-chance baseline. The directions disagree
between datasets.

**9. Does it outperform or add information beyond Raw/FastGP mean gradients?**
**No — it is strictly worse.** A plain raw-image gradient achieves AUC **1.0000** on
nuclei and **0.9375** on whole-cell, versus 0.875 and 0.576–0.597 for the deep
predictive-mean gradient. `|∇ fastgp|` matches or beats the deep model too (0.875 /
0.9375). The deep QEP mean adds no boundary information unavailable from a trivial,
essentially free alternative.

**10. Is the effect consistent enough across the two development datasets to justify a segmentation experiment?**
**No.** The contrast gain has **opposite signs** (nuclei −0.2169, whole-cell +0.1743);
the localization diagnostics disagree (nuclei edge/iface 1.120 vs whole-cell 0.669;
Stage-1 localization 1.453 vs 0.856); and the baseline dominance differs in degree
but holds in both. No pre-existing dataset-specific mechanism explains the sign flip.

## Criteria scorecard (preregistered; all six required for PROMISING)

| # | criterion | verdict |
|---|---|---|
| 1 | q=1.5 mean differs from q=2 beyond MC error | **PASS** (SNR 8.5 / 10.8; Stage-0 27.4) |
| 2 | difference spatially localized, not a global shift | **PASS** (non-uniformity 0.80 / 0.93) |
| 3 | improves mean-based discrimination on MERGED pairs | **FAIL** (AUC +0.0000 / +0.0208; nuclei contrast −0.2169) |
| 4 | gain ≥ comparable to paired seed variability | **FAIL** (q/seed = 1.171 / 0.951; AUC gain ≪ seed noise) |
| 5 | not explained by patch edges / inducing points | **PARTIAL** (nuclei edge/iface = 1.120) |
| 6 | direction consistent across datasets | **FAIL** (contrast gain signs opposite) |

# VERDICT

### DEEP-QEP SPATIAL-MEAN MECHANISM: **NOT SUPPORTED**

Three of six criteria fail outright and one is only partial. The distinction worth
recording precisely:

- **The mechanism exists.** Depth makes the predictive mean q-dependent, beyond MC
  error, where the shallow model is exactly q-invariant (0.000e+00). That is a real
  finding and it is new relative to Rounds 4 and 9.
- **The mechanism is not useful here.** Its magnitude equals random-seed noise, it
  does not improve merged-interface discrimination (AUC +0.000 on nuclei), its sign
  flips between datasets, part of its localization sits on patch edges, and a plain
  raw-image gradient outperforms it (AUC 1.000 vs 0.875).

Per the preregistered stop rule: **STOP. Do not tune q, architecture, number of
layers, or inducing points.** This is not an implementation blocker — the reference
demo reproduced exactly and the only compatibility need was installing `tqdm`.

## What this rules out

The hypothesis that *DeepQEP's q-dependent predictive mean supplies spatially
localized boundary evidence at adjacent-cell interfaces useful for resolving the
current pipeline's merge failures.* Combined with Rounds 4, 9 and 10, all three
routes by which q could have reached the segmentation decision are now closed:
the shallow mean (exactly q-invariant), shallow gradient distributional statistics
(rank-preserving rescaling, Round 9), and the deep mean (real but seed-sized,
non-specific, and beaten by a raw gradient).

The merge failure identified in Rounds 4–8 remains open, but the evidence now points
away from q as its solution and toward boundary evidence that is not derived from a
q-exponential process at all — note that the trivial `|∇ raw|` baseline separates
merged interfaces from interiors with AUC 1.000 on nuclei, which suggests the
information needed is already present in the image and the deficit is in how the
watershed consumes it.
