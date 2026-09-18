# Round 4: locating the failure and auditing QEP mechanisms against it

Run directory: `results/real_cellseg_round4_mechanism_20260916`  ·  git HEAD 1123b84  ·  see `logs/environment.txt`

Status labels: OBSERVED / DERIVED UNDER ASSUMPTIONS / HYPOTHESIS / UNVERIFIED.

All 8 round-3 held-out means were re-verified against the saved tables before anything else (all match to <5e-5). Every image used here has now informed research decisions and is exploratory data; a future generalization claim needs fresh images.

## 1. Which failure should we target?

**Adjacent-cell MERGING at the instance-separation stage.** OBSERVED, from 8344 ground-truth instances across all 10 images and 4 arms:

| dataset | arm | matched | **merged** | boundary_under | missing_foreground | split |
|---|---|---:|---:|---:|---:|---:|
| whole_cell | raw | 57.6% | **23.4%** | 13.4% | 1.7% | 2.2% |
| whole_cell | gp | 47.2% | **24.9%** | 19.7% | 3.1% | 2.1% |
| whole_cell | qep_q2 | 47.2% | **24.9%** | 19.7% | 3.1% | 2.1% |
| whole_cell | qep_q1.5 | 49.9% | **26.9%** | 16.6% | 2.6% | 1.7% |
| nuclei | raw | 80.6% | **13.4%** | 3.2% | 1.8% | 0.5% |
| nuclei | gp | 69.5% | **22.7%** | 3.7% | 3.5% | 0.4% |
| nuclei | qep_q2 | 69.5% | **22.7%** | 3.7% | 3.5% | 0.4% |
| nuclei | qep_q1.5 | 68.5% | **23.2%** | 4.1% | 3.4% | 0.5% |

Merging is 6-13x more common than missing foreground. This CORRECTS the round-3 conclusion that the dominant mode was 'under-detection': at the instance level the false negatives are mostly merges, not undetected cells. Pixel foreground recall and instance recall are different quantities and the gap proves it — whole-cell GP has pixel recall 0.574 but instance recall 0.405.

Smoothing makes merging WORSE: nuclei merged 13.4% (raw) -> 22.7% (GP). That is why Raw wins.

## 2. What evidence locates the cause?

Three independent lines, all OBSERVED.

**(a) Marker counts inside cells.** Merged cells are in the foreground but have no seed of their own:

| dataset | category | n | markers inside | fg coverage |
|---|---|---:|---:|---:|
| whole_cell | matched | 2065 | 1.384 | 0.690 |
| whole_cell | merged | 1024 | 0.693 | 0.720 |
| whole_cell | boundary_under | 711 | 3.201 | 0.479 |
| whole_cell | missing_foreground | 108 | 2.537 | 0.166 |
| nuclei | matched | 3063 | 1.398 | 0.870 |
| nuclei | merged | 871 | 0.526 | 0.873 |
| nuclei | boundary_under | 156 | 3.141 | 0.472 |
| nuclei | missing_foreground | 129 | 3.287 | 0.131 |

Merged cells: **0.693 / 0.526** markers each. Matched cells: **1.384 / 1.398**. Yet merged cells' foreground coverage is equal or HIGHER (0.720 vs 0.690; 0.873 vs 0.870). The cells are detected; they never get seeded.

**(b) Oracle stage replacement** (ORACLE / DIAGNOSTIC; not deployable, not an upper bound — a different marker rule could score higher):

| dataset | variant | AP@0.5 | merged |
|---|---|---:|---:|
| whole_cell | ORACLE foreground, markers recomputed | 0.3678 | 44.2 |
| whole_cell | raw: real pipeline | 0.2248 | 33.5 |
| whole_cell | raw: **ORACLE markers**, real foreground | **0.4350** | 39.0 |
| whole_cell | gp: real pipeline | 0.1821 | 34.8 |
| whole_cell | gp: **ORACLE markers**, real foreground | **0.2998** | 41.8 |
| whole_cell | qep_q1.5: real pipeline | 0.2107 | 37.2 |
| whole_cell | qep_q1.5: **ORACLE markers**, real foreground | **0.3282** | 46.8 |
| nuclei | ORACLE foreground, markers recomputed | 0.8785 | 35.0 |
| nuclei | raw: real pipeline | 0.7543 | 32.2 |
| nuclei | raw: **ORACLE markers**, real foreground | **0.9001** | 8.0 |
| nuclei | gp: real pipeline | 0.5822 | 51.0 |
| nuclei | gp: **ORACLE markers**, real foreground | **0.7179** | 32.5 |
| nuclei | qep_q1.5: real pipeline | 0.5677 | 52.0 |
| nuclei | qep_q1.5: **ORACLE markers**, real foreground | **0.6770** | 34.5 |

Oracle MARKERS on the real foreground beat oracle FOREGROUND with recomputed markers (whole-cell 0.4350 vs 0.3678; nuclei 0.9001 vs 0.8785). Seeding, not foreground, is the binding constraint. On nuclei, oracle markers cut merges 32.2 -> 8.0. For GP the same intervention only reaches 0.7179 with merges 32.5, because GP smoothing has already fused cells in the binary mask where no seeding rule can recover them.

**(c) Round-2/3 threshold confound resolved.** criterion_1 vs Li x global vs per-tile on development images, downstream frozen:

| dataset | rule | scope | fg Dice | AP@0.5 |
|---|---|---|---:|---:|
| whole_cell | criterion_1 | global | 0.8108 | 0.4805 |
| whole_cell | criterion_1 | per_tile | 0.7794 | 0.3669 |
| whole_cell | li | global | 0.8031 | 0.4671 |
| whole_cell | li | per_tile | 0.8202 | 0.4872 |
| nuclei | criterion_1 | global | 0.4408 | 0.1415 |
| nuclei | criterion_1 | per_tile | 0.6523 | 0.3314 |
| nuclei | li | global | 0.8522 | 0.6354 |
| nuclei | li | per_tile | 0.8762 | 0.6539 |

On nuclei the round-3 gain was mostly the ALGORITHM (criterion_1 -> Li at global scope: AP 0.1415 -> 0.6354), with scope adding less (0.6354 -> 0.6539). On whole-cell the algorithm barely matters. Rescoring old masks alone would NOT have separated these; the 2x2 was required.

## 3-4. Which QEP property could act, and through what computed quantity?

Full detail in `mechanism_cards.md`; `source_notes.md` records what each paper does and does not state.

**The honest answer: in the installed exact implementation, none.** OBSERVED, at fixed hyperparameters, max absolute difference versus q=2:

| quantity | q in {1.0,1.2,1.5,1.8,3.0} |
|---|---|
| ExactQEP predictive mean | **0.000e+00** |
| ExactQEP predictive variance | **0.000e+00** |
| ExactQEP variance across two datasets differing 3x in scale | **0.000e+00** |
| derivative-aware QEP (`Matern52KernelGrad`, 3 tasks) mean and variance | **0.000e+00** |
| **DeepQEP predictive mean** (identical init, identical RNG stream) | **3.89e-04** (q=1.5), **6.13e-04** (q=1.2) |

So every candidate boundary quantity — posterior mean, variance, `||E[grad f]||`, `E[||grad f||]`, boundary-event probabilities — is q-invariant for exact single-layer QEP, because `qpytorch/models/exact_qep.py` is structurally identical to `gpytorch/models/exact_gp.py` (verified by diff). q enters only `log_prob`, hence only hyperparameter learning, and a GP given those hyperparameters reproduces the image exactly (36/36 pixel-identical, round-2 check).

The one exception is **Deep Q-EP** (Chang, Obite, Zhou, Lan, PMLR v289), where the layer is sampled from a q-dependent distribution and passed non-linearly onward, so the channel is measurably non-zero. That is the only live QEP candidate, and its link to seed placement is HYPOTHESIS, not evidence.

## 5. What did the pilots demonstrate?

Both pilots targeted the located failure with non-QEP mechanisms and **both failed their pre-registered criteria** (written to `pilot_configs/` before any pilot score was computed; criterion (a) was merged count falling >=15% relative on at least one dataset).

| pilot | arm | AP@0.5 frozen -> pilot | merged frozen -> pilot | verdict |
|---|---|---|---|---|
| P1 elevation w=0.2 | raw | 0.2248 -> 0.2674 | 33.5 -> 33.8 | (a) FAIL |
| P1 elevation w=0.2 | gp | 0.1821 -> 0.2185 | 34.8 -> 34.2 | (a) FAIL |
| P1 elevation w=0.2 | qep_q1.5 | 0.2107 -> 0.2260 | 37.2 -> 39.0 | (a) FAIL |
| P2 markers wm=0.5 | raw | 0.2248 -> 0.2393 | 33.5 -> 38.0 | (a) FAIL |
| P2 markers wm=0.5 | gp | 0.1821 -> 0.1740 | 34.8 -> 35.5 | (a) FAIL |
| P2 markers wm=0.5 | qep_q1.5 | 0.2107 -> 0.1873 | 37.2 -> 36.8 | (a) FAIL |

What they DID show: Pilot 1's elevation raises whole-cell AP for every arm (+0.043 raw) but by cutting SPLITS (5.8 -> 3.2), not merges. Pilot 2 selected wm=0.5 on development (merged 147.5 -> 144.5, AP 0.4671 -> 0.4962) and that did NOT transfer: held-out merges rose.

What they FAILED to show: any reduction in merging. Best was -1.7%. Raw already emits **1137 markers for ~155 whole-cell cells** (7.3x over-seeded) and still merges 33.5, so the problem is not a shortage of seeds but their placement — the surplus sits in noise specks that cleanup removes, while a fused pair still shares one seed. HYPOTHESIS for why: whole-cell intensity is not reliably peaked at cell centres, so no global blend of distance and intensity puts one seed per fused cell.

## 6. Does anything survive both controls?

**No QEP benefit survives.** q=1.5 beat its Gaussian twin in Pilot 1 (0.2260 vs 0.2185) but lost to Raw (0.2674), failing the pre-registered QEP criterion which required beating both. In Pilot 2 q=1.5 also lost to Raw (0.1873 vs 0.2393). Under the frozen round-3 pipeline Raw remains best on both datasets.

## 7. Is the effect q, parameter learning, architecture, or postprocessing?

**Postprocessing dominates; any exact-QEP q effect is parameter learning.** OBSERVED: the largest single lever measured anywhere in this project is postprocessing — explicit markers gave +0.23/+0.17 AP (round 2, 32/32 cases) and the threshold algorithm gave +0.49 on nuclei (Phase 2C). For the exact model, q cannot change any predictive output, so whatever q does is confined to hyperparameter selection and is GP-reproducible. Architecture (depth) is the only untested channel, and it must be compared against a Deep GP with identical machinery before being attributed to q.

## 8. ONE most informative next experiment

**Take the oracle-marker result apart: measure how much of the +0.21 AP headroom is recoverable from a seed rule that uses only per-cell shape/size information available at inference, by replacing global peak-picking with per-connected-component seeding — and evaluate it on FRESH images not used so far.**

Concretely: for each foreground component, estimate an expected cell count from its area divided by a size statistic learned on the two development images, then place exactly that many seeds inside the component by constrained farthest-point selection on the distance transform. This is deployable (no GT at inference), it attacks the measured cause directly (0.5-0.7 seeds per merged cell versus 1.4 for matched), and it is cheap because it reuses the cached binaries with no model fitting.

Why this instead of Deep Q-EP: the failure is seed PLACEMENT inside an already correct foreground. Oracle markers prove +0.21 AP is available there. Nothing measured connects a q<2 latent prior to seed placement, and a Deep Q-EP pilot costs multiple hours plus a matched Deep GP control. If per-component seeding recovers most of the oracle headroom, the merge problem is solved without any model change and the QEP question can be re-posed on whatever failure remains. If it recovers little, that is strong evidence the missing information is genuinely in the image/reconstruction, which is the first real justification for trying a learned representation such as Deep Q-EP.


**All 10 images are now exploratory.** Any generalization claim requires fresh independent evaluation images, which this repository does not currently contain.

