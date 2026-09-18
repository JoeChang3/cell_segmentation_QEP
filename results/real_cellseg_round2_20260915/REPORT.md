# Round 2: reproducible markers, GP/q=2 reconciliation, held-out validation

Run directory: `results/real_cellseg_round2_20260915`  
run_id: `round2-v1_ee35f34412ef`  
Protocol: round2-v1

## Headline

1. **Explicit watershed markers improve the shared pipeline in 32/32 held-out (image, arm) cases.** Mean AP@0.5 nuclei 0.1685 -> 0.3952; whole-cell 0.0468 -> 0.2198. Same cached binaries, no refitting.

2. **The round-1 GP vs QEP q=2 gap is explained and eliminated.** It was unseeded stochastic log-determinant estimation, not q and not an inherent noise floor. GP and QEP q=2 are now bitwise identical on all 8 held-out images (dAP@0.5 = 0.00e+00).

3. **q=1.5 shows no repeatable advantage.** Nuclei: better on 1/4 images, mean +0.0479 driven entirely by figure_4 (+0.2768) with the other three negative. Whole-cell: 2/4, mean -0.0044.

4. **Reconstruction over raw is modest and dataset-dependent.** Nuclei GP 0.3982 vs raw 0.3383; whole-cell GP 0.2210 vs raw 0.2206 (no difference).


## 1. Marker reproducibility (Step 1)

| check | result |
|---|---|
| legacy separation reproduces round-1 saved instance masks (up to label renaming) | 10/10 arms YES |
| postprocessing run twice: markers identical | 4/4 |
| postprocessing run twice: partition equivalent | 4/4 |
| postprocessing run twice: max abs AP delta | 0.00e+00 |

## 2. Development marker study (Step 1)

min_distance chosen by maximizing MEAN AP@0.5 across deployable arms on development images only, never per method. Selected: **nuclei 15, whole-cell 9** (nuclei verified as an interior optimum: md=18/21/25 give 0.3624/0.3510/0.3354).

| dataset | method | AP@0.5 legacy | AP@0.5 peak(selected md) | #pred legacy | #pred peak |
|---|---|---:|---:|---:|---:|
| nuclei | raw | 0.0213 | 0.4122 | 819 | 297 |
| nuclei | gp | 0.0315 | 0.3722 | 750 | 293 |
| nuclei | qep_q2 | 0.0255 | 0.3664 | 714 | 289 |
| nuclei | qep_q1.5 | 0.0347 | 0.3348 | 685 | 264 |
| nuclei | gp_legacy | 0.0161 | 0.0338 | 237 | 68 |
| whole_cell | raw | 0.1247 | 0.4781 | 616 | 339 |
| whole_cell | gp | 0.1158 | 0.4171 | 638 | 375 |
| whole_cell | qep_q2 | 0.1201 | 0.4087 | 632 | 376 |
| whole_cell | qep_q1.5 | 0.1330 | 0.4160 | 628 | 356 |
| whole_cell | gp_legacy | 0.1158 | 0.3432 | 647 | 321 |

Oracle-foreground runs are in `oracle_foreground_diagnostics.csv`. They substitute the ground-truth foreground and are **diagnostics only** - not a deployable method and not a mathematical upper bound.


## 3. GP / QEP q=2 equivalence (Step 2A)

| stage | finding |
|---|---|
| fixed parameters, identical training pixels | max abs prediction difference **0.000e+00** at all three tested parameter settings |
| independent training from library defaults | trajectories **never** differ; identical lengthscale/outputscale/noise/mean/loss at every iteration |
| first differing stage | none, once the RNG is controlled |
| root cause of the round-1 gap | gpytorch `max_cholesky_size=800` < N=3000, so the exact-MLL log-determinant used stochastic Lanczos quadrature with 10 **random** probe vectors (`deterministic_probes=False`). Round 1 ran arms sequentially in one process without reseeding torch. |
| repeatability study (same arm, same data, no reseed) | fitted lengthscale 0.02957 / 0.03135 / 0.03093 across three identical reruns (spread 1.78e-3); noise 0.01241 / 0.01222 / 0.01339 |
| with the fix (exact_logdet + per-tile torch seed) | bitwise identical across reruns; GP == QEP q=2 on all 8 held-out images |

The round-1 phrase "pipeline noise floor" was **not** justified and is withdrawn: the variation was an unseeded-RNG defect with an identifiable cause, not irreducible noise. `reconRMSE` has been renamed **`rmse_to_raw`**; it measures how much the reconstruction changed the observed image, not accuracy against an unknown clean image, which does not exist for real microscopy.


## 4. Legacy GP provenance (Step 2B)

| item | value |
|---|---|
| 0.583864 location | `data/combined_ap_table.csv`, Threshold=0.50, Pair=whole_cell_figure_1, column GP_Method_AP (per-image, not aggregated) |
| reproduced by | operation (1): recomputing `compute_ap_from_ious` on `data/whole_cell_test_images/whole_cell_figure_1/ious_gp.csv` -> **0.583864 exactly**, TP=275 FP=68 FN=128 |
| historical IoU matrix shape | 403 GT x **343 predicted** |
| 0.1158 location | round-1 new shared pipeline, whole_cell gp/gp_legacy arm with markers=None, n_pred 638/647 |
| predicted-instance evidence | markers=None gives 647 (gap 304 from 343); peak md=7 gives 357 (gap 14), md=9 gives 321 (gap 22). The historical separation behaved like **marker-based**, not markers=None |
| residual unexplained | matching n_pred is not sufficient: peak md=7 still scores 0.3194 vs the historical 0.583864, so the historical foreground/mask quality was also better (FP only 68 of 343) |
| MISSING artifact 1 | the historical GP **instance mask array** was never saved; only the IoU matrix and a rendered `GP_boundaries.png` |
| MISSING artifact 2 | no R/EBImage runtime here, so operation (4) cannot be run; 17 R scripts exist under `r_reference/` |
| operation (2) status | runnable in principle but not run: the literal `generate_gp_masks_test` predicts on ~68k tile pixels at once (OOM-killed in round 1) and uses an unseeded subsample, so it has no single well-defined output |

**Bounded conclusion:** the discrepancy is *mostly* attributable to instance separation (markers=None vs marker-based, worth roughly +0.20 AP here), with a residual gap that cannot be closed without the two missing artifacts above.


## 5. Metric correctness (Step 2C)

All 8 checks pass: identity, empty prediction, empty truth, exact-boundary equal split, unequal 70/30 split, pure merge, label-permutation invariance, and a double-match guard on the real development masks (max duplicate prediction reuse = 0, so matching is effectively one-to-one at tau>=0.5).

No metric bug was found, so no separate corrected metric version was needed; the project AP convention TP/(TP+FP+FN) is preserved. **mean matched IoU** = max IoU per ground-truth instance, averaged over ALL ground-truth instances (undetected GT contributes 0).


## 6. Image manifest (Step 3.1)

| dataset | image | HxW | square | #GT | role |
|---|---|---|---|---:|---|
| nuclei | nuclei_figure_1 | 1128x962 | no | 330 | development |
| nuclei | nuclei_figure_2 | 250x374 | no | 77 | heldout_candidate |
| nuclei | nuclei_figure_3 | 492x510 | no | 251 | heldout_candidate |
| nuclei | nuclei_figure_4 | 608x960 | no | 254 | heldout_candidate |
| nuclei | nuclei_figure_5 | 1024x1024 | yes | 151 | heldout_candidate |
| whole_cell | whole_cell_figure_1 | 600x602 | no | 403 | development |
| whole_cell | whole_cell_figure_2 | 800x800 | yes | 41 | heldout_candidate |
| whole_cell | whole_cell_figure_3 | 600x602 | no | 417 | heldout_candidate |
| whole_cell | whole_cell_figure_4 | 1000x1000 | yes | 73 | heldout_candidate |
| whole_cell | whole_cell_figure_5 | 1000x1000 | yes | 89 | heldout_candidate |

No exact duplicates; max pairwise downsample correlation 0.4044, so the 8 held-out images are treated as independent. **4 held-out images are square**, which is exactly where the legacy `align_mask_to_reference` would have silently left the ground truth transposed; the native-orientation loader removes that hazard.


## 7. Held-out per-image results (Step 3)

| Dataset | Image | Method | AP@0.5 | AP@0.75 | TP@0.5 | FP@0.5 | FN@0.5 | #pred | #GT | Runtime (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nuclei | figure_2 | Raw | 0.8025 | 0.2920 | 65 | 4 | 12 | 69 | 77 | 0 |
| nuclei | figure_2 | GP | 0.4659 | 0.0840 | 41 | 11 | 36 | 52 | 77 | 41 |
| nuclei | figure_2 | QEP q=2 | 0.4659 | 0.0840 | 41 | 11 | 36 | 52 | 77 | 40 |
| nuclei | figure_2 | QEP q=1.5 | 0.4045 | 0.0965 | 36 | 12 | 41 | 48 | 77 | 41 |
| nuclei | figure_2 | GP-legacy-config | 0.0000 | 0.0000 | 0 | 0 | 77 | 0 | 77 | 41 |
| nuclei | figure_2 | ImageJ (saved masks) | 0.9231 | 0.2295 | 72 | 1 | 5 | 73 | 77 | 0 |
| nuclei | figure_3 | Raw | 0.1785 | 0.0241 | 58 | 74 | 193 | 132 | 251 | 1 |
| nuclei | figure_3 | GP | 0.1600 | 0.0217 | 52 | 74 | 199 | 126 | 251 | 162 |
| nuclei | figure_3 | QEP q=2 | 0.1600 | 0.0217 | 52 | 74 | 199 | 126 | 251 | 162 |
| nuclei | figure_3 | QEP q=1.5 | 0.1470 | 0.0153 | 51 | 96 | 200 | 147 | 251 | 165 |
| nuclei | figure_3 | GP-legacy-config | 0.0401 | 0.0035 | 11 | 23 | 240 | 34 | 251 | 165 |
| nuclei | figure_3 | ImageJ (saved masks) | 0.4745 | 0.1245 | 186 | 141 | 65 | 327 | 251 | 0 |
| nuclei | figure_4 | Raw | 0.1874 | 0.0300 | 92 | 237 | 162 | 329 | 254 | 12 |
| nuclei | figure_4 | GP | 0.4494 | 0.2494 | 160 | 102 | 94 | 262 | 254 | 487 |
| nuclei | figure_4 | QEP q=2 | 0.4494 | 0.2494 | 160 | 102 | 94 | 262 | 254 | 479 |
| nuclei | figure_4 | QEP q=1.5 | 0.7263 | 0.3871 | 199 | 20 | 55 | 219 | 254 | 487 |
| nuclei | figure_4 | GP-legacy-config | 0.6386 | 0.2067 | 182 | 31 | 72 | 213 | 254 | 483 |
| nuclei | figure_4 | ImageJ (saved masks) | 0.7647 | 0.1384 | 221 | 35 | 33 | 256 | 254 | 0 |
| nuclei | figure_5 | Raw | 0.1849 | 0.1130 | 83 | 298 | 68 | 381 | 151 | 15 |
| nuclei | figure_5 | GP | 0.5174 | 0.2118 | 119 | 79 | 32 | 198 | 151 | 646 |
| nuclei | figure_5 | QEP q=2 | 0.5174 | 0.2118 | 119 | 79 | 32 | 198 | 151 | 638 |
| nuclei | figure_5 | QEP q=1.5 | 0.5067 | 0.1789 | 113 | 72 | 38 | 185 | 151 | 642 |
| nuclei | figure_5 | GP-legacy-config | 0.7018 | 0.1734 | 120 | 20 | 31 | 140 | 151 | 637 |
| nuclei | figure_5 | ImageJ (saved masks) | 0.5628 | 0.4237 | 121 | 64 | 30 | 185 | 151 | 0 |
| whole_cell | figure_2 | Raw | 0.1384 | 0.0169 | 22 | 118 | 19 | 140 | 41 | 1 |
| whole_cell | figure_2 | GP | 0.1241 | 0.0252 | 18 | 104 | 23 | 122 | 41 | 635 |
| whole_cell | figure_2 | QEP q=2 | 0.1241 | 0.0252 | 18 | 104 | 23 | 122 | 41 | 632 |
| whole_cell | figure_2 | QEP q=1.5 | 0.1293 | 0.0311 | 19 | 106 | 22 | 125 | 41 | 650 |
| whole_cell | figure_2 | GP-legacy-config | 0.1600 | 0.0211 | 20 | 84 | 21 | 104 | 41 | 630 |
| whole_cell | figure_2 | ImageJ (saved masks) | 0.0726 | 0.0105 | 13 | 138 | 28 | 151 | 41 | 0 |
| whole_cell | figure_3 | Raw | 0.4730 | 0.0339 | 245 | 101 | 172 | 346 | 417 | 0 |
| whole_cell | figure_3 | GP | 0.4241 | 0.0534 | 229 | 123 | 188 | 352 | 417 | 358 |
| whole_cell | figure_3 | QEP q=2 | 0.4241 | 0.0534 | 229 | 123 | 188 | 352 | 417 | 359 |
| whole_cell | figure_3 | QEP q=1.5 | 0.4030 | 0.0556 | 218 | 124 | 199 | 342 | 417 | 357 |
| whole_cell | figure_3 | GP-legacy-config | 0.3607 | 0.0626 | 198 | 132 | 219 | 330 | 417 | 356 |
| whole_cell | figure_3 | ImageJ (saved masks) | 0.0877 | 0.0000 | 67 | 347 | 350 | 414 | 417 | 0 |
| whole_cell | figure_4 | Raw | 0.1517 | 0.0459 | 27 | 105 | 46 | 132 | 73 | 1 |
| whole_cell | figure_4 | GP | 0.2394 | 0.0539 | 34 | 69 | 39 | 103 | 73 | 641 |
| whole_cell | figure_4 | QEP q=2 | 0.2394 | 0.0539 | 34 | 69 | 39 | 103 | 73 | 642 |
| whole_cell | figure_4 | QEP q=1.5 | 0.2361 | 0.0409 | 34 | 71 | 39 | 105 | 73 | 651 |
| whole_cell | figure_4 | GP-legacy-config | 0.2460 | 0.0467 | 31 | 53 | 42 | 84 | 73 | 641 |
| whole_cell | figure_4 | ImageJ (saved masks) | 0.1082 | 0.0206 | 29 | 195 | 44 | 224 | 73 | 0 |
| whole_cell | figure_5 | Raw | 0.1196 | 0.0232 | 33 | 187 | 56 | 220 | 89 | 1 |
| whole_cell | figure_5 | GP | 0.0962 | 0.0148 | 30 | 223 | 59 | 253 | 89 | 642 |
| whole_cell | figure_5 | QEP q=2 | 0.0962 | 0.0148 | 30 | 223 | 59 | 253 | 89 | 642 |
| whole_cell | figure_5 | QEP q=1.5 | 0.0980 | 0.0156 | 29 | 207 | 60 | 236 | 89 | 649 |
| whole_cell | figure_5 | GP-legacy-config | 0.1037 | 0.0102 | 28 | 181 | 61 | 209 | 89 | 636 |
| whole_cell | figure_5 | ImageJ (saved masks) | 0.0533 | 0.0025 | 20 | 286 | 69 | 306 | 89 | 0 |

## 8. Dataset-level summary

| dataset | method | n | AP@0.5 mean | sd | min | max | AP@0.75 mean | mean IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| nuclei | GP | 4 | 0.3982 | 0.1614 | 0.1600 | 0.5174 | 0.1417 | 0.4846 |
| nuclei | GP-legacy-config | 4 | 0.3451 | 0.3766 | 0.0000 | 0.7018 | 0.0959 | 0.3287 |
| nuclei | ImageJ (saved masks) | 4 | 0.6813 | 0.2018 | 0.4745 | 0.9231 | 0.2290 | 0.6682 |
| nuclei | QEP q=1.5 | 4 | 0.4461 | 0.2404 | 0.1470 | 0.7263 | 0.1695 | 0.4934 |
| nuclei | QEP q=2 | 4 | 0.3982 | 0.1614 | 0.1600 | 0.5174 | 0.1417 | 0.4846 |
| nuclei | Raw | 4 | 0.3383 | 0.3095 | 0.1785 | 0.8025 | 0.1148 | 0.4820 |
| whole_cell | GP | 4 | 0.2210 | 0.1489 | 0.0962 | 0.4241 | 0.0368 | 0.4718 |
| whole_cell | GP-legacy-config | 4 | 0.2176 | 0.1119 | 0.1037 | 0.3607 | 0.0351 | 0.4605 |
| whole_cell | ImageJ (saved masks) | 4 | 0.0805 | 0.0232 | 0.0533 | 0.1082 | 0.0084 | 0.4245 |
| whole_cell | QEP q=1.5 | 4 | 0.2166 | 0.1376 | 0.0980 | 0.4030 | 0.0358 | 0.4696 |
| whole_cell | QEP q=2 | 4 | 0.2210 | 0.1489 | 0.0962 | 0.4241 | 0.0368 | 0.4718 |
| whole_cell | Raw | 4 | 0.2206 | 0.1687 | 0.1196 | 0.4730 | 0.0300 | 0.4762 |

n=4 images per dataset. No significance is claimed. ImageJ masks are pre-existing `original_ImageJ_masks.tif` produced outside this repo; any manual tuning is unknown.


## 9. Remaining uncertainties

- **n=4 per dataset.** Standard deviations are large (nuclei Raw sd 0.31, GP-legacy sd 0.38). No statistical claim is supportable.

- **min_distance is dataset-specific and development-selected.** It was not validated on independent data, and nuclei md=15 vs whole-cell md=9 may reflect cell size rather than dataset.

- **The historical 0.583864 is only partly explained** (see section 4); two artifacts are missing.

- **GP-legacy-config is unstable**: AP@0.5 = 0.0000 on nuclei_figure_2 (criterion_1 collapsed to all-background on every tile) but 0.7018 on nuclei_figure_5. Its unstandardized float32 path is fragile and should not be used as a reference without that caveat.

- **Thresholding is now the leading suspect.** With markers fixed, remaining errors are dominated by foreground quality: FP counts stay high (e.g. whole_cell_figure_5 GP: 223 FP for 89 GT).

- No claim is made about marginal-likelihood unboundedness at small q; that was not tested here.


## 10. Recommended next experiment (one)

**Replace `criterion_1` with a foreground rule chosen on the two development images, holding markers and cleanup at the now-frozen settings, and evaluate on the same 8 held-out images.**

Rationale: markers are fixed and reproducible, and GP == QEP q=2 exactly, so the pipeline is now a clean instrument. The dominant residual error is foreground quality - `criterion_1` collapsed to all-background on 3/16 development tiles and produced 0 predictions on an entire held-out image, and it is an absolute-scale rule (`percentage x max(image)`) sensitive to each reconstruction's intensity distribution. Until foreground is stable, any GP-vs-QEP reconstruction comparison is measured through a noisy gate. This also directly tests whether reconstruction helps once the gate is fair, which the current data leaves ambiguous (nuclei +0.06, whole-cell +0.00 over raw).

