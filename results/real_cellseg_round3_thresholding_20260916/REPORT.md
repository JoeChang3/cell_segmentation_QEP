# Round 3: foreground thresholding

Run directory: `results/real_cellseg_round3_thresholding_20260916`

Scope: thresholding only. GP/QEP models, kernels, q values, optimization, watershed markers (peak_local_max, min_distance nuclei=15 / whole-cell=9), cleanup (eliminate_small_areas 50px) and evaluation (AP = TP/(TP+FP+FN)) are unchanged from round 2. Reconstructions are the cached round-2 arrays; no GP/QEP model was refitted except to regenerate the two DEVELOPMENT images, whose round-1 arrays predated the round-2 determinism fix.

## Headline

- **Thresholding was a major confounder on nuclei and a non-issue on whole-cell.** Held-out nuclei AP@0.5 for Raw went 0.3383 -> **0.7543**; whole-cell Raw 0.2206 -> 0.2248.

- **Raw now beats GP and QEP on BOTH datasets.** Reconstruction does not help once foreground extraction is fixed.

- **q=1.5 shows no advantage over q=2 that holds across datasets.** Nuclei 2/4 images, mean -0.0146. Whole-cell 4/4 images but only +0.0286 mean.

- **GP == QEP q=2 in 8/8 held-out images.**


## 1. What criterion_1 does and why it failed

`criterion_1` (Modified_Functions_RGasp.py:53) sweeps p = 0.00..1.00, thresholds at `p * nanmax(image)`, counts foreground pixels, Gaussian-smooths the |diff| of that curve, takes its argmax, then walks forward to the first index where consecutive smoothed diffs change by less than `0.05 * std(diff_curve)`. Four defects, all confirmed on real data:

| defect | evidence |
|---|---|
| anchored to `nanmax`, a single-pixel order statistic | the whole sweep rescales with one bright pixel |
| grid spans [0, max], assuming background near zero | nuclei_figure_1 has range 105..255, so 41% of the p-grid selects the entire image; GP reconstructions run NEGATIVE (to -75 on whole_cell_figure_3), which the grid never covers |
| systematic under-detection on nuclei | mean precision 0.920 but mean recall only 0.574; keeps roughly the top 4-10% of pixels where ground-truth foreground is 9-22% |
| silent all-background default when the stability walk never fires | round-2 held-out nuclei_figure_2 / gp_legacy produced **0 predictions, AP = 0.0000** |

The collapse is a **per-tile** phenomenon: round 2 applied criterion_1 tile by tile. Applied globally to a stitched reconstruction it does not collapse on any of the 40 image/arm cases, so per-tile criterion_1 is reported separately as the round-2 baseline. Global-vs-per-tile is therefore a second difference between the rounds besides the rule itself, and is stated as such.


Step-6 controls: **criterion_1 is the only rule tested that is NOT affine-invariant** (`fraction * max` shifts inconsistently under I -> aI + b). li, otsu, yen, triangle, quantile and robust_mad are all scale- AND affine-invariant. All rules are exactly deterministic: identical threshold, identical mask, equivalent partition on repeated runs.


## 2. Candidate rules tested

criterion_1, Otsu, Li, Yen, Triangle, quantile p in {0.80, 0.85, 0.90, 0.95}, robust MAD `(I - median) / (1.4826 * MAD) >= c` for c in {1.5, 2, 3, 4}. 13 candidates, no larger search. Local/adaptive thresholding was deliberately NOT added: the reconstructions show no illumination gradient that defeats a global rule, and adding it only to enlarge the comparison was not warranted.


## 3. Development-only selection

Selection used the two development images only, aggregating foreground Dice across ALL FOUR reconstruction arms (never per arm, never per image). Rules collapsing on any arm are ineligible.

| dataset | rule | Dice mean | Dice min (worst arm) | precision | recall |
|---|---|---:|---:|---:|---:|
| nuclei | li | 0.8522 | 0.8393 | 0.864 | 0.841 |
| nuclei | quantile_p0.8 | 0.8485 | 0.8387 | 0.900 | 0.802 |
| nuclei | otsu | 0.8426 | 0.8287 | 0.915 | 0.781 |
| nuclei | yen | 0.8242 | 0.8210 | 0.938 | 0.735 |
| nuclei | *criterion_1 (baseline)* | 0.4408 | 0.3287 | 0.993 | 0.291 |
| whole_cell | robust_mad_c1.5 | 0.8123 | 0.8066 | 0.949 | 0.710 |
| whole_cell | criterion_1 | 0.8108 | 0.8026 | 0.949 | 0.708 |
| whole_cell | li | 0.8031 | 0.7881 | 0.952 | 0.695 |
| whole_cell | triangle | 0.7993 | 0.7366 | 0.855 | 0.790 |
| whole_cell | *criterion_1 (baseline)* | 0.8108 | 0.8026 | 0.949 | 0.708 |

Automated winners: li (nuclei), robust_mad_c1.5 (whole-cell). **Override: li for BOTH datasets** - parameter-free, within 0.0092 mean Dice of the whole-cell winner, and with one development image per dataset a 0.009 margin does not justify two rules. Note that on whole-cell criterion_1 was already essentially best (0.8108 vs 0.8123), so thresholding was never the whole-cell bottleneck.


## 4-5. Frozen rule

**nuclei: Li (global, parameter-free). whole-cell: Li (global, parameter-free).** Written to `selected_threshold_config.json` before any round-3 held-out score was computed.


## 6. Held-out foreground results

| dataset | arm | Dice | IoU | precision | recall | fg fraction | GT fg fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| nuclei | raw | 0.8597 | 0.7549 | 0.922 | 0.809 | 11.0% | 12.3% |
| nuclei | gp | 0.8215 | 0.6998 | 0.795 | 0.851 | 13.0% | 12.3% |
| nuclei | qep_q2 | 0.8215 | 0.6998 | 0.795 | 0.851 | 13.0% | 12.3% |
| nuclei | qep_q1.5 | 0.8125 | 0.6880 | 0.772 | 0.861 | 13.5% | 12.3% |
| whole_cell | raw | 0.7484 | 0.5985 | 0.961 | 0.614 | 18.2% | 28.8% |
| whole_cell | gp | 0.7146 | 0.5582 | 0.961 | 0.574 | 16.7% | 28.8% |
| whole_cell | qep_q2 | 0.7146 | 0.5582 | 0.961 | 0.574 | 16.7% | 28.8% |
| whole_cell | qep_q1.5 | 0.7258 | 0.5713 | 0.959 | 0.588 | 17.2% | 28.8% |

## 7. Held-out segmentation results

| Dataset | Image | Method | AP@0.5 | AP@0.75 | TP@0.5 | FP@0.5 | FN@0.5 | #pred | #GT |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| nuclei | 2 | Raw | 0.7805 | 0.5052 | 64 | 5 | 13 | 69 | 77 |
| nuclei | 2 | GP | 0.4565 | 0.0635 | 42 | 15 | 35 | 57 | 77 |
| nuclei | 2 | QEP q=2 | 0.4565 | 0.0635 | 42 | 15 | 35 | 57 | 77 |
| nuclei | 2 | QEP q=1.5 | 0.3895 | 0.0312 | 37 | 18 | 40 | 55 | 77 |
| nuclei | 2 | ImageJ (saved masks) | 0.9231 | 0.2295 | 72 | 1 | 5 | 73 | 77 |
| nuclei | 3 | Raw | 0.6655 | 0.3011 | 183 | 24 | 68 | 207 | 251 |
| nuclei | 3 | GP | 0.4700 | 0.1395 | 141 | 49 | 110 | 190 | 251 |
| nuclei | 3 | QEP q=2 | 0.4700 | 0.1395 | 141 | 49 | 110 | 190 | 251 |
| nuclei | 3 | QEP q=1.5 | 0.4570 | 0.1224 | 138 | 51 | 113 | 189 | 251 |
| nuclei | 3 | ImageJ (saved masks) | 0.4745 | 0.1245 | 186 | 141 | 65 | 327 | 251 |
| nuclei | 4 | Raw | 0.8141 | 0.4353 | 219 | 15 | 35 | 234 | 254 |
| nuclei | 4 | GP | 0.6857 | 0.3447 | 192 | 26 | 62 | 218 | 254 |
| nuclei | 4 | QEP q=2 | 0.6857 | 0.3447 | 192 | 26 | 62 | 218 | 254 |
| nuclei | 4 | QEP q=1.5 | 0.6918 | 0.3409 | 193 | 25 | 61 | 218 | 254 |
| nuclei | 4 | ImageJ (saved masks) | 0.7647 | 0.1384 | 221 | 35 | 33 | 256 | 254 |
| nuclei | 5 | Raw | 0.7572 | 0.5916 | 131 | 22 | 20 | 153 | 151 |
| nuclei | 5 | GP | 0.7168 | 0.3562 | 124 | 22 | 27 | 146 | 151 |
| nuclei | 5 | QEP q=2 | 0.7168 | 0.3562 | 124 | 22 | 27 | 146 | 151 |
| nuclei | 5 | QEP q=1.5 | 0.7326 | 0.3304 | 126 | 21 | 25 | 147 | 151 |
| nuclei | 5 | ImageJ (saved masks) | 0.5628 | 0.4237 | 121 | 64 | 30 | 185 | 151 |
| whole_cell | 2 | Raw | 0.1333 | 0.0200 | 18 | 94 | 23 | 112 | 41 |
| whole_cell | 2 | GP | 0.1429 | 0.0141 | 18 | 85 | 23 | 103 | 41 |
| whole_cell | 2 | QEP q=2 | 0.1429 | 0.0141 | 18 | 85 | 23 | 103 | 41 |
| whole_cell | 2 | QEP q=1.5 | 0.1525 | 0.0149 | 18 | 77 | 23 | 95 | 41 |
| whole_cell | 2 | ImageJ (saved masks) | 0.0726 | 0.0105 | 13 | 138 | 28 | 151 | 41 |
| whole_cell | 3 | Raw | 0.4705 | 0.0266 | 247 | 108 | 170 | 355 | 417 |
| whole_cell | 3 | GP | 0.3089 | 0.0066 | 181 | 169 | 236 | 350 | 417 |
| whole_cell | 3 | QEP q=2 | 0.3089 | 0.0066 | 181 | 169 | 236 | 350 | 417 |
| whole_cell | 3 | QEP q=1.5 | 0.3614 | 0.0147 | 202 | 142 | 215 | 344 | 417 |
| whole_cell | 3 | ImageJ (saved masks) | 0.0877 | 0.0000 | 67 | 347 | 350 | 414 | 417 |
| whole_cell | 4 | Raw | 0.1585 | 0.0547 | 29 | 110 | 44 | 139 | 73 |
| whole_cell | 4 | GP | 0.2039 | 0.0578 | 31 | 79 | 42 | 110 | 73 |
| whole_cell | 4 | QEP q=2 | 0.2039 | 0.0578 | 31 | 79 | 42 | 110 | 73 |
| whole_cell | 4 | QEP q=1.5 | 0.2378 | 0.0536 | 34 | 70 | 39 | 104 | 73 |
| whole_cell | 4 | ImageJ (saved masks) | 0.1082 | 0.0206 | 29 | 195 | 44 | 224 | 73 |
| whole_cell | 5 | Raw | 0.1370 | 0.0132 | 37 | 181 | 52 | 218 | 89 |
| whole_cell | 5 | GP | 0.0729 | 0.0032 | 21 | 199 | 68 | 220 | 89 |
| whole_cell | 5 | QEP q=2 | 0.0729 | 0.0032 | 21 | 199 | 68 | 220 | 89 |
| whole_cell | 5 | QEP q=1.5 | 0.0912 | 0.0101 | 25 | 185 | 64 | 210 | 89 |
| whole_cell | 5 | ImageJ (saved masks) | 0.0533 | 0.0025 | 20 | 286 | 69 | 306 | 89 |

### Dataset means (n=4 held-out images)

| dataset | method | AP@0.5 | sd | AP@0.75 | mean matched IoU |
|---|---|---:|---:|---:|---:|
| nuclei | Raw | 0.7543 | 0.0637 | 0.4583 | 0.7078 |
| nuclei | GP | 0.5822 | 0.1381 | 0.2260 | 0.5872 |
| nuclei | QEP q=2 | 0.5822 | 0.1381 | 0.2260 | 0.5872 |
| nuclei | QEP q=1.5 | 0.5677 | 0.1699 | 0.2062 | 0.5746 |
| nuclei | ImageJ (saved masks) | 0.6813 | 0.2018 | 0.2290 | 0.6682 |
| whole_cell | Raw | 0.2248 | 0.1641 | 0.0286 | 0.4854 |
| whole_cell | GP | 0.1821 | 0.1000 | 0.0204 | 0.4517 |
| whole_cell | QEP q=2 | 0.1821 | 0.1000 | 0.0204 | 0.4517 |
| whole_cell | QEP q=1.5 | 0.2107 | 0.1170 | 0.0233 | 0.4647 |
| whole_cell | ImageJ (saved masks) | 0.0805 | 0.0232 | 0.0084 | 0.4245 |

## 8. Round 2 vs Round 3

| question | answer |
|---|---|
| 1. catastrophic cases eliminated? | Among the four deployable arms neither round produced an all-background full image. The round-2 collapse was confined to `gp_legacy` (nuclei_figure_2, 0 predictions); li rescues that foreground (0.0% -> 73.7%) and removes the collapse (1 -> 0), but that image still scores 0 because the gp_legacy reconstruction is itself degenerate. |
| 2. foreground improved? | Nuclei yes, substantially: Dice +0.34 (raw), +0.18 (gp), +0.11 (q=1.5). Whole-cell slightly WORSE, -0.04. |
| 3. variance reduced? | Nuclei yes (raw Dice sd 0.2206 -> 0.0301). Whole-cell slightly increased (GP 0.0376 -> 0.0553). |
| 4. ranking more stable? | Yes on nuclei: AP sd fell for every arm (raw 0.3095 -> 0.0637). |
| 5. does reconstruction beat Raw? | **No.** Raw best on both: nuclei GP -0.1721, q=1.5 -0.1866 vs Raw; whole-cell GP -0.0427, q=1.5 -0.0141. |
| 6. repeatable q=1.5 advantage? | **No.** Nuclei 2/4, mean -0.0146. Whole-cell 4/4 but mean only +0.0286 where all arms sit at 0.18-0.22. |
| 7. GP == q=2? | Yes, identical AP@0.5 in 8/8 held-out images. |
| 8. dominant remaining failure | Under-detection. Nuclei GP FN=234 vs FP=112; whole-cell GP FN=369, spurious=271. Merges (60-96 / 61-65) exceed splits (6-23) everywhere. |

## 9. Does reconstruction help relative to Raw?

**No.** With thresholding fixed, Raw is the best arm on both datasets, decisively on nuclei (0.7543 vs 0.5822 for GP/q=2), and Raw now also beats ImageJ on nuclei (0.7543 vs 0.6813). The round-2 impression that GP helped nuclei (0.3982 vs raw 0.3383) looks like an artifact of criterion_1: GP smoothing compresses the intensity distribution into a range where `fraction * max` misbehaves less, so the smoothed arms were penalized less than the noisier raw image. With a scale- and affine-invariant rule the advantage disappears and reverses.


## 10. Does q=1.5 beat q=2 / GP?

**Not repeatably.** The one consistent signal is whole-cell, where q=1.5 beats q=2 on 4/4 images - but the mean gain is +0.0286 AP where every arm scores 0.18-0.22, with n=4. On nuclei the effect reverses (2/4, mean -0.0146). Because GP == q=2 exactly, this IS a genuine q effect rather than numerical noise, but it is small, dataset-dependent, and does not close the gap to Raw on either dataset.


## 11. Dominant remaining failure mode

**Under-detection (false negatives), not instance separation.** Held-out nuclei foreground recall averages 0.843 at precision 0.821; whole-cell recall 0.587 at precision 0.960. Every rule tested, including the selected one, is precision-heavy and recall-light. Whole-cell additionally carries a large spurious count (243-283), so its errors are two-sided.


## 12. Recommended next experiment (one)

**Run the frozen round-3 pipeline on the ground-truth foreground (oracle-foreground diagnostic) for the held-out WHOLE-CELL images and compare against the real-foreground result.**

Rationale: nuclei is now largely handled by Raw + Li + peak markers (AP@0.5 0.7543, above ImageJ). Whole-cell is not (0.22 at best) and did not improve from the threshold fix, so its bottleneck lies elsewhere. The oracle-foreground contrast separates 'foreground extraction is still wrong' from 'even perfect foreground cannot be separated into these cells', which decides whether the next effort belongs in reconstruction, thresholding, or instance separation. It is a diagnostic, not a deployable method, and needs no model change.


**Not recommended on this evidence:** a more complex QEP model. Reconstruction currently loses to doing nothing on both datasets, so added model capacity has no demonstrated deficit to address.

