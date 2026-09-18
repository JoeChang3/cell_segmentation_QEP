# Round 6 — Is the Fast-GP reconstruction's value downstream-dependent?

2026-09-17. Two development images only. Rounds 1–5 preserved.

| item | value |
|---|---|
| Cell_Seg_QEP commit | `a48381de87bf59490f4c8a9ec09670270a45caa3` (branch `chatgpt-review-cellseg-20260917`) |
| reference repo path | `/Users/zchan/eclipse-workspace/cell_segmentation_original` |
| reference repo remote | `https://github.com/UncertaintyQuantification/cell_segmentation.git` |
| reference commit SHA | **`44714c2e0be958fe796a8fd4bdbc220dae3c23dd`** (`44714c2`, 2025-09-04) |
| reference repo modified? | **No** — clean working tree, read-only throughout |
| images | `data/nuclear_test_images/nuclei_figure_1/original_fig.png` (1128x962, GT 330)<br>`data/whole_cell_test_images/whole_cell_figure_1/original_fig.jpg` (600x602, GT 403) |
| R | 4.5.1; **EBImage 4.52.0**, **RobustGaSP 0.6.8**, magick 2.9.1, pracma 2.4.6 |

**EBImage was installed for this task** into the user library
`/Users/zchan/Library/R/arm64/4.5/library` via `BiocManager::install`, from
Bioconductor 3.22 pre-built arm64 binaries. No compilation, no `sudo`, no change
to the system R library. So **B and D are real R runs with real RobustGaSP and
real EBImage — not Python approximations.**

Scripts: `audits/round6_paper_downstream.R`,
`audits/round6_paper_downstream_supplied.R`,
`experiments/real_data/round6_two_by_two.py`.

---

## Primary 2 x 2 — AP@0.5

| | Current corrected downstream | Paper downstream (real R) |
|---|---|---|
| **Raw / NoGP** | **A** nuclei 0.6933 · whole-cell 0.5276 | **B** nuclei 0.4015 · whole-cell 0.4291 |
| **Paper Fast-GP** | **C** nuclei 0.6457 · whole-cell 0.5276 | **D** nuclei **0.6481** · whole-cell **0.5839** |

D uses the **paper-faithful** Fast-GP fit (single L-BFGS-B run in R; the
degenerate plateau solution, `beta1=26.04 beta2=5.679e6 nu=0.0822` on nuclei and
`beta1=3.805e7 beta2=37.99 nu=0.00225` on whole-cell). The robust/multi-start fit
is reported separately below and is **not** in this table.

## Full metrics (same Python evaluator for every cell)

### nuclei_figure_1 — GT = 330

| cell | AP@0.5 | AP@0.75 | TP | FP | FN | #pred | #GT | merge | split | missed | spurious | mean matched IoU | Dice | IoU | precision | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A Raw + current | **0.6933** | **0.3568** | 260 | 45 | 70 | 305 | 330 | 27 | 0 | 70 | 11 | 0.6777 | 0.8393 | 0.7231 | 0.8632 | 0.8167 |
| B NoGP + paper | 0.4015 | 0.1478 | 218 | 213 | 112 | 431 | 330 | 15 | 8 | 112 | **152** | 0.5703 | 0.7619 | 0.6154 | 0.8686 | 0.6786 |
| C FastGP + current | 0.6457 | 0.3397 | 246 | 51 | 84 | 297 | 330 | 34 | 3 | 84 | 9 | 0.6769 | **0.8785** | **0.7833** | 0.9394 | 0.8250 |
| D FastGP + paper | 0.6481 | 0.2840 | 256 | 65 | 74 | 321 | 330 | 22 | 1 | 74 | 23 | 0.6693 | 0.8504 | 0.7397 | 0.9520 | 0.7684 |
| B' published NoGP | 0.1821 | 0.0585 | 106 | 252 | 224 | 358 | 330 | 2 | 6 | 224 | 188 | 0.3175 | 0.5370 | 0.3671 | 0.9489 | 0.3745 |

### whole_cell_figure_1 — GT = 403

| cell | AP@0.5 | AP@0.75 | TP | FP | FN | #pred | #GT | merge | split | missed | spurious | mean matched IoU | Dice | IoU | precision | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A Raw + current | 0.5276 | 0.0795 | 258 | 86 | 145 | 344 | 403 | 67 | 15 | 145 | 12 | 0.5554 | 0.8275 | 0.7058 | 0.9427 | 0.7374 |
| B NoGP + paper | 0.4291 | 0.1303 | 224 | 119 | 179 | 343 | 403 | 84 | 20 | 179 | 36 | 0.5235 | 0.8141 | 0.6864 | 0.7697 | **0.8639** |
| C FastGP + current | 0.5276 | 0.0702 | 258 | 86 | 145 | 344 | 403 | 67 | 19 | 145 | 9 | 0.5517 | 0.8231 | 0.6994 | 0.9462 | 0.7283 |
| D FastGP + paper | **0.5839** | 0.1711 | 275 | 68 | 128 | 343 | 403 | 68 | 12 | 128 | 19 | 0.5918 | 0.8584 | 0.7519 | 0.9113 | 0.8113 |
| B' published NoGP | 0.5660 | **0.2327** | 270 | 74 | 133 | 344 | 403 | 68 | 13 | 133 | 24 | 0.5958 | **0.8716** | **0.7725** | 0.8695 | 0.8738 |

## Runtime (s)

| dataset | cell | reconstruction | thresholding | watershed+cleanup | total |
|---|---|---|---|---|---|
| nuclei | A | 1.77 | — (0.32 downstream) | — | 7.07 |
| nuclei | B | 0.00 | 2.59 | 9.59 | 12.35 |
| nuclei | C | 3.41 | — (0.32 downstream) | — | 3.73 |
| nuclei | D | 17.16 | 2.93 | 1.67 | 21.95 |
| whole_cell | A | 0.15 | — | — | 0.29 |
| whole_cell | B | 0.00 | 1.02 | 0.71 | 1.84 |
| whole_cell | C | 2.50 | — | — | 2.64 |
| whole_cell | D | 8.97 | 1.18 | 0.57 | 10.85 |

Note the watershed time itself: 9.59 s for B on nuclei vs 1.67 s for D — B's
distance map contains far more basins to flood.

## Step 10 — interaction decomposition (n = 1 per dataset; descriptive only)

| dataset | metric | A | B | C | D | C−A | D−B | B−A | D−C | **interaction (D−B)−(C−A)** |
|---|---|---|---|---|---|---|---|---|---|---|
| nuclei | AP@0.5 | 0.6933 | 0.4015 | 0.6457 | 0.6481 | **−0.0477** | **+0.2466** | −0.2919 | +0.0024 | **+0.2943** |
| nuclei | AP@0.75 | 0.3568 | 0.1478 | 0.3397 | 0.2840 | −0.0171 | +0.1362 | −0.2090 | −0.0557 | **+0.1533** |
| nuclei | Dice | 0.8393 | 0.7619 | 0.8785 | 0.8504 | +0.0392 | +0.0884 | −0.0774 | −0.0281 | +0.0493 |
| whole_cell | AP@0.5 | 0.5276 | 0.4291 | 0.5276 | 0.5839 | **+0.0000** | **+0.1547** | −0.0985 | +0.0563 | **+0.1547** |
| whole_cell | AP@0.75 | 0.0795 | 0.1303 | 0.0702 | 0.1711 | −0.0093 | +0.0408 | +0.0508 | +0.1009 | **+0.0501** |
| whole_cell | Dice | 0.8275 | 0.8141 | 0.8231 | 0.8584 | −0.0044 | +0.0443 | −0.0134 | +0.0353 | +0.0487 |

**The interaction is positive on every metric and both datasets**, and on AP@0.5
it is large (+0.2943, +0.1547) relative to the main effects.

## Mechanism — why the interaction exists

The pre-cleanup watershed label counts make it concrete:

| dataset | cell | labels before cleanup | labels after cleanup | spurious @0.5 |
|---|---|---|---|---|
| nuclei | B (raw + paper) | **2846** | 431 | 152 |
| nuclei | D (FastGP + paper) | **484** | 321 | 23 |
| whole_cell | B | 577 | 343 | 36 |
| whole_cell | D | 474 | 343 | 19 |

The paper's downstream calls `EBImage::watershed(dist_map)` with defaults
`tolerance = 1, ext = 1` on the **positive** distance map, and EBImage performs
its **own** seed detection. On a raw binary foreground the distance map is
pixel-noisy, producing thousands of shallow local maxima — 2846 basins on nuclei
against 330 true cells. The Fast-GP reconstruction removes exactly that
pixel-scale noise, so the same seeder finds 484 basins instead.

Our current downstream never had this problem: it supplies **explicit**
`peak_local_max` markers with `min_distance = 15/9`, which suppresses spurious
seeds by construction. So the GP's noise-removal is **redundant** there, and its
only remaining effect is the cost it carries — more merged cells (nuclei merge
27 → 34, AP@0.5 −0.0477).

**In one sentence: the Fast-GP reconstruction is a substitute for explicit
marker control. It is valuable precisely when the downstream relies on
automatic, tolerance-based seeding, and near-worthless once robust markers are
supplied.**

## Secondary diagnostic — robust/multi-start fit + paper downstream

Kept out of the primary table because it is not the published algorithm.

| dataset | AP@0.5 | AP@0.75 | Dice | TP | FP | FN | #pred | merge | split | vs D (paper fit) | vs B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nuclei | **0.7135** | 0.2998 | 0.8533 | 269 | 47 | 61 | 316 | 18 | 0 | **+0.0654** | +0.3121 |
| whole_cell | 0.5258 | 0.1509 | 0.8346 | 255 | 82 | 148 | 337 | 72 | 16 | −0.0581 | +0.0967 |

Mixed: repairing the optimizer degeneracy helps nuclei (and 0.7135 is the single
best AP@0.5 anywhere in this study, above A = 0.6933) but hurts whole-cell. It
does not change the qualitative 2 x 2 conclusion — `D_robust > B` on both.

## Step 11 — historical reproduction check

The reference repo ships saved IoU matrices. AP recomputed from them with the
paper's own `compute_ap_from_ious` (verified identical in definition to our
evaluator: per-GT-row argmax, `>=` inclusive, AP = TP/(TP+FP+FN)):

| dataset | historical file | shape | AP@0.5 | AP@0.75 | our new run | Δ | #pred new vs hist |
|---|---|---|---|---|---|---|---|
| nuclei | `ious_gp.csv` | 330x317 | 0.6547 | 0.2863 | D = 0.6481 | **−0.0066** | 321 vs 317 |
| nuclei | `ious_noGP.csv` | 330x358 | **0.1821** | 0.0585 | B' = **0.1821** | **−0.0000** | **358 vs 358** |
| nuclei | `ious_noImageGP.csv` | 330x545 | 0.3318 | 0.1261 | not run | — | — |
| nuclei | `ious_imagej.csv` | 330x481 | 0.4586 | 0.2515 | — | — | — |
| whole_cell | `ious_gp.csv` | 403x349 | 0.5765 | 0.1695 | D = 0.5839 | **+0.0073** | 343 vs 349 |
| whole_cell | `ious_imagej.csv` | 403x410 | 0.3416 | 0.0025 | — | — | — |

**The published NoGP arm reproduces exactly** — AP 0.1821 vs 0.1821 and #pred
358 vs 358 — which is expected, because that arm contains no GP and therefore no
optimizer. The GP arms reproduce to within 0.007 AP. The residual is attributable
to the optimizer: the profiled likelihood has a flat `beta → ∞` plateau on which
`beta` is unidentifiable (documented in
`audits/PAPER_FAST_GP_WIRING_REPORT.md`), so different runs land on different
plateau points with identical objective but predictive means differing at ~1e-6,
which is enough to flip a handful of `percentage * max(tile)` threshold
decisions. **Algorithm reproduction: confirmed. Exact historical number
reproduction: confirmed for NoGP, within 0.007 AP for GP.**

## Validation performed before any number was trusted

Evaluator sanity checks (all PASS): GT vs itself AP@0.5 = AP@0.75 =
1.000000000000; two exact squares AP = 1.0; one of two detected AP = 0.5 with
TP1/FP0/FN1; merge detector returns `merged=1` on a blob spanning two GT objects;
foreground self-Dice = self-IoU = 1.0.

Orientation alignment (Step 6), verified by arrays rather than by eye:

- The drivers' `t(integer_mask)[, nrow:1][, ncol:1]` was shown numerically to
  equal **plain `t()`** — the two flips cancel (`identical(final, t(M)) TRUE`).
- `EBImage::readImage` returns `(width, height)` = 962x1128, so after `t()` the
  GT is `(height, width)` = natural, matching our Python loader.
- The R-driver GT foreground is **element-wise identical** to the Python-loaded
  GT on both images, with the same label count (330, 403).
- Candidate orientations of the R label array scored by foreground IoU against
  the Python GT: nuclei identity **0.7411** vs flipud 0.1019 / fliplr 0.1100 /
  rot180 0.1138; whole-cell identity **0.7569** vs 0.1836 / 0.1904 / 0.1863.
  Identity wins by a wide margin, asserted in code.
- A and C were re-scored from the Round-5 stored masks and asserted to reproduce
  the Round-5 AP to `< 1e-12` before use.

---

# Answers

## 1. What exactly is the original paper's NoGP arm?

`Nuclear_Real_Analysis/GP_vs_NoGP/Segmentation_Functions_NoGP (no conflict function names).R`,
function `generate_GP_Masks_test2`, called from
`Nuclear_Data_Generate_IoU (GP vs no GP).R:213`. It reproduces Figures S1/S2 of
the supplement (README line 16).

1. **Does NoGP send raw pixels straight into the same thresholding?** Yes for the
   *tiles*. `generate_GP_Masks_test2` is byte-identical to
   `generate_GP_Masks_test` except one line: `predmean_mat <- img_matrix` instead
   of calling `separable_GP`. Same `get_proportion` tiling, same magick raw/255
   channel-1 intensities, same outlier handling, same stitching.
2. **Any other smoothing?** No. None at all.
3. **Does threshold selection still use a GP / RobustGaSP?** **This is where the
   two ablations differ, and it is the critical detail.** There are *two*:
   - `Segmentation_Functions_NoGP`: the `rgasp(...)` and `predict(...)` lines in
     `criterion_1_2` are **commented out**, and so is
     `diff_pixel_counts <- smoothed$mean`. So "no GP" means **no image GP AND no
     RobustGaSP threshold GP** — the criterion curve is used raw.
   - `Segmentation_Functions_NoImageGP`: the rgasp lines are **active**. This one
     removes only the image GP. The `diff` between the two files is exactly those
     four lines and a comment — nothing else.
4. **Different parameters or preprocessing?** No — `delta = 0.01`,
   `nugget = TRUE`, identical tiling and intensity handling.
5. **Same watershed/cleanup?** Same watershed (`distmap` + `EBImage::watershed`).
   **Different cleanup**: `eliminate_small_areas2(GP_masks)` takes *no* size
   argument and uses `mean_obj_size * 0.15` (interior) and `mean_obj_size * 0.05`
   (boundary-touching), whereas the GP arm uses
   `eliminate_small_areas(GP_masks, remove_size_threshold = 50)` with `50` and
   `50/5 = 10`. **The 15% / 5% values are confirmed in code**, relative to the
   mean object area.
6. **Whole-cell NoGP?** **It does not exist.** `NoGP` appears only under
   `Nuclear_Real_Analysis/GP_vs_NoGP/`; `Whole_Cell_Data_Generate_IoU.R` sources
   only `src/Modified_Functions_RGasp.R` and calls `generate_GP_Masks_test`.

**Therefore the paper's own published GP-vs-NoGP comparison is confounded three
ways at once**: image GP, RobustGaSP criterion smoothing, and the cleanup rule
all change together. It cannot serve as the 2 x 2 control. Our **B** is a
constructed ablation that changes only `predmean_mat <- img_matrix`; the
published arm is reported as **B'**. The size of the confound is visible: on
nuclei B = 0.4015 but B' = 0.1821 (the confounds cost a further 0.2194 AP), while
on whole-cell B = 0.4291 and B' = 0.5660 (they *help* by 0.1369). The confound
acts in **opposite directions** on the two datasets.

Our Python `Raw` (A) is also **not** the paper's NoGP: A uses a single global Li
threshold with explicit peak markers; B uses per-tile `percentage * max(tile)`
criterion_1 with RobustGaSP and automatic EBImage seeding. A ≠ B by construction,
and indeed B − A = −0.2919 (nuclei) and −0.0985 (whole-cell).

## 2. Was the original threshold method reproduced faithfully?

**YES.** Real R, real `RobustGaSP 0.6.8`, no substitution. Traced and reproduced
exactly:

- candidates are **proportions of the per-tile maximum**:
  `threshold_value <- percentage * max(mat)`, grid `seq(0, 1, by = 0.01)` (101
  candidates) — not absolute intensities, not quantiles.
- evaluated **per tile**, on that tile's own `predmean_mat`.
- foreground count per candidate via `sum(mat > threshold_value)`; difference
  curve `diff_pixel_counts <- abs(diff(pixel_counts))` (100 points).
- smoothed by `rgasp(percentages[-1], diff_pixel_counts, nugget.est = TRUE)`
  then `predict(diff_mod, testing_input = as.matrix(percentages[-1]))$mean`,
  which **replaces** `diff_pixel_counts`.
- selection: walk forward from `which.max` until
  `|d[i] − d[i−1]| < 0.05 * sd(d)`; chosen value is
  `percentages[stable_index + 1]`.
- fallback: if no stable point is found, the tile becomes **all background** and
  the recorded percentage is 1.
- outlier tiles: `|n_connected − mean| > 2 * sd` → re-threshold at the mean of
  the non-outlier percentages; if that yields > 99% foreground, revert to all
  background and set the percentage to 1. Observed: 1 outlier tile (nuclei D,
  whole-cell B and D), 2 outlier tiles (nuclei B), 0 reverted.
- parameters are **tile-specific** (each tile selects its own percentage); the
  only shared quantity is the GP `(beta1, beta2, nu)` triple from tile (1,1).

Auditable artefacts saved for tile 1 of both datasets and both arms in
`threshold_diagnostics/`: the candidate grid, the raw criterion curve, the
RobustGaSP fitted mean with 95% bands, the selected percentage and its absolute
threshold, the fitted `rgasp` `beta_hat` / nugget / `sigma2`, plus the resulting
binary tile and the tile's predmean. RobustGaSP's own optimizer trace is in
`logs/r_nuclei.log` / `logs/r_whole_cell.log`.

## 3. Was EBImage watershed reproduced faithfully?

**YES.** `EBImage 4.52.0`, the genuine Bioconductor implementation.

- function: `EBImage::watershed(x, tolerance = 1, ext = 1)` — called as
  `watershed(dist_map)`, i.e. **both defaults**.
- input: `distmap(as.Image(combined_thresholded1))` with
  `metric = "euclidean"` (the default).
- **the POSITIVE distance map is passed, not negated.** This differs from our
  current pipeline, which uses `-distance_transform_edt`. EBImage floods from
  maxima of the positive map; skimage floods from minima of the negated map.
- seeds: EBImage performs its **own internal** tolerance-based seed detection.
  No markers are supplied anywhere in the paper code.
- label output: `segmented_image@.Data`, numeric, background 0, contiguous
  positive integers.
- border handling: none — no border exclusion at the watershed stage.

No Python approximation is used or labelled as EBImage anywhere in this round.

## 4. Was the cleanup reproduced faithfully?

**YES**, for both variants, traced from source rather than prose:

- GP arm and our B: `eliminate_small_areas(GP_masks, size_threshold)` with
  `remove_size_threshold = 50` (the `generate_GP_Masks_test` default).
  Rules: `area < 50 && !on_boundary` → remove; `on_boundary && area < 50/5 = 10`
  → remove. `on_boundary` = the label touches row 1, row n, column 1, or
  column n.
- published NoGP arm (B'): `eliminate_small_areas2(GP_masks)`, **no size
  argument**, using `mean_obj_size = mean(label_counts)`:
  `area < mean_obj_size*0.15 && !on_boundary` → remove;
  `on_boundary && area < mean_obj_size*0.05` → remove. **15% / 5% confirmed.**
- morphology: **none** before or after watershed in either variant.
- label renumbering: **none** — labels are zeroed in place, so the final mask has
  gaps in its label sequence. Our evaluator keys on unique positive values, so
  this is harmless; verified via the GT-vs-itself and orientation checks.

Masks **before and after** cleanup are saved: `watershed_precleanup/*.npz`
carries `labels_precleanup` alongside `n_labels_precleanup` /
`n_labels_final`, and the raw R dumps are in
`r_outputs/<ds>/*_watershed_precleanup.csv`.

## 5. Are B and D true paper-downstream runs, or approximations?

**True paper-downstream runs.** Every stage executed in R with the reference
repo's own sourced functions, real `RobustGaSP`, and real `EBImage`. Nothing in
B or D is a Python reimplementation. The only Python involvement is (i) reading
the R CSV dumps and (ii) scoring with the shared evaluator, which is required so
that A/B/C/D are comparable.

Two honest qualifications:

1. **B is a constructed ablation, not a published arm** — required, because the
   paper's own NoGP changes three things at once (answer 1) and there is no
   whole-cell NoGP at all. B changes exactly one line of
   `generate_GP_Masks_test`. This is stated everywhere it appears.
2. **D's GP fit is the paper-faithful single-run L-BFGS-B result**, which lands
   on the degenerate plateau on both images. That is what the published code
   does, so it is the right choice for D — but it means D inherits an optimizer
   pathology, quantified in the secondary table.

## 6. On nuclei: is D > B? Is C > A?

- **D > B: YES, decisively.** 0.6481 vs 0.4015 = **+0.2466** AP@0.5
  (AP@0.75 +0.1362, Dice +0.0884). TP 256 vs 218, spurious 23 vs 152.
- **C > A: NO.** 0.6457 vs 0.6933 = **−0.0477** AP@0.5 (AP@0.75 −0.0171).
  Foreground Dice does improve (+0.0392), but instance AP does not — merges rise
  27 → 34.

## 7. On whole-cell: is D > B? Is C > A?

- **D > B: YES.** 0.5839 vs 0.4291 = **+0.1547** AP@0.5 (AP@0.75 +0.0408,
  Dice +0.0443). TP 275 vs 224.
- **C > A: NO — exactly zero.** 0.5276 vs 0.5276 = **+0.0000**, with identical
  TP/FP/FN (258/86/145). AP@0.75 is slightly worse (−0.0093). As Round 5 noted,
  `paper_fast_gp` barely smooths this image (correlation to raw 0.9997), so it
  hands the current downstream almost the same input.

## 8. Does the sign of the reconstruction effect depend on the downstream pipeline?

**Yes, unambiguously, on both datasets.**

| dataset | C − A (current downstream) | D − B (paper downstream) | sign flips? |
|---|---|---|---|
| nuclei | **−0.0477** | **+0.2466** | **yes: negative → positive** |
| whole_cell | **+0.0000** | **+0.1547** | **yes: null → positive** |

The interaction `(D−B) − (C−A)` is **+0.2943** and **+0.1547** on AP@0.5, and
positive on AP@0.75 and Dice as well. The main effect of reconstruction is
essentially zero-or-negative under our downstream and clearly positive under the
paper's.

So the answer to the task's core question is **yes — the value of the original
Fast-GP reconstruction is dependent on the original paper's downstream
segmentation pipeline.** Mechanistically it substitutes for explicit marker
control (see *Mechanism* above): it is worth a lot when the watershed must find
its own seeds by tolerance on a noisy distance map, and worth nothing once
`peak_local_max` markers with a minimum separation are supplied.

## 9. Does paper Fast-GP + paper downstream reproduce a meaningful advantage over the corresponding NoGP control?

**Yes.** Against the clean constructed control: **+0.2466** AP@0.5 on nuclei and
**+0.1547** on whole-cell, with consistent gains in AP@0.75, Dice, IoU, TP and
mean matched IoU, and large reductions in spurious detections (152 → 23 on
nuclei). Against the paper's own published NoGP arm on nuclei the gap is larger
still (0.6481 vs 0.1821 = +0.4660), but roughly half of that is attributable to
the two non-GP confounds rather than to the reconstruction.

Two caveats that must travel with this result:

- **n = 1 image per dataset.** No variance, no significance, no generalization.
- **It is an advantage *within* the paper's pipeline, not an absolute one.** On
  nuclei, A (Raw + our corrected downstream) still scores 0.6933 > D's 0.6481, so
  the paper's full pipeline does not beat simply replacing its downstream. On
  whole-cell, D (0.5839) *is* the best of the four primary cells.

## 10. What discrepancy remains between this run and the historical paper setup?

Small and explained:

1. **GP arms differ by ≤ 0.007 AP** from the historical saved IoU matrices
   (nuclei −0.0066, whole-cell +0.0073; #pred 321 vs 317 and 343 vs 349). Cause:
   the profiled likelihood's flat `beta → ∞` plateau makes `beta` unidentifiable,
   so R's L-BFGS-B can terminate at different plateau points with identical
   objective, shifting predictive means at ~1e-6 — enough to flip a few
   `percentage * max(tile)` decisions. The **NoGP arm, which has no optimizer,
   reproduces exactly** (0.1821/0.1821, 358/358 predictions), which corroborates
   this explanation.
2. **The historical masks themselves are not in the repo** — only the IoU
   matrices, boundary PNGs and one `original_fig_seg.npy`. So a mask-level
   element-wise comparison against the historical run is not possible; the
   comparison is at the AP/IoU-matrix level. Stated rather than worked around.
3. **`ious_noImageGP.csv` exists for nuclei (AP@0.5 = 0.3318)** but we did not
   run that third ablation; it is listed for completeness.
4. **No whole-cell NoGP exists historically**, so whole-cell B has no published
   counterpart to check against — by design it is our constructed ablation.
5. **Separate the two kinds of reproduction**: (A) *algorithm* reproduction is
   confirmed — threshold rule, RobustGaSP smoothing, EBImage watershed and both
   cleanup variants all run as written; (B) *historical numerical* reproduction
   is exact for NoGP and within 0.007 AP for GP.

Nothing here contradicts the published paper. The finding is one of
**attribution**: the reconstruction's contribution is real inside the paper's
pipeline, and it is the paper's automatic seeding — not the reconstruction — that
creates the need for it.
