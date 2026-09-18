# Audit note — reproducing the original paper's downstream segmentation in R

2026-09-17. Companion to `audits/PAPER_FAST_GP_WIRING_REPORT.md` (reconstruction
stage) and `results/real_cellseg_round6_paper_downstream_20260917/REPORT.md`
(the 2 x 2 experiment).

Reference repo, **read-only, clean working tree, not in this project's index**:

```
/Users/zchan/eclipse-workspace/cell_segmentation_original
https://github.com/UncertaintyQuantification/cell_segmentation.git
44714c2e0be958fe796a8fd4bdbc220dae3c23dd   (44714c2, 2025-09-04, "Update on Sept 4")
```

## Dependency resolution — the blocker from the previous audit is cleared

The earlier audit reported EBImage MISSING, which blocked any faithful
paper-downstream claim. Resolved this session, non-destructively:

| package | before | after | how |
|---|---|---|---|
| BiocManager | MISSING | 1.30.27 | `install.packages` → user lib |
| **EBImage** | **MISSING** | **4.52.0** | `BiocManager::install("EBImage", lib=<user lib>)`, Bioconductor 3.22 |
| RobustGaSP | 0.6.8 | 0.6.8 | already present |
| magick / pracma | 2.9.1 / 2.4.6 | unchanged | already present |

Target library: `/Users/zchan/Library/R/arm64/4.5/library` (created; it did not
exist). Installed from **pre-built arm64 binaries** — no compilation, so the
absent system `fftw` was not an obstacle. Dependencies pulled: `bitops`,
`BiocGenerics`, `abind`, `tiff`, `locfit`, `fftwtools`, `RCurl`, `BiocVersion`.
**No `sudo`. The system R library at
`/Library/Frameworks/R.framework/.../library` was not touched.** R 4.5.1.

Scripts consume the user lib via an explicit
`.libPaths(c("/Users/zchan/Library/R/arm64/4.5/library", .libPaths()))`, so the
change is opt-in per script rather than global.

## The executed downstream, traced from the call graph

Entry points, **OBSERVED**:

```
Nuclear_Real_Analysis/Nuclear_Data_Generate_IoU.R:205        generate_GP_Masks_test(file_path, nugget=T)
Whole_Cell_Real_Analysis/Whole_Cell_Data_Generate_IoU.R:204  generate_GP_Masks_test(file_path, nugget=T)
Nuclear_Real_Analysis/GP_vs_NoGP/...(GP vs no GP).R:209      generate_GP_Masks_test(file_path, nugget=T)
Nuclear_Real_Analysis/GP_vs_NoGP/...(GP vs no GP).R:213      generate_GP_Masks_test2(file_path, nugget=T)
```

**Nuclei and whole-cell share the identical downstream** — both call
`generate_GP_Masks_test` with all defaults; only the image differs. Checked, not
assumed.

Stage by stage (`src/Modified_Functions_RGasp.R:606`):

| stage | exact call / rule | defaults |
|---|---|---|
| read | `magick::image_read`; `as.numeric(cropped_img[[1]])[,,1]` | raw/255, channel 1 only |
| tiling | `get_proportion(size, target_min=200, target_max=400)` over divisors 1/4,1/3,1/2,1; then `crop <- size %/% num_pieces` | nuclei 4x4 of 282x240; whole-cell 3x3 of 200x200 |
| reconstruction | `separable_GP_param_est` on tile (1,1) **once**, then `separable_GP` per tile | omitted entirely in the NoGP arms |
| threshold | `criterion_1(predmean_mat, delta=0.01, nugget=TRUE)` | see below |
| component count | `bwlabel(thresholded_image)` | EBImage |
| outlier tiles | `outlier_threshold <- 2`; `|n − mean| > 2*sd` → re-threshold at mean of non-outlier percentages; `>0.99` foreground → all background, pct = 1 | — |
| stitch | into `matrix(0, img_height, img_width)`; `floor` tiling leaves up to `crop−1` trailing rows/cols uncovered | nuclei: 960 of 962 cols covered |
| distance transform | `distmap(as.Image(combined_thresholded1))` | `metric = "euclidean"` |
| watershed | `EBImage::watershed(dist_map)` | **`tolerance = 1, ext = 1`**; POSITIVE map; EBImage's own seed detection; no markers supplied |
| cleanup | `eliminate_small_areas(GP_masks_raw, remove_size_threshold = 50)` | `area<50 & !boundary` → drop; `boundary & area<10` → drop |
| labels | `segmented_image@.Data`, background 0, **no renumbering after cleanup** (label sequence has gaps) | — |

### `criterion_1` in detail — **OBSERVED, reproduced with real RobustGaSP**

```r
percentages      <- seq(0, 1, by = delta)                    # 101 candidates
pixel_counts     <- sapply(percentages, \(t) sum(mat > t * max(mat)))
diff_pixel_counts<- abs(diff(pixel_counts))                  # 100 points
diff_mod         <- rgasp(percentages[-1], diff_pixel_counts, nugget.est = nugget)
smoothed         <- predict(diff_mod, testing_input = as.matrix(percentages[-1]))
diff_pixel_counts<- smoothed$mean                            # REPLACED by the GP mean
th               <- 0.05 * sd(diff_pixel_counts)
# walk forward from which.max until |d[i] - d[i-1]| < th
estimated_percentage <- percentages[stable_index + 1]
# if never stable: thresholded_image <- all zeros, estimated_percentage <- 1
```

Key facts: candidates are **proportions of the per-tile maximum**, not absolute
intensities and not quantiles; the criterion is **per tile**; each tile selects
its **own** percentage; the only shared quantity across tiles is the GP
`(beta1, beta2, nu)` triple. A `ksmooth` line is present but commented out.

Auditable dumps for tile 1 (both datasets, both arms) in
`results/real_cellseg_round6_paper_downstream_20260917/threshold_diagnostics/`:
candidate grid, raw criterion curve, `rgasp` fitted mean with 95% bands, selected
percentage and its absolute value, fitted `beta_hat` / nugget / `sigma2`, the
binary tile, and the tile predmean. RobustGaSP's optimizer trace is in `logs/`.

## The three arms in the repo, and why the published ablation is confounded

| file / function | image GP | rgasp on criterion curve | cleanup |
|---|---|---|---|
| `src/Modified_Functions_RGasp.R::generate_GP_Masks_test` | **yes** | **yes** | `eliminate_small_areas(·, 50)` → 50 / 10 |
| `GP_vs_NoGP/Segmentation_Functions_NoImageGP::generate_GP_Masks_test2` | no | **yes** | `eliminate_small_areas2` → mean·0.15 / mean·0.05 |
| `GP_vs_NoGP/Segmentation_Functions_NoGP::generate_GP_Masks_test2` | no | **no (commented out)** | `eliminate_small_areas2` → mean·0.15 / mean·0.05 |

`diff` between the two `Segmentation_Functions_*` files is **exactly** the four
rgasp/`smoothed$mean` lines plus one comment. Verified this session.

**Consequence: the paper's published "GP vs no GP" comparison (Figures S1/S2)
varies three things simultaneously** — the image GP, the RobustGaSP criterion
smoothing, and the cleanup rule. It is therefore not a clean ablation of the
reconstruction, and cannot be used as the control cell of a 2 x 2. The
`NoImageGP` variant removes one confound but keeps the cleanup difference.

Measured size of the confound (AP@0.5, our runs / recomputed historical):

| dataset | clean ablation (our B) | published NoGP (B') | confound contribution |
|---|---|---|---|
| nuclei | 0.4015 | 0.1821 | **−0.2194** (confounds hurt) |
| whole_cell | 0.4291 | 0.5660 | **+0.1369** (confounds help) |

The confound acts in **opposite directions** on the two datasets, so it cannot be
dismissed as a constant offset.

There is **no whole-cell NoGP arm anywhere in the repo** — `NoGP` appears only
under `Nuclear_Real_Analysis/GP_vs_NoGP/`, and `README.md:16` scopes it to the
nuclear channel. Whole-cell B is necessarily a constructed ablation.

## Orientation, established by arrays not by eye

1. The drivers' `process_image_mask` ends with
   `rotated <- t(M)[, nrow(M):1]; mirrored <- rotated[, ncol(rotated):1]`. Run on
   a 2x3 probe: `identical(mirrored, t(M))` → **TRUE**. **The two flips cancel;
   the net operation is a plain transpose.**
2. `EBImage::readImage` returns `(width, height)` — 962 x 1128 for a 962x1128
   `image_info`. After `t()` it is `(height, width)` = natural, matching
   `py_core/segmentation_eval.py::load_instance_mask`.
3. `combined_thresholded1` is allocated as `matrix(0, img_height, img_width)`, so
   everything downstream of it (distmap, watershed, labels) is already natural.
4. Cross-check: the R-driver GT foreground is **element-wise identical** to the
   Python-loaded GT on both images, with matching label counts (330, 403).
5. Orientation of the R **label** arrays scored by foreground IoU against the
   Python GT: nuclei identity **0.7411** vs flipud 0.1019 / fliplr 0.1100 /
   rot180 0.1138; whole-cell identity **0.7569** vs 0.1836 / 0.1904 / 0.1863.
   Identity wins; asserted in code, so a regression would raise.

**No mask was flipped anywhere.** Non-square dimensions (1128x962, 600x602) also
make an accidental transpose non-conformable, which is a second independent
guard.

## Fidelity verdicts

| stage | verdict | basis |
|---|---|---|
| threshold rule (`criterion_1`) | **YES — faithful** | real R, real `RobustGaSP 0.6.8`; no Gaussian filter / Li / Otsu / spline substitution anywhere in B or D |
| distance transform | **YES** | `EBImage::distmap`, euclidean |
| watershed | **YES — faithful** | real `EBImage 4.52.0`, `watershed(dist_map)` at defaults `tolerance=1, ext=1`, positive map, EBImage's own seeding |
| outlier-tile handling | **YES** | reproduced verbatim, incl. the `>0.99` revert branch; observed 1–2 outlier tiles per run, 0 reverts |
| cleanup | **YES — both variants** | `50 / 10` for the GP arm; `mean·0.15 / mean·0.05` for the published NoGP arm, **15%/5% confirmed in code**; no morphology; no relabelling |
| stitching / orientation | **YES** | verified by element-wise GT comparison and an orientation sweep |
| NoGP control for the 2 x 2 | **CONSTRUCTED ABLATION** (labelled) | the published arm is confounded three ways and does not exist for whole-cell |
| Fast-GP fit inside D | **paper-faithful single L-BFGS-B run** | lands on the degenerate `beta → ∞` plateau on both images, as the published code does |

## Historical reproduction

Recomputed AP from the repo's saved IoU matrices using the paper's own
`compute_ap_from_ious` — whose definition is **identical** to ours (per-GT-row
argmax IoU, `>=` inclusive, `AP = TP/(TP+FP+FN)`), which independently validates
evaluator compatibility:

| dataset | file | historical AP@0.5 | our run | Δ | #pred |
|---|---|---|---|---|---|
| nuclei | `ious_noGP.csv` | 0.1821 | B' 0.1821 | **−0.0000** | **358 vs 358** |
| nuclei | `ious_gp.csv` | 0.6547 | D 0.6481 | −0.0066 | 321 vs 317 |
| nuclei | `ious_noImageGP.csv` | 0.3318 | not run | — | — |
| whole_cell | `ious_gp.csv` | 0.5765 | D 0.5839 | +0.0073 | 343 vs 349 |

The **NoGP arm reproduces exactly**. It contains no GP and hence no optimizer.
The GP arms differ by ≤ 0.007 AP, consistent with the flat `beta → ∞` plateau
documented in `audits/PAPER_FAST_GP_WIRING_REPORT.md`: `beta` is unidentifiable
there, so different terminations give predictive means differing at ~1e-6, enough
to flip a few `percentage * max(tile)` decisions.

Separating the two senses of reproduction:

- **(A) algorithm reproduction: CONFIRMED.** Every downstream stage runs as
  written, in R, with the genuine packages.
- **(B) historical numerical reproduction: exact for NoGP; within 0.007 AP for
  the GP arms.** Historical *masks* are not shipped in the repo (only IoU
  matrices, boundary PNGs, and one `original_fig_seg.npy`), so no element-wise
  mask comparison against the historical run is possible. Stated, not
  worked around.

## The finding this enables

The reconstruction's value is **downstream-dependent**, and the mechanism is
identified rather than inferred. Pre-cleanup watershed basin counts:

| dataset | raw + paper downstream | Fast-GP + paper downstream | GT |
|---|---|---|---|
| nuclei | **2846** basins | **484** basins | 330 |
| whole_cell | 577 | 474 | 403 |

EBImage's tolerance-based automatic seeding finds thousands of shallow maxima in
the distance map of a pixel-noisy raw binary. The Fast-GP reconstruction removes
that pixel-scale noise. Our current downstream never had the problem because it
supplies explicit `peak_local_max` markers with `min_distance = 15/9`.

**The Fast-GP reconstruction substitutes for explicit marker control.** Under the
paper's automatic seeding it is worth +0.2466 / +0.1547 AP@0.5; under explicit
markers it is worth −0.0477 / +0.0000. Both readings are correct; they are
answers to different questions.
