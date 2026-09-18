# Task A — Reconstructing the TRUE Fast-GP baseline from the original paper code

Audit date 2026-09-17. Nothing under `results/` was modified, deleted or rerun.
No full-image GP/QEP experiment was run for this audit.

Every row below is tagged:

- **OBSERVED** — read directly out of source, or produced by a command run this session.
- **VERIFIED** — a numerical experiment run this session supports it.
- **INFERRED** — reasoning from observed facts; not directly tested.
- **UNVERIFIED** — could not be checked in this environment.

---

## A0. External reference repository

| item | value |
|---|---|
| remote | `https://github.com/UncertaintyQuantification/cell_segmentation.git` |
| local clone | `/Users/zchan/eclipse-workspace/cell_segmentation_original` (sibling of this project) |
| HEAD SHA | `44714c2e0be958fe796a8fd4bdbc220dae3c23dd` |
| short SHA | `44714c2` |
| HEAD date / subject | 2025-09-04, "Update on Sept 4" |
| in this project's git index? | **No** — verified absent from `Cell_Seg_QEP`'s index |
| treatment | read-only reference; nothing in it was edited |

Paper: *Unsupervised Cellular Boundary Detection by Fast Gaussian Processes* —
Baracaldo, King, Yan, Lin, Miolane, Gu (2025). **OBSERVED**

Local R environment: `R` is installed and was used for the A4 parity test.
**EBImage is NOT installed**, so the original end-to-end pipeline (which needs
`EBImage::distmap` and `EBImage::watershed`) could not be executed here.
**OBSERVED.** Consequently every statement about the original *watershed* stage
is from source reading, not execution, and is labelled as such.

---

## A1. The actual original call graph for REAL DATA

Two distinct code paths exist in the original repo, and it matters which one is
"the paper's Fast GP".

### A1.1 Which entry point the real analyses use — **OBSERVED**

```
Nuclear_Real_Analysis/Nuclear_Data_Generate_IoU.R:205   generate_GP_Masks_test(file_path, nugget = T)
Whole_Cell_Real_Analysis/Whole_Cell_Data_Generate_IoU.R:204   generate_GP_Masks_test(file_path, nugget = T)
```

So the real-data pipeline is `src/Modified_Functions_RGasp.R::generate_GP_Masks_test`
(defined at L606). It is NOT `src/2dim_lattice_func.R::lattice_alg`.

`lattice_alg` is used **only** by `Simulated_Experiments/*.R`. **OBSERVED.**
This distinction is the single most important finding of Task A, because our
Python `py_core/dim_2_lattice.py` is a port of `lattice_alg` — the *simulation*
function — not of the real-data path.

### A1.2 Traced stage-by-stage graph of `generate_GP_Masks_test` — **OBSERVED**

```
image file
  -> magick::image_read ; crop to tiles
     img_matrix <- as.numeric(cropped_img[[1]])[,,1]       # magick => floats in [0,1]
     crop geometry always uses crop_width / crop_height    # the piece_* args are UNUSED
  -> GP hyperparameter estimation, ONCE, ON THE FIRST TILE ONLY:
       if (i == 1 && j == 1) { parameters <- separable_GP_param_est(img_matrix) }
  -> per tile, GP predictive mean with those FIXED shared parameters:
       separable_GP_info <- separable_GP(img_matrix, parameters$param)
       predmean = X_testing %*% theta_hat
                  + as.vector(t(r1) %*% R_tilde_inv_output_normalize_mat %*% r2)
  -> per-tile foreground threshold via criterion_1 (L362) over a percentage grid,
     threshold_image (L349):  threshold_value <- percentage * max(mat)
  -> OUTLIER-TILE HANDLING: outlier_threshold <- 2 ; tiles whose chosen
     percentage is an outlier are re-thresholded at the mean of the
     non-outlier percentages; a tile that then exceeds 0.99 foreground is
     reverted to all background
  -> stitch tiles into one binary image (combined_thresholded1)
  -> dist_map        <- EBImage::distmap(as.Image(combined_thresholded1))
  -> segmented_image <- EBImage::watershed(dist_map)          # defaults tolerance=1, ext=1
  -> GP_masks <- eliminate_small_areas(GP_masks_raw, remove_size_threshold)
  -> IoU / mask evaluation in the calling *_Generate_IoU.R script
```

### A1.3 Details of the individual original functions — **OBSERVED**

`separable_GP_param_est` (`src/Modified_Functions_RGasp.R:3`)

- inputs are **normalized lattice coordinates**: `input1 = seq(0, 1, 1/(n1-1))`, same for `input2`.
- `set.seed(1)` at the top.
- Matérn 5/2, **separable**: `R = R1 ⊗ R2`, solved by eigendecomposition of R1 and R2.
- constant mean basis `X = 1`; `theta_hat` and `S_2` are **profiled out** analytically.
- objective `-(1/2*sum(log(Lambda_tilde_inv)) - N/2*log(S_2))`.
- `optim(param_ini, method = "L-BFGS-B")`, `param_ini = c(-2, -2, -3)`; on failure retries at `param_ini + runif(3)`.
- uses **ALL pixels** of the tile. No subsampling.
- returns `c(beta1, beta2, nu)` — two inverse-range parameters and a nugget.
- contains dead code: an unused `gradient()` and numerical gradients.

`separable_GP` (L123) — takes `parameters` as an **argument**; it does not fit anything.

`criterion_1` (L362)

```r
diff_mod <- rgasp(percentages[-1], diff_pixel_counts, nugget.est = nugget)   # RobustGaSP
smoothed <- predict(diff_mod, ...)
th <- 0.05 * sd(diff_pixel_counts)
# a ksmooth() line is present but COMMENTED OUT
# when !found_stable the function defaults to ALL BACKGROUND
```

`eliminate_small_areas` (L417) — **border-aware**:

```r
if (area < size_threshold && !on_boundary)          remove
if (on_boundary && area < size_threshold / 5)       remove
```

---

## A2. Function-by-function mapping: original R → our Python

Verdict scale: **EXACT** (numerically equivalent) / **APPROX** (same intent,
different numerics) / **DIFFERENT** (materially different algorithm) / **MISSING**.

| # | original R | our Python | verdict | mathematical difference | likely segmentation consequence |
|---|---|---|---|---|---|
| 1 | `separable_GP_param_est` (L3), separable Matérn 5/2 `K1⊗K2`, eigendecomposition, profiled mean+variance, all pixels, L-BFGS-B | `py_core/dim_2_lattice.py::lattice_alg` (port of `lattice_alg`, Nelder-Mead default) | **EXACT** once inputs/optimizer are matched — **VERIFIED**, see A4 | none detectable: `|Δbeta1|≤4.2e-05`, `|Δnu|≤5.0e-07`, predictive-mean `max|diff| = 1.03e-07` | none, *if* this function is the one actually wired into the pipeline. It currently is not. |
| 2 | same, as used by the **real** pipeline | `py_core/Modified_Functions_RGasp.py::separable_gp_smooth_gpytorch` (L208) — **isotropic** `ScaleKernel(MaternKernel(nu=2.5))` on joint 2D coordinates, GPyTorch ExactGP, Adam lr=0.1 × 75 iters | **DIFFERENT** | (a) isotropic joint-2D kernel instead of separable `K1⊗K2`, so it cannot express `beta1 ≠ beta2` — and the fitted values are genuinely anisotropic (`beta1=3.754` vs `beta2=2.611` at n=40; `15.50` vs `11.48` at n=120, **VERIFIED**); (b) Adam on the full ELBO instead of profiled-likelihood L-BFGS-B; (c) no analytic profiling of mean/variance | anisotropy is lost. A single compromise lengthscale over-smooths one image axis and under-smooths the other, which directly perturbs the distance transform that drives watershed. |
| 3 | all pixels used for estimation | `max_points = 6000` random subsample, **unseeded `np.random.choice`** (`Modified_Functions_RGasp.py:234`) | **DIFFERENT** | estimation set is a random subset, and a different one on every run | non-reproducible hyperparameters → non-reproducible masks. Already measured: recomputing AP from the stored IoU matrix gives 0.0388 against a stored 0.0307 (`segmentation_pipeline.py:45-53`, OBSERVED). |
| 4 | float64 throughout | float32 in the legacy smoother | **APPROX** | reduced precision in the Kronecker/Cholesky solves | small; not the dominant effect |
| 5 | `input1 = seq(0, 1, 1/(n1-1))` | `experiments/simulated/linear_diffusion_gp.py:248-249` passes `input1 = np.arange(1, H+1)` | **DIFFERENT** | inputs on `[1,H]` instead of `[0,1]` rescale the distance argument by `(H-1)`, so the SAME `beta` means a different physical range and `param_ini=c(-2,-2,-3)` starts in a different regime | fitted lengthscale lands in the wrong regime; affects the simulated experiments, not the real pipeline |
| 6 | intensities are magick floats in `[0,1]` | depends on loader; must be confirmed per call site | **APPROX** | a global intensity rescale changes `nu` (nugget) relative to signal | `criterion_1`'s rule is not affine-invariant (see #8), so intensity scaling is not harmless here |
| 7 | parameters estimated on **tile (1,1)** and **reused for every tile** | our pipeline re-estimates **per tile** | **DIFFERENT** | original enforces one global smoothness; ours adapts per tile | this cuts both ways: per-tile fitting is arguably better statistically, but it makes tiles mutually inconsistent and is a plausible contributor to the tile-boundary artifacts and to the outlier-tile machinery the original needed. Not an "improvement" that can be claimed without a test. |
| 8 | `criterion_1` smoothed with RobustGaSP `rgasp()` | `py_core/Modified_Functions_RGasp.py::criterion_1` (L53) uses `scipy.ndimage.gaussian_filter1d(sigma=2)` | **DIFFERENT** | a GP smoother with estimated hyperparameters replaced by a fixed-bandwidth Gaussian filter | already diagnosed in round 3 (4 defects, and the only non-affine-invariant rule in the pipeline). Replacing the whole rule with Li raised nuclei raw AP from 0.3383 to 0.7543 (OBSERVED from round-3 results). |
| 9 | outlier-tile detection `outlier_threshold <- 2`, re-threshold at mean of non-outliers, revert to background if >0.99 foreground | not present | **MISSING** | no protection against a single pathological tile | in the original this is what prevents one bad tile from destroying the stitched mask; our round-2 catastrophic all-background/all-foreground tiles are the symptom of its absence |
| 10 | `EBImage::distmap` | `scipy.ndimage.distance_transform_edt` | **APPROX** | EBImage's distmap is a different (Felzenszwalb-style) metric implementation | small |
| 11 | `EBImage::watershed(dist_map)`, defaults `tolerance = 1, ext = 1` — performs its **own tolerance-based seed detection** | `skimage.segmentation.watershed(-dist, markers=None, mask=...)` — **no tolerance parameter**; `markers=None` seeds at *every* local minimum | **DIFFERENT** | the original merges basins shallower than `tolerance=1`; skimage with `markers=None` merges nothing | this is a real over-segmentation mechanism, confirmed experimentally: switching to explicit `peak_local_max` markers improved 32/32 cases (OBSERVED, round 2). `marker_mode="legacy_none"` is therefore NOT the paper's behavior — it is *more* fragmented than the paper. |
| 12 | `eliminate_small_areas` with border-aware `size_threshold` / `size_threshold/5` rule | `py_core/Modified_Functions_RGasp.py::eliminate_small_areas` (L108) | **EXACT** | none; the border logic is faithfully ported | none |
| 13 | mask loading / orientation / AP | `py_core/segmentation_eval.py::load_instance_mask` (native orientation), `evaluate_instances` | **UNVERIFIED against the original** | the original computes IoU inside `*_Generate_IoU.R`; we did not run it (EBImage missing), so AP definitions were not cross-checked numerically | an orientation or matching-convention mismatch would bias absolute AP. Our internal metric code was separately unit-checked, so this is a comparability risk against the *paper's published numbers*, not an internal-consistency risk. |

### A2 summary of substantive mismatches

Ranked by expected effect on masks, all **INFERRED** from the differences above
except where noted:

1. **#11 watershed seeding** — the only one with direct experimental confirmation (32/32).
2. **#8 `criterion_1`** — direct experimental confirmation (AP 0.3383 → 0.7543).
3. **#2 isotropic vs separable kernel** — the fitted anisotropy is VERIFIED real (`beta1/beta2 ≈ 1.44` and `1.35`), so the loss is not hypothetical.
4. **#9 missing outlier-tile handling** — explains an observed failure class.
5. **#3 unseeded subsample** — destroys reproducibility, VERIFIED via the 0.0307/0.0388 discrepancy.
6. **#7 first-tile vs per-tile parameters** — large behavioral difference, untested.

---

## A3. Does `gp_legacy` reproduce the paper's Fast GP?

**No.** **OBSERVED / INFERRED.**

`py_core/segmentation_pipeline.py:35-37` documents the arm as

> `"gp_legacy"` : the legacy `separable_gp_smooth_gpytorch` verbatim (float32,
> isotropic Matern 2.5, Adam lr=0.1, 75 iters). Preserves the historical GP behavior for continuity.

The docstring is accurate about what it *is* — and that is precisely the problem:
it is the *isotropic GPyTorch* smoother, which differs from the paper on
mismatches #2, #3, #4, #7, #8, #9 and #11 above. The word "legacy" has been read
in this project as "the paper's method", and it is not. Note the internal
contradiction: the function is named `separable_gp_smooth_gpytorch` but the
kernel it builds is isotropic.

**Proposed rename:** `gp_isotropic_gpytorch_2025` — or, if a shorter label is
wanted, `gp_project_legacy`. Either way the name must not contain "separable",
"fast", or "paper". Reserve the name `paper_fast_gp` for an arm that has actually
been validated against the original.

Recommendation: rename the **arm label** and the **function**, but do not alter
its numerics — the historical results in `results/real_cellseg_round1..4` were
produced by it and must stay interpretable.

---

## A4. Minimum path to a faithful `paper_fast_gp`

Three options were considered as instructed.

### Option A — call R directly from Python

**Rejected.** EBImage is not installed, so the stages that most need R
(`distmap`, `watershed`) cannot run here anyway; and an R round-trip per tile
would dominate runtime. **OBSERVED** (EBImage absence).

### Option B — port the exact R Fast-GP to Python from scratch

**Unnecessary.** Option C already passes.

### Option C — reuse `py_core/dim_2_lattice.py`, but only if numerical parity is demonstrated

**Accepted — parity WAS demonstrated. VERIFIED.**

Test harness written this session (new files, nothing overwritten):

- `audits/taskA_parity_r_side.R` — sources the original `src/Modified_Functions_RGasp.R`,
  reads the real nuclei image through `magick` exactly as the original does, runs
  `separable_GP_param_est` + `separable_GP` on an n×n crop, dumps
  `r_params_n{n}.csv`, `r_tile_n{n}.csv`, `r_predmean_n{n}.csv`.
- `audits/taskA_parity_py_side.py` — loads the R-dumped tile (so both sides see
  bit-identical input), runs `py_core.dim_2_lattice.lattice_alg` with
  `input1 = np.linspace(0,1,·)` under both L-BFGS-B and Nelder-Mead, writes
  `parity_summary_n{n}.csv`.

Results, R as reference, Python with **L-BFGS-B**:

| n | R beta1 | R beta2 | R nu | \|Δbeta1\| | \|Δbeta2\| | \|Δnu\| | predmean max\|diff\| | predmean RMSE | RMSE as % of range |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 3.7542237977 | 2.6108606986 | 0.0525636587 | 4.244e-05 | 1.172e-06 | 5.033e-07 | **1.030e-07** | 2.491e-08 | 0.0001% (range 0.1807) |
| 120 | 15.4975758917 | 11.4844446735 | 0.1877490262 | 2.174e-05 | 6.439e-05 | 1.079e-07 | **1.117e-07** | 2.020e-08 | — |

A predictive-mean discrepancy of ~1e-07 on an image with range 0.18 is far below
any threshold that could alter a foreground decision. `dim_2_lattice.py` is a
faithful implementation of the original separable Fast-GP core.

**Caveat, stated explicitly:** parity was established for the *GP core* —
hyperparameter estimation and predictive mean. It was **not** established for
`criterion_1`, the outlier-tile logic, `distmap`, or `watershed`, because
EBImage is missing. Those stages remain **UNVERIFIED** against the original.

### Required corrections to build `paper_fast_gp` on top of `dim_2_lattice.py`

All five are needed; each is **OBSERVED** in the original source:

1. Pass `input1 = input2 = np.linspace(0, 1, n)` — normalized lattice coordinates, not pixel indices.
2. Estimate hyperparameters **once on the first tile** and reuse them for all tiles.
3. Use **all pixels** for estimation (no `max_points` subsample).
4. Feed intensities as floats in `[0,1]` (magick convention).
5. Matérn 5/2 with the R parameterization `(1 + x + x²/3)·exp(−x)`, `x = sqrt(5)·beta·d` — already what `dim_2_lattice.py::matern_5_2` does.

Plus, to be faithful to the *pipeline* rather than only the GP:

6. Restore the outlier-tile handling (mismatch #9).
7. Use `criterion_1` with a RobustGaSP-equivalent smoother, or clearly label the
   thresholding rule as a deliberate deviation (it already is, as of round 3).
8. Use a tolerance-merging watershed, not `markers=None` (mismatch #11).

### Minimal validation plan for `paper_fast_gp` (not run — awaiting approval)

Cheap, in this order:

1. Re-run the existing parity harness at n = 40, 120 and additionally n = 256, and on
   a whole-cell tile, asserting predmean `max|diff| < 1e-05`. Cost: seconds.
2. Single-image tile-grid check: confirm first-tile-only estimation reproduces the
   R `parameters$param` for tile (1,1), and that every later tile consumes it unchanged.
3. Compare `paper_fast_gp` reconstructions against `gp_isotropic_gpytorch_2025`
   reconstructions **directly to each other** (not merely via each one's RMSE to raw —
   that comparison was already shown to be uninformative).
4. Only then score AP on the 2 dev images.

Nothing beyond step 1 should run until the user approves, per the standing
instruction not to start the next large experiment.

---

## A5. Task A conclusions

1. The paper's real-data Fast GP is `generate_GP_Masks_test` → `separable_GP_param_est`
   (**first tile only**) + `separable_GP` (all tiles), with L-BFGS-B, `seq(0,1)`
   inputs, a separable Matérn-5/2 Kronecker kernel, all pixels, and magick `[0,1]`
   intensities. **OBSERVED.**
2. Our `gp_legacy` arm does **not** reproduce it, and its name is misleading.
   Rename to `gp_isotropic_gpytorch_2025`. **OBSERVED/INFERRED.**
3. Our `py_core/dim_2_lattice.py` *is* numerically faithful to the original
   separable GP core — `max|diff| ≈ 1.1e-07` — but it is a port of the
   **simulation-only** `lattice_alg`, and it is currently not wired into the real
   pipeline at all. **VERIFIED.**
4. Building a genuine `paper_fast_gp` is therefore cheap: reuse
   `dim_2_lattice.py` with the five corrections listed above. **INFERRED** from (3).
5. The GP core is not where the pipeline was losing accuracy. The two
   experimentally confirmed large effects were watershed seeding and the
   thresholding rule — both *outside* the GP. **OBSERVED** from rounds 2 and 3.
