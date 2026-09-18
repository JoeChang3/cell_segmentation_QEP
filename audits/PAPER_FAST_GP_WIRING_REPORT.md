# Part A — Wiring a faithful `paper_fast_gp` into the real-data pipeline

Date 2026-09-17. No Round 1–4 result was modified or deleted. No full benchmark
was run. Reconstruction only; no AP claim anywhere in this report.

Tags: **OBSERVED** (read from source, or produced by a command run this session),
**VERIFIED** (a numerical experiment supports it), **INFERRED**, **UNVERIFIED**.

Reference repository (read-only, not in this project's git index):

| item | value |
|---|---|
| remote | `https://github.com/UncertaintyQuantification/cell_segmentation.git` |
| local | `/Users/zchan/eclipse-workspace/cell_segmentation_original` |
| commit | `44714c2e0be958fe796a8fd4bdbc220dae3c23dd` (short `44714c2`) |
| date / subject | 2025-09-04, "Update on Sept 4" |

R environment: `magick 2.9.1`, `pracma 2.4.6`, `RobustGaSP 0.6.8`, `plot3D 1.4.2`,
`dplyr 1.1.4` present. **EBImage MISSING** — so `distmap`, `watershed`,
`as.Image`, `bwlabel` cannot run, and no end-to-end R segmentation was possible.
`separable_GP_param_est` and `separable_GP` do NOT need EBImage, so the
reconstruction-stage parity test below DID run against real R output. **OBSERVED.**

---

## A1. The original real-data GP path, re-confirmed

### A1.1 Entry point — **OBSERVED**

```
Nuclear_Real_Analysis/Nuclear_Data_Generate_IoU.R:205     generate_GP_Masks_test(file_path, nugget = T)
Whole_Cell_Real_Analysis/Whole_Cell_Data_Generate_IoU.R:204  generate_GP_Masks_test(file_path, nugget = T)
```

Confirmed chain: `generate_GP_Masks_test` (L606) → `separable_GP_param_est` (L3)
→ `separable_GP` (L123).

**`lattice_alg` and `separable_GP` are NOT the same function** — checked, not
assumed. Differences, all **OBSERVED**:

| | `src/2dim_lattice_func.R::lattice_alg` | `Modified_Functions_RGasp.R::separable_GP_param_est` + `separable_GP` |
|---|---|---|
| used by | `Simulated_Experiments/*.R` only | the two real-data analyses |
| optimizer | `optim_method = 'Nelder-Mead'` (default arg) | `optim(..., method = "L-BFGS-B")` (L106) |
| fit + predict | one function does both | **split**: estimate once, predict many times |
| inputs | `input1`, `input2` are ARGUMENTS | hard-coded `seq(0, 1, 1/(n-1))` |
| testing grid | `testing_input1/2` arguments | hard-coded to the training grid |

This split is exactly what makes first-tile-only estimation possible, and is why
our `py_core/dim_2_lattice.py` (a port of `lattice_alg`) could not be dropped in
as-is.

### A1.2 Traced pipeline — **OBSERVED**

```
image file
  -> magick::image_read; get_proportion(H), get_proportion(W) choose the tile size
     crop ALWAYS uses crop_width/crop_height; the piece_width/piece_height
     computed at L651-652 are UNUSED by the crop at L655
     img_matrix <- as.numeric(cropped_img[[1]])[,,1]
  -> if (i == 1 && j == 1) parameters <- separable_GP_param_est(img_matrix)   # ONCE
  -> separable_GP(img_matrix, parameters$param)                               # every tile
  -> criterion_1 per tile (RobustGaSP rgasp smoothing of the difference curve)
  -> outlier tiles: outlier_threshold <- 2; re-threshold at the mean of the
     non-outlier percentages; revert to all-background if > 0.99 foreground
  -> stitch
  -> dist_map <- EBImage::distmap(as.Image(combined_thresholded1))
  -> segmented_image <- EBImage::watershed(dist_map)     # defaults tolerance=1, ext=1
  -> eliminate_small_areas(GP_masks_raw, remove_size_threshold)
```

### A1.3 The documentation items requested

**Parameter definitions** — **OBSERVED**

| symbol | meaning |
|---|---|
| `beta1`, `beta2` | **INVERSE range** (not lengthscale) along axis 1 (rows) and axis 2 (cols). Optimized as `log beta`; `param_ini` = `-2` for both. |
| `nu` | **Nugget**, added to the Kronecker eigenvalues: `Lambda = kron(evals2, evals1) + nu`. Optimized as `log nu`, init `-3`. Because `sigma^2` is profiled out, `nu` is a **ratio to the signal variance**, not an absolute noise level. |
| kernel | Matérn 5/2, `(1 + x + x^2/3) exp(-x)`, `x = sqrt(5) * beta * d` (L39-42). Only this branch is live; `Exp_funct` / `double_Exp_funct` lines are commented out. |
| inputs | `input1 = seq(0, 1, 1/(n1-1))`, `input2 = seq(0, 1, 1/(n2-1))`. Normalized lattice coordinates, so `beta` is dimensionless in tile units and `param_ini` is calibrated to that scale. |
| objective | `-(1/2*sum(log(Lambda_tilde_inv)) - N/2*log(S_2))` (L81) |

**Mean handling** — **OBSERVED.** Constant basis `X = matrix(1, N, 1)`.
`theta_hat` is profiled analytically. Critically, `separable_GP_param_est`
returns **only** `list(param = c(beta, nu))` (L121) — not `theta_hat`. So
`separable_GP` **re-profiles `theta_hat` from each tile's own data** at L245.
Confirmed in our port: the four tiles of the wiring check produced `theta_hat`
= 0.21307, 0.13844, 0.19300, 0.17800. **VERIFIED.**

**Variance handling** — **OBSERVED.** `S_2` is profiled (L195, L245-248 region)
and used in the objective, but **cancels out of the predictive mean**: the
predictor is `X theta + r1' (R + nu I)^-1 (y - X theta)`, which is invariant to
`sigma^2`. So there is **no output-scale parameter in the reconstruction at all**.
We return `S_2` and `sigma2_hat = S_2/N` as diagnostics only.

**First-tile estimation and reuse** — **OBSERVED**, L659-664:

```r
if (i == 1 && j == 1) { parameters <- separable_GP_param_est(img_matrix) }
separable_GP_info <- separable_GP(img_matrix, parameters$param)
```

**Normalization / scaling** — **OBSERVED.** magick yields raw/255. Verified:
`nuclei_figure_1` has `img_min = 105` and magick reports a tile minimum of
`0.5333333333`; the global `105/255 = 0.411765` also matches the whole-image
range. So it is a **plain /255, not a min-max rescale**. Our Python loaders keep
[0,255]; the estimator is exactly equivariant under `y -> c*y` (objective shifts
by the constant `-N log c`, so `beta` and `nu` are invariant and `predmean`
scales by `c`), so `paper_fast_gp` normalizes internally and returns in the
caller's scale. **VERIFIED** by construction and by the wiring check
(`input_scale = 255.0`, recon range `[0.08, 229.35]` for a raw range `[0, 229]`).

**Clipping / output transformations** — **OBSERVED: none.** `z_lim` at L271 feeds
only a commented-out `image2D` call. `predmean` is returned unclipped, and can
exceed the input range (our whole-cell recon max 229.35 vs raw 229.0).

**Matrix orientation** — **VERIFIED empirically, not assumed.** Tested this
session:

```
image_info            : 962 x 1128  (width x height)
dim(as.numeric(img[[1]])) : 1128 x 962 x 1
=> nrow matches HEIGHT
```

So `img_matrix` is `(rows, cols)` — natural image orientation — and
`n1 = crop_height`, `n2 = crop_width`. The stitch at L727-730 (`piece_height <-
nrow(predmean_mat)`) is therefore self-consistent; there is **no orientation bug**
in the original. The algorithm is covariant under transpose (transposing the
input transposes the output and swaps `beta1`/`beta2`), so feeding tiles in our
native orientation is correct.

Two incidental **OBSERVED** findings: `gradient(as.matrix(output_mat))` (L15,
L135, from `pracma`) and the `output_numer_grad*` blocks are **dead code** —
computed and never used. `separable_GP` also returns finite-difference gradients
`grad1`, `grad2`, `grad_magnitude` (L340-346), and `generate_GP_Masks_test`
**never reads them** — only `predmean_mat`. Our port therefore does not compute
them.

Also **OBSERVED**: `num_pieces_* = floor(size / crop)`, so up to `crop-1`
trailing rows/columns are never covered by any tile. For `nuclei_figure_1`,
4 x 240 = 960 of 962 columns are covered; the last 2 columns stay 0 in the
original's `combined_predmean`.

---

## A2. Real-tile numerical parity — **the headline result**

Tile: the **true first tile** of each development image, extracted with exactly
`generate_GP_Masks_test`'s geometry.

| dataset | image | H x W | tiling | tile (1,1) |
|---|---|---|---|---|
| nuclei | `nuclei_figure_1/original_fig.png` | 1128 x 962 | 4 x 4 of 282 x 240 | 282 x 240 |
| whole_cell | `whole_cell_figure_1/original_fig.jpg` | 600 x 602 | 3 x 3 of 200 x 200 | 200 x 200 |

Both sides read **bit-identical pixels**: the R script dumps the tile, the Python
script loads that dump. Independently verified that `magick` and `imageio` decode
these files identically — `max|magick - imageio| = 5.55e-16`, **0 of 67680** and
**0 of 40000** pixels differing above `1e-9`, for the PNG and the JPEG
respectively. **VERIFIED.**

Scripts: `audits/taskA2_real_tile_r_side.R`, `audits/taskA2_real_tile_py_side.py`.

### A2.1 The linear algebra is EXACT — **VERIFIED**

Feeding **R's own fitted `(beta1, beta2, nu)`** into our `reconstruct_tile`
isolates the translation from the optimizer:

| dataset | \|Δtheta_hat\| | \|ΔS_2\| | predmean max\|diff\| | predmean RMSE | corr |
|---|---|---|---|---|---|
| nuclei | 2.887e-15 | 7.390e-13 | **5.329e-15** | 1.210e-15 | 1.000000000000 |
| whole_cell | 3.275e-15 | 3.183e-12 | **5.285e-14** | 6.758e-15 | 1.000000000000 |

This is machine precision. **Our port of `separable_GP` is numerically exact on
real tiles**, including the profiled local mean and the profiled local scale.

Independently, the R script asserts that its own recomputation of `theta_hat`
reproduces `separable_GP`'s output: `max|diff| = 0.000e+00` on both tiles.

### A2.2 The OPTIMIZER disagrees, and the objective is non-convex — **VERIFIED**

With each side running its own optimizer from the paper's `param_ini = (-2,-2,-3)`:

| dataset | side | beta1 | beta2 | nu | objective f | degenerate axis |
|---|---|---|---|---|---|---|
| nuclei | R L-BFGS-B | 26.0388376235 | **5679456.5183** | 0.0822243107 | 129310.4853 | cols |
| nuclei | PY L-BFGS-B | 35.5041219944 | 23.6888486972 | 0.1705953183 | **116922.5219** | none |
| whole_cell | R L-BFGS-B | **38046968.0349** | 37.9923051735 | 0.0022514151 | 61119.2354 | rows |
| whole_cell | PY L-BFGS-B | 34302.0315535 | 37.9928426984 | 0.0022513291 | 61119.2354 | rows |

Resulting predictive-mean agreement:

| dataset | predmean max\|diff\| | RMSE | corr |
|---|---|---|---|
| nuclei | 4.258e-02 | 7.971e-03 | 0.993367 |
| whole_cell | **1.953e-06** | 1.618e-07 | 1.000000000000 |

So on **whole_cell the two sides agree** (both land in the same degenerate
basin, f identical to 4 dp). On **nuclei they do not**, and the reason is now
established:

**The objective was evaluated at both points using R's OWN objective code:**

```
R-side objective at R's optimum        : 129310.4852588164
R-side objective at Python's optimum   : 116922.5219016823    <- LOWER by 12388
R-side objective at param_ini(-2,-2,-3): 194021.3380799486
```

and cross-checked with our Python objective: `129310.4852588165` /
`116922.5219016827` — the two implementations agree to ~1e-9 relative, so this is
the **same function**, and Python's point is genuinely better.

R's `optim` nevertheless reports success:

```
convergence code : 0   (0 = success)
message          : CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH
fn/gr counts     : 24 / 24
```

**Not** a finite-difference-step artifact: sweeping R's `ndeps` over
`1e-3 … 1e-8` gives `f = 129310.4853` every time, with `beta2` landing anywhere
in `7.0e5 … 5.7e6`. Identical objective, wildly different `beta2` ⇒ **`beta2` is
unidentifiable there**.

Multi-start (`audits/taskA2_objective_landscape.py`, 16 starts):

| dataset | distinct local optima found | best f | paper's `param_ini` lands at | R lands at |
|---|---|---|---|---|
| nuclei | **2** | 116922.5219 | **116922.5219** (best) | 129310.4853 |
| whole_cell | **5** | **41016.7491** | 61119.2354 | 61119.2354 |

The profile along `log beta2` exhibits the plateau directly (nuclei, `beta1`/`nu`
at R's values):

| log beta2 | beta2 | f |
|---|---|---|
| 2.0 | 7.39 | 120919.63 |
| 3.0 | 20.09 | **117057.33** (minimum along this ray) |
| 5.0 | 1.48e2 | 122691.77 |
| 8.0 | 2.98e3 | 129310.4853 |
| 12.0 | 1.63e5 | 129310.4853 |
| 15.55 | 5.68e6 | 129310.4853 ← R's point |
| 21.0 | 1.32e9 | 129310.4853 |

Flat to 4 dp from `beta2 = 2981` to `1.3e9`. **The `beta -> infinity` limit is a
genuine flat local optimum**, and it corresponds to `R2 = I`: zero correlation
across that axis, i.e. an axis-aligned smoother that smooths along one direction
only.

### A2.3 What this means, stated carefully

- The paper's **model, objective, and linear algebra** are reproduced exactly
  (5e-15). **VERIFIED.**
- The paper's **estimator is not at the global optimum of its own objective on
  either real development tile** (gaps 12388 and 20102), and returns a
  degenerate single-axis fit on both. **VERIFIED.**
- This is **not a bug in the R code** and not a translation error. It is a
  non-convex profiled likelihood with a flat boundary plateau, combined with a
  single fixed initialization and no restart-on-poor-convergence (the existing
  `while (!is.numeric(...))` retry catches only *errors*). **OBSERVED/INFERRED.**
- Consequence: "reproduce the paper exactly" and "fit the paper's model
  correctly" are **different targets** here. We implement the first as the
  default and make the second reachable by an explicit flag.

---

## A3. Implementation

**New file: `py_core/paper_fast_gp.py`.** Reconstruction only — no thresholding,
no markers, no watershed, no cleanup, no metrics.

| API | role |
|---|---|
| `SharedParams` | `(beta1, beta2, nugget)` + provenance, `neg_log_lik`, `n_obj_evals`, `runtime_sec`; `effective_ranges_px()`; `degenerate_axes` |
| `TileReconstruction` | `pred_mean`, `theta_hat`, `s_2`, `sigma2_hat`, `runtime_sec`, `diagnostics` |
| `estimate_shared_params(tile, ...)` | port of `separable_GP_param_est` |
| `reconstruct_tile(tile, shared)` | port of `separable_GP` |
| `tile_grid(H, W)` | replicates `generate_GP_Masks_test`'s tiling incl. the uncovered remainder |
| `reconstruct_image(image, ...)` | whole image: first-tile estimation + reuse |

Requirements from the brief, and how each is met:

| requirement | status |
|---|---|
| separable row/column covariance | `R1 = matern_5_2(R01, beta1)`, `R2 = matern_5_2(R02, beta2)`, Kronecker eigendecomposition. **Done** |
| distinct beta1 and beta2 | two free parameters. **Done** (and the fits are strongly anisotropic in practice) |
| original parameterization | inverse-range `beta`, nugget on the eigenvalues, `log` scale, `param_ini = (-2,-2,-3)`, Matérn 5/2 `x = sqrt(5) beta d`. **Done** |
| shared params from the first tile | `reconstruct_image` estimates only when `shared is None`; `run_segmentation` holds `shared_pfg` across the tile loop. **Done, VERIFIED**: identical `beta1`/`nugget` on all 4 tiles of the wiring check |
| reuse on later tiles | as above. **Done** |
| tile-specific mean/scale preserved | `theta_hat` and `S_2` re-profiled per tile, matching L245. **Done, VERIFIED**: 4 distinct `theta_hat` values |
| all lattice pixels | no subsampling anywhere; no RNG in estimation. **Done** |
| Kronecker/eigendecomposition, not isotropic ExactGP | `scipy.linalg.eigh` on `R1`, `R2`; no GPyTorch in this path. **Done** |
| downstream segmentation outside | module contains none. **Done** |
| returns recon, shared params, tile params, diagnostics, runtime | all present. **Done** |

Kernel and objective are imported from `py_core/dim_2_lattice.py`
(`matern_5_2`, `neg_log_lik_eigen_with_nugget`) so the math has a single source
of truth.

**Degeneracy guard (new, not in the original).** `SharedParams.degenerate_axes`
flags any axis whose effective range falls below `DEGENERATE_RANGE_PX = 0.5`, and
`estimate_shared_params` raises a `RuntimeWarning`. It fired correctly on
whole_cell (`effective range rows = 7.887e-03 px`) and stayed silent on nuclei.
**VERIFIED.** It changes no numbers.

**Opt-in multi-start.** `n_restarts = 0` is the default and is paper-faithful.
`n_restarts > 0` searches for a better basin and is documented in the code as a
deliberate deviation.

### Renaming, without breaking saved results

`py_core/segmentation_pipeline.py`:

- `METHODS` gains `"paper_fast_gp"`.
- New `METHOD_LABELS` dict and `method_label(method, q=None)` helper map the
  **key** `"gp_legacy"` to the **display label** `"gp_isotropic_gpytorch_2025"`.
  The stored key is unchanged, so every existing `results/*.csv` still resolves.
  **VERIFIED**: `method_label("gp_legacy") -> "gp_isotropic_gpytorch_2025"`.
- The module docstring now states explicitly that `gp_legacy` is **not** the
  paper's Fast GP and points to `paper_fast_gp`.
- No new code refers to the arm as "gp_legacy" in user-facing text.

Nothing in `results/` was rewritten.

---

## A4. Smoke test — reconstruction only

`audits/taskA4_paper_fast_gp_smoke.py`; outputs in
`results/audit_paper_fast_gp_smoke_20260917/`. Two development images. The
isotropic arm is run on the **same tile grid** so the comparison is
tiling-matched. **No AP was computed.**

### Fitted shared parameters (tile (1,1)) — **VERIFIED**

| dataset | arm | beta1 | beta2 | nu | f | eff. range rows | eff. range cols | degenerate |
|---|---|---|---|---|---|---|---|---|
| nuclei | paper_fast_gp | 35.50395 | 23.68895 | 0.170597 | 116922.5219 | 7.915 px | 10.09 px | none |
| nuclei | paper_fast_gp_ms8 | 35.50395 | 23.68895 | 0.170597 | 116922.5219 | 7.915 px | 10.09 px | none |
| whole_cell | paper_fast_gp | 25232.17199 | 37.99262 | 0.002251 | 61119.2354 | **0.0079 px** | 5.238 px | **rows** |
| whole_cell | paper_fast_gp_ms8 | 81.01078 | 62.33656 | 0.001954 | **41016.7491** | 2.456 px | 3.192 px | none |

On nuclei the paper-faithful run already finds the best basin, and multi-start
changes nothing (`paper_fast_gp` vs `ms8`: RMSE **0.0000**, max|diff| **0.0000**).
On whole_cell it does not.

### Reconstruction statistics — **VERIFIED**

| dataset | arm | runtime | RMSE-to-raw | max-to-raw | corr-to-raw | mean\|grad\| | rec std |
|---|---|---|---|---|---|---|---|
| nuclei | raw | — | 0.0000 | 0.0000 | 1.000000 | 4.9721 | 19.5631 |
| nuclei | gp_isotropic_gpytorch_2025 | **504.0 s** | 7.7356 | 69.2965 | 0.921729 | 1.1736 | 17.0753 |
| nuclei | paper_fast_gp | **3.5 s** | 5.0520 | 32.1148 | 0.966086 | 1.5431 | 18.8361 |
| nuclei | paper_fast_gp_ms8 | 19.1 s | 5.0520 | 32.1148 | 0.966086 | 1.5431 | 18.8361 |
| whole_cell | raw | — | 0.0000 | 0.0000 | 1.000000 | 8.1958 | 40.9612 |
| whole_cell | gp_isotropic_gpytorch_2025 | **273.3 s** | 9.8702 | 113.2645 | 0.973863 | 6.1990 | 36.8106 |
| whole_cell | paper_fast_gp | **14.5 s** | 1.0652 | 16.8049 | 0.999662 | 8.0873 | 40.9056 |
| whole_cell | paper_fast_gp_ms8 | 55.7 s | 0.5899 | 9.6843 | 0.999896 | 8.1471 | 40.9427 |

### Reconstructions compared DIRECTLY to each other — **VERIFIED**

Round 2 established that similar RMSE-to-raw does not establish similarity, so
the pairwise table is the one that matters:

| dataset | pair | RMSE | max\|diff\| |
|---|---|---|---|
| nuclei | gp_isotropic_gpytorch_2025 vs paper_fast_gp | **5.6854** | **56.2128** |
| nuclei | paper_fast_gp vs paper_fast_gp_ms8 | 0.0000 | 0.0000 |
| whole_cell | gp_isotropic_gpytorch_2025 vs paper_fast_gp | **9.6876** | **110.4061** |
| whole_cell | paper_fast_gp vs paper_fast_gp_ms8 | 0.8340 | 13.5310 |

The two GP arms produce **substantially different images** — differences of 56
and 110 grey levels on a 0–255 scale. They are not interchangeable, and the
round-1–4 "GP" arm cannot stand in for the paper's method.

Saved: `{nuclei,whole_cell}_reconstructions.npz`,
`figures/{nuclei,whole_cell}_reconstruction_compare.png` (reconstruction row +
signed difference row), `reconstruction_summary.csv`,
`shared_parameter_table.csv`, `per_tile_diagnostics.csv`.

### Two substantive observations — **VERIFIED, no AP claim attached**

1. **`paper_fast_gp` is 19–144x faster** than the isotropic arm (3.5 s vs 504 s;
   14.5 s vs 273 s) because it uses the Kronecker structure instead of a generic
   6000-point ExactGP with Adam.
2. **On whole_cell the paper's GP barely smooths at all** (`corr-to-raw =
   0.999662`, `nu = 0.00225`). Whatever the downstream result turns out to be,
   `paper_fast_gp` and `raw` will be nearly the same input on that image. Flagged
   because it bears directly on how informative a Raw-vs-Fast-GP comparison can
   be there. **Not** an AP prediction.

### End-to-end wiring check (400 x 400 crop, no AP claim) — **VERIFIED**

```
Raw             0.1s  n_instances=160  recon=[0.00,229.00]
paper_fast_gp   2.6s  n_instances=146  recon=[0.08,229.35]   recon_s=2.54
  tile0: beta1=25232.1720 beta2=37.9926 nu=0.002251 degen=rows shared_from=tile_index=0
  shared params identical on all 4 tiles: True
  theta_hat re-profiled per tile: [0.21307, 0.13844, 0.193, 0.178]
  input_scale=255.0
```

Confirms first-tile estimation, cross-tile reuse, per-tile mean re-profiling,
the degeneracy warning, and scale round-tripping. Instance counts are printed
only to show the path executes.

---

## Answers to the four A4 questions

**1. Was real-tile R/Python parity demonstrated?**

**Partly, and the split matters.**

- **YES for the reconstruction given parameters** — machine precision on both
  real tiles (`max|diff|` 5.3e-15 and 5.3e-14; `theta_hat` 2.9e-15; `S_2`
  7.4e-13). **VERIFIED.**
- **YES end-to-end on whole_cell** — `max|diff| = 1.953e-06`, corr
  `1.000000000000`. **VERIFIED.**
- **NO end-to-end on nuclei** — `max|diff| = 4.258e-02`, corr 0.993367,
  because R's optimizer stops at a point 12388 worse in its own objective.
  **VERIFIED.** This is an optimization difference on a non-convex objective, not
  a translation error, and the evidence for that is A2.1 + the objective
  cross-evaluation.
- **NOT TESTED**: `criterion_1`, outlier-tile handling, `distmap`, `watershed`,
  `eliminate_small_areas` end-to-end, and AP — EBImage is missing. **UNVERIFIED.**

**2. Exact differences remaining**

Inside the reconstruction stage:

| # | difference | severity |
|---|---|---|
| 1 | On nuclei our optimizer finds a better optimum than R's; R's fit is degenerate in `beta2`. Same objective, different basin. | material, and arguably in our favour |
| 2 | Within the degenerate plateau `beta` is unidentifiable (R reported `beta2` values spanning 7e5–5.7e6 at identical `f`; our whole_cell `beta1` = 25232 vs R's 3.8e7 at identical `f = 61119.2354`). Reconstructions still agree to 2e-6. | cosmetic for the reconstruction, but means reported `beta` is not comparable |
| 3 | Uncovered remainder strip: the original leaves it 0; we default to `remainder="raw"` (pass raw pixels through) to avoid injecting a dark edge. `remainder="zero"` restores the original. | deliberate, documented, switchable |
| 4 | `run_segmentation`'s tiling lets the LAST tile absorb the remainder, whereas the original always crops a fixed size. Pre-existing, shared by every arm, so it does not bias the comparison. Not changed. | neutral for comparisons |
| 5 | We do not compute `separable_GP`'s finite-difference gradients. The original computes and never uses them. | none |
| 6 | Our retry-on-nonfinite uses a seeded `default_rng(1)` where R uses `set.seed(1)` + `runif`. Only reachable when `optim` errors; never triggered here. | none observed |

Outside the reconstruction stage (unchanged this session, carried over):
`criterion_1` uses `gaussian_filter1d` not RobustGaSP; outlier-tile handling is
still absent; watershed is skimage, not EBImage's tolerance-merging version.

**3. Is `paper_fast_gp` suitable for a later fair benchmark?**

**Yes, for the reconstruction stage, with two conditions stated up front.**

Suitable because: the model, parameterization, objective, Kronecker computation,
first-tile-reuse structure and per-tile mean profiling are all reproduced, the
linear algebra is exact to 5e-15, it is deterministic (no RNG), it is fast enough
to run everywhere (3.5 s/image), and it is wired into the same downstream code as
every other arm.

Conditions:

1. **Report which basin was used.** Default (`n_restarts=0`) is paper-faithful
   and degenerate on whole_cell; `n_restarts>0` fits the model better but is not
   the paper's estimator. Running both is cheap and is the honest option.
2. **It is the reconstruction stage only.** It does not make the *pipeline*
   equal to the paper's, because `criterion_1`, outlier handling and the
   watershed still differ. A benchmark using it can compare **reconstructions
   under our fixed downstream**; it cannot yet claim to reproduce the paper's
   published AP.

**4. Exact code paths added/modified**

Added (new files):

- `py_core/paper_fast_gp.py`
- `audits/taskA2_real_tile_r_side.R`
- `audits/taskA2_real_tile_py_side.py`
- `audits/taskA2_objective_landscape.py`
- `audits/taskA4_paper_fast_gp_smoke.py`
- `audits/PAPER_FAST_GP_WIRING_REPORT.md`
- `results/audit_paper_fast_gp_smoke_20260917/` (new directory)
- `audits/parity/real_*` and `real_tile_*` CSVs

Modified (one file, four localized edits):

- `py_core/segmentation_pipeline.py`
  - module docstring: corrected the `gp_legacy` description, documented `paper_fast_gp`
  - `METHODS` + new `METHOD_LABELS` / `method_label()`
  - `run_segmentation`: new `paper_fast_gp_restarts` keyword, `shared_pfg`
    accumulator, and a `paper_fast_gp` branch in the per-tile dispatch
  - new import of `estimate_shared_params`, `reconstruct_tile`

**Unchanged, as required:** thresholding (`criterion_1`, `foreground_threshold`),
marker generation (`instance_separation`), watershed, `eliminate_small_areas`,
all metrics (`segmentation_eval`), and every existing arm's numerics. No
`results/` file from Rounds 1–4 was modified or deleted.
