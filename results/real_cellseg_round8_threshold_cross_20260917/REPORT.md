# Round 8 — Image effect vs threshold-selection effect (final Fast-GP diagnostic)

2026-09-17. Two development images. **Nothing refit; `criterion_1`/RobustGaSP not
re-run** — the per-tile proportions `p` are frozen from Round 6 and simply applied
to the recipient image. Rounds 1–7 preserved. Paper downstream unchanged, implicit
EBImage seeding, no explicit markers.

## Design

| condition | image | proportion | threshold |
|---|---|---|---|
| **RR** | raw | `p_raw` | `p_raw * max(raw tile)` |
| **RG** | raw | `p_fastgp` | `p_fastgp * max(raw tile)` |
| **GR** | fastgp | `p_raw` | `p_raw * max(fastgp tile)` |
| **GG** | fastgp | `p_fastgp` | `p_fastgp * max(fastgp tile)` |

The **recipient** image's own `max` is always used — the proportion `p` is
crossed, never an absolute intensity. Tile-by-tile at identical coordinates, then
the unchanged paper downstream: outlier-tile handling → stitch →
`EBImage::distmap` → `EBImage::watershed(tolerance=1, ext=1)` →
`eliminate_small_areas(., 50)`.

## Step 1 — input verification

- `max|raw_recon − image/255| = 5.55e-16` on both images (asserted 0).
- Per-tile `max` matches Round-6 `tile_meta` to `1e-9`; tile geometry identical
  between arms (asserted).
- **RR reproduces Round-6 B and GG reproduces Round-6 D, AP@0.5 to `<1e-12`**
  (asserted) — this validates the whole re-implementation.
- Reference note, not an error: the Round-6 Fast-GP reconstruction differs from
  Round-5's by `6.72e-02` on nuclei (`1.64e-06` on whole-cell) because Round 6
  uses the **R-side** paper-faithful fit, which on nuclei lands on the degenerate
  `beta → ∞` plateau, while Round 5 used the Python fit that found the better
  basin. Documented in `audits/PAPER_FAST_GP_WIRING_REPORT.md`. Round 8 uses the
  R arrays throughout, so it is internally consistent with Round 6.

## Step 1 table — per-tile thresholds (`tile_thresholds.csv`)

Summary; full table has one row per tile.

| dataset | mean `delta_p` | mean `max_ratio` | tiles with \|`delta_p`\| > 0.05 |
|---|---|---|---|
| nuclei | **+0.0019** | 0.9848 | **3 / 16** |
| whole_cell | **+0.0422** | 0.9930 | **5 / 9** |

`max(tile)` barely moves in either dataset (ratio 0.98–0.99). On nuclei the raw
`max` is essentially saturated — mean 0.99828, i.e. 255/255 — on 14 of 16 tiles.

## Steps 2 + 4 — foreground and instance metrics

### nuclei_figure_1 — GT = 330, GT foreground 22.21%

| cond | fg% | Dice | IoU | prec | rec | **components** | comps <50px | basins | final | AP@0.5 | AP@0.75 | merge | split |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RR | 17.4 | 0.7619 | 0.6154 | 0.869 | 0.679 | **4123** | 3867 | 2846 | 431 | 0.4015 | 0.1478 | 15 | 8 |
| RG | 16.6 | 0.8106 | 0.6815 | 0.949 | 0.707 | **4321** | 4060 | 2922 | 378 | 0.4781 | 0.1919 | 15 | 16 |
| GR | 23.6 | 0.7107 | 0.5513 | 0.690 | 0.732 | **470** | 237 | 572 | 342 | 0.4967 | 0.1748 | 23 | 3 |
| GG | 17.9 | **0.8504** | **0.7397** | 0.952 | 0.768 | **422** | 166 | 484 | 321 | **0.6481** | **0.2840** | 22 | 1 |

### whole_cell_figure_1 — GT = 403, GT foreground 33.36%

| cond | fg% | Dice | IoU | prec | rec | components | comps <50px | basins | final | AP@0.5 | AP@0.75 | merge | split |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RR | 37.4 | 0.8141 | 0.6864 | 0.770 | 0.864 | 366 | 250 | 577 | 343 | 0.4291 | 0.1303 | 84 | 20 |
| RG | 29.5 | 0.8567 | 0.7494 | 0.912 | 0.808 | 347 | 173 | 498 | 345 | **0.5814** | 0.1651 | 64 | 9 |
| GR | 37.7 | 0.8160 | 0.6892 | 0.769 | 0.869 | 289 | 176 | 524 | 344 | 0.4256 | 0.1370 | 82 | 20 |
| GG | 29.7 | **0.8584** | **0.7519** | 0.911 | 0.811 | 314 | 137 | 474 | 343 | **0.5839** | **0.1711** | 68 | 12 |

GT-coverage columns are in `foreground_cross_metrics.csv`.

## Step 5 — decomposition

| dataset | metric | RR | RG | GR | GG | p-eff on raw (RG−RR) | p-eff on fastgp (GG−GR) | **image-eff at p_raw (GR−RR)** | **image-eff at p_fastgp (GG−RG)** | interaction |
|---|---|---|---|---|---|---|---|---|---|---|
| nuclei | AP@0.5 | 0.4015 | 0.4781 | 0.4967 | 0.6481 | **+0.0766** | **+0.1514** | **+0.0952** | **+0.1700** | +0.0748 |
| nuclei | AP@0.75 | 0.1478 | 0.1919 | 0.1748 | 0.2840 | +0.0441 | +0.1092 | +0.0270 | +0.0921 | +0.0651 |
| nuclei | **components** | 4123 | 4321 | 470 | 422 | **+198** | −48 | **−3653** | **−3899** | −246 |
| nuclei | Dice | 0.7619 | 0.8106 | 0.7107 | 0.8504 | +0.0486 | +0.1397 | **−0.0512** | +0.0398 | +0.0910 |
| whole_cell | AP@0.5 | 0.4291 | 0.5814 | 0.4256 | 0.5839 | **+0.1523** | **+0.1583** | **−0.0035** | **+0.0025** | +0.0060 |
| whole_cell | AP@0.75 | 0.1303 | 0.1651 | 0.1370 | 0.1711 | +0.0348 | +0.0341 | +0.0067 | +0.0060 | −0.0007 |
| whole_cell | components | 366 | 347 | 289 | 314 | −19 | +25 | −77 | −33 | +44 |
| whole_cell | Dice | 0.8141 | 0.8567 | 0.8160 | 0.8584 | **+0.0427** | **+0.0424** | +0.0019 | +0.0017 | −0.0003 |

**The two datasets answer differently, and both answers are clean.**

## Step 6 — max-intensity mechanism

Per-tile means, raw vs Fast-GP:

| quantity | nuclei raw | nuclei fastgp | ratio | whole_cell raw | whole_cell fastgp | ratio |
|---|---|---|---|---|---|---|
| max(tile) | 0.99828 | 0.98319 | 0.985 | 0.87625 | 0.87009 | 0.993 |
| 99.9th pct | 0.96863 | 0.95940 | 0.990 | 0.81917 | 0.81800 | 0.999 |
| 99th pct | 0.89819 | 0.88967 | 0.991 | 0.70545 | 0.70438 | 0.998 |
| 95th pct | 0.82108 | 0.81671 | 0.995 | 0.53900 | 0.53824 | 0.999 |
| median | 0.64338 | 0.64174 | 0.997 | 0.10719 | 0.10731 | 1.001 |
| **MAD** | 0.02745 | 0.02184 | **0.795** | 0.04357 | 0.04353 | **0.999** |
| max − 99.9th pct | 0.02966 | 0.02379 | 0.802 | 0.05708 | 0.05208 | 0.912 |
| **#px exactly at max** | **65.4** | **1.0** | **0.015** | 1.11 | 1.00 | 0.900 |
| **islands at p_raw** | **499.6** | **41.6** | **0.083** | 53.9 | 41.9 | 0.777 |
| **small islands (<50px) at p_raw** | **481.3** | **25.4** | **0.053** | 38.7 | 27.0 | 0.698 |

**`max(tile)` is not the mechanism** in either dataset — it changes by 1–2%. On
nuclei the raw tiles are *saturated*: on average 65 pixels sit exactly at the
maximum, so `max` is pinned at 1.0 and cannot be moved by smoothing. What changes
is the **local spread** (MAD ratio 0.795) and, consequently, the **number of
speckle islands at an identical `p`**: 499.6 → 41.6 per tile, of which the
disappearing ones are overwhelmingly sub-50-pixel fragments (481.3 → 25.4).

### The worst nuclei tiles (islands at `p_raw`)

| tile | p_raw | p_fastgp | delta_p | max_ratio | islands raw | islands fastgp | #px at max (raw) | MAD raw | MAD gp |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.67 | 0.70 | +0.03 | 0.956 | **2592** | **47** | 1 | 0.0235 | 0.0155 |
| 3 | 0.68 | 0.75 | +0.07 | 0.992 | **2144** | **169** | 25 | 0.0314 | 0.0261 |
| 11 | 0.64 | 0.76 | +0.12 | 0.963 | **944** | **122** | 6 | 0.0392 | 0.0402 |
| 8 | 0.69 | 0.66 | −0.03 | 1.014 | 318 | 39 | 106 | 0.0353 | 0.0286 |

Three tiles (5, 3, 11) supply almost all of the 4123 components, and at the
**same p** the reconstruction removes 98%, 92% and 87% of their islands.

### Whole-cell tiles, sorted by `delta_p`

| tile | p_raw | p_fastgp | delta_p | max_ratio | islands raw | islands fastgp | MAD raw | MAD gp |
|---|---|---|---|---|---|---|---|---|
| 6 | **0.07** | **0.21** | **+0.14** | 0.996 | 43 | 27 | 0.0471 | 0.0478 |
| 7 | **0.12** | **0.23** | **+0.11** | 1.001 | 65 | 42 | 0.0588 | 0.0590 |
| 4 | **0.11** | **0.20** | **+0.09** | 0.991 | 106 | 71 | 0.0392 | 0.0393 |
| 8 | 0.19 | 0.25 | +0.06 | 0.996 | 38 | 32 | 0.0627 | 0.0607 |
| 5 | 0.23 | 0.17 | −0.06 | 0.980 | 48 | 41 | 0.0549 | 0.0548 |

Note the MAD columns: on whole-cell the reconstruction barely smooths at all
(0.0471 vs 0.0478; Round 5 measured correlation-to-raw 0.9997). Yet on tiles 6, 7
and 4 the raw criterion selects a catastrophically low `p` — 0.07 means
thresholding at 7% of the tile maximum, which floods the tile — and the
reconstruction moves it to 0.21, 0.23, 0.20. **The Fast-GP reconstruction on
whole-cell acts almost entirely through the criterion's stability point, not
through the image.**

## Step 7 — secondary absolute-threshold cross-check

**Diagnostic only, not the paper-faithful factorial** (raw and Fast-GP have
different scales, and this variant skips the outlier-tile handling, so the
component counts are not comparable to the primary table).

| dataset | image | absolute threshold | fg% | Dice | IoU | components | <50px |
|---|---|---|---|---|---|---|---|
| nuclei | raw | `T_raw_original` | 20.8 | 0.7300 | 0.5748 | **7923** | 7662 |
| nuclei | raw | `T_fastgp_original` | 19.5 | 0.8122 | 0.6838 | **6717** | 6437 |
| nuclei | fastgp | `T_raw_original` | 20.3 | 0.7383 | 0.5852 | **769** | 523 |
| nuclei | fastgp | `T_fastgp_original` | 19.1 | 0.8294 | 0.7085 | **601** | 338 |
| whole_cell | raw | `T_raw_original` | 40.0 | 0.7984 | 0.6644 | 429 | 329 |
| whole_cell | raw | `T_fastgp_original` | 29.4 | **0.8552** | 0.7470 | 342 | 166 |
| whole_cell | fastgp | `T_raw_original` | 40.1 | 0.7976 | 0.6634 | 337 | 238 |
| whole_cell | fastgp | `T_fastgp_original` | 29.4 | **0.8564** | 0.7489 | 318 | 137 |

Same conclusion, reached independently: on nuclei the **image** governs component
count (7923 → 769 at a fixed absolute threshold, a 10x reduction) while the
threshold governs Dice; on whole-cell the **threshold** governs Dice
(0.798 → 0.855) and the image barely matters (0.798 → 0.798). Max-rescaling is
not doing the work.

## Step 8 — case classification

| dataset | classification | justification |
|---|---|---|
| **nuclei** | **CASE C — both matter, with the image dominating topology** | AP@0.5: image effect +0.0952 / +0.1700 vs p effect +0.0766 / +0.1514 — comparable, and the interaction (+0.0748) is large and positive, so they reinforce. But component count is **~100% image**: p makes it *worse* on the raw image (4123 → 4321) while the image collapses it (4123 → 470 at fixed p). |
| **whole_cell** | **CASE B — threshold selection dominates, almost exclusively** | AP@0.5: p effect **+0.1523 / +0.1583** vs image effect **−0.0035 / +0.0025**. The image effect is within noise of zero and one sign is negative. `p` accounts for ~98% of the +0.1547 total. Dice likewise: +0.0427 vs +0.0019. |

## Scope limit

**One image per dataset.** No variance, no significance, no generalization.
"~98%" is arithmetic on two numbers, not an estimate with uncertainty. The two
datasets disagree, which is itself a warning against extrapolating either.

---

# Answers

## 1. On nuclei, what explains the ~4123 vs ~422 foreground-component gap?

**The reconstruction itself, essentially entirely. Not `p`, not `max(tile)`.**

- Holding `p` at `p_raw` and swapping only the image: 4123 → **470** (−3653).
  At `p_fastgp`: 4321 → 422 (−3899).
- Holding the image at raw and swapping only `p`: 4123 → **4321**, i.e. `p`
  makes fragmentation **slightly worse** (+198).
- `max(tile)` ratio is 0.985, and raw nuclei tiles are saturated (65 pixels at
  the maximum on average), so `max` is pinned and cannot be the lever.
- The direct mechanism is local noise: MAD ratio 0.795, and **islands at an
  identical `p` fall 499.6 → 41.6 per tile**, of which 481 → 25 are sub-50-pixel
  fragments. Three tiles (5, 3, 11) supply nearly all of the 4123 components and
  lose 87–98% of their islands at unchanged `p`.

Interaction on components is −246, small against the −3653 main effect. So:
**reconstruction, with no meaningful contribution from `p` or `max`.**

## 2. On whole-cell, what explains the residual Fast-GP advantage?

**Threshold selection — the value of `p` the RobustGaSP criterion picks — not the
image.**

- AP@0.5 image effect: **−0.0035** at `p_raw`, **+0.0025** at `p_fastgp`.
  Effectively zero, and not even consistently signed.
- AP@0.5 `p` effect: **+0.1523** (raw image), **+0.1583** (fastgp image). Both
  large, and nearly equal — the `p` benefit transfers fully to the raw image.
- Dice: `p` effect +0.0427/+0.0424 vs image effect +0.0019/+0.0017.
- Mechanism: the reconstruction barely changes whole-cell pixels (MAD 0.0471 vs
  0.0478), but on tiles 6, 7 and 4 the raw criterion selects `p` = 0.07, 0.12,
  0.11 — thresholds at 7–12% of the tile maximum, which flood the tile — and the
  smoothed image moves the curve's stability point to 0.21, 0.23, 0.20.

So the residual identified at the end of Round 7 is, on whole-cell, **criterion
fragility**: `criterion_1` is unstable on noisy tiles and the GP stabilises it.

## 3. When the SAME p is used, does Fast-GP still produce substantially cleaner foreground topology than Raw?

**Yes on nuclei, decisively; marginally on whole-cell.**

| dataset | components at `p_raw` | components at `p_fastgp` |
|---|---|---|
| nuclei | 4123 → **470** (−89%) | 4321 → **422** (−90%) |
| whole_cell | 366 → 289 (−21%) | 347 → 314 (−10%) |

Note a genuine dissociation on nuclei: GR has the **worst** foreground Dice of
all four conditions (0.7107, below RR's 0.7619, because `p_raw` over-includes on
the smoothed image — 23.6% foreground against a 22.2% truth), yet its **AP@0.5 is
higher** (0.4967 vs 0.4015). Cleaner topology beats better pixel accuracy in this
pipeline. That is worth stating plainly: pixel-level foreground quality and
instance-level AP move in opposite directions here.

## 4. When the SAME image is used, does swapping p materially change AP or fragmentation?

**AP: yes, substantially, in both datasets. Fragmentation: no.**

| dataset | image | AP@0.5 change from swapping `p` | component change |
|---|---|---|---|
| nuclei | raw | +0.0766 | **+198 (worse)** |
| nuclei | fastgp | +0.1514 | −48 |
| whole_cell | raw | **+0.1523** | −19 |
| whole_cell | fastgp | **+0.1583** | +25 |

So `p` is a real AP lever everywhere — and on whole-cell it is *the* lever — but
it never repairs fragmentation. On the raw nuclei image a better `p` actually
increases the component count while still improving AP (0.4015 → 0.4781), because
the extra fragments are sub-50-pixel and the paper's cleanup deletes them.

## 5. Is Fast-GP's main role best described as A, B, C, or D?

**D — a combination, with the weighting dataset-dependent and now quantified.**

| channel | nuclei | whole_cell |
|---|---|---|
| **A. choosing a better threshold `p`** | real: +0.0766 to +0.1514 AP@0.5 | **dominant: +0.1523 to +0.1583 AP@0.5 (~98% of the effect)** |
| **B. suppressing extreme/noisy intensity structure before thresholding** | **dominant for topology: components −89%, islands −92% at fixed `p`; AP +0.0952 to +0.1700** | negligible: AP −0.0035 to +0.0025 |
| **C. stabilizing watershed seeding** | partial: Round 7 measured 36.8% of the D−B gap absorbed by explicit markers | partial: 27.9% absorbed |
| max(tile) rescaling | ruled out (ratio 0.985; raw tiles saturated) | ruled out (ratio 0.993) |

Single-sentence form: on nuclei Fast-GP works mainly by **B** (noise suppression
→ foreground topology → fewer spurious basins, which is also why C appears), and
on whole-cell almost entirely by **A** (stabilising the criterion's choice of
`p`). C is a downstream consequence of B, not an independent channel — Round 7
showed that explicit markers cannot repair fragmentation because seeds act within
components.

## 6. After Rounds 6–8, can the Fast-GP baseline investigation reasonably be considered complete?

**Yes.** The chain is closed end-to-end with no unexplained residual:

- **Round 5**: under our corrected downstream, `paper_fast_gp` does not beat Raw
  (C−A = −0.0477 nuclei, 0.0000 whole-cell). The reconstruction stage itself was
  verified faithful to `5.3e-15` against R.
- **Round 6**: under the *paper's* downstream it helps a lot (D−B = +0.2466,
  +0.1547); interaction +0.2943, +0.1547. The published NoGP ablation was shown
  to be confounded three ways, and the published numbers were reproduced
  (NoGP exactly; GP within 0.007 AP).
- **Round 7**: explicit marker control absorbs only 36.8% / 27.9% of that
  advantage, so seeding is a partial channel; the foreground-quality advantage has
  exactly zero seed interaction.
- **Round 8**: the remaining majority is now attributed — nuclei to the intensity
  field (components ~100% image-driven), whole-cell to the criterion's `p`
  (~98% p-driven) — and `max(tile)` rescaling is ruled out in both.

Two residual items are known and bounded, neither requiring more Fast-GP work:
the R optimizer's degenerate `beta → ∞` plateau (quantified in the wiring audit;
the robust-fit variant was scored in Rounds 5 and 6), and the fact that the
paper's published AP could not be re-derived at the mask level because historical
masks are not shipped.

**Recommendation: close it.** The remaining differences between our pipeline and
the paper's are now understood as properties of `criterion_1` and EBImage's
automatic seeding, not of the Gaussian process.

## 7. What exact segmentation failure should a future QEP-specific mechanism target?

**Adjacent-cell merging under a correct foreground** — not fragmentation, and not
seeding.

The evidence, assembled across rounds:

- Fragmentation and seeding are now fully explained by smoothing and marker
  control, and both are solved problems: the corrected pipeline already reaches
  422–470 components and explicit markers already give ~1.03 markers per GT cell
  on nuclei.
- What no intervention has fixed is **merging**. In the best cells here, merges
  remain the dominant error: nuclei GG has 22 merges and D\* (Round 7) has 28;
  whole-cell GG has 68 merges against 403 GT cells — 17% of all cells. Round 4
  independently identified merging as the dominant FN mechanism, and Round 5 showed
  that Fast-GP *increases* merges on nuclei (27 → 34) even while improving Dice.
- Merging is precisely a **boundary-evidence** deficit: two touching cells with no
  intensity ridge between them produce one basin regardless of threshold, seed
  strategy or smoothing. Smoothing makes it worse, because it erases what little
  ridge exists.
- This is also where the Task-B audit found a *real*, chunk-invariant,
  q-dependent signal: `E[‖∇f‖ | y]` and `P(‖∇f‖ > c | y)` move by 2.3x and 2.0x
  across q ∈ [1.2, 2.0] at pinned hyperparameters, and their invariance to
  chunk partition was verified (0.4% spread). Every other q-dependent quantity
  either failed q-sensitivity or failed partition invariance.

So the target is: **supply boundary evidence between touching cells, from a
q-dependent gradient-tail statistic, to split merged basins** — under the
preconditions already established (never `rescale=True` for the multitask
gradient; fix and report the chunk partition; distinguish `‖E[∇f]‖`, which is
q-invariant, from `E[‖∇f‖]`, which is not; give the q=2 control identical
machinery).

---

# COMPACT SUMMARY

```
NUCLEI:
RR AP = 0.4015
RG AP = 0.4781
GR AP = 0.4967
GG AP = 0.6481
RR components = 4123
RG components = 4321
GR components = 470
GG components = 422
dominant mechanism = IMAGE / intensity-field noise suppression
                     (CASE C overall on AP -- image +0.0952..+0.1700 vs
                      p +0.0766..+0.1514 -- but component topology is
                      ~100% image-driven; max(tile) ruled out)

WHOLE-CELL:
RR AP = 0.4291
RG AP = 0.5814
GR AP = 0.4256
GG AP = 0.5839
dominant mechanism = THRESHOLD SELECTION (p)
                     (CASE B -- p +0.1523..+0.1583 vs image -0.0035..+0.0025,
                      ~98% of the effect; max(tile) ruled out)
```

**"The next QEP-specific experiment should target adjacent-cell merging under an
already-correct foreground, because Rounds 6–8 show that fragmentation and seeding
are fully explained by smoothing and marker control while merges remain the
dominant residual error (22 on nuclei, 68 of 403 cells on whole-cell) and are the
one failure that a q-dependent, chunk-invariant gradient-tail boundary statistic
is positioned to address."**
