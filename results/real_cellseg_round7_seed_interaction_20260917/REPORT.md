# Round 7 — Is Fast-GP's paper-pipeline benefit mediated by watershed seeding?

2026-09-17. Two development images. **Nothing refit** — every reconstruction,
RobustGaSP threshold, binary foreground and EBImage distance map is read from
the Round-6 R dumps. Rounds 1–6 preserved.

## Step 4 decision — stated, not hidden

`EBImage::watershed(x, tolerance = 1, ext = 1)` has **no seed/marker argument**;
it cannot accept external markers. `EBImage::propagate(x, seeds, mask, lambda)`
does take seeds but is Voronoi-like propagation, not watershed flooding.

So B\*/D\* use **skimage watershed with explicit markers on the same flooding
surface and mask**:

| stage | B\*/D\* |
|---|---|
| elevation | `-(EBImage distmap from R)` — EBImage floods maxima of the positive map, skimage floods minima, so the surface is negated. The R distmap is reused verbatim, **not recomputed**. |
| mask | the R paper-threshold binary foreground, asserted byte-identical to B/D |
| cleanup | the paper's `eliminate_small_areas(., 50)` |

**This is a controlled seeding intervention, not a fully paper-faithful EBImage
run.** `EBImage::propagate` with the identical seeds is reported separately as an
EBImage-native cross-check.

## Step 1 — reuse verified

- B and D reproduce Round-6 `ap50`, `ap75` and `fg_dice` to `< 1e-12` (asserted).
- Our Python `eliminate_small_areas` port reproduces R's cleanup **partition
  exactly** on all four cells (`py-cleanup == R: True`), validated by applying it
  to Round-6's saved pre-cleanup labels and comparing to R's saved final labels.
- Basin counts reproduced from saved Round-6 data, confirming the earlier
  observation: nuclei raw **2846** pre-cleanup basins, Fast-GP **484**.

## Step 2 — marker definition (frozen, not tuned here)

From `results/real_cellseg_round3_thresholding_20260916/frozen_config.json`,
chosen before this experiment and preserved at dataset level:

| setting | value |
|---|---|
| implementation | `skimage.feature.peak_local_max` via `py_core.instance_separation.make_peak_markers` |
| source surface | **EBImage distmap from R**, reused verbatim |
| `min_distance` | **nuclei 15, whole_cell 9** |
| units | pixels, `p_norm = inf` (Chebyshev) |
| `threshold_abs` / `threshold_rel` | `None` / `None` |
| `exclude_border` | `False` |
| mask restriction | `labels =` paper binary foreground |
| markerless-component fallback | one marker at that component's distance-transform argmax (first in C raster order) |
| watershed connectivity | 1 |
| label construction | each peak gets its own integer id; peaks are never fused by `ndi.label` |
| GT used to generate/tune markers | **No** |
| tuned per arm | **No** — identical settings for Raw and Fast-GP |

## Results

### nuclei_figure_1 — GT = 330

| cell | AP@0.5 | AP@0.75 | Dice | TP | FP | FN | #pred | merge | split | missed | spurious | mean matched IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B Raw, implicit | 0.4015 | 0.1478 | 0.7619 | 218 | 213 | 112 | 431 | 15 | 8 | 112 | 152 | 0.5703 |
| B\* Raw, explicit | **0.5082** | 0.1676 | 0.7619 | 216 | 95 | 114 | 311 | 26 | 2 | 114 | 35 | 0.5694 |
| D FastGP, implicit | 0.6481 | **0.2840** | 0.8504 | 256 | 65 | 74 | 321 | 22 | 1 | 74 | 23 | **0.6693** |
| D\* FastGP, explicit | **0.6640** | 0.2782 | 0.8504 | 253 | 51 | 77 | 304 | 28 | 0 | 77 | 11 | 0.6641 |

### whole_cell_figure_1 — GT = 403

| cell | AP@0.5 | AP@0.75 | Dice | TP | FP | FN | #pred | merge | split | missed | spurious | mean matched IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B Raw, implicit | **0.4291** | 0.1303 | 0.8141 | 224 | 119 | 179 | 343 | 84 | 20 | 179 | 36 | 0.5235 |
| B\* Raw, explicit | 0.4168 | 0.1330 | 0.8141 | 218 | 120 | 185 | 338 | 84 | 25 | 185 | 28 | 0.5222 |
| D FastGP, implicit | **0.5839** | **0.1711** | 0.8584 | 275 | 68 | 128 | 343 | 68 | 12 | 128 | 19 | **0.5918** |
| D\* FastGP, explicit | 0.5285 | 0.1463 | 0.8584 | 260 | 89 | 143 | 349 | 68 | 22 | 143 | 17 | 0.5790 |

Foreground Dice is **identical** within each column pair by construction (the
binary is untouched; asserted in code). That is what makes the decomposition
below interpretable.

## Step 5 — seeds, components and basins

| dataset | cell | fg components | markers | basins pre-cleanup | final instances |
|---|---|---|---|---|---|
| nuclei | B | **4123** | — (implicit) | **2846** | 431 |
| nuclei | B\* | **4123** | **4173** (3778 rescued) | 4173 | 311 |
| nuclei | D | **422** | — (implicit) | **484** | 321 |
| nuclei | D\* | **422** | **466** (152 rescued) | 466 | 304 |
| whole_cell | B | 366 | — | 577 | 343 |
| whole_cell | B\* | 366 | 600 (220 rescued) | 600 | 338 |
| whole_cell | D | 314 | — | 474 | 343 |
| whole_cell | D\* | 314 | 495 (111 rescued) | 495 | 349 |

**The decisive number is the foreground component count, not the basin count.**
The nuclei raw binary is shattered into **4123 connected components for 330 true
cells**; the Fast-GP binary has **422**. Explicit markers cannot repair that —
each component still needs at least one seed, so B\* ends up with *more* seeds
(4173) than B had basins (2846). The final count only falls to 311 because the
paper's `eliminate_small_areas(., 50)` then deletes the fragments.

## Step 7 — the mechanism test

`Delta_implicit = D − B`, `Delta_explicit = D* − B*`,
`seed-mediated interaction = Delta_implicit − Delta_explicit`.

### nuclei

| metric | B | D | B\* | D\* | D−B | D\*−B\* | interaction | % of D−B removed |
|---|---|---|---|---|---|---|---|---|
| **AP@0.5** | 0.4015 | 0.6481 | 0.5082 | 0.6640 | **+0.2466** | **+0.1558** | +0.0908 | **36.8%** |
| AP@0.75 | 0.1478 | 0.2840 | 0.1676 | 0.2782 | +0.1362 | +0.1106 | +0.0256 | 18.8% |
| Dice | 0.7619 | 0.8504 | 0.7619 | 0.8504 | +0.0884 | +0.0884 | **+0.0000** | **0.0%** |
| merge | 15 | 22 | 26 | 28 | +7 | +2 | +5 | 71.4% |
| split | 8 | 1 | 2 | 0 | −7 | −2 | −5 | 71.4% |

### whole_cell

| metric | B | D | B\* | D\* | D−B | D\*−B\* | interaction | % of D−B removed |
|---|---|---|---|---|---|---|---|---|
| **AP@0.5** | 0.4291 | 0.5839 | 0.4168 | 0.5285 | **+0.1547** | **+0.1116** | +0.0431 | **27.9%** |
| AP@0.75 | 0.1303 | 0.1711 | 0.1330 | 0.1463 | +0.0408 | +0.0133 | +0.0275 | 67.4% |
| Dice | 0.8141 | 0.8584 | 0.8141 | 0.8584 | +0.0443 | +0.0443 | **+0.0000** | **0.0%** |
| merge | 84 | 68 | 84 | 68 | −16 | −16 | **+0.0000** | **0.0%** |
| split | 20 | 12 | 25 | 22 | −8 | −3 | −5 | 62.5% |

### Effect of explicit seeding alone

| dataset | change | AP@0.5 | AP@0.75 | #pred | merge | split | spurious |
|---|---|---|---|---|---|---|---|
| nuclei | B\* − B | **+0.1068** | +0.0198 | 431→311 | 15→26 | 8→2 | 152→35 |
| nuclei | D\* − D | +0.0159 | −0.0058 | 321→304 | 22→28 | 1→0 | 23→11 |
| whole_cell | B\* − B | **−0.0123** | +0.0027 | 343→338 | 84→84 | 20→25 | 36→28 |
| whole_cell | D\* − D | **−0.0554** | −0.0248 | 343→349 | 68→68 | 12→22 | 19→17 |

## Step 8 — marker count vs marker placement

GT used **only to describe** markers that were already generated without it.
Nothing here fed back into marker generation or parameter selection.

| dataset | cell | markers | fg comps | markers/comp (mean) | comps >1 marker | % GT cells ≥1 marker | **markers per GT cell** | GT cells 0 markers | GT cells >1 marker |
|---|---|---|---|---|---|---|---|---|---|
| nuclei | B\* | 4173 | 4123 | 1.012 | 31 | 98.2% | **6.524** | 6 | **274** |
| nuclei | D\* | 466 | 422 | 1.104 | 33 | 91.2% | **1.027** | 29 | **28** |
| whole_cell | B\* | 600 | 366 | 1.639 | 38 | 73.7% | 0.804 | **106** | 27 |
| whole_cell | D\* | 495 | 366→314 | 1.576 | 68 | 81.4% | 0.891 | **75** | 30 |

**Placement, not count, is where the arms differ on nuclei.** Markers per GT
cell: **6.524** for raw vs **1.027** for Fast-GP — essentially one seed per cell,
which is the ideal. 274 of 330 GT cells receive more than one raw marker versus
28 with Fast-GP.

On whole-cell the failure is the opposite: **under**-seeding. 106 of 403 GT cells
(26%) get no marker at all under raw and 75 (19%) under Fast-GP, because
`min_distance = 9` on a fragmented foreground cannot place a seed in every cell.
That is why explicit markers *hurt* whole-cell (−0.0123, −0.0554) while helping
nuclei raw (+0.1068).

## Secondary — EBImage-native `propagate` with the identical seeds

A different algorithm (Voronoi-like propagation, not flooding), so a cross-check
only, not a paper-faithful watershed.

| dataset | cell | AP@0.5 | AP@0.75 | Dice | #pred | merge | split |
|---|---|---|---|---|---|---|---|
| nuclei | B\*prop | 0.0400 | 0.0073 | 0.7619 | 917 | 7 | 49 |
| nuclei | D\*prop | 0.5985 | 0.2140 | 0.8504 | 311 | 33 | 9 |
| whole_cell | B\*prop | 0.2964 | 0.0741 | 0.8141 | 380 | 105 | 67 |
| whole_cell | D\*prop | 0.4179 | 0.0777 | 0.8584 | 374 | 89 | 62 |

`D*prop − B*prop` = **+0.5585** (nuclei), **+0.1215** (whole-cell). The Fast-GP
advantage survives — and on nuclei grows — under an entirely different
EBImage-native seeded algorithm. So the persistence of the advantage in B\*/D\*
is not an artifact of substituting skimage for EBImage.

## Scope limit

**One image per dataset.** No variance, no significance test, no generalization.
Percentages like "36.8% of the gap" are descriptive arithmetic on two numbers,
not estimates with uncertainty.

---

# Answers

## 1. Does explicit marker control improve Raw under the paper downstream?

**Mixed — yes on nuclei, no on whole-cell.**

- nuclei: 0.4015 → **0.5082**, **+0.1068** AP@0.5. Spurious detections collapse
  152 → 35 and #pred 431 → 311, because the seeds are now placed on distance
  peaks rather than found by tolerance on a noisy surface. Splits 8 → 2.
- whole_cell: 0.4291 → 0.4168, **−0.0123**. Explicit markers slightly hurt,
  because `min_distance = 9` leaves 106 of 403 GT cells with no seed at all.

## 2. Does explicit marker control improve FastGP under the paper downstream?

**Essentially no, and on whole-cell it hurts.**

- nuclei: 0.6481 → 0.6640, **+0.0159** on AP@0.5 but **−0.0058** on AP@0.75.
  Negligible and not consistent across thresholds.
- whole_cell: 0.5839 → 0.5285, **−0.0554**, with splits rising 12 → 22.

Read together with answer 1: explicit markers help the arm that needs seed
discipline (raw) and do nothing for the arm that already has it (Fast-GP). That
is the signature of a shared mechanism — but a partial one, see answer 7.

## 3. What happens to D − B after implicit seeding is replaced by explicit seeding?

It **shrinks but remains strongly positive**:

| dataset | D − B (implicit) | D\* − B\* (explicit) | absorbed by seeding | remaining |
|---|---|---|---|---|
| nuclei | **+0.2466** | **+0.1558** | +0.0908 (**36.8%**) | **63.2%** |
| whole_cell | **+0.1547** | **+0.1116** | +0.0431 (**27.9%**) | **72.1%** |

So seed control accounts for roughly a third of the nuclei advantage and a
quarter of the whole-cell advantage. The majority survives.

## 4. How much of the original FastGP advantage is explained by each sub-mechanism?

| sub-mechanism | nuclei | whole_cell | evidence |
|---|---|---|---|
| **fewer excessive seeds** | **36.8%** of the AP@0.5 gap | **27.9%** | the `D−B` vs `D*−B*` contraction above |
| **better marker placement** | **the dominant part of that 36.8%** | small | markers per GT cell 6.524 → 1.027; GT cells with >1 marker 274 → 28. The raw arm's problem is not the total (4173 markers for 4123 components ≈ 1.012 per component) but that those components are cell *fragments*, so a single cell collects ~6.5 seeds |
| **fewer splits** | 71.4% seed-mediated (D−B = −7 → D\*−B\* = −2) | 62.5% (−8 → −3) | split columns above |
| **fewer merges** | **not a Fast-GP benefit at all** — merges *rise* with Fast-GP (15 → 22), and rise further under explicit seeds (26 → 28) | **−16, and 100% seed-INdependent** (interaction exactly 0.0000) | merge columns above |
| **foreground / threshold change** | **0% seed-mediated, and it is the largest single component** | same | Dice +0.0884 / +0.0443 with interaction **exactly 0.0000**; fg components 4123 → 422 (nuclei), 366 → 314 (whole-cell) |

The cleanest single statement: the **entire foreground-quality advantage is
seed-independent by construction**, and it is large. On nuclei the reconstruction
reduces foreground fragmentation by a factor of ~10 (4123 → 422 components), and
no marker strategy can undo that, because seeds cannot merge separate components.

## 5. On nuclei, does the enormous Raw-vs-FastGP basin difference disappear under explicit marker control?

**No. It does not disappear — it is relocated, and in raw's case it gets worse.**

| | raw | FastGP |
|---|---|---|
| basins, implicit | 2846 | 484 |
| markers, explicit | **4173** | **466** |
| fg components | **4123** | **422** |

Explicit seeding does not equalise the counts because the disparity is not a
seeding artifact: the **raw thresholded foreground itself has 4123 connected
components for 330 cells**. Every component must receive at least one seed
(3778 of them needed the markerless-component fallback), so explicit control
*raises* raw's seed count from 2846 to 4173. What rescues B\* is not fewer seeds
but the paper's own small-area cleanup deleting the fragments (4173 → 311 final).

This reframes the Round-6 basin observation: the 2846-vs-484 gap was a
**symptom of foreground fragmentation**, not a property of EBImage's seeding.

## 6. On whole-cell, is the FastGP advantage seed-mediated, or is there a residual reconstruction benefit?

**Largely residual.** Only 27.9% of the AP@0.5 advantage is absorbed by explicit
seeding; **72.1% remains**. Two components are provably seed-independent:

- foreground Dice +0.0443 with interaction **exactly 0.0000**;
- the merge advantage −16 with interaction **exactly 0.0000** — Fast-GP prevents
  16 merges, and it does so identically under both seeding regimes.

Whole-cell is also the dataset where explicit seeding is actively harmful
(−0.0554 for Fast-GP), because the frozen `min_distance = 9` under-seeds a
fragmented foreground. So on whole-cell the paper's implicit seeding is not the
weak link at all.

## 7. Does this support "Fast-GP helps the original pipeline mainly because smoothing stabilizes automatic watershed seeding"?

**PARTIALLY SUPPORTED.**

Quantitatively, the word that fails is **"mainly"**:

- Seeding accounts for **36.8%** (nuclei) and **27.9%** (whole-cell) of the
  AP@0.5 advantage. The majority — **63.2%** and **72.1%** — survives explicit
  seed control.
- The foreground-quality advantage has a seeding interaction of **exactly
  0.0000** on both datasets: Dice +0.0884 and +0.0443 are untouched by the
  intervention, by construction.
- The whole-cell merge advantage (−16) is likewise **exactly 0.0000**
  seed-mediated.
- The advantage persists, and on nuclei nearly doubles (+0.5585), under a
  completely different EBImage-native seeded algorithm (`propagate`).

What *is* supported: seed stabilization is a **real and substantial** channel —
about a third of the effect on nuclei, most of the split-error reduction
(71.4% / 62.5%), and the reason explicit markers rescue raw nuclei by +0.1068.

What is **not** supported: that it is the main channel. The larger channel is
that Fast-GP changes **which pixels pass the paper's per-tile
`percentage * max(tile)` threshold**, producing a foreground that is both more
accurate (Dice) and ~10x less fragmented (4123 → 422 components on nuclei). Seeds
operate *within* components and can never repair that.

## 8. What ONE remaining mechanism is needed to explain the residual benefit?

**Foreground formation under the paper's per-tile proportional threshold.**

The paper thresholds each tile at `percentage * max(tile)` with the percentage
chosen by the RobustGaSP-smoothed criterion curve. On a raw tile, pixel noise
both (a) inflates `max(tile)`, shifting every candidate threshold, and (b) makes
the retained foreground speckled — hence 4123 components for 330 cells. Fast-GP's
smoothing removes that noise *before* the threshold is selected, so the same rule
yields a cleaner and far more connected foreground.

This is the single residual mechanism, and it is directly testable with one cheap
transplant experiment that needs no refitting: **hold the seeding regime fixed
and swap only the foreground** — run the paper downstream on the Fast-GP binary
with the raw reconstruction's distance map and vice versa, or more simply score
`raw + Fast-GP's binary` against `raw + raw's binary`. If the residual 63–72%
transfers with the binary, foreground formation is confirmed as the dominant
channel.

Not run. Reporting the seed-interaction result and stopping.
