# Round 5 — Corrected-baseline experiment

2026-09-17. Two **development** images only (`nuclei_figure_1`,
`whole_cell_figure_1`). Held-out images were not inspected or run. Every
previous result directory is intact; the only tracked source file modified in
this line of work is `py_core/segmentation_pipeline.py`.

Script: `experiments/real_data/round5_corrected_baseline.py`.
Log: `logs/round5.log`. Config: `round5_config.json`.

## Design

Reconstruction is the **only** thing that differs between arms. The downstream is
read from `results/real_cellseg_round3_thresholding_20260916/selected_threshold_config.json`
and applied byte-identically to all six arms — **nothing was retuned per arm**:

| stage | setting |
|---|---|
| threshold | Round-3 selected rule: **Li**, applied GLOBALLY to the stitched reconstruction |
| markers | `peak_local_max`, `min_distance` nuclei = 15, whole_cell = 9, `exclude_border=False`, Chebyshev |
| cleanup | `eliminate_small_areas(50)` |
| watershed | `-distance_transform_edt`, connectivity 1, mask = foreground |
| evaluation | `evaluate_instances`, AP = TP/(TP+FP+FN) |

**Caching.** `raw`, `gp_legacy`, `qep_q2`, `qep_q1.5` reuse the Round-3 cached
`predmean` arrays verbatim — no GP or QEP model was refitted. Verified
beforehand that shapes match and that the cached `raw` predmean is bit-identical
to the loaded image (`max|diff| = 0.0`). Only the two `paper_fast_gp` arms were
computed.

**Arm 3 is not the published algorithm.** `paper_fast_gp_robust` uses the same
statistical model with a **modified optimizer** (multi-start, `n_restarts = 8`).
It is labelled `paper_fast_gp (MODIFIED opt: multi-start)` in every table.

**Terminology.** `rmse_to_raw` is not reconstruction accuracy — no clean
ground-truth image exists. It only measures how far an arm moved from its input.

## Results

### nuclei_figure_1 — 1128 x 962, GT = 330

| arm | thr | fg% | Dice | AP@0.5 | AP@0.75 | TP | FP | FN | #pred | #GT | merge | split | recon s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Raw | 178.67 | 21.0 | 0.8393 | **0.6933** | **0.3568** | 260 | 45 | 70 | 305 | 330 | 27 | 0 | 1.8 | 7.1 |
| paper_fast_gp (paper-faithful opt) | 179.14 | 19.5 | **0.8785** | 0.6457 | 0.3397 | 246 | 51 | 84 | 297 | 330 | 34 | 3 | 3.4 | 3.7 |
| paper_fast_gp (MODIFIED opt) | 179.14 | 19.5 | **0.8785** | 0.6457 | 0.3397 | 246 | 51 | 84 | 297 | 330 | 34 | 3 | 18.5 | 18.8 |
| gp_isotropic_gpytorch_2025 | 169.06 | 39.9 | 0.6422 | **0.1499** | 0.0170 | 70 | 137 | 260 | 207 | 330 | 66 | 4 | 650.3 | 650.7 |
| QEP q=2 | 177.79 | 21.8 | 0.8569 | 0.6250 | 0.2973 | 240 | 54 | 90 | 294 | 330 | 38 | 2 | 649.6 | 649.9 |
| QEP q=1.5 | 177.77 | 21.9 | 0.8558 | 0.5985 | 0.2967 | 234 | 61 | 96 | 295 | 330 | 39 | 5 | 670.3 | 670.7 |

### whole_cell_figure_1 — 600 x 602, GT = 403

| arm | thr | fg% | Dice | AP@0.5 | AP@0.75 | TP | FP | FN | #pred | #GT | merge | split | recon s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Raw | 54.94 | 26.1 | 0.8275 | 0.5276 | 0.0795 | 258 | 86 | 145 | 344 | 403 | 67 | 15 | 0.2 | 0.3 |
| paper_fast_gp (paper-faithful opt) | 55.47 | 25.7 | 0.8231 | 0.5276 | 0.0702 | 258 | 86 | 145 | 344 | 403 | 67 | 19 | 2.5 | 2.6 |
| paper_fast_gp (MODIFIED opt) | 55.51 | 25.6 | 0.8224 | **0.5369** | 0.0745 | 262 | 85 | 141 | 347 | 403 | 65 | 17 | 11.2 | 11.3 |
| gp_isotropic_gpytorch_2025 | 50.71 | 29.1 | **0.8505** | 0.4168 | **0.1330** | 213 | 108 | 190 | 321 | 403 | 81 | 26 | 357.3 | 357.4 |
| QEP q=2 | 62.45 | 23.4 | 0.7881 | 0.4453 | 0.0358 | 232 | 118 | 171 | 350 | 403 | 62 | 18 | 359.5 | 359.6 |
| QEP q=1.5 | 59.64 | 24.7 | 0.8087 | 0.4503 | 0.0705 | 231 | 110 | 172 | 341 | 403 | 67 | 16 | 358.7 | 358.8 |

### Fitted `paper_fast_gp` parameters and optimizer diagnostics

| dataset | arm | beta1 | beta2 | nugget | objective f | obj evals | eff. range rows | eff. range cols | degenerate | recon s |
|---|---|---|---|---|---|---|---|---|---|---|
| nuclei | paper-faithful | 35.503951 | 23.688954 | 0.170597 | 116922.5219 | 112 | 7.915 px | 10.089 px | none | 3.41 |
| nuclei | MODIFIED | 35.503951 | 23.688954 | 0.170597 | 116922.5219 | 628 | 7.915 px | 10.089 px | none | 18.47 |
| whole_cell | paper-faithful | **25232.17** | 37.992622 | 0.002251 | 61119.2354 | 156 | **0.0079 px** | 5.238 px | **rows** | 2.50 |
| whole_cell | MODIFIED | **81.0108** | 62.336563 | 0.001954 | **41016.7491** | 668 | 2.456 px | 3.192 px | none | 11.20 |

**Which basin differs.** On **nuclei the two fits are the same point** — identical
`beta1`, `beta2`, `nugget` and objective, and the reconstructions are
bit-identical (corr 1.000000, RMSE 0.0000, max|diff| 0.0000). On **whole_cell they
differ materially**: the paper-faithful optimizer lands in the degenerate
`beta1 → ∞` plateau (row correlation range 0.0079 px, i.e. an axis-aligned
smoother that does not smooth across rows at all) at `f = 61119.2354`, while
multi-start finds an interior optimum at `f = 41016.7491` — better by **20102** in
the paper's own objective — with both axes on a sane pixel scale.

Tile-specific mean parameters are saved for all 9 whole-cell tiles and all 16
nuclei tiles (`paper_fast_gp_per_tile.csv`). They are re-profiled per tile as the
original does; whole_cell `theta_hat` ranges 0.1384–0.2131 across tiles.
`paper_fast_gp_fits/*.npz` holds the reconstruction, shared params, per-tile
`theta_hat` / `s_2` / `sigma2_hat`, tile offsets and runtime.

### Reconstruction-to-reconstruction diagnostics

Not accuracy — these compare arms to each other.

| dataset | pair | correlation | RMSE between | max\|diff\| |
|---|---|---|---|---|
| nuclei | Raw vs paper_fast_gp | 0.966086 | 5.0520 | 32.1148 |
| nuclei | paper_fast_gp vs gp_isotropic | 0.888734 | 8.8900 | 61.6720 |
| nuclei | paper-faithful vs MODIFIED | **1.000000** | **0.0000** | **0.0000** |
| whole_cell | Raw vs paper_fast_gp | 0.999662 | 1.0652 | 16.8049 |
| whole_cell | paper_fast_gp vs gp_isotropic | 0.945360 | 14.0494 | 143.2721 |
| whole_cell | paper-faithful vs MODIFIED | 0.999793 | 0.8340 | 13.5310 |

## Scope limit

**One image per dataset.** No variance estimate, no significance test, and no
held-out confirmation. Every statement below is about these two images under this
one frozen downstream. Differences of a few AP hundredths are not resolvable at
n = 1.

---

# Answers

## 1. Does the actual paper Fast-GP reconstruction improve segmentation over Raw on either development image?

**No — not on instance segmentation, on either image. It does improve foreground
segmentation on nuclei.**

The two metrics disagree, and that is the substantive finding:

| | nuclei | whole_cell |
|---|---|---|
| foreground Dice | **0.8785 vs 0.8393 = +0.0392 (better)** | 0.8231 vs 0.8275 = −0.0044 (tie/marginally worse) |
| AP@0.5 | 0.6457 vs 0.6933 = **−0.0476 (worse)** | 0.5276 vs 0.5276 = **exactly equal** |
| AP@0.75 | 0.3397 vs 0.3568 = −0.0171 (worse) | 0.0702 vs 0.0795 = −0.0093 (worse) |

On nuclei the mechanism is visible in the counts: Fast-GP finds a *cleaner
foreground* (Dice +0.039) but **merges more cells** — merge count 27 → 34, FN
70 → 84, #pred 305 → 297. Smoothing closes the gaps between touching nuclei, so
better foreground costs instance separation. This is the same merge-dominated
failure mode Round 4 identified, now made worse by the reconstruction.

On whole_cell `paper_fast_gp` produces **numerically identical** TP/FP/FN
(258/86/145) to Raw. That is expected rather than surprising: it barely smooths
there (`corr to raw = 0.999662`, `nugget = 0.00225`), so it hands the downstream
almost the same image. The comparison on that image is close to uninformative by
construction.

## 2. Does the previous conclusion "GP reconstruction hurts" survive when the correct paper Fast-GP model is used?

**In direction, partly. In magnitude, no — and the original basis for the claim
was an arm that is simply broken.**

| arm | nuclei AP@0.5 | Δ vs Raw | whole_cell AP@0.5 | Δ vs Raw |
|---|---|---|---|---|
| Raw | 0.6933 | — | 0.5276 | — |
| gp_isotropic_gpytorch_2025 | 0.1499 | **−0.5434** | 0.4168 | −0.1108 |
| paper_fast_gp | 0.6457 | **−0.0476** | 0.5276 | **0.0000** |

The old arm loses 0.543 AP on nuclei — it collapses (foreground 39.9%, Dice
0.642, only 70 of 330 cells recovered, 66 merges). The paper's Fast GP loses
0.048, i.e. the harm is **~11x smaller**, and on whole_cell it vanishes
entirely (exact tie).

So the honest restatement is: *"GP reconstruction catastrophically hurts"* was a
statement about `gp_isotropic_gpytorch_2025`, not about the paper's method. What
survives is a much weaker claim — **the paper's Fast-GP reconstruction does not
help instance segmentation here, and slightly hurts it on nuclei** — together
with the new observation that it *does* help foreground extraction on nuclei.

## 3. Does robust optimization materially change the answer relative to the paper-faithful optimizer?

**No, on either image — even though on whole_cell it materially changes the
fit.** This is a clean dissociation.

- **nuclei: no change at all.** Both optimizers reach the identical point
  (`f = 116922.5219`, same beta/nugget to 6 dp) and the reconstructions are
  bit-identical (max|diff| = 0.0000). Every metric is identical. Multi-start
  cost 18.5 s instead of 3.4 s for nothing.
- **whole_cell: the fit changes a lot, the answer barely moves.** The objective
  improves by 20102 (61119.2354 → 41016.7491), `beta1` goes 25232 → 81.01, the
  degenerate row axis is repaired (0.0079 px → 2.456 px), and the
  reconstructions genuinely differ (corr 0.999793, RMSE 0.834, max|diff| 13.53).
  Yet AP@0.5 moves only 0.5276 → 0.5369 (+0.0093, 4 more TP), AP@0.75
  0.0702 → 0.0745, Dice 0.8231 → 0.8224.

At n = 1 image, +0.0093 AP is not a resolvable difference. So fixing the
optimizer pathology found in the wiring audit does **not** rescue the Fast-GP
baseline, and the conclusions in answers 1, 2 and 5 are the same under either
optimizer.

## 4. How different is the old isotropic GPyTorch GP from `paper_fast_gp`, in both reconstruction and segmentation?

**Very different in both — they are not substitutes.**

*Reconstruction* (compared directly to each other, not via either one's RMSE to
raw): correlation **0.8887** (nuclei) and **0.9454** (whole_cell); RMSE between
them **8.89** and **14.05**; max absolute difference **61.7** and **143.3** grey
levels on a 0–255 scale. On whole_cell the two arms disagree by more than half
the dynamic range at some pixels.

*Segmentation*:

| | nuclei | whole_cell |
|---|---|---|
| AP@0.5 | 0.1499 vs 0.6457 — a **4.3x** difference | 0.4168 vs 0.5276 |
| Dice | 0.6422 vs 0.8785 | **0.8505 vs 0.8231** (isotropic better) |
| AP@0.75 | 0.0170 vs 0.3397 — a **20x** difference | **0.1330 vs 0.0702** (isotropic better) |
| TP / #pred | 70 / 207 vs 246 / 297 | 213 / 321 vs 258 / 344 |
| merges | 66 vs 34 | 81 vs 67 |
| runtime | 650.3 s vs 3.4 s (**191x**) | 357.3 s vs 2.5 s (**143x**) |

The old arm over-smooths badly: 39.9% foreground on nuclei against a 22.2% ground
truth, and it loses 260 of 330 cells. Notably it is **not** uniformly worse —
on whole_cell it has the best foreground Dice (0.8505) and the best AP@0.75
(0.1330) of any arm, while having a poor AP@0.5. So its errors are qualitatively
different, not merely larger.

Conclusion: results from Rounds 1–4 that used `gp_legacy` cannot be read as
statements about the paper's Fast GP. They differ by 4–20x in AP and by up to
143 grey levels in the reconstruction.

## 5. Does QEP q=1.5 beat the CORRECT Fast-GP baseline under identical downstream processing?

**No. It loses on both images on essentially every metric.**

| metric | nuclei: q=1.5 vs paper_fast_gp | whole_cell: q=1.5 vs paper_fast_gp |
|---|---|---|
| foreground Dice | 0.8558 vs 0.8785 → **−0.0227** | 0.8087 vs 0.8231 → **−0.0144** |
| AP@0.5 | 0.5985 vs 0.6457 → **−0.0472** | 0.4503 vs 0.5276 → **−0.0773** |
| AP@0.75 | 0.2967 vs 0.3397 → **−0.0430** | 0.0705 vs 0.0702 → +0.0003 (negligible) |
| TP | 234 vs 246 | 231 vs 258 |
| merges | 39 vs 34 | 67 vs 67 |

The single non-negative entry (+0.0003 AP@0.75 on whole_cell) is far below
resolution at n = 1.

q = 1.5 also does not beat q = 2 consistently: nuclei AP@0.5 0.5985 vs 0.6250
(q = 2 better), whole_cell 0.4503 vs 0.4453 (q = 1.5 better by 0.0050). No stable
q advantage, consistent with Rounds 2–4 and with the audit finding that every
mask in these arms is a deterministic function of the q-invariant posterior mean.

**Overall ranking on AP@0.5 (both images): Raw first.** Raw is best on nuclei
(0.6933) and tied-best on whole_cell (0.5276). Under this frozen downstream, no
reconstruction arm — the paper's Fast GP included — beats simply not
reconstructing.
