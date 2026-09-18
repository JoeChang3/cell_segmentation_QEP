# Reproducibility snapshot — 2026-09-17

Snapshot of the environment that produced the Round 1–4 real-cell-segmentation
results. Nothing in the installed environment was modified to create this file.

## Repository

| item | value |
|---|---|
| local path | `/Users/zchan/eclipse-workspace/Cell_Seg_QEP` |
| remote | https://github.com/JoeChang3/cell_segmentation_QEP |
| branch | `chatgpt-review-cellseg-20260917` |
| branched from | `main` at `1123b84` ("move legacy scripts") |

## Runtime actually used for Rounds 1–4

| item | value |
|---|---|
| python executable | `/Users/zchan/miniforge3/envs/gpytorch_arm/bin/python` |
| python version | 3.10.19 |
| platform | macOS-15.3.2-arm64-arm-64bit (Apple silicon, `arm64`) |
| compute | **CPU only.** `torch.cuda.is_available() == False`; MPS was available but not used — every run was CPU with `torch.set_default_dtype(torch.float64)` unless a script states float32 |
| `torch.get_num_threads()` | 8 (individual commands set `OMP_NUM_THREADS` between 1 and 8) |

### Package versions and real import paths

| package | version | import path |
|---|---|---|
| torch | 2.10.0 | `…/gpytorch_arm/lib/python3.10/site-packages/torch` |
| gpytorch | 1.15.1 | `…/gpytorch_arm/lib/python3.10/site-packages/gpytorch` |
| qpytorch | 0.2 | `…/gpytorch_arm/lib/python3.10/site-packages/qpytorch` |
| scikit-image | 0.25.2 | `…/gpytorch_arm/lib/python3.10/site-packages/skimage` |
| numpy | 1.26.4 | `…/site-packages/numpy` |
| scipy | 1.15.3 | `…/site-packages/scipy` |
| pandas | 2.3.3 | `…/site-packages/pandas` |
| matplotlib | 3.10.8 | `…/site-packages/matplotlib` |

### qpytorch / gpytorch provenance

- `qpytorch` 0.2 installed by **pip** (`INSTALLER` = `pip`) into the conda env
  `gpytorch_arm`. It is a **released wheel, not a VCS or local checkout**: no
  `direct_url.json` is present, so pip recorded no Git URL or commit.
- **A qpytorch Git commit hash is therefore NOT determinable** from this
  installation. Upstream project URLs recorded in the wheel metadata:
  - Home-page: https://lanzithinking.github.io/qepytorch/
  - Source: https://github.com/lanzithinking/qepytorch/
- `gpytorch` 1.15.1 is likewise a pip-installed release in the same env.
- `Diff_QEP/GPyTorch/` (a vendored gpytorch fork in the sibling reference repo)
  was deliberately **not** used or vendored into this project; all QEP code here
  imports the installed `qpytorch`.

There is a second, older environment on this machine
(`/Users/zchan/opt/anaconda3/envs/Gpytorch310`, qpytorch 0.1.1, torch 2.2.2).
It was used only for early API probing and produced none of the Round 1–4
results.

## Latest reports (included in this branch)

| round | report |
|---|---|
| Round 1 | `results/real_cellseg_round1/` — metrics/config CSVs + JSON (no standalone REPORT.md; findings are in the session record) |
| Round 2 | `results/real_cellseg_round2_20260915/REPORT.md` |
| Round 3 | `results/real_cellseg_round3_thresholding_20260916/REPORT.md` |
| Round 4 | `results/real_cellseg_round4_mechanism_20260916/REPORT.md` |
| Round 4 extras | `…/mechanism_cards.md`, `…/source_notes.md` |
| cell-blob synthetic | `results/cell_blobs_qep_report/REPORT.md` |

## What stayed LOCAL and was not pushed

| path | size | reason |
|---|---|---|
| `results/**/masks/`, `binary_masks/`, `instance_masks/`, `npz/` | ~151 MB over 304 `.npz`/`.npy` files | large generated arrays; regenerable from the scripts |
| `results/real_cellseg_round*/figures/`, `results/cell_blobs_*/figures/` | ~24 MB over 41 PNGs | large generated figures; not needed to review code |
| `results/*.png`, `results/*.pdf` (linear-diffusion plots) | ~1.8 MB | generated figures from the retired linear-diffusion negative control |
| `results/real_cellseg_round*/logs/*.log` | ~64 KB | console logs, already excluded by the pre-existing `*.log` rule; the numbers they contain are reproduced in the REPORT.md files |
| `results/cell_blobs_qep_{exact,exact_fixnoise,variational}/` arrays | ~24 MB | synthetic-benchmark arrays; the summary lives in `cell_blobs_qep_report/` |

The microscopy dataset under `data/` (172 MB, 10 image/GT/ImageJ triples) was
**already tracked on `main` before this snapshot** and is unchanged here.

## Reproducing a round

Rounds reuse cached reconstructions; only Round 1/2 fit models. Entry points:

```
# Round 2 held-out (fits GP/QEP; hours on CPU)
python experiments/real_data/round2_step3_heldout.py --out <dir>
# Round 3 thresholding (reuses cached reconstructions)
python experiments/real_data/round3_step1_threshold_diagnostics.py --out <dir> --dev-dir <dir>
python experiments/real_data/round3_step4_heldout.py --out <dir> --dev-dir <dir>
# Round 4 diagnostics (reuses cached outputs, no fitting)
python experiments/real_data/round4_phase1_failure_inventory.py --out <dir>
python experiments/real_data/round4_phase2_interventions.py --out <dir>
# Standalone verification (fast, self-contained)
python experiments/simulated/qep_power_semantics_check.py
```

Seeds: training-pixel sampling uses `np.random.default_rng([seed, tile_index])`
with `seed=0`; torch is seeded per tile as `seed*1000 + tile_index`; the exact
log-determinant path is forced on (`exact_logdet=True`) because gpytorch's
default `max_cholesky_size=800` otherwise estimates it with random probe
vectors, which made identical models fit differently.
