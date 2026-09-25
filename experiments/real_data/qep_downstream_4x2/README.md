# Approved QEP reconstruction × downstream experiment

Status on 2026-09-24 (America/Phoenix): cloud execution was blocked at preflight.
The subsequent MacBook run passed non-QEP anchors and stopped on newly generated
nuclei Q2-C reproduction mismatch. Historical-cache recovery is now available;
the full 16-row experiment is not yet complete. See CACHE_RECOVERY.md in the
associated results directory.

Base: `fa746853ba007552bb1d5a43340d91f6796096b1`.
R reference: `44714c2e0be958fe796a8fd4bdbc220dae3c23dd`.

Scope: nuclei_figure_1 and whole_cell_figure_1 only; Raw, frozen R-side FastGP,
QEP q=2 and q=1.5, each crossed with real paper downstream P and corrected
downstream C. No optimization knobs are exposed by this runner.

## Execution

Use the original research environment, or an isolated environment with Python
3.10.19, torch 2.10.0 (CPU), gpytorch 1.15.1, qpytorch 0.2, numpy 1.26.4,
scikit-image 0.25.2, pandas 2.3.3, plus scipy/imageio/tifffile/Pillow/matplotlib.
The last five packages' installed versions are recorded, not claimed to be a
complete archived environment lock. Anchor checks remain mandatory.

R must be 4.5.1 with EBImage 4.52.0, RobustGaSP 0.6.8, magick 2.9.1 and pracma
2.4.6. Use the original packages, not Python approximations. No installer is
automatically invoked by the runner.

From the main repository root, replacing the reference path as appropriate:

```bash
python experiments/real_data/qep_downstream_4x2/run.py \
  --reference /absolute/path/to/cell_segmentation_original \
  --out results/qep_downstream_4x2_run --preflight

python experiments/real_data/qep_downstream_4x2/run.py \
  --reference /absolute/path/to/cell_segmentation_original \
  --out results/qep_downstream_4x2_run
```

An explicit R executable can be passed using `--rscript /absolute/path/to/Rscript`.
Exit code 2 means a failed preflight. Runtime/anchor failures raise an error and
write STOPPED.json. A successful scientific run writes completion.json and
REPORT.md. Do not treat a partial metrics.csv as completed evidence.

## Integrity and scientific gates

- Frozen tracked code and input hashes are checked before imports/training.
- FastGP restores committed R parameters using the reference R separable_GP;
  it does not refit or substitute the Round-5 Python optimum. The paper's zero
  remainder convention is preserved. Recorded runtime excludes historical fitting.
- Both Raw-P/FastGP-P anchors and Raw-C are checked on both images before QEP
  training. TP/FP/FN at .5/.75, failure counts and foreground Dice must match.
- QEP uses frozen seed 0, 75 Adam steps, learning rate .1, 3000 pixels/tile,
  isotropic Matern 2.5, float64, exact log determinant and prediction chunks 8192.
  The historical float32 reconstruction cache conversion is retained.
- QEP-C must reproduce Round 5 before its P result is computed. No silent
  tolerance relaxation, optimizer repair, or parameter sweep follows a failure.
- P/C consume identical saved reconstruction arrays (hash checked). P divides
  raw intensity units by 255 explicitly; no clipping or min-max normalization.
- The supplied R adapter preserves the frozen downstream algorithm, while
  parameterizing paths, recording tile thresholds, and setting RNG seed 1 at
  entry. This explicit seed initialization is new bookkeeping, and its
  compatibility is gated by the historical anchors; it is not assumed proved.
- All 16 rows, per-tile diagnostics, masks, AP50/AP75, foreground scores,
  component/marker diagnostics, merge/split counts and lost matches are saved.
  EBImage's implicit marker count is marked unavailable, not equated to labels.
- Q15 continuation thresholds: AP50 gain >= .02 over FastGP-P and >= .01 over
  the best same-downstream Raw/FastGP/Q2 control. The design's AP75 decline
  <= .01 and new split/GT <= .01 are operationalized conservatively against
  every listed control; all actual contrasts are retained. Q2 is the Gaussian
  limit control, despite its implementation class name.
- The split diagnostic compares GT identities, not net split-count change;
  lost matches are reported separately. These are not independent sample tests.

## Validation and limits

Python syntax and dependency-preflight execution can be checked without R.
The R scripts and full numerical workflow have **not** been executed here.
Baseline parity, training runtime and successful end-to-end execution remain
unverified until the required environment is available.

The worker saves each newly generated QEP tile and its diagnostics, but does
not yet implement verified cross-run cache resume. Use a fresh output directory
after a scientific run stops; never concatenate incompatible partial results.
The original audited core modules and previous results are unchanged.

## Preferred continuation using the recovered historical caches

Add this argument to the normal run command:

```bash
--qep-cache-dir /Users/zchan/eclipse-workspace/Cell_Seg_QEP/results/real_cellseg_round3_thresholding_20260916/masks
```

Choose a new `--out` directory (for example `results/qep_downstream_4x2_macbook_cached`).
All four historical files must pass the committed cache manifest before any
scientific work. They are then independently checked as arrays before scoring.
This route performs no QEP training. It reruns the fast non-QEP anchors and all
fixed downstream comparisons, rather than merging result tables from two runs.
Historical QEP-C anchors are unchanged. The original failed run is retained.
