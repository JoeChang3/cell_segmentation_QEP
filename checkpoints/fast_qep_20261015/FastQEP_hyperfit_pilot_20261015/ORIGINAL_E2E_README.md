# Fast-QEP in the original R segmentation pipeline

At the time of this protocol, the runner was ready for execution. Sampling diagnostics had passed on two fixed real-image tiles. The planned comparison ran Fast-GP and Fast-QEP (q=1.5) on all 25 native tiles of two development images, changing only the reconstructor. See `payload/experiments/real_data/fast_qep_e2e_20261015/PROTOCOL.md`.

## Run on macOS

Extract `FastQEP_R_pipeline_20261015.zip` into Downloads and activate the Python environment used for the project.

```bash
cd ~/Downloads/FastQEP_R_pipeline_20261015
caffeinate -i python run.py --run
```

The runner checks the environment and reproduces the original R baselines before sampling. The package runs separately from the Cell_Seg_QEP checkout. The default R reference path is `/Users/zchan/eclipse-workspace/cell_segmentation_original`. Override it with `--reference /path/to/cell_segmentation_original`. To run the checks alone, use `python run.py --preflight`. A later `--run` verifies and reuses the preflight results. On macOS, `caffeinate` keeps the computer awake.

## Requirements and limits

- Python >=3.10 with NumPy, SciPy, pandas, Pillow and threadpoolctl.
- R 4.5.1 with EBImage 4.52.0, RobustGaSP 0.6.8, magick 2.9.1 and pracma 2.4.6.
- A separate checkout of the verified original R reference commit.
- At least 8 GiB of free disk space. Tiles run sequentially with one BLAS thread.

The script reports missing dependencies. If `~/Library/R/arm64/4.5/library` exists and `R_LIBS_USER` is unset, the R subprocess uses that library. The R packages execute their own compiled C/C++ code. Extrapolation from two cloud tiles suggested about 80 minutes plus preflight and output. Mac runtime can differ. Sampling stops at 15 minutes per tile, two hours total, or a failed diagnostic. The runner never increases the sample count, switches seeds or skips a failed check automatically.

## Comparisons

- Fast-GP (q=2) and Fast-QEP (q=1.5) both use the original RobustGaSP, EBImage and cleanup steps.
- The Gaussian-mean control uses the existing proposal rule for smoothing strength, without label or AP tuning.
- Segmentations of the four individual chain means reveal whether small reconstruction changes cause unstable threshold jumps. Inspect all chains.
- The historical raw and Fast-GP baselines must be reproduced before new sampling.

Both images are development images. AP uses the frozen TP/(TP+FP+FN) definition. Assess AP75, merges and splits, previously correct instances lost, the Gaussian control and Monte Carlo stability alongside AP50. These two images cannot establish generalization or isolate a QEP-specific advantage on their own.

## Outputs and failure handling

Results are written to `payload/results/fast_qep_e2e_20261015/`.

- `run.log` records terminal output. Report failures with this log and `last_stop.json`.
- `environment.json` records Python/R versions and environment issues.
- `preflight/R_anchor_checks.json` records the R baseline checks.
- `tiles/` contains tile diagnostics, sample-block means and checkpoints.
- Completed runs also produce `REPORT.md`, `metrics.csv`, `comparisons.json` and `MC_stability.json`.

After an interruption between tiles, rerun the same command to verify and skip completed tiles. An interrupted or failed tile is retained for diagnosis. Preserve the failure logs and thresholds. Ctrl+C stops execution. A partially sampled MCMC chain cannot be resumed by this runner.

SHA-256 checks pin code and inputs. The original metric functions are loaded through the AST to avoid unrelated Torch/skimage dependencies. Four historical segmentation outputs serve as checks. The developer entry point `--local-checks` skips R; reproducing the Mac R preflight requires the R environment above.
