# Fast-QEP hyperparameter learning on the first native tile

Extract this package to its own directory alongside the earlier FastQEP_R_pipeline_20261015 package. The original execution bundle froze 53 files. The new entry point is the top-level `run.py`.

## Scope

The pilot implements the marginal-likelihood gradient, a fixed Stein control variate, raw-moment Kronecker calculations and candidate ECM updates for m/s/nu. Checks cover the analytical GP limit at q=2 and independent quadrature at q=1.5. The question is whether any QEP adjustment near refitted GP parameters can be resolved above Monte Carlo noise. Whole-image segmentation and full hyperparameter optimization are separate stages.

## Environment and correctness checks

Use the previously tested gpytorch_arm environment with Python >=3.10, NumPy, SciPy, Pillow and threadpoolctl. This stage runs in Python. Subsequent end-to-end segmentation requires the original R pipeline. The scripts do not install or upgrade packages.

```bash
conda activate gpytorch_arm
cd ~/Downloads/FastQEP_hyperfit_pilot_20261015
python run.py --check
```

The check performs small sampling and gradient-integration tests and verifies the archived first-tile R reconstruction. It prints the absolute path to a `*_review.zip` for review. Real-image fitting starts in the pilot.

## First-tile pilot

After confirming the protocol, run:

```bash
caffeinate -i python run.py --pilot
```

Use the first native tile from each development image (nuclei and whole-cell). Refit the analytic GP from two fixed starts. Run four q=1.5 chains at both the original R and refitted GP parameter points to estimate gradients, Monte Carlo error and differences between reconstructions. This step excludes label-based optimization, AP scoring, watershed and q optimization. ECM suggestions are recorded as candidates.

Sampling is capped at 900 seconds per target, GP fitting at 300 seconds per dataset, and the entire pilot at 7200 seconds. These limits do not guarantee convergence. Failures stop execution and preserve diagnostics.

Outputs go to `payload/results/fast_qep_hyperfit_pilot_20261015/`. The entry point prints the review ZIP path on completion or failure. Retain failed runs and all chains. Checks and pilot outputs use separate directories. Existing directories prevent accidental overwrites, and failed targets do not resume automatically. See `payload/experiments/real_data/fast_qep_hyperfit_20261015/PROTOCOL.md`.

## Interpretation

A resolved direction supports error-controlled QEP fitting. Its segmentation impact would require a separate test. An unresolved direction limits the evidence at the tested parameter point and sampling precision. Failed numerical checks require computational diagnosis. Improvements explained by GP refitting alone cannot establish a QEP-specific benefit. The runner keeps the original checkout and prior results intact.
