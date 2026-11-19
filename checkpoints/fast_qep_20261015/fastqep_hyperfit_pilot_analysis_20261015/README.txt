Fast-QEP hyperparameter-gradient diagnostic on the first native tile
15 October 2026 (America/Phoenix)

Finding

All four sampling targets passed the prespecified numerical, convergence and mean-precision checks. Neither the coordinatewise QEP–GP gradient differences nor the QEP gradients alone crossed the decision threshold at these parameter points and precision. Other parameter points and joint directions remain open.

Executed pilot

1. First tile of the nuclei development image: 282 × 240 (67,680 pixels).
2. First tile of the whole-cell development image: 200 × 200 (40,000 pixels).
3. Refit the analytic GP from two frozen starts per dataset and select by GP model evidence.
4. Sample q=1.5 QEP gradients at the original R and the refitted GP parameters.
5. Run four chains per target with 2,000 warmup and 4,000 retained draws per chain. Evaluate a five-dimensional gradient every tenth retained draw, for 1,600 gradient observations per target.

The pilot took 726.89 seconds (12.1 minutes), excluding preliminary correctness checks and report preparation. It ran in the cloud Python environment. See environment.json in the results for versions.

Gradient differences

Dataset       Parameter point       Maximum |QEP−GP score| / MCSE    Passed cutoff
Nuclei        Original R                          1.031                 No
Nuclei        Refitted GP                         2.498                 No
Whole-cell    Original R                          1.119                 No
Whole-cell    Refitted GP                         0.235                 No

MCSE denotes Monte Carlo standard error. The rule required differences above 3 MCSE and a scaled absolute difference above 1e-8. This descriptive screen measures numerical resolution, without p-values or multiplicity adjustment. The thresholds and seeds were fixed before seeing the results. At the original R parameters, the nuclei column-kernel derivative and the whole-cell row-kernel derivative were on exactly flat plateaus. Those coordinates gave no local information.

GP controls

For both images, the common nondegenerate start produced higher GP evidence. The selected solutions passed the stationarity check and lay within the specified bounds. Selection covered two starts in a bounded domain.

Nuclei:     beta_row=35.50397, beta_col=23.68896, nu=0.1705975
Whole-cell: beta_row=81.01076, beta_col=62.33656, nu=0.001954292

GP log evidence (solution from original R start to solution from common start):
Nuclei:     151042.7113 to 163430.6746
Whole-cell:  94055.9179 to 114158.4042

These values compare the two GP fits. The pilot did not estimate QEP marginal evidence or test whether higher GP evidence improves segmentation. Future QEP comparisons need the refitted GP control to account for optimization quality.

Numerical checks and limitations

The input and code hashes, preliminary numerical checks, and original R first-tile reconstruction checks passed. All four targets reached the planned sample count. Retained draws showed no HMC divergences, nonfinite proposals or global-step numerical failures. All monitored latent variables and primary gradients met the frozen R-hat and ESS criteria. The maximum gradient R-hat was about 1.00338. Minimum bulk and tail gradient ESS were about 1320.8 and 1421.8. All pixel-mean MCSE checks passed.

A zero-mean control variate can increase variance at q=1.5. At the refitted whole-cell point, the fixed control produced row- and column-kernel gradient MCSEs about 7.53 and 7.12 times those of the raw estimator, while reducing error for the mean gradient. We used the prespecified estimator and rules and saved both raw and corrected traces. Chain convergence alone does not establish adequate gradient precision.

Research decision and scope

The pilot provided no direct evidence that hyperparameter training improves QEP segmentation. Before full training, improve gradient precision and design an error-controlled rule for accepting parameter updates. At this stage that work had not begun. The q range, sampling budget and seeds stayed fixed.

No QEP hyperparameter update was accepted. Nuisance ECM suggestions are recorded candidates. No whole-image QEP reconstruction, RobustGaSP thresholding, EBImage watershed, cleanup, AP evaluation or held-out image evaluation was run. The prior end-to-end AP tie between fixed-parameter Fast-QEP and Fast-GP still applies. One tile from one development image per dataset cannot establish generalization or statistical significance.

Files in the original delivery

REPORT_EN.txt          Detailed report for discussion.
analysis.json          Summary of frozen outputs and chain means.
summarize_pilot.py     Read-only summarization, without fitting or sampling.
source/                Original frozen source and input ZIP.
results/               Per-target summary.json, score_traces.npz, estimates.npz,
                       GP fit histories, logs, versions and verification records.
provenance.json        Source ZIP hashes and run status.
delivery_manifest.json SHA-256 values for the delivery.

To regenerate the summary from the original package:
python summarize_pilot.py results/fast_qep_hyperfit_pilot_20261015 regenerated_review

To reproduce the experiment, unpack the executable from source into a new directory and run python run.py --pilot. The runner refuses to overwrite existing outputs. The full protocol is in PROTOCOL.md in that bundle. Reviewing the results requires no rerun on the Mac.
