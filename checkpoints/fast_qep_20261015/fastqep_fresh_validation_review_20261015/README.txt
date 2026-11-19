Fast-QEP independent sampling with frozen control weights
15 October 2026

Finding

The two validation targets finished in 370.20 seconds (about 6.2 minutes). Both passed the convergence, numerical and prespecified precision-improvement checks. The revised weights improved gradient precision, but no QEP parameter direction was resolved. The prespecified stopping rule paused further expansion of this local training procedure.

Dataset        Convergence    Precision check    Max |QEP−GP gradient|/MCSE    Eligible direction
Nuclei            Pass              Pass                    0.647                     None
Whole-cell        Pass              Pass                    0.712                     None

A direction required a difference above 3 MCSE, a scaled magnitude above 1e-8 and agreement in the sign of the mean difference across the four fresh chains. None of the ten coordinates exceeded 3 MCSE, and each had mixed signs. This is a descriptive screen for numerical resolution without p-values or multiplicity adjustment.

Precision improvement

The frozen rule passed with fresh seeds. Compared with the original fixed b=1 correction, row/column kernel MCSEs decreased by 41.78% / 45.09% for nuclei and 89.90% / 90.43% for whole cells. Relative to raw estimates without control variates, the whole-cell row/column MCSE ratios were 0.879 / 1.006. Most of the whole-cell gain came from removing noise introduced by the original correction. The comparison measures estimator precision.

Technical checks

Each target used four chains with 2,000 warmup and 4,000 retained samples per chain. Gradients were evaluated every tenth retained sample. The new fixtures were 8202 (nuclei) and 8212 (whole-cell), with base seed 20261015, distinct from the old 8102/8112 chains used to determine the weights. All weights were fixed from the old draws before execution.

Maximum weighted-gradient R-hat was about 1.00208 / 1.00020 for nuclei / whole-cell. Minimum bulk ESS was about 1282 / 1469 and minimum tail ESS about 1373 / 1489. HMC divergences, nonfinite proposals and global-step numerical failures during retained sampling were all zero. Checks passed for the frozen source, parameters, weights, R reconstruction anchors and outputs.

Interpretation and limits

Fresh sampling reproduced the repair to precision lost under the original control. At q=1.5, neither fitted GP point produced enough evidence of a local direction to justify full QEP training. The stopping rule rules out automatic longer runs, seed retries or threshold relaxation. Further work on this branch would require a new theoretical or model-based rationale. The screen cannot prove exact zero gradients or exclude other optima, Fast-QEP models or joint parameter directions.

Only the first tile of one nuclei image and one whole-cell development image was used. The images, model, q, GP parameters and sampler were fixed. The replication concerns Monte Carlo randomness on the same images. No QEP parameter update, whole-image reconstruction, RobustGaSP thresholding, EBImage watershed, cleanup or new AP scoring took place. The saved nuisance ECM suggestion was never applied. The earlier end-to-end AP tie between fixed-parameter Fast-QEP and Fast-GP still applies.

Files in the original delivery

REPORT_EN.txt gives detailed results and decision rules.
analysis.json gives verified summaries, gradients, MCSE, four-chain directions and precision ratios.
fresh_gradient_validation.png plots fresh-sample gradients relative to MCSE.
results/fresh_validation/ contains outputs, gradient traces, saved states, estimates, logs and checks.
source/ contains the two frozen ZIPs with model inputs, sampler, weights and runner.
summarize_validation.py and write_report.py verify and summarize the results.
provenance.json and delivery_manifest.json record sources and checksums.

Reading the results requires no rerun or new Mac software. To reproduce the original delivery, unpack the two source ZIPs into sibling directories. From the precision diagnostic directory run validate_new_samples.py with --bundle pointing at the original executable package. By default the runner performs checks only. Add --run to resample and use a fresh output path. The source protocol retains its pre-execution NOT RUN status. The results directory records the subsequently authorized and completed run.
