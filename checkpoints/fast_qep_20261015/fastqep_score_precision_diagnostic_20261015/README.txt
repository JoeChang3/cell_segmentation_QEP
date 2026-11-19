Fast-QEP gradient precision diagnostic
15 October 2026

Finding

The original fixed control variate overcorrected, especially for whole-cell kernel gradients. Revised weights improved precision. None of the QEP gradient differences crossed the original decision threshold. The measured improvement concerns the estimator; model training and segmentation were outside this analysis.

Methods

The analysis reused four existing targets with four gradient chains per target. It preserved the original files and pilot conclusion. There was no new sampling, model change, QEP parameter update, thresholding, watershed or AP evaluation.

For each parameter, G is the raw gradient and G−C the original corrected gradient. We compared G (b=0), G−C (b=1) and G−bC with b determined solely by variation in calibration chains. First calibrate on chains 1 and 2 and estimate standard errors on chains 3 and 4, then swap the pairs. Estimate covariance from batch means of 20 gradient observations and constrain b to [0,1]. All estimators use the same two evaluation chains within a split. Neither the sign nor apparent significance determined the weights. No other batch lengths or control terms were tried.

The splits reuse the same previously examined data. This is an exploratory offline diagnostic. The two splits do not yield independent evidence or a pooled independent standard error.

Kernel gradients at the refitted GP parameters

Parameter         MCSE reduction vs fixed control     MCSE ratio to raw gradient
Nuclei row                 40.9%–42.2%                        0.611–0.624
Nuclei column              36.2%–42.6%                        0.598–0.693
Whole-cell row             87.6%–87.9%                        0.972–0.990
Whole-cell column          86.4%–87.8%                        0.917–1.008

The ranges describe the two swapped splits. MCSE is Monte Carlo standard error. For whole-cell data the large reductions mainly remove noise introduced by overcorrection, restoring precision close to the raw estimator. The nuclei kernels also gained precision relative to the raw estimator.

Reason for overcorrection

A zero-mean control preserves the expected value with fixed weights. Its variance is Var(G)+b² Var(C)−2b Cov(G,C). At the refitted whole-cell point the row and column control terms had batch-mean standard deviations about 6.69 and 7.36 times those of their raw gradients, with correlations of only about 0.19 and 0.31. Subtracting the full control added noise. Kernel weights from the two calibration splits were about 0.02–0.05. This finding is specific to the estimator and samples used here.

Gradient evidence

All weighted evaluations passed the convergence checks. Every absolute gradient difference divided by MCSE remained below 3, with a maximum of about 1.90. Of 18 informative parameter/target combinations, 12 met the fixed technical precision-improvement rule on both splits. Two kernel-axis derivatives at the original R parameters were identically zero and were excluded from the informative comparisons.

The 3-MCSE and scaled-amplitude cutoffs are descriptive numerical criteria. They carry no p-value or multiplicity correction. The observations leave open nonzero gradients, other parameter points and other models. They provide insufficient evidence for full QEP training.

Next validation as planned at the time

Candidate weights were computed from all old chains using the same rule and frozen in candidate_weights.json. The plan called for fresh seeds at the two refitted GP points, q=1.5, four chains, 2,000 warmup and 4,000 retained draws per chain, with a 30-minute cap. The protocol holds weights, model and GP parameters fixed and excludes segmentation. If precision and convergence pass but no direction emerges, stop automatic expansion of local gradient training. See NEXT_VALIDATION_PROTOCOL.txt.

Read-only readiness checks covered the source, R anchors, GP parameters and analytic gradients. readiness_check.json recorded READY_NOT_RUN before fresh sampling.

Files and reproduction in the original delivery

REPORT_EN.txt contains the detailed report.
precision_diagnostic.png shows weights and errors at the two refitted GP points.
results/analysis.json records splits, raw/fixed/weighted gradients and MCSEs.
results/weighted_evaluation_traces.npz contains the split-evaluation traces.
inputs/ contains the unmodified pilot gradient arrays and summary.json.
PROTOCOL.txt records the offline rules fixed before computation.
input_provenance.json and the manifests identify and verify the inputs.

Reproduce the offline analysis without R or images:
python analyze.py --out regenerated_analysis

Check the fresh-sample runner without sampling, using the previous executable bundle:
python validate_new_samples.py --bundle /path/to/FastQEP_hyperfit_pilot_20261015

The original protocol reserved --run for explicitly approved fresh sampling. FastQEP_hyperfit_pilot_20261015.zip remains the frozen dependency for the model, sampler and image inputs. The diagnostic package omitted a duplicate of that large archive. Reviewing the results requires no Mac rerun.
