# BBBC020 external validation of GP replacement in the original R whole-cell pipeline

17 November 2026, Phoenix. We fixed q at 1.15. GP and QEP were fitted independently without labels and each used the original R thresholding, watershed and cleanup steps. BBBC020 contains 25 C1 fields. The whole-cell annotations cover 517 instances in 20 fields. The five 30-minute fields lack corresponding whole-cell annotations. All 20 eligible fields met the prespecified inclusion rule. None exactly duplicated the grayscale pixel array of any of the project's ten earlier images. Each image contributed 16 tiles. All fits finished before scoring, and comparisons were paired by field.

| Endpoint (equal weight per field unless marked total) | GP | QEP | Difference |
|---|---:|---:|---:|
| Original pipeline, mean AP50 | 0.078836 | 0.081850 | +0.003014 |
| Original pipeline, mean AP75 | 0.045264 | 0.046284 | +0.001020 |
| Merges, total | 48 | 49 | +1 |
| Splits, total | 74 | 75 | +1 |
| Shared GP thresholds, mean AP75 | 0.045264 | 0.045205 | −0.000059 |

The median paired AP50 difference was +0.000019. QEP improved on 10 fields, tied on 4 and scored lower on 6. The descriptive paired field-bootstrap 95% interval was [−0.000342, +0.008361]. The prespecified criteria required a mean gain of at least +0.01, improvement on more than half the fields, and a positive interval lower bound. None was met. The fields came from one experiment with unknown donor and biological-replicate relationships, so the interval describes this cohort.

At IoU 0.50, GP matched 176 reference instances and QEP matched 177. Both matched 89 at IoU 0.75. The project's AP is TP/(TP+FP+FN), computed per field and then averaged. Computing IoU for each reference mask preserves overlapping annotations, which occurred in all 20 fields. The data provider omitted some border, severely overlapping or ambiguous cells. Whole-field scoring can therefore count real but unannotated predictions as false positives. Both methods had low absolute AP against the supplied reference.

The largest AP50 improvement was +0.046324 on the third control field. In a post hoc sensitivity calculation, excluding that field reduced the mean gain to +0.000734 across the other 19 fields. The primary result includes all 20. In the auxiliary comparison, reusing each field's GP threshold fractions gave a mean QEP–GP AP75 difference of −0.000059, with a descriptive interval of [−0.000721, +0.000534]. The small AP75 gain seen on development images did not replicate on average in this cohort.

All 40 independent fits and 640 tile reconstructions finished. Replaying all 40 native threshold paths exactly reproduced foreground masks and instance partitions. One GP first-tile fit raised a numerical-gradient line-search warning. A separate refit with the October analytic gradient converged, with a log-likelihood difference of 1.75e-10. Changing QEP quadrature order gave a maximum normalized intensity difference below 5e-11 across all tiles. An independent importance-sampling check of the QEP marginal likelihood on the first and last fields in filename order met the prespecified tolerances. See the per-field tables and reproduction instructions in this directory.

Direct replacement yielded a small positive mean AP50 difference. The result fell short of the prespecified magnitude and consistency criteria, supporting closure of this comparison. Other models, data and application targets require separate evaluation.

[BBBC020 data and annotation conventions](https://bbbc.broadinstitute.org/BBBC020)
