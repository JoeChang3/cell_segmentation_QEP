# QEP reconstruction × downstream development pilot

All 16 frozen combinations completed. These are development-image observations, not generalization or significance results.

| Dataset | Reconstruction | Downstream | AP50 | AP75 | Merge | Split | FG Dice |
|---|---|---|---:|---:|---:|---:|---:|
| nuclei | raw | P | 0.401473 | 0.147813 | 15 | 8 | 0.761948 |
| nuclei | raw | C | 0.693333 | 0.356838 | 27 | 0 | 0.839310 |
| nuclei | fastgp | P | 0.648101 | 0.284024 | 22 | 1 | 0.850393 |
| nuclei | fastgp | C | 0.620155 | 0.314465 | 33 | 3 | 0.862481 |
| whole_cell | raw | P | 0.429119 | 0.130303 | 84 | 20 | 0.814065 |
| whole_cell | raw | C | 0.527607 | 0.079480 | 67 | 15 | 0.827496 |
| whole_cell | fastgp | P | 0.583864 | 0.171115 | 68 | 12 | 0.858411 |
| whole_cell | fastgp | C | 0.527607 | 0.070201 | 67 | 19 | 0.823097 |
| nuclei | q2 | C | 0.625000 | 0.297297 | 38 | 2 | 0.856865 |
| nuclei | q2 | P | 0.460000 | 0.164894 | 33 | 9 | 0.783736 |
| nuclei | q15 | C | 0.598465 | 0.296680 | 39 | 5 | 0.855843 |
| nuclei | q15 | P | 0.539759 | 0.185529 | 30 | 5 | 0.802663 |
| whole_cell | q2 | C | 0.445298 | 0.035763 | 62 | 18 | 0.788088 |
| whole_cell | q2 | P | 0.491968 | 0.171924 | 74 | 18 | 0.855048 |
| whole_cell | q15 | C | 0.450292 | 0.070504 | 67 | 16 | 0.808653 |
| whole_cell | q15 | P | 0.490835 | 0.165605 | 73 | 10 | 0.852405 |

AP is the frozen TP/(TP+FP+FN) metric. Q2 is the same QEP implementation at the Gaussian limit.

Q15/Q2 differences include q-dependent hyperparameter fitting and likelihood effects; they are not a direct fixed-parameter mean effect.

See comparisons.csv for all control contrasts, AP75 changes, new splits and lost matches; downstream_contrasts.csv for C minus P.

Engineering gates are descriptive predeclared continuation criteria. AP75/split safety is conservatively required against all listed controls. No p-values are computed.

FastGP runtime excludes the historical optimizer: its frozen R-side fit is restored. Runtime columns cannot support a full-fit speed ranking.

When historical QEP caches are used, reconstruction/total runtime is unavailable (null), not zero. Historical full-pipeline runtime is retained separately in cache provenance.

No automatic follow-up search or held-out evaluation is authorized by these results.
