# BBBC020 external validation · 17 November 2026

This experiment compares GP and QEP (q = 1.15) in the original R whole-cell segmentation pipeline. It includes all 20 BBBC020 C1 fields with whole-cell annotations. The primary comparison fits each model independently and uses its own thresholds.

| Mean field AP50, GP | Mean field AP50, QEP | Mean paired difference | QEP wins / ties / losses |
| ---: | ---: | ---: | ---: |
| 0.078836 | 0.081850 | +0.003014 | 10 / 4 / 6 |

The mean gain fell below the prespecified +0.0100 target. The descriptive 95% paired field-bootstrap interval was [−0.000342, +0.008361]. The group-meeting slide displayed +0.0031. The table above uses the underlying per-field scores.

- [Per-field AP50 and AP75](field_scores.csv) contains one row per field and arm, including TP, FP and FN. Compare `fit_gp` and `fit_q115` for the primary result. The third arm reuses GP thresholds with QEP.
- [Paired differences](field_deltas.csv), [summary](summary.csv), and [plot](paired_fields.png) give the field-level contrasts and their aggregate.
- [Frozen protocol](PROTOCOL.md), [cohort manifest](data/manifest.json), and [results report](RESULTS.md) describe the population, endpoint and interpretation.
- [Run instructions and source-data attribution](REPRODUCE.md), [scripts](code/), [GP and R reference code](gp/), and [per-field fitted parameters, tile thresholds and predicted masks](results/) accompany the original exports.
- [SHA-256 checksums](SHA256.json) cover the archived payload files in this edition.
