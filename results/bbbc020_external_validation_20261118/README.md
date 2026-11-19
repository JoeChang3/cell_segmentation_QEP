# BBBC020 external validation

This directory archives the original code, protocol, cohort manifest, fitted parameters and field-level metrics. field_scores.csv contains one row per field and arm. Averaging AP50 across 20 matched fields gives GP 0.0788359764 and QEP (q = 1.15) 0.0818498990, a paired difference of 0.0030139226. The prespecified target was 0.01.

REPRODUCE.md describes the original distribution. Binary masks and raw input images are not included in this Git snapshot. For a fresh run, use an empty results directory in a working copy, retrieve the public BBBC020 inputs and follow the pinned environment and fitting steps there. The CSVs suffice to recalculate the reported means without fitting.
