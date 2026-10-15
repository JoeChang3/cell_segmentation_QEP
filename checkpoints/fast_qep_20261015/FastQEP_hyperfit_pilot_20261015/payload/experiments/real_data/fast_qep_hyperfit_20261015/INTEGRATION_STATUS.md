# Integration status - 2026-10-15

## Result

The uploaded executable bundle matches the previously recorded SHA-256 exactly:
64baa638d989a9a8e3032d00d65eb75eeb2b911b0b7425c4fa78581540a50df2.
All 53 original manifest entries match. Their source and data are unchanged.
The model is the declared finite-tile covariance-matched qED prior with independent
Gaussian noise, not an unverified substitute for the full original QEP papers.

The new package adds evidence scores, exact-identity affine Stein score controls,
raw-moment nuisance ECM candidates, an analytic two-start GP optimizer, and a
bounded first-tile score runner. It is ready for the described score pilot.
It does NOT yet implement accepted multi-step QEP hyperparameter optimization.
No real-image GP refitting, QEP sampling, segmentation, or held-out run occurred
while preparing this bundle.

## Verification actually completed

- Previous 15 mathematical checks: PASS.
- All 15 integration test groups: PASS (groups include multiple q/dimension cases).
- Dense physical score vs integrated spectral score: maximum error <1.7e-13.
- Independent integration of raw/controlled scores: maximum error <1.8e-14.
- q=2 controlled score equals analytic evidence gradient for each tested draw.
- Unchanged sampler with q=1.5: controlled scores agree with quadrature within
  five conservative MCSE; tested R-hat and bulk/tail ESS gates pass.
- Observer preserves chain states; scores match the fixed every-tenth-draw rule.
- Raw-moment ECM candidates agree with independent formulas within 3e-16.
- Two-start GP fitting exercised on a fixed numerical fixture; stationarity passes.
- Matrices of both ORIGINAL tile sizes tested at beta=5 and beta=20 using
  artificial values: q=2 pointwise error <2.0e-9; kernel finite-difference error
  per pixel <2.8e-11. These are numerical checks, not cell experiments.
- Root --check entrypoint completed, including archived R first-tile parity:
  nuclei 6.44e-15; whole-cell 5.04e-14 (normalized intensity).

Full details: validation/integration_checks.json, previous_math_checks.json and
archived_R_anchor_checks.json. Timings in these files are cloud timings, not Mac
performance guarantees. The validated cloud environment uses Python 3.12.14,
NumPy 2.3.5 and SciPy 1.17.0. Run --check in the user's existing gpytorch_arm
environment before any pilot; it performs numerical checks without changing
installed packages. Production outputs are not pre-populated in the new ZIP.

## Why score-specific controls were added

Old mean-control-variate values cannot be treated as posterior samples when
forming nonlinear gradient moments. The new controls instead have their own
zero-expectation identity under the actual posterior and are independently
verified. They reduce gradient noise in the small fixture; their effectiveness
on the real first tiles remains to be measured. Both raw and controlled estimates
are retained. No new mean estimator or sampler transition was substituted.

## Scope of the next experiment

Only the two original first tiles, two fixed GP starts, then q=1.5 scores at
archived R and refitted GP parameters (four sampling targets). No q sweep,
AP-based selection, QEP optimizer step acceptance, whole-image reconstruction or
downstream segmentation. Positive score diagnostics would justify a separately
specified error-controlled fitting procedure. An unresolved diagnostic is not
a refutation of QEP or proof that no other optimum exists.

Future claims require an equally fitted GP and a parameter-source control;
improvement over an old optimizer plateau alone is insufficient for a QEP claim.
