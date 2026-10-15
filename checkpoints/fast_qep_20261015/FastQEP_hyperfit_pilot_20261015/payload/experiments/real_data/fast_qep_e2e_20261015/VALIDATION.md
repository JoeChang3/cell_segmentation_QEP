# Execution preparation validation

Status: **READY FOR MAC R PREFLIGHT**, not a new segmentation result.

- All 25 analytic q=2 tiles checked against frozen original Python formulas.
- Full-image analytic q=2 versus archived R: maximum normalized absolute error
  7.8020e-15 (nuclei) and 1.1614e-13 (whole-cell), below the frozen 1e-10 gate.
- Four historical Raw/Fast-GP paper-downstream outputs reproduce all frozen
  AP50/AP75 TP/FP/FN, merge/split/missed/spurious, foreground and instance metrics.
  Archived masks/reconstructions were verified against their source artifact hashes.
- Real-crop fast/legacy IoU parity, deterministic density/Jacobian checks and
  both unchanged 2D q=2/q=1.5 sampler reference gates pass.
- A separate synthetic 20-coordinate runner fixture verifies actual sampler ->
  checkpoint -> image assembly; zero trailing columns; pooled/chain consistency;
  no-resampling recovery; and blocking on failed, changed or corrupted checkpoints.
  This fixture is an engineering test, not a real-image experiment.
- Launching the real preflight on this host correctly stops at missing Rscript,
  before any full-image QEP sampling. No R behavior was simulated or substituted.
- Previous sources/results listed by the latest inference manifest remain intact;
  counts and source hashes are recorded in validation/STATUS.json.

R is not installed on the cloud host. Actual R-version, fresh R reconstruction,
RobustGaSP/EBImage baseline-partition parity, all 25 real QEP tile gates and final
segmentation remain to be executed on the Mac. The runner enforces that order.

Validation records in validation/ are historical preparation records; the execution
host generates its own fresh preflight records under the result directory.
