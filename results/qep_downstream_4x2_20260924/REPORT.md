# QEP reconstruction and downstream comparison: environment preflight

24 September 2026 (America/Phoenix)

**The preflight ran. New training runs: 0. New segmentation results: 0 of 16.**

## Completed work

- Checked out the main research repository at `fa746853ba007552bb1d5a43340d91f6796096b1` and the original paper's R reference at `44714c2e0be958fe796a8fd4bdbc220dae3c23dd`.
- All four MD5 checks passed for the two figure_1 development images and reference masks.
- Wrote the driver for four reconstructors and two downstream pipelines, the original-R downstream adapter and the script to recover frozen FastGP parameters.
- The Python syntax check passed. The preflight exited with status 2 after finding missing dependencies.
- This stage preserved the core implementation and Rounds 1–10. It ran no new q sweep or evaluation on other images.

## Environment blocker

The environment had Python 3.12.14, while the frozen run used 3.10.19. Rscript, torch, gpytorch, qpytorch, scikit-image, imageio and tifffile were unavailable. The installed numpy and pandas versions also differed. Conda could not finish creating an isolated environment because a proxy returned HTTP 403 (CONNECT tunnel failed) while downloading dependency-index shards. The original RobustGaSP and EBImage pipeline could not run. No approximate Python replacement or incompatible dependency set was used.

The blocker concerns execution. Section 5 of the approved protocol excludes comparisons using measurements that fail compatibility or baseline-reproduction checks. No new model-performance conclusion follows from this preflight.

## Remaining execution steps

Follow `experiments/real_data/qep_downstream_4x2/README.md` in the original research environment or one with compatible critical dependencies. Both repository checkouts and working Python and R installations are required.

1. Verify dependencies and input files.
2. Reproduce the historical Raw-P, FastGP-P and Raw-C anchors on both images.
3. Run fixed-configuration QEP reconstruction and reproduce the historical QEP-C anchor.
4. Run QEP-P and export all 16 rows and control differences.

An anchor mismatch stops execution. Each reconstruction is generated once. Its input array hashes must match across P and C. FastGP uses the R reference implementation and archived R-side parameters. The Round 5 Python optimum belongs to a separate comparison. All 16 combinations were already approved. The outstanding requirement was an executable environment.

## Validation limits and decision rules

Only Python syntax and the preflight path were tested. The R scripts, numerical reproduction and full execution still needed validation. The driver saves tile-level QEP diagnostics and arrays. Cross-run cache resumption was unavailable, so failed runs need a new output directory.

The prespecified engineering thresholds require an AP50 improvement of at least 0.02 over the original method and at least 0.01 over the strongest Raw/FastGP/Q2 control with the same downstream pipeline. The driver conservatively checks the AP75 and added-split safeguards against every listed control and retains all differences. The thresholds are engineering criteria. This record supplies no new evidence about QEP performance relative to FastGP.
