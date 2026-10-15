# Fast-QEP replacement: frozen development-image end-to-end protocol

Prepared 2026-10-15, before any full-image Fast-QEP sampling or segmentation.
Authoritative historical checkpoint: fa746853ba007552bb1d5a43340d91f6796096b1.
Original R reference: 44714c2e0be958fe796a8fd4bdbc220dae3c23dd.

## Question and scope

Does replacing only the original Fast-GP reconstruction with the audited finite-tile
Fast-QEP posterior mean improve instance segmentation under the original R pipeline?
Use only nuclei_figure_1 and whole_cell_figure_1, one development image each.
This is a mechanism/application feasibility test, not held-out generalization or a
statistical significance claim. Preserve all previous negative results.

## Frozen model and computation

Use the existing MODEL_CONTRACT.md: separable Matern-5/2 covariance, archived R
beta1/beta2/nu, per-tile GLS mean and signal scale S/n, covariance-matched finite-n
qED prior and independent Gaussian observation noise. q=2 is analytic Fast-GP;
q=1.5 is the sole QEP arm. No hyperparameter fitting, q search, clipping, float32
rounding, altered tiling, extra padding, or downstream tuning.

Nuclei: 16 tiles of 282x240 in a 1128x962 image. Whole-cell: nine tiles of 200x200
in a 600x602 image. Order: column block outer, row block inner, one-based tile ID.
Both retain the original two zero trailing columns. Tile dimensions are part of
the finite-dimensional model; smaller tiles are not an interchangeable speedup.

Use unchanged joint_mh.py and Observer from fast_qep_joint_mh_20261015. Four chains,
2,000 warmup and 4,000 retained draws each, frozen warmup-only adaptation, HMC plus
exact global independence MH. The deterministic tilted-Gaussian proposal is MH
corrected; its mean is not substituted for the QEP posterior mean. Estimate the
posterior mean using the previously validated fixed score control variate.
Seeds: base 20261015, fixture IDs 1000+tile for nuclei, 2000+tile for whole-cell;
the unchanged sampler defines chain, trajectory and global-update seed streams.
Do not reuse selected pilot draws or choose seeds by their results.

Each tile must satisfy the existing gates: every monitored rank/folded split Rhat
<1.01, bulk and tail ESS >400, no retained HMC divergence/nonfinite proposal or
global numerical failure, positive HMC/global acceptance in every chain, completed
draw count, pixel MCSE p99 <=0.1 and maximum <=0.5 in 0-255 units. Maximum sampling
time 900 seconds/tile and 7,200 seconds cumulative. A failed gate stops the run;
no automatic extension, seed retry, fallback model, or suppressed failure.

## Preconditions, before full-image QEP sampling

1. Verify all bundled source/input hashes and original R source against its frozen
   commit. Require Python >=3.10 with NumPy, SciPy, pandas, Pillow, threadpoolctl.
   Python versions are recorded, and numerical checks must pass; Torch is unused.
2. Require R 4.5.1, EBImage 4.52.0, RobustGaSP 0.6.8, magick 2.9.1, pracma 2.4.6.
   No packages are installed or system settings changed automatically.
3. Validate the unchanged metric bodies against the four archived Raw/Fast-GP R
   outputs; validate analytic q=2 on every tile against the archived R full image
   (maximum normalized error <1e-10) and original Python formulas.
4. Re-run the unchanged 2D deterministic/Jacobian and small-chain reference gates
   on the execution host (not a new real-data experiment).
5. Run actual R Fast-GP reconstruction, then actual R downstream for Raw and
   Fast-GP on both images. Require analytic q=2/R maximum normalized error <1e-10,
   historical TP/FP/FN at 0.50/0.75 and failure counts exactly, foreground metrics
   within 1e-10, and the archived foreground/instance partition exactly (label
   renumbering allowed). Any mismatch blocks sampling; do not relax tolerances.
   Fast-GP AP50 anchors: nuclei 0.6481012658227848, whole-cell 0.583864118895966.

## Arms and downstream

Primary: Fast-GP analytic q=2 versus pooled-four-chain Fast-QEP q=1.5.
Secondary diagnostic: the deterministic proposal Gaussian mean g_eta, equivalently
a GP mean with nugget ratio eta*nu and the same per-tile mean/covariance scale.
Eta is determined by the already frozen proposal rule, without labels/AP. This
is a smoothing-strength control, not a fitted optimal GP or an exact QEP mean.
If it matches a QEP improvement, a uniquely QEP distributional benefit is unproven.
Raw is an anchor, not a newly tuned competitor.

For every arm, use the byte-unchanged R adapter from qep_downstream_4x2:
RobustGaSP criterion_1 and original outlier handling -> EBImage positive-distance
watershed with original implicit seeds -> eliminate_small_areas(50). The native
C/C++ inside those R packages is executed through Rscript, not rewritten in Python.
Use double-precision CSV at 17 significant digits, normalized by /255 in R.

Because the observed q effect is small, also pass each of the four chain-specific
mean reconstructions through the same R downstream. Report all four, AP range,
foreground disagreement, and improved/lost GT identities. These are Monte Carlo
stability diagnostics, not four independent data replicates; do not pick the best.

## Endpoints and interpretation

Primary: legacy AP50 = TP/(TP+FP+FN), retaining the original per-GT-best-IoU matching
rule. Secondary: AP75, merge/split/missed/spurious counts, foreground Dice/IoU,
per-GT gained/lost matches and new splits among formerly matched GT, runtime,
threshold/outlier changes and pixel MCSE. Failure categories are nonexclusive.
Reuse frozen evaluator function bodies through AST loading to avoid unrelated
Torch/skimage imports; record hashes and test cached anchors.

Report both images separately, including ties and harm. Descriptive support for
the intended merge-reduction route on an image requires AP50 increase, fewer
merges, no AP75 decrease, no increased split count, and no newly split previously
matched GT. These are conservative decision flags, not significance tests.
Also report whether AP50 gain stays positive in all four chain-mean segmentations.
An AP-only gain with other harm is a tradeoff. MC-unstable signs are unresolved.
No gain is evidence against this fixed substitution on these images, not a
refutation of QEP theory or every possible Fast-QEP construction. A positive
result needs a later, separately frozen held-out evaluation before generalization.

## Execution and recovery

Preflight is run before long sampling. Save a completed tile only after its gate
results and artifact hashes are written; resume skips only verified PASSED tiles
with the same source, input, runtime versions and configuration. An interrupted
or failed tile is retained and blocks automatic restart; review its logs first.
Do not overwrite existing project files or prior results. Keep >=8 GiB free.
Sample sequentially with one BLAS thread. Rough cloud projection is about 80
minutes plus preprocessing/output; Mac speed is unknown and the fixed cap applies.
The separate execution bundle does not alter the user's checkout or vendor the
original R reference repository.
