# Fast-QEP hyperparameter score pilot, 2026-10-15

Status: implementation validated locally; this protocol precedes real-image fitting.
The user approved route-1 feasibility/integration. This bundle prepares the next
first-tile experiment for review; building/checking it does not launch that experiment.

## Target and hypothesis

Keep the existing finite-n covariance-matched qED prior and independent Gaussian
noise: y=m1+v+epsilon, Cov(v)=sK, Cq=sK/a(n,q), noise variance=s*nu.
q=1.5 only; q=2 analytic GP. K is the original separable Matern-5/2 with
normalized coordinates and column-major vectorization. No changed radius,
per-coordinate q prior, jitter, eigenvalue clipping or tile reduction.

Theta=(m, log s, log nu, log beta_row, log beta_col). Unlike the prior fixed run,
the Gaussian profiles initialize QEP; they are not claimed to optimize QEP.
Fisher scores are derivatives at fixed physical latent v. In particular the
log-s score includes the noise likelihood contribution because tau2=s*nu.

Question: at archived R parameters and at a refitted GP stationary point, is the
QEP evidence gradient distinguishable from numerical sampling noise? No AP claim.

## Exact score controls

In white coordinates let Uq be the actual negative log posterior, P the fixed
Gaussian precision, mu the Gaussian white posterior mean. A fixed affine vector
field F gives a zero-mean control C=F dot grad(Uq) - div(F). Coefficients are fixed
before sampling, never estimated from outcomes. The boundary term vanishes for
these affine fields under the confining white posterior used here.

For a Gaussian-reference complete score g2=z'Bz+l'z+c, with B symmetric, take
F=P^-1 B(z-mu)+P^-1(2Bmu+l). Then at q=2, g2-C is the analytic evidence score
for every z. For q=1.5, use its TRUE complete score minus the SAME valid control.
This is variance reduction for evidence scores, not a Gaussian approximation to
the QEP target. Raw scores are saved alongside controlled scores.

Whitened kernel derivatives D=K^-1/2 (dK/dlog beta) K^-1/2 are evaluated through
row/column factors. m, log-s, log-nu and kernel controls are all checked against
independent integration; q=2 is checked pointwise. Old posterior-mean controls
are never fed to nonlinear moments; nuisance ECM moments use raw latent samples.

## Images and GP control

Original first tile only: nuclei 282x240; whole-cell 200x200. Byte-frozen images,
/255 normalization and archived R parameters. All existing 53 bundle hashes are
checked. Fixed q=2 reconstructions match the archived R first tile within 1e-10.
Ground-truth masks may be present and hashed as frozen inputs; they are not
decoded, used for fitting, or scored. R is not executed in this pre-segmentation
pilot. Later segmentation still requires the actual R downstream.

GP evidence is analytic; profile m and s exactly. Optimize (log beta_row,
log beta_col, log nu), evidence per pixel, L-BFGS-B, maxiter=200, maxls=20,
ftol=1e-12, gtol=1e-8. Two starts: archived R and beta_row=beta_col=20, nu=.1.
Choose the highest GP evidence of these two, never AP. Report both fits, success,
boundary hits and projected gradient. Require success and maximum projected
gradient per pixel <=1e-6 for the selected point before QEP sampling.

Declared pilot domain: beta in [5, max(archived beta,20*(axis_size-1))] on each
axis; nu in [1e-6,10]. Lower beta bounds restrict numerically delicate very-long
ranges; the upper domain includes the archived identity-kernel plateau. These
are computational restrictions, not full-model optima or evidence of preferable
biology. Record boundaries and condition diagnostics. Any future QEP fitting
comparison must use the identical domain. A nonpositive eigenvalue stops the
fit; no fallback or hidden regularization. GP fitting cap 300 seconds/dataset.

## Sampling and score precision

Use byte-unchanged joint_mh.py and its warmup-only adaptation. Four chains,
2000 warmup and 4000 retained draws; 8-12 HMC steps; base seed 20261015.
Target order/fixture IDs: nuclei archived 8101, nuclei refitted 8102,
whole-cell archived 8111, whole-cell refitted 8112. All coordinates retain a
single joint radius. Proposal root, acceptance and numerical failure rules are
unchanged. No seed retry, automatic extension or replacement proposal.

Full original diagnostics run on all retained draws. Calculate exact hyperparameter
scores at every tenth retained draw (400 per chain) to bound matrix-operation
cost. This is thinning of a scalar estimator, not a chunked posterior or a new
radius. Save raw and controlled score traces and their rank/folded split R-hat,
bulk/tail ESS and conservative MCSE. Score controls must satisfy R-hat<1.01 and
both ESS>400; constant zero derivatives on a flat kernel axis are reported as
zero information, not an informative direction.

Retain original gates for all monitored latent quantities and sampler numerics:
R-hat<1.01, bulk/tail ESS>400; no divergence/nonfinite proposal/global failure;
positive HMC/global acceptance in every chain; completed draw count. Pixel mean
MCSE in 0-255 units: p99<=.1, max<=.5. At most 900 seconds per sampling target,
7200 seconds overall pilot work after local correctness checks. Failed gates stop.

Scale the m derivative by the tile SD and divide all five scores by n. Report
QEP score, MCSE and QEP-minus-analytic-GP score. A descriptive resolved flag
requires abs(score difference)>3*MCSE AND scaled abs difference>1e-8. Also report
the equivalent flag for QEP's own score. These are screens, not hypothesis-test
p-values or a multiple-testing correction. Boundary directions must be interpreted
within the feasible domain, and tiny/flat derivatives do not rule out other basins.

## Updates and stopping interpretation

One raw-moment conditional ECM candidate for m/s/nu is saved but NOT applied.
It is not accepted optimization, not an MCMC-ascent guarantee and not a QEP fit.
An error-controlled full optimizer and step acceptance require a later design;
do not silently run the prior review's eight-step fitting budget in this pilot.

Resolved, reproducible QEP directions justify evaluating a bounded learning
procedure. Unresolved directions limit support at these two parameter points;
they do not disprove alternative optima. Numerical failure is inconclusive for
the model. No full images, new thresholding, watershed, segmentation AP or held-out
tests are produced. Existing negative results remain unchanged.

For a later whole-image comparison retain archived R, refitted GP, refitted QEP,
and GP-at-QEP-parameters; complete the distribution-by-parameter-source 2x2 when
attributing gains. Optimize only on image intensities; keep original downstream
fixed. First-tile shared beta/nu and per-tile nuisance fitting must be symmetric.

## Recovery

Separate outputs for --check and --pilot. Existing output directories are refused.
No automatic resume of failed inference. Every run records source/input hashes,
package versions and status; a review ZIP is produced on completion or failure.
No source or previous results are overwritten. Use the existing gpytorch_arm env.
