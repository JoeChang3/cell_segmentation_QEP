# HMC plus tilted-Gaussian global MH, frozen before execution

Run label: 2026-10-15 UTC (2026-10-14 in America/Phoenix).
Continue the explicitly approved Fast-QEP inference investigation. Preserve the
MODEL_CONTRACT.md target and every earlier result. No q or model search.

## Why this particular change

The previous conditional radial update passed the two small reference gates and
nuclei tile 2, but whole-cell tile 2 retained radial bulk ESS 346.30 (<400), with
lag-1 correlation about .94. Its three failing quantities are functions of the
same radius. A move holding direction fixed remains narrow. Earlier independent
GP proposals moved all coordinates but failed on nuclei because the fixed GP
radial location differed from the QEP posterior location.

Use a local radial tangent to construct ONE tilted Gaussian proposal. Correct
every global proposal with the exact full-target MH ratio. Retain ordinary HMC
to explore directions and to relax the same dispersed starts. This is a change
to the transition kernel, not a change of likelihood, prior, fitted amplitude,
q, reconstruction estimator or downstream. The proposal mean is NOT the QEP mean.

## Deterministic proposal rule

Let d_j=s0*lambda_j and T=sum(v_j^2/d_j). For one fixed eta>0 define

    g_eta(v) = Normal(m_eta, diag(s_eta^2)),
    m_eta,j = d_j*w_j/(d_j+eta*tau²),
    s_eta,j² = d_j*tau²/(d_j+eta*tau²).

Set eta=kappa(E_g_eta[T]) where

    kappa(t)=n(2-q)/(2t)+(q/2)*a^(q/2)*t^(q/2-1).

This local-tangent/moment self-consistency is a proposal heuristic, not an exact
posterior moment or KL optimum. q=2 returns eta=1 exactly. For q=1.5 scan 121
fixed log-spaced eta points 1e-6 to 1e6. Require exactly one detected root bracket
(or one exact scan root), then brentq xtol=rtol=1e-12, maxiter=100, relative
equation residual <=1e-10. Failure blocks that run. No alternative root choice,
bracket expansion, optimization against AP or proposal-scale sweep.

At this fixed eta, log(pi_q(v|y)/g_eta(v)), up to a constant, is

    R_eta(T)=n(q-2)/4*log T - .5*(a*T)^(q/2) + .5*eta*T.

Evaluate it centered at T0=E_g_eta[T], using log1p/expm1 and subtracting the
linear terms to reduce cancellation. At q=2 return exactly zero. Use the
independence acceptance min(1,exp(R_eta(T_new)-R_eta(T_current))). Transform an
accepted full latent vector back to the original white z coordinates. The
Jacobian is accounted for by using the matching latent target AND latent
proposal; check directly against transformed white densities. No radial chunks.

## Kernel and budget

- HMC as before: fixed Gaussian-reference mass, 8–12 leapfrog steps, initial
  epsilon .25, target acceptance .85, max epsilon 1.5, warmup-only adaptation.
- Each iteration: HMC, then one global MH proposal; recompute white potential
  after the global step. No conditional radial slice in this kernel.
- Four chains, starts Gaussian-reference white mean plus 3 posterior SD normal
  vectors, unchanged construction; do not inherit or select old final states.
- Seeds [20261015,fixture,chain] for HMC; [20261015,fixture,999] for length;
  [20261015,fixture,chain,888] for global proposals and global acceptance.
- No sampling beyond 900 seconds per fixture, excluding proposal setup and
  result postprocessing. Record both component acceptance rates, rejected-step
  runs, global log weights, HMC energy errors, initial and final states.

## Gates

1. Deterministic n=2,20,40000, both q: full latent density/proposal ratio vs
   centered R differences (<1e-7); transformed white/Jacobian ratio (<1e-7);
   inverse whitening round trip (<1e-10); root residual <=1e-10; exact q=2
   proposal mean/variance and zero log weights. Disabling global updates must
   reproduce the old stream_hmc draws and HMC metadata exactly (40+40, ID619).
2. Same 2D references y=(1.2,-.8), prior covariance diag(.4,1.6), tau²=.35.
   q=2 ID601, q=1.5 ID602; each chain 2,000 warmup +20,000 retained draws.
   Reuse existing independent quadrature values. Ordinary q=2 means <=5 MCSE;
   ordinary q=1.5 means <=3 MCSE; q=1.5 controlled means additionally absolute
   error <=.001 and MCSE <=.001; second/cross moments <=5 MCSE. q=2 centered
   covariance and white radius second/fourth moments <=5 MCSE. All prescribed
   nonconstant statistics Rhat<1.01 and bulk/tail ESS>400. No retained HMC
   divergence abs(deltaH)>100, nonfinite event or global numerical failure;
   positive retained acceptance in each component/chain; q=2 global acceptance
   exactly one. Attempt both small fixtures once; any failure blocks image runs.
3. Only if both small fixtures pass: the SAME nuclei and whole-cell tile 2,
   original R parameter reuse, byte hashes and q2 parity checks unchanged.
   IDs603/604; 2,000 warmup +4,000 retained per chain. Attempt both once.
   Monitor the prior fixed 64 spectral indices and globals, prior T, target/GP
   log weight, PLUS target/tilted-proposal log weight. Apply the same Rhat/ESS,
   completion, finite-numerics and component-acceptance gates. Controlled pixel
   MCSE p99<=.1 and max<=.5 in 0–255 units. Save ordinary and unchanged score-
   controlled mean estimates, full-coordinate block means and all fixed traces.

No seed retries, chain extensions, eta sweeps, alternative kernels, changed
estimators, extra tiles, full-image reconstruction or downstream segmentation.
A genuine coding bug may be corrected transparently with failure preserved;
a statistical gate failure is a result and stops expansion. Passing finite-run
gates is not proof of global mixing, coverage of all tails or segmentation gain.
Gaussian proposal weights may remain unbounded near the origin or in tails;
tilting does not establish uniform/geometric ergodicity of independence MH.

Postprocess saved results only: compare prescribed diagnostics and computational
cost with earlier runs (different seeds, no claim of controlled speed benchmark),
plot trajectories and mean differences with physical units, identify failures.
No new R execution is available; do not replace RobustGaSP/EBImage by Python.
