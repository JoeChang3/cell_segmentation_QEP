# Fast-QEP model contract: fixed-grid prior replacement

Status: declared model for the next inference validation, not a complete
process-level theorem or a reproduction of the two papers' full latent-variable
architectures. This records the EXISTING target; it does not change past runs.

## Probability model

For one complete original tile with n pixels, column-major vectorization:

    K = K_col(beta_col) tensor K_row(beta_row), both original Matern-5/2;
    a(n,q) = 2^(2/q) Gamma(n/2+2/q) / [n Gamma(n/2)];
    C_q = s0 K / a(n,q);
    v ~ q-ED_n(0,C_q);
    epsilon ~ N(0,tau² I), independent of v;
    y = m 1 + v + epsilon, tau² = s0 nu.

The q-ED density includes r^[n(q-2)/4]. Do not delete the log-r term, replace
the prior by independent scalar q coefficients, or allocate one radius per
computational chunk. All n coordinates share the vector's radius.

## Parameters and comparison

- q=2 analytic Fast-GP control; one fixed q=1.5 candidate.
- Keep original R beta and nu; same tile GLS m and profiled s0=S/n for both q.
- These are empirical-Bayes plug-in values. No q-dependent fitting and no
  integration over parameter uncertainty is performed in the current comparison.
- Exact covariance matching is intentional. It is a finite-n amplitude
  reparameterization of the paper's q-ED family, not the identical convention
  as holding its dimension-scaled kernel amplitude fixed.
- Same original tile shape, coordinates, grayscale convention and pixel order.
  Changing tile size changes this finite-dimensional model.
- Missing observations, nonuniform noise and nonlinear observation operators
  are outside this contract.

## Target and estimator

With K=U Lambda U^T, w=U^T(y-m1), D_j=sqrt(s0 lambda_j/a),

    h(z)=D z ||z||^(2/q-1), z initially has N(0,I) prior;
    log pi(z|y)=constant - ||w-h(z)||²/(2 tau²) - ||z||²/2;
    reconstruction=m1 + U E[h(z)|y].

No extra Jacobian belongs in this z-density. A change to log radial coordinates
does require the polar-measure factor described in MATH_NOTE.md.

The fixed score control variate estimates the SAME posterior mean. Controlled
values are not posterior samples for variance/quantiles. Save ordinary means too.
Do not replace E[h(z)] by h(E[z]) or by a MAP optimizer.

At q=2 this gives the original Fast-GP posterior mean. Full q=2 parameter-refitting
parity is a separate issue from the completed fixed-parameter reconstruction parity.

## Interpretation

The model is a legitimate finite-n prior replacement with Gaussian observation
noise. Its validity does not require the disputed additive/conditional q-ED
closure statements. Do not call it the full joint spatial QEP posterior across
arbitrary grids. Do not call it the original Reg_Rep or DeepQEP training model.

Sampler failure is computational evidence only. Passing inference checks enables
a comparison of model outputs, not a claim of improved reconstruction or instance
segmentation. No AP-driven sampler or q selection is permitted.
