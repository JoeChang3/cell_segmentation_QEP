"""Original white HMC composed with an exact global latent-space MH update."""
import time
import numpy as np
from scipy.optimize import brentq
from py_core.fast_qep_prototype import leapfrog
from independence_mh import to_white


class TiltedGaussian:
    def __init__(self, target):
        self.target = target
        d = target.gaussian_root ** 2
        self.d = d
        def moments(eta):
            mean = d * target.observations / (d + eta * target.noise_variance)
            var = d * target.noise_variance / (d + eta * target.noise_variance)
            return mean, var, float(np.sum((mean * mean + var) / d))
        self.moments = moments
        def kappa(t):
            return target.n * (2-target.q)/(2*t) + target.q/2 * target.a**(target.q/2) * t**(target.q/2-1)
        self.kappa = kappa
        def equation(eta):
            return eta-kappa(moments(eta)[2])
        self.brackets = []
        if target.q == 2:
            self.eta = 1.
        else:
            grid = np.logspace(-6, 6, 121)
            ff = np.array([equation(e) for e in grid])
            if not np.isfinite(ff).all():
                raise ValueError('Nonfinite proposal equation.')
            exact = list(np.flatnonzero(ff == 0))
            sign = list(np.flatnonzero(ff[:-1] * ff[1:] < 0))
            if len(exact)+len(sign) != 1:
                raise ValueError('Require exactly one detected proposal root; no fallback.')
            if exact:
                self.eta = float(grid[exact[0]])
                self.brackets = [[self.eta, self.eta]]
            else:
                i = sign[0]; lo, hi = float(grid[i]), float(grid[i+1])
                self.brackets = [[lo, hi]]
                self.eta = float(brentq(equation, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100))
        self.mean, self.variance, self.t0 = moments(self.eta)
        self.sd = np.sqrt(self.variance)
        self.residual = abs(equation(self.eta))/max(1., self.eta)
        if not np.isfinite(self.residual) or self.residual > 1e-10:
            raise ValueError('Proposal equation residual too large.')
        self.alpha = target.q/2
        self.c = (target.a*self.t0)**self.alpha
        self.linear = target.n*(target.q-2)/4 - .5*self.c*self.alpha + .5*self.eta*self.t0

    def quadratic(self, v):
        t = np.sum(v*v/self.d, axis=-1)
        if np.any(t <= 0) or not np.isfinite(t).all():
            raise ValueError('Nonpositive or nonfinite prior quadratic form.')
        return t

    def log_weight(self, v):
        if self.target.q == 2:
            return np.zeros(np.shape(v)[:-1])
        t = self.quadratic(v)
        ell = np.log1p((t-self.t0)/self.t0)
        result = (.5*self.eta*self.t0*(np.expm1(ell)-ell)
                  -.5*self.c*(np.expm1(self.alpha*ell)-self.alpha*ell)
                  +self.linear*ell)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite global MH log weight.')
        return result

    def record(self):
        return {'eta': self.eta, 'reference_E_T': self.t0,
                'reference_E_T_over_n': self.t0/self.target.n,
                'root_relative_residual': self.residual, 'detected_brackets': self.brackets,
                'fixed_before_sampling': True, 'changes_target_model': False}


def global_step(target, proposal, z, rngs):
    v = target.latent(z)
    proposed = np.stack([proposal.mean+proposal.sd*r.normal(size=target.n) for r in rngs])
    old_weight = proposal.log_weight(v)
    new_weight = proposal.log_weight(proposed)
    accepted = np.array([np.log(r.uniform()) for r in rngs]) < np.minimum(0, new_weight-old_weight)
    # Transform the matched proposal and target together; no extra Jacobian in MH ratio.
    proposed_white = to_white(target, proposed)
    if not np.isfinite(proposed_white).all():
        raise ValueError('Nonfinite inverse-white proposal.')
    return (np.where(accepted[:,None], proposed_white, z), accepted,
            np.where(accepted, new_weight, old_weight), new_weight)


def sample(target, config, fixture_id, observer, max_seconds=900, progress=None, global_enabled=True):
    if config.chains != 4 or not 1 <= config.warmup <= 5000 or not 4 <= config.draws <= 20000:
        raise ValueError('Outside fixed prototype bounds.')
    setup_started = time.perf_counter()
    proposal = TiltedGaussian(target)
    setup_seconds = time.perf_counter()-setup_started
    rngs = [np.random.default_rng(np.random.SeedSequence([config.base_seed, fixture_id, c])) for c in range(4)]
    global_rngs = [np.random.default_rng(np.random.SeedSequence([config.base_seed, fixture_id, c, 888])) for c in range(4)]
    length_rng = np.random.default_rng(np.random.SeedSequence([config.base_seed, fixture_id, 999]))
    center = target.gaussian_white_mean()
    z = np.stack([center+3*r.normal(size=target.n)/np.sqrt(target.mass_diag) for r in rngs])
    initial = z.copy(); value, _ = target.potential_gradient(z)
    logeps = np.full(4, np.log(config.initial_step_size)); logbar = logeps.copy()
    mu = np.log(10*config.initial_step_size); hbar = np.zeros(4)
    accepted_count = np.zeros(4, dtype=int); global_count = np.zeros(4, dtype=int)
    warmup_global_count = np.zeros(4, dtype=int)
    rejection_run = np.zeros(4, dtype=int); longest_rejection = np.zeros(4, dtype=int)
    divergences = np.zeros(4, dtype=int); nonfinite = np.zeros(4, dtype=int); max_delta = np.zeros(4)
    traces = {k: np.zeros((4, config.draws)) for k in
              ['hmc_delta_H', 'hmc_accepted', 'global_accepted', 'global_log_weight', 'proposal_log_weight']}
    started = time.perf_counter(); retained = 0; completed = True; failure = None; global_failures = 0
    for iteration in range(config.warmup+config.draws):
        if time.perf_counter()-started > max_seconds:
            completed = False; failure = 'TIME_BUDGET'; break
        momentum = np.stack([r.normal(size=target.n) for r in rngs])*np.sqrt(target.mass_diag)
        steps = int(length_rng.integers(config.min_steps, config.max_steps+1))
        proposed, p_final, v_final = leapfrog(target, z, momentum, np.exp(logeps), steps)
        with np.errstate(over='ignore', invalid='ignore'):
            delta = v_final+.5*np.sum(p_final**2/target.mass_diag, axis=1)
            delta -= value+.5*np.sum(momentum**2/target.mass_diag, axis=1)
        finite = np.isfinite(delta) & np.all(np.isfinite(proposed), axis=1)
        log_accept = np.where(finite, np.minimum(0, -delta), -np.inf)
        accepted = np.array([np.log(r.uniform()) for r in rngs]) < log_accept
        z = np.where(accepted[:,None], proposed, z); value = np.where(accepted, v_final, value)
        if iteration < config.warmup:
            t = iteration+1; rate = 1/(t+10)
            hbar = (1-rate)*hbar+rate*(config.target_accept-np.exp(log_accept))
            logeps = np.clip(mu-np.sqrt(t)/.05*hbar, np.log(1e-5), np.log(config.max_step_size))
            weight = t**-.75; logbar = weight*logeps+(1-weight)*logbar
            if t == config.warmup:
                logeps = logbar.copy()
        ga = np.zeros(4, dtype=bool); lw = np.zeros(4); plw = np.zeros(4)
        if global_enabled:
            try:
                z, ga, lw, plw = global_step(target, proposal, z, global_rngs)
                value, _ = target.potential_gradient(z)
                if not np.isfinite(value).all():
                    raise FloatingPointError('Nonfinite post-global potential.')
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                global_failures += 1; completed = False; failure = repr(exc); break
        if iteration < config.warmup:
            warmup_global_count += ga
        else:
            observer(z, retained)
            for k, v in [('hmc_delta_H', delta), ('hmc_accepted', accepted), ('global_accepted', ga),
                         ('global_log_weight', lw), ('proposal_log_weight', plw)]:
                traces[k][:,retained] = v
            retained += 1; accepted_count += accepted; global_count += ga
            divergences += (~finite) | (abs(delta)>100); nonfinite += ~finite
            max_delta = np.maximum(max_delta, np.where(finite, abs(delta), np.inf))
            rejection_run = np.where(ga, 0, rejection_run+1)
            longest_rejection = np.maximum(longest_rejection, rejection_run)
        if progress and (iteration+1)%1000 == 0:
            progress(iteration+1, config.warmup+config.draws, time.perf_counter()-started)
    result = {
        'completed': completed, 'failure': failure, 'retained_per_chain': retained,
        'runtime_sec': time.perf_counter()-started, 'proposal_setup_sec': setup_seconds,
        'proposal': proposal.record(), 'step_sizes': np.exp(logeps).tolist(),
        'acceptance_rate': (accepted_count/max(1, retained)).tolist(),
        'global_acceptance_rate': (global_count/max(1, retained)).tolist(),
        'warmup_global_acceptance_rate': (warmup_global_count/config.warmup).tolist(),
        'max_consecutive_global_rejections': longest_rejection.tolist(),
        'divergences_abs_delta_H_gt_100': divergences.tolist(), 'max_abs_delta_H': max_delta.tolist(),
        'nonfinite_retained_hmc_proposals': nonfinite.tolist(), 'global_numerical_failures': global_failures,
        'seed_components': [[config.base_seed, fixture_id, c] for c in range(4)],
        'global_seed_components': [[config.base_seed, fixture_id, c, 888] for c in range(4)],
        'trajectory_length_seed': [config.base_seed, fixture_id, 999],
        'global_enabled': global_enabled, 'adaptation_after_warmup': False,
        'mass_is_fixed_gaussian_reference': True,
    }
    return result, initial, z.copy(), {k: v[:,:retained] for k, v in traces.items()}
