"""Memory-bounded observer version of the validated fixed-grid HMC loop."""
import time
import numpy as np
from py_core.fast_qep_prototype import leapfrog


def stream_hmc(target, config, fixture_id, observer, max_seconds=900, progress=None):
    if config.chains != 4 or not 1 <= config.warmup <= 5000 or not 4 <= config.draws <= 20000:
        raise ValueError('Outside the validated prototype bounds.')
    rngs = [np.random.default_rng(np.random.SeedSequence([config.base_seed, fixture_id, i]))
            for i in range(config.chains)]
    length_rng = np.random.default_rng(np.random.SeedSequence([config.base_seed, fixture_id, 999]))
    center = target.gaussian_white_mean()
    z = np.stack([center + 3*rng.normal(size=target.n)/np.sqrt(target.mass_diag) for rng in rngs])
    initial_positions = z.copy()
    value, _ = target.potential_gradient(z)
    logeps = np.full(config.chains, np.log(config.initial_step_size))
    logbar = logeps.copy()
    mu = np.log(10*config.initial_step_size)
    hbar = np.zeros(config.chains)
    accept_count = np.zeros(config.chains, dtype=int)
    divergent_count = np.zeros(config.chains, dtype=int)
    max_abs_delta = np.zeros(config.chains)
    retained = 0
    started = time.perf_counter()
    completed = True
    for iteration in range(config.warmup + config.draws):
        if time.perf_counter()-started > max_seconds:
            completed = False
            break
        momenta = np.stack([rng.normal(size=target.n) for rng in rngs])*np.sqrt(target.mass_diag)
        steps = int(length_rng.integers(config.min_steps, config.max_steps+1))
        proposed, p_final, v_final = leapfrog(target, z, momenta, np.exp(logeps), steps)
        with np.errstate(over='ignore', invalid='ignore'):
            delta = v_final + .5*np.sum(p_final**2/target.mass_diag, axis=1)
            delta -= value + .5*np.sum(momenta**2/target.mass_diag, axis=1)
        finite = np.isfinite(delta) & np.all(np.isfinite(proposed), axis=1)
        log_accept = np.where(finite, np.minimum(0, -delta), -np.inf)
        accepted = np.array([np.log(rng.uniform()) for rng in rngs]) < log_accept
        z = np.where(accepted[:, None], proposed, z)
        value = np.where(accepted, v_final, value)
        if iteration < config.warmup:
            t = iteration+1
            eta = 1/(t+10)
            hbar = (1-eta)*hbar + eta*(config.target_accept-np.exp(log_accept))
            logeps = np.clip(mu-np.sqrt(t)/.05*hbar, np.log(1e-5), np.log(config.max_step_size))
            weight = t**-.75
            logbar = weight*logeps + (1-weight)*logbar
            if t == config.warmup:
                logeps = logbar.copy()
        else:
            observer(z, retained)
            retained += 1
            accept_count += accepted
            divergent_count += (~finite) | (abs(delta)>100)
            max_abs_delta = np.maximum(max_abs_delta, np.where(finite, abs(delta), np.inf))
        if progress and (iteration+1) % 250 == 0:
            progress(iteration+1, config.warmup+config.draws, time.perf_counter()-started)
    return {
        'completed': completed, 'retained_per_chain': retained,
        'runtime_sec': time.perf_counter()-started,
        'step_sizes': np.exp(logeps).tolist(),
        'acceptance_rate': (accept_count/max(1, retained)).tolist(),
        'divergences_abs_delta_H_gt_100': divergent_count.tolist(),
        'max_abs_delta_H': max_abs_delta.tolist(),
        'seed_components': [[config.base_seed, fixture_id, i] for i in range(config.chains)],
        'trajectory_length_seed': [config.base_seed, fixture_id, 999],
        'mass_is_fixed_gaussian_reference': True, 'adaptation_after_warmup': False,
    }, initial_positions, z.copy()


class BlockObserver:
    """Store full-coordinate block means and predetermined diagnostic traces."""
    def __init__(self, target, draws, block=100):
        if draws % (block*5):
            raise ValueError('Draw count must support both prescribed batch sizes.')
        self.target, self.block = target, block
        self.indices = np.unique(np.linspace(0, target.n-1, min(64, target.n), dtype=int))
        shape = (4, draws//block, target.n)
        self.raw_blocks = np.zeros(shape)
        self.cv_blocks = np.zeros(shape)
        self.traces = np.zeros((4, draws, 4+3*len(self.indices)))
        self.names = ['radius', 'log_likelihood', 'potential', 'latent_squared_norm']
        self.names += [f'{kind}_{i}' for kind in ['z', 'latent', 'controlled'] for i in self.indices]
        self.count = 0

    def __call__(self, z, j):
        h = self.target.latent(z)
        potential, gradient = self.target.potential_gradient(z)
        cv = h-self.target.score_coefficient*gradient
        self.raw_blocks[:, j//self.block] += h/self.block
        self.cv_blocks[:, j//self.block] += cv/self.block
        global_stats = np.column_stack([
            np.linalg.norm(z, axis=1),
            -.5*np.sum((h-self.target.observations)**2, axis=1)/self.target.noise_variance,
            potential, np.sum(h*h, axis=1)])
        self.traces[:, j] = np.column_stack([global_stats,z[:,self.indices],h[:,self.indices],cv[:,self.indices]])
        self.count = j+1

    def complete_blocks(self):
        nblocks = self.count//self.block
        return self.raw_blocks[:, :nblocks], self.cv_blocks[:, :nblocks]


def pixel_block_summary(grid, blocks):
    """Covariance-aware MCSE by transforming entire block vectors to pixels."""
    chains, count, n = blocks.shape
    if count < 10 or count % 5:
        raise ValueError('Require complete batches of 100 and 500.')
    pixel = np.empty((chains,count,*grid.shape))
    for c in range(chains):
        for b in range(count):
            pixel[c,b] = grid.inverse(blocks[c,b])
    chain_means = pixel.mean(axis=1)
    mean = chain_means.mean(axis=0)
    small = np.sqrt(pixel.var(axis=1,ddof=1).mean(axis=0)/(chains*count))
    coarse = pixel.reshape(chains,count//5,5,*grid.shape).mean(axis=2)
    large = np.sqrt(coarse.var(axis=1,ddof=1).mean(axis=0)/(chains*(count//5)))
    between = chain_means.std(axis=0,ddof=1)/np.sqrt(chains)
    return {'mean':mean,'chain_means':chain_means,
            'mcse':np.maximum.reduce([small,large,between]),
            'mcse_block100':small,'mcse_block500':large,'mcse_between':between}


def verify_stream_equivalence():
    from py_core.fast_qep_prototype import SpectralPosterior,HMCConfig,sample_hmc
    config = HMCConfig(warmup=40,draws=40)
    comparisons = []
    for q in (2.,1.5):
        target = SpectralPosterior(np.array([.4,1.6]),np.array([1.2,-.8]),1.,.35,q)
        reference, metadata = sample_hmc(target,config,707)
        actual = np.zeros_like(reference)
        result, _, _ = stream_hmc(target,config,707,
                                  lambda z,j: actual.__setitem__((slice(None),j),z))
        err = float(np.max(abs(actual-reference)))
        assert err == 0
        for key in ['step_sizes','acceptance_rate','divergences_abs_delta_H_gt_100','max_abs_delta_H']:
            assert result[key] == metadata[key]
        comparisons.append({'q':q,'max_draw_difference':err,'sampler_metadata_matches':True})
    return comparisons
