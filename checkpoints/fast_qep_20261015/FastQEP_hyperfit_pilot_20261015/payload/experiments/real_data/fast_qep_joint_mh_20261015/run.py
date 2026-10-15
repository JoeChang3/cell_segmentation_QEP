"""Frozen correctness gates followed by the two existing tile-2 targets."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREV = ROOT/'results/fast_qep_radial_hmc_20261014'
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT/'audits/fast_qep_radial_diagnosis_20261014'),
               str(ROOT/'audits/fast_qep_prototype_20261014'),
               str(ROOT/'experiments/real_data/fast_qep_real_pilot_20261014')]
from py_core.fast_qep_prototype import SpectralPosterior, HMCConfig
from joint_mh import TiltedGaussian, sample
from independence_mh import to_white, centered_log_weight
from stream_hmc import stream_hmc, pixel_block_summary
from diagnostics import summarize
from run_pilot import prepare, save, digest, load_original

PREV_RUN = ROOT/'experiments/real_data/fast_qep_radial_hmc_20261014/run.py'
spec = importlib.util.spec_from_file_location('previous_radial_validation', PREV_RUN)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)


class Observer(old.Observer):
    def __init__(self, target, draws):
        super().__init__(target, draws)
        self.proposal = TiltedGaussian(target)
        self.global_weight = np.empty((4, draws))

    def __call__(self, z, j):
        super().__call__(z, j)
        self.global_weight[:,j] = self.proposal.log_weight(self.target.latent(z))

    def diagnostics(self):
        result = super().diagnostics()
        if self.target.q != 2:
            result['target_to_tilted_log_weight'] = summarize(self.global_weight[:,:self.count])
        return result


def config(draws):
    return HMCConfig(warmup=2000, draws=draws, base_seed=20261015)


def checks():
    rng = np.random.default_rng(106001); rows = []
    for n in (2, 20, 40000):
        for q in (2., 1.5):
            target = SpectralPosterior(np.linspace(.4, 1.6, n), np.linspace(-.8, 1.2, n), 1., .35, q)
            prop = TiltedGaussian(target)
            z = rng.normal(size=(4, n))*np.array([.7, .9, 1.1, 1.3])[:,None]
            v = target.latent(z); inv = to_white(target, v)
            roundtrip = float(np.max(abs(inv-z)))
            assert roundtrip < 1e-10
            t = prop.quadratic(v); r = target.a*t
            logprior = n*(q-2)/4*np.log(r)-.5*r**(q/2)-np.log(target.root_scatter).sum()
            loglik = -.5*np.sum((v-target.observations)**2, axis=1)/target.noise_variance
            logproposal = -.5*np.sum((v-prop.mean)**2/prop.variance, axis=1)-np.log(prop.sd).sum()
            direct = logprior+loglik-logproposal
            lw = prop.log_weight(v)
            error = float(np.max(abs((lw-lw[0])-(direct-direct[0]))))
            rho = np.linalg.norm(z, axis=1)
            logjac = np.log(target.root_scatter).sum()+np.log(2/q)+n*(2/q-1)*np.log(rho)
            white_ratio = -target.potential_gradient(z)[0]-logjac-logproposal
            jac_error = float(np.max(abs((white_ratio-white_ratio[0])-(lw-lw[0]))))
            previous = centered_log_weight(target, v)+.5*(prop.eta-1)*(t-n)
            old_error = float(np.max(abs((previous-previous[0])-(lw-lw[0]))))
            assert max(error, jac_error, old_error) < 1e-7
            if q == 2:
                assert prop.eta == 1 and np.array_equal(lw, np.zeros(4))
                assert np.max(abs(prop.mean-target.gaussian_mean())) < 1e-12
                assert np.max(abs(prop.variance-target.gaussian_root**2/target.mass_diag)) < 1e-12
            rows.append({'n': n, 'q': q, 'proposal': prop.record(), 'roundtrip_max_error': roundtrip,
                         'latent_density_ratio_error': error, 'white_jacobian_ratio_error': jac_error,
                         'previous_weight_ratio_error': old_error})
    equality = []
    for q in (2., 1.5):
        target = SpectralPosterior(np.array([.4, 1.6]), np.array([1.2, -.8]), 1., .35, q)
        cfg = HMCConfig(warmup=40, draws=40, base_seed=20261015)
        aa = np.empty((4, 40, 2)); bb = np.empty_like(aa)
        ma, ia, fa = stream_hmc(target, cfg, 619, lambda z,j: aa.__setitem__((slice(None),j), z))
        mb, ib, fb, _ = sample(target, cfg, 619, lambda z,j: bb.__setitem__((slice(None),j), z), global_enabled=False)
        assert np.array_equal(aa, bb) and np.array_equal(ia, ib) and np.array_equal(fa, fb)
        for k in ('step_sizes', 'acceptance_rate', 'divergences_abs_delta_H_gt_100', 'max_abs_delta_H'):
            assert ma[k] == mb[k]
        equality.append({'q': q, 'all_draws_identical': True})
    return {'density_checks': rows, 'disabled_global_equivalence': equality, 'passed': True}


def gates(diag, sampler, q):
    return {
        'convergence': all(d['rhat_rank_folded_split'] < 1.01 and d['ess_bulk'] > 400 and d['ess_tail'] > 400 for d in diag.values()),
        'sampler_numerics': sampler['completed'] and sampler['global_numerical_failures'] == 0
            and sum(sampler['divergences_abs_delta_H_gt_100']) == 0
            and sum(sampler['nonfinite_retained_hmc_proposals']) == 0
            and min(sampler['acceptance_rate']) > 0 and min(sampler['global_acceptance_rate']) > 0,
        'q2_exact_global_acceptance': q != 2 or sampler['global_acceptance_rate'] == [1., 1., 1., 1.],
    }


def small(out):
    refs = json.loads((ROOT/'audits/fast_qep_prototype_20261014/validation/deterministic_checks.json').read_text())['quadrature']
    results = []
    for fid, q in [(601, 2.), (602, 1.5)]:
        target = SpectralPosterior(np.array([.4, 1.6]), np.array([1.2, -.8]), 1., .35, q)
        obs = Observer(target, 20000); z = np.empty((4, 20000, 2))
        def record(zz, j):
            obs(zz, j); z[:,j] = zz
        print(f'Start 2D q={q}; proposal={obs.proposal.record()}', flush=True)
        sampler, initial, final, traces = sample(target, config(20000), fid, record,
            progress=lambda i,n,s: print(f'2D q={q}: {i}/{n}, {s:.1f}s', flush=True) if i%5000 == 0 else None)
        z = z[:,:obs.count]
        np.savez_compressed(out/f'2d_q{q:g}_chains.npz', white_draws=z, trace=obs.traces[:,:obs.count],
            trace_names=np.asarray(obs.names), extra_trace=obs.extra[:,:obs.count], initial_z=initial, final_z=final, **traces)
        result = {'q': q, 'sampler': sampler, 'passed': False}
        if sampler['completed']:
            diag = obs.diagnostics(); gate = gates(diag, sampler, q)
            v = target.latent(z); cv = target.score_corrected_latent(z); truth = refs[str(q)]; comparisons = []
            def compare(name, values, reference, multiple, absolute=False):
                d = summarize(values); error = abs(d['mean']-reference)
                passed = error <= multiple*d['mcse'] and (not absolute or (error <= .001 and d['mcse'] <= .001))
                comparisons.append({'quantity': name, 'reference': reference, 'estimate': d['mean'],
                    'abs_error': error, 'mcse': d['mcse'], 'mcse_multiple': multiple, 'passed': passed})
            for j in range(2):
                compare(f'raw_mean_{j}', v[:,:,j], truth['mean'][j], 5 if q == 2 else 3)
                compare(f'second_moment_{j}', v[:,:,j]**2, truth['second_moments'][j], 5)
                if q != 2:
                    compare(f'controlled_mean_{j}', cv[:,:,j], truth['mean'][j], 3, True)
            compare('cross_moment', v[:,:,0]*v[:,:,1], truth['cross_moment'], 5)
            if q == 2:
                m = target.gaussian_mean(); var = target.gaussian_root**2/target.mass_diag
                for j in range(2):
                    compare(f'centered_variance_{j}', (v[:,:,j]-m[j])**2, var[j], 5)
                compare('centered_cross', (v[:,:,0]-m[0])*(v[:,:,1]-m[1]), 0., 5)
                mz = target.gaussian_white_mean(); vz = 1/target.mass_diag
                er2 = float(np.sum(mz*mz+vz)); vr2 = float(np.sum(2*vz*vz+4*mz*mz*vz))
                r2 = np.sum(z*z, axis=-1)
                compare('white_radius_squared', r2, er2, 5)
                compare('white_radius_fourth', r2*r2, er2*er2+vr2, 5)
            gate['reference_moments'] = all(c['passed'] for c in comparisons)
            result.update({'gates': gate, 'passed': all(gate.values()), 'diagnostics': diag,
                           'comparisons': comparisons, **old.diag_summary(diag)})
        save(out/f'2d_q{q:g}_summary.json', result); results.append(result)
        print(f'2D q={q}: passed={result["passed"]}', flush=True)
    return results


def real(out):
    results = []; original = load_original()
    for fid, (ds, grid, first_ref, _, meta) in enumerate(prepare(), 603):
        h, w = grid.shape
        image = np.asarray(Image.open(ROOT/meta['input_path']), dtype=float)
        if image.ndim == 3:
            image = image[...,0]
        tile = image[h:2*h,:w]/255
        ref = grid.gaussian_reference(tile, first_ref.nugget_ratio)
        previous = json.loads((PREV/f'{ds}_input.json').read_text())
        assert hashlib.sha256(tile.tobytes()).hexdigest() == previous['tile_normalized_float64_sha256']
        assert meta['input_sha256'] == previous['input_sha256']
        assert abs(ref.mean-previous['tile_gls_mean']) < 1e-14 and abs(ref.signal_variance-previous['tile_signal_variance']) < 1e-14
        p = meta['r_parameters']
        sp = original['SharedParams'](float(p['beta1']), float(p['beta2']), float(p['nugget_nu']), h, w, 'same fixed tile2', 0., 'none', 0, 0.)
        error = float(np.max(abs(original['reconstruct_tile'](tile, sp).pred_mean-ref.prediction)))
        assert error < 1e-10
        metadata = {**previous, 'q2_vs_original_python_max_error': error}
        save(out/f'{ds}_input.json', metadata)
        target = grid.posterior(ref, 1.5); obs = Observer(target, 4000)
        print(f'Start {ds} tile 2, n={grid.n}; proposal={obs.proposal.record()}', flush=True)
        sampler, initial, final, traces = sample(target, config(4000), fid, obs,
            progress=lambda i,n,s: print(f'{ds}: {i}/{n}, {s:.1f}s', flush=True))
        assert sampler['proposal'] == obs.proposal.record()
        post_started = time.perf_counter(); raw, cv = obs.complete_blocks()
        np.savez_compressed(out/f'{ds}_chains.npz', trace=obs.traces[:,:obs.count], trace_names=np.asarray(obs.names),
            spectral_indices=obs.indices, extra_trace=obs.extra[:,:obs.count], raw_spectral_blocks=raw,
            controlled_spectral_blocks=cv, initial_z=initial, final_z=final, **traces)
        result = {'input': metadata, 'sampler': sampler, 'passed': False}
        if sampler['completed']:
            diag = obs.diagnostics(); gate = gates(diag, sampler, 1.5)
            cvs = pixel_block_summary(grid, cv); raws = pixel_block_summary(grid, raw)
            mean = (ref.mean+cvs['mean'])*255; mcse = cvs['mcse']*255; q2 = ref.prediction*255; delta = mean-q2
            gate['pixel_mcse'] = np.quantile(mcse, .99) <= .1 and mcse.max() <= .5
            result.update({'passed': all(gate.values()), 'gates': gate, **old.diag_summary(diag),
                'mcse_p99': float(np.quantile(mcse, .99)), 'mcse_max': float(mcse.max()),
                'mean_absolute_q_change': float(np.mean(abs(delta))), 'max_absolute_q_change': float(np.max(abs(delta))),
                'fraction_q_change_exceeds_3_mcse': float(np.mean(abs(delta) > 3*mcse)), 'diagnostics': diag})
            np.savez_compressed(out/f'{ds}_estimates.npz', raw=tile*255, q2=q2, q15_controlled=mean,
                q15_raw=(ref.mean+raws['mean'])*255, q15_mcse=mcse, raw_mcse=raws['mcse']*255,
                controlled_chain_means=(ref.mean+cvs['chain_means'])*255,
                raw_chain_means=(ref.mean+raws['chain_means'])*255, radius_traces=obs.traces[:,:,0])
            print(f'{ds}: passed={result["passed"]}, max Rhat={result["max_rhat"]:.6f}, '
                  f'min ESS={result["min_bulk_ess"]:.1f}/{result["min_tail_ess"]:.1f}', flush=True)
        result['postprocessing_runtime_sec'] = time.perf_counter()-post_started
        save(out/f'{ds}_summary.json', result); results.append(result)
        del obs, raw, cv
    return results


def main():
    out = ROOT/'results/fast_qep_joint_mh_20261015'
    if out.exists():
        raise RuntimeError('Refusing overwrite or seed retry.')
    out.mkdir()
    before = json.loads((PREV/'manifest.json').read_text())
    previous = dict(before['previous_result_sha256'])
    previous.update({str(p.relative_to(ROOT)): digest(p) for p in PREV.iterdir() if p.is_file()})
    assert all(digest(ROOT/p) == h for p,h in previous.items())
    assert all(digest(ROOT/p) == h for p,h in before['source_sha256'].items())
    sources = [HERE/'PROTOCOL.md', HERE/'joint_mh.py', Path(__file__), PREV_RUN,
               ROOT/'py_core/fast_qep_prototype.py', ROOT/'py_core/paper_fast_gp.py', ROOT/'py_core/dim_2_lattice.py',
               ROOT/'audits/fast_qep_radial_diagnosis_20261014/independence_mh.py',
               ROOT/'experiments/real_data/fast_qep_real_pilot_20261014/stream_hmc.py',
               ROOT/'experiments/real_data/fast_qep_real_pilot_20261014/run_pilot.py',
               ROOT/'audits/fast_qep_prototype_20261014/diagnostics.py',
               ROOT/'audits/fast_qep_feasibility_20261014/verify_math.py',
               ROOT/'audits/fast_qep_paper_alignment_20261014/MODEL_CONTRACT.md']
    save(out/'manifest.json', {'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'previous_result_sha256': previous, 'python': sys.version, 'numpy': np.__version__, 'threads': 1,
        'small_config': asdict(config(20000)), 'real_config': asdict(config(4000)),
        'date_UTC': '2026-10-15', 'date_America_Phoenix': '2026-10-14'})
    completion = {'full_image_or_segmentation_run': False, 'new_model_or_q_search': False, 'real_image_run': False}
    try:
        save(out/'deterministic_checks.json', checks()); print('Deterministic checks PASS.', flush=True)
        ss = small(out); completion['small_pass'] = {str(r['q']): r['passed'] for r in ss}
        if all(r['passed'] for r in ss):
            completion['real_image_run'] = True; rr = real(out)
            completion['real_pass'] = {r['input']['dataset']: r['passed'] for r in rr}
            completion['status'] = 'PASS_TWO_LOCAL_DIAGNOSTICS' if all(r['passed'] for r in rr) else 'STOP_REAL_NUMERICAL_GATE'
        else:
            completion['status'] = 'STOP_SMALL_REFERENCE_GATE'
    except Exception as exc:
        completion.update({'status': 'STOP_IMPLEMENTATION_OR_RUNTIME_ERROR', 'exception': repr(exc)})
        raise
    finally:
        unchanged = all(digest(ROOT/p) == h for p,h in previous.items())
        save(out/'integrity.json', {'old_results_unchanged': unchanged, 'checked_files': len(previous)})
        completion['old_results_unchanged'] = unchanged; save(out/'completion.json', completion)
        assert unchanged


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
