"""Check by default. --run starts the separately approved fresh validation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import zipfile
from contextlib import redirect_stdout, redirect_stderr

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from threadpoolctl import threadpool_limits
from diagnostics import summarize

HERE = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for stream in self.streams: stream.write(s); stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams: stream.flush()


def verify(bundle):
    for root, name in ((bundle, 'hyperfit_manifest.json'), (HERE, 'validation_manifest.json'),
                       (HERE, 'input_manifest.json'), (HERE, 'protocol_freeze.json')):
        record = json.loads((root/name).read_text())
        for relative, h in record['files'].items():
            if digest(root/relative) != h:
                raise RuntimeError('Source checksum mismatch: '+str(root/relative))
    sys.path.insert(0, str(bundle/'payload/experiments/real_data/fast_qep_hyperfit_20261015'))
    import pilot
    weights = json.loads((HERE/'candidate_weights.json').read_text())
    work = pilot.prepare_inputs()
    for item, fixed in zip(work, weights['targets']):
        if item['dataset'] != fixed['dataset'] or len(fixed['weights']) != 5:
            raise RuntimeError('Dataset/weight ordering mismatch')
        theta = np.asarray(fixed['theta'])
        original = json.loads((HERE/'inputs'/item['dataset']/'refitted_GP/summary.json').read_text())
        if not np.array_equal(theta, original['theta']):
            raise RuntimeError('GP theta changed')
        ll, score = pilot.gaussian_evidence(item['tile'], theta)
        if not np.allclose(score, fixed['gp_analytic_score'], rtol=1e-7, atol=1e-6):
            raise RuntimeError('Analytic GP score mismatch')
        if not np.all((np.array(fixed['weights']) >= 0) & (np.array(fixed['weights']) <= 1)):
            raise RuntimeError('Weights outside frozen domain')
    return pilot, work, weights


def weighted_review(folder, fixed, record):
    with np.load(folder/'score_traces.npz') as f:
        raw, old = f['raw'], f['controlled']
        names = f['names'].tolist()
    b = np.asarray(fixed['weights'])
    selected = raw-b*(raw-old)
    selected[:, :, b == 1] = old[:, :, b == 1]
    selected[:, :, b == 0] = raw[:, :, b == 0]
    rows = []
    for j, name in enumerate(names):
        summaries = {label: summarize(x[:, :, j]) for label, x in
                     [('raw', raw), ('fixed', old), ('weighted', selected)]}
        w, scale, gp = summaries['weighted'], fixed['score_scale'][j], fixed['gp_analytic_score'][j]
        chain_difference = selected[:, :, j].mean(axis=1)-gp
        direction_resolved = abs(w['mean']-gp) > 3*w['mcse'] and abs((w['mean']-gp)*scale) > 1e-8
        consistent_sign = bool(np.all(chain_difference > 0) or np.all(chain_difference < 0))
        rows.append({'name': name, 'weight': float(b[j]), 'summaries': summaries,
                     'scaled_qep_score': w['mean']*scale, 'scaled_mcse': w['mcse']*scale,
                     'gp_analytic_score': gp, 'chain_difference': chain_difference.tolist(),
                     'weighted_to_raw_mcse': w['mcse']/summaries['raw']['mcse'],
                     'weighted_to_fixed_mcse': w['mcse']/summaries['fixed']['mcse'],
                     'descriptive_direction_resolved': bool(direction_resolved),
                     'all_four_chains_same_sign': consistent_sign,
                     'candidate_direction': bool(direction_resolved and consistent_sign)})
    convergence = all(r['summaries']['weighted']['rhat_rank_folded_split'] < 1.01 and
                      r['summaries']['weighted']['ess_bulk'] > 400 and
                      r['summaries']['weighted']['ess_tail'] > 400 for r in rows)
    precision = all(r['weighted_to_raw_mcse'] <= 1.2 for r in rows) and all(
        r['weighted_to_fixed_mcse'] <= .8 for r in rows if r['name'].startswith('log_beta'))
    result = {'dataset': fixed['dataset'], 'passed_original_gates': record['passed'],
              'weighted_convergence_pass': convergence, 'precision_repair_supported': precision,
              'weights_updated_using_fresh_data': False, 'scores': rows,
              'qep_hyperparameters_optimized': False, 'segmentation_run': False}
    np.savez_compressed(folder/'weighted_scores.npz', scores=selected, weights=b, names=names)
    dump(folder/'weighted_summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=HERE.parent/'FastQEP_hyperfit_pilot_20261015')
    p.add_argument('--run', action='store_true')
    p.add_argument('--out', type=Path, default=HERE/'fresh_validation')
    args = p.parse_args()
    with threadpool_limits(limits=1):
        pilot, work, weights = verify(args.bundle.resolve())
        if not args.run:
            print(json.dumps({'status': 'READY_NOT_RUN', 'source_hashes_verified': True,
                              'R_anchors_pass': True, 'GP_theta_and_scores_match': True,
                              'fresh_sampling_started': False, 'weights': [
                                  {'dataset': t['dataset'], 'weights': t['weights']} for t in weights['targets']]}, indent=2))
            return
        out = args.out.resolve()
        if out.exists(): raise RuntimeError('Refusing overwrite/retry: '+str(out))
        out.mkdir(parents=True)
        outcome = {'status': 'STARTED', 'targets': [], 'new_qep_fit': False, 'segmentation_run': False}
        start, error = time.perf_counter(), None
        with (out/'run.log').open('w') as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                dump(out/'environment.json', {'python': sys.version, 'numpy': np.__version__,
                     'threads': 1, 'source_bundle': str(args.bundle.resolve()),
                     'candidate_weights_sha256': digest(HERE/'candidate_weights.json'),
                     'validation_manifest_sha256': digest(HERE/'validation_manifest.json')})
                for item, fixed in zip(work, weights['targets']):
                    remaining = 1800-(time.perf_counter()-start)
                    if remaining <= 0: raise RuntimeError('VALIDATION_TIME_BUDGET')
                    label = 'frozen_weight_validation'
                    record = pilot.run_target(item, label, np.asarray(fixed['theta']), out,
                                              fixed['new_fixture_id'], remaining)
                    if not record['passed']: raise RuntimeError('ORIGINAL_NUMERICAL_GATE')
                    r = weighted_review(out/item['dataset']/label, fixed, record)
                    outcome['targets'].append(r)
                    dump(out/'summary.json', outcome)
                    if not r['weighted_convergence_pass']: raise RuntimeError('WEIGHTED_SCORE_GATE')
                outcome['status'] = 'COMPLETED'
            except Exception as exc:
                error = exc
                outcome.update(status='FAILED', error=repr(exc))
                traceback.print_exc()
            finally:
                outcome['runtime_seconds'] = time.perf_counter()-start
                dump(out/'summary.json', outcome)
                dump(out/'manifest.json', {'files': {str(f.relative_to(out)): digest(f)
                     for f in sorted(out.rglob('*')) if f.is_file() and f.name not in ('manifest.json', 'run.log')}})
        with zipfile.ZipFile(out.parent/(out.name+'_review.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file(): z.write(f, out.name+'/'+str(f.relative_to(out)))
        print(outcome['status'])
        if error: raise SystemExit(1)


if __name__ == '__main__':
    main()
