"""Offline analysis of frozen scores only. No model fitting or sampling."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from diagnostics import summarize

HERE = Path(__file__).resolve().parent
BLOCK = 20
SPLITS = [('forward', [0, 1], [2, 3]), ('reverse', [2, 3], [0, 1])]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def batches(x):
    if x.shape[1] % BLOCK:
        raise ValueError('Incomplete batch')
    return x.reshape(x.shape[0], -1, BLOCK).mean(axis=-1)


def covariance(g, c):
    gb, cb = batches(g), batches(c)
    gc, cc = gb-gb.mean(axis=1, keepdims=True), cb-cb.mean(axis=1, keepdims=True)
    denominator = gb.shape[0]*(gb.shape[1]-1)
    vg, vc, cov = [float(x/denominator) for x in
                   (np.sum(gc*gc), np.sum(cc*cc), np.sum(gc*cc))]
    correlation = cov/np.sqrt(vg*vc) if vg > 0 and vc > 0 else None
    return {'batch_variance_raw': vg, 'batch_variance_control': vc,
            'batch_covariance_raw_control': cov, 'batch_correlation': correlation,
            'batch_variance_fixed': vg+vc-2*cov,
            'batch_count_per_chain': gb.shape[1]}


def fit_weight(g, c):
    stats = covariance(g, c)
    vc = stats['batch_variance_control']
    unconstrained = stats['batch_covariance_raw_control']/vc if vc else 0.
    b = float(np.clip(unconstrained, 0., 1.))
    vg, cov = stats['batch_variance_raw'], stats['batch_covariance_raw_control']
    vb = vg+b*b*vc-2*b*cov
    # The empirical constrained quadratic minimum must beat both endpoints.
    tolerance = 1e-10*max(vg, vc, 1.)
    if vb > min(vg, stats['batch_variance_fixed'])+tolerance:
        raise ArithmeticError('Weight did not minimize the calibration quadratic')
    return {**stats, 'unbounded_weight': float(unconstrained), 'weight': b,
            'zero_control_variance': vc == 0, 'batch_variance_weighted': vb}


def ratio(a, b):
    return float(a/b) if b else None


def summary(x, gp, scale):
    r = summarize(x)
    r['chain_means'] = x.mean(axis=1).tolist()
    r['scaled_mean'] = r['mean']*scale
    r['scaled_mcse'] = r['mcse']*scale
    r['qep_minus_gp'] = r['mean']-gp
    r['scaled_qep_minus_gp'] = r['qep_minus_gp']*scale
    r['signed_difference_over_mcse'] = ratio(r['qep_minus_gp'], r['mcse'])
    r['exploratory_old_screen_crossed'] = bool(abs(r['qep_minus_gp']) > 3*r['mcse']
                                             and abs(r['scaled_qep_minus_gp']) > 1e-8)
    r['diagnostics_pass'] = bool(r['rhat_rank_folded_split'] < 1.01 and
                                 r['ess_bulk'] > 400 and r['ess_tail'] > 400)
    return r


def main(out):
    out = Path(out).resolve()
    if out.exists():
        raise RuntimeError('Refusing to overwrite an existing analysis: '+str(out))
    for manifest in ('protocol_freeze.json', 'input_manifest.json'):
        m = json.loads((HERE/manifest).read_text())
        for name, h in m['files'].items():
            if digest(HERE/name) != h:
                raise ValueError('Input/protocol checksum mismatch: '+name)
    out.mkdir(parents=True)
    all_rows, targets = [], []
    weighted_traces = {}
    for dataset in ('nuclei', 'whole_cell'):
        for point in ('archived_R', 'refitted_GP'):
            folder = HERE/'inputs'/dataset/point
            with np.load(folder/'score_arrays.npz') as f:
                raw, fixed, names = f['raw'], f['controlled'], f['names'].tolist()
            original = json.loads((folder/'summary.json').read_text())
            if raw.shape != (4, 400, 5) or fixed.shape != raw.shape:
                raise ValueError('Wrong shape')
            if not np.isfinite(raw).all() or not np.isfinite(fixed).all():
                raise ValueError('Nonfinite input')
            control = raw-fixed
            rows = []
            for j, name in enumerate(names):
                prior = original['scores'][j]
                if name != prior['name']:
                    raise ValueError('Parameter ordering mismatch')
                # Independent saved summaries verify trace interpretation.
                for field, arr in [('raw', raw), ('controlled', fixed)]:
                    if not np.isclose(arr[:, :, j].mean(), prior[field]['mean'], rtol=1e-12, atol=1e-10):
                        raise ValueError('Saved score summary disagrees with trace')
                full = covariance(raw[:, :, j], control[:, :, j])
                row = {'dataset': dataset, 'parameter_source': point, 'name': name,
                       'gp_analytic_score': prior['gp_analytic_score'], 'score_scale': prior['score_scale'],
                       'uninformative_zero': bool(np.all(raw[:, :, j] == 0) and np.all(fixed[:, :, j] == 0)),
                       'all_chains_descriptive_variance_decomposition': full, 'splits': []}
                for split, train, evaluation in SPLITS:
                    if set(train) & set(evaluation):
                        raise ValueError('Calibration/evaluation overlap')
                    fit = fit_weight(raw[train, :, j], control[train, :, j])
                    b = fit['weight']
                    g, c, old = raw[evaluation, :, j], control[evaluation, :, j], fixed[evaluation, :, j]
                    selected = old.copy() if b == 1 else g.copy() if b == 0 else g-b*c
                    weighted_traces[f'{dataset}__{point}__{name}__{split}'] = selected
                    stats = {key: summary(value, prior['gp_analytic_score'], prior['score_scale'])
                             for key, value in [('raw', g), ('fixed', old), ('weighted', selected)]}
                    sf = {'split': split, 'calibration_chains_zero_based': train,
                          'evaluation_chains_zero_based': evaluation, 'calibration': fit,
                          'evaluation_variance': covariance(g, c), 'estimates': stats,
                          'weighted_to_fixed_mcse': ratio(stats['weighted']['mcse'], stats['fixed']['mcse']),
                          'weighted_to_raw_mcse': ratio(stats['weighted']['mcse'], stats['raw']['mcse'])}
                    sf['technical_screen_pass'] = bool(not row['uninformative_zero'] and
                        sf['weighted_to_fixed_mcse'] is not None and sf['weighted_to_fixed_mcse'] <= .8 and
                        sf['weighted_to_raw_mcse'] is not None and sf['weighted_to_raw_mcse'] <= 1.2 and
                        stats['weighted']['diagnostics_pass'])
                    row['splits'].append(sf)
                row['technical_candidate_both_splits'] = all(s['technical_screen_pass'] for s in row['splits'])
                weighted = [s['estimates']['weighted'] for s in row['splits']]
                row['exploratory_signal_both_splits_same_sign'] = bool(not row['uninformative_zero'] and
                    all(s['exploratory_old_screen_crossed'] and s['diagnostics_pass'] for s in weighted) and
                    weighted[0]['qep_minus_gp']*weighted[1]['qep_minus_gp'] > 0)
                rows.append(row)
                all_rows.append(row)
            targets.append({'dataset': dataset, 'parameter_source': point,
                            'technical_candidates': [r['name'] for r in rows if r['technical_candidate_both_splits']],
                            'exploratory_signals': [r['name'] for r in rows if r['exploratory_signal_both_splits_same_sign']]})
    result = {'type': 'POST_HOC_OFFLINE_DIAGNOSTIC', 'new_sampling': False, 'qep_fit': False,
              'segmentation_run': False, 'original_pilot_decisions_changed': False,
              'independent_scientific_validation': False, 'score_batch_size': BLOCK,
              'coefficient_selection_uses_evaluation_samples': False,
              'source_hashes': {name: digest(HERE/name) for name in
                                ('PROTOCOL.txt', 'protocol_freeze.json', 'input_manifest.json', 'analyze.py', 'diagnostics.py')},
              'target_summary': targets, 'rows': all_rows,
              'limitations': ['Both reciprocal splits use the same previously inspected four chains.',
                              'Weights minimize a finite calibration batch criterion, not proven asymptotic MSE.',
                              'No pooling of reciprocal split means or MCSEs.',
                              'MCSE and between-chain diagnostics from two evaluation chains remain noisy.',
                              'Any crossed score screen is exploratory and needs fresh validation.']}
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(out/'weighted_evaluation_traces.npz', **weighted_traces)
    print(json.dumps(targets, indent=2))
    for r in all_rows:
        if r['uninformative_zero']:
            continue
        print(r['dataset'], r['parameter_source'], r['name'],
              'weights', [round(s['calibration']['weight'], 6) for s in r['splits']],
              'MCSE/fixed', [round(s['weighted_to_fixed_mcse'], 5) for s in r['splits']],
              'MCSE/raw', [round(s['weighted_to_raw_mcse'], 5) for s in r['splits']],
              'candidate', r['technical_candidate_both_splits'],
              'z_exploratory', [round(s['estimates']['weighted']['signed_difference_over_mcse'], 3) for s in r['splits']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=HERE/'results')
    main(p.parse_args().out)
