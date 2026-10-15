"""Read-only review of the frozen pilot outputs. Does not fit or sample."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, dest):
    root, dest = Path(root).resolve(), Path(dest).resolve()
    completion = json.loads((root / 'completion.json').read_text())
    bad = [p for p, h in completion['files'].items() if sha(root / p) != h]
    if bad:
        raise ValueError('Result checksum mismatch: ' + str(bad))
    report = {
        'source_output': str(root),
        'completion': completion['status'],
        'verified_result_files': len(completion['files']),
        'new_segmentation_results': False,
        'qep_hyperparameter_optimization_performed': False,
        'gp_fits': [], 'targets': [],
    }
    if 'error' in completion:
        report['stop_reason'] = completion['error']
    lines = ['FAST-QEP FIRST-TILE HYPERPARAMETER SCORE PILOT',
             '2026-10-15 | Read-only review of the frozen run', '',
             'Scope: one original first tile from each development image;',
             'analytic GP refit and q=1.5 evidence-score estimation only.',
             'No QEP parameter update was accepted. No full-image reconstruction,',
             'thresholding, watershed, AP evaluation or held-out test was run.',
             'The nuisance ECM candidate in each target is logged but not applied.',
             '', 'Run status: ' + report['completion']]
    if 'stop_reason' in report:
        lines += ['Stop reason: ' + report['stop_reason'],
                  'A gate failure is not evidence that QEP is scientifically invalid.']
    for dataset in ('nuclei', 'whole_cell'):
        gp_file = root / dataset / 'gp_refit.json'
        if not gp_file.exists():
            lines += ['', dataset + ': not reached.']
            continue
        gp = json.loads(gp_file.read_text())
        selected = next(f for f in gp['fits'] if f['start'] == gp['selected_start'])
        item = {'dataset': dataset, 'selected_start': gp['selected_start'],
                'stationarity_pass': gp['stationarity_pass'],
                'selected_theta': gp['theta'],
                'selected_parameters': dict(zip(['mean', 'signal_variance', 'noise_ratio',
                                                 'beta_rows', 'beta_cols'],
                                                [gp['theta'][0], *np.exp(gp['theta'][1:]).tolist()])),
                'selected_boundary_hits': selected['boundary_hits'],
                'selected_projected_gradient_per_pixel_max': selected['projected_gradient_per_pixel_max'],
                'fits': [{k: v for k, v in f.items() if k != 'history'} for f in gp['fits']]}
        report['gp_fits'].append(item)
        lines += ['', 'GP CONTROL: ' + dataset,
                  'Selected start: ' + gp['selected_start'],
                  'Stationarity gate: ' + str(gp['stationarity_pass']),
                  'Boundary hits (beta rows, beta cols, nu): ' + str(selected['boundary_hits']),
                  'Parameters: ' + json.dumps(item['selected_parameters']),
                  'Two-start fits (log evidence; higher is better within this GP model):']
        for f in gp['fits']:
            lines += [f"  {f['start']}: {f['log_evidence']:.9f}; success={f['success']}; "
                      f"projected gradient/n={f['projected_gradient_per_pixel_max']:.6g}"]
        for parameter_source in ('archived_R', 'refitted_GP'):
            folder = root / dataset / parameter_source
            summary_path = folder / 'summary.json'
            if not summary_path.exists():
                lines += ['', dataset + '/' + parameter_source + ': not reached.']
                continue
            r = json.loads(summary_path.read_text())
            target = {'dataset': dataset, 'parameter_source': parameter_source,
                      'passed': r['passed'], 'sampler_completed': r['sampler']['completed'],
                      'gates': r.get('gates'), 'sampler_seconds': r['sampler']['runtime_sec'],
                      'retained_per_chain': r['sampler']['retained_per_chain'],
                      'hmc_acceptance': r['sampler']['acceptance_rate'],
                      'global_acceptance': r['sampler']['global_acceptance_rate'],
                      'scores': []}
            lines += ['', 'QEP TARGET: ' + dataset + '/' + parameter_source,
                      f"Gates passed: {r['passed']}; sampler seconds: {target['sampler_seconds']:.2f}"]
            if r.get('scores'):
                score_traces = np.load(folder / 'score_traces.npz')
                for j, s in enumerate(r['scores']):
                    cv = s['controlled']
                    ratio = abs(s['qep_minus_gp']) / cv['mcse'] if cv['mcse'] else None
                    chain_means = score_traces['controlled'][:, :, j].mean(axis=1) * s['score_scale']
                    row = {k: s[k] for k in ('name', 'scaled_qep_score', 'scaled_mcse',
                                            'descriptive_q_difference_resolved',
                                            'descriptive_qep_score_resolved')}
                    row.update(scaled_gp_score=s['gp_analytic_score']*s['score_scale'],
                               scaled_qep_minus_gp=s['qep_minus_gp']*s['score_scale'],
                               abs_q_difference_over_mcse=ratio,
                               rhat=cv['rhat_rank_folded_split'],
                               ess_bulk=cv['ess_bulk'], ess_tail=cv['ess_tail'],
                               scaled_chain_means=chain_means.tolist(),
                               raw_score_mean=s['raw']['mean'], raw_score_mcse=s['raw']['mcse'],
                               controlled_score_mean=cv['mean'], controlled_score_mcse=cv['mcse'],
                               controlled_to_raw_mcse_ratio=(cv['mcse']/s['raw']['mcse']
                                                            if s['raw']['mcse'] else None))
                    target['scores'].append(row)
                    lines += [f"  {s['name']}: QEP={row['scaled_qep_score']:+.8g}, "
                              f"GP={row['scaled_gp_score']:+.8g}, "
                              f"difference={row['scaled_qep_minus_gp']:+.8g}, "
                              f"MCSE={row['scaled_mcse']:.5g}; "
                              f"difference resolved={row['descriptive_q_difference_resolved']}, "
                              f"QEP own score resolved={row['descriptive_qep_score_resolved']}",
                              f"    Rhat={row['rhat']:.6f}, bulk ESS={row['ess_bulk']:.1f}, "
                              f"tail ESS={row['ess_tail']:.1f}; chain means={row['scaled_chain_means']}",
                              f"    Raw MCSE={row['raw_score_mcse']:.6g}; controlled MCSE="
                              f"{row['controlled_score_mcse']:.6g}; ratio={row['controlled_to_raw_mcse_ratio']}"]
                target.update(maximum_latent_rhat=r['maximum_rhat'],
                              maximum_score_rhat=max(s['controlled']['rhat_rank_folded_split'] for s in r['scores']),
                              min_score_bulk_ess=min(s['controlled']['ess_bulk'] for s in r['scores']),
                              min_score_tail_ess=min(s['controlled']['ess_tail'] for s in r['scores']),
                              pixel_mcse_p99=r['pixel_mcse_p99'], pixel_mcse_max=r['pixel_mcse_max'],
                              qep_gp_mean_absolute_difference=r['qep_gp_mean_absolute_difference'],
                              resolved_differences=[s['name'] for s in r['scores'] if s['descriptive_q_difference_resolved']],
                              resolved_own_scores=[s['name'] for s in r['scores'] if s['descriptive_qep_score_resolved']])
                lines += [f"  Mean MCSE (0-255 intensity): p99={r['pixel_mcse_p99']:.6g}, max={r['pixel_mcse_max']:.6g}",
                          f"  QEP/GP mean absolute difference at same theta (0-255): {r['qep_gp_mean_absolute_difference']:.6g}"]
            report['targets'].append(target)
    lines += ['', 'INTERPRETATION RULES',
              'All scores are divided by tile pixel count; the mean score is also multiplied by tile SD.',
              'Resolved means both |QEP-GP| > 3 MCSE and scaled |QEP-GP| > 1e-8, as frozen before the run.',
              'This is a descriptive numerical screen, not a p-value or multiplicity-adjusted test.',
              'Parameter scores have different scales; compare each to its own MCSE, not across coordinates.',
              'Positive log-beta score means a locally preferred increase in inverse length scale,',
              'subject to the declared feasible domain; it does not establish better cell boundaries.',
              'A resolved score at a GP stationary point motivates a bounded QEP learning test.',
              'It does not demonstrate a finite-step evidence gain, segmentation gain, or superiority to GP.',
              'An unresolved score cannot rule out other basins or all possible QEP models.',
              'The predeclared coordinatewise screen is not an equivalence test or a test of every joint direction.',
              'A zero-mean control is unbiased but does not guarantee lower variance at q=1.5.',
              'Inspect controlled/raw MCSE ratios: some controls increased rather than decreased error.',
              'The predeclared estimator and screen were retained; no post-hoc estimator was substituted.',
              'GP log-evidence differences here compare GP fits only; no QEP marginal evidence was estimated.',
              'A zero beta derivative on the archived identity-kernel plateau is zero information.',
              'Existing fixed-parameter Fast-QEP/Fast-GP AP ties remain unchanged.',
              'Any later superiority claim needs a refitted GP control, parameter-source controls,',
              'identical downstream processing and independent images after development.', '']
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (dest / 'REPORT_EN.txt').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
