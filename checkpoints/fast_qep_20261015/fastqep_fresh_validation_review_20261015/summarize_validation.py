"""Read-only audit of the frozen-weight fresh-sample validation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(root, calibration, destination):
    root, calibration, destination = map(lambda p: Path(p).resolve(), (root, calibration, destination))
    outcome = json.loads((root/'summary.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    for relative, h in manifest['files'].items():
        if sha(root/relative) != h:
            raise ValueError('Frozen result hash mismatch: '+relative)
    for source_manifest in ('validation_manifest.json', 'input_manifest.json', 'protocol_freeze.json'):
        for relative, h in json.loads((calibration/source_manifest).read_text())['files'].items():
            if sha(calibration/relative) != h:
                raise ValueError('Frozen calibration/source hash mismatch: '+relative)
    weights = json.loads((calibration/'candidate_weights.json').read_text())
    env = json.loads((root/'environment.json').read_text())
    if env['candidate_weights_sha256'] != sha(calibration/'candidate_weights.json'):
        raise ValueError('Weights changed since run start')
    review = {'run_status': outcome['status'], 'runtime_seconds': outcome['runtime_seconds'],
              'validated_result_files': len(manifest['files']), 'new_sampling': True,
              'independent_microscopy_images': False, 'qep_optimization_run': False,
              'segmentation_run': False, 'weights_changed_after_new_sampling': False,
              'targets': [], 'source_result_manifest_sha256': sha(root/'manifest.json')}
    for r in outcome['targets']:
        fixed = next(t for t in weights['targets'] if t['dataset'] == r['dataset'])
        folder = root/r['dataset']/'frozen_weight_validation'
        old = json.loads((folder/'summary.json').read_text())
        if old['theta'] != fixed['theta']:
            raise ValueError('Model parameters changed')
        seeds = [[fixed['base_seed'], fixed['new_fixture_id'], c] for c in range(4)]
        if old['sampler']['seed_components'] != seeds:
            raise ValueError('Unexpected seeds')
        with np.load(folder/'score_traces.npz') as f:
            raw, original_controlled = f['raw'], f['controlled']
        with np.load(folder/'weighted_scores.npz') as f:
            weighted, saved_b = f['scores'], f['weights']
        if raw.shape != (4, 400, 5) or not np.array_equal(saved_b, fixed['weights']):
            raise ValueError('Score shape or weight mismatch')
        direct = raw-(raw-original_controlled)*saved_b
        error = float(np.max(abs(weighted-direct)))
        if error > 1e-8:
            raise ValueError('Saved weighted traces do not match frozen linear correction')
        for j, s in enumerate(r['scores']):
            if not np.isclose(weighted[:, :, j].mean(), s['summaries']['weighted']['mean'], rtol=1e-12, atol=1e-10):
                raise ValueError('Saved weighted mean mismatch')
        numerical_pass = old['passed'] and r['weighted_convergence_pass']
        t = {'dataset': r['dataset'], 'numerical_convergence_pass': numerical_pass,
             'precision_repair_supported': r['precision_repair_supported'],
             'fresh_seed_components': seeds, 'weights': fixed['weights'],
             'theta': fixed['theta'], 'sampler_seconds': old['sampler']['runtime_sec'],
             'pixel_mcse_p99': old['pixel_mcse_p99'], 'pixel_mcse_max': old['pixel_mcse_max'],
             'maximum_latent_rhat': old['maximum_rhat'], 'weighted_trace_formula_max_error': error,
             'retained_per_chain': old['sampler']['retained_per_chain'],
             'divergences': old['sampler']['divergences_abs_delta_H_gt_100'],
             'nonfinite_proposals': old['sampler']['nonfinite_retained_hmc_proposals'],
             'global_failures': old['sampler']['global_numerical_failures'],
             'hmc_acceptance': old['sampler']['acceptance_rate'],
             'global_acceptance': old['sampler']['global_acceptance_rate'],
             'scores': [], 'candidate_directions': []}
        for s in r['scores']:
            w = s['summaries']['weighted']
            ratio = (w['mean']-s['gp_analytic_score'])/w['mcse']
            row = {k: s[k] for k in ('name', 'weight', 'scaled_qep_score', 'scaled_mcse',
                    'gp_analytic_score', 'chain_difference', 'weighted_to_raw_mcse',
                    'weighted_to_fixed_mcse', 'descriptive_direction_resolved',
                    'all_four_chains_same_sign', 'candidate_direction')}
            row.update(mean=w['mean'], mcse=w['mcse'], signed_difference_over_mcse=ratio,
                       rhat=w['rhat_rank_folded_split'], ess_bulk=w['ess_bulk'], ess_tail=w['ess_tail'],
                       raw_mcse=s['summaries']['raw']['mcse'], fixed_mcse=s['summaries']['fixed']['mcse'])
            t['scores'].append(row)
            if s['candidate_direction']:
                t['candidate_directions'].append(s['name'])
        t['maximum_abs_difference_over_mcse'] = max(abs(s['signed_difference_over_mcse']) for s in t['scores'])
        t['maximum_weighted_rhat'] = max(s['rhat'] for s in t['scores'])
        t['minimum_weighted_bulk_ess'] = min(s['ess_bulk'] for s in t['scores'])
        t['minimum_weighted_tail_ess'] = min(s['ess_tail'] for s in t['scores'])
        t['kernel_mcse_reduction_pct'] = {s['name']: 100*(1-s['weighted_to_fixed_mcse'])
                                         for s in t['scores'] if s['name'].startswith('log_beta')}
        review['targets'].append(t)
    complete = outcome['status'] == 'COMPLETED' and len(review['targets']) == 2
    review['all_numerical_gates_pass'] = complete and all(t['numerical_convergence_pass'] for t in review['targets'])
    review['precision_repair_supported_both_datasets'] = complete and all(t['precision_repair_supported'] for t in review['targets'])
    if review['all_numerical_gates_pass'] and review['precision_repair_supported_both_datasets']:
        review['decision'] = ('BOUNDED_UPDATE_DESIGN_MAY_BE_CONSIDERED' if any(t['candidate_directions'] for t in review['targets'])
                              else 'STOP_AUTOMATIC_EXPANSION_OF_THIS_LOCAL_GRADIENT_TRAINING_ROUTE')
    else:
        review['decision'] = 'VALIDATION_INCONCLUSIVE_OR_PRECISION_REPAIR_NOT_CONFIRMED'
    review['claim_scope'] = ('Two previously used development-image first tiles, one fixed q and one GP-fitted parameter point per image. '
                             'Fresh MCMC randomness is independent of calibration, but the microscopy data are not held out. '
                             'An unresolved direction is not a proof of exact zero gradients or impossibility of every Fast-QEP model.')
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'analysis.json').write_text(json.dumps(review, indent=2, allow_nan=False)+'\n')
    print(json.dumps(review, indent=2, allow_nan=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
