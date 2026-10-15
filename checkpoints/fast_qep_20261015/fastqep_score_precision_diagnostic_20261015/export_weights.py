"""Freeze candidate weights from ALL OLD scores for a future fresh-sample run."""
from pathlib import Path
import json
import numpy as np
from analyze import HERE, digest, fit_weight


def main():
    destination = HERE/'candidate_weights.json'
    if destination.exists():
        raise RuntimeError('Refusing to overwrite frozen candidate weights')
    result = {'status': 'CANDIDATE_NOT_INDEPENDENTLY_VALIDATED',
              'estimated_from': 'all four OLD pilot chains; never future validation draws',
              'rule': 'centered batch covariance, batch size 20, scalar coefficient clipped to [0,1]',
              'offline_analysis_sha256': digest(HERE/'results/analysis.json'), 'targets': []}
    for dataset, fixture in [('nuclei', 8202), ('whole_cell', 8212)]:
        folder = HERE/'inputs'/dataset/'refitted_GP'
        r = json.loads((folder/'summary.json').read_text())
        with np.load(folder/'score_arrays.npz') as f:
            raw, fixed, names = f['raw'], f['controlled'], f['names'].tolist()
        fits = [fit_weight(raw[:, :, j], raw[:, :, j]-fixed[:, :, j]) for j in range(5)]
        result['targets'].append({'dataset': dataset, 'theta': r['theta'], 'names': names,
                                  'weights': [f['weight'] for f in fits], 'weight_fits': fits,
                                  'gp_analytic_score': r['gp_analytic_score'],
                                  'score_scale': [s['score_scale'] for s in r['scores']],
                                  'base_seed': 20261015, 'new_fixture_id': fixture,
                                  'calibration_source_summary_sha256': digest(folder/'summary.json'),
                                  'calibration_score_arrays_sha256': digest(folder/'score_arrays.npz')})
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps([{'dataset':t['dataset'],'weights':t['weights']} for t in result['targets']],indent=2))


if __name__ == '__main__':
    main()
