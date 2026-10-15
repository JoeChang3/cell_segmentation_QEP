"""Recompute saved-output reviews in temporary space; never fit or sample."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(a, b, path='root'):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError(f'Key mismatch at {path}')
        for key in a:
            if key != 'source_output':  # Absolute output path changes on relocation.
                compare(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError(f'Length mismatch at {path}')
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'Numerical mismatch at {path}: {a} != {b}')
    elif a != b:
        raise ValueError(f'Value mismatch at {path}: {a!r} != {b!r}')


def read(path):
    return json.loads(path.read_text())


def main():
    # Integrity is checked first; all scientific source files remain untouched.
    load_module(HERE / 'verify_checkpoint.py', 'checkpoint_integrity').main()
    offline = HERE / 'fastqep_score_precision_diagnostic_20261015'
    pilot_review = HERE / 'fastqep_hyperfit_pilot_analysis_20261015'
    fresh_review = HERE / 'fastqep_fresh_validation_review_20261015'
    sys.path.insert(0, str(offline))
    results = []
    with tempfile.TemporaryDirectory(prefix='fastqep_saved_review_') as temporary:
        out = Path(temporary)
        with contextlib.redirect_stdout(io.StringIO()):
            load_module(pilot_review / 'summarize_pilot.py', 'saved_pilot_review').main(
                HERE / 'FastQEP_hyperfit_pilot_20261015/payload/results/fast_qep_hyperfit_pilot_20261015',
                out / 'pilot')
            compare(read(out / 'pilot/analysis.json'), read(pilot_review / 'analysis.json'))
            results.append('GP refit and QEP score pilot review agrees')
            load_module(offline / 'analyze.py', 'saved_offline_review').main(out / 'offline')
            compare(read(out / 'offline/analysis.json'), read(offline / 'results/analysis.json'))
            results.append('Offline control-variate analysis agrees')
            load_module(fresh_review / 'summarize_validation.py', 'saved_fresh_review').main(
                offline / 'fresh_validation', offline, out / 'fresh')
            compare(read(out / 'fresh/analysis.json'), read(fresh_review / 'analysis.json'))
            results.append('Fresh validation weights, seeds, traces and decision agree')
    print(json.dumps({'saved_output_review': 'PASS', 'checks': results,
                      'new_sampling_or_model_fitting': False,
                      'archived_files_modified': False}, indent=2))


if __name__ == '__main__':
    main()
