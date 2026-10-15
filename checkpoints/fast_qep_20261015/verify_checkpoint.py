"""Read-only integrity checks; no imports of scientific runners or sampling."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root, records):
    for relative, expected in records.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(HERE):
            raise ValueError(f'Path outside checkpoint: {relative}')
        expected_hash = expected['sha256'] if isinstance(expected, dict) else expected
        if not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f'Missing file or checksum mismatch: {path}')
        if isinstance(expected, dict) and path.stat().st_size != expected['bytes']:
            raise ValueError(f'Size mismatch: {path}')
    return len(records)


def main():
    inventory = json.loads((HERE / 'CHECKPOINT_MANIFEST.json').read_text())
    verified = check(HERE, inventory['files'])
    bundle = HERE / 'FastQEP_hyperfit_pilot_20261015'
    offline = HERE / 'fastqep_score_precision_diagnostic_20261015'
    manifests = [
        (bundle, bundle / 'hyperfit_manifest.json'),
        (bundle / 'payload', bundle / 'payload/experiments/real_data/fast_qep_e2e_20261015/bundle_manifest.json'),
        (offline, offline / 'input_manifest.json'),
        (offline, offline / 'protocol_freeze.json'),
        (offline, offline / 'validation_manifest.json'),
        (offline, offline / 'delivery_manifest.json'),
        (offline / 'fresh_validation', offline / 'fresh_validation/manifest.json'),
    ]
    for label in ('check', 'pilot'):
        folder = bundle / f'payload/results/fast_qep_hyperfit_{label}_20261015'
        manifests.append((folder, folder / 'completion.json'))
    frozen_checks = {str(path.relative_to(HERE)): check(root, json.loads(path.read_text())['files'])
                     for root, path in manifests}
    feasibility = HERE / 'fastqep_hyperfit_review_20261015'
    sums = {}
    for line in (feasibility / 'SHA256SUMS.txt').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        sums[relative] = expected
    frozen_checks['feasibility_SHA256SUMS'] = check(feasibility, sums)
    mac = HERE / 'fastqep_e2e_review_20261015'
    original = json.loads((mac / 'completion.json').read_text())['artifacts']
    supplied = {p: h for p, h in original.items() if (mac / p).is_file()}
    missing = {p: h for p, h in original.items() if not (mac / p).is_file()}
    availability = json.loads((HERE / 'mac_artifact_availability.json').read_text())
    if missing != availability['not_supplied']:
        raise ValueError('Mac artifact availability record differs from archive')
    mac_verified = check(mac, supplied)
    fresh = json.loads((HERE / 'fastqep_fresh_validation_review_20261015/analysis.json').read_text())
    print(json.dumps({
        'integrity': 'PASS', 'verified_checkpoint_files': verified,
        'original_manifest_checks': frozen_checks,
        'mac_completion_entries_verified': mac_verified,
        'mac_artifacts_not_supplied': len(missing),
        'fresh_validation_decision': fresh['decision'],
        'new_sampling_or_model_fitting': False,
    }, indent=2))


if __name__ == '__main__':
    main()
