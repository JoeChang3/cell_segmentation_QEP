"""Engineering checks only: checkpoint identity, recovery, and failure blocking."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Verification output already exists')
    args.out.mkdir(parents=True)
    path = Path(__file__).with_name('run.py')
    spec = importlib.util.spec_from_file_location('e2e_runner_under_test', path)
    runner = importlib.util.module_from_spec(spec); sys.modules[spec.name] = runner; spec.loader.exec_module(runner)
    joint, metrics = runner.load_modules(); np = joint.np
    from py_core.fast_qep_prototype import KroneckerGrid
    from threadpoolctl import threadpool_limits
    grid = KroneckerGrid.matern52((4,5), 2.2, 1.7)
    raw = 50 + .4*np.sin(np.arange(28).reshape(4,7))
    work = dict(ds='nuclei', grid=grid, raw=raw, nu=.3, tiles=[(1,0,0)])
    checks = {}
    with threadpool_limits(limits=1):
        values = runner.reconstruct(args, 'engineering_fixture_only', work, joint)
        checks['real_sampler_to_checkpoint_and_assembly'] = True
        checks['original_uncovered_columns_remain_zero'] = bool(np.all(values['q15'][:,5:]==0))
        checks['pooled_mean_matches_four_chain_mean'] = bool(np.allclose(values['q15'], values['chain_means'].mean(axis=0), atol=1e-12, rtol=0))
        old_sample = joint.sample
        def forbidden(*a, **kw): raise AssertionError('Unexpected resampling of saved tile')
        joint.sample = forbidden
        resumed = runner.reconstruct(args, 'engineering_fixture_only', work, joint)
        checks['resume_without_resampling_identical'] = all(np.array_equal(values[k], resumed[k]) for k in values)
        try:
            runner.reconstruct(args, 'changed_signature', work, joint)
        except RuntimeError as exc:
            checks['changed_identity_blocks_resume'] = 'Cannot automatically retry' in str(exc)
        checkpoint = args.out/'tiles/nuclei/tile01/checkpoint.json'
        original = checkpoint.read_text(); obj = json.loads(original); obj['status'] = 'FAILED'
        checkpoint.write_text(json.dumps(obj))
        try:
            runner.reconstruct(args, 'engineering_fixture_only', work, joint)
        except RuntimeError as exc:
            checks['failed_tile_cannot_be_retried'] = 'Cannot automatically retry' in str(exc)
        finally:
            checkpoint.write_text(original)
        artifact = args.out/'tiles/nuclei/tile01/estimates.npz'
        content = artifact.read_bytes(); artifact.write_bytes(content+b'changed')
        try:
            runner.reconstruct(args, 'engineering_fixture_only', work, joint)
        except RuntimeError as exc:
            checks['tampered_tile_artifact_blocks_resume'] = 'Changed/missing saved artifact' in str(exc)
        finally:
            artifact.write_bytes(content); joint.sample = old_sample
    # Nonconsecutive label IDs and label-renumbering invariance used by R anchors.
    aa = np.array([[0,4,4],[11,11,0]], dtype=np.int32)
    bb = np.array([[0,9,9],[3,3,0]], dtype=np.int32)
    cc = np.array([[0,9,3],[3,3,0]], dtype=np.int32)
    checks['partition_check_ignores_only_label_names'] = bool(np.array_equal(metrics.canonical(aa), metrics.canonical(bb))
        and not np.array_equal(metrics.canonical(aa), metrics.canonical(cc)))
    runner.dump(args.out/'runner_checks.json', dict(passed=all(checks.values()), checks=checks,
        synthetic_fixture_only=True, real_R_executed=False, real_image_QEP_sampled=False))
    if not all(checks.values()): raise RuntimeError(checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
