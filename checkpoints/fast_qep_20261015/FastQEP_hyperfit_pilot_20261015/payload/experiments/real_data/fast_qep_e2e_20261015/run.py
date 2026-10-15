"""Frozen Fast-QEP substitution; real R preflight precedes all image sampling."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUT = ROOT/'results/fast_qep_e2e_20261015'
REFERENCE_COMMIT = '44714c2e0be958fe796a8fd4bdbc220dae3c23dd'
R_VERSIONS = {'R': '4.5.1', 'EBImage': '4.52.0', 'RobustGaSP': '0.6.8',
              'magick': '2.9.1', 'pracma': '2.4.6'}
PACKAGES = ['numpy', 'scipy', 'pandas', 'Pillow', 'threadpoolctl']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def table(path, records):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for row in records for k in row)))
        writer.writeheader(); writer.writerows(records)


def artifacts(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p not in {folder/'checkpoint.json', folder/'preflight_complete.json'}}


def verify_artifacts(folder, hashes):
    for relative, expected in hashes.items():
        if not (folder/relative).is_file() or digest(folder/relative) != expected:
            raise RuntimeError(f'Changed/missing saved artifact: {folder/relative}')


def checked(cmd, **kwargs):
    return subprocess.check_output(list(map(str, cmd)), text=True, stderr=subprocess.STDOUT,
                                   timeout=120, **kwargs).strip()


def environment(args, require_r=True):
    manifest = json.loads((HERE/'bundle_manifest.json').read_text())
    errors = []
    for relative, expected in manifest['files'].items():
        if not (ROOT/relative).is_file() or digest(ROOT/relative) != expected:
            errors.append(f'Frozen source/input differs: {relative}')
    if sys.version_info < (3, 10):
        errors.append('Python >=3.10 required')
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f'Missing Python package: {package}')
    record = dict(python=platform.python_version(), executable=sys.executable,
                  platform=platform.platform(), packages=packages,
                  manifest_sha256=digest(HERE/'bundle_manifest.json'))
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'; os.environ[key] = '1'
    local_r = Path.home()/'Library/R/arm64/4.5/library'
    if not env.get('R_LIBS_USER') and local_r.is_dir():
        env['R_LIBS_USER'] = str(local_r)
    if require_r:
        record['reference'] = str(args.reference)
        try:
            record['reference_head'] = checked(['git', '-C', args.reference, 'rev-parse', 'HEAD'])
            checked(['git', '-C', args.reference, 'merge-base', '--is-ancestor',
                     REFERENCE_COMMIT, record['reference_head']])
            for relative, expected in manifest['reference_files'].items():
                if digest(args.reference/relative) != expected:
                    errors.append(f'Original R source changed: {relative}')
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f'Original R reference verification failed: {exc}')
        rscript = shutil.which(args.rscript)
        record['rscript'] = rscript
        record['R_LIBS_USER'] = env.get('R_LIBS_USER')
        if not rscript:
            errors.append(f'Rscript unavailable: {args.rscript}')
        else:
            expr = 'cat("R=",as.character(getRversion()),"\\n",sep="");' + \
                'for(p in c("EBImage","RobustGaSP","magick","pracma")) ' + \
                'cat(p,"=",if(requireNamespace(p,quietly=TRUE)) as.character(packageVersion(p)) else "MISSING","\\n",sep="")'
            try:
                found = dict(line.split('=', 1) for line in checked([rscript, '-e', expr], env=env).splitlines() if '=' in line)
                record['R_packages'] = found
                for key, version in R_VERSIONS.items():
                    if found.get(key) != version:
                        errors.append(f'{key}: found {found.get(key)}, expected {version}')
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f'R dependency probe failed: {exc}')
        free = shutil.disk_usage(args.out).free
        record['free_disk_bytes'] = free
        if free < 8*1024**3:
            errors.append('Need at least 8 GiB free before starting/resuming')
    record['errors'] = errors
    dump(args.out/('environment.json' if require_r else 'local_environment.json'), record)
    if errors:
        raise RuntimeError('\n'.join(errors))
    identity = {k: v for k, v in record.items() if k not in ('free_disk_bytes', 'errors')}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return record, signature, env


def load_modules():
    sys.path[:0] = [str(HERE), str(ROOT), str(ROOT/'experiments/real_data/fast_qep_joint_mh_20261015')]
    path = ROOT/'experiments/real_data/fast_qep_joint_mh_20261015/run.py'
    spec = importlib.util.spec_from_file_location('frozen_joint_validation', path)
    joint = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = joint; spec.loader.exec_module(joint)
    import frozen_metrics as metrics
    return joint, metrics


def data(joint, metrics):
    manifest = rows(ROOT/'results/real_cellseg_round3_thresholding_20260916/image_manifest.csv')
    historical = rows(ROOT/'results/qep_downstream_4x2_completed_20260924/metrics.csv')
    prepared = []
    for ds, grid, first_ref, _, meta in joint.prepare():
        entry = next(m for m in manifest if m['dataset'] == ds and m['role'] == 'development')
        raw, gt = metrics.gray(ROOT/entry['path_image']), metrics.ground_truth(ROOT/entry['path_gt'])
        if raw.shape != gt.shape or len(joint.np.unique(gt[gt > 0])) != int(entry['n_gt_instances']):
            raise RuntimeError('Input/GT geometry mismatch')
        tiles = []
        h, w = grid.shape
        for col in range(raw.shape[1]//w):
            for row in range(raw.shape[0]//h):
                tiles.append((len(tiles)+1, row*h, col*w))
        if len(tiles) != (16 if ds == 'nuclei' else 9):
            raise RuntimeError('Unexpected original tiling')
        anchors = {method: next(r for r in historical if r['dataset'] == ds and r['method'] == method and r['downstream'] == 'P')
                   for method in ('raw', 'fastgp')}
        prepared.append(dict(ds=ds, grid=grid, nu=first_ref.nugget_ratio, meta=meta,
                             entry=entry, raw=raw, gt=gt, tiles=tiles, anchors=anchors))
    return prepared


def analytic(work, joint):
    np = joint.np
    gp = np.zeros_like(work['raw']); control = gp.copy(); checks = []
    grid = work['grid']; h, w = grid.shape; original = joint.load_original()
    p = work['meta']['r_parameters']
    params = original['SharedParams'](float(p['beta1']), float(p['beta2']), work['nu'],
                                     h, w, 'frozen whole-image tiles', 0., 'none', 0, 0.)
    for tile_id, y, x in work['tiles']:
        tile = work['raw'][y:y+h, x:x+w]/255
        ref = grid.gaussian_reference(tile, work['nu'])
        error = float(np.max(abs(ref.prediction-original['reconstruct_tile'](tile, params).pred_mean)))
        if error >= 1e-10:
            raise RuntimeError('q=2/original Python parity failed')
        prop = joint.TiltedGaussian(grid.posterior(ref, 1.5))
        gp[y:y+h, x:x+w] = ref.prediction*255
        control[y:y+h, x:x+w] = (ref.mean+grid.inverse(prop.mean))*255
        checks.append(dict(tile=tile_id, y=y, x=x, q2_original_error=error, proposal=prop.record()))
    return gp, control, checks


def local_checks(out, joint, metrics, prepared):
    np = joint.np; report = []
    for work in prepared:
        ds = work['ds']; print(f'Checking archived anchors and all analytic tiles: {ds}', flush=True)
        gp, control, tiles = analytic(work, joint)
        with np.load(HERE/'anchors'/ds/'fastgp_reconstruction.npz') as z:
            error = float(np.max(abs(gp-z['predmean']))/255)
        if error >= 1e-10:
            raise RuntimeError(f'{ds}: analytic/archived R reconstruction mismatch: {error}')
        for method in ('raw', 'fastgp'):
            with np.load(HERE/'anchors'/ds/f'{method}_P_segmentation.npz') as z:
                row, _ = metrics.score(work['gt'], z['labels'], z['foreground'])
                bad = metrics.anchor_differences(row, work['anchors'][method])
                if bad:
                    raise RuntimeError(f'Frozen metric anchor failed: {ds}/{method}: {bad}')
                # A small real crop checks the AST-loaded fast/legacy IoU paths.
                parity = metrics.check_iou_parity(work['gt'][:100,:100], z['labels'][:100,:100], 6)
                if parity > 1e-14:
                    raise RuntimeError('Fast/legacy IoU parity failed')
            report.append(dict(dataset=ds, method=method, metrics=row, archived_anchor_pass=True,
                               analytic_vs_archived_R_normalized_max=error, iou_parity_error=parity))
        work['analytic_gp'], work['control'] = gp, control
        dump(out/f'{ds}_analytic_tiles.json', tiles)
    dump(out/'cached_anchor_checks.json', report)
    dump(out/'deterministic_checks.json', joint.checks())
    small = joint.small(out)
    if not all(x['passed'] for x in small):
        raise RuntimeError('Small reference gate failed; no image sampling permitted')
    dump(out/'local_checks.json', {'passed': True, 'real_R_executed': False,
                                  'full_image_QEP_sampled': False, 'anchor_rows': len(report)})


def rcall(args, env, script, argv, log):
    with open(log, 'w') as f:
        try:
            subprocess.run([args.rscript, str(ROOT/'experiments/real_data/qep_downstream_4x2'/script),
                            *map(str, argv)], env=env, stdout=f, stderr=subprocess.STDOUT,
                           check=True, timeout=900)
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f'R call failed; see {log}: {exc}') from exc


def downstream(args, env, work, folder, method, reconstruction, metrics, np):
    folder.mkdir(parents=True, exist_ok=True)
    reconstruction_csv = folder/f'{method}.csv'
    np.savetxt(reconstruction_csv, reconstruction, delimiter=',', fmt='%.17g')
    started = time.perf_counter()
    rcall(args, env, 'paper_downstream.R', [work['ds'], ROOT/work['entry']['path_image'],
          reconstruction_csv, folder, method, args.reference], folder/f'{method}_R.log')
    binary = np.loadtxt(folder/f'{method}_combined_binary.csv', delimiter=',') > 0
    labels = metrics.validate_labels(np.loadtxt(folder/f'{method}_labels_final.csv', delimiter=','), work['gt'].shape)
    row, states = metrics.score(work['gt'], labels, binary)
    summary = rows(folder/f'{method}_summary.csv')[0]
    row.update(dataset=work['ds'], method=method, downstream='original_R',
               n_precleanup=int(summary['n_labels_precleanup']), n_outlier_tiles=int(summary['n_outlier_tiles']),
               runtime_downstream_s=time.perf_counter()-started)
    np.savez_compressed(folder/f'{method}_segmentation.npz', labels=labels, foreground=binary, **states)
    dump(folder/f'{method}_metrics.json', row)
    print(f"{work['ds']} {method}: AP50={row['ap50']:.8f}, merge={row['merged']}, split={row['split']}", flush=True)
    return row, states, labels, binary


def preflight(args, signature, env, joint, metrics, prepared):
    np = joint.np; folder = args.out/'preflight'; complete = folder/'preflight_complete.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        if record['signature'] != signature:
            raise RuntimeError('Execution environment/source changed since preflight')
        verify_artifacts(folder, record['artifacts'])
        print('Verified previous preflight: PASS', flush=True)
        return
    if folder.exists():
        raise RuntimeError(f'Incomplete/failed preflight retained at {folder}; review before rerunning')
    folder.mkdir()
    local_checks(folder, joint, metrics, prepared)
    anchor_rows = []
    for work in prepared:
        ds = work['ds']; dsdir = folder/ds; dsdir.mkdir()
        np.savetxt(dsdir/'raw_input.csv', work['raw'], delimiter=',', fmt='%.17g')
        start = time.perf_counter()
        rcall(args, env, 'restore_fastgp.R', [args.reference, dsdir/'raw_input.csv',
              ROOT/f'audits/parity/real_{ds}_r_params.csv', dsdir/'r_fastgp.csv'], dsdir/'restore_fastgp.log')
        reconstruction_time = time.perf_counter()-start
        r_gp = np.loadtxt(dsdir/'r_fastgp.csv', delimiter=',')
        if r_gp.shape != work['raw'].shape or not np.isfinite(r_gp).all():
            raise RuntimeError('Invalid R reconstruction')
        err = float(np.max(abs(r_gp-work['analytic_gp']))/255)
        if err >= 1e-10:
            raise RuntimeError(f'Python q2 versus actual R failed: {ds}, error {err}')
        # Primary GP goes through the same Python numerical representation as QEP.
        # R reconstruction and archived mask checks protect the baseline identity.
        for method, a in [('raw', work['raw']), ('fastgp', work['analytic_gp'])]:
            row, _, labels, binary = downstream(args, env, work, dsdir, method, a, metrics, np)
            bad = metrics.anchor_differences(row, work['anchors'][method])
            with np.load(HERE/'anchors'/ds/f'{method}_P_segmentation.npz') as cached:
                same_fg = np.array_equal(binary, cached['foreground'])
                same_partition = np.array_equal(metrics.canonical(labels), metrics.canonical(cached['labels']))
            check = dict(dataset=ds, method=method, differences=bad, same_foreground=same_fg,
                         same_partition=same_partition, python_q2_vs_R_max_normalized=err,
                         R_reconstruction_runtime_s=reconstruction_time)
            anchor_rows.append(check); dump(folder/'R_anchor_checks.json', anchor_rows)
            if bad or not same_fg or not same_partition:
                raise RuntimeError(f'R anchor mismatch: {check}')
        np.savez_compressed(dsdir/'reconstructions.npz', fastgp=work['analytic_gp'], gp_eta=work['control'])
    dump(complete, dict(signature=signature, status='PASS_ALL_PREFLIGHT_GATES', artifacts=artifacts(folder)))
    print('PASS_ALL_PREFLIGHT_GATES — actual R anchors and numerical gates passed.', flush=True)


def reconstruct(args, signature, work, joint):
    np = joint.np; ds = work['ds']; grid = work['grid']; h, w = grid.shape
    qep = np.zeros_like(work['raw']); mcse = qep.copy()
    chain_means = np.zeros((4, *qep.shape)); raw_qep = qep.copy(); total = 0.
    for other in (args.out/'tiles').glob('*/*/checkpoint.json'):
        previous = json.loads(other.read_text())
        if previous['status'] == 'PASSED':
            total += previous['sampler_runtime_s']
    for tile_id, y, x in work['tiles']:
        folder = args.out/'tiles'/ds/f'tile{tile_id:02d}'
        checkpoint = folder/'checkpoint.json'; tile = work['raw'][y:y+h,x:x+w]/255
        tile_hash = hashlib.sha256(np.ascontiguousarray(tile, dtype='<f8').tobytes()).hexdigest()
        identity = dict(signature=signature, dataset=ds, tile=tile_id, y=y, x=x,
                        tile_sha256=tile_hash, fixture_id=(1000 if ds == 'nuclei' else 2000)+tile_id)
        if checkpoint.exists():
            old = json.loads(checkpoint.read_text())
            if old['identity'] != identity or old['status'] != 'PASSED':
                raise RuntimeError(f'Cannot automatically retry changed/incomplete/failed tile: {folder}')
            verify_artifacts(folder, old['artifacts'])
            print(f'{ds} tile {tile_id}: verified passed checkpoint', flush=True)
        else:
            if folder.exists():
                raise RuntimeError(f'Incomplete tile directory: {folder}')
            if total >= 7200:
                raise RuntimeError('Cumulative 7200-second sampling cap reached')
            folder.mkdir(parents=True)
            dump(checkpoint, dict(identity=identity, status='STARTED'))
            ref = grid.gaussian_reference(tile, work['nu']); target = grid.posterior(ref, 1.5)
            obs = joint.Observer(target, 4000)
            print(f'{ds} tile {tile_id}/{len(work["tiles"])}: START, n={grid.n}', flush=True)
            sampler, initial, final, traces = joint.sample(target, joint.config(4000), identity['fixture_id'], obs,
                max_seconds=min(900, 7200-total),
                progress=lambda i,n,s: print(f'  {ds} tile {tile_id}: {i}/{n}, {s:.1f}s', flush=True) if i%1000==0 else None)
            total += sampler['runtime_sec']
            raw, cv = obs.complete_blocks()
            np.savez_compressed(folder/'chains.npz', trace=obs.traces[:,:obs.count], trace_names=np.asarray(obs.names),
                spectral_indices=obs.indices, extra_trace=obs.extra[:,:obs.count],
                global_weight=obs.global_weight[:,:obs.count], raw_spectral_blocks=raw,
                controlled_spectral_blocks=cv, initial_z=initial, final_z=final, **traces)
            result = dict(identity=identity, sampler=sampler, passed=False)
            if sampler['completed']:
                diag = obs.diagnostics(); gate = joint.gates(diag, sampler, 1.5)
                cvs = joint.pixel_block_summary(grid, cv); raws = joint.pixel_block_summary(grid, raw)
                se = cvs['mcse']*255; mean = (ref.mean+cvs['mean'])*255
                gate['pixel_mcse'] = bool(np.quantile(se,.99) <= .1 and se.max() <= .5)
                gate['cumulative_sampling_time'] = total <= 7200
                result.update(passed=all(gate.values()), gates=gate, diagnostics=diag,
                              **joint.old.diag_summary(diag), mcse_p99=float(np.quantile(se,.99)),
                              mcse_max=float(se.max()), mean_absolute_q_change=float(np.mean(abs(mean-ref.prediction*255))))
                np.savez_compressed(folder/'estimates.npz', q15=mean, q15_mcse=se, q2=ref.prediction*255,
                    raw_q15=(ref.mean+raws['mean'])*255, chain_means=(ref.mean+cvs['chain_means'])*255)
            joint.save(folder/'summary.json', result)
            dump(checkpoint, dict(identity=identity, status='PASSED' if result['passed'] else 'FAILED',
                                 sampler_runtime_s=sampler['runtime_sec'], artifacts=artifacts(folder)))
            if not result['passed']:
                raise RuntimeError(f'Tile numerical/runtime gate failed: {folder}')
            print(f'{ds} tile {tile_id}: PASS, Rhat={result["max_rhat"]:.6f}, MCSE99={result["mcse_p99"]:.6g}', flush=True)
            del obs, raw, cv, cvs, raws
        with np.load(folder/'estimates.npz') as z:
            qep[y:y+h,x:x+w] = z['q15']; mcse[y:y+h,x:x+w] = z['q15_mcse']
            raw_qep[y:y+h,x:x+w] = z['raw_q15']; chain_means[:,y:y+h,x:x+w] = z['chain_means']
    return dict(q15=qep, q15_mcse=mcse, q15_raw=raw_qep, chain_means=chain_means)


def finish(args, env, prepared, joint, metrics):
    np = joint.np; all_rows = []; contrasts = []
    for work in prepared:
        ds = work['ds']; pre = args.out/'preflight'/ds; folder = args.out/'segmentation'/ds
        baseline = json.loads((pre/'fastgp_metrics.json').read_text())
        all_rows.extend([json.loads((pre/f'{method}_metrics.json').read_text()) for method in ('raw','fastgp')])
        with np.load(pre/'fastgp_segmentation.npz') as z:
            before = {k: z[k] for k in ('matched', 'split')}; baseline_fg = z['foreground']
        with np.load(args.out/'reconstruction'/f'{ds}.npz') as z:
            arms = [('fastqep_q15', z['q15']), ('gp_eta', z['gp_eta'])]
            arms += [(f'q15_chain{i+1}', z['chain_means'][i]) for i in range(4)]
        for method, a in arms:
            row, states, _, fg = downstream(args, env, work, folder, method, a, metrics, np)
            changes = metrics.identity_changes(before, states)
            contrast = dict(dataset=ds, method=method,
                **{'delta_'+k: row[k]-baseline[k] for k in ('ap50','ap75','merged','split','missed','spurious','fg_dice')},
                foreground_changed_pixels=int(np.count_nonzero(fg != baseline_fg)), **changes)
            contrast['descriptive_merge_route_support'] = bool(contrast['delta_ap50'] > 0 and contrast['delta_merged'] < 0
                and contrast['delta_ap75'] >= 0 and contrast['delta_split'] <= 0
                and changes['new_split_previously_matched']['count'] == 0)
            contrasts.append(contrast); all_rows.append(row)
            table(args.out/'metrics.csv', all_rows); dump(args.out/'comparisons.json', contrasts)
    stability = []
    for work in prepared:
        ds = work['ds']; cc = [x for x in contrasts if x['dataset'] == ds and x['method'].startswith('q15_chain')]
        dd = [x['delta_ap50'] for x in cc]
        stability.append(dict(dataset=ds, chain_delta_ap50=dd, all_chain_AP50_gains_positive=all(x>0 for x in dd),
                              minimum_delta_ap50=min(dd), maximum_delta_ap50=max(dd)))
    dump(args.out/'MC_stability.json', stability)
    lines = ['# Fast-QEP original-pipeline replacement — development images', '',
             'Actual R downstream executed. All 25 original tiles passed frozen numerical gates.', '',
             '| Dataset | Arm | AP50 | AP75 | Merge | Split | Missed | Dice |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in all_rows:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['ap50']:.6f} | {row['ap75']:.6f} | {row['merged']} | {row['split']} | {row['missed']} | {row['fg_dice']:.6f} |")
    lines += ['', 'AP is the frozen legacy TP/(TP+FP+FN) score, not area under a precision-recall curve.',
              'Chain rows diagnose Monte Carlo stability; they are not independent image replicates.',
              'gp_eta is a fixed Gaussian smoothing-strength control, not an optimal fitted GP.',
              'See comparisons.json for gained/lost GT identities, harm and descriptive support flags;',
              'MC_stability.json for all four chain gains, and tile summaries for inference diagnostics.',
              'One development image per dataset: no significance or generalization claim.',
              'No new q search, downstream tuning, or held-out evaluation was performed.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')


class Tee:
    def __init__(self, stream, logfile): self.stream, self.logfile = stream, logfile
    def write(self, text): self.stream.write(text); self.logfile.write(text); self.flush()
    def flush(self): self.stream.flush(); self.logfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true', help='Includes actual R baseline checks; no full-image QEP')
    action.add_argument('--run', action='store_true', help='Preflight then full experiment; resume only passed tiles')
    action.add_argument('--local-checks', action='store_true', help='No R; validates code/cached anchors only')
    parser.add_argument('--reference', type=Path, default=Path('/Users/zchan/eclipse-workspace/cell_segmentation_original'))
    parser.add_argument('--rscript', default='Rscript')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); args.out = args.out.resolve(); args.reference = args.reference.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'run.log').open('a', buffering=1) as log:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
        try:
            print(f'Output: {args.out}\nMode: {"local checks" if args.local_checks else "run" if args.run else "preflight"}', flush=True)
            _, signature, env = environment(args, require_r=not args.local_checks)
            joint, metrics = load_modules()
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                prepared = data(joint, metrics)
                if args.local_checks:
                    folder = args.out/'local_checks'
                    if folder.exists(): raise RuntimeError('Local check outputs already exist')
                    folder.mkdir(); local_checks(folder, joint, metrics, prepared)
                    print('LOCAL_CHECKS_PASS; actual R and full-image QEP NOT executed.', flush=True)
                    return
                preflight(args, signature, env, joint, metrics, prepared)
                if args.preflight: return
                if (args.out/'completion.json').exists():
                    record = json.loads((args.out/'completion.json').read_text())
                    if record.get('status') == 'COMPLETED':
                        verify_artifacts(args.out, record['artifacts'])
                        print('Already completed; verified saved results.'); return
                recon_dir = args.out/'reconstruction'; recon_dir.mkdir(exist_ok=True)
                for work in prepared:
                    estimates = reconstruct(args, signature, work, joint)
                    with joint.np.load(args.out/'preflight'/work['ds']/'reconstructions.npz') as z:
                        joint.np.savez_compressed(recon_dir/f'{work["ds"]}.npz', **estimates, fastgp=z['fastgp'], gp_eta=z['gp_eta'])
                finish(args, env, prepared, joint, metrics)
                hashes = artifacts(args.out)
                for key in ('completion.json', 'run.log', 'environment.json'):
                    hashes.pop(key, None)
                dump(args.out/'completion.json', dict(status='COMPLETED', signature=signature,
                    development_images=2, original_tiles=25, real_R_executed=True, artifacts=hashes))
                print('COMPLETED. Inspect REPORT.md, metrics.csv, comparisons.json and MC_stability.json.', flush=True)
        except Exception as exc:
            traceback.print_exc()
            dump(args.out/'last_stop.json', dict(status='STOPPED_WITHOUT_AUTOMATIC_RETRY', error=str(exc),
                                               real_R_required=not args.local_checks))
            return 1
        finally:
            sys.stdout, sys.stderr = original_out, original_err
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
