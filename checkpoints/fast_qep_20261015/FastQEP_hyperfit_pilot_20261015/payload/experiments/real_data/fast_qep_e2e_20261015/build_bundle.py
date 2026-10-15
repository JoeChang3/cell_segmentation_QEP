"""Package explicit audited dependencies; never copy or modify the user's checkout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    source_manifest = json.loads((ROOT/'results/qep_downstream_4x2_completed_20260924/artifact_manifest.json').read_text())
    for ds in ('nuclei', 'whole_cell'):
        for name in ('fastgp_reconstruction.npz', 'raw_P_segmentation.npz', 'fastgp_P_segmentation.npz'):
            source = args.cache/ds/name; expected = source_manifest['files'][f'{ds}/{name}']['sha256']
            if sha(source) != expected:
                raise RuntimeError(f'Historical cache hash mismatch: {source}')
            dst = HERE/'anchors'/ds/name; dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and sha(dst) != expected:
                raise RuntimeError(f'Existing anchor differs: {dst}')
            shutil.copy2(source, dst)
    relatives = [
        'py_core/__init__.py',
        'py_core/fast_qep_prototype.py', 'py_core/paper_fast_gp.py', 'py_core/dim_2_lattice.py',
        'py_core/metrics.py', 'py_core/segmentation_eval.py', 'py_core/foreground_threshold.py',
        'experiments/real_data/fast_qep_joint_mh_20261015/run.py',
        'experiments/real_data/fast_qep_joint_mh_20261015/joint_mh.py',
        'experiments/real_data/fast_qep_joint_mh_20261015/PROTOCOL.md',
        'experiments/real_data/fast_qep_radial_hmc_20261014/run.py',
        'experiments/real_data/fast_qep_radial_hmc_20261014/radial_hmc.py',
        'experiments/real_data/fast_qep_real_pilot_20261014/run_pilot.py',
        'experiments/real_data/fast_qep_real_pilot_20261014/stream_hmc.py',
        'experiments/real_data/qep_downstream_4x2/restore_fastgp.R',
        'experiments/real_data/qep_downstream_4x2/paper_downstream.R',
        'audits/fast_qep_feasibility_20261014/verify_math.py',
        'audits/fast_qep_prototype_20261014/diagnostics.py',
        'audits/fast_qep_prototype_20261014/validation/deterministic_checks.json',
        'audits/fast_qep_prototype_20261014/r_reference_validation/fastgp_r_reference.zip',
        'audits/fast_qep_radial_diagnosis_20261014/independence_mh.py',
        'audits/fast_qep_paper_alignment_20261014/MODEL_CONTRACT.md',
        'audits/parity/real_nuclei_r_params.csv', 'audits/parity/real_whole_cell_r_params.csv',
        'results/real_cellseg_round3_thresholding_20260916/image_manifest.csv',
        'results/qep_downstream_4x2_completed_20260924/metrics.csv',
        'results/qep_downstream_4x2_completed_20260924/artifact_manifest.json',
    ]
    with (ROOT/'results/real_cellseg_round3_thresholding_20260916/image_manifest.csv').open() as f:
        for row in csv.DictReader(f):
            if row['role'] == 'development':
                relatives.extend([row['path_image'], row['path_gt']])
    relatives += [str(f.relative_to(ROOT)) for f in HERE.rglob('*') if f.is_file()
                  and '__pycache__' not in str(f) and f.name != 'bundle_manifest.json']
    files = {relative: sha(ROOT/relative) for relative in sorted(set(relatives))}
    r_source = 'src/Modified_Functions_RGasp.R'
    committed = subprocess.check_output(['git','-C',str(args.reference),'show',
        '44714c2e0be958fe796a8fd4bdbc220dae3c23dd:'+r_source])
    if hashlib.sha256(committed).hexdigest() != sha(args.reference/r_source):
        raise RuntimeError('R reference source differs from frozen commit')
    manifest = dict(base_commit='fa746853ba007552bb1d5a43340d91f6796096b1',
                    reference_commit='44714c2e0be958fe796a8fd4bdbc220dae3c23dd', files=files,
                    reference_files={r_source: sha(args.reference/r_source)},
                    historical_anchor_source_archive_sha256=source_manifest['source_archive_sha256'])
    (HERE/'bundle_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    dest = args.destination/'FastQEP_R_pipeline_20261015'
    if dest.exists():
        raise RuntimeError(f'Build destination exists: {dest}')
    for relative in [*files, str((HERE/'bundle_manifest.json').relative_to(ROOT))]:
        target = dest/'payload'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    shutil.copy2(HERE/'README.md', dest/'README.md')
    (dest/'run.py').write_text('from pathlib import Path\nimport runpy\n'
        'runpy.run_path(str(Path(__file__).resolve().parent/"payload/experiments/real_data/fast_qep_e2e_20261015/run.py"), run_name="__main__")\n')
    archive = args.destination/(dest.name+'.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dest.rglob('*')):
            if f.is_file(): z.write(f, f.relative_to(args.destination))
    print(json.dumps({'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
                      'payload_files': len(files)+1}, indent=2))


if __name__ == '__main__':
    main()
