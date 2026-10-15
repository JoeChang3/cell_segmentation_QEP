"""Versioned command-line wrapper; never overwrites or silently resumes a run."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import traceback
import zipfile
from threadpoolctl import threadpool_limits
import runtime
from pilot import prepare_inputs,run_pilot,digest
from validate import checks

HERE=Path(__file__).resolve().parent
BUNDLE=runtime.ROOT.parent


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def verify_source():
    manifest=json.loads((BUNDLE/'hyperfit_manifest.json').read_text())
    bad=[relative for relative,expected in manifest['files'].items()
         if not (BUNDLE/relative).is_file() or digest(BUNDLE/relative)!=expected]
    if bad:raise RuntimeError('Source/input hash mismatch: '+str(bad))
    return {'manifest_sha256':digest(BUNDLE/'hyperfit_manifest.json'),
            'verified_files':len(manifest['files']),
            'original_execution_zip_sha256':manifest['original_execution_zip_sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--check',action='store_true',help='Mathematical and fixed-anchor checks only (default).')
    group.add_argument('--pilot',action='store_true',help='Fit GP controls and sample QEP scores on original first tiles only.')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    suffix='pilot' if args.pilot else 'check'
    out=(args.out or runtime.ROOT/f'results/fast_qep_hyperfit_{suffix}_20261015').resolve()
    if out.exists():raise SystemExit(f'Refusing overwrite/retry: {out}\nKeep the completed or failed run for review.')
    out.mkdir(parents=True)
    status={'status':'STARTED','mode':suffix,'full_image_segmentation':False,'qep_optimization':False}
    error=None
    with (out/'run.log').open('w') as log,redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
        try:
            print('Output:',out,flush=True)
            env={'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),
                 'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','Pillow','threadpoolctl']},
                 'mode':suffix,'threads':1,**verify_source()}
            (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
            with threadpool_limits(limits=1):
                checks(out/'validation')
                work=prepare_inputs()
                (out/'anchor_checks.json').write_text(json.dumps([w['input'] for w in work],indent=2)+'\n')
                print('Both original first-tile R anchors PASS. No new R process was run.',flush=True)
                if args.pilot:run_pilot(out,work)
            status['status']='COMPLETED'
        except Exception as exc:
            status.update(status='FAILED',error=repr(exc))
            traceback.print_exc()
            error=exc
        finally:
            status['files']={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*'))
                             if p.is_file() and p.name not in ['completion.json','run.log']}
            (out/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
            print(status['status'],flush=True)
    archive=out.parent/(out.name+'_review.zip')
    if archive.exists():raise SystemExit(f'Review archive already exists: {archive}')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,out.name+'/'+str(p.relative_to(out)))
    print('Upload for review:',archive,flush=True)
    if error:raise SystemExit(1)


if __name__=='__main__':main()
