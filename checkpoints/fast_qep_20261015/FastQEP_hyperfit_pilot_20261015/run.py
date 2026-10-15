#!/usr/bin/env python3
"""Safe launcher: checks by default; --pilot explicitly starts first-tile work."""
import importlib.metadata
import os
from pathlib import Path
import runpy
import sys

if sys.version_info<(3,10):
    raise SystemExit('Python >=3.10 required; use the existing gpytorch_arm environment.')
missing=[]
for package in ['numpy','scipy','Pillow','threadpoolctl']:
    try:importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:missing.append(package)
if missing:
    raise SystemExit(f'Interpreter: {sys.executable}\nMissing: {missing}\nActivate gpytorch_arm, then rerun.')
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
folder=Path(__file__).resolve().parent/'payload/experiments/real_data/fast_qep_hyperfit_20261015'
sys.path.insert(0,str(folder))
runpy.run_path(str(folder/'run.py'),run_name='__main__')
