"""Import the unchanged validated sampler in an isolated module namespace."""
from pathlib import Path
import importlib.util
import sys
from functools import lru_cache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


@lru_cache(maxsize=1)
def frozen_joint():
    folder = ROOT/'experiments/real_data/fast_qep_joint_mh_20261015'
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('unchanged_fastqep_joint', folder/'run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
