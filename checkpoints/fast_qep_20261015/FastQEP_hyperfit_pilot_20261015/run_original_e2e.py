from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent/"payload/experiments/real_data/fast_qep_e2e_20261015/run.py"), run_name="__main__")
