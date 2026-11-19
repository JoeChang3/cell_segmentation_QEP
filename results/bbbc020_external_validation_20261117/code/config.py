from pathlib import Path
import os,sys,json
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
REPO=Path(os.environ.get("CELLSEG_REPO",str(ROOT.parent/"independent_repo")))
PKG=REPO/"checkpoints/fast_qep_20261015/FastQEP_hyperfit_pilot_20261015/payload"
E2E=PKG/"experiments/real_data/fast_qep_e2e_20261015"
RSCRIPT=os.environ.get("RSCRIPT",str(ROOT/"r_env/bin/Rscript"))
sys.path[:0]=[str(ROOT/"gp"),str(PKG),str(E2E)]
from frozen_metrics import canonical,compute_ap_from_ious
from py_core.fast_qep_prototype import KroneckerGrid
from mixture import ScalePrior,evaluate

def geometry(size):
    divisor=next((d for d in (.25,1/3,.5,1) if 200<=size*d<=400),.25)
    count=size//int(size*divisor)
    return size//count,count

def field_names():
    return json.loads((ROOT/"data/manifest.json").read_text())["fields"]

def picture(name):
    return np.array(Image.open(ROOT/"data"/(name+".png")),dtype=float)
