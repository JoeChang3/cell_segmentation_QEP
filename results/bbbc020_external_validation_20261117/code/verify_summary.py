"""Independent arithmetic checks on final paired metric exports."""
from pathlib import Path
import numpy as np
import pandas as pd
from config import ROOT,field_names
scores=pd.read_csv(ROOT/"field_scores.csv")
assert len(scores)==3*len(field_names()) and not scores.duplicated(["arm","field"]).any()
for t in ["50","75"]:
    assert np.allclose(scores["ap"+t],scores["tp"+t]/(scores["tp"+t]+scores["fp"+t]+scores["fn"+t]),rtol=0,atol=2e-15)
summary=pd.read_csv(ROOT/"summary.csv")
for _,row in summary.iterrows():
    wide=scores.pivot(index="field",columns="arm",values=row.metric)
    delta=wide[row.arm]-wide.fit_gp
    assert row.win==(delta>0).sum() and row.loss==(delta<0).sum() and row.tie==(delta==0).sum()
    assert np.isclose(row.mean_delta,delta.mean(),rtol=0,atol=2e-15)
print("verified",len(scores),"field/arm records",len(summary),"paired summaries")
