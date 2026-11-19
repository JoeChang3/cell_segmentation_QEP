from config import ROOT,PKG,picture,field_names,geometry,np,pd
import json,sys
sys.path.insert(0,str(PKG/"experiments/real_data/fast_qep_hyperfit_20261015"))
from pilot import fit_gp
p=pd.read_csv(PKG/"audits/parity/real_whole_cell_r_params.csv").iloc[0]
starts=np.array([p.theta_hat,np.log(p.sigma2_hat),np.log(p.nugget_nu),np.log(p.beta1),np.log(p.beta2)])
checks=[]
for field in field_names():
    base=json.loads((ROOT/"results"/field/"fit_gp_fit.json").read_text())
    if base["best"]["success"]:continue
    raw=picture(field); h,w=[geometry(s)[0] for s in raw.shape]
    result=fit_gp(raw[:h,:w]/255,starts)
    best=max(result["fits"],key=lambda r:r["log_evidence"])
    delta=best["log_evidence"]-base["best"]["ll"]
    assert result["stationarity_pass"] and abs(delta)<1e-4
    checks.append(dict(field=field,independent_stationarity=result["stationarity_pass"],independent_ll=best["log_evidence"],ll_difference=delta,
                       gradient_per_pixel=best["projected_gradient_per_pixel_max"]))
(ROOT/"gp_warning_check.json").write_text(json.dumps(checks,indent=2))
print(checks)
