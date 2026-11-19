import json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2
from config import ROOT as here,PKG as payload,picture,geometry,field_names
exp=payload/"experiments/real_data"
from mixture import ScalePrior,evaluate
from py_core.fast_qep_prototype import KroneckerGrid,covariance_factor
sys.path[:0]=[str(exp/"fast_qep_joint_mh_20261015"),str(payload/"audits/fast_qep_radial_diagnosis_20261014")]
from joint_mh import TiltedGaussian
def check(ds):
    q=1.15
    out=here/"results"/ds
    raw=picture(ds)
    shape=tuple(geometry(size)[0] for size in raw.shape)
    tile=raw[:shape[0],:shape[1]]/255
    n=tile.size
    prior=ScalePrior(n,q=q)
    xx=np.linspace(-15*prior.sd,15*prior.sd,4001)
    pdf=np.exp(prior.logpdf(xx))
    normal=simpson(pdf,x=xx)
    prior_mean=simpson(pdf*np.exp(xx),x=xx)
    qradius_errors=[]
    k=n/2; b=2/q; loga=np.log(covariance_factor(n,q))
    for quant in [.001,.01,.25,.5,.75,.99,.999]:
        r=chi2.ppf(quant,n)
        h=b*np.log(r)-loga
        # Density of the log radius squared, computed from the model two ways.
        target=k*np.log(r)-r/2-k*np.log(2)-gammaln(k)-np.log(b)
        gpdf=k*(h-xx)-np.exp(h-xx)/2-k*np.log(2)-gammaln(k)
        mixture=np.log(simpson(np.exp(gpdf)*pdf,x=xx))
        qradius_errors.append(float(abs(target-mixture)))
    assert max(abs(normal-1),abs(prior_mean-1),max(qradius_errors))<1e-7
    result=dict(prior_integral=normal,prior_mean_S=prior_mean,max_radial_log_density_error=max(qradius_errors),checks=[])
    fits=json.loads((out/"fit_q115_fit.json").read_text())
    assert fits["q"]==q
    first=fits["tiles"][0]
    m,s=first["mean"],first["s"]
    br,bc,lnu=fits["best"]["x"][:3]
    nu=np.exp(lnu)
    g=KroneckerGrid.matern52(shape,np.exp(br),np.exp(bc))
    w=g.forward(tile-m)
    ll,f=evaluate(g.eigenvalues,w,s,nu,prior,order=24,predict=True)
    ref=g.gaussian_reference(tile,nu)
    ref.residual_spectral=w; ref.signal_variance=s; ref.noise_variance=s*nu
    target=g.posterior(ref,q); prop=TiltedGaussian(target)
    # Direct posterior importance sampling with a different probability representation.
    # Symmetry of the likelihood under orthogonal row/column changes preserves the evidence.
    eta,t0=prop.eta,prop.t0
    alpha=q/2
    c=(np.log(alpha)+n*q/4*loga-n/2*np.log(eta)+n*(q-2)/4*np.log(t0)
       -.5*(np.exp(loga)*t0)**alpha+.5*eta*t0)
    vle=s*(g.eigenvalues/eta+nu)
    normal_ll=-.5*(np.log(2*np.pi*vle).sum()+(w*w/vle).sum())
    rng=np.random.default_rng(1152026+n)
    weights=[]
    for _ in range(256):
        draws=prop.mean+prop.sd*rng.normal(size=(8,n))
        weights.extend(np.exp(prop.log_weight(draws)))
    weights=np.array(weights)
    is_ll=normal_ll+c+np.log(weights.mean())
    isse=weights.std(ddof=1)/np.sqrt(len(weights))/weights.mean()
    gp_ll=evaluate(g.eigenvalues,w,s,nu,None)[0]
    result.update(independent_IS_logevidence=is_ll,quadrature_logevidence=ll,
                  independent_IS_mcse=isse,standardized_difference=(is_ll-ll)/isse)
    assert abs(is_ll-ll)<5*isse
    assert abs(ll-first["ll"])<1e-6
    return result
if __name__=="__main__":
    checks={d:check(d) for d in [field_names()[0],field_names()[-1]]}
    print(json.dumps(checks,indent=2))
    (here/"quadrature_checks.json").write_text(json.dumps(checks,indent=2))
