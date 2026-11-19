from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json,sys
import numpy as np
from scipy.optimize import minimize
from config import ROOT as here,PKG as payload,picture,geometry,field_names,pd
from mixture import ScalePrior,evaluate
from py_core.fast_qep_prototype import KroneckerGrid

def gradients(fun,x,h=8e-5):
    eye=np.eye(len(x))*h
    return np.array([(fun(x+e)-fun(x-e))/(2*h) for e in eye])
def optimize(fun,x,bounds,h=8e-5):
    result=minimize(fun,x,method="L-BFGS-B",jac=lambda x:gradients(fun,x,h),bounds=bounds,
                    options=dict(ftol=2e-15,gtol=3e-9,maxiter=160,maxfun=400,maxls=30))
    grad=gradients(fun,result.x,h)
    proj=[0 if (abs(a-l)<1e-6 and g>0 or abs(a-u)<1e-6 and g<0) else abs(g)
          for a,g,(l,u) in zip(result.x,grad,bounds)]
    if not result.success and h==8e-5 and max(proj)<=1e-6:
        print("Numerical retry",result.message,flush=True)
        alt,pg=optimize(fun,result.x,bounds,h/2)
        alt["retried"]=True
        return alt,pg
    return result, float(max(proj))
def fit(ds,q):
    raw=picture(ds)
    p=pd.read_csv(payload/"audits/parity/real_whole_cell_r_params.csv").iloc[0]
    h,w=(geometry(s)[0] for s in raw.shape)
    shape=(h,w)
    out=here/"results"/ds
    out.mkdir(exist_ok=True,parents=True)
    prior=ScalePrior(h*w,q=q) if q!=2 else None
    # Same search domain for both families. The mean and variance offsets
    # are local coordinates centered on the original method's per-tile profiles.
    offsets=[(-.1,.1),(-.25,.25)]
    bounds=[(np.log(5),np.log(max(p.beta1,20*(h-1)))),
            (np.log(5),np.log(max(p.beta2,20*(w-1)))),(np.log(1e-6),np.log(10)),*offsets]
    tile=raw[:h,:w]/255
    @lru_cache(maxsize=15)
    def grid(br,bc): return KroneckerGrid.matern52(shape,np.exp(br),np.exp(bc))
    @lru_cache(maxsize=15)
    def input_arrays(br,bc):
        g=grid(br,bc)
        return g.eigenvalues,g.forward(tile),g.forward(np.ones(shape))
    def model(x,data,sd,order=12,predict=False):
        br,bc,lnu,dm,dv=x
        lam,yy,xx=data
        nu=np.exp(lnu)
        denom=lam+nu
        m0=np.sum(xx*yy/denom)/np.sum(xx*xx/denom)
        s0=np.mean((yy-m0*xx)**2/denom)
        m=m0+sd*dm
        s=s0*np.exp(dv)
        ll,f=evaluate(lam,yy-m*xx,s,nu,prior,order,predict)
        return ll,m,s,f
    sd=tile.std()
    @lru_cache(maxsize=128)
    def loss_tuple(x): return -model(x,input_arrays(*x[:2]),sd)[0]/(h*w)
    def loss(x): return loss_tuple(tuple(x))
    starts=[(p.beta1,p.beta2,p.nugget_nu),(20.,20.,.1)]
    fits=[]
    for vals in starts:
        x0=np.r_[np.log(vals),0.,0.]
        opt,pg=optimize(loss,x0,bounds)
        fits.append(dict(x=opt.x.tolist(),initial=x0.tolist(),ll=float(-opt.fun*h*w),gradient=pg,
                         retry=bool(opt.get("retried",False)),steps=int(opt.nit),function_calls=int(opt.nfev),success=bool(opt.success),message=str(opt.message)))
        print(ds,q,fits[-1],flush=True)
    best=max(fits,key=lambda f:f["ll"])
    if (not best["success"] and not best["retry"]) or best["gradient"]>1e-6: raise ArithmeticError(f"fit not stationary: {best}")
    bestx=np.array(best["x"])
    g=grid(*bestx[:2])
    recon=np.zeros_like(raw)
    tiles=[]
    for col in range(raw.shape[1]//w):
        for row in range(raw.shape[0]//h):
            iy,ix=row*h,col*w
            local=raw[iy:iy+h,ix:ix+w]/255
            data=(g.eigenvalues,g.forward(local),g.forward(np.ones(shape)))
            if row==0 and col==0:
                x=bestx
                stat=best["gradient"]
            else:
                def tileloss(delta): return -model(np.r_[bestx[:3],delta],data,local.std())[0]/(h*w)
                opt,stat=optimize(tileloss,np.zeros(2),offsets)
                if (not opt.success and not opt.get("retried",False)) or stat>1.e-6: raise ArithmeticError(f"tile fit {ds}/{q}/{row}/{col}: {opt.message} {stat}")
                x=np.r_[bestx[:3],opt.x]
            ll,m,s,f=model(x,data,local.std(),24,True)
            ll0,_,_,f0=model(x,data,local.std(),12,True)
            pred=m+g.inverse(f*(data[1]-m*data[2]))
            error=float(np.max(np.abs(g.inverse((f-f0)*(data[1]-m*data[2])))))
            if abs(ll-ll0)>1e-5 or error>1e-9: raise ArithmeticError("integration check failed")
            recon[iy:iy+h,ix:ix+w]=pred*255
            tiles.append(dict(row=row,col=col,mean=m,s=s,ll=ll,q_order_delta=ll-ll0,prediction_error=error,
                              gradient=stat,offsets=x[-2:].tolist(),retry=best["retry"] if row==0 and col==0 else bool(opt.get("retried",False))))
    tag="fit_gp" if q==2 else "fit_q115"
    np.savez_compressed(out/(tag+"_recon.npz"),predmean=recon)
    (out/(tag+"_fit.json")).write_text(json.dumps(dict(dataset=ds,q=q,bounds=bounds,
             fits=fits,best=best,tiles=tiles),indent=2))
    print("FROZEN",ds,q,flush=True)

def fit_job(job):
    from contextlib import redirect_stdout,redirect_stderr
    import traceback
    ds,q=job
    tag="fit_gp" if q==2 else "fit_q115"
    folder=here/"results"/ds
    folder.mkdir(exist_ok=True,parents=True)
    if (folder/(tag+"_fit.json")).is_file(): return
    with (folder/(tag+".log")).open("w",buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        try: fit(ds,q)
        except Exception:
            traceback.print_exc()
            raise
if __name__=="__main__":
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import os
    with ProcessPoolExecutor(int(os.getenv("N_WORKERS","7"))) as pool:
        tasks={pool.submit(fit_job,(ds,q)):(ds,q) for ds in field_names() for q in (2.,1.15)}
        for task in as_completed(tasks):
            print(tasks[task],str(task.exception()) if task.exception() else "complete",flush=True)
