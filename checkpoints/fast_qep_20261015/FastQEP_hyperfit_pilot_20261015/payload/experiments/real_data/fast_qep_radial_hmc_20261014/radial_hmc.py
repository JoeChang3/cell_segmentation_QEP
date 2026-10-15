"""Compose the existing white HMC transition with conditional radial slice.

Same finite-dimensional target; no new likelihood/prior or fitted parameters.
The conditional slice update is exact in ideal arithmetic and solved to explicit
root tolerances in this implementation. No claim of independent draws or global
mixing follows from its construction.
"""
import math
import time
import numpy as np
from scipy.optimize import brentq
from py_core.fast_qep_prototype import leapfrog

XTOL=1e-13
RTOL=1e-14
MAXITER=120
MAX_EXPANSIONS=40


class RadialConditional:
    def __init__(self,n,q,A,B,current_t):
        if not (n>=1 and 1<=q<=2 and A>0 and all(map(math.isfinite,(A,B,current_t)))):
            raise ValueError('Invalid radial conditional parameters.')
        self.n,self.q,self.A,self.B=n,q,A,B
        self.evaluations=0
        def negative_slope(t):
            self.evaluations+=1
            x=math.exp(2*t/q)
            value=math.exp(2*t)+2*A/q*x*x-2*B/q*x-n
            if not math.isfinite(value):raise FloatingPointError('Nonfinite radial mode equation.')
            return value
        width=1.
        for _ in range(MAX_EXPANSIONS):
            left,right=current_t-width,current_t+width
            if negative_slope(left)<0<negative_slope(right):break
            width*=2
        else:raise RuntimeError('Could not bracket radial mode.')
        self.mode=brentq(negative_slope,left,right,xtol=XTOL,rtol=RTOL,maxiter=MAXITER)
        self.c2=.5*math.exp(2*self.mode)
        self.c4=.5*A*math.exp(4*self.mode/q)
        self.cb=B*math.exp(2*self.mode/q)
        self.slope=n-2*self.c2-4*self.c4/q+2*self.cb/q

    def centered(self,t):
        self.evaluations+=1
        d=t-self.mode
        def remainder(x):return math.expm1(x)-x
        value=(-self.c2*remainder(2*d)-self.c4*remainder(4*d/self.q)
               +self.cb*remainder(2*d/self.q)+self.slope*d)
        if not math.isfinite(value):raise FloatingPointError('Nonfinite radial log density.')
        return value

    def interval(self,current_t,exponential):
        height=self.centered(current_t)-exponential
        if not math.isfinite(height) or height>=0:
            raise FloatingPointError('Invalid log slice height relative to the mode.')
        f=lambda t:self.centered(t)-height
        endpoints=[]
        for sign in (-1,1):
            width=.25
            for _ in range(MAX_EXPANSIONS):
                edge=self.mode+sign*width
                if f(edge)<0:break
                width*=2
            else:raise RuntimeError('Could not bracket a complete radial slice.')
            lo,hi=sorted([edge,self.mode])
            endpoints.append(brentq(f,lo,hi,xtol=XTOL,rtol=RTOL,maxiter=MAXITER))
        lo,hi=endpoints
        if not lo<hi or not lo-XTOL<=current_t<=hi+XTOL:
            raise RuntimeError('Radial slice does not contain current state.')
        residual=max(abs(f(lo)),abs(f(hi)))
        if residual>1e-7*(1+abs(height)):
            raise RuntimeError('Radial slice endpoint residual exceeds tolerance.')
        return lo,hi,height,residual


def radial_step(target,z,rngs):
    radius=np.linalg.norm(z,axis=1)
    if np.any(radius<=0) or not np.isfinite(radius).all():raise ValueError('Invalid white radius.')
    direction=z/radius[:,None]
    d=direction*target.root_scatter
    aa=np.sum(d*d,axis=1)/target.noise_variance
    bb=d@target.observations/target.noise_variance
    out=np.empty_like(z);widths=[];moves=[];residuals=[];evaluations=[]
    for c,rng in enumerate(rngs):
        old=math.log(radius[c])
        cond=RadialConditional(target.n,target.q,float(aa[c]),float(bb[c]),old)
        lo,hi,height,residual=cond.interval(old,float(rng.exponential()))
        new=float(rng.uniform(lo,hi))
        if cond.centered(new)<height-1e-7*(1+abs(height)):
            raise RuntimeError('Proposed radius falls outside its slice.')
        out[c]=direction[c]*math.exp(new)
        widths.append(hi-lo);moves.append(new-old)
        residuals.append(residual);evaluations.append(cond.evaluations)
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite radial output.')
    return out,np.array(widths),np.array(moves),max(residuals),sum(evaluations)


def sample(target,config,fixture_id,observer,max_seconds=900,progress=None,radial_enabled=True):
    if config.chains!=4 or not 1<=config.warmup<=5000 or not 4<=config.draws<=20000:
        raise ValueError('Outside fixed prototype bounds.')
    rngs=[np.random.default_rng(np.random.SeedSequence([config.base_seed,fixture_id,c])) for c in range(4)]
    radial_rngs=[np.random.default_rng(np.random.SeedSequence([config.base_seed,fixture_id,c,777])) for c in range(4)]
    length_rng=np.random.default_rng(np.random.SeedSequence([config.base_seed,fixture_id,999]))
    center=target.gaussian_white_mean()
    z=np.stack([center+3*rng.normal(size=target.n)/np.sqrt(target.mass_diag) for rng in rngs])
    initial=z.copy();value,_=target.potential_gradient(z)
    logeps=np.full(4,np.log(config.initial_step_size));logbar=logeps.copy()
    mu=np.log(10*config.initial_step_size);hbar=np.zeros(4)
    accept_count=np.zeros(4,dtype=int);divergences=np.zeros(4,dtype=int)
    max_delta=np.zeros(4);stagnant=np.zeros(4,dtype=int);nonfinite_proposals=np.zeros(4,dtype=int)
    trace={name:np.zeros((4,config.draws)) for name in
           ['slice_width','radial_log_move','hmc_delta_H','hmc_accepted']}
    started=time.perf_counter();retained=0;completed=True;failure=None
    max_residual=0.;scalar_evaluations=0;radial_failures=0
    for iteration in range(config.warmup+config.draws):
        if time.perf_counter()-started>max_seconds:
            completed=False;failure='TIME_BUDGET';break
        momenta=np.stack([rng.normal(size=target.n) for rng in rngs])*np.sqrt(target.mass_diag)
        steps=int(length_rng.integers(config.min_steps,config.max_steps+1))
        proposed,p_final,v_final=leapfrog(target,z,momenta,np.exp(logeps),steps)
        with np.errstate(over='ignore',invalid='ignore'):
            delta=v_final+.5*np.sum(p_final**2/target.mass_diag,axis=1)
            delta-=value+.5*np.sum(momenta**2/target.mass_diag,axis=1)
        finite=np.isfinite(delta)&np.all(np.isfinite(proposed),axis=1)
        log_accept=np.where(finite,np.minimum(0,-delta),-np.inf)
        accepted=np.array([np.log(rng.uniform()) for rng in rngs])<log_accept
        z=np.where(accepted[:,None],proposed,z);value=np.where(accepted,v_final,value)
        if iteration<config.warmup:
            t=iteration+1;eta=1/(t+10)
            hbar=(1-eta)*hbar+eta*(config.target_accept-np.exp(log_accept))
            logeps=np.clip(mu-np.sqrt(t)/.05*hbar,np.log(1e-5),np.log(config.max_step_size))
            weight=t**-.75;logbar=weight*logeps+(1-weight)*logbar
            if t==config.warmup:logeps=logbar.copy()
        widths=np.zeros(4);moves=np.zeros(4)
        if radial_enabled:
            try:
                z,widths,moves,residual,evaluations=radial_step(target,z,radial_rngs)
                value,_=target.potential_gradient(z)
                if not np.isfinite(value).all():raise FloatingPointError('Nonfinite post-radial potential.')
                max_residual=max(max_residual,residual);scalar_evaluations+=evaluations
            except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
                radial_failures+=1;completed=False;failure=f'{type(exc).__name__}: {exc}';break
        if iteration>=config.warmup:
            observer(z,retained)
            trace['slice_width'][:,retained]=widths;trace['radial_log_move'][:,retained]=moves
            trace['hmc_delta_H'][:,retained]=delta;trace['hmc_accepted'][:,retained]=accepted
            retained+=1;accept_count+=accepted;divergences+=(~finite)|(abs(delta)>100)
            nonfinite_proposals+=~finite
            max_delta=np.maximum(max_delta,np.where(finite,abs(delta),np.inf))
            stagnant+=abs(moves)<=1e-12
        if progress and (iteration+1)%1000==0:
            progress(iteration+1,config.warmup+config.draws,time.perf_counter()-started)
    result={'completed':completed,'failure':failure,'retained_per_chain':retained,
        'runtime_sec':time.perf_counter()-started,'step_sizes':np.exp(logeps).tolist(),
        'acceptance_rate':(accept_count/max(1,retained)).tolist(),
        'divergences_abs_delta_H_gt_100':divergences.tolist(),'max_abs_delta_H':max_delta.tolist(),
        'nonfinite_retained_hmc_proposals':nonfinite_proposals.tolist(),
        'radial_solver_failures':radial_failures,'max_slice_endpoint_log_residual':max_residual,
        'radial_scalar_evaluations':scalar_evaluations,'radial_stagnation_count':stagnant.tolist(),
        'seed_components':[[config.base_seed,fixture_id,c] for c in range(4)],
        'radial_seed_components':[[config.base_seed,fixture_id,c,777] for c in range(4)],
        'trajectory_length_seed':[config.base_seed,fixture_id,999],
        'radial_enabled':radial_enabled,'adaptation_after_warmup':False,'mass_is_fixed_gaussian_reference':True}
    return result,initial,z.copy(),{k:v[:,:retained] for k,v in trace.items()}
