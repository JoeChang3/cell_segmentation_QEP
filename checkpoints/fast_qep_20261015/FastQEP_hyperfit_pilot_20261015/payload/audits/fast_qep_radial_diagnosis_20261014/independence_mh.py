"""Gaussian-posterior independent proposals; exact radial MH correction."""
import time
import numpy as np


def centered_log_weight(target,v):
    if target.q==2: return np.zeros(np.shape(v)[:-1])
    t=np.sum((v/target.gaussian_root)**2,axis=-1)
    if np.any(t<=0) or not np.isfinite(t).all():
        raise ValueError('Nonpositive/nonfinite prior quadratic form.')
    logratio=np.log1p((t-target.n)/target.n)
    return (target.n*(target.q-2)/4*logratio
            -.5*(target.a*target.n)**(target.q/2)*np.expm1(target.q/2*logratio)
            +.5*(t-target.n))


def to_white(target,v):
    x=v/target.root_scatter
    radius=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(radius<=0): raise ValueError('Zero-vector transform not used by this diagnostic.')
    return x*radius**(target.q/2-1)


def sample(target,warmup,draws,fixture_id,observer,initial_white=None,max_seconds=900,progress=None):
    rngs=[np.random.default_rng(np.random.SeedSequence([20261014,fixture_id,c])) for c in range(4)]
    mean=target.gaussian_mean()
    d=target.gaussian_root**2
    sd=np.sqrt(d*target.noise_variance/(d+target.noise_variance))
    v=(np.stack([mean+sd*rng.normal(size=target.n) for rng in rngs]) if initial_white is None
       else target.latent(initial_white))
    initial=v.copy();logw=centered_log_weight(target,v)
    accepted_count=np.zeros(4,dtype=int);rejections=np.zeros(4,dtype=int)
    max_rejections=np.zeros(4,dtype=int)
    trace=np.empty((4,draws));proposal_trace=np.empty((4,draws))
    started=time.perf_counter();retained=0;completed=True
    for iteration in range(warmup+draws):
        if time.perf_counter()-started>max_seconds:
            completed=False;break
        proposal=np.stack([mean+sd*rng.normal(size=target.n) for rng in rngs])
        proposed_logw=centered_log_weight(target,proposal)
        if not np.isfinite(proposed_logw).all(): raise ValueError('Nonfinite MH weight.')
        accepted=np.array([np.log(rng.uniform()) for rng in rngs])<np.minimum(0,proposed_logw-logw)
        v=np.where(accepted[:,None],proposal,v)
        logw=np.where(accepted,proposed_logw,logw)
        if iteration>=warmup:
            observer(to_white(target,v),retained)
            trace[:,retained]=logw;proposal_trace[:,retained]=proposed_logw
            retained+=1;accepted_count+=accepted
            rejections=np.where(accepted,0,rejections+1)
            max_rejections=np.maximum(max_rejections,rejections)
        if progress and (iteration+1)%1000==0:
            progress(iteration+1,warmup+draws,time.perf_counter()-started)
    return {'completed':completed,'retained_per_chain':retained,
            'runtime_sec':time.perf_counter()-started,
            'acceptance_rate':(accepted_count/max(1,retained)).tolist(),
            'max_consecutive_retained_rejections':max_rejections.tolist(),
            'seed_components':[[20261014,fixture_id,c] for c in range(4)],
            'nonfinite_weights':0},initial,v.copy(),trace[:,:retained],proposal_trace[:,:retained]
