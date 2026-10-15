"""Small mathematical/inference checks only; no real-image fitting or sampling."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import eigh
import runtime
from hyperfit_core import state, EvidenceScores, ScoreGeometry, gaussian_evidence, nuisance_update
from py_core.fast_qep_prototype import KroneckerGrid, SpectralPosterior, HMCConfig
import math_reference as ref


def two_dimensional_fixture(q):
    theta = np.array([.1, np.log(1.3), np.log(.4/1.3), np.log(1.4), 0.])
    y = np.array([.7, -.4])
    k, _ = ref.kernel(theta[:4], (2,))
    lam, u = eigh(k)
    # Test-only 2x1 algebra fixture. Production uses full original 2D tiles.
    grid = KroneckerGrid((2,1), 1.4, 1., u, np.eye(1), lam)
    target = SpectralPosterior(lam, grid.forward(y[:,None]-.1), 1.3, .4, q)
    return theta, y, grid, target, EvidenceScores(target, ScoreGeometry(grid))


def independent_integral(q):
    theta, y, grid, target, scores = two_dimensional_fixture(q)
    phi = (np.arange(256)+.5)*2*np.pi/256
    k, _ = ref.kernel(theta[:4], (2,))
    direction = np.column_stack([np.cos(phi),np.sin(phi)])@np.linalg.cholesky(k).T
    joint = runtime.frozen_joint()
    def integrand(t):
        v = np.sqrt(target.signal_variance/target.a)*(2*t)**(1/q)*direction
        spectral = v@grid.row_vectors
        z = joint.to_white(target, spectral)
        raw, cv = scores(z)
        w = np.exp(-t-np.sum((y[None,:]-theta[0]-v)**2,axis=1)/(2*.4))/(2*np.pi*.4)
        return np.mean(w[:,None]*np.column_stack([np.ones(len(w)),raw,cv]),axis=0)
    vals, err = quad_vec(integrand,0,np.inf,epsabs=2e-11,epsrel=2e-10)
    return vals[1:6]/vals[0], vals[6:11]/vals[0], float(err)


def checks(out, run_mcmc=True):
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result={'kind':'local validation only', 'passed':False, 'tests':{}}
    tests=result['tests']
    from pilot import fit_gp, observer_class
    prior=ref.audit()
    assert prior['passed']
    (out/'previous_math_checks.json').write_text(json.dumps(prior,indent=2)+'\n')
    tests['previous_15_checks']=True
    rng=np.random.default_rng(2026101501)
    shape=(3,4)
    theta=np.array([.15,np.log(1.2),np.log(.45),np.log(1.7),np.log(2.3)])
    tile=rng.normal(size=shape)
    for q in (2.,1.5):
        grid,target,geo=state(tile,theta,q)
        zz=rng.normal(size=(12,grid.n))
        raw,cv=EvidenceScores(target,geo)(zz)
        errors=[]
        for c,z in enumerate(zz):
            physical=grid.inverse(target.latent(z)).ravel(order='F')
            _,dense=ref.log_joint_and_score(physical,tile.ravel(order='F'),theta,q,shape)
            errors.append(float(np.max(abs(raw[c]-dense))))
        assert max(errors)<1e-8
        tests[f'score_vs_independent_dense_q{q}']={'max_abs_error':max(errors)}
        if q==2:
            ll,g=gaussian_evidence(tile,theta,grid,geo)
            ll2,g2,_,_=ref.gaussian_evidence(tile.ravel(order='F'),theta,shape)
            err=float(max(abs(ll-ll2),np.max(abs(g-g2)),np.max(abs(cv-g))))
            assert err<1e-8
            tests['q2_controlled_score_is_analytic_for_every_draw']={'max_abs_error':err}
    for q in (2.,1.5):
        raw,cv,error=independent_integral(q)
        truth=ref.quadrature(np.array([.7,-.4]),two_dimensional_fixture(q)[0][:4],q)
        delta=float(max(np.max(abs(raw[:4]-truth['score'])),np.max(abs(cv[:4]-truth['score']))))
        assert delta<1e-8
        tests[f'raw_and_controlled_score_integrals_q{q}']={'max_abs_error':delta,'quadrature_error_bound':error}
        theta,y,grid,target,es=two_dimensional_fixture(q)
        nxt=nuisance_update(grid,target,y[:,None],grid.forward(truth['latent_mean'][:,None]),
                            float(truth['latent_second'].sum()),truth['radial_moment'])
        expected=ref.ecm_nuisance_update(y,theta[:4],q,truth)
        err=float(np.max(abs(nxt[:4]-expected)))
        assert err<1e-10
        tests[f'ECM_integration_q{q}']={'max_abs_error':err}
    if run_mcmc:
        joint=runtime.frozen_joint()
        tests['unchanged_sampler_deterministic_checks']=joint.checks()
        for q,fid in [(2.,7301),(1.5,7302)]:
            theta,y,grid,target,es=two_dimensional_fixture(q)
            cfg=HMCConfig(warmup=2000,draws=4000,base_seed=20261015)
            z=np.empty((4,cfg.draws,2))
            sampler,_,_,_=joint.sample(target,cfg,fid,lambda zz,j:z.__setitem__((slice(None),j),zz),max_seconds=120)
            assert sampler['completed'] and sampler['retained_per_chain']==cfg.draws
            assert not sum(sampler['divergences_abs_delta_H_gt_100'])
            assert sampler['global_numerical_failures']==0
            raw,cv=es(z)
            observer=observer_class()(target,grid,cfg.draws)
            saved=z.copy()
            for j in range(cfg.draws):observer(z[:,j],j)
            assert np.array_equal(saved,z)
            assert np.array_equal(observer.score_raw,raw[:,9::10])
            assert np.array_equal(observer.score_cv,cv[:,9::10])
            tests[f'observer_stride_and_no_mutation_q{q}']=True
            truth=ref.quadrature(y,theta[:4],q)['score']
            rows=[]
            for j in range(4):
                if q==2:
                    err=float(np.max(abs(cv[:,:,j]-truth[j])))
                    assert err<1e-9
                    rows.append({'coordinate':j,'max_abs_error':err})
                else:
                    d=joint.summarize(cv[:,:,j]); d_raw=joint.summarize(raw[:,:,j])
                    err=abs(d['mean']-truth[j])
                    assert err<=5*d['mcse']+1e-9
                    assert d['rhat_rank_folded_split']<1.01 and min(d['ess_bulk'],d['ess_tail'])>400
                    rows.append({'coordinate':j,'abs_error':err,'controlled':d,'raw':d_raw})
            tests[f'unchanged_sampler_score_expectation_q{q}']={'sampler':sampler,'coordinates':rows}
    # Fixed code exercise, not a superiority benchmark or real-image fit.
    xr,xc=np.meshgrid(np.linspace(0,1,8),np.linspace(0,1,7),indexing='ij')
    toy=.5+.1*np.sin(2*np.pi*xr)*np.cos(2*np.pi*xc)+.01*rng.normal(size=xr.shape)
    gg=KroneckerGrid.matern52(toy.shape,8.,9.)
    rr=gg.gaussian_reference(toy,.2)
    archive=np.array([rr.mean,np.log(rr.signal_variance),np.log(.2),np.log(8.),np.log(9.)])
    fitted=fit_gp(toy,archive,maximum_seconds=30)
    assert fitted['stationarity_pass']
    old_ll,_=gaussian_evidence(toy,archive,gg)
    best_ll=max(f['log_evidence'] for f in fitted['fits'])
    assert best_ll>=old_ll-1e-8
    tests['two_start_GP_optimizer_fixture']={'stationarity_pass':True,'initial_log_evidence':old_ll,
                                              'final_log_evidence':best_ll,'fitting':fitted}
    # Original-sized matrices with artificial values: validates numerical domain
    # and implementation cost without training or sampling a real image.
    large=[]
    for shape in [(282,240),(200,200)]:
        rr,cc=np.meshgrid(np.linspace(0,1,shape[0]),np.linspace(0,1,shape[1]),indexing='ij')
        toy=.3+.02*np.sin(4*rr)*np.cos(3*cc)
        for beta in (5.,20.):
            theta=np.array([.3,np.log(.01),np.log(.1),np.log(beta),np.log(beta)])
            grid,target,geo=state(toy,theta,2.)
            _,gg=gaussian_evidence(toy,theta,grid,geo)
            zz=rng.normal(size=(4,grid.n))
            started_score=time.perf_counter()
            _,cv=EvidenceScores(target,geo)(zz)
            seconds=time.perf_counter()-started_score
            err=float(np.max(abs(cv-gg)))
            assert err<1e-7
            fd=[]
            for j in (3,4):
                step=np.zeros(5);step[j]=1e-5
                fd.append((gaussian_evidence(toy,theta+step)[0]-gaussian_evidence(toy,theta-step)[0])/2e-5)
            fd_err=float(np.max(abs(np.array(fd)-gg[3:5]))/grid.n)
            assert fd_err<1e-8
            large.append({'shape':shape,'beta':beta,'q2_pointwise_error':err,
                          'kernel_FD_error_per_pixel':fd_err,'four_chain_score_eval_seconds':seconds,
                          'condition':geo.condition,'real_image_used':False})
    tests['original_sized_matrix_checks']=large
    result.update(passed=True,runtime_s=time.perf_counter()-started,real_image_fitted=False,real_image_sampled=False)
    (out/'integration_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('INTEGRATION CHECKS PASS',flush=True)
    return result


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):checks(Path(__file__).parent/'validation')
