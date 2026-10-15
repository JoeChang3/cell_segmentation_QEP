"""Frozen validation gates, then the two predetermined real tiles if permitted."""
from pathlib import Path
from dataclasses import asdict
import sys,json,time,hashlib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'audits/fast_qep_prototype_20261014'),
    str(ROOT/'audits/fast_qep_radial_diagnosis_20261014'),
    str(ROOT/'experiments/real_data/fast_qep_real_pilot_20261014')]
from py_core.fast_qep_prototype import SpectralPosterior,HMCConfig
from radial_hmc import RadialConditional,radial_step,sample
from stream_hmc import stream_hmc,BlockObserver,pixel_block_summary
from independence_mh import centered_log_weight
from diagnostics import summarize
from run_pilot import prepare,save,digest,load_original


class Observer(BlockObserver):
    def __init__(self,target,draws):
        super().__init__(target,draws)
        self.extra=np.empty((4,draws,2))

    def __call__(self,z,j):
        super().__call__(z,j)
        v=self.target.latent(z)
        self.extra[:,j,0]=np.sum((v/self.target.gaussian_root)**2,axis=1)
        self.extra[:,j,1]=centered_log_weight(self.target,v)

    def diagnostics(self):
        result={name:summarize(self.traces[:,:self.count,j]) for j,name in enumerate(self.names)
                if not (self.target.q==2 and name.startswith('controlled_'))}
        result['prior_quadratic_T']=summarize(self.extra[:,:self.count,0])
        if self.target.q!=2:
            result['target_to_gaussian_log_weight']=summarize(self.extra[:,:self.count,1])
        return result


def checks():
    rng=np.random.default_rng(104501);rows=[]
    for n in (2,20,40000):
        for q in (2.,1.5):
            target=SpectralPosterior(np.linspace(.4,1.6,n),np.linspace(-.8,1.2,n),1.,.35,q)
            z=rng.normal(size=(4,n));rho=np.linalg.norm(z,axis=1);u=z/rho[:,None]
            d=u[0]*target.root_scatter
            cond=RadialConditional(n,q,float(d@d/.35),float(d@target.observations/.35),float(np.log(rho[0])))
            tt=cond.mode+np.array([-.05,0,.05])
            direct=np.array([-float(target.potential_gradient(np.exp(t)*u[0])[0])+n*t for t in tt])
            actual=np.array([cond.centered(t) for t in tt])
            err=float(np.max(abs((direct-direct[1])-(actual-actual[1]))))
            assert err<1e-7
            lo,hi,height,resid=cond.interval(float(np.log(rho[0])),.7)
            assert lo<=np.log(rho[0])<=hi and lo<cond.mode<hi
            proposed,widths,moves,residual,evaluations=radial_step(target,z,
                [np.random.default_rng(np.random.SeedSequence([104501,n,int(q*10),c])) for c in range(4)])
            unit_error=float(np.max(abs(proposed/np.linalg.norm(proposed,axis=1)[:,None]-u)))
            assert unit_error<1e-12 and (widths>0).all()
            rows.append({'n':n,'q':q,'conditional_log_density_error':err,
                         'direction_max_error':unit_error,'slice_endpoint_residual':resid})
    equality=[]
    for q in (2.,1.5):
        target=SpectralPosterior(np.array([.4,1.6]),np.array([1.2,-.8]),1.,.35,q)
        config=HMCConfig(warmup=40,draws=40)
        old=np.empty((4,40,2));new=np.empty_like(old)
        aa,ia,fa=stream_hmc(target,config,519,lambda z,j:old.__setitem__((slice(None),j),z))
        bb,ib,fb,_=sample(target,config,519,lambda z,j:new.__setitem__((slice(None),j),z),radial_enabled=False)
        assert np.array_equal(old,new) and np.array_equal(ia,ib) and np.array_equal(fa,fb)
        for k in ('step_sizes','acceptance_rate','divergences_abs_delta_H_gt_100','max_abs_delta_H'):
            assert aa[k]==bb[k]
        equality.append({'q':q,'all_draws_identical':True})
    return {'conditional_and_slice_checks':rows,'disabled_radial_hmc_equivalence':equality,'passed':True}


def gates(diag,sampler):
    return {'convergence':all(d['rhat_rank_folded_split']<1.01 and d['ess_bulk']>400 and d['ess_tail']>400
                             for d in diag.values()),
            'sampler_numerics':sampler['completed'] and sampler['radial_solver_failures']==0
                and sum(sampler['divergences_abs_delta_H_gt_100'])==0
                and sum(sampler['nonfinite_retained_hmc_proposals'])==0
                and min(sampler['acceptance_rate'])>0}


def diag_summary(diag):
    return {'max_rhat':max(d['rhat_rank_folded_split'] for d in diag.values()),
            'min_bulk_ess':min(d['ess_bulk'] for d in diag.values()),
            'min_tail_ess':min(d['ess_tail'] for d in diag.values())}


def small(out):
    refs=json.loads((ROOT/'audits/fast_qep_prototype_20261014/validation/deterministic_checks.json').read_text())['quadrature']
    results=[]
    for fid,q in [(501,2.),(502,1.5)]:
        target=SpectralPosterior(np.array([.4,1.6]),np.array([1.2,-.8]),1.,.35,q)
        obs=Observer(target,20000);z=np.empty((4,20000,2))
        def record(zz,j):obs(zz,j);z[:,j]=zz
        print(f'Start 2D q={q}',flush=True)
        sampler,initial,final,traces=sample(target,HMCConfig(warmup=2000,draws=20000),fid,record,
            progress=lambda i,n,s:print(f'2D q={q}: {i}/{n}, {s:.1f}s',flush=True) if i%5000==0 else None)
        z=z[:,:obs.count]
        np.savez_compressed(out/f'2d_q{q:g}_chains.npz',white_draws=z,trace=obs.traces[:,:obs.count],
            trace_names=np.asarray(obs.names),extra_trace=obs.extra[:,:obs.count],initial_z=initial,final_z=final,**traces)
        result={'q':q,'sampler':sampler,'passed':False}
        if sampler['completed']:
            diag=obs.diagnostics();gate=gates(diag,sampler)
            v=target.latent(z);cv=target.score_corrected_latent(z);truth=refs[str(q)];comparisons=[]
            def compare(name,values,reference,multiple,absolute=False):
                d=summarize(values);error=abs(d['mean']-reference)
                passed=error<=multiple*d['mcse'] and (not absolute or (error<=.001 and d['mcse']<=.001))
                comparisons.append({'quantity':name,'reference':reference,'estimate':d['mean'],
                    'abs_error':error,'mcse':d['mcse'],'mcse_multiple':multiple,'passed':passed})
            for j in range(2):
                compare(f'raw_mean_{j}',v[:,:,j],truth['mean'][j],5 if q==2 else 3)
                compare(f'second_moment_{j}',v[:,:,j]**2,truth['second_moments'][j],5)
                if q!=2:compare(f'controlled_mean_{j}',cv[:,:,j],truth['mean'][j],3,True)
            compare('cross_moment',v[:,:,0]*v[:,:,1],truth['cross_moment'],5)
            if q==2:
                m=target.gaussian_mean();var=target.gaussian_root**2/target.mass_diag
                for j in range(2):compare(f'centered_variance_{j}',(v[:,:,j]-m[j])**2,var[j],5)
                compare('centered_cross', (v[:,:,0]-m[0])*(v[:,:,1]-m[1]),0.,5)
                mz=target.gaussian_white_mean();vz=1/target.mass_diag
                er2=float(np.sum(mz*mz+vz));vr2=float(np.sum(2*vz*vz+4*mz*mz*vz))
                r2=np.sum(z*z,axis=-1)
                compare('white_radius_squared',r2,er2,5)
                compare('white_radius_fourth',r2*r2,er2*er2+vr2,5)
            gate['reference_moments']=all(r['passed'] for r in comparisons)
            result.update({'gates':gate,'passed':all(gate.values()),'diagnostics':diag,
                           'comparisons':comparisons,**diag_summary(diag)})
        save(out/f'2d_q{q:g}_summary.json',result);results.append(result)
        print(f'2D q={q}: passed={result["passed"]}',flush=True)
    return results


def real(out):
    results=[];original=load_original()
    for fid,(ds,grid,first_ref,_,meta) in enumerate(prepare(),503):
        h,w=grid.shape
        image=np.asarray(Image.open(ROOT/meta['input_path']),dtype=float)
        if image.ndim==3:image=image[...,0]
        tile=image[h:2*h,:w]/255
        ref=grid.gaussian_reference(tile,first_ref.nugget_ratio)
        old=json.loads((ROOT/f'results/fast_qep_second_tiles_20261014/{ds}_input.json').read_text())
        assert hashlib.sha256(tile.tobytes()).hexdigest()==old['tile_normalized_float64_sha256']
        assert meta['input_sha256']==old['input_sha256']
        assert abs(ref.mean-old['tile_gls_mean'])<1e-14 and abs(ref.signal_variance-old['tile_signal_variance'])<1e-14
        p=meta['r_parameters'];sp=original['SharedParams'](float(p['beta1']),float(p['beta2']),float(p['nugget_nu']),
            h,w,'same fixed tile2 R parameters',0.,'none',0,0.)
        q2_error=float(np.max(abs(original['reconstruct_tile'](tile,sp).pred_mean-ref.prediction)))
        assert q2_error<1e-10
        metadata={**old,'initialization':'Gaussian-reference white mean plus 3 posterior SD normal vectors',
                  'q2_vs_original_python_max_error':q2_error}
        save(out/f'{ds}_input.json',metadata)
        target=grid.posterior(ref,1.5);obs=Observer(target,4000)
        print(f'Start {ds} original tile 2, n={grid.n}',flush=True)
        sampler,initial,final,traces=sample(target,HMCConfig(warmup=2000,draws=4000),fid,obs,
            progress=lambda i,n,s:print(f'{ds}: {i}/{n}, {s:.1f}s',flush=True))
        post_started=time.perf_counter()
        raw,cv=obs.complete_blocks()
        np.savez_compressed(out/f'{ds}_chains.npz',trace=obs.traces[:,:obs.count],trace_names=np.asarray(obs.names),
            spectral_indices=obs.indices,extra_trace=obs.extra[:,:obs.count],
            raw_spectral_blocks=raw,controlled_spectral_blocks=cv,initial_z=initial,final_z=final,**traces)
        result={'input':metadata,'sampler':sampler,'passed':False}
        if sampler['completed']:
            diag=obs.diagnostics();gate=gates(diag,sampler)
            cvs=pixel_block_summary(grid,cv);raws=pixel_block_summary(grid,raw)
            mean=(ref.mean+cvs['mean'])*255;mcse=cvs['mcse']*255;q2=ref.prediction*255;delta=mean-q2
            gate['pixel_mcse']=np.quantile(mcse,.99)<=.1 and mcse.max()<=.5
            result.update({'passed':all(gate.values()),'gates':gate,**diag_summary(diag),
                'mcse_p99':float(np.quantile(mcse,.99)),'mcse_max':float(mcse.max()),
                'mean_absolute_q_change':float(np.mean(abs(delta))),'max_absolute_q_change':float(np.max(abs(delta))),
                'fraction_q_change_exceeds_3_mcse':float(np.mean(abs(delta)>3*mcse)),
                'diagnostics':diag})
            np.savez_compressed(out/f'{ds}_estimates.npz',raw=tile*255,q2=q2,q15_controlled=mean,
                q15_raw=(ref.mean+raws['mean'])*255,q15_mcse=mcse,raw_mcse=raws['mcse']*255,
                controlled_chain_means=(ref.mean+cvs['chain_means'])*255,
                raw_chain_means=(ref.mean+raws['chain_means'])*255,radius_traces=obs.traces[:,:,0])
            print(f'{ds}: passed={result["passed"]}, max Rhat={result["max_rhat"]:.6f}, min ESS={result["min_bulk_ess"]:.1f}/{result["min_tail_ess"]:.1f}',flush=True)
        result['postprocessing_runtime_sec']=time.perf_counter()-post_started
        save(out/f'{ds}_summary.json',result);results.append(result)
        del obs,raw,cv
    return results


def main():
    out=ROOT/'results/fast_qep_radial_hmc_20261014'
    if out.exists():raise RuntimeError('Refusing overwrite/seed retry.')
    out.mkdir()
    previous=json.loads((ROOT/'audits/fast_qep_paper_alignment_20261014/formula_checks.json').read_text())['preexisting_result_sha256']
    assert all(digest(ROOT/p)==v for p,v in previous.items())
    files=[HERE/'PROTOCOL.md',HERE/'radial_hmc.py',Path(__file__),ROOT/'py_core/fast_qep_prototype.py',
        ROOT/'experiments/real_data/fast_qep_real_pilot_20261014/stream_hmc.py',
        ROOT/'experiments/real_data/fast_qep_real_pilot_20261014/run_pilot.py',
        ROOT/'audits/fast_qep_prototype_20261014/diagnostics.py',
        ROOT/'audits/fast_qep_radial_diagnosis_20261014/independence_mh.py',
        ROOT/'audits/fast_qep_paper_alignment_20261014/MODEL_CONTRACT.md',
        ROOT/'audits/fast_qep_paper_alignment_20261014/NEXT_INFERENCE_PLAN.md']
    save(out/'manifest.json',{'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files},
        'previous_result_sha256':previous,'python':sys.version,'numpy':np.__version__,'threads':1,
        'hmc_config_small':asdict(HMCConfig(warmup=2000,draws=20000)),
        'hmc_config_real':asdict(HMCConfig(warmup=2000,draws=4000))})
    completion={'full_image_or_segmentation_run':False,'new_model_or_q_search':False,'real_image_run':False}
    try:
        save(out/'deterministic_checks.json',checks());print('Deterministic checks PASS.',flush=True)
        ss=small(out);completion['small_pass']={str(r['q']):r['passed'] for r in ss}
        if all(r['passed'] for r in ss):
            completion['real_image_run']=True;rr=real(out)
            completion['real_pass']={r['input']['dataset']:r['passed'] for r in rr}
            completion['status']='PASS_TWO_LOCAL_DIAGNOSTICS' if all(r['passed'] for r in rr) else 'STOP_REAL_NUMERICAL_GATE'
        else:completion['status']='STOP_SMALL_REFERENCE_GATE'
    except Exception as exc:
        completion.update({'status':'STOP_IMPLEMENTATION_OR_RUNTIME_ERROR','exception':repr(exc)})
        raise
    finally:
        unchanged=all(digest(ROOT/p)==v for p,v in previous.items())
        save(out/'integrity.json',{'old_results_unchanged':unchanged,'checked_files':len(previous)})
        completion['old_results_unchanged']=unchanged;save(out/'completion.json',completion)
        assert unchanged


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
