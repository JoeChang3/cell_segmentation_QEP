"""Bounded first-tile score pilot, not full QEP optimization or segmentation."""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from PIL import Image
import runtime
from hyperfit_core import (NAMES, EvidenceScores, ScoreGeometry, gaussian_evidence,
                           state, nuisance_update)
from py_core.fast_qep_prototype import KroneckerGrid, HMCConfig

ROOT=runtime.ROOT
CFG=HMCConfig(warmup=2000,draws=4000,base_seed=20261015)
SCORE_STRIDE=10


def dump(path, value):
    runtime.frozen_joint().save(path,value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs():
    rows=list(csv.DictReader((ROOT/'results/real_cellseg_round3_thresholding_20260916/image_manifest.csv').open()))
    work=[]
    for ds,expected_shape in [('nuclei',(282,240)),('whole_cell',(200,200))]:
        row=next(r for r in rows if r['dataset']==ds and r['role']=='development')
        path=ROOT/row['path_image']
        # Do not decode or score ground-truth masks in this pilot.
        raw=np.asarray(Image.open(path),dtype=np.float64)
        if raw.ndim==3:raw=raw[...,0]
        p=next(csv.DictReader((ROOT/f'audits/parity/real_{ds}_r_params.csv').open()))
        shape=(int(p['n1']),int(p['n2']))
        if shape!=expected_shape or raw.shape!=(int(row['height']),int(row['width'])):
            raise RuntimeError('Unexpected original image/tile geometry')
        tile=raw[:shape[0],:shape[1]]/255
        grid=KroneckerGrid.matern52(shape,float(p['beta1']),float(p['beta2']))
        ref=grid.gaussian_reference(tile,float(p['nugget_nu']))
        with np.load(ROOT/f'experiments/real_data/fast_qep_e2e_20261015/anchors/{ds}/fastgp_reconstruction.npz') as f:
            error=float(np.max(abs(ref.prediction-f['predmean'][:shape[0],:shape[1]]/255)))
        if max(error,abs(ref.mean-float(p['theta_hat'])),
               abs(ref.signal_variance-float(p['sigma2_hat'])))>=1e-10:
            raise RuntimeError('Archived q2/R anchor mismatch')
        theta=np.array([ref.mean,np.log(ref.signal_variance),np.log(ref.nugget_ratio),
                        np.log(grid.beta_rows),np.log(grid.beta_cols)])
        work.append({'dataset':ds,'tile':tile,'theta':theta,'input':{
            'image_path':row['path_image'],'image_sha256':digest(path),
            'tile_float64_sha256':hashlib.sha256(tile.tobytes()).hexdigest(),
            'shape':shape,'tile_number':1,'origin':[0,0],
            'original_r_parameters':p,'q2_vs_archived_R_max_error':error}})
    return work


def bounds_for(tile, archive):
    # Pilot numerical domain: same domain for future GP/QEP fitting; boundary
    # hits are reported, not silently interpreted as unrestricted optima.
    upper=np.maximum(np.exp(archive[3:5]),20*(np.array(tile.shape)-1))
    return [(np.log(5.),np.log(upper[0])),(np.log(5.),np.log(upper[1])),
            (np.log(1e-6),np.log(10.))]


def fit_gp(tile, archive, maximum_seconds=300):
    bounds=bounds_for(tile,archive)
    starts=[archive[[3,4,2]],np.log([20.,20.,.1])]
    records=[]
    overall=time.perf_counter()
    for label,start in zip(['archived_R','common_nondegenerate'],starts):
        history=[]
        def objective(x):
            if time.perf_counter()-overall>maximum_seconds:
                raise RuntimeError('GP_FIT_TIME_BUDGET')
            grid=KroneckerGrid.matern52(tile.shape,*np.exp(x[:2]))
            ref=grid.gaussian_reference(tile,np.exp(x[2]))
            theta=np.array([ref.mean,np.log(ref.signal_variance),x[2],x[0],x[1]])
            ll,g=gaussian_evidence(tile,theta,grid)
            history.append({'x':x.tolist(),'log_evidence':ll})
            return -ll/grid.n,-g[[3,4,2]]/grid.n
        opt=minimize(objective,start,method='L-BFGS-B',jac=True,bounds=bounds,
                     options={'maxiter':200,'maxls':20,'ftol':1e-12,'gtol':1e-8})
        grid=KroneckerGrid.matern52(tile.shape,*np.exp(opt.x[:2]))
        ref=grid.gaussian_reference(tile,np.exp(opt.x[2]))
        theta=np.array([ref.mean,np.log(ref.signal_variance),opt.x[2],opt.x[0],opt.x[1]])
        ll,g=gaussian_evidence(tile,theta,grid)
        # Projected gradient of the minimized negative evidence per pixel.
        pg=-g[[3,4,2]]/grid.n
        hits=[]
        for j,(lo,hi) in enumerate(bounds):
            low=opt.x[j]-lo<=1e-7; high=hi-opt.x[j]<=1e-7
            hits.append('lower' if low else 'upper' if high else None)
            if low and pg[j]>0 or high and pg[j]<0:pg[j]=0
        records.append({'start':label,'theta':theta.tolist(),'log_evidence':ll,
                        'success':bool(opt.success),'message':str(opt.message),'iterations':opt.nit,
                        'function_evaluations':opt.nfev,'gradient':g.tolist(),
                        'projected_gradient_per_pixel_max':float(np.max(abs(pg))),
                        'boundary_hits':hits,'history':history})
    best=max(records,key=lambda r:r['log_evidence'])
    return {'theta':best['theta'],'selected_start':best['start'],'fits':records,
            'bounds_log_beta_rows_cols_nu':bounds,'runtime_sec':time.perf_counter()-overall,
            'stationarity_pass':best['success'] and best['projected_gradient_per_pixel_max']<=1e-6,
            'scope':'Best of two starts within declared numerical domain, not a global optimum.'}


def observer_class():
    joint=runtime.frozen_joint()
    class ScoreObserver(joint.Observer):
        def __init__(self,target,grid,draws):
            super().__init__(target,draws)
            self.evidence_scores=EvidenceScores(target,ScoreGeometry(grid))
            self.score_raw=np.zeros((4,draws//SCORE_STRIDE,5))
            self.score_cv=np.zeros_like(self.score_raw)
            self.score_count=0
            self.score_seconds=0.
        def __call__(self,z,j):
            super().__call__(z,j)
            if (j+1)%SCORE_STRIDE==0:
                t=time.perf_counter()
                raw,cv=self.evidence_scores(z)
                self.score_raw[:,self.score_count]=raw
                self.score_cv[:,self.score_count]=cv
                self.score_count+=1
                self.score_seconds+=time.perf_counter()-t
    return ScoreObserver


def score_summary(obs,tile,gp_score):
    joint=runtime.frozen_joint()
    scale=np.array([float(tile.std()),1.,1.,1.,1.])/tile.size
    rows=[]
    for j,name in enumerate(NAMES):
        raw=joint.summarize(obs.score_raw[:,:obs.score_count,j])
        controlled=joint.summarize(obs.score_cv[:,:obs.score_count,j])
        difference=controlled['mean']-gp_score[j]
        threshold=max(3*controlled['mcse'],1e-8/scale[j])
        rows.append({'name':name,'raw':raw,'controlled':controlled,
                     'gp_analytic_score':float(gp_score[j]),'qep_minus_gp':float(difference),
                     'score_scale':float(scale[j]),
                     'scaled_qep_score':float(controlled['mean']*scale[j]),
                     'scaled_mcse':float(controlled['mcse']*scale[j]),
                     'descriptive_q_difference_resolved':bool(abs(difference)>threshold),
                     'descriptive_qep_score_resolved':bool(abs(controlled['mean'])>threshold),
                     'interpretation':'3 MCSE descriptive screen, not a multiple-testing significance claim.'})
    return rows


def run_target(work,label,theta,out,fixture_id,remaining_seconds):
    joint=runtime.frozen_joint()
    folder=out/work['dataset']/label
    folder.mkdir(parents=True)
    tile=work['tile']
    grid,target,geo=state(tile,theta,1.5)
    ll,gp_score=gaussian_evidence(tile,theta,grid,geo)
    obs=observer_class()(target,grid,CFG.draws)
    print(f"START {work['dataset']}/{label}: original tile {grid.shape}, q=1.5",flush=True)
    sampler,initial,final,traces=joint.sample(target,CFG,fixture_id,obs,
        max_seconds=min(900,remaining_seconds),
        progress=lambda i,n,s:print(f"{work['dataset']}/{label}: {i}/{n}, {s:.1f}s",flush=True))
    record={'dataset':work['dataset'],'parameter_source':label,'theta':theta.tolist(),
            'sampler':sampler,'score_stride':SCORE_STRIDE,'score_count_per_chain':obs.score_count,
            'score_evaluation_seconds':obs.score_seconds,'passed':False,
            'kernel_condition':geo.condition,
            'gp_log_evidence':ll,'gp_analytic_score':gp_score.tolist()}
    # Preserve diagnostic data even on a time/numerical failure. No automatic retry.
    np.savez_compressed(folder/'score_traces.npz',raw=obs.score_raw[:,:obs.score_count],
                        controlled=obs.score_cv[:,:obs.score_count],names=np.asarray(NAMES),
                        global_traces=obs.traces[:,:obs.count,:4],initial_z=initial,final_z=final,
                        **traces)
    if not sampler['completed'] or sampler['retained_per_chain']!=CFG.draws:
        dump(folder/'summary.json',record)
        return record
    diagnostics=obs.diagnostics()
    gates=joint.gates(diagnostics,sampler,1.5)
    rows=score_summary(obs,tile,gp_score)
    gates['score_convergence']=all(r['controlled']['rhat_rank_folded_split']<1.01 and
        r['controlled']['ess_bulk']>400 and r['controlled']['ess_tail']>400 for r in rows)
    raw,cv=obs.complete_blocks()
    cvs=joint.pixel_block_summary(grid,cv)
    raw_spectral_mean=raw.mean(axis=(0,1))
    mcse=cvs['mcse']*255
    gates['pixel_mcse']=bool(np.quantile(mcse,.99)<=.1 and mcse.max()<=.5)
    # Raw moments only. The ECM suggestion is logged, NEVER applied as a fit.
    norm2=float(obs.traces[:,:obs.count,3].mean())
    qpower=float(np.mean((target.signal_variance*obs.extra[:,:obs.count,0])**(target.q/2)))
    suggested=nuisance_update(grid,target,tile,raw_spectral_mean,norm2,qpower)
    qmean=(theta[0]+cvs['mean'])*255
    gpmean=(theta[0]+grid.inverse(target.gaussian_mean()))*255
    np.savez_compressed(folder/'estimates.npz',qep_mean=qmean,qep_mcse=mcse,gp_same_theta=gpmean,
                        qep_chain_means=(theta[0]+cvs['chain_means'])*255)
    record.update(passed=all(gates.values()),gates=gates,scores=rows,diagnostics=diagnostics,
                  nuisance_ECM_suggestion_not_applied=suggested.tolist(),
                  pixel_mcse_p99=float(np.quantile(mcse,.99)),pixel_mcse_max=float(mcse.max()),
                  qep_gp_mean_absolute_difference=float(np.mean(abs(qmean-gpmean))),
                  maximum_rhat=max(d['rhat_rank_folded_split'] for d in diagnostics.values()))
    dump(folder/'summary.json',record)
    print(f"{work['dataset']}/{label}: {'PASS' if record['passed'] else 'STOP: GATE FAILURE'}",flush=True)
    return record


def run_pilot(out,work):
    start=time.perf_counter()
    records=[]
    for dataset_index,item in enumerate(work):
        folder=out/item['dataset'];folder.mkdir()
        dump(folder/'input.json',item['input'])
        print(f"Fitting analytic GP control: {item['dataset']} (two fixed starts)",flush=True)
        fits=fit_gp(item['tile'],item['theta'])
        dump(folder/'gp_refit.json',fits)
        if not fits['stationarity_pass']:
            raise RuntimeError('GP_STATIONARITY_GATE: inspect gp_refit.json; no QEP sampling')
        for index,(label,theta) in enumerate([('archived_R',item['theta']),('refitted_GP',np.array(fits['theta']))]):
            remaining=7200-(time.perf_counter()-start)
            if remaining<=0:raise RuntimeError('PILOT_TIME_BUDGET')
            record=run_target(item,label,theta,out,8101+dataset_index*10+index,remaining)
            records.append(record)
            dump(out/'pilot_summary.json',{'status':'IN_PROGRESS','targets':records})
            if not record['passed']:raise RuntimeError('SAMPLER_OR_SCORE_GATE: no automatic extension or seed retry')
    summary={'status':'COMPLETED','targets':records,'runtime_s':time.perf_counter()-start,
             'qep_hyperparameters_optimized':False,'segmentation_run':False,
             'question':'Is there a resolved QEP evidence-gradient difference at archived/refitted GP parameters?',
             'interpretation':'Positive scores suggest a fitting direction, not segmentation improvement; unresolved scores do not rule out every optimum.'}
    dump(out/'pilot_summary.json',summary)
    lines=['FIRST-TILE HYPERPARAMETER SCORE PILOT','',
           'No QEP parameter optimization, whole-image reconstruction or segmentation was run.','']
    for r in records:
        lines.append(f"{r['dataset']}/{r['parameter_source']}: gates PASS")
        for s in r['scores']:
            lines.append(f"  {s['name']}: scaled score {s['scaled_qep_score']:.6g}, MCSE {s['scaled_mcse']:.3g}; QEP-GP resolved={s['descriptive_q_difference_resolved']}")
    lines+=['','These are descriptive MC screens. Inspect GP boundary/stationarity diagnostics before interpreting them.']
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('COMPLETED: inspect REPORT.txt and pilot_summary.json',flush=True)
