"""Fixed first-tile real-data feasibility test; no parameter search."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile

import numpy as np
import scipy
from PIL import Image
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT/'audits/fast_qep_prototype_20261014'
sys.path[:0] = [str(ROOT),str(HERE),str(AUDIT)]
from py_core.fast_qep_prototype import KroneckerGrid,HMCConfig
from stream_hmc import stream_hmc,BlockObserver,pixel_block_summary,verify_stream_equivalence
from diagnostics import summarize


def digest(path,algorithm='sha256'):
    return hashlib.new(algorithm,Path(path).read_bytes()).hexdigest()


def save(path,data):
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,np.ndarray): return clean(x.tolist())
        if isinstance(x,(np.floating,float)):
            return float(x) if np.isfinite(x) else str(x)
        if isinstance(x,(np.integer,np.bool_)): return x.item()
        return x
    path.write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n')


def load_original():
    path = ROOT/'audits/fast_qep_feasibility_20261014/verify_math.py'
    spec = importlib.util.spec_from_file_location('original_review',path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.load_original_core()[1]


def prepare():
    manifest = list(csv.DictReader((ROOT/'results/real_cellseg_round3_thresholding_20260916/image_manifest.csv').open()))
    selected = [next(m for m in manifest if m['dataset']==ds and m['role']=='development')
                for ds in ['nuclei','whole_cell']]
    prepared = []
    original = load_original()
    for row in selected:
        ds = row['dataset']; path = ROOT/row['path_image']
        assert digest(path,'md5')==row['md5_image']
        assert digest(ROOT/row['path_gt'],'md5')==row['md5_gt']
        raw = np.asarray(Image.open(path),dtype=float)
        if raw.ndim==3: raw=raw[...,0]
        assert raw.shape==(int(row['height']),int(row['width']))
        p = next(csv.DictReader((ROOT/f'audits/parity/real_{ds}_r_params.csv').open()))
        shape = (int(p['n1']),int(p['n2']))
        tile = raw[:shape[0],:shape[1]]/255
        started=time.perf_counter()
        grid = KroneckerGrid.matern52(shape,float(p['beta1']),float(p['beta2']))
        ref=grid.gaussian_reference(tile,float(p['nugget_nu']))
        q2=ref.mean+grid.inverse(grid.posterior(ref,2).gaussian_mean())
        gp_seconds=time.perf_counter()-started
        sp=original['SharedParams'](float(p['beta1']),float(p['beta2']),float(p['nugget_nu']),
                                   *shape,'historical R fixed first tile',0.,'none',0,0.)
        old=original['reconstruct_tile'](tile,sp).pred_mean
        maxdiff=float(np.max(abs(old-q2)))
        assert maxdiff<1e-10
        mean_error=abs(ref.mean-float(p['theta_hat']))
        scale_error=abs(ref.signal_variance-float(p['sigma2_hat']))
        assert mean_error<1e-10 and scale_error<1e-10
        if ds=='nuclei':
            archive=AUDIT/'r_reference_validation/fastgp_r_reference.zip'
            with zipfile.ZipFile(archive) as z:
                rt=np.loadtxt(io.BytesIO(z.read('r_tile_n120.csv')),delimiter=',')
            assert np.max(abs(tile[:120,:120]-rt))<1e-14
        prepared.append((ds,grid,ref,tile,{
            'dataset':ds,'input_path':row['path_image'],'input_sha256':digest(path),
            'gt_path':row['path_gt'],'gt_sha256':digest(ROOT/row['path_gt']),
            'shape':shape,'pixels':grid.n,'q2_runtime_including_grid_sec':gp_seconds,
            'r_parameters':p,'q2_vs_original_python_max_error':maxdiff,
            'mean_vs_historical_r_error':mean_error,'scale_vs_historical_r_error':scale_error,
            'signal_variance':ref.signal_variance,'noise_variance':ref.noise_variance,
        }))
    return prepared


def make_figure(out,ds,arrays,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delta=arrays['q15_controlled']-arrays['q2']
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    lo,hi=np.percentile(arrays['raw'],[1,99])
    for ax,key,title in zip(axes[0],['raw','q2','q15_controlled'],['Raw','Fast-GP / q=2','QEP q=1.5 (MC estimate)']):
        ax.imshow(arrays[key],cmap='gray',vmin=lo,vmax=hi)
        ax.set_title(title); ax.set_axis_off()
    limit=max(float(np.max(abs(delta))),1e-8)
    im=axes[1,0].imshow(delta,cmap='coolwarm',vmin=-limit,vmax=limit)
    axes[1,0].set_title('q=1.5 minus q=2 (intensity units)')
    fig.colorbar(im,ax=axes[1,0],shrink=.7)
    im=axes[1,1].imshow(arrays['q15_mcse'],cmap='magma')
    axes[1,1].set_title('Estimated pixel MCSE')
    fig.colorbar(im,ax=axes[1,1],shrink=.7)
    for c in range(4):
        axes[1,2].plot(arrays['radius_traces'][c],lw=.5,alpha=.7,label=f'Chain {c+1}')
    axes[1,2].set_title('Retained latent radius traces')
    axes[1,2].set_xlabel('Retained iteration');axes[1,2].legend(fontsize=7)
    for ax in axes[1,:2]: ax.set_axis_off()
    status='PASSES numerical gates' if summary['numerical_gate_pass'] else 'FAILS numerical gates: exploratory estimate only'
    fig.suptitle(f'{ds}: fixed first tile; {status}\nNo clean-image ground truth; image difference is not improvement',fontsize=12)
    fig.savefig(out/f'{ds}_diagnostic.png',dpi=150)
    plt.close(fig)


def run(args):
    if args.out.exists(): raise RuntimeError('Output directory exists; refusing to overwrite.')
    args.out.mkdir(parents=True)
    config=HMCConfig(warmup=2000,draws=4000)
    stream_checks=verify_stream_equivalence()
    prepared=prepare()
    files=[HERE/'PROTOCOL.md',HERE/'run_pilot.py',HERE/'stream_hmc.py',
           ROOT/'py_core/fast_qep_prototype.py',ROOT/'py_core/dim_2_lattice.py',
           ROOT/'py_core/paper_fast_gp.py',AUDIT/'diagnostics.py',
           ROOT/'audits/fast_qep_feasibility_20261014/verify_math.py']
    record={'scope':'original first tiles of two development images; q=1.5 versus analytic q=2',
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'config':asdict(config),'threads':1,'stream_sampler_checks':stream_checks,
            'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files},
            'inputs':[x[-1] for x in prepared],
            'Rscript':shutil.which('Rscript'),'paper_downstream_available':False}
    save(args.out/'manifest.json',record)
    print('Input hashes, fixed R means/scales, q=2 and streamed-sampler checks PASS.',flush=True)
    results=[]
    for fid,(ds,grid,ref,tile,metadata) in enumerate(prepared,start=101):
        target=grid.posterior(ref,1.5)
        observer=BlockObserver(target,config.draws)
        print(f'Starting {ds}: n={grid.n}; 4 chains x (2000+4000).',flush=True)
        sample,initial,last=stream_hmc(target,config,fid,observer,900,
                    progress=lambda i,total,sec: print(f'{ds}: {i}/{total}, elapsed {sec:.1f}s',flush=True))
        blocks,cvblocks=observer.complete_blocks()
        np.savez_compressed(args.out/f'{ds}_chains.npz',
            trace=observer.traces[:,:observer.count],trace_names=np.asarray(observer.names),
            spectral_indices=observer.indices,raw_spectral_blocks=blocks,
            controlled_spectral_blocks=cvblocks,initial_positions=initial,final_positions=last)
        if not sample['completed']:
            result={**metadata,'sampler':sample,'numerical_gate_pass':False,'status':'STOP_TIME_BUDGET'}
            save(args.out/f'{ds}_summary.json',result);results.append(result)
            print(f'{ds}: STOP_TIME_BUDGET',flush=True)
            del observer,blocks,cvblocks
            continue
        diagnostics={name:summarize(observer.traces[:,:,j]) for j,name in enumerate(observer.names)}
        raw_stats=pixel_block_summary(grid,blocks)
        cv_stats=pixel_block_summary(grid,cvblocks)
        cv_mcse=cv_stats['mcse']*255
        q2=ref.prediction*255
        raw_mean=(ref.mean+raw_stats['mean'])*255
        controlled=(ref.mean+cv_stats['mean'])*255
        delta=controlled-q2
        rhat=max(x['rhat_rank_folded_split'] for x in diagnostics.values())
        bulk=min(x['ess_bulk'] for x in diagnostics.values())
        tail=min(x['ess_tail'] for x in diagnostics.values())
        gates={'convergence':rhat<1.01 and bulk>400 and tail>400,
               'no_retained_divergences':sum(sample['divergences_abs_delta_H_gt_100'])==0,
               'pixel_mcse':float(np.quantile(cv_mcse,.99))<=.1 and float(cv_mcse.max())<=.5}
        result={**metadata,'sampler':sample,'max_rhat':rhat,'min_bulk_ess':bulk,'min_tail_ess':tail,
                'controlled_pixel_mcse_p99':float(np.quantile(cv_mcse,.99)),
                'controlled_pixel_mcse_max':float(cv_mcse.max()),
                'raw_pixel_mcse_p99':float(np.quantile(raw_stats['mcse']*255,.99)),
                'mean_absolute_q_change':float(np.mean(abs(delta))),
                'max_absolute_q_change':float(np.max(abs(delta))),
                'rms_q_change':float(np.sqrt(np.mean(delta**2))),
                'fraction_q_change_exceeds_3_mcse':float(np.mean(abs(delta)>3*cv_mcse)),
                'raw_vs_controlled_mean_abs_difference':float(np.mean(abs(raw_mean-controlled))),
                'gates':gates,'numerical_gate_pass':all(gates.values()),
                'diagnostics':diagnostics,
                'status':'PASS_LOCAL_NUMERICAL_GATE' if all(gates.values()) else 'STOP_NUMERICAL_GATE'}
        arrays={'raw':tile*255,'q2':q2,'q15_raw':raw_mean,'q15_controlled':controlled,
                'q15_mcse':cv_mcse,'raw_mcse':raw_stats['mcse']*255,
                'raw_chain_means':(ref.mean+raw_stats['chain_means'])*255,
                'controlled_chain_means':(ref.mean+cv_stats['chain_means'])*255,
                'radius_traces':observer.traces[:,:,0]}
        np.savez_compressed(args.out/f'{ds}_estimates.npz',**arrays)
        save(args.out/f'{ds}_summary.json',result)
        make_figure(args.out,ds,arrays,result)
        results.append(result)
        print(f'{ds}: {result["status"]}; gates={gates}; max Rhat={rhat:.6f}; min ESS={bulk:.1f}; '
              f'pixel MCSE p99={np.quantile(cv_mcse,.99):.6f}',flush=True)
        del observer,blocks,cvblocks,raw_stats,cv_stats,arrays
    projected=sum(r['sampler']['runtime_sec']*(16 if r['dataset']=='nuclei' else 9) for r in results)
    numerical=all(r['numerical_gate_pass'] for r in results)
    runtime=projected<=7200 and all(r['sampler']['completed'] for r in results)
    save(args.out/'completion.json',{
        'stage_a_completed':all(r['sampler']['completed'] for r in results),
        'all_numerical_gates_pass':numerical,'projected_full_sampling_sec':projected,
        'runtime_gate_pass':runtime,'stage_b_executed':False,
        'status':('READY_FOR_MAC_DOWNSTREAM_PREFLIGHT' if numerical and runtime else 'STOP_BEFORE_FULL_IMAGE'),
        'next_stage_environment_blocker':'Rscript and original R downstream packages unavailable here',
        'dataset_status':{r['dataset']:r['status'] for r in results},
        'no_segmentation_metrics_produced':True})
    print('Stage A complete. No full-image or downstream run launched.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/fast_qep_real_pilot_20261014')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
