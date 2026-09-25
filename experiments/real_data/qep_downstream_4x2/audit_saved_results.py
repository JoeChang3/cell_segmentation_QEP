from pathlib import Path
import csv,json,hashlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import argparse
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--results',type=Path,required=True)
p.add_argument('--cache',type=Path,required=True)
args=p.parse_args()
repo=args.repo; results=args.results; cache=args.cache
rows=list(csv.DictReader((results/'metrics.csv').open())); assert len(rows)==16
assert len({(r['dataset'],r['method'],r['downstream']) for r in rows})==16
state={};checks=[]
for r in rows:
 ds,m,d=r['dataset'],r['method'],r['downstream'];nuc=ds=='nuclei'
 folder=repo/'data'/('nuclear_test_images' if nuc else 'whole_cell_test_images')/(ds+'_figure_1')
 g=np.asarray(Image.open(folder/'original_true_masks.png'))
 if g.ndim==3:g=g[...,0]
 gt=np.searchsorted(np.unique(g),g)
 raw=np.asarray(Image.open(folder/('original_fig.png' if nuc else 'original_fig.jpg')))
 if raw.ndim==3:raw=raw[...,0]
 if m=='raw':a=raw
 elif m=='fastgp':a=np.load(results/ds/'fastgp_reconstruction.npz')['predmean']
 else:a=np.load(cache/f"{ds}_{ds}_figure_1_{'qep_q2' if m=='q2' else 'qep_q1.5'}.npz")['predmean']
 assert hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()==r['input_sha256']
 z=np.load(results/ds/f'{m}_{d}_segmentation.npz');pr=z['labels'];fg=z['foreground'].astype(bool)
 assert pr.shape==gt.shape==a.shape and np.isfinite(pr).all() and pr.min()>=0
 assert np.all(pr==np.floor(pr)) and np.all(fg[pr>0])
 tl=np.unique(gt[gt>0]);pl=np.unique(pr[pr>0]); ti=np.zeros(int(gt.max())+1,int); pi=np.zeros(int(pr.max())+1,int)
 ti[tl]=np.arange(1,len(tl)+1);pi[pl]=np.arange(1,len(pl)+1)
 nt,npr=len(tl),len(pl)
 hist=np.bincount((ti[gt]*(npr+1)+pi[pr]).ravel(),minlength=(nt+1)*(npr+1)).reshape(nt+1,npr+1)
 inter=hist[1:,1:];ta=hist[1:,:].sum(1)[:,None];pa=hist[:,1:].sum(0)[None,:]
 iou=inter/(ta+pa-inter);best=iou.max(1);chosen=iou.argmax(1);cover=inter/ta
 calc=dict(n_gt=nt,n_pred=npr,mean_best_iou=float(best.mean()),merged=int(((cover>=.25).sum(0)>=2).sum()),split=int(((cover>=.25).sum(1)>=2).sum()),missed=int((best<.5).sum()),spurious=int(((iou.max(0)<.5)&((cover>=.25).sum(0)==0)).sum()),fg_components_4=ndi.label(fg)[1])
 for t,s in [(.5,'50'),(.75,'75')]:
  matched=best>=t;tp=int(matched.sum());fp=npr-len(np.unique(chosen[matched]));fn=nt-tp
  calc.update({f'tp{s}':tp,f'fp{s}':fp,f'fn{s}':fn,f'ap{s}':tp/(tp+fp+fn)})
 truth=gt>0;it=int((fg&truth).sum());nf=int(fg.sum());ng=int(truth.sum())
 calc.update(fg_dice=2*it/(nf+ng),fg_iou=it/(nf+ng-it),fg_precision=it/nf,fg_recall=it/ng,fg_fraction=fg.mean(),gt_fg_fraction=truth.mean())
 for k,v in calc.items():assert abs(float(v)-float(r[k]))<1e-12,(ds,m,d,k,v,r[k])
 state[ds,f'{m}-{d}']=(best>=.5,(cover>=.25).sum(1)>=2)
 checks.append(dict(dataset=ds,method=m,downstream=d,input_hash_verified=True,mask_metrics_verified=True))
for c in json.loads((results/'comparisons.json').read_text()):
 cm,cs=state[c['dataset'],c['candidate']];rm,rs=state[c['dataset'],c['control']]
 assert (np.flatnonzero(rm&~cm)+1).tolist()==c['lost_gt_ids']
 assert (np.flatnonzero(cs&~rs)+1).tolist()==c['newly_split_gt_ids']
 assert int((cs&~rs).sum())==c['newly_split']
 assert int((rm&~cm).sum())==c['lost_previously_matched']
for ds in ['nuclei','whole_cell']:
 for m in ['raw','fastgp','q2','q15']:
  assert len({r['input_sha256'] for r in rows if r['dataset']==ds and r['method']==m})==1
anchors=json.loads((results/'anchor_checks.json').read_text());assert len(anchors)==10 and all(x['passed'] for x in anchors)
for anchor in anchors:
 ds,m,d=anchor['dataset'],anchor['method'],anchor['downstream'];now=next(r for r in rows if (r['dataset'],r['method'],r['downstream'])==(ds,m,d))
 if m in ['q2','q15']:
  historical=list(csv.DictReader((repo/'results/real_cellseg_round5_corrected_baseline_20260917/development_segmentation_metrics.csv').open()))
  old=next(x for x in historical if x['dataset']==ds and x['method']==('qep_q2' if m=='q2' else 'qep_q1.5'))
 else:
  historical=list(csv.DictReader((repo/'results/real_cellseg_round6_paper_downstream_20260917/two_by_two_metrics.csv').open()))
  cell='D' if m=='fastgp' else 'B' if d=='P' else 'A'
  old=next(x for x in historical if x['dataset']==ds and x['cell']==cell)
 for k in ['tp50','fp50','fn50','tp75','fp75','fn75','merged','split','missed','spurious','fg_dice']:
  assert abs(float(now[k])-float(old[k]))<1e-12,(anchor,k)
summary=dict(status='PASS',n_masks_independently_scored=16,n_historical_anchors_verified=10,all_input_array_hashes_verified=True,all_PC_pairs_share_input=True,all_comparison_GT_identity_lists_verified=True,metric_tolerance=1e-12,scope='Independent NumPy/Pillow/SciPy audit of saved masks and arrays; no model training or downstream rerun',checks=checks)
(results/'INDEPENDENT_REVIEW.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
