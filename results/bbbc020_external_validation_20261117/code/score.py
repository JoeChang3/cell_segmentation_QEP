from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from config import ROOT,canonical,field_names,compute_ap_from_ious
from frozen_metrics import score as legacy_score
ARMS=["fit_gp","fit_q115","q115_gp_thresholds"]

def calculate(masks, labels):
    pred_ids,areas=np.unique(labels,return_counts=True)
    positive=pred_ids>0
    pred_ids,areas=pred_ids[positive],areas[positive]
    counts=np.stack([np.bincount(labels.ravel()[ref],minlength=int(labels.max())+1)[pred_ids] for ref in masks],axis=0)
    ref_areas=np.array([len(ref) for ref in masks])[:,None]
    iou=counts/(ref_areas+areas[None,:]-counts)
    table=pd.DataFrame(iou)
    summary=dict(n_gt=len(masks),n_pred=len(pred_ids),merged=int(((counts/ref_areas>=.25).sum(axis=0)>=2).sum()),
                 split=int(((counts/ref_areas>=.25).sum(axis=1)>=2).sum()))
    for cutoff,suffix in [(.5,"50"),(.75,"75")]:
        row=compute_ap_from_ious(table,cutoff)
        summary["ap"+suffix]=row["precision"]
        summary["recall"+suffix]=row["tp"]/len(masks)
        for k in ["tp","fp","fn"]: summary[k+suffix]=row[k]
    return summary

def test_metric():
    gt=np.zeros((20,20),int)
    gt[1:7,3:11]=1;gt[8:16,4:7]=2;gt[8:16,7:10]=3
    pred=gt.copy();pred[pred==3]=2;pred[1:7,7:11]=4
    masks=[np.flatnonzero(gt.ravel()==n) for n in [1,2,3]]
    row=calculate(masks,pred)
    orig,_=legacy_score(gt,pred,pred>0)
    for key in ["ap50","ap75","tp50","fp50","fn50","merged","split","n_pred","n_gt"]:
        assert row[key]==orig[key],key
    overlapping=[masks[0],np.array([24,25,26])]
    cover=np.zeros_like(pred);cover.ravel()[masks[0]]=1
    assert calculate(overlapping,cover)["merged"]==1

def main():
    test_metric()
    fields=field_names()
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    rows=[];gt_audit=[];audit=[]
    with ZipFile(ROOT/"data/cells.zip") as z:
        for field in fields:
            folder=ROOT/"results"/field
            if not (folder/"downstream.done").is_file():raise RuntimeError(f"{field}: segmentation unfinished")
            masks=[]
            seg=np.load(folder/"fit_gp_masks.npz")["labels"]
            overlap=np.zeros(seg.shape,dtype=np.uint8)
            for mask_path in manifest["reference_masks"][field]:
                mask=np.array(Image.open(BytesIO(z.read(mask_path))))
                assert mask.shape==seg.shape and set(np.unique(mask))=={0,255}
                overlap+=mask>0
                masks.append(np.flatnonzero(mask.ravel()))
            gt_audit.append(dict(field=field,objects=len(masks),overlap_pixels=int((overlap>1).sum())))
            for arm in ARMS:
                with np.load(folder/(arm+"_masks.npz")) as obj:
                    labels=obj["labels"];fg=obj["foreground"]
                    assert labels.shape==seg.shape
                    rows.append(dict(field=field,arm=arm,foreground_pixels=int(fg.sum()),**calculate(masks,labels)))
            for arm in ARMS[:2]:
                doc=json.loads((folder/(arm+"_fit.json")).read_text())
                audit.append(dict(field=field,arm=arm,tiles=len(doc["tiles"]),
                    retained_start_converged=doc["best"]["success"],stationary_retry_selected=doc["best"]["retry"],
                    tile_retries=sum(x["retry"] for x in doc["tiles"]),max_abs_log_evidence_grid_diff=max(abs(x["q_order_delta"]) for x in doc["tiles"]),
                    max_normalized_recon_grid_diff=max(x["prediction_error"] for x in doc["tiles"]),
                    max_projected_gradient=max(x["gradient"] for x in doc["tiles"])))
    scores=pd.DataFrame(rows);scores.to_csv(ROOT/"field_scores.csv",index=False)
    pd.DataFrame(gt_audit).to_csv(ROOT/"reference_audit.csv",index=False)
    pd.DataFrame(audit).to_csv(ROOT/"fit_audit.csv",index=False)
    base=scores[scores.arm=="fit_gp"].set_index("field")
    diffs=[];summaries=[]
    for arm in ARMS[1:]:
        other=scores[scores.arm==arm].set_index("field").loc[base.index]
        columns=["ap50","ap75","merged","split","recall50","recall75"]
        delta=other[columns]-base[columns]
        delta.insert(0,"arm",arm);diffs.append(delta)
        for metric in columns:
            d=delta[metric].to_numpy()
            rng=np.random.default_rng(20261117)
            resamples=rng.choice(d,(20000,len(d)),replace=True)
            lo,hi=np.quantile(resamples.mean(axis=1),[.025,.975])
            summaries.append(dict(arm=arm,metric=metric,fields=len(d),baseline_mean=base[metric].mean(),arm_mean=other[metric].mean(),
                                  mean_delta=d.mean(),median_delta=np.median(d),win=int((d>0).sum()),tie=int((d==0).sum()),loss=int((d<0).sum()),
                                  field_bootstrap_lo=lo,field_bootstrap_hi=hi,baseline_total=base[metric].sum(),candidate_total=other[metric].sum()))
    pd.concat(diffs).to_csv(ROOT/"field_deltas.csv")
    pd.DataFrame(summaries).to_csv(ROOT/"summary.csv",index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
if __name__=="__main__":main()
