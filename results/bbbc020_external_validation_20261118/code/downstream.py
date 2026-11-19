from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,tempfile,shutil,traceback,hashlib
from pathlib import Path
from config import ROOT,PKG,RSCRIPT,np,pd,geometry,canonical,field_names
ref=ROOT/"gp/reference"
source=ref/"src/Modified_Functions_RGasp.R"
assert hashlib.sha256(source.read_bytes()).hexdigest()=="f7840f7ac25e5ed3e4f98a3101a9ac1090f4b36238fec1b84225923897e0c391"
runner=PKG/"experiments/real_data/qep_downstream_4x2/paper_downstream.R"
def replay(directory,rec,ratios):
    h,w=rec.shape
    th,ny=geometry(h);tw,nx=geometry(w)
    mask=np.zeros((h,w),bool)
    assert len(ratios)==nx*ny
    for col in range(nx):
        for row in range(ny):
            sl=np.s_[row*th:(row+1)*th,col*tw:(col+1)*tw]
            mask[sl]=rec[sl]>ratios[col*ny+row]*rec[sl].max()
    mask.T.astype("uint8").tofile(directory/"mask.bin")
    subprocess.run([RSCRIPT,str(ROOT/"code/replay.R"),str(source),str(directory/"mask.bin"),str(directory/"labels.bin"),str(h),str(w)],check=True,stdout=subprocess.DEVNULL)
    labels=np.fromfile(directory/"labels.bin",dtype="<i4").reshape(rec.shape,order="F")
    return mask,labels
def process(name):
    folder=ROOT/"results"/name
    if (folder/"downstream.done").exists():return
    with tempfile.TemporaryDirectory(dir=ROOT) as work:
        d=Path(work)
        for arm in ["fit_gp","fit_q115"]:
            rec=np.load(folder/(arm+"_recon.npz"))["predmean"]
            np.savetxt(d/"input.csv",rec,delimiter=",",fmt="%.17g")
            with (folder/(arm+"_R.log")).open("w") as log:
                subprocess.run([RSCRIPT,str(runner),"whole_cell",str(ROOT/"data"/(name+".png")),str(d/"input.csv"),str(d),arm,str(ref)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            fg=np.loadtxt(d/(arm+"_combined_binary.csv"),delimiter=",",dtype=np.uint8)>0
            labels=np.loadtxt(d/(arm+"_labels_final.csv"),delimiter=",",dtype=np.int32)
            for suffix in ["_tiles.csv","_R_session.txt","_summary.csv"]:
                shutil.copyfile(d/(arm+suffix),folder/(arm+suffix))
            frac=pd.read_csv(folder/(arm+"_tiles.csv")).p_final.to_numpy()
            fg_check,labels_check=replay(d,rec,frac)
            assert np.array_equal(fg,fg_check) and np.array_equal(canonical(labels),canonical(labels_check)),f"replay failed {name}/{arm}"
            np.savez_compressed(folder/(arm+"_masks.npz"),labels=labels,foreground=fg)
            if arm=="fit_gp":gp_frac=frac
            else:
                fg,labels=replay(d,rec,gp_frac)
                np.savez_compressed(folder/"q115_gp_thresholds_masks.npz",labels=labels,foreground=fg)
        (folder/"downstream.done").write_text("both native paths reproduced; shared thresholds evaluated\n")
if __name__=="__main__":
    import time
    items=field_names()
    # Pipeline segmentation as fields finish; per-field score is kept separate.
    seen=set();tasks={}
    with ThreadPoolExecutor(int(os.getenv("R_WORKERS","3"))) as pool:
        while len(seen)<len(items):
            for field in items:
                if field not in seen and all((ROOT/"results"/field/(a+"_fit.json")).is_file() for a in ("fit_gp","fit_q115")):
                    seen.add(field); tasks[pool.submit(process,field)]=field
            if len(seen)<len(items):
                if (ROOT/"fit.finished").exists():break
                time.sleep(2)
        for task in as_completed(tasks):
            print(tasks[task],"complete" if not task.exception() else repr(task.exception()),flush=True)
