from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
root=Path(__file__).resolve().parents[1]
fields=pd.read_csv(root/"field_deltas.csv")
order=list(fields[fields.arm=="fit_q115"].field)
positions=np.arange(len(order))
fig,axes=plt.subplots(1,2,figsize=(9,7.3),sharey=True)
tasks=[("fit_q115","ap50","Primary comparison","AP50, native thresholds"),("q115_gp_thresholds","ap75","Threshold control","AP75, shared GP thresholds")]
for ax,(arm,metric,title,subtitle) in zip(axes,tasks):
    d=fields[fields.arm==arm].set_index("field").loc[order,metric].to_numpy()
    low=min(-.5,100*d.min()*1.2);high=max(.5,100*d.max()*1.2)
    ax.axvline(0,color="#718091",lw=.8)
    ax.hlines(positions,0,d*100,color="#a1b4c2",lw=1)
    ax.scatter(d*100,positions,c=np.where(d>0,"#127963",np.where(d<0,"#b15739","#546979")),s=24,zorder=3)
    ax.set(xlim=(low,high),xlabel="QEP minus GP (percentage points)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x",alpha=.15);ax.spines[['top','right','left']].set_visible(False)
    ax.set_title(title+"\n"+subtitle,loc="left",fontsize=11)
axes[0].set_yticks(positions,[x.replace("jw-","").replace("_c1","").replace("Kontrolle","Control ") for x in order],fontsize=9)
axes[0].invert_yaxis()
fig.suptitle("Whole-cell instance segmentation on 20 BBBC020 fields",x=.13,ha="left",fontsize=15)
fig.tight_layout(rect=(0,0,.99,.955))
fig.savefig(root/"paired_fields.png",dpi=180,bbox_inches="tight")
