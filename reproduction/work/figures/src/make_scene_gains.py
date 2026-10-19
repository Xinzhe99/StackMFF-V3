"""All-scene SSIM differences against a fixed external comparator per dataset."""
from pathlib import Path
import csv, hashlib, json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'work/paper/tables'))
import make_tables as tables

def main():
    source = ROOT/'work/eval/per_image.csv'
    rows = list(csv.DictReader(source.open(encoding='utf8')))
    means = tables.load()
    baselines = [m for _, methods in tables.GROUPS for m in methods if not m.startswith('Ours')]
    records = []
    plt.rcParams.update({'font.family':'serif', 'font.size':10, 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(8.6,3.1))
    for i, (dataset, _) in enumerate(tables.DATASETS):
        baseline = max(baselines, key=lambda m:means.get((m,dataset,'SSIM'),-np.inf))
        def values(method):
            selected = [r for r in rows if r['method']==method and r['dataset']==dataset and r['primary']=='True']
            assert len({r['scene'] for r in selected})==len(selected)
            return {r['scene']:float(r['SSIM']) for r in selected}
        ours, other = values(tables.OURS_NAME), values(baseline)
        assert set(ours)==set(other)
        gains = [{'scene':s,'ours':ours[s],'baseline':other[s],'delta':ours[s]-other[s]} for s in sorted(ours)]
        x = np.array([r['delta'] for r in gains])
        # Deterministic vertical packing prevents dense cohorts hiding observations.
        used=[]; yy=[]
        for value in x:
            candidates=[0]+[v for j in range(1,30) for v in (j*.075,-j*.075)]
            y=next(y for y in candidates if all(abs(value-a)>.00022 or abs(y-b)>.074 for a,b in used))
            used.append((value,y));yy.append(i+y)
        ax.scatter(x,yy,s=16,color=['#567fa5' if v>0 else '#ad3b52' for v in x],alpha=.85,edgecolors='white',linewidth=.3,zorder=3)
        ax.scatter([x.mean()],[i],marker='D',s=53,color='#ad3b52',edgecolors='white',linewidth=.7,zorder=4)
        wins=int((x>0).sum()); ties=int((x==0).sum())
        ax.text(1.025,i,f'{wins}/{len(x)}',transform=ax.get_yaxis_transform(),va='center',ha='left',fontsize=10)
        records.append({'dataset':dataset,'baseline':baseline,'count':len(x),'wins':wins,'ties':ties,'mean_delta':float(x.mean()),'scenes':gains})
    ax.axvline(0,color='#89939d',lw=.8,zorder=1)
    ax.set_yticks(range(4),[r['dataset']+'\nvs '+tables.DISPLAY.get(r['baseline'],r['baseline']) for r in records])
    ax.set_ylim(3.48,-.60)
    ax.set_xlim(-.0012,.021)
    ax.set_xticks([0,.005,.010,.015,.020])
    ax.set_xticklabels(['0','+0.005','+0.010','+0.015','+0.020'])
    ax.set_xlabel('SSIM difference: weighted fusion − fixed external baseline')
    ax.text(1.025,-.48,'Wins',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=9,color='#56606a')
    ax.scatter([],[],s=18,c='#567fa5',label='One scene')
    ax.scatter([],[],s=37,marker='D',c='#ad3b52',label='Dataset mean')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=9)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#89939d')
    ax.tick_params(axis='y',length=0,pad=9)
    ax.grid(axis='x',color='#e5e8ec',lw=.6,zorder=0)
    fig.subplots_adjust(left=.19,right=.89,bottom=.19,top=.86)
    out=ROOT/'work/figures/out/fig_scene_gains'
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'),dpi=180,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps({'metric':'Gaussian-window SSIM, BT.601 luma','selection':'For each dataset, select the external baseline with the highest unrounded dataset-mean SSIM in the complete current table; keep it fixed for every scene. All 206 scenes are shown.','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256(Path(tables.SUMMARY).read_bytes()).hexdigest(),'records':records},indent=2),encoding='utf8')
    print(json.dumps([{k:v for k,v in r.items() if k!='scenes'} for r in records]))

if __name__=='__main__':main()
