"""Derive the complete current cohort without modifying historical measurements."""
from pathlib import Path
from collections import defaultdict, Counter
import csv,json

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'benchmark';OUT=B/'current'

def read(p):
    with p.open(newline='',encoding='utf8') as f:
        r=csv.DictReader(f);return r.fieldnames,list(r)

def write(p,fields,rows):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    config=json.loads((B/'config_paper.json').read_text())
    methods=config['methods'];allowed=set(methods)
    assert len(allowed)==35 and not allowed.intersection({'ZMFF','ReDiffuse'})
    OUT.mkdir(exist_ok=True)
    fields,source=read(B/'per_image.csv');rows=[r for r in source if r['method'] in allowed]
    counts=Counter((r['method'],r['dataset']) for r in rows if r['primary']=='True')
    sizes={'Mobile Depth':11,'Middlebury':15,'FlyingThings3D':100,'Road-MF':80}
    assert counts==Counter({(m,d):n for m in methods for d,n in sizes.items()})
    assert sum(counts.values())==7210
    write(OUT/'per_image.csv',fields,rows)
    fields,source=read(B/'summary.csv');means=[r.copy() for r in source if r['method'] in allowed]
    groups=defaultdict(list)
    for r in means:groups[(r['dataset'],r['metric'])].append(r)
    for values in groups.values():
        assert len(values)==35
        ordered=sorted((float(r['mean']) for r in values),reverse=True)
        for r in values:
            v=float(r['mean']);positions=[i+1 for i,x in enumerate(ordered) if x==v]
            r['rank_in_dataset']=str(sum(positions)/len(positions));r['n_methods']='35'
    write(OUT/'summary.csv',fields,means)
    meta=[r for r in json.loads((B/'methods.json').read_text()) if r['id'] in allowed]
    (OUT/'methods.json').write_text(json.dumps(meta,indent=2)+'\n')
    scores={(r['method'],r['dataset'],r['metric']):float(r['mean']) for r in means}
    lines=['# Current complete-cohort comparison','',
           '30 baseline families (33 settings) and two proposed read-outs: 35 rows, each covering all 206 scenes. SSIM uses the Gaussian window; PSNR is in dB. The underlying scores are unchanged from the historical records.','',
           '| Method | Venue | MD SSIM / PSNR | Mid. SSIM / PSNR | FT3D SSIM / PSNR | Road-MF SSIM / PSNR |',
           '|---|---|---:|---:|---:|---:|']
    group=None
    for r in meta:
        if r['group']!=group:
            group=r['group'];lines.append(f'| **{group}** | | | | | |')
        cells=[f"{scores[r['id'],d,'SSIM']:.4f} / {scores[r['id'],d,'PSNR']:.2f}" for d in sizes]
        lines.append('| '+r['display']+' | '+r.get('venue','—')+' | '+' | '.join(cells)+' |')
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    report={'methods':35,'baseline_families':30,'baseline_settings':33,'scenes':206,
            'primary_images':7210,'method_dataset_cells':140,'deferred_methods':['ZMFF','ReDiffuse'],
            'measurement_values_changed':False,'ranks_recomputed_for_current_cohort':True}
    (OUT/'cohort.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
