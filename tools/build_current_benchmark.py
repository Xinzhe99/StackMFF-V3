"""Derive the complete current cohort without modifying historical measurements."""
from pathlib import Path
from collections import defaultdict, Counter
import csv,json,hashlib,math

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
    assert len(allowed)==36 and 'ReDiffuse' in allowed and 'ZMFF' not in allowed
    OUT.mkdir(exist_ok=True)
    fields,source=read(B/'per_image.csv')
    updated=set()
    for name in config.get('measurement_updates', []):
        update=(B/name).resolve()
        assert update.is_relative_to((B/'updates').resolve())
        manifest=json.loads((update/'manifest.json').read_text())
        assert manifest['status']=='validated_complete'
        for record in manifest['files']:
            p=(update/record['path']).resolve()
            assert p.is_relative_to(update)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'], p
        new_fields,new_rows=read(update/'per_image.csv')
        assert fields==new_fields
        replacements=set(manifest['methods'])
        assert not (updated & replacements), 'Overlapping update bundles'
        assert set(r['method'] for r in new_rows)==replacements
        source=[r for r in source if r['method'] not in replacements]+new_rows
        updated.update(replacements)
    rows=[r for r in source if r['method'] in allowed]
    counts=Counter((r['method'],r['dataset']) for r in rows if r['primary']=='True')
    sizes={'Mobile Depth':11,'Middlebury':15,'FlyingThings3D':100,'Road-MF':80}
    assert counts==Counter({(m,d):n for m in methods for d,n in sizes.items()})
    primary=[r for r in rows if r['primary']=='True']
    keys=[(r['method'],r['dataset'],r['scene']) for r in primary]
    assert len(keys)==len(set(keys))==7416
    for d in sizes:
        expected={r['scene'] for r in primary if r['method']=='CVT' and r['dataset']==d}
        for m in methods:
            assert {r['scene'] for r in primary if r['method']==m and r['dataset']==d}==expected
    write(OUT/'per_image.csv',fields,rows)
    fields,source=read(B/'summary.csv');means=[r.copy() for r in source if r['method'] in allowed]
    for name in config.get('measurement_updates', []):
        update=B/name
        new_fields,new_rows=read(update/'summary.csv')
        assert fields==new_fields
        replacements=set(r['method'] for r in new_rows)
        means=[r for r in means if r['method'] not in replacements]+new_rows
    groups=defaultdict(list)
    for r in means:groups[(r['dataset'],r['metric'])].append(r)
    for values in groups.values():
        assert len(values)==len(allowed)
        ordered=sorted((float(r['mean']) for r in values),reverse=True)
        for r in values:
            observations=[float(x[r['metric']]) for x in primary if x['method']==r['method'] and x['dataset']==r['dataset']]
            assert len(observations)==sizes[r['dataset']] and all(map(math.isfinite,observations))
            assert abs(sum(observations)/len(observations)-float(r['mean']))<1e-10
            v=float(r['mean']);positions=[i+1 for i,x in enumerate(ordered) if x==v]
            r['rank_in_dataset']=str(sum(positions)/len(positions));r['n_methods']=str(len(allowed))
    write(OUT/'summary.csv',fields,means)
    meta=[r for r in json.loads((B/'methods.json').read_text()) if r['id'] in allowed]
    for r in meta:
        r['datasets']=list(sizes)
    meta.sort(key=lambda r:methods.index(r['id']))
    (OUT/'methods.json').write_text(json.dumps(meta,indent=2)+'\n')
    scores={(r['method'],r['dataset'],r['metric']):float(r['mean']) for r in means}
    lines=['# Current complete-cohort comparison','',
           '31 baseline families (34 settings) and two proposed read-outs: 36 rows, each covering all 206 scenes. SSIM uses the Gaussian window; PSNR is in dB. The original measurements are retained, with the independently validated ReDiffuse update applied.','',
           '| Method | Venue | MD SSIM / PSNR | Mid. SSIM / PSNR | FT3D SSIM / PSNR | Road-MF SSIM / PSNR |',
           '|---|---|---:|---:|---:|---:|']
    group=None
    for r in meta:
        if r['group']!=group:
            group=r['group'];lines.append(f'| **{group}** | | | | | |')
        cells=[f"{scores[r['id'],d,'SSIM']:.4f} / {scores[r['id'],d,'PSNR']:.2f}" for d in sizes]
        lines.append('| '+r['display']+' | '+r.get('venue','—')+' | '+' | '.join(cells)+' |')
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    report={'methods':36,'baseline_families':31,'baseline_settings':34,'scenes':206,
            'primary_images':7416,'method_dataset_cells':144,'deferred_methods':['ZMFF'],
            'measurement_updates':config.get('measurement_updates', []),
            'historical_measurements_preserved':True,'ranks_recomputed_for_current_cohort':True}
    (OUT/'cohort.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
