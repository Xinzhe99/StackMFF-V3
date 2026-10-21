"""Validate all canonical pairs and create an isolated input-only view.

Reference images are deliberately excluded from inference views. Existing datasets,
scripts and inference jobs are never changed. Re-running this preparation is safe.
"""
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/audit_codex/resource_rebalance_2026-10-19'))
from remote_ops import call
HERE=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
from PIL import Image
from collections import Counter
import json, hashlib, os, datetime
root=Path('/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz')
data=root/'datasets/three_datasets_MFF'
work=root/'v3merge_work/pair_eval_20261019'
spec=[('Lytro',20,None),('MFFW',13,None),('MFI-WHU',120,'full_clear'),('RealMFF',710,'Fusion')]
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(data),'work_root':str(work),'datasets':{},'scenes':[]}
errors=[]
def meta(p):
 with Image.open(p) as im:
  im.load()
  return {'path':str(p.relative_to(data)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'width':im.width,'height':im.height,'mode':im.mode}
for ds,n,ref in spec:
 maps={}
 for part in ['A','B']+([ref] if ref else []):
  files=[p for p in (data/ds/part).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]
  maps[part]={p.stem:p for p in files}
  if len(maps[part])!=len(files):errors.append(ds+'/'+part+':duplicate stems')
  if set(maps[part])!={str(i) for i in range(1,n+1)}:errors.append(ds+'/'+part+':unexpected IDs '+str(len(files)))
 sizes=Counter();total_bytes=0
 for sid in sorted(maps['A'],key=lambda s:int(s)):
  try:row={'dataset':ds,'scene':sid,'a':meta(maps['A'][sid]),'b':meta(maps['B'][sid])}
  except Exception as e:errors.append(ds+'/'+sid+':'+repr(e));continue
  if ref:row['reference_candidate']=meta(maps[ref][sid]);row['reference_status']='pending_author_release_match'
  shape=(row['a']['width'],row['a']['height'])
  if any((m['width'],m['height'])!=shape for k,m in row.items() if isinstance(m,dict)):errors.append(ds+'/'+sid+':size mismatch')
  if row['a']['sha256']==row['b']['sha256']:errors.append(ds+'/'+sid+':identical inputs')
  sizes[str(shape)]+=1
  total_bytes+=sum(m['bytes'] for m in row.values() if isinstance(m,dict))
  manifest['scenes'].append(row)
 manifest['datasets'][ds]={'expected':n,'sizes':dict(sizes),'total_bytes':total_bytes,'pairing':'exact numeric filename stem, not list position','reference_folder':ref,'reference_status':'pending_author_release_match' if ref else 'none'}
if errors:raise RuntimeError(json.dumps(errors))
# Fail above before changing anything if even one source is invalid.
for row in manifest['scenes']:
 view=work/'inputs'/row['dataset']/'image stack'/row['scene'];view.mkdir(parents=True,exist_ok=True)
 for i,k in enumerate(['a','b'],1):
  src=data/row[k]['path'];dst=view/(str(i)+src.suffix.lower())
  if dst.is_symlink():
   if dst.resolve()!=src.resolve():raise RuntimeError('Conflicting link: '+str(dst))
  elif dst.exists():raise RuntimeError('Existing non-link: '+str(dst))
  else:dst.symlink_to(src)
work.mkdir(parents=True,exist_ok=True)
path=work/'manifest.json'
if path.exists():
 old=json.loads(path.read_text()); old.pop('created_utc',None);check=dict(manifest);check.pop('created_utc',None)
 if old!=check:raise RuntimeError('Manifest changed: review before replacement')
else:path.write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
'''
manifest=call('hainan',code,timeout=120)
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps({'datasets':manifest['datasets'],'pairs':len(manifest['scenes']),'remote_work':manifest['work_root']},indent=2))
