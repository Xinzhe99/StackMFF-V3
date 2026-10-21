"""Score verified references with the manuscript's unchanged PSNR/SSIM code."""
from pathlib import Path
import argparse,hashlib,json,math,msvcrt,os,sys,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/eval'))
from metrics_lib import load_gray,psnr,ssim_wang
from PIL import Image
from method_registry import METHODS
from reference_protocol import DATASETS,load_reference_protocol,lock_filename

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--loop',action='store_true')
 ap.add_argument('--datasets',nargs='+',choices=DATASETS,default=['MFI-WHU']);a=ap.parse_args()
 datasets={ds:load_reference_protocol(ds) for ds in sorted(set(a.datasets))}
 locks=[]
 for ds in datasets:
  lock=(HERE/lock_filename(ds)).open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);locks.append(lock)
 manifest=json.loads((HERE/'manifest.json').read_text())
 active={m['method'] for m in METHODS};metric_hash=sha(ROOT/'work/eval/metrics_lib.py')
 data=Path('E:/ImagefusionDatasets/three_datasets_MFF');n=0
 print(json.dumps({'reference_datasets':datasets,'active_configurations':len(active)}),flush=True)
 while True:
  progress=0
  for ds,proof in datasets.items():
   rows=[r for r in manifest['scenes'] if r['dataset']==ds]
   for root in [HERE/'results',HERE/'remote_results']:
    if not root.exists():continue
    for method in root.iterdir():
     if not method.is_dir() or method.name not in active:continue
     for r in rows:
      sid=r['scene'];fp=method/ds/(sid+('_rgb.png' if method.name=='SwinFusion' else '.png'))
      result=HERE/'reference_scores'/method.name/ds/(sid+'.json')
      if result.exists() or not fp.exists() or time.time()-fp.stat().st_mtime<5:continue
      refinfo=r['reference_candidate'];rp=data/refinfo['path']
      assert sha(rp)==refinfo['sha256']
      try:
       with Image.open(fp) as im:im.load();assert im.mode=='RGB' and im.size==(r['a']['width'],r['a']['height'])
       output_hash=sha(fp)
       ref=load_gray(rp);fused=load_gray(fp);assert ref.shape==fused.shape
       scores={'PSNR':psnr(ref,fused),'SSIM':ssim_wang(ref,fused)}
       assert math.isfinite(scores['SSIM']) and not math.isnan(scores['PSNR'])
       assert sha(fp)==output_hash,'Output changed during scoring'
       rec={'method':method.name,'dataset':ds,'scene':sid,'metrics':scores,'output_sha256':output_hash,'reference_sha256':sha(rp),'reference_proof_sha256':proof['proof_sha256'],'metric_source_sha256':metric_hash,'reference_encoding':proof['encoding'],'protocol':'PIL uint8 L; PSNR range255; Wang SSIM11 Gaussian1.5 valid; no resize/alignment'}
       if 'approval_sha256' in proof:rec['reference_approval_sha256']=proof['approval_sha256']
       result.parent.mkdir(parents=True,exist_ok=True);tmp=result.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2));os.replace(tmp,result)
       n+=1;progress+=1
      except Exception as e:print(json.dumps({'reference_error':[method.name,ds,sid,repr(e)]}),flush=True)
  print(json.dumps({'reference_scored_this_pass':progress,'reference_scored_this_process':n}),flush=True)
  if not a.loop:return
  time.sleep(60)
if __name__=='__main__':main()
