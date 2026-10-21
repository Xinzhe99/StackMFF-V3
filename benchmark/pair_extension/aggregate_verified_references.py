"""Separate complete-reference means from the eight source-based metrics."""
from pathlib import Path
import csv,hashlib,json,math,statistics
from method_registry import METHODS,PAUSED_METHOD_IDS
from reference_protocol import DATASETS,load_reference_protocol
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_inventory():
 inv={}
 for name in ['output_inventory.tsv','output_inventory_zhoushan.tsv']:
  f=HERE/name
  if not f.is_file():continue
  for line in f.read_text(encoding='utf8').splitlines():
   parts=line.split('\t')
   if len(parts)==5:inv.setdefault((parts[1],parts[2],parts[3]),set()).add(parts[4])
 return inv

def main():
 inv=load_inventory()
 manifest=json.loads((HERE/'manifest.json').read_text());expected={(r['dataset'],r['scene']):r for r in manifest['scenes']}
 proofs={ds:load_reference_protocol(ds) for ds in DATASETS}
 active={m['method'] for m in METHODS};metric_hash=sha(HERE.parents[1]/'work/eval/metrics_lib.py')
 records={}
 for p in (HERE/'reference_scores').glob('*/*/*.json'):
  r=json.loads(p.read_text());ds=r['dataset'];sid=r['scene'];method=r['method']
  if method in PAUSED_METHOD_IDS:continue
  assert method in active and ds in proofs,p
  row=expected[ds,sid]
  assert r['reference_proof_sha256']==proofs[ds]['proof_sha256'] and r['reference_sha256']==row['reference_candidate']['sha256'],p
  if 'approval_sha256' in proofs[ds]:assert r['reference_approval_sha256']==proofs[ds]['approval_sha256'],p
  assert r['metric_source_sha256']==metric_hash,p
  fn=sid+('_rgb.png' if method=='SwinFusion' else '.png')
  outputs=[root/method/ds/fn for root in [HERE/'results',HERE/'remote_results'] if (root/method/ds/fn).exists()]
  if outputs:
   assert all(sha(q)==r['output_sha256'] for q in outputs),p
  else:
   assert r['output_sha256'] in inv.get((method,ds,fn),set()),('no output evidence',p)
  assert math.isfinite(r['metrics']['SSIM']) and not math.isnan(r['metrics']['PSNR']),p
  key=method,ds,sid;assert key not in records;records[key]=r
 summaries=[];coverage=[]
 for m in METHODS:
  for ds in proofs:
   ids={sid for d,sid in expected if d==ds};rs=[r for (method,d,sid),r in records.items() if method==m['method'] and d==ds]
   complete={r['scene'] for r in rs}==ids
   coverage.append({'method':m['method'],'dataset':ds,'scored':len(rs),'expected':len(ids),'complete':complete})
   if complete:summaries.append({'method':m['method'],'dataset':ds,'n':len(ids),'PSNR':statistics.mean(r['metrics']['PSNR'] for r in rs),'SSIM':statistics.mean(r['metrics']['SSIM'] for r in rs)})
 for name,rows in [('reference_summary_complete_only',summaries),('reference_coverage',coverage)]:
  if rows:
   with (HERE/(name+'.csv')).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'reference_scored_outputs':len(records),'complete_method_datasets':len(summaries)}))
if __name__=='__main__':main()
