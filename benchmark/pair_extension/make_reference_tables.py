"""Typeset reference-based (PSNR/SSIM) rankings, refusing incomplete datasets."""
from pathlib import Path
import csv,json,math
from method_registry import METHODS
from reference_protocol import DATASETS
HERE=Path(__file__).resolve().parent
HEAD=[r'PSNR $\uparrow$',r'SSIM $\uparrow$']
DIGITS={'PSNR':2,'SSIM':4}

def load(name):
 p=HERE/(name+'.csv');return list(csv.DictReader(p.open(encoding='utf8'))) if p.exists() else []

def main():
 summary=load('reference_summary_complete_only');coverage=load('reference_coverage')
 measured={(r['method'],r['dataset']):r for r in summary}
 verdict={(r['method'],r['dataset']):r['complete']=='True' for r in coverage}
 expected=json.loads((HERE/'manifest.json').read_text())['datasets']
 out=HERE/'publication_draft/tables';out.mkdir(parents=True,exist_ok=True)
 report={}
 for ds in DATASETS:
  pairs=int(expected[ds]['expected'])
  missing=[]
  for m in METHODS:
   row=measured.get((m['method'],ds))
   if verdict.get((m['method'],ds)) is not True or row is None or int(row['n'])!=pairs:missing.append(m['method'])
  ready=not missing
  report[ds]={'complete_methods':len(METHODS)-len(missing),'expected_methods':len(METHODS),
   'expected_pairs':pairs,'table_generated':ready,'missing':missing}
  if not ready:continue
  values={n:sorted((float(measured[m['method'],ds][n]) for m in METHODS),reverse=True) for n in DIGITS}
  for n in DIGITS:
   assert len(values[n])==len(METHODS) and all(math.isfinite(v) for v in values[n])
  def cell(method,n):
   val=float(measured[method,ds][n]);assert math.isfinite(val)
   s=format(val,'.'+str(DIGITS[n])+'f')
   if val==values[n][0]:return r'\best{'+s+'}'
   if val==values[n][1]:return r'\second{'+s+'}'
   return s
  name={'RealMFF':'Real-MFF','MFFW':'MFFW2'}.get(ds,ds)
  lines=[r'\begin{table}[!htbp]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.95}',
   r'\caption{Two-input fusion on '+name+' ('+str(pairs)+r' pairs) against the per-scene ground truth. PSNR and SSIM are means over the complete dataset at native resolution; pink and blue mark the best and second-best unrounded means.}',
   r'\label{tab:pair_ref_'+ds.lower().replace('-','')+'}',r'\begin{adjustbox}{max width=\textwidth}',r'\begin{tabular}{@{}llrr@{}}',r'\toprule',
   'Method & Venue & '+' & '.join(HEAD)+r' \\',r'\midrule']
  previous=None
  for m in METHODS:
   if m['group']!=previous:
    if previous is not None:lines.append(r'\addlinespace[2pt]')
    previous=m['group']
   lines.append(m['label']+' & '+m['venue_citation']+' & '+' & '.join(cell(m['method'],n) for n in DIGITS)+r' \\')
  lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}'])
  (out/('tab_pair_ref_'+ds+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf8')
 p=HERE/'reference_table_readiness.json'
 p.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
