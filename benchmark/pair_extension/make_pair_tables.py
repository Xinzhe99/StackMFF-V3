"""Require every active configuration before typesetting dataset rankings."""
from pathlib import Path
import csv,json,math
from method_registry import METHODS
from aggregate_scores import METRICS
HERE=Path(__file__).resolve().parent
HEAD=[r'NMI $\uparrow$',r'FMI$_w$ $\uparrow$',r'$Q^{AB/F}$ $\uparrow$',r'$Q_P$ $\uparrow$',r'$Q_Y$ $\uparrow$',r'$Q_{CB}$ $\uparrow$',r'$Q_{CV}$ $\downarrow$',r'VIFF $\uparrow$']

def main():
 p=HERE/'summary_complete_only.csv';rows=list(csv.DictReader(p.open())) if p.exists() else []
 v={(r['method'],r['dataset']):r for r in rows};out=HERE/'publication_draft/tables';out.mkdir(parents=True,exist_ok=True)
 report={};manifest=json.loads((HERE/'manifest.json').read_text())
 for ds,meta in manifest['datasets'].items():
  missing=[m['method'] for m in METHODS if (m['method'],ds) not in v]
  report[ds]={'complete_methods':len(METHODS)-len(missing),'expected_methods':len(METHODS),'missing':missing}
  if missing:continue
  values={n:sorted({float(v[m['method'],ds][n]) for m in METHODS},reverse=n!='QCV') for n in METRICS}
  def cell(method,n):
   val=float(v[method,ds][n]);assert math.isfinite(val)
   s=format(val,'.2f' if n=='QCV' else '.4f')
   if val==values[n][0]:return r'\best{'+s+'}'
   if len(values[n])>1 and val==values[n][1]:return r'\second{'+s+'}'
   return s
  name={'RealMFF':'Real-MFF','MFFW':'MFFW2'}.get(ds,ds)
  lines=[r'\begin{table}[!htbp]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{0.95}',
   r'\caption{Two-input fusion on '+name+' ('+str(meta['expected'])+r' pairs). All means use the complete dataset at native resolution. Arrows indicate the preferred direction. Pink and blue mark the best and second-best unrounded means. Training-data provenance and fixed-input adaptations are described in the accompanying protocol.}',
   r'\label{tab:pair_'+ds.lower().replace('-','')+'}',r'\begin{adjustbox}{max width=\textwidth}',r'\begin{tabular}{@{}llrrrrrrrr@{}}',r'\toprule',
   'Method & Venue & '+' & '.join(HEAD)+r' \\',r'\midrule']
  previous=None
  for m in METHODS:
   if m['group']!=previous:
    if previous is not None:lines.append(r'\addlinespace[2pt]')
    previous=m['group']
   lines.append(m['label']+' & '+m['venue_citation']+' & '+' & '.join(cell(m['method'],n) for n in METRICS)+r' \\')
  lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}'])
  (out/('tab_pair_'+ds+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf8')
 (out.parent/'table_readiness.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
