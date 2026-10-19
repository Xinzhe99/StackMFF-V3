"""Restore citation-backed venue abbreviations in table method labels.

Run after table generation. This changes labels only, never metric cells.
"""
from pathlib import Path
import re
import argparse
from make_tables import DISPLAY, CITE, VENUE

KEYS = {DISPLAY.get(k,k):k for k in VENUE}
KEYS.update({'IFCNN':'IFCNN-MAX', 'StackMFF':'StackMFF_release',
             'StackMFF~V2':'StackMFF V2', 'V2':'StackMFF V2'})
for label in ['LFSF', 'LFSF-RF']:
    KEYS[label]=label
    CITE[label]='li2026lfsf'
    VENUE[label]=r'Neurocomput.~\cite{li2026lfsf}'

def rewrite(text):
    lines=[]
    for line in text.splitlines():
        if ' & ' not in line:
            lines.append(line); continue
        name, rest=line.split(' & ',1)
        bare=re.sub(r'~?\\cite\{[^}]+\}','',name).strip()
        marker=r'$^\ast$' if bare.endswith(r'$^\ast$') else ''
        lookup=bare[:-len(marker)] if marker else bare
        key=KEYS.get(lookup)
        if key in CITE and key in VENUE:
            venue=VENUE[key].split(r'~\cite')[0]
            name=bare+' ('+venue+')'+r'~\cite{'+CITE[key]+'}'
        cells=rest.split(' & ')
        for i,cell in enumerate(cells):
            for method in ['StackMFF V2','StackMFF','CSNet','MFIF-GAN','DeFusion']:
                match=re.fullmatch(re.escape(method)+r'(\s*\\\\)?\s*',cell)
                if not match:continue
                key=KEYS[method]
                venue=VENUE[key].split(r'~\cite')[0]
                cells[i]=r'\shortstack{'+method+r'\\('+venue+r')~\cite{'+CITE[key]+'}}'+(match.group(1) or '')
                break
        rest=' & '.join(cells)
        lines.append(name+' & '+rest)
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('directory',nargs='?',default=str(Path(__file__).parent))
    args=ap.parse_args()
    changed=0
    for p in Path(args.directory).glob('tab_*.tex'):
        t=p.read_text(encoding='utf8');u=rewrite(t)
        if u!=t:
            p.write_text(u,encoding='utf8');changed+=1
    print(f'Updated method/venue labels in {changed} tables.')
