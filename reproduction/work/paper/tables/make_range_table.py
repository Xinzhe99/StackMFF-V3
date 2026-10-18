"""Supplementary table: fused values outside the recorded range on the real stacks (real_eval/range_violation.py)."""
import csv, os
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(H, '../../real_eval/range_per_image.csv'))))
S = defaultdict(list)
for r in rows:
    S[(r['method'], r['dataset'])].append(float(r['frac_out']))
M = [('Ours', 'Ours (weighted)'), ('Ours_hard', 'Ours (hard sel.)'), ('StackMFF_V2', 'StackMFF V2'), ('StackMFF_V1', 'StackMFF'),
     ('CSNet', 'CSNet'), ('MFIF-GAN', 'MFIF-GAN'), ('DeFusion', 'DeFusion')]
D = [('CCCID', 'CCCID (100)'), ('WBC', 'WBC (200)'), ('Araujo', 'Camera bursts (10)')]
L = [r'\begin{table}[t]', r'\centering', r'\scriptsize',
     r'\caption{Fused values outside a common input envelope on the real stacks (percentage of color-channel values; tolerance 2 gray levels). The envelope includes both Pillow and OpenCV decodings of the registered files, so JPEG decoder differences are not counted as fusion errors.}',
     r'\label{tab:range}', r'\begin{tabular}{lccc}', r'\toprule', 'Method & ' + ' & '.join(d[1] for d in D) + r' \\', r'\midrule']
for k, n in M:
    L.append(n + ' & ' + ' & '.join('%.2f' % (100 * sum(S[(k, d)]) / len(S[(k, d)])) for d, _ in D) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(os.path.join(H, 'tab_range.tex'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
