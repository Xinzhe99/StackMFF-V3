"""Real stacks (no reference): Q^{X/F} and MEF-SSIM per dataset. Source: work/real_eval/real_summary.md (server)."""
import os
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHODS = [('Ours', 'Ours (weighted)'), ('Ours_hard', 'Ours (hard sel.)'), ('StackMFF_V2', 'StackMFF V2'), ('StackMFF_V1', 'StackMFF'),
           ('CSNet', 'CSNet'), ('MFIF-GAN', 'MFIF-GAN'), ('DeFusion', 'DeFusion')]
DS = [('CCCID', 'CCCID (100)'), ('WBC', 'WBC (200)'), ('Araujo', 'Camera bursts (10)')]
v = {}
for l in open(f'{W}/real_eval/real_summary_reg.md'):
    c = [x.strip() for x in l.strip().strip('|').split('|')]
    if len(c) == 5 and c[0] in dict(DS):
        v[(c[0], c[1])] = (float(c[3]), float(c[4]))
L = [r'\begin{table}[!htbp]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
     r'\caption{No-reference evaluation on real focal stacks ($Q^{\mathbf{X}/F}$ / MEF-SSIM, higher is better; best and second-best highlighted in pink and blue). Number of stacks in parentheses.}',
     r'\label{tab:real}', r'\begin{tabular}{l' + 'c' * len(DS) + '}', r'\toprule',
     'Method & ' + ' & '.join(d for _, d in DS) + r' \\', r'\midrule']
srt = {(d, i): sorted({v[(d, m)][i] for m, _ in METHODS}, reverse=True) for d, _ in DS for i in (0, 1)}
hl = lambda x, d, i: (r'\best{%.3f}' if x == srt[(d, i)][0] else (r'\second{%.3f}' if x == srt[(d, i)][1] else '%.3f')) % x
for m, disp in METHODS:
    cells = []
    for d, _ in DS:
        q, e = v[(d, m)]
        fq = hl(q, d, 0)
        fe = hl(e, d, 1)
        cells.append(f'{fq} / {fe}')
    L.append(disp + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{W}/paper/tables/tab_real.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[9:]))
