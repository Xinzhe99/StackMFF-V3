"""Supplementary table from eval/stats.md: Friedman test, Nemenyi CD, average ranks and Wilcoxon-Holm results."""
import re
import os
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = 'Ours_final_soft'
blocks, cur = [], None
for l in open(f'{W}/eval/stats.md'):
    m = re.match(r'## (.+?) / (\w+): n=(\d+) images, k=(\d+) methods, Friedman p=([0-9.e+-]+), Nemenyi CD=([0-9.na]+)', l)
    if m:
        cur = dict(ds=m.group(1), met=m.group(2), n=int(m.group(3)), k=int(m.group(4)), fp=float(m.group(5)), cd=float(m.group(6)), rows=[]); blocks.append(cur); continue
    if cur and l.startswith('| ') and not l.startswith('| method') and not l.startswith('|---'):
        c = [x.strip() for x in l.strip().strip('|').split('|')]
        cur['rows'].append((c[0], float(c[2]), float(c[3]) if c[3] else None))
KMD = next((b['k'] for b in blocks if b['ds'].startswith('Mobile')), 0)
KOTHER = next((b['k'] for b in blocks if b['ds'].startswith('Flying')), 0)
DISP = {'Ours_final_hard': 'Ours (hard)', 'StackMFF_release': 'StackMFF', 'Helicon Focus 8 - A': 'Helicon (A)', 'Helicon Focus 8 - B': 'Helicon (B)'}
L = [r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
     '\\caption{Statistical comparison over the scenes of each benchmark (%d entries on Mobile Depth and Middlebury, %d on the other two). Friedman test over all entries, Nemenyi critical difference (CD, $\\alpha=0.05$) of the average ranks, average rank of our model, the best-ranked competing entry, the number of entries whose average rank is within the CD of ours (including our hard-selection variant). With so many entries and only 11 or 15 scenes, the Nemenyi test is conservative on Mobile Depth and Middlebury; the paired Wilcoxon tests in the main text are the more sensitive comparison.}' % (KMD, KOTHER),
     r'\label{tab:stats}', r'\begin{tabular}{llcccccc}', r'\toprule',
     r'Benchmark & Metric & $n$ & Friedman $p$ & CD & Rank (ours) & Next best (rank) & Within CD \\', r'\midrule']
for b in blocks:
    rk = {m: r for m, r, _ in b['rows']}
    our = rk[REF]
    others = sorted([(r, m) for m, r, _ in b['rows'] if m not in (REF, 'Ours_final_hard')])
    within = sum(1 for m, r in rk.items() if m != REF and r - our <= b['cd'])
    pmax = max(p for m, _, p in b['rows'] if p is not None and m != 'Ours_final_hard')
    fp = '$<10^{-300}$' if b['fp'] == 0 else f"${b['fp']:.1e}$".replace('e-', r'\times10^{-') .replace('$', '$', 1)
    if b['fp'] != 0:
        mant, ex = f"{b['fp']:.1e}".split('e'); fp = f'${mant}\\times10^{{{int(ex)}}}$'
    L.append(f"{b['ds']} & {b['met']} & {b['n']} & {fp} & {b['cd']:.1f} & {our:.2f} & {DISP.get(others[0][1], others[0][1])} ({others[0][0]:.2f}) & {within} \\\\")
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{W}/paper/tables/tab_supp_stats.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[10:]))
