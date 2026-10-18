"""Robustness table: change of SSIM / PSNR w.r.t. the unperturbed stacks, pooled over the 206 benchmark scenes
(scene-weighted mean over datasets; shuffle seeds averaged). Source: work/robustness/robust_summary.csv."""
import csv
from collections import defaultdict
import os
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUPS = [('Original', ['original']), ('Shuffled', ['shuffle_s0', 'shuffle_s1', 'shuffle_s2']), ('Reversed', ['reverse']),
          ('+2 blurred', ['distract_blur_k2']), ('+4 blurred', ['distract_blur_k4']),
          ('+2 dupl.', ['distract_dup_k2']), ('+4 dupl.', ['distract_dup_k4']),
          ('$-$25\\%', ['drop_mid25']), ('$-$50\\%', ['drop_mid50']),
          ('8 images', ['subsample_n8']), ('4 images', ['subsample_n4']), ('2 images', ['subsample_n2'])]
METHODS = [('Ours_final', 'Ours'), ('StackMFF V2', 'StackMFF V2'), ('StackMFF_release', 'StackMFF'),
           ('CSNet', 'CSNet'), ('MFIF-GAN', 'MFIF-GAN'), ('DeFusion', 'DeFusion')]
acc = defaultdict(lambda: [0.0, 0.0, 0])
for r in csv.DictReader(open(f'{W}/eval/summary.csv')):   # unperturbed, scene-weighted over the 4 benchmarks
    if r['metric'] in ('SSIM', 'PSNR'):
        a = acc[('original', 'Ours_final' if r['method'] == 'Ours_final_soft' else r['method'])]; n = int(r['n_images'])
        a[0 if r['metric'] == 'SSIM' else 1] += n * float(r['mean'])
        if r['metric'] == 'SSIM': a[2] += n
for r in csv.DictReader(open(f'{W}/robustness/robust_summary.csv')):
    n = int(r['n']); a = acc[(r['variant'], r['method'])]
    a[0] += n * float(r['SSIM']); a[1] += n * float(r['PSNR']); a[2] += n
def val(vs, m, i):
    xs = [acc[(v, m)][i] / acc[(v, m)][2] for v in vs if acc[(v, m)][2]]
    return sum(xs) / len(xs) if xs else float('nan')
def table(i, label, cap, fmt, star):
    tab = {m: [val(vs, m, i) for _, vs in GROUPS] for m, _ in METHODS}
    L = [r'\begin{table}[!htbp]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}', cap, label,
         r'\begin{tabular}{l' + 'c' * len(METHODS) + '}', r'\toprule',
         'Perturbation & ' + ' & '.join(d for _, d in METHODS) + r' \\', r'\midrule']
    for j, (g, _) in enumerate(GROUPS):
        row = [tab[m][j] for m, _ in METHODS]
        srt = sorted({round(x, 6) for x in row}, reverse=True)
        cells = []
        for x in row:
            t = fmt.format(x); rk = srt.index(round(x, 6)) + 1
            cells.append(r'\best{' + t + '}' if rk == 1 else (r'\second{' + t + '}' if rk == 2 else t))
        L.append(g + ' & ' + ' & '.join(cells) + r' \\')
        if j == 0:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L) + '\n'


NOTE = r'averaged over the 206 scenes of the four benchmarks, i.e.\ weighted by the size of each benchmark (FlyingThings3D and Road-MF contribute 87\% of the scenes), with three seeds for shuffling; the values therefore differ from the per-benchmark means of the main comparison. Distractors are blurred copies or exact duplicates of random images; $-25\%$/$-50\%$ remove the central images; $n$ images keeps $n$ uniformly spaced images. Best and second-best per row are highlighted in pink and blue.'
open(f'{W}/paper/tables/tab_robust.tex', 'w').write(table(0, r'\label{tab:robust}', r'\caption{Robustness to perturbed stacks: SSIM ' + NOTE + '}', '{:.3f}', False))
open(f'{W}/paper/tables/tab_supp_robust_psnr.tex', 'w').write(table(1, r'\label{tab:robust_psnr}', r'\caption{Robustness to perturbed stacks: PSNR (dB) ' + NOTE + '}', '{:.2f}', False))
print(open(f'{W}/paper/tables/tab_robust.tex').read())
