"""Fusion strategies on the final model (no retraining): hard selection, weighted fusion with Gaussian smoothing
sigma of the probabilities. Benchmark scores as in training monitoring (gray, skimage SSIM defaults)."""
import csv
import os
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = [('Mobile Depth', 'MD'), ('Middlebury', 'Mid.'), ('FlyingThings3D', 'FT3D'), ('Road-MF', 'Road-MF')]
import os
FINAL = os.environ.get('FINAL', '1') == '1'
EP_CSV = f'{W}/new_model/final_epochs.csv' if FINAL else f'{W}/new_model/Hms_epochs.csv'
SM_CSV = f'{W}/figures/data/final/smooth.csv' if FINAL else f'{W}/new_model/smooth.csv'
eps = list(csv.DictReader(open(EP_CSV)))
sel = max((r for r in eps if r['fusion'] == 'soft'), key=lambda r: float(r['mean_SSIM']))['epoch']
hard = next(r for r in eps if r['epoch'] == sel and r['fusion'] == 'hard')
rows = [('Hard selection', hard)]
for r in csv.DictReader(open(SM_CSV)):
    s = float(r['sigma'])
    rows.append((('Weighted, no smoothing' if s == 0 else f'Weighted, $\\sigma={s:g}$') + (' (default)' if s == 0.5 else ''), r))
L = [r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{2.5pt}',
     r'\caption{Fusion strategies applied to the same trained model (SSIM / PSNR; gray images, SSIM with the default $7\times7$ window, as in training monitoring). Best and second-best per column in pink and blue.}',
     r'\label{tab:fusion}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{l' + 'c' * (len(DS) + 1) + '}', r'\toprule',
     'Fusion & ' + ' & '.join(d for _, d in DS) + r' & Mean \\', r'\midrule']
def hl(x, col, fmt):
    srt = sorted({round(float(r_[col]), 6) for _, r_ in rows}, reverse=True)
    t = fmt.format(x); k = srt.index(round(x, 6)) + 1
    return r'\best{' + t + '}' if k == 1 else (r'\second{' + t + '}' if k == 2 else t)
for name, r in rows:
    cells = [hl(float(r[d + '_SSIM']), d + '_SSIM', '{:.4f}') + ' / ' + hl(float(r[d + '_PSNR']), d + '_PSNR', '{:.2f}') for d, _ in DS]
    cells.append(hl(float(r['mean_SSIM']), 'mean_SSIM', '{:.4f}') + ' / ' + hl(float(r['mean_PSNR']), 'mean_PSNR', '{:.2f}'))
    L.append(name + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
open(f'{W}/paper/tables/tab_fusion.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[10:]))
