"""Supplementary table: benchmark scores (old convention, gray, skimage SSIM) of every epoch of the final run,
weighted fusion, with the epoch selected on the benchmarks and the epoch selected on the held-out split marked."""
import csv
import os
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = [('Mobile Depth', 'MD'), ('Middlebury', 'Mid'), ('FlyingThings3D', 'FT3D'), ('Road-MF', 'Road-MF')]
import os, re
SRC = os.environ.get('EPOCHS_CSV', f'{W}/new_model/final_epochs.csv')
LOG = os.environ.get('TRAIN_LOG', f'{W}/new_model/final_s1nh_train.out')
rows = [r for r in csv.DictReader(open(SRC)) if r['fusion'] == 'soft']
# epoch with the best held-out pixel accuracy, from the training log ("best val acc X @ epoch E")
VAL_BEST = None
if os.path.exists(LOG):
    m = re.findall(r'best val acc [0-9.]+ @ epoch (\d+)', open(LOG).read())
    VAL_BEST = int(m[-1]) if m else None
if VAL_BEST is None:
    raise RuntimeError(f'Cannot establish held-out best epoch from {LOG}; fetch the final_s1nh training log first.')
sel = max(rows, key=lambda r: float(r['mean_SSIM']))['epoch']
L = [r'\begin{table}[t]', r'\centering', r'\caption{Benchmark scores (SSIM / PSNR, weighted fusion with $\sigma=0.5$) of every training epoch of the final model. '
     r'$^\dagger$: epoch selected by the mean SSIM over the four benchmarks (reported in the paper); $^\ddagger$: epoch with the highest pixel accuracy on the held-out synthetic split. '
     r'Scores computed on gray images with the SSIM defaults used during training monitoring, so they differ slightly from the main tables.}',
     r'\label{tab:epochs}', r'\scriptsize', r'\begin{tabular}{l' + 'c' * (len(DS) + 1) + '}', r'\toprule',
     'Epoch & ' + ' & '.join(s for _, s in DS) + r' & Mean \\', r'\midrule']
for r in rows:
    e = int(r['epoch'])
    marks = (r'\dagger' if r['epoch'] == sel else '') + (r'\ddagger' if e == VAL_BEST else '')
    mark = '$^{' + marks + '}$' if marks else ''
    cells = [f"{float(r[d + '_SSIM']):.4f} / {float(r[d + '_PSNR']):.2f}" for d, _ in DS]
    L.append(f'{e + 1}{mark} & ' + ' & '.join(cells) + f" & {float(r['mean_SSIM']):.4f} / {float(r['mean_PSNR']):.2f}" + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{W}/paper/tables/tab_supp_epochs.tex', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
