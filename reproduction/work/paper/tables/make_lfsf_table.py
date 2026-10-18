"""Comparison with LFSF (no public code) using the values reported in its paper (BT.601 luma, full image, the same four
benchmarks), next to our unified evaluation. -> tab_lfsf.tex"""
import csv, os

import os
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
DS = [('Mobile Depth', 'MD'), ('Middlebury', 'Mid.'), ('FlyingThings3D', 'FT3D'), ('Road-MF', 'Road-MF')]
# Li et al., Neurocomputing 707 (2027) 135167, main comparison table: (SSIM, PSNR)
REPORTED = {'LFSF': {'Mobile Depth': (0.9799, 38.86), 'Middlebury': (0.9546, 32.81), 'FlyingThings3D': (0.9663, 34.07), 'Road-MF': (0.9925, 38.26)},
            'LFSF-RF': {'Mobile Depth': (0.9766, 36.29), 'Middlebury': (0.9617, 34.68), 'FlyingThings3D': (0.9712, 35.43), 'Road-MF': (0.9981, 46.91)}}
OURS = [('Ours_final_soft', 'Ours (weighted)'), ('Ours_final_hard', 'Ours (hard selection)')]


def main():
    v = {}
    for r in csv.DictReader(open(f'{W}/eval/summary.csv')):
        if r['metric'] in ('SSIM', 'PSNR'):
            v[(r['method'], r['dataset'], r['metric'])] = float(r['mean'])
    rows = [(lab, {d: REPORTED[lab][d] for d, _ in DS}, True) for lab in REPORTED] + \
           [(lab, {d: (v[(m, d, 'SSIM')], v[(m, d, 'PSNR')]) for d, _ in DS}, False) for m, lab in OURS]
    best = {(d, i): max(r[1][d][i] for r in rows) for d, _ in DS for i in (0, 1)}
    L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\caption{Comparison with LFSF \cite{li2026lfsf}, for which no code is available, using the values reported '
         r'by its authors on the same benchmarks (BT.601 luma, full image). Each cell is SSIM / PSNR (dB). The reported '
         r'values come from a different evaluation pipeline and are not re-evaluated; the choice of pipeline alone can '
         r'move the PSNR by a few tenths of a dB (Section~\ref{sec:protocol}). $^{\ast}$Reported values. LFSF weights '
         r'the observed values with a softmax over the stack; LFSF-RF refines its output with a rectified-flow model '
         r'and therefore synthesizes intensities. Highest value of each quantity in bold.}',
         r'\label{tab:lfsf}', r'\begin{tabular}{lcccc}', r'\toprule',
         'Method & ' + ' & '.join(lab for _, lab in DS) + r' \\', r'\midrule']
    for lab, vals, reported in rows:
        cells = []
        for d, _ in DS:
            ssim, psnr = vals[d]
            a = ('\\textbf{%.4f}' % ssim) if abs(ssim - best[(d, 0)]) < 1e-9 else ('%.4f' % ssim)
            b = ('\\textbf{%.2f}' % psnr) if abs(psnr - best[(d, 1)]) < 1e-9 else ('%.2f' % psnr)
            cells.append('%s / %s' % (a, b))
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{HERE}/tab_lfsf.tex', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
