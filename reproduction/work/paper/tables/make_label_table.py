"""Supplementary table: focus-label accuracy on the rendered benchmarks (eval/label_acc/label_acc.py, tau=5)."""
import csv, os
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '../../eval/label_acc')
def mean(f, ds, c):
    r = [float(x[c]) for x in csv.DictReader(open(os.path.join(D, f))) if x['dataset'] == ds]
    return sum(r) / len(r)
DS = ['Middlebury', 'FlyingThings3D', 'Road-MF']
L = [r'\begin{table}[t]', r'\centering', r'\scriptsize',
     r'\caption{Focus-label accuracy on the rendered benchmarks. The reference label of a pixel is the input image closest to the reference (mean absolute difference over the color channels); only decisive pixels are counted, where the second-closest image is at least 5 gray levels further away (fraction of the pixels in the last row). The label of a method is the input image closest to its result. Exact agreement / agreement within one image (\%).}',
     r'\label{tab:label}', r'\begin{tabular}{lccc}', r'\toprule', 'Method & ' + ' & '.join(DS) + r' \\', r'\midrule']
for f, n in (('ours.csv', 'Ours (hard sel.)'), ('v2.csv', 'StackMFF V2')):
    L.append(n + ' & ' + ' & '.join('%.1f / %.1f' % (100 * mean(f, d, 'exact_t5'), 100 * mean(f, d, 'within1_t5')) for d in DS) + r' \\')
L += [r'\midrule', 'Decisive pixels & ' + ' & '.join('%.1f' % (100 * mean('ours.csv', d, 'frac_decisive_t5')) for d in DS) + r' \\',
      r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(os.path.join(H, 'tab_label.tex'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[8:12]))
