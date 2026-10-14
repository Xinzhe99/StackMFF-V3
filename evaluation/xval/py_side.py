"""Python side of the cross-validation: same gray cases, metrics_lib implementations."""
import os, sys, csv
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import metrics_lib as M
rd = lambda p: np.array(Image.open(p), dtype=np.uint8)
rows = []
for line in open(os.path.join(HERE, 'cases', 'cases.txt')):
    d, m, n = line.split(); n = int(n); D = os.path.join(HERE, 'cases', d)
    gt = rd(f'{D}/gt.png'); f = rd(f'{D}/fused_{m}.png'); fc = rd(f'{D}/fused_crop_{m}.png')
    Ls = [rd(f'{D}/layer_crop_{k:02d}.png') for k in range(1, n + 1)]
    mi, nmi = M.mi_nmi(gt, f)
    g2 = [M.grad_orient(Ls[0]), M.grad_orient(Ls[1])]
    rows.append(dict(dir=d, method=m, SSIM=M.ssim_wang(f, gt), PSNR=M.psnr(f, gt), VIF=M.vifp(gt, f), MI=mi, NMI=nmi,
                     Qabf2_crop_matlabrule=M.qabf_stack(g2, fc, rule='matlab'),
                     Qabf2_crop_minmax=M.qabf_stack(g2, fc, rule='minmax'),
                     QabfN_crop_minmax=M.qabf_stack([M.grad_orient(x) for x in Ls], fc, rule='minmax'),
                     MEF_crop=M.mef_from_field(M.mef_precompute(np.stack(Ls)), fc)[0]))
    print(d, m, 'ok', flush=True)
with open(os.path.join(HERE, 'python_results.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
