"""Export grayscale cases for Octave/MATLAB cross-validation (identical uint8 inputs for both sides)."""
import os, sys, json
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics_lib as M
from common import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases')
SCENES = [('Mobile Depth', 'balls'), ('Middlebury', 'Adirondack'), ('FlyingThings3D', '0000000'), ('Road-MF', '1')]
METH = ['StackMFF V4', 'CVT', 'Helicon Focus 8 - A', 'MFF-GAN']
CROP = 192  # Qabf.m and mef_ssim.m use per-pixel loops; crops keep Octave runtime small
cases = []
def save(a, p):
    Image.fromarray(a).save(p)
for ds, sc in SCENES:
    k = norm(sc)
    gt = M.load_gray(gt_index(ds)[k])
    layers = [M.resize_to(M.load_gray(f), gt.shape)[0] for f in stack_files(stack_index(ds)[k])]
    H, W = gt.shape
    y0, x0 = (H - CROP) // 2, (W - CROP) // 2
    cid0 = f'{norm(ds)}_{k}'
    d = os.path.join(OUT, cid0); os.makedirs(d, exist_ok=True)
    save(gt, os.path.join(d, 'gt.png'))
    save(gt[y0:y0+CROP, x0:x0+CROP], os.path.join(d, 'gt_crop.png'))
    for i, L in enumerate(layers):
        save(L[y0:y0+CROP, x0:x0+CROP], os.path.join(d, f'layer_crop_{i+1:02d}.png'))
    for m in METH:
        r = M.resize_to(M.load_gray(result_index(m, ds)[k]), gt.shape)[0]
        mm = norm(m)
        save(r, os.path.join(d, f'fused_{mm}.png'))
        save(r[y0:y0+CROP, x0:x0+CROP], os.path.join(d, f'fused_crop_{mm}.png'))
        cases.append({'dir': cid0, 'method': mm, 'n_layers': len(layers)})
json.dump(cases, open(os.path.join(OUT, 'cases.json'), 'w'), indent=1)
with open(os.path.join(OUT, 'cases.txt'), 'w') as f:
    for c in cases:
        f.write(f"{c['dir']} {c['method']} {c['n_layers']}\n")
print(len(cases), 'cases')
