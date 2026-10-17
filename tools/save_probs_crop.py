# -*- coding: utf-8 -*-
"""Save the per-image probabilities P of one benchmark scene without and with smoothing (sigma 0.5), the colour
images and the reference, for the hard-versus-weighted fusion figure.
  python tools/save_probs_crop.py --dataset Middlebury --scene ArtL --out probs_ArtL.npz --bench_root <benchmarks>
"""
import argparse, os, re, sys
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import smooth_probs
from stackmff_v3.model_io import DEFAULT_WEIGHTS, load
from predict_benchmark import pad_to_64, index_gt

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='Mobile Depth'); ap.add_argument('--scene', default='keyboard')
ap.add_argument('--ckpt', default=DEFAULT_WEIGHTS); ap.add_argument('--run_dir', default='')
ap.add_argument('--bench_root', default=os.environ.get('STACKMFF_BENCH_ROOT', 'data/benchmarks'))
ap.add_argument('--sigma', type=float, default=0.5)
ap.add_argument('--out', required=True)
a = ap.parse_args()
root = os.path.join(a.bench_root, a.dataset)
d = os.path.join(root, 'image stack', a.scene)
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
fs = sorted([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.bmp'))], key=nat)
bgr = np.stack([cv2.imread(os.path.join(d, f)) for f in fs])
gray = np.stack([cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in bgr]).astype(np.float32) / 255
N, H, W = gray.shape
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
m = load(a.run_dir, a.ckpt, dev, K=1)
with torch.no_grad():
    P0 = m.forward_probs(pad_to_64(torch.from_numpy(gray)[None].to(dev)))[:, :, :H, :W].float()
    P1 = smooth_probs(P0, a.sigma)
gt = cv2.imread(index_gt(os.path.join(root, 'AiF'))[a.scene])
np.savez_compressed(a.out, P0=P0[0].cpu().numpy().astype(np.float16), P1=P1[0].cpu().numpy().astype(np.float16),
                    rgb=bgr[..., ::-1], gt=gt[..., ::-1], names=np.array(fs))
print('saved', N, H, W)
