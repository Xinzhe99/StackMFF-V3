# -*- coding: utf-8 -*-
"""Inference time of the final model with the protocol of the paper's efficiency table.

Protocol: the 11 Mobile Depth stacks at full resolution, data resident on the GPU before timing, batch 1, no_grad,
cudnn.benchmark off, TF32 off (strict fp32 unless --precision fp16), one warm-up run on the first stack (not
recorded), cuda.synchronize around the timed region; file I/O and model loading are excluded. Per stack:
RGB stack -> BT.601 grey [1,N,H,W] -> reflect pad to x64 -> forward_probs (scales averaged) -> Gaussian smoothing of
P (sigma) -> probability-weighted sum of the RGB images -> 8 bit.

  python tools/time_final.py --stack_root "<benchmarks>/Mobile Depth/image stack" [--scales 1] [--precision fp16]
"""
import argparse, json, os, re, sys, time
import cv2, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.model_io import DEFAULT_WEIGHTS, load  # noqa: E402
from stackmff_v3.network import smooth_probs  # noqa: E402

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def list_stacks(root):
    out = []
    for sc in sorted(os.listdir(root), key=natural_key):
        d = os.path.join(root, sc)
        if os.path.isdir(d):
            files = sorted([f for f in os.listdir(d) if f.lower().endswith(EXTS)], key=natural_key)
            out.append((sc, [os.path.join(d, f) for f in files]))
    return out


def load_gpu(paths, dev):
    ims = [torch.from_numpy(np.ascontiguousarray(cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1])).to(dev) for p in paths]
    torch.cuda.synchronize(dev)
    return ims


def pad_to_64(x):
    H, W = x.shape[-2:]
    ph, pw = (-H) % 64, (-W) % 64
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect' if (ph < H and pw < W) else 'replicate')
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack_root', required=True, help='folder with one sub-folder per stack (Mobile Depth/image stack)')
    ap.add_argument('--ckpt', default=DEFAULT_WEIGHTS); ap.add_argument('--run_dir', default='')
    ap.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    ap.add_argument('--scales', default='1,0.75', help='inference scales whose probabilities are averaged')
    ap.add_argument('--sigma', type=float, default=0.5, help='Gaussian smoothing of P (0: none)')
    ap.add_argument('--compile', action='store_true', help='torch.compile the network (compiled once per input size before timing)')
    ap.add_argument('--out', default='', help='optional JSON file with the per-stack times')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    model = load(a.run_dir, a.ckpt, dev)
    model.ms_scales = tuple(float(v) for v in a.scales.split(','))
    if a.compile:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256
        model.forward_logits = torch.compile(model.forward_logits, dynamic=False)

    def fuse_stack(imgs):
        st = torch.stack(imgs, 0).float()                                   # N,H,W,3 RGB 0..255
        H, W = st.shape[1:3]
        g = (0.299 * st[..., 0] + 0.587 * st[..., 1] + 0.114 * st[..., 2]).div_(255.0)[None]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=a.precision == 'fp16'):
            P = model.forward_probs(pad_to_64(g))[:, :, :H, :W].float()
        P = (smooth_probs(P, a.sigma) if a.sigma > 0 else P)[0]             # N,H,W
        return (P[..., None] * st).sum(0).clamp_(0, 255).round_().to(torch.uint8)

    stacks = list_stacks(a.stack_root)
    if a.compile:   # one compilation per input size, excluded from the timing like the warm-up
        for _, paths in stacks:
            fuse_stack(load_gpu(paths, dev))
    imgs0 = load_gpu(stacks[0][1], dev)
    fuse_stack(imgs0); torch.cuda.synchronize(dev); del imgs0
    times = []
    for sc, paths in stacks:
        imgs = load_gpu(paths, dev)
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        fuse_stack(imgs)
        torch.cuda.synchronize(dev); times.append(time.perf_counter() - t0)
        print(f'{sc}: N={len(paths)} t={times[-1]:.4f}s', flush=True)
        del imgs
    res = {'mean_s_per_stack': float(np.mean(times)), 'std_s': float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
           'per_stack_s': times, 'n_stacks': len(times), 'precision': a.precision, 'scales': model.ms_scales,
           'sigma': a.sigma, 'compile': a.compile, 'device': torch.cuda.get_device_name(dev)}
    print(json.dumps({k: v for k, v in res.items() if k != 'per_stack_s'}, indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
