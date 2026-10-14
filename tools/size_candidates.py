# -*- coding: utf-8 -*-
"""Size candidates for StackMFFV3 vs StackMFF V4, under one counting convention.

thop_MACs : thop module MACs only (the convention behind the old tables: V2 2.75 G, V4 0.51 G, N=2, 256x256);
            thop does not see attention matmuls.
full_GFLOPs: torch.utils.flop_counter.FlopCounterMode (all matmul/conv/SDPA, FLOPs = 2*MACs).
time_MD   : mean forward time (fp32, batch 1, eval, cuda sync) over the 11 real Mobile Depth stacks
            at native resolution. GPUs are shared with other jobs, so times are indicative only.
"""
import glob, os, sys, time, re
import numpy as np, torch, thop, cv2
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import StackMFFV3

V4DIR = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_image_stack_fusion/StackMFFV4'
MD = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/dff_and_mff/Mobile Depth/image stack'


def load_v4():
    sys.path.insert(0, V4DIR)
    from ablation_models.network_spatial_pool_ratio_0_0625 import StackMFF_V4
    return StackMFF_V4()


class Wrap(torch.nn.Module):
    def __init__(self, m, ours):
        super().__init__(); self.m, self.ours = m, ours
    def forward(self, x):
        return self.m.forward_logits(x) if self.ours else self.m(x)


def nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def md_stacks():
    out = []
    for d in sorted(os.listdir(MD)):
        fs = sorted([f for f in glob.glob(os.path.join(MD, d, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))], key=nat)
        out.append(np.stack([cv2.imread(f, 0) for f in fs]).astype(np.float32) / 255.0)
    return out


def measure(name, model, ours, stacks, dev):
    w = Wrap(model, ours).eval()
    n_par = sum(p.numel() for p in model.parameters())
    x = torch.rand(1, 2, 256, 256)
    with torch.no_grad():
        macs, _ = thop.profile(Wrap(model, ours).eval(), inputs=(x,), verbose=False)
    w = w.to(dev)
    fl = {}
    for N in (2, 8):
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            w(torch.rand(1, N, 256, 256, device=dev))
        fl[N] = fc.get_total_flops() / 1e9
    ts = []
    with torch.no_grad():
        for s in stacks[:2]:
            w(torch.from_numpy(s)[None].to(dev))
        for s in stacks:
            xx = torch.from_numpy(s)[None].to(dev)
            torch.cuda.synchronize(); t0 = time.time()
            w(xx)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
    print(f'| {name} | {n_par/1e6:.3f} | {macs/1e9:.3f} | {fl[2]:.3f} | {fl[8]:.3f} | {np.mean(ts):.4f} |', flush=True)


if __name__ == '__main__':
    dev = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False
    stacks = md_stacks()
    print(f'Mobile Depth stacks: {len(stacks)}, N = {[s.shape[0] for s in stacks]}, size {stacks[0].shape[1:]}')
    print('| model | params (M) | thop MACs N=2 256^2 (G) | full GFLOPs N=2 | full GFLOPs N=8 | time MD (s) |')
    print('|---|---|---|---|---|---|')
    try:
        measure('StackMFF V4 (ablation_models/network_spatial_pool_ratio_0_0625)', load_v4(), False, stacks, dev)
    except Exception as e:
        print('V4 failed:', repr(e))
    cfgs = [('A C32/64 r(1,2,1)', dict(C1=32, C2=64, rounds=(1, 2, 1))),
            ('B C32/64 r(1,3,1)', dict(C1=32, C2=64, rounds=(1, 3, 1))),
            ('C C40/80 r(1,2,1)', dict(C1=40, C2=80, rounds=(1, 2, 1), heads=(2, 4))),
            ('D C48/96 r(1,2,1)', dict(C1=48, C2=96, rounds=(1, 2, 1), heads=(3, 6))),
            ('E C48/96 r(1,3,1)', dict(C1=48, C2=96, rounds=(1, 3, 1), heads=(3, 6))),
            ('F C32/96 r(1,3,1)', dict(C1=32, C2=96, rounds=(1, 3, 1), heads=(2, 6))),
            ('G C32/64 r(1,2,1) inter_pool=2', dict(C1=32, C2=64, rounds=(1, 2, 1), inter_pool=2))]
    for n, c in cfgs:
        torch.manual_seed(0)
        measure('ours ' + n, StackMFFV3(**c), True, stacks, dev)
