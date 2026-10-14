# Extra speed check on an idle GPU: Mobile Depth real stacks, batch 1, fp32 vs fp16 autocast.
import time, numpy as np, torch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from size_candidates import md_stacks, Wrap
from stackmff_v3.network import StackMFFV3
dev = torch.device('cuda:0'); stacks = md_stacks()
cfgs = [('A C32/64 p1', dict(C1=32, C2=64)), ('G C32/64 p2', dict(C1=32, C2=64, inter_pool=2)),
        ('H C48/96 p2', dict(C1=48, C2=96, heads=(3, 6), inter_pool=2)), ('I C40/80 p2', dict(C1=40, C2=80, inter_pool=2)),
        ('D C48/96 p1', dict(C1=48, C2=96, heads=(3, 6)))]
for name, c in cfgs:
    m = Wrap(StackMFFV3(**c), True).eval().to(dev)
    npar = sum(p.numel() for p in m.parameters()) / 1e6
    res = []
    for amp in (False, True):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            for s in stacks[:2]: m(torch.from_numpy(s)[None].to(dev))
            ts = []
            for s in stacks:
                x = torch.from_numpy(s)[None].to(dev); torch.cuda.synchronize(); t = time.time(); m(x); torch.cuda.synchronize(); ts.append(time.time() - t)
        res.append(np.mean(ts))
    print(f'| {name} | {npar:.3f} | {res[0]:.4f} | {res[1]:.4f} |', flush=True)
