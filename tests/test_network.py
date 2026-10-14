# -*- coding: utf-8 -*-
"""Unit tests for StackMFFV3: shapes, permutation equivariance, training-step memory."""
import sys
import time
import torch
import torch.nn.functional as F

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import StackMFFV3

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
dev = torch.device('cuda:0')
model = StackMFFV3().to(dev)
fails = []

print('=== (a) shapes ===')
for N in [2, 7, 24]:
    for (H, W) in [(256, 256), (360, 640)]:
        x = torch.rand(2, N, H, W, device=dev)
        model.train()
        with torch.no_grad():
            lg = model(x)
        model.eval()
        with torch.no_grad():
            fused, focus = model(x)
            fs, probs = model.fuse(x, 'soft')[0], model.forward_probs(x)
        ok = (lg.shape == (2, N, H, W) and fused.shape == (2, 1, H, W) and focus.shape == (2, H, W)
              and focus.dtype == torch.long and fs.shape == (2, 1, H, W)
              and torch.allclose(probs.sum(1), torch.ones_like(probs[:, 0]), atol=1e-5))
        print(f'N={N:2d} HxW={H}x{W}: logits{tuple(lg.shape)} fused{tuple(fused.shape)} '
              f'focus{tuple(focus.shape)} soft{tuple(fs.shape)} -> {"PASS" if ok else "FAIL"}')
        if not ok:
            fails.append(f'shape N={N} {H}x{W}')
        # hard fusion must equal gather from input
        g = torch.gather(x, 1, focus.unsqueeze(1))
        if not torch.equal(g, fused):
            fails.append(f'hard gather mismatch N={N} {H}x{W}')

print('=== (b) permutation equivariance over N (eval mode, fp32) ===')
model.eval()
for N in [2, 7, 24]:
    for (H, W) in [(256, 256), (360, 640)]:
        x = torch.rand(1, N, H, W, device=dev)
        perm = torch.randperm(N, device=dev)
        with torch.no_grad():
            l1 = model.forward_logits(x)
            l2 = model.forward_logits(x[:, perm])
        err = (l2 - l1[:, perm]).abs().max().item()
        scale = l1.abs().max().item()
        ok = err < 1e-4
        print(f'N={N:2d} HxW={H}x{W}: max|logits(x[perm]) - logits(x)[perm]| = {err:.3e} '
              f'(logit scale {scale:.3f}) -> {"PASS" if ok else "FAIL"}')
        if not ok:
            fails.append(f'equivariance N={N} {H}x{W} err={err}')

print('=== (d) training-step peak GPU memory: batch 8, N=24, 256x256, bf16 autocast ===')
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.rand(8, 24, 256, 256, device=dev)
y = torch.randint(0, 24, (8, 256, 256), device=dev)
torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
times = []
for it in range(4):
    t0 = time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize()
    times.append(time.time() - t0)
peak = torch.cuda.max_memory_allocated()
rel_grad = model.stage1[0].intra_a.rel_pos_bias_table.grad
ok = torch.isfinite(loss).item() and rel_grad is not None and torch.isfinite(rel_grad).all().item()
print(f'loss={loss.item():.4f} finite grads incl. rel-pos table={ok}')
print(f'base allocated={base/2**30:.3f} GiB, peak allocated={peak/2**30:.3f} GiB, '
      f'step time (last 3 avg, incl. first warm-up excluded)={sum(times[1:])/3:.3f} s')
if not ok:
    fails.append('grad check')

print('=== SUMMARY ===')
print('ALL PASS' if not fails else f'FAILURES: {fails}')
