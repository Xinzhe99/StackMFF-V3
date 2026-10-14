# -*- coding: utf-8 -*-
"""
Parameter and compute count for StackMFFV3.

MACs = thop (nn.Linear / nn.Conv2d / nn.LayerNorm modules)
     + attention matmul MACs (QK^T and AV inside F.scaled_dot_product_attention,
       counted by wrapping the function; thop does not see these).
Counted at input [1, N, H, W] using forward_logits (the network's training output).
"""
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import thop

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import StackMFFV3

_ATTN = {'macs': 0}
_orig_sdpa = F.scaled_dot_product_attention


def _counting_sdpa(q, k, v, attn_mask=None, **kw):
    lead = math.prod(q.shape[:-2])
    L, S, d = q.shape[-2], k.shape[-2], q.shape[-1]
    _ATTN['macs'] += 2 * lead * L * S * d          # QK^T and AV
    return _orig_sdpa(q, k, v, attn_mask=attn_mask, **kw)


class LogitsOnly(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_logits(x)


def count(model, N, H, W):
    model.eval()
    wrap = LogitsOnly(model)
    x = torch.rand(1, N, H, W)
    _ATTN['macs'] = 0
    F.scaled_dot_product_attention = _counting_sdpa
    try:
        with torch.no_grad():
            macs_mod, _ = thop.profile(wrap, inputs=(x,), verbose=False)
    finally:
        F.scaled_dot_product_attention = _orig_sdpa
    attn = _ATTN['macs']
    return macs_mod, attn, macs_mod + attn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--H', type=int, default=256)
    p.add_argument('--W', type=int, default=256)
    p.add_argument('--Ns', type=int, nargs='+', default=[2, 8, 24])
    p.add_argument('--inter_pool', type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(0)
    model = StackMFFV3(inter_pool=args.inter_pool)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params total: {n_params} ({n_params/1e6:.3f} M)")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Input H x W = {args.H} x {args.W}, inter_pool={args.inter_pool}")
    for N in args.Ns:
        mod, attn, tot = count(model, N, args.H, args.W)
        print(f"N={N:2d}: thop module MACs={mod/1e9:.4f} G | attention matmul MACs={attn/1e9:.4f} G | "
              f"total MACs={tot/1e9:.4f} G (FLOPs~{2*tot/1e9:.4f} G)")


if __name__ == '__main__':
    main()
