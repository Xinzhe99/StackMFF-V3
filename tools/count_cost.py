# -*- coding: utf-8 -*-
"""Parameters and GFLOPs of the final model for one fusion call (default: two 256x256 images), at one inference scale
and at the two default scales (1, 0.75), as reported in the efficiency table of the paper (0.156 M, 1.94 / 3.03 GFLOPs).

GFLOPs: torch.utils.flop_counter.FlopCounterMode (all matrix products and convolutions, FLOPs = 2 x MACs) over
forward_probs. FlopCounterMode does not see the math fallback of scaled_dot_product_attention on the CPU, so the
attention products are counted with an explicit implementation of the same mathematics (--sdpa-native: uncorrected).
--thop additionally reports thop MACs (module convention of earlier work: Linear/Conv/LayerNorm, no attention products).

  python tools/count_cost.py [--N 2 --size 256] [--thop]
  python tools/count_cost.py --override single_scale=False fine_head=True     # an ablation configuration
"""
import argparse, ast, os, sys
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.model_io import FINAL_ARGS, build  # noqa: E402


def manual_sdpa(q, k, v, attn_mask=None, **kw):
    s = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    if attn_mask is not None:
        s = s + attn_mask
    return torch.softmax(s, dim=-1) @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=2); ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--override', nargs='*', default=[], help='architecture arguments, e.g. inter_pool=1 intra=conv')
    ap.add_argument('--sdpa-native', dest='native', action='store_true')
    ap.add_argument('--thop', action='store_true')
    a = ap.parse_args()
    if not a.native:
        F.scaled_dot_product_attention = manual_sdpa
    cfg = dict(FINAL_ARGS)
    for kv in a.override:
        k, v = kv.split('=', 1)
        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v
    torch.set_num_threads(2)
    m = build(cfg).eval()
    params = sum(p.numel() for p in m.parameters())
    x = torch.rand(1, a.N, a.size, a.size)
    out = {}
    for label, ms in (('one scale', (1.0,)), ('two scales (1, 0.75)', (1.0, 0.75))):
        m.ms_scales = ms
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m.forward_probs(x)
        out[label] = fc.get_total_flops() / 1e9
    print(f'config: {cfg}')
    print(f'params {params / 1e6:.3f} M ({params})')
    for k, v in out.items():
        print(f'GFLOPs, {k}, N={a.N}, {a.size}x{a.size}: {v:.2f}')
    if a.thop:
        import thop

        class Wrap(torch.nn.Module):
            def __init__(s, m):
                super().__init__(); s.m = m

            def forward(s, x):
                return s.m.forward_probs(x)
        for ms in ((1.0,), (1.0, 0.75)):
            m.ms_scales = ms
            macs, _ = thop.profile(Wrap(m), inputs=(x,), verbose=False)
            print(f'thop MACs, scales {ms}: {macs / 1e9:.2f} G')


if __name__ == '__main__':
    main()
