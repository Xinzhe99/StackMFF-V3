# -*- coding: utf-8 -*-
"""Unit tests for the final StackMFF V3 model: shapes, probabilities, permutation equivariance of the logits and order
invariance of the fused images (hard and weighted fusion with two-scale inference and smoothing), parameter count,
and (CUDA only) one training step in bf16 autocast. Random weights unless the released weights are present.

  python tests/test_network.py        (or: python -m pytest tests)
Runs on the CPU when no GPU is available (smaller inputs).
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.model_io import DEFAULT_WEIGHTS, FINAL_ARGS, build, load_checkpoint  # noqa: E402
from stackmff_v3.network import smooth_probs  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SIZES = [(256, 256), (360, 640)] if DEV.type == 'cuda' else [(128, 128), (90, 150)]
NS = [2, 7, 24] if DEV.type == 'cuda' else [2, 7, 13]


def make_model():
    torch.manual_seed(0)
    if os.path.exists(DEFAULT_WEIGHTS):
        return load_checkpoint(DEFAULT_WEIGHTS, DEV)
    return build(FINAL_ARGS).to(DEV).eval()


def test_param_count():
    m = build(FINAL_ARGS)
    n = sum(p.numel() for p in m.parameters())
    assert n == 156172, n
    assert m.single_scale and not m.fine_head and m.cur_K == 1 and tuple(m.ms_scales) == (1.0, 0.75)


def test_shapes_and_probs():
    m = make_model()
    for N in NS:
        for H, W in SIZES:
            x = torch.rand(2, N, H, W, device=DEV)
            with torch.no_grad():
                lg = m.forward_logits(x)
                fused, focus = m(x)
                soft = m.fuse(x, 'soft')[0]
                P = m.forward_probs(x, sigma=0.5)
            assert lg.shape == (2, N, H, W) and fused.shape == (2, 1, H, W) and focus.shape == (2, H, W)
            assert focus.dtype == torch.long and soft.shape == (2, 1, H, W) and P.shape == (2, N, H, W)
            assert torch.allclose(P.sum(1), torch.ones_like(P[:, 0]), atol=1e-5)
            assert torch.equal(torch.gather(x, 1, focus.unsqueeze(1)), fused)        # hard fusion copies inputs
            assert (soft >= x.amin(1, keepdim=True) - 1e-6).all() and (soft <= x.amax(1, keepdim=True) + 1e-6).all()


def test_equivariance_and_order_invariance():
    """Exact in float64 (up to rounding); in float32 the deviations are rounding errors of the attention sums."""
    m = make_model()
    g = torch.Generator().manual_seed(1)
    worst32 = worst64 = 0.0
    for N in NS:
        for H, W in SIZES:
            x = torch.rand(1, N, H, W, generator=g).to(DEV)
            perm = torch.randperm(N, generator=g).to(DEV)
            with torch.no_grad():
                l1, l2 = m.forward_logits(x), m.forward_logits(x[:, perm])
                P1, P2 = m.forward_probs(x), m.forward_probs(x[:, perm])            # two scales (1, 0.75)
                s1 = (smooth_probs(P1, 0.5) * x).sum(1)
                s2 = (smooth_probs(P2, 0.5) * x[:, perm]).sum(1)
            err = (l2 - l1[:, perm]).abs().max().item() / max(1.0, l1.abs().max().item())   # relative to the logit scale
            worst32 = max(worst32, err)
            assert err < 1e-4, (N, H, W, err)
            assert (P2 - P1[:, perm]).abs().max().item() < 1e-4
            assert (s2 - s1).abs().max().item() < 1e-4                                # weighted fusion: order-free
            h1 = torch.gather(x, 1, P1.argmax(1, keepdim=True))
            h2 = torch.gather(x[:, perm], 1, P2.argmax(1, keepdim=True))
            assert (h1 != h2).float().mean().item() < 1e-3                            # hard selection (near-ties aside)
    m64 = m.double()
    for N in NS:
        x = torch.rand(1, N, *SIZES[0], generator=g, dtype=torch.float64).to(DEV)
        perm = torch.randperm(N, generator=g).to(DEV)
        with torch.no_grad():
            l1, l2 = m64.forward_logits(x), m64.forward_logits(x[:, perm])
        worst64 = max(worst64, (l2 - l1[:, perm]).abs().max().item())
    assert worst64 < 1e-9, worst64
    print(f'max deviation of logits(x[perm]) from logits(x)[perm]: float32 {worst32:.1e} (relative), '
          f'float64 {worst64:.1e}')


def test_train_step():
    if DEV.type != 'cuda':
        print('skipped (no GPU)')
        return
    m = build(FINAL_ARGS).to(DEV).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.rand(8, 24, 256, 256, device=DEV)
    y = torch.randint(0, 24, (8, 256, 256), device=DEV)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = m(x)
    loss = F.cross_entropy(logits.float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    g = m.u4a.intra_a.rel_pos_bias_table.grad
    assert torch.isfinite(loss).item() and g is not None and torch.isfinite(g).all().item()
    print(f'train step OK, peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB (batch 8, N=24, 256x256)')


if __name__ == '__main__':
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f'device {DEV}, weights {"released" if os.path.exists(DEFAULT_WEIGHTS) else "random"}')
    fails = []
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith('test_')]:
        try:
            fn()
            print(f'{name}: PASS')
        except AssertionError as e:
            fails.append(name)
            print(f'{name}: FAIL {e}')
    print('ALL PASS' if not fails else f'FAILURES: {fails}')
    sys.exit(1 if fails else 0)
