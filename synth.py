# -*- coding: utf-8 -*-
"""On-the-fly synthesis of training stacks at training resolution.

The original data were synthesised at the native resolution of each source image (kernel 2*d+1 px for depth
distance d) and then resized to 256, so the blur difference between neighbouring layers depended on the source
resolution and was below one pixel for large images. Here the stack is synthesised after resizing:
  q(p) = depth level in {0..N-1} (uniform bins, as before); layer i at p = B_{|q(p)-i|}(p),
  B_d = AiF blurred with radius r_d = s * d (s ~ U[smin, smax] px per level and per sample),
  kernel = disk (defocus, probability p_disk) or Gaussian with sigma = r_d / 2;
then per-layer gain jitter and Gaussian noise; layers are shuffled and the label permuted accordingly.
"""
import math, os, random
import numpy as np
import torch
import torch.nn.functional as F


class PairDataset(torch.utils.data.Dataset):
    """(AiF, depth) pairs from make_cache_aif.py, random crop `size` + flips. Each item carries a stack size N
    drawn once per item (uniform in [nmin, nmax]) so that the grouped batch sampler can batch equal N."""

    def __init__(self, files, size=256, nmin=2, nmax=24, seed=0):
        self.files, self.size = files, size
        rng = random.Random(seed)
        self.stack_sizes = [rng.randint(nmin, nmax) for _ in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i]); a, z = d['aif'], d['depth']
        S = a.shape[0]; t = random.randint(0, S - self.size); l = random.randint(0, S - self.size)
        a = a[t:t + self.size, l:l + self.size]; z = z[t:t + self.size, l:l + self.size]
        if random.random() < 0.5:
            a, z = a[:, ::-1], z[:, ::-1]
        if random.random() < 0.5:
            a, z = a[::-1], z[::-1]
        x = torch.from_numpy(np.stack([np.ascontiguousarray(a), np.ascontiguousarray(z)])).float()   # 2,S,S
        return x, torch.zeros(1, dtype=torch.long), self.stack_sizes[i]


def pair_files(root, names, split, fraction, seed=0):
    files = []
    for n in names:
        d = os.path.join(root, n, split)
        if os.path.isdir(d):
            fs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz'))
            if fraction < 1.0:
                fs = random.Random(seed).sample(fs, max(1, int(len(fs) * fraction)))
            files += fs
    return files


def _kernel(r, disk, device):
    if r < 0.25:
        return None
    if disk:
        R = int(math.ceil(r)); g = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
        dist = torch.sqrt(g[None] ** 2 + g[:, None] ** 2)
        k = (r + 0.5 - dist).clamp(0, 1)
    else:
        sig = max(r / 2.0, 0.3); R = int(math.ceil(3 * sig)); g = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
        k1 = torch.exp(-g ** 2 / (2 * sig * sig)); k = k1[None] * k1[:, None]
    return k / k.sum()


@torch.no_grad()
def synth_batch(xp, N, smin=0.4, smax=1.6, p_disk=0.5, rmax=20.0, gain=0.03, noise=0.01):
    """xp: [B,2,S,S] (AiF 0..255, depth 0..255) on GPU -> stack [B,N,S,S] in [0,1], label [B,S,S] long.
    Per sample: all N blur levels in one convolution (kernels zero-padded to a common size), layers gathered."""
    B, _, S, _ = xp.shape
    dev = xp.device
    aif = xp[:, :1] / 255.0
    q = (xp[:, 1] * N / 256.0).floor().long().clamp_(0, N - 1)                       # B,S,S
    out = torch.empty(B, N, S, S, device=dev)
    lab = torch.empty(B, S, S, dtype=torch.long, device=dev)
    lev = torch.arange(N, device=dev).view(N, 1, 1)
    for b in range(B):
        s = random.uniform(smin, smax); disk = random.random() < p_disk
        ks = [_kernel(min(s * d, rmax), disk, dev) for d in range(N)]
        R = max([k.shape[0] // 2 for k in ks if k is not None] + [0])
        W = torch.zeros(N, 1, 2 * R + 1, 2 * R + 1, device=dev)
        for d, k in enumerate(ks):
            if k is None:
                W[d, 0, R, R] = 1.0
            else:
                r = k.shape[0] // 2
                W[d, 0, R - r:R + r + 1, R - r:R + r + 1] = k
        Bd = F.conv2d(F.pad(aif[b:b + 1], (R, R, R, R), mode='reflect'), W)[0]        # N(levels),S,S
        dist = (q[b][None] - lev).abs()                                               # N(layers),S,S
        layers = torch.gather(Bd, 0, dist)
        g = 1 + (torch.rand(N, 1, 1, device=dev) * 2 - 1) * gain
        layers = (layers * g + torch.randn_like(layers) * random.uniform(0, noise)).clamp_(0, 1)
        perm = torch.randperm(N, device=dev)                                          # level i -> position perm[i]
        out[b, perm] = layers
        lab[b] = perm[q[b]]
    return out, lab
