# -*- coding: utf-8 -*-
"""Runtime and peak GPU memory of our model versus the stack size N (random 640x360 grey stacks, fp32, idle GPU),
plus StackMFF V2 (official, native stack) and an iterated pairwise network (IFCNN, official) for comparison.
  python time_vs_n.py --arch h|loop --ckpt ... --out time_vs_n.csv
"""
import argparse, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import StackMFFV3, StackMFFLoop
from predict_benchmark import pad_to_64
B = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz'


def timeit(fn, reps=5):
    fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return float(np.median(ts)), torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='h'); ap.add_argument('--ckpt', default=''); ap.add_argument('--out', required=True)
    ap.add_argument('--Ns', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64])
    ap.add_argument('--K', type=int, default=3)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False; torch.set_grad_enabled(False)
    H, W = 360, 640
    if a.arch == 'loop':
        m = StackMFFLoop(C1=48, C2=96, heads=(3, 6), inter_pool=2).cuda().eval(); m.cur_K = a.K
    else:
        m = StackMFFV3(C1=48, C2=96, heads=(3, 6), inter_pool=2).cuda().eval()
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location='cpu'); m.load_state_dict(sd.get('model', sd))
    sys.path.insert(0, os.path.join(B, 'project_image_stack_fusion/StackMFF_v2'))
    from Stack_MFF_V2_Net_simple import StackMFF_V2
    v2 = StackMFF_V2(); sd = torch.load(os.path.join(B, 'project_image_stack_fusion/StackMFF_v2/epoch_29.pth'), map_location='cpu')
    v2.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}); v2 = v2.cuda().eval()
    sys.path.insert(0, os.path.join(B, 'project_image_pair_fusion/IFCNN/Code'))
    rows = []
    for N in a.Ns:
        x = torch.rand(1, N, H, W, device='cuda')
        t_o, m_o = timeit(lambda: (m.forward_probs(pad_to_64(x))[:, :, :H, :W] * x).sum(1))
        xv = F.interpolate(x, size=(384, 640), mode='bilinear', align_corners=False)
        try:
            t_v, m_v = timeit(lambda: v2(xv))
        except RuntimeError:
            t_v, m_v = float('nan'), float('nan'); torch.cuda.empty_cache()
        rows.append({'N': N, 'ours_s': t_o, 'ours_MiB': m_o, 'v2_s': t_v, 'v2_MiB': m_v})
        print(rows[-1], flush=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]


if __name__ == '__main__':
    main()
