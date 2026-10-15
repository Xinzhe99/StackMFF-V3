# -*- coding: utf-8 -*-
"""Save, for selected benchmark scenes, the per-pixel confidence max_n P_n, the argmax index and the weighted
grey fusion of a checkpoint (for the failure-case / confidence figure).
  python save_conf.py --ckpt ... --scenes "Mobile Depth/keyboard" "Middlebury/Adirondack" --out conf_maps.npz
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import StackMFFV3
from predict_benchmark import list_stacks, read_gray, pad_to_64

ROOT = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/dff_and_mff'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--C1', type=int, default=48); ap.add_argument('--C2', type=int, default=96)
    ap.add_argument('--heads', type=int, nargs=2, default=[3, 6]); ap.add_argument('--inter_pool', type=int, default=2)
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    m = StackMFFV3(C1=a.C1, C2=a.C2, heads=tuple(a.heads), inter_pool=a.inter_pool).to(dev).eval()
    sd = torch.load(a.ckpt, map_location='cpu'); m.load_state_dict(sd.get('model', sd))
    out = {}
    for sc in a.scenes:
        ds, name = sc.split('/')
        paths = dict(list_stacks(os.path.join(ROOT, ds, 'image stack')))[name]
        st = np.stack([read_gray(p) for p in paths]).astype(np.float32)
        N, H, W = st.shape
        with torch.no_grad():
            P = m.forward_probs(pad_to_64(torch.from_numpy(st / 255).unsqueeze(0).to(dev)))[0, :, :H, :W].float().cpu().numpy()
        key = sc.replace(' ', '_').replace('/', '__')
        out[key + '__conf'] = P.max(0).astype(np.float16)
        out[key + '__idx'] = P.argmax(0).astype(np.uint8)
        out[key + '__soft'] = np.clip((P * st).sum(0), 0, 255).astype(np.uint8)
    np.savez_compressed(a.out, **out)
    print('saved', list(out))


if __name__ == '__main__':
    main()
