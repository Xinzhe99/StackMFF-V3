# Save the per-layer probabilities P (with and without smoothing), the colour layers and the reference inside one
# window of a benchmark scene, for the hard-vs-weighted fusion figure.
import argparse, os, re, sys, numpy as np, torch, cv2
B = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz'
sys.path.insert(0, os.path.join(B, 'v3merge_work/new_model/code'))
from stackmff_v3.network import StackMFFV3, smooth_probs
from predict_benchmark import pad_to_64, index_gt
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='Mobile Depth'); ap.add_argument('--scene', default='keyboard')
ap.add_argument('--ckpt', default=os.path.join(B, 'v3merge_work/runs/Hms_c48_p2_s175/model_save/epoch_0.pth'))
ap.add_argument('--out', required=True); ap.add_argument('--run_dir', default='')
a = ap.parse_args()
root = os.path.join(B, 'datasets/dff_and_mff', a.dataset)
d = os.path.join(root, 'image stack', a.scene)
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
fs = sorted([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.bmp'))], key=nat)
bgr = np.stack([cv2.imread(os.path.join(d, f)) for f in fs])
gray = np.stack([cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in bgr]).astype(np.float32) / 255
N, H, W = gray.shape
if a.run_dir:
    from stackmff_v3.model_io import load
    m = load(a.run_dir, a.ckpt)
else:
    m = StackMFFV3(C1=48, C2=96, heads=(3, 6), inter_pool=2).cuda().eval()
    sd = torch.load(a.ckpt, map_location='cpu'); m.load_state_dict(sd.get('model', sd))
with torch.no_grad():
    P0 = m.forward_probs(pad_to_64(torch.from_numpy(gray)[None].cuda()))[:, :, :H, :W].float()
    P1 = smooth_probs(P0, 1.0)
gt = cv2.imread(index_gt(os.path.join(root, 'AiF'))[a.scene])
np.savez_compressed(a.out, P0=P0[0].cpu().numpy().astype(np.float16), P1=P1[0].cpu().numpy().astype(np.float16),
                    rgb=bgr[..., ::-1], gt=gt[..., ::-1], names=np.array(fs))
print('saved', N, H, W)
