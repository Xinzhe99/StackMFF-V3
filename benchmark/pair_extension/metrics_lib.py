"""Metric library for the StackMFF stack-fusion benchmark.

Conventions: all computation in float64; inputs are uint8 grayscale (ITU-R BT.601 via PIL 'L').
Every function documents its reference (paper / MATLAB file it follows).
Higher is better for every metric returned here.
"""
import math
import numpy as np
from PIL import Image
import cv2
from scipy.signal import convolve2d
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as sk_ssim

EPS = np.finfo(np.float64).eps  # == MATLAB eps


# ----------------------------------------------------------------- I/O
def load_gray(path):
    """Load as 8-bit grayscale, ITU-R BT.601 luma via PIL convert('L')."""
    im = Image.open(path)
    if im.mode != 'L':
        im = im.convert('L')
    return np.array(im, dtype=np.uint8, copy=True)


def resize_to(img, hw):
    """Bicubic resize (PIL) to (H, W) if needed. Returns (img, resized_flag)."""
    h, w = hw
    if img.shape == (h, w):
        return img, False
    out = Image.fromarray(img).resize((w, h), Image.BICUBIC)
    return np.array(out, dtype=np.uint8, copy=True), True


# ----------------------------------------------------------------- filters
def gauss1d(n, sigma):
    r = (n - 1) / 2.0
    x = np.arange(n, dtype=np.float64) - r
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def valid_sep(img, g):
    """'valid' correlation of a 2-D array with a separable symmetric kernel g
    (equivalent to filter2(fspecial('gaussian',n,sigma), img, 'valid'))."""
    n = len(g)
    H, W = img.shape
    t = np.zeros((H - n + 1, W))
    for i in range(n):
        t += g[i] * img[i:i + H - n + 1, :]
    o = np.zeros((H - n + 1, W - n + 1))
    for j in range(n):
        o += g[j] * t[:, j:j + W - n + 1]
    return o


def box_valid(a, k=11):
    """Exact 'valid' k x k box SUM for integer arrays (int64 cumulative sums)."""
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=a.dtype)
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


# ----------------------------------------------------------------- SSIM
def ssim_wang(x, y, L=255.0, K1=0.01, K2=0.03):
    """Wang et al. 2004 SSIM as in ssim_index.m: 11x11 Gaussian sigma=1.5,
    C1=(K1 L)^2, C2=(K2 L)^2, mean of the map over the 'valid' region."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    g = gauss1d(11, 1.5)
    mx = valid_sep(x, g)
    my = valid_sep(y, g)
    sxx = valid_sep(x * x, g) - mx * mx
    syy = valid_sep(y * y, g) - my * my
    sxy = valid_sep(x * y, g) - mx * my
    smap = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    return float(smap.mean())


def ssim_sk7(x, y):
    """skimage defaults (7x7 uniform window, sample covariance), data_range=255."""
    return float(sk_ssim(x.astype(np.float64), y.astype(np.float64), data_range=255))


# ----------------------------------------------------------------- PSNR
def psnr(x, y):
    d = x.astype(np.float64) - y.astype(np.float64)
    mse = float(np.mean(d * d))
    return float('inf') if mse == 0.0 else 10.0 * math.log10(255.0 ** 2 / mse)


# ----------------------------------------------------------------- VIFp
def vifp(ref, dist, sigma_nsq=2.0):
    """Pixel-domain multiscale VIFp, transliteration of vifp_mscale (Sheikh & Bovik 2006).
    4 scales, window N=2^(4-s+1)+1 with sigma N/5, 'valid' filtering, decimate by 2."""
    ref = ref.astype(np.float64)
    dist = dist.astype(np.float64)
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        g = gauss1d(N, N / 5.0)
        if scale > 1:
            ref = valid_sep(ref, g)
            dist = valid_sep(dist, g)
            ref = ref[0::2, 0::2]
            dist = dist[0::2, 0::2]
        mu1 = valid_sep(ref, g)
        mu2 = valid_sep(dist, g)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        s1 = valid_sep(ref * ref, g) - mu1_sq
        s2 = valid_sep(dist * dist, g) - mu2_sq
        s12 = valid_sep(ref * dist, g) - mu1_mu2
        s1[s1 < 0] = 0
        s2[s2 < 0] = 0
        gg = s12 / (s1 + 1e-10)
        sv = s2 - gg * s12
        m = s1 < 1e-10
        gg[m] = 0
        sv[m] = s2[m]
        s1[m] = 0
        m = s2 < 1e-10
        gg[m] = 0
        sv[m] = 0
        m = gg < 0
        sv[m] = s2[m]
        gg[m] = 0
        sv[sv <= 1e-10] = 1e-10
        num += np.sum(np.log10(1.0 + gg * gg * s1 / (sv + sigma_nsq)))
        den += np.sum(np.log10(1.0 + s1 / sigma_nsq))
    return float(num / den)


# ----------------------------------------------------------------- MI / NMI
def mi_nmi(a, b):
    """256-bin joint histogram, log2. NMI = 2 MI / (H(A)+H(B)). Returns (MI, NMI)."""
    a = a.astype(np.int64).ravel()
    b = b.astype(np.int64).ravel()
    joint = np.bincount(a * 256 + b, minlength=65536).reshape(256, 256).astype(np.float64)
    p = joint / joint.sum()
    pa = p.sum(1)
    pb = p.sum(0)
    nz = p > 0
    mi = float(np.sum(p[nz] * np.log2(p[nz] / np.outer(pa, pb)[nz])))
    ha = float(-np.sum(pa[pa > 0] * np.log2(pa[pa > 0])))
    hb = float(-np.sum(pb[pb > 0] * np.log2(pb[pb > 0])))
    nmi = 2.0 * mi / (ha + hb) if (ha + hb) > 0 else 1.0
    return mi, nmi


# ----------------------------------------------------------------- Q^{AB/F} (N-source)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)  # h3 in Qabf.m
SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)  # h1 in Qabf.m
TG, KG, DG = 0.9994, -15.0, 0.5
TA, KA, DA = 0.9879, -22.0, 0.8
QL = 1.0  # exponent L


def grad_orient(img):
    """Sobel gradient magnitude g and orientation a (atan(sy/sx); pi/2 where sx==0),
    zero-padded 'same' convolution as conv2(...,'same') in Qabf.m."""
    x = img.astype(np.float64)
    sx = convolve2d(x, SOBEL_X, mode='same', boundary='fill', fillvalue=0.0)
    sy = convolve2d(x, SOBEL_Y, mode='same', boundary='fill', fillvalue=0.0)
    g = np.sqrt(sx * sx + sy * sy)
    safe = np.where(sx == 0, 1.0, sx)
    ang = np.where(sx == 0, np.pi / 2, np.arctan(sy / safe))
    return g, ang


def _ratio(gn, gf, rule):
    mx = np.maximum(gn, gf)
    mn = np.minimum(gn, gf)
    if rule == 'minmax':          # standard Xydeas-Petrovic: min/max, 0 when both are 0
        return np.where(mx > 0, mn / np.where(mx > 0, mx, 1.0), 0.0)
    if rule == 'matlab':          # Qabf.m: if gA>gF: gF/gA; if equal: gF (not 1); else gA/gF
        out = np.empty_like(gn)
        big = gn > gf
        eq = gn == gf
        lt = ~(big | eq)
        out[big] = gf[big] / gn[big]
        out[eq] = gf[eq]
        out[lt] = gn[lt] / gf[lt]
        return out
    raise ValueError(rule)


def q_pair(gn, an, gf, af, rule='minmax'):
    G = _ratio(gn, gf, rule)
    Qg = TG / (1.0 + np.exp(KG * (G - DG)))
    A = 1.0 - np.abs(an - af) / (np.pi / 2)
    Qa = TA / (1.0 + np.exp(KA * (A - DA)))
    return Qg * Qa


def qabf_stack(src_grads, fused, rule='minmax'):
    """Q^{AB/F} generalized to N sources: sum_n sum_p w_n Q_nF / sum_n sum_p w_n,
    w_n = g_n^L (L=1). src_grads: list of (g, a) per source layer. For N=2 this is
    exactly the Xydeas-Petrovic Qabf."""
    gf, af = grad_orient(fused)
    num = 0.0
    den = 0.0
    for gn, an in src_grads:
        Q = q_pair(gn, an, gf, af, rule)
        w = gn ** QL
        num += float(np.sum(w * Q))
        den += float(np.sum(w))
    return num / den


# ----------------------------------------------------------------- MEF-SSIM (Ma, Zeng, Wang 2015)
_G11 = None


def _g2d():
    global _G11
    if _G11 is None:
        g = gauss1d(11, 1.5)
        _G11 = np.outer(g, g)  # == fspecial('gaussian',11,1.5), sums to 1
    return _G11


OFFS = [(di, dj) for di in range(11) for dj in range(11)]  # C-order patch offsets
C_MEF = (0.03 * 255) ** 2


def mef_precompute(layers):
    """Method-independent part of mef_ssim.m (single scale, wSize=11, K=0.03).
    layers: (N,H,W) uint8. Returns Rf with shape (121,Hp,Wp): the normalised
    structure-consistency block rBlock for every valid patch centre (float64).
    Box statistics are exact (integer arithmetic), numerator/denominator follow
    mef_ssim.m: num = ||sum_k x_k - mean||, den = sum_k ||x_k - mu_k||."""
    X = layers.astype(np.int64)
    N, H, W = X.shape
    Hp, Wp = H - 10, W - 10
    S1 = np.stack([box_valid(X[n]) for n in range(N)])
    S2 = np.stack([box_valid(X[n] * X[n]) for n in range(N)])
    Dint = 121 * S2 - S1 * S1                      # 121 * sum of squared deviations
    mu = S1 / 121.0
    norm_ = np.sqrt(np.maximum(Dint, 0) / 121.0)   # ||x_k - mu_k||
    ed = norm_ + 0.001                              # sqrt(wSize^2 * sigmaSq) + 0.001
    den = norm_.sum(0)
    T = X.sum(0)
    T1 = box_valid(T)
    T2 = box_valid(T * T)
    num = np.sqrt(np.maximum(121 * T2 - T1 * T1, 0) / 121.0)
    R = (num + EPS) / (den + EPS)
    R = np.clip(R, EPS, 1 - EPS)
    p = np.minimum(np.tan(np.pi / 2 * R), 10.0)
    wmap = (ed / 11.0) ** p[None] + EPS
    wmap = wmap / wmap.sum(0)[None]
    maxed = ed.max(0)
    c = wmap / ed
    M = (c * mu).sum(0)
    Rf = np.empty((121, Hp, Wp), dtype=np.float64)
    for m, (di, dj) in enumerate(OFFS):
        acc = np.zeros((Hp, Wp), dtype=np.float64)
        for n in range(N):
            acc += c[n] * X[n, di:di + Hp, dj:dj + Wp]
        Rf[m] = acc - M
    nr = np.sqrt(np.sum(Rf * Rf, axis=0))
    scale = np.where(nr > 0, maxed / np.where(nr > 0, nr, 1.0), 1.0)
    Rf *= scale[None]
    return Rf


def mef_from_field(Rf, fused):
    """Fused-dependent part of mef_ssim.m (fast form). Uses sigma1^2 = sum w r^2 - mu1^2,
    sigma2^2 = sum w f^2 - mu2^2 and sigma12 = sum w r f - mu1 mu2 (valid because sum w = 1).
    Agreement with the centred two-pass form (mef_from_field_centered) is checked in validate.py."""
    g = gauss1d(11, 1.5)
    f = fused.astype(np.float64)
    K, Hp, Wp = Rf.shape
    w = _g2d().ravel()
    R2 = Rf.reshape(K, Hp * Wp)
    mu1 = (w @ R2).reshape(Hp, Wp)
    s1 = np.einsum('m,mp,mp->p', w, R2, R2).reshape(Hp, Wp) - mu1 * mu1
    cross = np.zeros((Hp, Wp))
    for m, (di, dj) in enumerate(OFFS):
        cross += w[m] * Rf[m] * f[di:di + Hp, dj:dj + Wp]
    mu2 = valid_sep(f, g)
    s2 = valid_sep(f * f, g) - mu2 * mu2
    s12 = cross - mu1 * mu2
    qmap = (2 * s12 + C_MEF) / (s1 + s2 + C_MEF)
    return float(qmap.mean()), qmap


def mef_from_field_centered(Rf, fused):
    """Centred two-pass form of the fused-dependent part of mef_ssim.m (kept for cross-checks)."""
    g2 = _g2d()
    f = fused.astype(np.float64)
    Hp, Wp = Rf.shape[1], Rf.shape[2]
    mu1 = np.zeros((Hp, Wp))
    mu2 = np.zeros((Hp, Wp))
    for m, (di, dj) in enumerate(OFFS):
        w = g2[di, dj]
        mu1 += w * Rf[m]
        mu2 += w * f[di:di + Hp, dj:dj + Wp]
    s1 = np.zeros((Hp, Wp))
    s2 = np.zeros((Hp, Wp))
    s12 = np.zeros((Hp, Wp))
    for m, (di, dj) in enumerate(OFFS):
        w = g2[di, dj]
        dr = Rf[m] - mu1
        df = f[di:di + Hp, dj:dj + Wp] - mu2
        s1 += w * dr * dr
        s2 += w * df * df
        s12 += w * dr * df
    qmap = (2 * s12 + C_MEF) / (s1 + s2 + C_MEF)
    return float(qmap.mean()), qmap


def conv_valid_2d(img, k):
    kh, kw = k.shape
    H, W = img.shape
    o = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(kh):
        for j in range(kw):
            o += k[i, j] * img[i:i + H - kh + 1, j:j + W - kw + 1]
    return o


def mef_literal(imgSeq, fI, K=0.03):
    """Literal (loop) transliteration of mef_ssim.m, used only for validation."""
    X = imgSeq.astype(np.float64).transpose(1, 2, 0)  # (H, W, N) as in MATLAB
    fI = fI.astype(np.float64)
    H, W, N = X.shape
    wSize = 11
    bd = 5
    sWin = np.ones((11, 11)) / 121.0
    window = _g2d()
    Hp, Wp = H - 2 * bd, W - 2 * bd
    mu = np.zeros((Hp, Wp, N))
    ed = np.zeros((Hp, Wp, N))
    for k in range(N):
        img = X[:, :, k]
        mu[:, :, k] = conv_valid_2d(img, sWin)
        sigmaSq = conv_valid_2d(img * img, sWin) - mu[:, :, k] ** 2
        ed[:, :, k] = np.sqrt(np.maximum(wSize ** 2 * sigmaSq, 0)) + 0.001
    R = np.zeros((Hp, Wp))
    for i in range(bd, H - bd):
        for j in range(bd, W - bd):
            vecs = X[i - bd:i + bd + 1, j - bd:j + bd + 1, :].reshape(121, N)
            den = 0.0
            for k in range(N):
                den += np.linalg.norm(vecs[:, k] - mu[i - bd, j - bd, k])
            s = vecs.sum(1)
            num = np.linalg.norm(s - s.mean())
            R[i - bd, j - bd] = (num + EPS) / (den + EPS)
    R = np.where(R > 1, 1 - EPS, R)
    R = np.where(R < 0, EPS, R)
    p = np.tan(np.pi / 2 * R)
    p = np.where(p > 10, 10, p)
    maxEd = ed.max(2)
    wMap = (ed / wSize) ** p[:, :, None] + EPS
    wMap = wMap / wMap.sum(2, keepdims=True)
    C = (K * 255) ** 2
    qMap = np.zeros((Hp, Wp))
    w = window.ravel()
    for i in range(bd, H - bd):
        for j in range(bd, W - bd):
            blocks = X[i - bd:i + bd + 1, j - bd:j + bd + 1, :]
            rB = np.zeros((11, 11))
            for k in range(N):
                rB = rB + wMap[i - bd, j - bd, k] * (blocks[:, :, k] - mu[i - bd, j - bd, k]) / ed[i - bd, j - bd, k]
            nrm = np.linalg.norm(rB.ravel())
            if nrm > 0:
                rB = rB / nrm * maxEd[i - bd, j - bd]
            rV = rB.ravel()
            fV = fI[i - bd:i + bd + 1, j - bd:j + bd + 1].ravel()
            mu1 = np.sum(w * rV)
            mu2 = np.sum(w * fV)
            s1 = np.sum(w * (rV - mu1) ** 2)
            s2 = np.sum(w * (fV - mu2) ** 2)
            s12 = np.sum(w * (rV - mu1) * (fV - mu2))
            qMap[i - bd, j - bd] = (2 * s12 + C) / (s1 + s2 + C)
    return float(qMap.mean()), qMap


# ----------------------------------------------------------------- alignment
def shift_phase(gt, res):
    """Translation (dy, dx) that registers res onto gt (skimage phase correlation, 1/10 px)."""
    shift, err, _ = phase_cross_correlation(gt.astype(np.float64), res.astype(np.float64),
                                            upsample_factor=10)
    return float(shift[0]), float(shift[1])


_ORB = None


def gt_keypoints(gt):
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=2000)
    return _ORB.detectAndCompute(gt, None)


def align_to_gt(res, gt, gt_kp, shift):
    """ORB + RANSAC homography (cv2.findHomography, reproj 5 px) from res to gt, then warp.
    Falls back to the phase-correlation translation if the homography is unreliable.
    Returns (warped uint8, info dict)."""
    H, W = gt.shape
    info = {'method': 'homography', 'matches': 0, 'inliers': 0, 'scale': None, 'fallback': False}
    Hm = None
    kp1, d1 = _ORB.detectAndCompute(res, None) if _ORB is not None else gt_keypoints(res)
    kp2, d2 = gt_kp
    if d1 is not None and d2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        ms = sorted(bf.match(d1, d2), key=lambda m: m.distance)
        info['matches'] = len(ms)
        if len(ms) >= 8:
            src = np.float32([kp1[m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in ms]).reshape(-1, 1, 2)
            Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if Hm is not None and mask is not None:
                info['inliers'] = int(mask.sum())
                sc = math.sqrt(abs(np.linalg.det(Hm[:2, :2])))
                info['scale'] = float(sc)
                if info['inliers'] < 10 or not np.all(np.isfinite(Hm)) or not (0.5 < sc < 2.0):
                    Hm = None
    if Hm is None:
        info['fallback'] = True
        info['method'] = 'translation'
        dy, dx = shift
        Hm = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        info['scale'] = 1.0
    # uncovered border pixels are filled by edge replication (not 0), so that a small crop/scale
    # change in the commercial output is not scored as a black frame; valid_frac is still logged
    warped = cv2.warpPerspective(res, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    info['H'] = Hm.tolist()  # kept so that figures can apply the same warp to colour images
    ones = np.full(res.shape, 255, dtype=np.uint8)
    valid = cv2.warpPerspective(ones, Hm, (W, H), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    info['valid_frac'] = float(np.mean(valid > 0))
    return warped, info


# ----------------------------------------------------------------- all metrics for one fused image
def all_metrics(img, gt, src_grads, Rf):
    m = {}
    m['SSIM'] = ssim_wang(img, gt)
    m['SSIM_sk7'] = ssim_sk7(gt, img)
    m['PSNR'] = psnr(img, gt)
    m['VIF'] = vifp(gt, img)
    mi, nmi = mi_nmi(gt, img)
    m['MI'] = mi
    m['NMI'] = nmi
    m['Qabf_N'] = qabf_stack(src_grads, img)
    m['MEF_SSIM'], _ = mef_from_field(Rf, img)
    return m
