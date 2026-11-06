"""Shared code for Assignment 2: data loading, A1 augmentation, features, hand mask,
procedural stress-test generator, procedural 'camera clip' generator, ArtAug-style enhancement.

IMPORTANT (honesty note): Step-Video-T2V and ArtAug are GPU-heavy generative models that could not
be run in the sandbox used to build this repo (no GPU, no network). `stress_condition`, `make_clip`
and `enhance` are *classical, procedural stand-ins* that play the same role in the pipeline.
Everything they produce is SYNTHETIC and is labelled as such everywhere.
"""
import os, json, glob
import numpy as np, cv2
from skimage.feature import hog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ['peace', 'thumb', 'stop', 'ok']
IMG = 128
DATA = os.path.join(ROOT, 'data', 'real')

# ------------------------------------------------------------------ data
def load_real(data=DATA):
    meta = json.load(open(f'{data}/metadata.json'))
    files = sorted(meta['labels'])
    paths = [f"{data}/{meta['labels'][f]}/{f}" for f in files]
    y = np.array([CLASSES.index(meta['labels'][f]) for f in files])
    groups = np.array([meta['groups'][f] for f in files])
    return files, paths, y, groups

def preprocess(path_or_img, size=IMG):
    im = cv2.imread(path_or_img) if isinstance(path_or_img, str) else path_or_img
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if isinstance(path_or_img, str) else im
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)

def load_full_rgb(path, long_side=256):
    """RGB image resized so the long side == long_side (used for synthesis at higher res)."""
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    s = long_side / max(im.shape[:2])
    return cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)

# ------------------------------------------------------------------ A1 augmentation (unchanged)
def augment(img, rng):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rng.uniform(-25, 25), rng.uniform(0.85, 1.15))
    M[:, 2] += rng.uniform(-0.08, 0.08, 2) * w
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5: out = np.ascontiguousarray(out[:, ::-1])
    out = out.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    if rng.random() < 0.3: out = cv2.GaussianBlur(out, (5, 5), 0)
    out = out + rng.normal(0, 4, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ features (A1)
def feat_pixels(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
def feat_hog(img):
    g = _clahe.apply(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')

# ------------------------------------------------------------------ hand mask (GrabCut + skin prior)
def hand_mask(img_rgb, size=160):
    """Rough hand mask (uint8 0/1, same HxW as input). Used for background replacement only."""
    h, w = img_rgb.shape[:2]; s = size / max(h, w)
    im = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (max(8, int(w*s)), max(8, int(h*s))), interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    Cr, Cb = ycc[..., 1].astype(int), ycc[..., 2].astype(int)
    skin = (Cr > 133) & (Cr < 180) & (Cb > 77) & (Cb < 135)
    H, W = skin.shape
    gm = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    gm[skin] = cv2.GC_PR_FGD
    b = max(1, int(.03 * min(H, W)))
    gm[:b] = cv2.GC_BGD; gm[-b:] = cv2.GC_BGD; gm[:, :b] = cv2.GC_BGD; gm[:, -b:] = cv2.GC_BGD
    cs = (slice(int(.3*H), int(.7*H)), slice(int(.3*W), int(.7*W)))
    gm[cs] = np.where(skin[cs], cv2.GC_FGD, cv2.GC_PR_BGD)
    bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(im, gm, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
        m = ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        m = skin.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if m.mean() < 0.03: m = skin.astype(np.uint8)          # fall-back
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)  # soft alpha in [0,1]

# ------------------------------------------------------------------ SYNTHETIC: stress conditions
CONDITIONS = ['lowlight', 'backlight', 'overexposed', 'motion_blur', 'defocus', 'sensor_noise',
              'color_cast', 'camera_tilt', 'new_background']

def _random_background(h, w, rng):
    kind = rng.integers(0, 4)
    if kind == 0:   # smooth gradient
        a, b = rng.uniform(30, 225, 3), rng.uniform(30, 225, 3)
        t = np.linspace(0, 1, h)[:, None, None]
        bg = a * (1 - t) + b * t; bg = np.repeat(bg, w, axis=1)
    elif kind == 1:  # tiles
        t = rng.integers(16, 40); base = rng.uniform(60, 200)
        yy, xx = np.mgrid[:h, :w]; chk = (((yy // t) + (xx // t)) % 2)[..., None]
        bg = np.where(chk, base, base * 0.8) * rng.uniform(0.9, 1.1, 3)
    elif kind == 2:  # wood-ish stripes
        base = np.array([150, 105, 65]) * rng.uniform(0.6, 1.2)
        xs = (np.sin(np.arange(w) / rng.uniform(3, 9)) * 12 + rng.normal(0, 6, w))[None, :, None]
        bg = np.repeat(base[None, None, :], h, 0).repeat(w, 1) + xs
    else:            # speckled concrete
        bg = np.full((h, w, 3), rng.uniform(80, 190)) + rng.normal(0, 18, (h, w, 1))
        bg = cv2.GaussianBlur(bg.astype(np.float32), (3, 3), 0)
    return np.clip(bg, 0, 255).astype(np.float32)

def _motion_kernel(length, angle):
    k = np.zeros((length, length), np.float32); k[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length / 2 - .5, length / 2 - .5), angle, 1)
    k = cv2.warpAffine(k, M, (length, length)); return k / max(k.sum(), 1e-6)

def stress_condition(img, cond, rng, severity=None, mask=None):
    """Apply one environmental stress to an RGB uint8 image -> RGB uint8. SYNTHETIC."""
    sev = rng.uniform(0.5, 1.0) if severity is None else severity
    f = img.astype(np.float32); h, w = f.shape[:2]
    if cond == 'lowlight':
        g = 0.55 - 0.40 * sev                                  # 0.15 .. 0.55 of original
        f = 255 * (f / 255) ** 1.6 * (g / 0.4)
        f = f + rng.normal(0, 3 + 6 * sev, f.shape)
    elif cond == 'backlight':
        cx = rng.uniform(.2, .8) * w; cy = rng.uniform(0, .4) * h
        yy, xx = np.mgrid[:h, :w]; d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(h, w)
        glow = np.exp(-(d / .35) ** 2)[..., None]
        f = f * (0.85 - 0.45 * sev) + 255 * glow * (0.35 + .5 * sev)
    elif cond == 'overexposed':
        f = f * (1.3 + 1.2 * sev) + 20 * sev
    elif cond == 'motion_blur':
        L = int(5 + 18 * sev) | 1
        f = cv2.filter2D(f, -1, _motion_kernel(L, rng.uniform(0, 180)))
    elif cond == 'defocus':
        k = int(3 + 10 * sev) | 1
        f = cv2.GaussianBlur(f, (k, k), 0)
    elif cond == 'sensor_noise':
        f = f * 0.8 + rng.normal(0, 10 + 28 * sev, f.shape)
        f = f + rng.normal(0, 8 * sev, f.shape[:2] + (1,))
    elif cond == 'color_cast':
        tint = np.array([1.0, 1.0, 1.0]); i = rng.integers(0, 3)
        tint[i] = 1 + .35 * sev; tint[(i + 1) % 3] = 1 - .25 * sev
        f = f * tint
    elif cond == 'camera_tilt':
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.choice([-1, 1]) * (15 + 40 * sev), rng.uniform(.8, 1.1))
        f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif cond == 'new_background':
        m = hand_mask(img) if mask is None else mask
        bg = _random_background(h, w, rng)
        f = f * m[..., None] + bg * (1 - m[..., None])
    else:
        raise ValueError(cond)
    return np.clip(f, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ SYNTHETIC: short camera clips
def make_clip(img_rgb, rng, n_frames=24, out=224, cond=None):
    """Turn ONE still into a short 'handheld camera' clip (pan/zoom/roll with ease-in-out,
    velocity-dependent motion blur, exposure drift, optional stress condition).
    Role: stand-in for a Step-Video-T2V gesture clip. Returns list of RGB uint8 frames. SYNTHETIC."""
    h, w = img_rgb.shape[:2]; s = out / max(h, w)
    base = cv2.resize(img_rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    H, W = base.shape[:2]
    if cond is not None:
        base = stress_condition(base, cond, rng)
    a0, a1 = rng.uniform(-12, 12, 2); z0, z1 = 1.0, rng.uniform(1.05, 1.35)
    t0, t1 = rng.uniform(-.06, .06, 2) * W, rng.uniform(-.06, .06, 2) * W
    gain0, gain1 = rng.uniform(.7, 1.0), rng.uniform(.8, 1.2)
    shake = rng.normal(0, 1.2, (n_frames, 2))
    ease = lambda t: t * t * (3 - 2 * t)
    prev = None; frames = []
    for i in range(n_frames):
        t = ease(i / (n_frames - 1))
        ang = a0 + (a1 - a0) * t; z = z0 + (z1 - z0) * t
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, z)
        M[:, 2] += t0 + (t1 - t0) * t + shake[i]
        fr = cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        speed = 0 if prev is None else np.hypot(*(M[:, 2] - prev))
        if speed > 1.5:
            L = int(min(15, 3 + speed * 1.5)) | 1
            fr = cv2.filter2D(fr, -1, _motion_kernel(L, rng.uniform(0, 180) if i == 1 else 0))
        fr = fr * (gain0 + (gain1 - gain0) * t) + rng.normal(0, 2, fr.shape)
        frames.append(np.clip(fr, 0, 255).astype(np.uint8)); prev = M[:, 2].copy()
    return frames

BANNER = 18
def write_clip(frames, path, fps=12, watermark='SYNTHETIC - procedural clip (not a real person)'):
    """mp4 with a black banner strip (BANNER px) under the picture carrying the SYNTHETIC label."""
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h + BANNER))
    for fr in frames:
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        bar = np.zeros((BANNER, w, 3), np.uint8)
        cv2.putText(bar, watermark, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(np.vstack([bgr, bar]))
    vw.release()

def sample_frames(video_path, k=6, crop_bottom=0):
    """Auto-sample k evenly spaced frames from an mp4 (works for Step-Video-T2V output too)."""
    cap = cv2.VideoCapture(video_path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for idx in np.linspace(0, max(n - 1, 0), k).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, fr = cap.read()
        if ok:
            if crop_bottom: fr = fr[:-crop_bottom]
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release(); return out

# ------------------------------------------------------------------ ArtAug-style enhancement (classical stand-in)
def enhance(img, target_luma=0.45):
    """Exposure / colour / contrast 'polish' for dark or washed-out frames.
    Classical stand-in for ArtAug's polishing step: gray-world white balance, auto-gamma to a target mean
    luminance, CLAHE on L, light denoise, unsharp mask. RGB uint8 -> RGB uint8."""
    f = img.astype(np.float32)
    m = f.reshape(-1, 3).mean(0); f = np.clip(f * (m.mean() / np.maximum(m, 1e-3)) ** 0.6, 0, 255)  # soft gray-world
    lum = max(f.mean() / 255, 1e-3)
    gamma = np.clip(np.log(target_luma) / np.log(lum), 0.35, 2.2)
    f = 255 * (f / 255) ** gamma
    u8 = np.clip(f, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    u8 = cv2.bilateralFilter(u8, 5, 25, 5)
    blur = cv2.GaussianBlur(u8, (0, 0), 1.2)
    return np.clip(cv2.addWeighted(u8, 1.4, blur, -0.4, 0), 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ synthetic training set builder
def synth_from_image(full_rgb, rng, n_stress=6, n_clips=2, frames_per_clip=3, use_enhance=False):
    """Return list of 128x128 RGB synthetic training images derived from ONE real image."""
    out = []
    mask = None
    conds = list(rng.choice(CONDITIONS, size=n_stress, replace=False if n_stress <= len(CONDITIONS) else True))
    for c in conds:
        if c == 'new_background' and mask is None: mask = hand_mask(full_rgb)
        im = stress_condition(full_rgb, c, rng, mask=mask if c == 'new_background' else None)
        out.append(im)
    for _ in range(n_clips):
        cond = rng.choice(CONDITIONS[:7] + [None, None])
        fr = make_clip(full_rgb, rng, cond=None if cond is None else str(cond))
        idx = np.linspace(2, len(fr) - 3, frames_per_clip).astype(int)
        out += [fr[i] for i in idx]
    return [cv2.resize(enhance(im) if use_enhance else im, (IMG, IMG), interpolation=cv2.INTER_AREA) for im in out]
