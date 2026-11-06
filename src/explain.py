"""Explainability for the HOG+SVM pipeline: occlusion-sensitivity heatmap + nearest-neighbour gallery.
(Grad-CAM needs a CNN; occlusion is model-agnostic and works for any classifier.)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, joblib
from common import *

def feats(imgs): return np.stack([feat_hog(enhance(i)) for i in imgs])

def predict(model, img128):
    s = model.decision_function(feats([img128]))[0]
    e = np.exp(s - s.max()); return int(s.argmax()), e / e.sum(), s     # softmax of SVM scores = relative score

def occlusion_map(model, img128, patch=32, stride=16):
    """Drop of the predicted-class SVM score when a grey patch hides each region (red = important)."""
    k, _, s0 = predict(model, img128); base = s0[k]
    pos = list(range(0, IMG - patch + 1, stride)); batch = []
    for y0 in pos:
        for x0 in pos:
            o = img128.copy(); o[y0:y0+patch, x0:x0+patch] = 128; batch.append(o)
    sc = model.decision_function(feats(batch))[:, k].reshape(len(pos), len(pos))
    drop = base - sc; acc = np.zeros((IMG, IMG), np.float32); cnt = np.zeros((IMG, IMG), np.float32)
    for a, y0 in enumerate(pos):
        for b, x0 in enumerate(pos):
            acc[y0:y0+patch, x0:x0+patch] += drop[a, b]; cnt[y0:y0+patch, x0:x0+patch] += 1
    h = acc / np.maximum(cnt, 1); h = np.clip(h, 0, None); return h / (h.max() + 1e-9)

def overlay(img128, heat, size=256):
    im = cv2.resize(img128, (size, size)); hm = cv2.applyColorMap((cv2.resize(heat, (size, size)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(im, 0.55, cv2.cvtColor(hm, cv2.COLOR_BGR2RGB), 0.45, 0)

def neighbours(bank, img128, k=4, exclude=None):
    f = feat_hog(enhance(img128)); d = np.linalg.norm(bank - f, axis=1)
    if exclude is not None: d[exclude] = np.inf
    return np.argsort(d)[:k], np.sort(d)[:k]

def load_bundle(p=os.path.join(ROOT, 'results', 'model.joblib')): return joblib.load(p)
