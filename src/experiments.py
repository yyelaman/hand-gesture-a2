"""Assignment 2 experiments: A1 baselines vs. A2 pipeline (synthetic stress data + enhancement).

Protocols
  P1  5-fold Stratified-Group CV x 3 repeats on the 68 REAL photos (same protocol as A1) -> clean metrics,
      plus the same held-out photos pushed through every synthetic stress condition (per-condition accuracy).
  P2  Leave-One-Environment-Out (4 pseudo-environments found by clustering the photo background) -> accuracy
      per environment: a *real* domain shift (different floor / table / lighting), no synthetic test data.
  P3  Training curves (MLP on HOG, per-epoch) for P2 folds.
  P4  Latency of the full pipeline.
No synthetic image derived from a test photo is ever used for training (synthetic data is generated from
training photos of each fold only).
"""
import os, sys, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from common import *

RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
N_AUG = 10
STRESS = ['clean'] + CONDITIONS
files, paths, y, groups = load_real()
N = len(y)
X128 = np.stack([preprocess(p) for p in paths])
FULL = [load_full_rgb(p, 256) for p in paths]

# ----------------------------------------------------------- pseudo-environments (real domain shift)
def env_feat(im):
    h, w = im.shape[:2]; lab = cv2.cvtColor(cv2.resize(im, (96, int(96 * h / w))), cv2.COLOR_RGB2LAB).astype(float)
    h, w = lab.shape[:2]; m = np.ones((h, w), bool); m[int(.15*h):int(.85*h), int(.2*w):int(.8*w)] = False
    b = lab[m]; return [b[:, 0].mean(), b[:, 1].mean(), b[:, 2].mean(), b[:, 0].std()]
EF = np.array([env_feat(im) for im in FULL])
km = KMeans(4, n_init=10, random_state=0).fit((EF - EF.mean(0)) / EF.std(0))
order = np.argsort([-EF[km.labels_ == c, 0].mean() for c in range(4)])       # bright -> dark
remap = {c: i for i, c in enumerate(order)}
ENV = np.array([remap[c] for c in km.labels_])
ENV_NAMES = None  # named after visual inspection in report (see figures/environments.png)

# ----------------------------------------------------------- configs
CFGS = {
    'A1 baseline: pixels+LogReg (+aug)': dict(feat='pix', model='lr',  synth=False, enh=False),
    'A1 improved: HOG+SVM (+aug)':       dict(feat='hog', model='svm', synth=False, enh=False),
    'A2: HOG+SVM + synthetic':           dict(feat='hog', model='svm', synth=True,  enh=False),
    'A2: HOG+SVM + enhance':             dict(feat='hog', model='svm', synth=False, enh=True),
    'A2 full: + synthetic + enhance':    dict(feat='hog', model='svm', synth=True,  enh=True),
}
def featurize(imgs, cfg):
    if cfg['feat'] == 'pix': return np.stack([feat_pixels(i) for i in imgs])
    return np.stack([feat_hog(enhance(i) if cfg['enh'] else i) for i in imgs])
def make_model(cfg):
    if cfg['model'] == 'lr': return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    return SVC(kernel='rbf', C=10, gamma='scale')

_SYN = {}
def synth_for(i, seed):
    k = (i, seed)
    if k not in _SYN:
        _SYN[k] = synth_from_image(FULL[i], np.random.default_rng([seed, i, 7]))
    return _SYN[k]
_AUG = {}
def aug_for(i, seed):
    k = (i, seed)
    if k not in _AUG:
        r = np.random.default_rng([seed, i, 3]); _AUG[k] = [augment(X128[i], r) for _ in range(N_AUG)]
    return _AUG[k]

def train_set(tr, seed, cfg):
    imgs, lab = [X128[i] for i in tr], [y[i] for i in tr]
    src = ['real'] * len(tr)
    for i in tr:
        a = aug_for(i, seed); imgs += a; lab += [y[i]] * len(a); src += ['aug'] * len(a)
        if cfg['synth']:
            s = synth_for(i, seed); imgs += s; lab += [y[i]] * len(s); src += ['synth'] * len(s)
    return imgs, np.array(lab), src

# stress test images for the held-out photos: fixed seed, independent from training synthesis seeds
_STR = {}
def stress_img(i, cond):
    k = (i, cond)
    if k not in _STR:
        if cond == 'clean': _STR[k] = X128[i]
        else:
            r = np.random.default_rng([4242, i, CONDITIONS.index(cond)])
            _STR[k] = cv2.resize(stress_condition(FULL[i], cond, r), (IMG, IMG), interpolation=cv2.INTER_AREA)
    return _STR[k]

def macro(yt, yp):
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
    return dict(accuracy=accuracy_score(yt, yp), precision=p, recall=r, f1=f)

# ----------------------------------------------------------- P1: repeated grouped CV + stress
def run_p1(seeds=(0, 1, 2)):
    res = {n: [] for n in CFGS}; preds = {n: [] for n in CFGS}
    stress = {n: {c: [] for c in STRESS} for n in CFGS}      # per-image correctness lists
    for seed in seeds:
        for name in CFGS: preds[name].append(np.zeros(N, int))
        sc = {n: {c: np.zeros(N, int) for c in STRESS} for n in CFGS}
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X128, y, groups):
            for name, cfg in CFGS.items():
                imgs, lab, _ = train_set(tr, seed, cfg)
                m = make_model(cfg).fit(featurize(imgs, cfg), lab)
                preds[name][-1][te] = m.predict(featurize([X128[i] for i in te], cfg))
                for c in STRESS:
                    sc[name][c][te] = (m.predict(featurize([stress_img(i, c) for i in te], cfg)) == y[te])
        for name in CFGS:
            res[name].append(macro(y, preds[name][-1]))
            for c in STRESS: stress[name][c].append(sc[name][c])
        print('P1 seed', seed, 'done', flush=True)
    return res, preds, stress

# ----------------------------------------------------------- P2: leave-one-environment-out
def run_p2(seed=0):
    preds = {n: np.zeros(N, int) for n in CFGS}
    for e in range(4):
        te = np.where(ENV == e)[0]; tr = np.where(ENV != e)[0]
        for name, cfg in CFGS.items():
            imgs, lab, _ = train_set(tr, seed, cfg)
            m = make_model(cfg).fit(featurize(imgs, cfg), lab)
            preds[name][te] = m.predict(featurize([X128[i] for i in te], cfg))
    return preds

# ----------------------------------------------------------- P3: training curves (MLP on HOG, LOEO folds)
def run_curves(epochs=60, seed=0):
    out = {}
    variants = {'A1 data (real+aug)': dict(feat='hog', synth=False, enh=False),
                'A2 data (+synthetic)': dict(feat='hog', synth=True, enh=False),
                'A2 data (+synthetic, +enhance)': dict(feat='hog', synth=True, enh=True)}
    for vn, cfg in variants.items():
        curves = []
        for e in range(4):
            te = np.where(ENV == e)[0]; tr = np.where(ENV != e)[0]
            imgs, lab, _ = train_set(tr, seed, cfg)
            sc = StandardScaler().fit(featurize(imgs, cfg)); Xtr = sc.transform(featurize(imgs, cfg))
            Xte = sc.transform(featurize([X128[i] for i in te], cfg))
            clf = MLPClassifier((256,), alpha=1e-2, batch_size=64, learning_rate_init=1e-3, random_state=seed)
            rows = []
            for ep in range(epochs):
                clf.partial_fit(Xtr, lab, classes=np.arange(4))
                rows.append([clf.loss_, log_loss(y[te], clf.predict_proba(Xte), labels=range(4)),
                             clf.score(Xtr, lab), clf.score(Xte, y[te])])
            curves.append(rows)
        out[vn] = np.mean(curves, axis=0).tolist()      # epochs x [train_loss, val_loss, train_acc, val_acc]
        print('curves', vn, 'final val acc', round(out[vn][-1][3], 3), flush=True)
    return out

# ----------------------------------------------------------- P4: latency
def run_latency(n=100):
    cfg = CFGS['A2 full: + synthetic + enhance']; tr = np.arange(N)
    imgs, lab, _ = train_set(tr, 0, cfg); m = make_model(cfg).fit(featurize(imgs, cfg), lab)
    cfg1 = CFGS['A1 improved: HOG+SVM (+aug)']; imgs1, lab1, _ = train_set(tr, 0, cfg1); m1 = make_model(cfg1).fit(featurize(imgs1, cfg1), lab1)
    ts = {'preprocess': [], 'A1 HOG+SVM total': [], 'A2 enhance+HOG+SVM total': []}
    for k in range(n):
        raw = cv2.imread(paths[k % N]); t0 = time.perf_counter()
        im = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), (IMG, IMG), interpolation=cv2.INTER_AREA); t1 = time.perf_counter()
        m1.predict(featurize([im], cfg1)); t2 = time.perf_counter()
        m.predict(featurize([im], cfg)); t3 = time.perf_counter()
        ts['preprocess'].append((t1 - t0) * 1e3); ts['A1 HOG+SVM total'].append((t2 - t1 + t1 - t0) * 1e3)
        ts['A2 enhance+HOG+SVM total'].append((t3 - t2 + t1 - t0) * 1e3)
    return {k: dict(mean_ms=float(np.mean(v)), p95_ms=float(np.percentile(v, 95)), fps=float(1e3 / np.mean(v))) for k, v in ts.items()}

def boot_ci(correct, n=2000, seed=0):
    r = np.random.default_rng(seed); c = np.asarray(correct, float)
    s = [c[r.integers(0, len(c), len(c))].mean() for _ in range(n)]; return np.percentile(s, [2.5, 97.5])

if __name__ == '__main__':
    t0 = time.time()
    res, preds, stress = run_p1()
    summary = {}
    for n in CFGS:
        d = pd.DataFrame(res[n])
        corr = np.mean([(preds[n][s] == y) for s in range(3)], axis=0)
        lo, hi = boot_ci(corr)
        summary[n] = {**{m: [float(d[m].mean()), float(d[m].std(ddof=0))] for m in d.columns}, 'acc_ci95': [float(lo), float(hi)]}
    base = np.mean([(preds['A1 improved: HOG+SVM (+aug)'][s] == y) for s in range(3)], axis=0)
    full = np.mean([(preds['A2 full: + synthetic + enhance'][s] == y) for s in range(3)], axis=0)
    r = np.random.default_rng(1); diffs = [ (full - base)[r.integers(0, N, N)].mean() for _ in range(2000)]
    paired = dict(delta_acc=float((full - base).mean()), ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
    stress_tab = {n: {c: float(np.mean([v.mean() for v in stress[n][c]])) for c in STRESS} for n in CFGS}
    p2 = run_p2()
    env_tab = {n: {f'E{e}': float((p2[n][ENV == e] == y[ENV == e]).mean()) for e in range(4)} for n in CFGS}
    for n in CFGS:
        env_tab[n]['overall'] = float((p2[n] == y).mean()); env_tab[n]['macro_f1'] = float(macro(y, p2[n])['f1'])
    curves = run_curves()
    lat = run_latency()
    cms = {n: sum(confusion_matrix(y, p, labels=range(4)) for p in preds[n]).tolist() for n in CFGS}
    json.dump(dict(summary=summary, paired_full_vs_A1=paired, stress=stress_tab, loeo=env_tab, env_sizes=np.bincount(ENV).tolist(),
                   env_class_counts=[np.bincount(y[ENV == e], minlength=4).tolist() for e in range(4)],
                   curves=curves, latency=lat, confusion=cms, env_assign=ENV.tolist(), files=files,
                   n_synth_per_real=len(synth_for(0, 0)), n_aug_per_real=N_AUG),
              open(f'{RES}/results.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1)); print('paired', paired); print('LOEO', json.dumps(env_tab, indent=1))
    print('stress', pd.DataFrame(stress_tab).round(2).to_string()); print('lat', lat); print('time', time.time() - t0)
