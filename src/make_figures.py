import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import StratifiedGroupKFold
import experiments as E, explain as X
from common import *
R = json.load(open(f'{ROOT}/results/results.json')); F = f'{ROOT}/figures'; os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
ENVN = ['E0 light table/floor', 'E1 street granite', 'E2 lap / jeans', 'E3 dark tile']
SHORT = {'A1 baseline: pixels+LogReg (+aug)': 'A1 pixels+LogReg', 'A1 improved: HOG+SVM (+aug)': 'A1 HOG+SVM',
         'A2: HOG+SVM + synthetic': 'A2 +synthetic', 'A2: HOG+SVM + enhance': 'A2 +enhance', 'A2 full: + synthetic + enhance': 'A2 full'}
COL = ['#9aa5b1', '#4C72B0', '#DD8452', '#55A868', '#C44E52']

# 1 dataset composition
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
real = np.bincount(E.y, minlength=4); aug = real * 10; syn = real * 12
ax[0].bar(CLASSES, real, label='real photos', color='#4C72B0'); ax[0].bar(CLASSES, aug, bottom=real, label='classic aug (x10)', color='#8fb3d9')
ax[0].bar(CLASSES, syn, bottom=real + aug, label='SYNTHETIC (x12)', color='#DD8452'); ax[0].legend(fontsize=7); ax[0].set_title('Training images per class (all 68 photos)')
tot = [real.sum(), aug.sum(), syn.sum()]; ax[1].pie(tot, labels=[f'real {tot[0]}', f'aug {tot[1]}', f'synthetic {tot[2]}'], colors=['#4C72B0', '#8fb3d9', '#DD8452'], autopct='%1.0f%%', textprops={'fontsize': 8})
ax[1].set_title('Share of the final training set'); plt.tight_layout(); plt.savefig(f'{F}/dataset.png', dpi=170); plt.close()

# 2 synthetic grid + environments
rng = np.random.default_rng(3); i = int(np.where(E.y == 2)[0][4]); im = E.FULL[i]
tiles = [('original', im)] + [(c, stress_condition(im, c, rng)) for c in CONDITIONS] + [('lowlight+enhance', enhance(stress_condition(im, 'lowlight', rng, severity=1.0)))]
fig, axs = plt.subplots(3, 6, figsize=(11, 6.4))
for a, (t, m) in zip(axs.ravel(), tiles): a.imshow(m); a.set_title(t, fontsize=8); a.axis('off')
for a in axs.ravel()[len(tiles):]:
    a.axis('off')
fr = make_clip(im, np.random.default_rng(5))
for a, k in zip(axs.ravel()[len(tiles):], [0, 5, 10, 15, 20, 23]): a.imshow(fr[k]); a.set_title(f'SYNTH clip frame {k}', fontsize=8)
plt.suptitle('SYNTHETIC stress conditions & clip frames (procedural stand-ins) - all derived from one real photo', fontsize=9); plt.tight_layout(); plt.savefig(f'{F}/synthetic_samples.png', dpi=150); plt.close()
fig, axs = plt.subplots(4, 6, figsize=(9, 8.4))
for e in range(4):
    idx = np.where(E.ENV == e)[0][:6]
    for k in range(6):
        a = axs[e, k]; a.axis('off')
        if k < len(idx): a.imshow(E.FULL[idx[k]])
        if k == 0: a.set_title(f'{ENVN[e]} (n={int((E.ENV == e).sum())})', loc='left', fontsize=9, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{F}/environments.png', dpi=110); plt.close()

# 3 metrics bars
names = list(E.CFGS); fig, ax = plt.subplots(figsize=(7.5, 3)); w = .2
for k, m in enumerate(['accuracy', 'precision', 'recall', 'f1']):
    ax.bar(np.arange(5) + (k - 1.5) * w, [R['summary'][n][m][0] for n in names], w, yerr=[R['summary'][n][m][1] for n in names], label=m, capsize=1.5)
ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[n] for n in names], fontsize=8); ax.set_ylim(0, .8); ax.legend(ncol=4, fontsize=7); ax.set_title('Clean real test photos, 5-fold grouped CV x3 (macro avg, mean+-std over repeats)')
plt.tight_layout(); plt.savefig(f'{F}/metrics.png', dpi=170); plt.close()

# 4 stress heatmap
conds = ['clean'] + CONDITIONS; M = np.array([[R['stress'][n][c] for c in conds] for n in names])
fig, ax = plt.subplots(figsize=(8, 2.7)); im_ = ax.imshow(M, cmap='YlGn', vmin=.2, vmax=.7)
for a in range(5):
    for b in range(len(conds)): ax.text(b, a, f'{M[a,b]*100:.0f}', ha='center', va='center', fontsize=8)
ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=30, ha='right', fontsize=8); ax.set_yticks(range(5)); ax.set_yticklabels([SHORT[n] for n in names], fontsize=8)
ax.set_title('Accuracy (%) by SYNTHETIC test condition', fontsize=9); plt.tight_layout(); plt.savefig(f'{F}/stress.png', dpi=170, bbox_inches='tight'); plt.close()

# 5 LOEO
fig, ax = plt.subplots(figsize=(7.5, 3)); w = .16
for k, n in enumerate(names): ax.bar(np.arange(5) + (k - 2) * w, [R['loeo'][n][f'E{e}'] for e in range(4)] + [R['loeo'][n]['overall']], w, label=SHORT[n], color=COL[k])
ax.set_xticks(range(5)); ax.set_xticklabels([f'{ENVN[e]}\n(n={R["env_sizes"][e]})' for e in range(4)] + ['overall'], fontsize=7); ax.set_ylim(0, .9); ax.legend(fontsize=6, ncol=5, loc='upper center'); ax.set_title('Leave-one-environment-out accuracy (REAL domain shift)')
plt.tight_layout(); plt.savefig(f'{F}/loeo.png', dpi=170); plt.close()

# 6 curves
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3))
for (n, c), col in zip(R['curves'].items(), ['#4C72B0', '#DD8452', '#C44E52']):
    c = np.array(c); ax[0].plot(c[:, 0], color=col, label=n + ' (train)'); ax[0].plot(c[:, 1], color=col, ls='--', label=n + ' (val)'); ax[1].plot(c[:, 2], color=col, label=n + ' (train)'); ax[1].plot(c[:, 3], color=col, ls='--', label=n + ' (val)')
ax[0].set_title('Loss'); ax[1].set_title('Accuracy'); ax[0].set_xlabel('epoch'); ax[1].set_xlabel('epoch'); ax[0].legend(fontsize=5.5); ax[0].set_yscale('log')
plt.suptitle('MLP-on-HOG training curves, mean of 4 leave-one-environment-out folds', fontsize=9); plt.tight_layout(); plt.savefig(f'{F}/curves.png', dpi=170); plt.close()

# 7 confusion
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
for a, n in zip(ax, ['A1 improved: HOG+SVM (+aug)', 'A2 full: + synthetic + enhance']):
    cm = np.array(R['confusion'][n]); cn = cm / cm.sum(1, keepdims=True); a.imshow(cn, cmap='Blues', vmin=0, vmax=1)
    for i_ in range(4):
        for j in range(4): a.text(j, i_, f'{cn[i_,j]*100:.0f}%', ha='center', va='center', color='white' if cn[i_, j] > .5 else 'black')
    a.set_xticks(range(4)); a.set_xticklabels(CLASSES); a.set_yticks(range(4)); a.set_yticklabels(CLASSES); a.set_title(SHORT[n]); a.set_xlabel('predicted'); a.set_ylabel('true')
plt.tight_layout(); plt.savefig(f'{F}/confusion.png', dpi=170); plt.close()

# 8 explainability (model never saw these photos: fold 0, seed 0)
cfg = E.CFGS['A2 full: + synthetic + enhance']; tr, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(E.X128, E.y, E.groups)))
imgs, lab, _ = E.train_set(tr, 0, cfg); m = E.make_model(cfg).fit(E.featurize(imgs, cfg), lab); bank = np.stack([feat_hog(enhance(E.X128[i])) for i in tr])
pred = m.predict(E.featurize([E.X128[i] for i in te], cfg)); ok = [j for j in range(len(te)) if pred[j] == E.y[te[j]]]; bad = [j for j in range(len(te)) if pred[j] != E.y[te[j]]]
sel = []
for c in range(4):
    cand = [j for j in ok if E.y[te[j]] == c]
    if cand: sel.append(cand[0])
sel = sel[:3] + bad[:2]
fig, axs = plt.subplots(len(sel), 7, figsize=(11, 1.9 * len(sel)))
for r, j in enumerate(sel):
    i = te[j]; im128 = E.X128[i]; k, pr, _ = X.predict(m, im128); h = X.occlusion_map(m, im128); nn, _ = X.neighbours(bank, im128, 4)
    axs[r, 0].imshow(im128); axs[r, 0].set_title(f'true: {CLASSES[E.y[i]]}', fontsize=8); axs[r, 1].imshow(X.overlay(im128, h)); axs[r, 1].set_title(f'pred: {CLASSES[k]} ({pr[k]*100:.0f}%)', fontsize=8, color='green' if k == E.y[i] else 'red')
    for q in range(4):
        t = tr[nn[q]]; axs[r, 2 + q].imshow(E.X128[t]); axs[r, 2 + q].set_title(f'NN{q+1}: {CLASSES[E.y[t]]}', fontsize=7)
    axs[r, 6].imshow(enhance(im128)); axs[r, 6].set_title('after enhance', fontsize=7)
    for a in axs[r]: a.axis('off')
plt.suptitle('Explainability on UNSEEN photos: occlusion heatmap (red = evidence) + 4 nearest training photos in HOG space', fontsize=9); plt.tight_layout(); plt.savefig(f'{F}/explain.png', dpi=130); plt.close()

# 9 diagrams
def box(ax, x, y_, w, h, t, c='#e8eef7'):
    ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle='round,pad=0.02', fc=c, ec='#333', lw=.8)); ax.text(x + w / 2, y_ + h / 2, t, ha='center', va='center', fontsize=7.5)
def arr(ax, x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, lw=.9))
fig, ax = plt.subplots(figsize=(10, 3.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis('off')
box(ax, .1, 2.3, 1.7, .9, 'text prompt\n"a hand showing a\npeace sign, dim room"', '#fff3d6'); box(ax, 2.2, 2.3, 1.9, .9, 'Step-Video-T2V\n(30B DiT, offline Colab/GPU)\nOR procedural stand-in', '#fde0c8')
box(ax, 4.5, 2.3, 1.6, .9, 'short clip\n(watermarked\nSYNTHETIC)', '#fde0c8'); box(ax, 6.5, 2.3, 1.5, .9, 'auto-sample\nframes\n(ingest_clips.py)', '#fde0c8'); box(ax, 8.3, 2.3, 1.5, .9, 'ArtAug-style\npolish of dark /\nwashed frames', '#fde0c8')
for x in [1.8, 4.1, 6.1, 8.0]: arr(ax, x, 2.75, x + .4, 2.75)
box(ax, .1, .3, 1.7, .9, '68 real photos\n(A1 dataset)', '#dfeedd'); box(ax, 2.2, .3, 1.9, .9, 'classic aug x10\n(A1)', '#dfeedd'); box(ax, 4.5, .3, 1.6, .9, 'HOG features', '#dfeedd'); box(ax, 6.5, .3, 1.5, .9, 'RBF-SVM', '#dfeedd'); box(ax, 8.3, .3, 1.5, .9, 'label +\nocclusion heatmap\n+ NN gallery', '#dfeedd')
for x in [1.8, 4.1, 6.1, 8.0]: arr(ax, x, .75, x + .4, .75)
arr(ax, 9.05, 2.3, 5.3, 1.25); ax.text(7.4, 1.75, 'synthetic training frames\n(never from test photos)', fontsize=7, ha='center', color='#a0522d')
plt.savefig(f'{F}/pipeline.png', dpi=170, bbox_inches='tight'); plt.close()
print('figures done')
