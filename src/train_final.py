"""Train the final A2 model on ALL real photos (+aug +synthetic, enhancement on) and save results/model.joblib"""
import os, sys, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as E
from common import *
cfg = E.CFGS['A2 full: + synthetic + enhance']; tr = np.arange(E.N)
imgs, lab, src = E.train_set(tr, 0, cfg)
m = E.make_model(cfg).fit(E.featurize(imgs, cfg), lab)
bank = np.stack([feat_hog(enhance(i)) for i in E.X128])
joblib.dump(dict(model=m, bank=bank, paths=[os.path.relpath(p, ROOT) for p in E.paths], labels=E.y, classes=CLASSES,
                 n_train=len(imgs), n_real=len(tr), n_synth=src.count('synth')), os.path.join(ROOT, 'results', 'model.joblib'))
print('trained on', len(imgs), 'imgs', {s: src.count(s) for s in set(src)})
