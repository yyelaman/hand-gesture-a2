"""Build the on-disk SYNTHETIC part of the dataset from the real photos.
  data/synthetic/stress/<class>/*.jpg   stress-condition images (watermarked 'SYNTHETIC')
  data/synthetic/clips/<class>/*.mp4    short procedural camera clips (banner 'SYNTHETIC')
  data/synthetic/frames/<class>/*.jpg   frames auto-sampled from the clips (the ingest path used for Step-Video-T2V clips)
  data/synthetic/manifest.json
Usage: python src/make_synthetic.py
"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
out = os.path.join(ROOT, 'data', 'synthetic'); shutil.rmtree(out, ignore_errors=True)
for k in ('stress', 'clips', 'frames'):
    for c in CLASSES: os.makedirs(f'{out}/{k}/{c}')
files, paths, y, groups = load_real(); man = []
def stamp(im):
    bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR).copy()
    cv2.putText(bgr, 'SYNTHETIC', (4, bgr.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, 'SYNTHETIC', (4, bgr.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return bgr
for i, (f, p) in enumerate(zip(files, paths)):
    full = load_full_rgb(p, 256); c = CLASSES[y[i]]; stem = os.path.splitext(f)[0]
    rng = np.random.default_rng([0, i, 7]); mask = None
    for cond in rng.choice(CONDITIONS, 6, replace=False):
        if cond == 'new_background' and mask is None: mask = hand_mask(full)
        im = stress_condition(full, str(cond), rng, mask=mask if cond == 'new_background' else None)
        fn = f'SYN_{stem}_{cond}.jpg'; cv2.imwrite(f'{out}/stress/{c}/{fn}', stamp(im), [cv2.IMWRITE_JPEG_QUALITY, 92])
        man.append(dict(file=f'stress/{c}/{fn}', kind='stress', source=f, label=c, condition=str(cond)))
    for k in range(2):
        cond = rng.choice(CONDITIONS[:7] + [None, None]); cond = None if cond is None else str(cond)
        fr = make_clip(full, rng, cond=cond); name = f'SYN_{stem}_clip{k}'
        write_clip(fr, f'{out}/clips/{c}/{name}.mp4')
        man.append(dict(file=f'clips/{c}/{name}.mp4', kind='clip', source=f, label=c, condition=cond or 'camera_motion_only'))
        for j, im in enumerate(sample_frames(f'{out}/clips/{c}/{name}.mp4', 3, crop_bottom=BANNER)):
            fn = f'{name}_f{j}.jpg'; cv2.imwrite(f'{out}/frames/{c}/{fn}', cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            man.append(dict(file=f'frames/{c}/{fn}', kind='clip_frame', source=f, label=c, condition=cond or 'camera_motion_only'))
json.dump(dict(note='ALL FILES IN THIS FOLDER ARE SYNTHETIC (procedural stand-ins, derived from the 68 real photos). '
               'Never mix them into a TEST set; derive them only from training photos.', items=man),
          open(f'{out}/manifest.json', 'w'), indent=1)
print(len(man), 'synthetic items;', sum(m['kind'] == 'clip' for m in man), 'clips')
