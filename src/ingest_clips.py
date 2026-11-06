"""Ingest generated gesture clips (Step-Video-T2V from Colab, or the procedural ones) -> training frames.
Usage: python src/ingest_clips.py CLIPS_DIR OUT_DIR [--frames 6] [--crop-bottom 0]
CLIPS_DIR must contain one sub-folder per class (peace/ thumb/ stop/ ok/) with .mp4 files.
Output frames are written to OUT_DIR/<class>/ and can be added to training (never to the test set)."""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('clips'); ap.add_argument('out')
ap.add_argument('--frames', type=int, default=6); ap.add_argument('--crop-bottom', type=int, default=0)
a = ap.parse_args(); n = 0
for c in CLASSES:
    os.makedirs(f'{a.out}/{c}', exist_ok=True)
    for v in sorted(glob.glob(f'{a.clips}/{c}/*.mp4')):
        for j, im in enumerate(sample_frames(v, a.frames, a.crop_bottom)):
            cv2.imwrite(f'{a.out}/{c}/{os.path.splitext(os.path.basename(v))[0]}_f{j}.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR)); n += 1
print('wrote', n, 'frames to', a.out)
