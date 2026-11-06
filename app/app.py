"""Hand-gesture demo (Assignment 2). Run:  python app/app.py   ->  http://127.0.0.1:5000
upload / webcam -> prediction -> occlusion heatmap + nearest neighbours; latency + environment; stress-test generator."""
import os, sys, io, time, base64, platform
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, cv2
from flask import Flask, request, jsonify, render_template
from common import *
import explain as X
B = X.load_bundle(); MODEL, BANK = B['model'], B['bank']
app = Flask(__name__); LAT = []
X.predict(MODEL, np.zeros((IMG, IMG, 3), np.uint8)); X.occlusion_map(MODEL, np.zeros((IMG, IMG, 3), np.uint8))   # warm-up

def b64(rgb, q=88):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q]); return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
def watermark(rgb, text='SYNTHETIC'):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy(); h = bgr.shape[0]
    cv2.putText(bgr, text, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA); cv2.putText(bgr, text, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
def read_img(f):
    arr = np.frombuffer(f.read(), np.uint8); im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None: raise ValueError('not an image')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); s = 512 / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im

@app.route('/')
def index(): return render_template('index.html', conditions=CONDITIONS, server=f'{platform.system()} {platform.machine()}, {os.cpu_count()} CPU, Python {platform.python_version()}, CPU-only (no GPU)')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try: im = read_img(request.files['image'])
    except Exception as e: return jsonify(error=str(e)), 400
    cond = request.form.get('condition', 'none'); synthetic = cond != 'none'
    if synthetic: im = stress_condition(im, cond, np.random.default_rng(int(request.form.get('seed', 0))), severity=float(request.form.get('severity', .8)))
    t0 = time.perf_counter(); x = cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA)
    k, pr, _ = X.predict(MODEL, x); t1 = time.perf_counter()
    heat = X.occlusion_map(MODEL, x); t2 = time.perf_counter()
    nn, dist = X.neighbours(BANK, x, 4); t3 = time.perf_counter()
    root_img = lambda i: cv2.cvtColor(cv2.imread(os.path.join(ROOT, B['paths'][i])), cv2.COLOR_BGR2RGB)
    nns = [dict(img=b64(cv2.resize(root_img(i), (160, 160))), label=CLASSES[B['labels'][i]]) for i in nn]
    inf_ms = (t1 - t0) * 1e3; LAT.append(inf_ms)
    shown = watermark(cv2.resize(im, (256, 256))) if synthetic else cv2.resize(im, (256, 256))
    return jsonify(label=CLASSES[k], scores={c: float(p) for c, p in zip(CLASSES, pr)}, input=b64(shown),
                   enhanced=b64(cv2.resize(enhance(x), (256, 256))), heatmap=b64(X.overlay(x, heat)), neighbours=nns, synthetic=synthetic, condition=cond,
                   latency=dict(inference_ms=round(inf_ms, 2), heatmap_ms=round((t2 - t1) * 1e3, 1), nn_ms=round((t3 - t2) * 1e3, 2), avg_inference_ms=round(float(np.mean(LAT)), 2), n=len(LAT)))
if __name__ == '__main__': app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)))
