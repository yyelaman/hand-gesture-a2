"""Record the demo video (silent, with captions) of the running app with Playwright. Needs the app on PORT 5055."""
import os, sys, time, glob, shutil
from playwright.sync_api import sync_playwright
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); out = os.path.join(ROOT, 'demo'); os.makedirs(out, exist_ok=True)
tmp = os.path.join(out, '_rec'); shutil.rmtree(tmp, ignore_errors=True)
def cap(page, t, sec):
    page.evaluate("""t=>{let d=document.getElementById('capbar');if(!d){d=document.createElement('div');d.id='capbar';d.style.cssText='position:fixed;left:0;right:0;bottom:0;background:rgba(0,0,0,.82);color:#fff;font:600 20px system-ui;padding:14px 22px;z-index:99';document.body.appendChild(d)}d.textContent=t}""", t); page.wait_for_timeout(int(sec * 1750))
def img(c, n): return glob.glob(f'{ROOT}/data/real/{c}/*.jpg')[n]
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'])
    ctx = b.new_context(viewport={'width': 1180, 'height': 760}, record_video_dir=tmp, record_video_size={'width': 1180, 'height': 760}); pg = ctx.new_page()
    pg.goto('http://127.0.0.1:5055'); 
    cap(pg, 'Assignment 2 - Hand gesture recognition: A1 baseline upgraded with a synthetic-data pipeline', 6)
    cap(pg, 'Consent notice: images stay in memory; all generated media is watermarked SYNTHETIC', 7)
    cap(pg, 'Environment panel: browser + CPU-only server; latency is measured per request', 6)
    pg.set_input_files('#file', img('peace', 2)); cap(pg, 'Step 1: upload a photo (or use the camera button)', 4)
    pg.click('#run'); pg.wait_for_selector('#out', state='visible'); cap(pg, 'Prediction + relative scores + latency (avg over requests)', 8)
    pg.evaluate("document.getElementById('out').scrollIntoView()"); cap(pg, 'Explanation: occlusion heatmap (red = evidence) and the 4 nearest training photos', 10)
    pg.evaluate("window.scrollTo(0,0)")
    for cond, g, n, msg in [('lowlight', 'stop', 3, 'Stress test: SYNTHETIC low light - the enhancement panel recovers exposure before classification'),
                            ('motion_blur', 'thumb', 5, 'Stress test: SYNTHETIC motion blur (what a T2V camera move would produce)'),
                            ('new_background', 'ok', 2, 'Stress test: SYNTHETIC background replacement (hand mask + procedural background)'),
                            ('overexposed', 'peace', 6, 'Stress test: SYNTHETIC over-exposure')]:
        pg.set_input_files('#file', img(g, n)); pg.select_option('#cond', cond); pg.click('#run'); pg.wait_for_timeout(1500)
        pg.evaluate("document.getElementById('out').scrollIntoView()"); cap(pg, msg, 11); pg.evaluate("window.scrollTo(0,0)")
    pg.click('#cam'); cap(pg, 'Camera mode: start camera, capture a frame, then Predict (same pipeline)', 7)
    cap(pg, 'Results (report): +SYNTHETIC data raises accuracy under stress (e.g. motion blur 35% -> 53%) but clean/LOEO gains are within noise', 12)
    cap(pg, 'Limits: 68 photos, in-distribution stress generator, Step-Video-T2V / ArtAug replaced by classical stand-ins - see report', 10)
    ctx.close(); b.close()
v = glob.glob(f'{tmp}/*.webm')[0]
os.system(f'ffmpeg -y -loglevel error -i "{v}" -c:v libx264 -pix_fmt yuv420p -crf 26 "{out}/demo.mp4"'); shutil.rmtree(tmp)
print('ok')
