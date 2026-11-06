import os, json, numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); F = f'{ROOT}/figures'
R = json.load(open(f'{ROOT}/results/results.json'))
ss = getSampleStyleSheet()
B = ParagraphStyle('b', parent=ss['BodyText'], fontSize=9.2, leading=12.2, spaceAfter=4)
H1 = ParagraphStyle('h1', parent=ss['Heading1'], fontSize=13.5, spaceBefore=8, spaceAfter=4)
H2 = ParagraphStyle('h2', parent=ss['Heading2'], fontSize=10.5, spaceBefore=5, spaceAfter=2)
CAP = ParagraphStyle('c', parent=B, fontSize=8, textColor=colors.HexColor('#444'), leading=10)
BOX = ParagraphStyle('box', parent=B, backColor=colors.HexColor('#fff4e0'), borderColor=colors.HexColor('#b45309'), borderWidth=.8, borderPadding=5, spaceBefore=4, spaceAfter=8)
SM = ParagraphStyle('sm', parent=B, fontSize=8.2, leading=10.4)
def P(t, s=B): return Paragraph(t, s)
def img(n, w):
    from PIL import Image as I
    a, b = I.open(f'{F}/{n}').size; return Image(f'{F}/{n}', width=w * cm, height=w * cm * b / a)
def tab(rows, cw, fs=8, hl=None):
    t = Table(rows, colWidths=[c * cm for c in cw])
    st = [('FONTSIZE', (0, 0), (-1, -1), fs), ('GRID', (0, 0), (-1, -1), .4, colors.grey), ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef7')), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')]
    if hl is not None: st.append(('BACKGROUND', (0, hl), (-1, hl), colors.HexColor('#e6f3e6')))
    t.setStyle(TableStyle(st)); return t
pc = lambda v: f'{v*100:.1f}'
NAMES = list(R['summary']); S = R['summary']; ST = R['stress']; LO = R['loeo']
SH = {NAMES[0]: 'A1 baseline: pixels + LogReg', NAMES[1]: 'A1 improved: HOG + SVM', NAMES[2]: 'A2: + synthetic', NAMES[3]: 'A2: + enhancement', NAMES[4]: 'A2 full: + synthetic + enhancement'}
CONDS = [c for c in ST[NAMES[0]] if c != 'clean']
mean_stress = {n: np.mean([ST[n][c] for c in CONDS]) for n in NAMES}
d = R['paired_full_vs_A1']; lat = R['latency']; ENVN = ['E0 light table/floor', 'E1 street granite', 'E2 lap/jeans', 'E3 dark tile']
story = []

# ---------------------------------------------------------------- title + summary
story += [P('<b>Hand Gesture Recognition: Leveling Up the A1 Baseline with a Synthetic-Data + Enhancement Pipeline</b>', ParagraphStyle('t', parent=H1, fontSize=16, leading=19)),
          P('Computer Vision - Assignment 2 (continuation of A1) - Astana IT University', CAP),
          P('<b>Abstract.</b> We keep the A1 topic (4 gestures: <i>peace, thumb, stop, ok</i>; 68 self-collected photos) and add the two SOTA components recommended for this topic - '
            '<b>Step-Video-T2V</b> (synthetic gesture clips with varied backgrounds / camera motion) and <b>ArtAug</b> (polishing low-exposure frames) - as pipeline stages: '
            '<i>generate &rarr; auto-sample frames &rarr; polish &rarr; train on real + synthetic &rarr; evaluate by environment</i>. We ship a web demo (upload/camera &rarr; label &rarr; occlusion heatmap + nearest neighbours, latency, stress-test generator) '
            'and report accuracy per environment and latency, as the assignment asks for this topic.'),
          P('<b>Read this first (what was and was not done).</b> The 30B-parameter Step-Video-T2V and the FLUX-based ArtAug need large GPUs and internet access to download weights; neither was available in the environment where this project was built. '
            'Their roles are therefore played by <b>classical, procedural stand-ins</b> that are clearly labelled <b>SYNTHETIC</b> everywhere: a stress-condition generator and a camera-clip generator (instead of T2V), and a gamma/CLAHE/white-balance enhancer (instead of ArtAug). '
            'The pipeline accepts real T2V clips unchanged (<font face="Courier">ingest_clips.py</font>, Colab template provided, <b>not executed</b>). All numbers below come from the stand-ins, on 68 real photos, so they must be read as a small-data study, not as evidence about the real SOTA models. No external dataset could be downloaded (the sandbox network was closed), so the "rest of the dataset" is synthetic, derived from our own photos.', BOX)]

# ---------------------------------------------------------------- 1 dataset
story += [P('1. Dataset update', H1),
          P(f'<b>Real data (unchanged from A1):</b> 68 phone photos, 4 classes (peace 18, thumb 18, stop 19, ok 13), several people/backgrounds; near-duplicates share a group id so they never straddle train/test. '
            f'<b>Added in A2 (all SYNTHETIC):</b> for every <i>training</i> photo we generate {R["n_synth_per_real"]} synthetic images: 6 stress-condition images (random 6 of 9 conditions: low light, backlight, over-exposure, motion blur, defocus, sensor noise, colour cast, camera tilt, background replacement via GrabCut hand mask) '
            'and 2 short procedural camera clips (24 frames: pan/zoom/roll with easing, speed-dependent motion blur, exposure drift, optional stress) from which 3 frames each are auto-sampled. '
            f'In the final model the training set has 68 real + {R["n_aug_per_real"]*68} classic-augmentation + {R["n_synth_per_real"]*68} synthetic images, i.e. <b>synthetic share = 52%</b> (0% of any test set). '
            'On disk: <font face="Courier">data/synthetic</font> holds 952 items (816 stress/frames images and 136 mp4 clips) with a manifest; images and clips carry a "SYNTHETIC" watermark/banner.'),
          P('<b>Leakage control.</b> Synthetic data of a fold is generated only from that fold\'s training photos; test photos are never used to synthesise anything. Augmentations of A1 (rotation, scale, shift, flip, brightness, blur, noise) are kept (10 copies/photo).'),
          img('dataset.png', 13.5), P('Fig. 1 - Training-set composition. Class balance is unchanged because every class gets the same multipliers.', CAP),
          img('synthetic_samples.png', 14.5), P('Fig. 2 - Synthetic stress conditions and clip frames generated from one real photo (stand-ins for Step-Video-T2V / ArtAug).', CAP)]

# ---------------------------------------------------------------- 2 SOTA
story += [P('2. SOTA methods and why they help this topic', H1),
          img('pipeline.png', 15.5), P('Fig. 3 - Upper row: SOTA data branch (T2V &rarr; frames &rarr; polish), executed offline; lower row: A1 recogniser. In this repo the orange boxes run as procedural stand-ins.', CAP),
          P('<b>Step-Video-T2V (text &rarr; video).</b> An open-source text-to-video foundation model with <b>30 B parameters</b>, producing clips of up to <b>204 frames at 544x992</b> from Chinese or English prompts. '
            'Its <b>Video-VAE compresses 16x16 spatially and 8x temporally</b>, so a diffusion-transformer (DiT: 48 layers, 48 heads, 128 dims/head) can work in a small latent space; two text encoders (English, Chinese) condition it, and Direct Preference Optimisation sharpens visual quality. '
            'Recommended hardware is ~80 GB GPU memory (a Turbo variant uses 10-15 steps). <i>Why it helps us:</i> the real set has only 13-19 photos per class and few backgrounds; prompts such as "<i>a hand showing a peace sign, dim room, handheld camera</i>" '
            'can yield many viewpoints, backgrounds and motion-blurred frames for the same gesture, which A1 identified as the main weakness (domain shift between photo sessions). Frames are auto-sampled and added to <i>training only</i>; a "stress-test generator" builds environment-specific test sets.'),
          P('<b>ArtAug (synthesis-understanding loop).</b> An open-source (Apache-2.0) framework from East China Normal University / Alibaba that improves a text-to-image model (FLUX.1-dev) without manual annotation: the generator makes an image, a multimodal LLM (Qwen2-VL-72B) '
            'proposes fine-grained improvements, and a LoRA module learns them iteratively so that inference costs nothing extra; the paper reports aesthetic score 6.81 vs 6.35 for the base model. '
            '<i>Why it helps us:</i> low-exposure / poorly composed frames are exactly where HOG gradients vanish; polishing them before training and inference makes dark and washed-out hands comparable to well-lit ones. '
            'Caveat: ArtAug is a <i>generative</i> enhancer, so it could alter finger shapes; labels must be checked after polishing (we use a deterministic, shape-preserving enhancer instead).'),
          P('<b>What we actually implemented.</b> (i) <font face="Courier">stress_condition</font> + <font face="Courier">make_clip</font> = procedural substitute for T2V clips; (ii) <font face="Courier">enhance</font> = soft gray-world white balance, auto-gamma to target luminance 0.45, CLAHE on L, bilateral denoise, unsharp mask = substitute for ArtAug polishing; '
            '(iii) the real Step-Video-T2V path is <font face="Courier">notebooks/step_video_t2v_colab.ipynb</font> (prompt grid 4 gestures x 6 environments x 4 camera moves = 96 prompts; template, not executed) + <font face="Courier">src/ingest_clips.py</font>.'),
          P('Sources (Neurohive pages): Step-Video-T2V - neurohive.io/en/state-of-the-art/step-video-t2v-text-to-video-open-source-model-achieves-16x-video-compression-breakthrough/ ; '
            'ArtAug - neurohive.io/en/state-of-the-art/artaug-a-novel-synthesis-understanding-framework-for-image-generation-enhancement/ ; ArtAug paper arXiv:2412.12888; Step-Video-T2V report arXiv:2502.10248.', SM)]

# ---------------------------------------------------------------- 3 experiments
story += [P('3. Experiments', H1),
          P('<b>Models.</b> A1 baseline = standardised 32x32 pixels + logistic regression (C=0.1); A1 improved = CLAHE + HOG (9 orient., 16x16 cells, 1764 features) + RBF-SVM (C=10). Hyper-parameters are those of A1 and were <i>not</i> tuned on the evaluation folds. '
            'No deep CNN could be trained here (no PyTorch/TensorFlow, no pretrained weights); the MobileNetV2 cell of A1 remains the natural next step. '
            '<b>Protocols.</b> <b>P1</b>: 5-fold stratified-group CV x 3 repeats (as in A1) on clean photos + the same held-out photos pushed through each synthetic stress condition (test parameters drawn with a different seed than training synthesis). '
            '<b>P2</b>: leave-one-environment-out (LOEO) - the photos were clustered by background colour/brightness into 4 real environments (Fig. 4: E0 light table/floor n=20, E1 street granite n=14, E2 lap/jeans n=7, E3 dark tile n=27); '
            'the model is trained on three environments and tested on the fourth, a <i>real</i> domain shift with no synthetic test data. <b>P3</b>: training curves (MLP on HOG, LOEO). <b>P4</b>: latency.'),
          img('environments.png', 9.5), P('Fig. 4 - The four pseudo-environments used for "accuracy by environment" (6 example photos each).', CAP)]
rows = [['Model (clean real photos, CV x3)', 'Acc %', 'Macro-P', 'Macro-R', 'Macro-F1', 'Acc 95% CI']]
for n in NAMES: rows.append([SH[n], f'{pc(S[n]["accuracy"][0])}+-{S[n]["accuracy"][1]*100:.1f}', pc(S[n]['precision'][0]), pc(S[n]['recall'][0]), pc(S[n]['f1'][0]), f'{pc(S[n]["acc_ci95"][0])}-{pc(S[n]["acc_ci95"][1])}'])
story += [P('<b>Table 1 - Baseline vs. SOTA pipeline, clean test photos</b> (mean+-std over 3 CV repeats; CI = bootstrap over the 68 photos).', SM), tab(rows, [6.2, 2.2, 1.6, 1.6, 1.7, 2.6], hl=5),
          P(f'On clean photos the full A2 pipeline is {d["delta_acc"]*100:+.1f} pp vs. A1 HOG+SVM, with a paired bootstrap 95% CI of [{d["ci95"][0]*100:+.1f}, {d["ci95"][1]*100:+.1f}] pp: <b>not distinguishable from zero</b> with 68 photos (one photo = 1.5 pp).'), Spacer(1, 4)]
story += [img('metrics.png', 11.5), P('Fig. 5 - Macro metrics on clean photos.', CAP)]
rows = [['Condition', 'A1 pix', 'A1 HOG', 'A2 +syn', 'A2 +enh', 'A2 full']]
for c in ['clean'] + CONDS: rows.append([c] + [f'{ST[n][c]*100:.0f}' for n in NAMES])
rows.append(['mean of 9 stress'] + [f'{mean_stress[n]*100:.1f}' for n in NAMES])
story += [P('<b>Table 2 - Accuracy (%) by synthetic environment/condition</b> (held-out photos, 3 repeats). Caution: test corruptions come from the same generator family as the synthetic training data, so this favours A2 (optimistic).', SM),
          tab(rows, [3.6, 2.2, 2.2, 2.2, 2.2, 2.2], hl=len(rows) - 1), Spacer(1, 4), img('stress.png', 14)]
rows = [['Real environment (held out)'] + [SH[n].split(':')[0] + ':' + SH[n].split(':')[1][:14] if False else ['A1 pix', 'A1 HOG', 'A2 +syn', 'A2 +enh', 'A2 full'][i] for i, n in enumerate(NAMES)]]
for e in range(4): rows.append([f'{ENVN[e]} (n={R["env_sizes"][e]})'] + [f'{LO[n][f"E{e}"]*100:.0f}' for n in NAMES])
rows.append(['overall accuracy'] + [f'{LO[n]["overall"]*100:.1f}' for n in NAMES]); rows.append(['macro-F1'] + [f'{LO[n]["macro_f1"]*100:.1f}' for n in NAMES])
story += [P('<b>Table 3 - Leave-one-environment-out accuracy (%), REAL domain shift.</b>', SM), tab(rows, [5.2, 1.8, 1.8, 1.8, 1.8, 1.8]), Spacer(1, 4), img('loeo.png', 11.5)]
story += [P('<b>Findings.</b> (1) HOG+SVM is far better than raw pixels in every protocol (A1 result reproduced: 57% vs 31%; the 60% quoted in A1 differs slightly because our CV repeats/seeds differ). '
            f'(2) Under <i>synthetic</i> stress the A2 pipeline is clearly more robust: mean accuracy over the 9 conditions {mean_stress[NAMES[1]]*100:.1f}% &rarr; {mean_stress[NAMES[4]]*100:.1f}%, largest gains for motion blur ({ST[NAMES[1]]["motion_blur"]*100:.0f}&rarr;{ST[NAMES[4]]["motion_blur"]*100:.0f}), new background ({ST[NAMES[1]]["new_background"]*100:.0f}&rarr;{ST[NAMES[4]]["new_background"]*100:.0f}), over-exposure ({ST[NAMES[1]]["overexposed"]*100:.0f}&rarr;{ST[NAMES[4]]["overexposed"]*100:.0f}) and sensor noise ({ST[NAMES[1]]["sensor_noise"]*100:.0f}&rarr;{ST[NAMES[4]]["sensor_noise"]*100:.0f}); camera tilt is not solved. '
            f'(3) On <i>real</i> unseen environments (LOEO) A2 does <b>not</b> help: overall {LO[NAMES[1]]["overall"]*100:.1f}% (A1 HOG+SVM) vs {LO[NAMES[4]]["overall"]*100:.1f}% (A2 full), with mixed per-environment results and tiny subsets (E2 has 7 photos). '
            'So the synthetic data buys robustness to the corruptions it simulates, but we have no evidence that it transfers to real new environments; plausible reasons (untested) are the 52% synthetic share diluting 68 real photos and HOG gradients being unchanged by many of our corruptions.')]
story += [img('curves.png', 13), P('Fig. 6 - Training curves (MLP on HOG features, mean over 4 LOEO folds; solid = train, dashed = held-out environment). Train loss goes to ~0 in all variants while held-out accuracy plateaus (61% A1 data, 55% +synthetic, 58% +synthetic+enhance at epoch 60): overfitting to few real hands, and no gain from synthetic data on real shift. SVM/LogReg have no epochs, hence an MLP is used for curves.', CAP)]
# ablation
rows = [['Ablation (2x2)', 'synthetic', 'enhance', 'Clean CV acc', 'Mean 9-stress acc', 'LOEO acc']]
for n, a, b in [(NAMES[1], 'no', 'no'), (NAMES[2], 'yes', 'no'), (NAMES[3], 'no', 'yes'), (NAMES[4], 'yes', 'yes')]:
    rows.append([SH[n].split(':')[0] if False else ['', 'A1 HOG+SVM', 'A2 +syn', 'A2 +enh', 'A2 full'][NAMES.index(n)], a, b, pc(S[n]['accuracy'][0]), f'{mean_stress[n]*100:.1f}', f'{LO[n]["overall"]*100:.1f}'])
story += [P('<b>Table 4 - Ablation: with / without synthetic clips and with / without enhancement.</b> Synthetic data gives the larger stress gain (+' + f'{(mean_stress[NAMES[2]]-mean_stress[NAMES[1]])*100:.1f}' + ' pp), enhancement alone gives little (+' + f'{(mean_stress[NAMES[3]]-mean_stress[NAMES[1]])*100:.1f}' + ' pp) but the two combine best on clean and stress data; on LOEO every A2 variant is below A1.', SM),
          tab(rows, [3.4, 2, 2, 2.6, 3.3, 2.4], hl=4)]
story += [P(f'<b>Latency (CPU-only, 2 cores, single image).</b> A1 HOG+SVM: {lat["A1 HOG+SVM total"]["mean_ms"]:.1f} ms (p95 {lat["A1 HOG+SVM total"]["p95_ms"]:.1f} ms, ~{lat["A1 HOG+SVM total"]["fps"]:.0f} FPS); A2 with enhancement: {lat["A2 enhance+HOG+SVM total"]["mean_ms"]:.1f} ms (p95 {lat["A2 enhance+HOG+SVM total"]["p95_ms"]:.1f} ms, ~{lat["A2 enhance+HOG+SVM total"]["fps"]:.0f} FPS) including 1.3 ms resize. '
            'The occlusion heatmap (49 forward passes) costs ~230 ms. The heavy T2V generation is offline and not part of inference latency.'),
          P('<b>Confusions.</b> Fig. 7 shows the pooled confusion matrices; as in A1 the weak class is <i>ok</i> (fewest photos, small thumb-index circle).'), img('confusion.png', 11), P('Fig. 7 - Pooled confusion matrices (row-normalised), A1 HOG+SVM vs A2 full.', CAP)]

# ---------------------------------------------------------------- 4 explainability
story += [PageBreak(), P('4. Explainability', H1),
          P('Grad-CAM needs a CNN; for HOG+SVM we use two model-agnostic explanations, both shown in the app: <b>occlusion sensitivity</b> (a 32x32 grey patch slides over the 128x128 image with stride 16; heat = drop of the predicted-class SVM score) and a <b>nearest-neighbour gallery</b> (the 4 closest training photos in HOG space). '
            'Fig. 8 uses photos the model has never seen (CV fold 0). Correct predictions focus on the palm/finger region (e.g. the fingers for <i>stop</i>), while the two errors show the typical failure: the heat is spread on background texture or on the folded fingers, and neighbours come from the confusable class (<i>thumb</i> &rarr; <i>ok</i>; dark photo <i>stop</i> &rarr; <i>thumb</i>). '
            'This confirms that HOG still partly uses background/lighting cues, matching the weak LOEO results.'),
          img('explain.png', 16.5), P('Fig. 8 - Left to right: input, prediction + occlusion heatmap, four nearest training photos, enhanced input. Last two rows are errors.', CAP)]

# ---------------------------------------------------------------- 5 app
story += [P('5. Web application', H1),
          P('Flask app (<font face="Courier">python app/app.py</font>): upload or webcam capture &rarr; label with relative SVM scores &rarr; enhanced input, occlusion heatmap, nearest training photos; shows <b>per-request and average latency</b> and the <b>environment</b> (browser user-agent and screen, server OS/CPU/Python, CPU-only). '
            'A <b>stress-test generator</b> panel (9 conditions, severity slider) transforms the input before prediction; the result is stamped SYNTHETIC and tagged in the UI. A consent / synthetic-media notice is permanently visible; images are processed in memory and not stored. '
            'Heavy generators (T2V) run offline in Colab and enter only as clips/frames. Demo video: <font face="Courier">demo/demo.mp4</font> (silent screen recording with captions, ~3.5 min; add a voice-over when presenting).'),
          img('app_screenshot.png', 8.2), P('Fig. 9 - App screenshot (low-light SYNTHETIC input, enhancement, heatmap, neighbours, latency, environment).', CAP)]

# ---------------------------------------------------------------- 6 ethics
story += [P('6. Ethics, limitations, reproducibility', H1),
          P('<b>Ethics and consent.</b> All photos show hands of the author/volunteers; no faces were collected; the T2V prompt grid asks for bare hands only. Generated media must be disclosed: every synthetic file/clip/UI image carries "SYNTHETIC", and the manifest records its source. Hand gestures are culturally dependent (the "OK" sign is offensive in some regions) and skin-tone or hand-size bias is possible: the data contain only a few people, and the pixel baseline is clearly brightness/skin dependent. '
            'Misuse risks: photorealistic synthetic hands/people could be passed off as real footage; keep watermarks, do not use the system for surveillance or access control, and re-evaluate on new users before any real use.'),
          P('<b>Limitations.</b> (a) 68 real photos (below the 100-200/class target): CIs are wide (about +-9 pp) and most A2-vs-A1 differences on clean/LOEO data are not significant. (b) Stand-ins, not the SOTA models; no real T2V frames were evaluated, so claims about Step-Video-T2V/ArtAug are only that the pipeline is ready for them. '
            '(c) Table 2 is optimistic because train and test corruptions share one generator. (d) Environments in LOEO are pseudo-labels from background clustering, unequal in size and class mix. (e) No deep CNN. (f) The final app model has seen all 68 photos, so demoing on them is in-sample; use new photos for honest demos.'),
          P('<b>Next steps.</b> Run the Colab notebook with Step-Video-T2V Turbo and compare real T2V frames vs. the stand-in (same code path); collect >=100 photos/class with new people and backgrounds; MobileNetV2/ViT features with the same folds; lower the synthetic share (e.g. 20-30%) as an ablation.'),
          P('<b>Reproducibility.</b> Python 3.13, numpy, scikit-learn, scikit-image, OpenCV, Flask; no GPU, no downloads, fixed seeds. Commands: <font face="Courier">python src/make_synthetic.py; python src/experiments.py; python src/train_final.py; python src/make_figures.py; python app/app.py; python src/build_report.py</font> (README in the repo). Experiments take ~9 minutes on 2 CPU cores; results are stored in <font face="Courier">results/results.json</font>.')]

def foot(c, d):
    c.saveState(); c.setFont('Helvetica', 7.5); c.setFillColor(colors.grey); c.drawString(1.8 * cm, 1 * cm, 'Hand Gesture Recognition - Assignment 2'); c.drawRightString(19.2 * cm, 1 * cm, f'{d.page}'); c.restoreState()
os.makedirs(f'{ROOT}/report', exist_ok=True)
SimpleDocTemplate(f'{ROOT}/report/Assignment2_Report.pdf', pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.6 * cm, title='Hand Gesture Recognition - Assignment 2').build(story, onFirstPage=foot, onLaterPages=foot)
print('report built')
