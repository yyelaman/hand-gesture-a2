# Hand Gesture Recognition - Assignment 2 (A1 topic upgraded with synthetic-data + enhancement pipeline)

Classes: peace, thumb, stop, ok. Data: 68 real photos (A1) + procedurally generated SYNTHETIC stress images/clips.

> **Honesty note.** Step-Video-T2V and ArtAug are GPU-heavy models that were not run here. Their pipeline roles are filled by
> classical stand-ins (`stress_condition`, `make_clip`, `enhance` in `src/common.py`). `notebooks/step_video_t2v_colab.ipynb`
> is an unexecuted template to produce real clips offline; `src/ingest_clips.py` ingests them.

## Run
```
pip install -r requirements.txt
python src/make_synthetic.py        # data/synthetic (images, mp4 clips, manifest) - all watermarked SYNTHETIC
python src/experiments.py           # ~9 min on 2 CPU -> results/results.json
python src/train_final.py           # results/model.joblib
python src/make_figures.py          # figures/
python app/app.py                   # http://127.0.0.1:5000  (upload / camera, heatmap, NN, latency, stress generator)
python src/build_report.py          # report/Assignment2_Report.pdf
```
Layout: `src/` code, `app/` Flask demo, `data/real` A1 photos, `data/synthetic`, `results/`, `figures/`, `report/`, `demo/demo.mp4`, `notebooks/`.
Seeds are fixed; no pretrained weights or internet needed. Model/protocol details are in the report.
