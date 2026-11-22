# KOA-ORDNet

**An ordinal-aware, class-imbalance-robust and explainable deep learning framework for automated knee osteoarthritis (KOA) severity grading from radiographs.**

KOA-ORDNet grades knee X-rays on the Kellgren–Lawrence (KL) scale (0–4). It addresses four gaps in prior KL-grading work:

| Gap | KOA-ORDNet component | Code |
|---|---|---|
| RG1: KL grades treated as unordered classes | Cumulative-link ordinal head (4 units, unit *k* = P(y > k)) | [`koa_ordnet/model.py`](koa_ordnet/model.py) |
| RG2: Severe grades (3–4) are rare | Inverse-frequency class weights + focal ordinal BCE | [`koa_ordnet/losses.py`](koa_ordnet/losses.py) |
| RG3: No external validation | Frozen model tested on a second, independent dataset | [`evaluate.py --external`](evaluate.py) |
| RG4: Grad-CAM only shown, never measured | Grad-CAM + IoU against the tibiofemoral joint region | [`koa_ordnet/gradcam.py`](koa_ordnet/gradcam.py) |

## Method

- **Backbone:** ResNet-50, ImageNet-pretrained, final FC removed.
- **Preprocessing:** resize to 224×224, CLAHE, 3-channel replication, ImageNet normalisation.
- **Augmentation:** rotation ±10°, small translation and zoom. Horizontal flip is excluded on purpose, because it would change the left/right knee orientation.
- **Loss (proposed):** per-threshold BCE with a focal term (γ = 2), weighted by inverse class frequency.
- **Training:** Adam (lr 1e-4, wd 1e-5), batch 32, ReduceLROnPlateau (×0.5, patience 3), up to 60 epochs, early stopping on validation macro-F1 (patience 8).
- **Metrics:** accuracy, macro-F1, per-grade recall, Grade 3–4 recall, Quadratic Weighted Kappa, MAE, Grad-CAM IoU.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                      # unit tests + synthetic smoke run (CPU, ~1 min)
```

### 1. Get the data

Download the *Knee Osteoarthritis Severity Grading* dataset (Chen, 2018, Mendeley Data, doi:10.17632/56rmx5bjcr.1; also on Kaggle). Place it so that you have:

```
data/kneeKL224/train/{0,1,2,3,4}/*.png
data/kneeKL224/val/{0,1,2,3,4}/*.png
data/kneeKL224/test/{0,1,2,3,4}/*.png
```

For the cross-dataset test, put a second, independent KOA dataset in `data/external/{0,1,2,3,4}/`. It must never be used for training.

### 2. Train (GPU recommended: Kaggle or Colab T4/P100)

```bash
python train.py --loss ce        # baseline: ResNet-50 + cross-entropy
python train.py --loss ordinal   # baseline: ordinal loss (Chen et al. style)
python train.py --loss koa       # proposed: KOA-ORDNet
```

Each run is roughly 1–2 hours on a T4. CPU training is not practical for the full dataset.

### 3. Evaluate

```bash
python evaluate.py --external data/external
```

This writes `results/results.json` and `results/RESULTS.md`, with Table II (in-distribution) and Table III (cross-dataset) filled in **from the trained checkpoints**.

### 4. Demo

```bash
mkdir models && cp runs/koa_ordnet/koa/best.pt models/koa_ordnet.pt
streamlit run app.py
```

Upload a knee X-ray to get the KL grade, per-grade probabilities, the Grad-CAM heatmap and its IoU score.

## Results

Results will be published in [`results/RESULTS.md`](results/) once the full training runs are complete. No numbers are reported here before then.

## Project layout

```
koa_ordnet/   config, data + CLAHE, model, losses, metrics, Grad-CAM
train.py      training with early stopping on val macro-F1
evaluate.py   Table II / III + Grad-CAM IoU, from checkpoints
app.py        Streamlit demo
tests/        unit tests + end-to-end synthetic smoke test
.github/      daily CI health check
```

## Disclaimer

Research code. Not a medical device and not for clinical diagnosis.
