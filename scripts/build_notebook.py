"""Generates notebooks/koa_ordnet_kaggle.ipynb (run: python scripts/build_notebook.py)."""
import json
from pathlib import Path

cells = [
    ("markdown", """# KOA-ORDNet: training on Kaggle

Before running:
1. **Settings → Accelerator → GPU T4 x2** (or P100)
2. **Settings → Internet → On**
3. **Add Input** → search *Knee Osteoarthritis Dataset with Severity Grading* → Add

Then **Run All**. It trains the three models from the paper (CE baseline, ordinal baseline, KOA-ORDNet), evaluates them, and saves the trained model and results under `/kaggle/working`. Total time: about 3–5 hours on a T4."""),
    ("code", """import torch, subprocess, sys
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - turn on GPU in Settings!')
assert torch.cuda.is_available(), 'Turn on GPU: Settings -> Accelerator -> GPU'"""),
    ("code", """!rm -rf /kaggle/working/KOA-ORDNet
!git clone -q https://github.com/abhay93682-lab/KOA-ORDNet /kaggle/working/KOA-ORDNet
%cd /kaggle/working/KOA-ORDNet
!pip install -q opencv-python-headless scikit-learn"""),
    ("code", """# Find the dataset folder automatically: the folder that has train/0 ... train/4
from pathlib import Path
DATA = None
for p in sorted(Path('/kaggle/input').rglob('train')):
    if p.is_dir() and all((p / str(g)).is_dir() for g in range(5)) and (p.parent / 'test').is_dir():
        DATA = str(p.parent); break
assert DATA, 'Dataset not found: Add Input -> "Knee Osteoarthritis Dataset with Severity Grading"'
print('DATA =', DATA)
for s in ('train', 'val', 'test'):
    print(s, [len(list((Path(DATA) / s / str(g)).glob('*'))) for g in range(5)])"""),
    ("markdown", "## Train the three models (paper Table II rows)"),
    ("code", "!python train.py --data \"$DATA\" --loss ce --workers 4"),
    ("code", "!python train.py --data \"$DATA\" --loss ordinal --workers 4"),
    ("code", "!python train.py --data \"$DATA\" --loss koa --workers 4"),
    ("markdown", "## Evaluate (Table II, Grad-CAM IoU). Add an external dataset with `--external <path>` for Table III."),
    ("code", """!python evaluate.py --data "$DATA" --workers 4
print(open('results/RESULTS.md').read())"""),
    ("code", """# Save everything you need to /kaggle/working (download from the Output tab)
!mkdir -p /kaggle/working/output
!cp runs/koa_ordnet/koa/best.pt /kaggle/working/output/koa_ordnet.pt
!cp -r results /kaggle/working/output/
!for k in ce ordinal koa; do cp runs/koa_ordnet/$k/history.json /kaggle/working/output/history_$k.json; done
!ls -la /kaggle/working/output /kaggle/working/output/results"""),
    ("markdown", """## Next
Download `output/koa_ordnet.pt` and `output/results/` from the **Output** tab.
- Put `koa_ordnet.pt` in the repo's `models/` folder to power the Streamlit demo.
- Copy `results/RESULTS.md` into the repo and use these numbers in the paper's Table II."""),
]

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"},
                 "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "koa_ordnet_kaggle.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)

# ---------------- Google Colab version ----------------
colab_cells = [
    ("markdown", """# KOA-ORDNet: training on Google Colab

1. **Runtime → Change runtime type → T4 GPU → Save**
2. **Runtime → Run all**. Allow Google Drive access when asked: checkpoints and results are saved to
   `MyDrive/KOA-ORDNet`, so nothing is lost if Colab disconnects.

Each model takes about 1–2 hours on a T4. Free Colab sessions can stop after a few hours: if that happens,
run all again and finished models are skipped automatically."""),
    ("code", """import torch
assert torch.cuda.is_available(), 'Turn on GPU: Runtime -> Change runtime type -> T4 GPU'
print('GPU:', torch.cuda.get_device_name(0))"""),
    ("code", """from google.colab import drive
drive.mount('/content/drive')
SAVE = '/content/drive/MyDrive/KOA-ORDNet'
import os; os.makedirs(SAVE, exist_ok=True); print('Saving to', SAVE)"""),
    ("code", """!rm -rf /content/KOA-ORDNet
!git clone -q https://github.com/abhay93682-lab/KOA-ORDNet /content/KOA-ORDNet
%cd /content/KOA-ORDNet
!pip install -q opencv-python-headless scikit-learn kagglehub"""),
    ("code", """# Download the Chen (2018) dataset from Kaggle (public, CC BY 4.0)
import kagglehub
from pathlib import Path
root = kagglehub.dataset_download('shashwatwork/knee-osteoarthritis-dataset-with-severity')
DATA = None
for p in sorted(Path(root).rglob('train')):
    if p.is_dir() and all((p / str(g)).is_dir() for g in range(5)) and (p.parent / 'test').is_dir():
        DATA = str(p.parent); break
assert DATA, f'Dataset layout not found under {root}'
print('DATA =', DATA)
for s in ('train', 'val', 'test'):
    print(s, [len(list((Path(DATA) / s / str(g)).glob('*'))) for g in range(5)])"""),
    ("markdown", "## Train the three models (skips any model already finished in Drive)"),
    ("code", """import os, subprocess
RUNS = f'{SAVE}/runs'
for loss in ['ce', 'ordinal', 'koa']:
    if os.path.exists(f'{RUNS}/{loss}/history.json'):
        print(f'== {loss}: already trained, skipping'); continue
    print(f'== training {loss}')
    !python train.py --data "$DATA" --out "$RUNS" --loss $loss --workers 2"""),
    ("markdown", """## Evaluate: Table II, Table III and Grad-CAM IoU, from the trained checkpoints
External test set (RG3): *Digital Knee X-ray Images* (Gornale & Patravali, 2020, Mendeley Data,
doi:10.17632/t9ndx37v5h.1), KL labels from Medical Expert I. It is never used for training or tuning."""),
    ("code", """ext_root = kagglehub.dataset_download('tommyngx/digital-knee-xray')
EXT = next(str(p) for p in Path(ext_root).rglob('MedicalExpert-I') if p.is_dir())
print('EXT =', EXT)
!python evaluate.py --data "$DATA" --runs "$RUNS" --external "$EXT" --workers 2
!cp -r results "$SAVE"/
!cp "$RUNS"/koa/best.pt "$SAVE"/koa_ordnet.pt
print(open('results/RESULTS.md').read())
print('Saved in Google Drive:', SAVE)"""),
]
colab = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in colab_cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU",
                 "colab": {"provenance": [], "gpuType": "T4"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "koa_ordnet_colab.ipynb"
out.write_text(json.dumps(colab, indent=1))
print("wrote", out)
