import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from koa_ordnet.gradcam import GradCAM, cam_iou  # noqa: E402
from koa_ordnet.losses import FocalOrdinalLoss, inverse_freq_weights, ordinal_targets  # noqa: E402
from koa_ordnet.metrics import compute_metrics  # noqa: E402
from koa_ordnet.model import KOAOrdNet, logits_to_grade, logits_to_probs  # noqa: E402


def test_ordinal_targets():
    t = ordinal_targets(torch.tensor([0, 2, 4]))
    assert t.tolist() == [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]


def test_logits_to_grade_and_probs():
    logits = torch.tensor([[5., 5., -5., -5.]])
    assert logits_to_grade(logits, "ordinal").item() == 2
    p = logits_to_probs(logits, "ordinal")
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-4)
    assert p.argmax().item() == 2


def test_focal_ordinal_penalises_far_errors_more():
    loss = FocalOrdinalLoss(inverse_freq_weights(np.array([10, 10, 10, 10, 10])))
    y = torch.tensor([0])
    near = loss(torch.tensor([[3., -3., -3., -3.]]), y)   # predicts grade 1
    far = loss(torch.tensor([[3., 3., 3., 3.]]), y)       # predicts grade 4
    assert far > near


def test_inverse_freq_weights_favour_rare():
    w = inverse_freq_weights(np.array([100, 50, 30, 10, 5]))
    assert w[4] > w[0]


def test_metrics():
    m = compute_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert m["accuracy"] == 1.0 and m["qwk"] == 1.0 and m["mae"] == 0.0 and m["grade34_recall"] == 1.0


def test_model_forward_and_gradcam():
    model = KOAOrdNet(head="ordinal", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    assert model(x).shape == (1, 4)
    cam_fn = GradCAM(model, model.target_layer)
    cam, grade = cam_fn(x)
    cam_fn.remove()
    assert cam.shape == (224, 224) and 0 <= grade <= 4
    assert 0.0 <= cam_iou(cam) <= 1.0


def test_end_to_end_smoke(tmp_path):
    data = tmp_path / "data"
    subprocess.run([sys.executable, "scripts/make_synthetic.py", str(data), "--per-class", "2"], cwd=ROOT, check=True)
    runs = tmp_path / "runs"
    subprocess.run([sys.executable, "train.py", "--data", str(data), "--out", str(runs), "--loss", "koa",
                    "--epochs", "1", "--batch-size", "4", "--workers", "0", "--no-pretrained"], cwd=ROOT, check=True)
    assert (runs / "koa" / "best.pt").exists()
