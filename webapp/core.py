"""Model loading, analysis, history and PDF report for the KOA-ORDNet web app."""
from __future__ import annotations

import io
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import sys

import cv2
import numpy as np
import streamlit as st

# torch and the model code are imported lazily inside the functions that need them, so the
# Home/About pages render immediately instead of waiting for PyTorch to import.

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "koa_ordnet.pt"
WEIGHTS_URL = "https://github.com/abhay93682-lab/KOA-ORDNet/releases/latest/download/koa_ordnet.pt"
RESULTS = ROOT / "results" / "results.json"
SAMPLES = ROOT / "samples"

KL = {
    0: {"name": "Normal", "color": "#22c55e",
        "desc": "No radiographic features of osteoarthritis."},
    1: {"name": "Doubtful", "color": "#38bdf8",
        "desc": "Doubtful joint space narrowing and possible osteophytic lipping."},
    2: {"name": "Mild", "color": "#facc15",
        "desc": "Definite osteophytes and possible joint space narrowing."},
    3: {"name": "Moderate", "color": "#fb923c",
        "desc": "Multiple osteophytes, definite joint space narrowing and some sclerosis."},
    4: {"name": "Severe", "color": "#ef4444",
        "desc": "Large osteophytes, marked joint space narrowing, severe sclerosis and bone deformity."},
}


@st.cache_data
def test_metrics() -> dict:
    """Real test-set metrics from evaluate.py (results/results.json)."""
    try:
        return json.loads(RESULTS.read_text())
    except Exception:
        return {}


def ensure_weights() -> bool:
    import torch
    if WEIGHTS.exists():
        return True
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.hub.download_url_to_file(WEIGHTS_URL, str(WEIGHTS), progress=False)
        return True
    except Exception:
        WEIGHTS.unlink(missing_ok=True)
        return False


@st.cache_resource(show_spinner="Loading KOA-ORDNet model…")
def load_model():
    import torch
    from koa_ordnet.model import KOAOrdNet
    if not ensure_weights():
        return None, None
    c = torch.load(WEIGHTS, map_location="cpu")
    m = KOAOrdNet(head=c["head"], pretrained=False)
    m.load_state_dict(c["state_dict"])
    return m.eval(), c["head"]


def device_label() -> str:
    torch = sys.modules.get("torch")
    return "GPU" if torch is not None and torch.cuda.is_available() else "CPU"


@dataclass
class Analysis:
    id: str
    filename: str
    timestamp: str
    grade: int
    confidence: float
    probs: list
    iou: float
    seconds: float
    original: np.ndarray            # grayscale uint8, original size
    preprocessed: np.ndarray        # CLAHE 224x224 uint8
    heatmap: np.ndarray             # RGB uint8
    overlay: np.ndarray             # RGB uint8
    expert_grade: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def severity(self) -> str:
        return KL[self.grade]["name"]


def analyze(data: bytes, filename: str, expert_grade: int | None = None) -> Analysis:
    import torch
    from koa_ordnet.data import clahe_resized, preprocess
    from koa_ordnet.gradcam import GradCAM, cam_iou, overlay
    from koa_ordnet.model import logits_to_probs
    model, head = load_model()
    t0 = time.time()
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not read this file as an image.")
    pre = clahe_resized(gray)
    x = torch.from_numpy(preprocess(gray).transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        probs = logits_to_probs(model(x), head)[0].numpy()
    cam_fn = GradCAM(model, model.target_layer)
    cam, grade = cam_fn(x)
    cam_fn.remove()
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Analysis(
        id=uuid.uuid4().hex[:8].upper(),
        filename=filename,
        timestamp=datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST"),
        grade=int(grade),
        confidence=float(probs[grade]),
        probs=[float(p) for p in probs],
        iou=cam_iou(cam),
        seconds=time.time() - t0,
        original=gray,
        preprocessed=pre,
        heatmap=heat,
        overlay=overlay(cv2.resize(gray, (224, 224)), cam),
        expert_grade=expert_grade,
    )


# ---------------- history (per browser session) ----------------
def history() -> list[Analysis]:
    return st.session_state.setdefault("history", [])


def add_to_history(a: Analysis) -> None:
    history().insert(0, a)
    st.session_state["current"] = a


# ---------------- PDF report ----------------
def _png(img: np.ndarray) -> io.BytesIO:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return io.BytesIO(buf.tobytes())


def pdf_report(a: Analysis) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(12, 8)
    pdf.cell(0, 8, "KOA-ORDNet  |  Knee X-ray Analysis Report")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(12, 17)
    pdf.cell(0, 5, f"Report ID {a.id}   |   {a.timestamp}   |   File: {a.filename[:60]}")

    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(12, 36)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, f"Predicted Kellgren-Lawrence grade: KL {a.grade} ({a.severity})", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Model confidence: {a.confidence*100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    if a.expert_grade is not None:
        pdf.cell(0, 6, f"Expert label (test-set sample): KL {a.expert_grade} ({KL[a.expert_grade]['name']})",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 6, f"Radiographic description: {KL[a.grade]['desc']}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Grade probabilities", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for k, p in enumerate(a.probs):
        pdf.cell(40, 6, f"KL {k}  {KL[k]['name']}")
        pdf.set_fill_color(37, 99, 235 if k == a.grade else 180)
        pdf.cell(max(0.5, 100 * p), 4.5, "", fill=True)
        pdf.cell(0, 6, f"  {p*100:.1f}%", new_x="LMARGIN", new_y="NEXT")

    y = pdf.get_y() + 4
    pdf.set_font("Helvetica", "B", 10)
    for i, (title, img) in enumerate([("Original radiograph", a.original),
                                      ("Preprocessed (CLAHE, 224x224)", a.preprocessed),
                                      ("Grad-CAM overlay", a.overlay)]):
        x = 12 + i * 63
        pdf.set_xy(x, y)
        pdf.cell(60, 6, title)
        pdf.image(_png(cv2.resize(img, (224, 224))), x=x, y=y + 7, w=58, h=58)
    pdf.set_xy(12, y + 70)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, f"Grad-CAM IoU with the tibiofemoral joint region: {a.iou:.2f}. "
                         "Model: ResNet-50 with a cumulative-link ordinal head (4 threshold outputs), trained with a "
                         "class-weighted focal-ordinal loss on the Chen (2018) dataset. "
                         "Test-set performance of this model: accuracy 60.81%, QWK 0.8090 (see project results).")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(185, 28, 28)
    pdf.multi_cell(0, 5, "Disclaimer: research prototype for academic demonstration only. "
                         "Not a medical device and not a substitute for a radiologist's diagnosis.")
    return bytes(pdf.output())
