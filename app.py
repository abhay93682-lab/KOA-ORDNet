"""Streamlit demo: upload a knee X-ray, get KL grade + Grad-CAM heatmap + IoU score.

    streamlit run app.py

Looks for weights at models/koa_ordnet.pt; if missing, downloads them from the
latest GitHub release. You can also upload a checkpoint in the sidebar.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from koa_ordnet.data import preprocess
from koa_ordnet.gradcam import GradCAM, cam_iou, overlay
from koa_ordnet.model import KOAOrdNet, logits_to_probs

GRADE_TEXT = {0: "Grade 0 - None", 1: "Grade 1 - Doubtful", 2: "Grade 2 - Minimal",
              3: "Grade 3 - Moderate", 4: "Grade 4 - Severe"}
DEFAULT_WEIGHTS = Path("models/koa_ordnet.pt")
# Trained weights published as a GitHub release asset; downloaded once on first run.
WEIGHTS_URL = "https://github.com/abhay93682-lab/KOA-ORDNet/releases/latest/download/koa_ordnet.pt"


def ensure_weights() -> None:
    if DEFAULT_WEIGHTS.exists():
        return
    DEFAULT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    try:
        with st.spinner("Downloading trained model (~95 MB, first run only)..."):
            torch.hub.download_url_to_file(WEIGHTS_URL, str(DEFAULT_WEIGHTS), progress=False)
    except Exception:
        DEFAULT_WEIGHTS.unlink(missing_ok=True)

st.set_page_config(page_title="KOA-ORDNet", page_icon="🦵", layout="centered")
st.title("KOA-ORDNet")
st.caption("Ordinal, imbalance-robust and explainable KL grading of knee X-rays. Research demo, not a medical device.")


@st.cache_resource
def load_model(data: bytes | None):
    src = io.BytesIO(data) if data else (DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else None)
    if src is None:
        return None, None
    c = torch.load(src, map_location="cpu")
    m = KOAOrdNet(head=c["head"], pretrained=False)
    m.load_state_dict(c["state_dict"])
    return m.eval(), c["head"]


ensure_weights()
up_ckpt = st.sidebar.file_uploader("Model checkpoint (.pt)", type=["pt"])
model, head = load_model(up_ckpt.getvalue() if up_ckpt else None)
if model is None:
    st.warning("No trained weights found. Train with `train.py`, then copy `runs/koa_ordnet/koa/best.pt` "
               "to `models/koa_ordnet.pt` or upload it in the sidebar.")
    st.stop()

img_file = st.file_uploader("Knee X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])
if img_file:
    gray = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(preprocess(gray).transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        probs = logits_to_probs(model(x), head)[0].numpy()
    cam_fn = GradCAM(model, model.target_layer)
    cam, grade = cam_fn(x)
    cam_fn.remove()

    st.subheader(GRADE_TEXT[grade])
    c1, c2 = st.columns(2)
    c1.image(cv2.resize(gray, (224, 224)), caption="Input", use_container_width=True)
    c2.image(overlay(cv2.resize(gray, (224, 224)), cam), caption="Grad-CAM (box = joint region)", use_container_width=True)
    st.metric("Grad-CAM IoU with joint region", f"{cam_iou(cam):.2f}")
    st.bar_chart({f"KL {k}": float(p) for k, p in enumerate(probs)})
