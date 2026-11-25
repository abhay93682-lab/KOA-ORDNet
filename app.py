"""KOA-ORDNet web app (Streamlit).

    streamlit run app.py

Pages: Home, Analyze X-Ray, Results dashboard (grade, severity, confidence, KL scale, CLAHE view,
Grad-CAM heatmap/overlay, PDF report), History and About. Trained weights are read from
models/koa_ordnet.pt, or downloaded once from the latest GitHub release.
"""
import streamlit as st

st.set_page_config(page_title="KOA-ORDNet · Knee OA Severity AI", page_icon="🦵", layout="wide")

from webapp.pages import PAGES  # noqa: E402

st.navigation(list(PAGES.values()), position="top").run()
