"""The five pages of the KOA-ORDNet web app."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from .core import KL, SAMPLES, add_to_history, analyze, history, load_model, pdf_report, test_metrics
from .ui import badge, card, kl_cards, setup


def _zoom(img: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h / factor), int(w / factor)
    y, x = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- Home
def home() -> None:
    setup()
    m = test_metrics().get("koa", {}).get("in_distribution", {})
    st.markdown(
        """<div class="k-hero"><span class="k-chip">◎ AI-Powered Medical Radiograph Intelligence</span>
           <h1>Instant &amp; Automated<br><span class="k-grad">Knee Osteoarthritis Severity Assessment</span></h1>
           <p class="k-lead">Upload a knee X-ray radiograph to get Kellgren–Lawrence (KL 0–4) severity grading,
           Grad-CAM heatmap analysis and a downloadable PDF report.</p></div>""", unsafe_allow_html=True)
    _, b1, b2, _ = st.columns([1.3, 1, 1, 1.3])
    if b1.button("Analyze X-Ray Now  →", type="primary", width="stretch"):
        st.switch_page(PAGES["analyze"])
    if b2.button("View Analysis History", width="stretch"):
        st.switch_page(PAGES["history"])
    st.write("")
    c = st.columns(4)
    stats = [
        (f'{m.get("qwk", 0):.4f}', "#38bdf8", "QWK Agreement", f'Quadratic Weighted Kappa on the held-out test set (n = {m.get("n", 0)})'),
        ("ResNet-50", "#34d399", "Ordinal Backbone", "4 cumulative binary threshold outputs P(y > k)"),
        ("Grad-CAM", "#fbbf24", "Explainable AI", "Layer4 activation heatmaps, scored by IoU"),
        ("Focal Loss", "#fb923c", "Class-Imbalance Robust", "Inverse-frequency weighted focal-ordinal loss"),
    ]
    for col, (v, color, t, d) in zip(c, stats):
        with col:
            card(f'<div class="k-stat" style="color:{color}">{v}</div><div class="k-stat-t">{t}</div>'
                 f'<div class="k-stat-d">{d}</div>')
    st.write(""); st.write("")
    st.markdown("### How it works")
    steps = [("Pre-processing", "Resize to 224×224 and CLAHE contrast enhancement of the joint space."),
             ("Ordinal ResNet-50", "Four threshold units estimate P(KL > k); the grade is the number above 0.5."),
             ("Imbalance-aware training", "Class-weighted focal-ordinal loss for rare severe grades."),
             ("Explainability", "Grad-CAM heatmap plus an IoU score against the joint region.")]
    c = st.columns(4)
    for i, (col, (t, d)) in enumerate(zip(c, steps), 1):
        with col:
            card(f'<div class="k-step"><div class="k-num">{i}</div><div><div class="k-stat-t">{t}</div>'
                 f'<div class="k-stat-d">{d}</div></div></div>')


# ------------------------------------------------------------------------ Analyze
def analyze_page() -> None:
    setup()
    st.markdown('<div class="k-eyebrow">Step 1</div>', unsafe_allow_html=True)
    st.markdown("## Analyze a Knee X-Ray")
    st.caption("Use an anteroposterior (front-view) knee radiograph, ideally cropped around the joint. PNG or JPG.")
    model, _ = load_model()
    if model is None:
        st.error("The trained model could not be downloaded. Please try again in a minute.")
        return
    left, right = st.columns([1.4, 1])
    with left:
        up = st.file_uploader("Upload radiograph", type=["png", "jpg", "jpeg"])
    samples = sorted(SAMPLES.glob("*.png"))
    with right:
        names = ["—"] + [f'KL {p.stem[2]} sample ({KL[int(p.stem[2])]["name"]})' for p in samples]
        pick = st.selectbox("…or try a test-set sample", names)
        st.caption("Samples are the first test image of each grade (Chen 2018, CC BY 4.0).")
    data, fname, expert = None, None, None
    if up is not None:
        data, fname = up.getvalue(), up.name
    elif pick != "—":
        p = samples[names.index(pick) - 1]
        data, fname, expert = p.read_bytes(), p.name, int(p.stem[2])
    if data:
        p1, p2 = st.columns([1, 2.4], vertical_alignment="center")
        p1.image(data, caption=fname, width="stretch")
        p2.markdown(f'<div class="k-stat-t">Ready to analyze</div><p class="k-note">{fname}</p>', unsafe_allow_html=True)
        if p2.button("Run AI Analysis  →", type="primary"):
            with st.spinner("Pre-processing, grading and computing Grad-CAM…"):
                try:
                    add_to_history(analyze(data, fname, expert))
                except ValueError as e:
                    st.error(str(e)); return
            st.switch_page(PAGES["results"])


# ------------------------------------------------------------------------ Results
def results() -> None:
    setup()
    a = st.session_state.get("current")
    if a is None:
        st.info("No analysis yet. Upload an X-ray on the Analyze X-Ray page.")
        if st.button("Analyze X-Ray", type="primary"):
            st.switch_page(PAGES["analyze"])
        return
    h1, h2, h3 = st.columns([2.6, 1, 1], vertical_alignment="center")
    h1.markdown("## Complete AI Analysis Dashboard")
    if "pdf" not in a.extra:
        a.extra["pdf"] = pdf_report(a)
    h2.download_button("⬇  Download PDF Report", a.extra["pdf"], file_name=f"KOA-ORDNet_report_{a.id}.pdf",
                       mime="application/pdf", type="primary", width="stretch")
    if h3.button("Analyze Another Radiograph", width="stretch"):
        st.switch_page(PAGES["analyze"])

    c = KL[a.grade]["color"]
    expert = (f'<p class="k-note">Expert label for this test-set sample: <b>KL {a.expert_grade} '
              f'({KL[a.expert_grade]["name"]})</b></p>') if a.expert_grade is not None else ""
    st.markdown(
        f"""<div class="k-card" style="display:flex; gap:24px; flex-wrap:wrap; justify-content:space-between">
          <div style="flex:1 1 520px">
            <div class="k-eyebrow">AI Analysis Result</div>
            <div class="k-big">PREDICTED KL GRADE: <span style="color:{c}">{a.grade}</span></div>
            {badge(a.grade)}<span class="k-conf">Model Confidence: <span style="color:#38bdf8">{a.confidence*100:.2f}%</span></span>
            <p style="margin-top:16px; color:#b6c2d9">{KL[a.grade]["desc"]}</p>{expert}
            <p class="k-note">Report ID {a.id} · {a.timestamp} · {a.filename} · {a.seconds:.1f}s</p>
          </div>
          <div class="k-card" style="flex:0 1 300px; height:auto"><dl class="k-kv">
            <dt>Architecture</dt><dd>ResNet-50 Ordinal</dd>
            <dt>Preprocessing</dt><dd>CLAHE + 224×224</dd>
            <dt>Explainability</dt><dd style="color:#38bdf8">Layer4 Grad-CAM</dd></dl></div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("### Kellgren–Lawrence Grading Scale")
    kl_cards(a.grade)
    st.write("")
    st.markdown("### Grade probabilities")
    bars = "".join(
        f'<div style="display:flex; align-items:center; gap:14px; margin:9px 0">'
        f'<div style="width:150px; font-weight:{800 if k == a.grade else 500}">KL {k} · {KL[k]["name"]}</div>'
        f'<div style="flex:1; background:#0b1220; border:1px solid #1f2b45; border-radius:999px; height:14px; overflow:hidden">'
        f'<div style="width:{p*100:.1f}%; height:100%; background:{KL[k]["color"] if k == a.grade else "#2563eb"}; '
        f'opacity:{1 if k == a.grade else .55}"></div></div>'
        f'<div style="width:64px; text-align:right; font-variant-numeric:tabular-nums">{p*100:.1f}%</div></div>'
        for k, p in enumerate(a.probs))
    card(bars)

    st.write("")
    z = st.segmented_control("Zoom", ["1×", "1.5×", "2×", "3×"], default="1×")
    f = float((z or "1×").rstrip("×"))
    i1, i2 = st.columns(2)
    with i1:
        card('<div class="k-imgtitle">Original Radiograph</div>')
        st.image(_zoom(a.original, f), width="stretch")
    with i2:
        card('<div class="k-imgtitle">Preprocessed Image (CLAHE + 224×224)'
             '<span class="k-tag">Normalized for ResNet-50</span></div>')
        st.image(_zoom(a.preprocessed, f), width="stretch")

    st.write("")
    st.markdown('<div class="k-eyebrow y">Visual Explainability Engine</div>', unsafe_allow_html=True)
    st.markdown(f"## Explainable AI – Grad-CAM Explanation for Predicted KL Grade {a.grade}")
    st.markdown('<p class="k-note">Highlighted regions show the image areas that contributed most to the prediction '
                '(e.g. joint space narrowing, osteophytes, subchondral sclerosis). The white box marks the joint '
                'region used for the IoU score. This visualization supports model transparency and is not a '
                'standalone medical diagnosis.</p>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.markdown('<div class="k-imgtitle" style="text-align:center">Original Image</div>', unsafe_allow_html=True)
    g1.image(cv2.resize(a.original, (224, 224)), width="stretch")
    g2.markdown('<div class="k-imgtitle" style="text-align:center; color:#fbbf24">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
    g2.image(a.heatmap, width="stretch")
    g3.markdown('<div class="k-imgtitle" style="text-align:center; color:#38bdf8">Grad-CAM Overlay</div>', unsafe_allow_html=True)
    g3.image(a.overlay, width="stretch")
    st.metric("Grad-CAM IoU with joint region", f"{a.iou:.2f}",
              help="Overlap between the high-attention area (≥ 0.5) and the fixed tibiofemoral joint box.")


# ------------------------------------------------------------------------ History
def history_page() -> None:
    setup()
    st.markdown("## Analysis History")
    st.caption("Every X-ray analyzed in this browser session, newest first.")
    h = history()
    if not h:
        st.info("No analyses yet.")
        if st.button("Analyze X-Ray", type="primary"):
            st.switch_page(PAGES["analyze"])
        return
    df = pd.DataFrame([{"Report ID": a.id, "Time": a.timestamp, "File": a.filename, "KL grade": a.grade,
                        "Severity": a.severity, "Confidence (%)": round(a.confidence * 100, 2),
                        "Expert label": a.expert_grade if a.expert_grade is not None else "",
                        "Grad-CAM IoU": round(a.iou, 2)} for a in h])
    st.dataframe(df, hide_index=True, width="stretch")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.download_button("Download CSV", df.to_csv(index=False), "koa_ordnet_history.csv", "text/csv", width="stretch")
    if c2.button("Clear history", width="stretch"):
        st.session_state["history"] = []; st.session_state.pop("current", None); st.rerun()
    st.write("")
    ids = [f"{a.id} · {a.filename} · KL {a.grade}" for a in h]
    sel = st.selectbox("Open a past report", ids)
    if st.button("Open in Results dashboard", type="primary"):
        st.session_state["current"] = h[ids.index(sel)]
        st.switch_page(PAGES["results"])


# -------------------------------------------------------------------------- About
def about() -> None:
    setup()
    st.markdown("## About KOA-ORDNet")
    st.markdown(
        "**KOA-ORDNet: An Ordinal-Aware, Class-Imbalance-Robust and Explainable Deep Learning Framework for "
        "Automated Knee Osteoarthritis Severity Grading from Radiographs**  \n"
        "Abhay Kumar, Dr. Gesu Thakur, Dr. Ankush Joshi · College of Smart Computing, COER University, Roorkee  \n"
        "Presented at ICE2CPT-2026, COER University, 19–21 November 2026.")
    st.markdown("### Research gaps addressed")
    c = st.columns(4)
    gaps = [("RG1", "Ordinal structure ignored", "Cumulative-link ordinal head"),
            ("RG2", "Severe class imbalance", "Class-weighted focal-ordinal loss"),
            ("RG3", "No external validation", "Frozen model tested on an independent dataset"),
            ("RG4", "Explainability not measured", "Grad-CAM scored by IoU")]
    for col, (k, t, d) in zip(c, gaps):
        with col:
            card(f'<div class="k-stat" style="font-size:22px; color:#60a5fa">{k}</div>'
                 f'<div class="k-stat-t">{t}</div><div class="k-stat-d">→ {d}</div>')
    st.write("")
    st.markdown("### Measured results")
    r = test_metrics()
    names = {"ce": "ResNet-50 + Cross-Entropy", "ordinal": "ResNet-50 + Ordinal loss", "koa": "KOA-ORDNet (deployed)"}
    rows = []
    for k, n in names.items():
        if k in r:
            i, o = r[k]["in_distribution"], r[k].get("out_of_distribution", {})
            rows.append({"Model": n, "Accuracy": f'{i["accuracy"]*100:.2f}%', "Macro-F1": f'{i["macro_f1"]:.4f}',
                         "QWK": f'{i["qwk"]:.4f}', "MAE": f'{i["mae"]:.4f}',
                         "Grade 3–4 recall": f'{i["grade34_recall"]*100:.2f}%',
                         "External QWK": f'{o.get("qwk", float("nan")):.4f}'})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Test set: Chen (2018), n = 1656. External: Digital Knee X-ray Images (Gornale & Patravali, 2020), "
                   "n = 1650. Trained on Google Colab (T4).")
    st.markdown("### Links")
    st.markdown("- Code: https://github.com/abhay93682-lab/KOA-ORDNet  \n"
                "- Trained weights: https://github.com/abhay93682-lab/KOA-ORDNet/releases/latest  \n"
                "- Full results: https://github.com/abhay93682-lab/KOA-ORDNet/blob/main/results/RESULTS.md")
    st.warning("Research prototype for academic demonstration. Not a medical device and not a substitute for a "
               "radiologist's diagnosis.")


PAGES = {
    "home": st.Page(home, title="Home", icon=":material/monitor_heart:", url_path="home", default=True),
    "analyze": st.Page(analyze_page, title="Analyze X-Ray", icon=":material/memory:", url_path="analyze"),
    "results": st.Page(results, title="Results", icon=":material/task_alt:", url_path="results"),
    "history": st.Page(history_page, title="History", icon=":material/history:", url_path="history"),
    "about": st.Page(about, title="About", icon=":material/info:", url_path="about"),
}
