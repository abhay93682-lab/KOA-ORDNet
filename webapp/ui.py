"""Shared look and feel: theme CSS, top bar and small HTML components."""
from __future__ import annotations

import html

import streamlit as st

from .core import KL, device_label

CSS = """
<style>
:root{
  --bg:#0b1220; --bg2:#0f172a; --card:#111a2e; --card2:#0d1527; --line:#1f2b45;
  --ink:#e6ecf7; --muted:#94a3b8; --blue:#2563eb; --cyan:#38bdf8; --violet:#818cf8;
}
html, body, [data-testid="stAppViewContainer"], .stApp{background:radial-gradient(1200px 600px at 50% -10%, #13203d 0%, var(--bg) 55%) fixed !important; color:var(--ink);}
[data-testid="stHeader"]{background:rgba(11,18,32,.92) !important; border-bottom:1px solid var(--line); backdrop-filter:blur(8px);}
.block-container{padding-top:4.2rem; max-width:1240px;}
h1,h2,h3,h4{color:var(--ink) !important; letter-spacing:-.01em;}
p, li, label, .stMarkdown{color:var(--ink);}
[data-testid="stCaptionContainer"]{color:var(--muted) !important;}
a{color:var(--cyan) !important;}

.k-brand{display:flex; align-items:center; justify-content:space-between; gap:16px; padding:10px 0 18px; border-bottom:1px solid var(--line); margin-bottom:26px;}
.k-logo{display:flex; align-items:center; gap:12px;}
.k-mark{width:44px; height:44px; border-radius:11px; background:linear-gradient(135deg,#0ea5e9,#2563eb); display:grid; place-items:center; box-shadow:0 6px 20px rgba(37,99,235,.35);}
.k-name{font-weight:800; font-size:20px; line-height:1.1;}
.k-sub{font-size:12.5px; color:var(--muted);}
.k-pill{display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border:1px solid var(--line); border-radius:999px; font-size:13px; font-weight:600; background:var(--card2);}
.k-dot{width:8px; height:8px; border-radius:50%; background:#22c55e; box-shadow:0 0 0 3px rgba(34,197,94,.18);}

.k-chip{display:inline-block; padding:6px 16px; border:1px solid #2b3d66; border-radius:999px; font-size:13px; color:#c7d2fe; background:rgba(37,99,235,.08);}
.k-hero{text-align:center; padding:18px 0 8px;}
.k-hero h1{font-size:clamp(32px,5vw,56px) !important; font-weight:900; line-height:1.08; margin:18px 0 10px;}
.k-grad{background:linear-gradient(90deg,#38bdf8,#60a5fa 45%,#a78bfa); -webkit-background-clip:text; background-clip:text; color:transparent;}
.k-lead{font-size:18px; color:#b6c2d9 !important; max-width:760px; margin:0 auto 18px !important; text-align:center;}

.k-card{background:linear-gradient(180deg,var(--card),var(--card2)); border:1px solid var(--line); border-radius:16px; padding:22px 24px; height:100%;}
.k-stat{font-size:30px; font-weight:800; margin-bottom:6px;}
.k-stat-t{font-weight:700; font-size:15px;}
.k-stat-d{font-size:13px; color:var(--muted); margin-top:4px;}
.k-eyebrow{font-size:12.5px; letter-spacing:.12em; font-weight:700; color:var(--muted); text-transform:uppercase;}
.k-eyebrow.y{color:#facc15;}
.k-big{font-size:clamp(34px,4.6vw,54px); font-weight:900; margin:8px 0 14px; letter-spacing:-.02em;}
.k-badge{display:inline-block; padding:8px 18px; border-radius:999px; font-weight:700; font-size:16px; margin-right:10px; border:1px solid;}
.k-conf{display:inline-block; padding:8px 18px; border-radius:999px; font-weight:700; font-size:15px; border:1px solid var(--line); background:var(--card2);}
.k-kv dt{font-size:12px; letter-spacing:.1em; color:var(--muted); text-transform:uppercase; margin-top:10px;}
.k-kv dd{margin:2px 0 0; font-weight:700; font-size:16px;}

.k-kl{border:1px solid var(--line); border-radius:14px; padding:16px; background:var(--card2); height:100%; position:relative;}
.k-kl.on{border:2px solid var(--blue); box-shadow:0 0 0 4px rgba(37,99,235,.18);}
.k-kl-h{display:flex; justify-content:space-between; align-items:center; font-weight:800; font-size:18px;}
.k-kl-b{font-size:12px; font-weight:700; padding:3px 10px; border-radius:999px; border:1px solid;}
.k-kl-d{font-size:12.5px; color:var(--muted); margin-top:10px;}
.k-flag{position:absolute; top:-11px; right:12px; background:var(--blue); color:#fff; font-size:10.5px; font-weight:800; padding:2px 9px; border-radius:999px; letter-spacing:.05em;}
.k-imgtitle{font-weight:700; font-size:16px; margin-bottom:8px;}
.k-tag{font-size:12px; color:var(--cyan); border:1px solid #1e4b6b; padding:3px 10px; border-radius:999px; background:rgba(56,189,248,.08); float:right;}
.k-note{font-size:13px; color:var(--muted);}
.k-step{display:flex; gap:14px; align-items:flex-start;}
.k-num{flex:none; width:34px; height:34px; border-radius:10px; background:rgba(37,99,235,.15); border:1px solid #2b3d66; display:grid; place-items:center; font-weight:800; color:#93c5fd;}

[data-testid="stImage"] img{border-radius:10px; background:#000; max-height:430px; object-fit:contain;}
div.stButton > button, div.stDownloadButton > button{border-radius:10px; font-weight:700; padding:.6rem 1.2rem; border:1px solid #2b3d66;}
div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"]{background:var(--blue); border-color:var(--blue); box-shadow:0 8px 24px rgba(37,99,235,.35);}
[data-testid="stFileUploaderDropzone"]{background:var(--card2); border:1.5px dashed #2b3d66; border-radius:14px;}
[data-testid="stMetric"]{background:var(--card2); border:1px solid var(--line); border-radius:12px; padding:12px 16px;}
[data-testid="stDataFrame"]{border:1px solid var(--line); border-radius:12px;}
footer{visibility:hidden;}
</style>
"""

ICON = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>')


def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f"""<div class="k-brand">
              <div class="k-logo"><div class="k-mark">{ICON}</div>
                <div><div class="k-name">KOA-ORDNet</div><div class="k-sub">Ordinal-Aware Knee Osteoarthritis AI</div></div></div>
              <span class="k-pill"><span class="k-dot"></span>Model Online ({device_label()})</span>
            </div>""", unsafe_allow_html=True)


def card(inner: str) -> None:
    st.markdown(f'<div class="k-card">{inner}</div>', unsafe_allow_html=True)


def badge(grade: int) -> str:
    c = KL[grade]["color"]
    return (f'<span class="k-badge" style="color:{c}; border-color:{c}; background:{c}1f">'
            f'Severity: {html.escape(KL[grade]["name"])}</span>')


def kl_cards(selected: int | None) -> None:
    cols = st.columns(5)
    for g, col in enumerate(cols):
        c = KL[g]["color"]
        on = " on" if g == selected else ""
        flag = '<span class="k-flag">PREDICTED GRADE</span>' if g == selected else ""
        col.markdown(
            f'<div class="k-kl{on}">{flag}<div class="k-kl-h">KL {g}'
            f'<span class="k-kl-b" style="color:{c}; border-color:{c}; background:{c}1a">{KL[g]["name"]}</span></div>'
            f'<div class="k-kl-d">{KL[g]["desc"]}</div></div>', unsafe_allow_html=True)
