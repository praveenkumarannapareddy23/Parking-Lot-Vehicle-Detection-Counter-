"""Streamlit front end for the vehicle counter (bonus feature).

Run with:
    streamlit run app.py

Everything the brief asks for is reachable from this page: pick an image or a
video, detect cars / motorcycles / buses / trucks, see boxes, labels and
confidence scores, read the total and per-class counts, optionally restrict
counting to a Region of Interest, and save the processed result.

It calls exactly the same pipeline functions as the CLI, so the two front ends
can never disagree about what a count means.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from vehicle_counter import annotate, config
from vehicle_counter.detector import VehicleDetector
from vehicle_counter.pipeline import process_image, process_video, source_kind
from vehicle_counter.roi import ROI, ROI_RULES

st.set_page_config(
    page_title="Vehicle Counter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────── #
st.markdown(
    """
    <style>
    /* ── Google Fonts ───────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 999999 !important;
        pointer-events: none;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto;
    }

    /* ── Sidebar Toggle Arrow Button ────────────────────────────────────── */
    button[data-testid="stHeaderIconButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[aria-label*="sidebar"],
    button[aria-label*="Sidebar"],
    button[aria-label*="collapse"],
    button[aria-label*="expand"] {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: linear-gradient(135deg, #1e1b4b 0%, #2e1065 100%) !important;
        border: 1.5px solid rgba(108, 99, 255, 0.6) !important;
        border-radius: 12px !important;
        color: #c4b5fd !important;
        padding: 6px 10px !important;
        margin: 8px !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.35) !important;
    }
    button[data-testid="stHeaderIconButton"]:hover,
    [data-testid="collapsedControl"]:hover,
    [data-testid="stSidebarCollapseButton"]:hover {
        background: linear-gradient(135deg, #2e1065 0%, #4c1d95 100%) !important;
        border-color: #a78bfa !important;
        color: #ffffff !important;
        transform: scale(1.08) !important;
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.6) !important;
    }
    button[data-testid="stHeaderIconButton"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #c4b5fd !important;
        color: #c4b5fd !important;
        width: 1.4rem !important;
        height: 1.4rem !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px !important;
    }

    /* ── Hero banner ────────────────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #1a1033 0%, #2d1b69 30%, #1e1145 60%, #0d1b2a 100%);
        border-radius: 20px;
        padding: 2.5rem 2.8rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(108, 99, 255, 0.2);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15), 0 0 60px rgba(108, 99, 255, 0.05);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -80px; right: -60px;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(108,99,255,0.25) 0%, transparent 70%);
        border-radius: 50%;
        animation: pulse-glow 4s ease-in-out infinite;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -100px; left: 5%;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(0,210,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
        animation: pulse-glow 5s ease-in-out infinite reverse;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.15); }
    }
    .hero h1 {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.4rem !important;
        position: relative; z-index: 1;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        color: #a5b4c8 !important;
        font-size: 1rem !important;
        margin: 0 !important;
        position: relative; z-index: 1;
        font-weight: 400;
    }
    .hero .badge-row {margin-top: 1rem; position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 0.5rem;}
    .hero .badge {
        display: inline-block;
        background: rgba(108,99,255,0.15);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 24px;
        padding: 0.3rem 0.9rem;
        font-size: 0.78rem;
        color: #c4b5fd;
        font-weight: 500;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .hero .badge:hover {
        background: rgba(108,99,255,0.3);
        border-color: rgba(108,99,255,0.5);
        transform: translateY(-1px);
    }

    /* ── Section headers ────────────────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(108,99,255,0.25);
    }
    .section-header h3 {
        margin: 0 !important;
        font-size: 1.1rem !important;
        color: #e8e8ec !important;
        font-weight: 600 !important;
    }
    .section-icon {
        font-size: 1.15rem;
    }

    /* ── KPI cards ──────────────────────────────────────────────────────── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.9rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: rgba(26, 29, 41, 0.8);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .kpi-card:hover {
        border-color: rgba(108,99,255,0.3);
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .kpi-card .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .kpi-card .kpi-label {
        font-size: 0.75rem;
        color: #8892a8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
    .kpi-total .kpi-value {color: #ffffff;}
    .kpi-total::before {background: linear-gradient(90deg, #6C63FF, #00D2FF);}
    .kpi-car .kpi-value {color: #4ade80;}
    .kpi-car::before {background: #4ade80;}
    .kpi-truck .kpi-value {color: #f87171;}
    .kpi-truck::before {background: #f87171;}
    .kpi-bus .kpi-value {color: #60a5fa;}
    .kpi-bus::before {background: #60a5fa;}
    .kpi-motorcycle .kpi-value {color: #fb923c;}
    .kpi-motorcycle::before {background: #fb923c;}
    .kpi-default .kpi-value {color: #a78bfa;}
    .kpi-default::before {background: #a78bfa;}

    /* ── Status bar ─────────────────────────────────────────────────────── */
    .status-bar {
        background: rgba(26, 29, 41, 0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: #a5b4c8;
        backdrop-filter: blur(10px);
    }
    .status-bar .status-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 6px currentColor;
    }
    .dot-green {background: #4ade80; color: #4ade80;}
    .dot-blue {background: #6C63FF; color: #6C63FF;}
    .dot-amber {background: #fbbf24; color: #fbbf24;}

    /* ── Radio button pills (Choose Source & ROI) ────────────────────────── */
    div[data-testid="stRadio"] > div,
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.8rem !important;
        width: 100% !important;
        margin-bottom: 0.8rem !important;
    }
    div[data-testid="stRadio"] > div > label,
    div[role="radiogroup"] > label {
        flex: 1 1 auto !important;
        min-height: 42px !important;
        height: 42px !important;
        background: rgba(26, 29, 41, 0.7) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 0.45rem 1.1rem !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(15px) !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        color: #e8e8ec !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        white-space: nowrap !important;
    }
    div[data-testid="stRadio"] > div > label:hover,
    div[role="radiogroup"] > label:hover {
        border-color: rgba(108, 99, 255, 0.5) !important;
        background: rgba(108, 99, 255, 0.12) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(108, 99, 255, 0.15) !important;
    }
    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked),
    div[role="radiogroup"] > label:has(input:checked) {
        border-color: #6C63FF !important;
        background: rgba(108, 99, 255, 0.22) !important;
        box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.3), 0 4px 16px rgba(108, 99, 255, 0.25) !important;
        color: #ffffff !important;
    }

    /* ── Styled Section Cards / Panels ─────────────────────────────────── */
    .glass-card-panel {
        background: rgba(26, 29, 41, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(20px);
    }

    /* ── Run button area ────────────────────────────────────────────────── */
    .run-area {
        background: rgba(108,99,255,0.08);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin: 0.8rem 0;
    }

    /* ── Result panel ───────────────────────────────────────────────────── */
    .result-panel {
        background: rgba(26, 29, 41, 0.5);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        backdrop-filter: blur(20px);
    }
    .result-panel h2 {
        font-size: 1.4rem !important;
        color: #e8e8ec !important;
        margin-top: 0 !important;
    }

    /* ── Download buttons ───────────────────────────────────────────────── */
    .dl-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .dl-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        background: rgba(108,99,255,0.1);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 12px;
        padding: 0.8rem 1.1rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #c4b5fd;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .dl-btn:hover {
        background: rgba(108,99,255,0.2);
        border-color: rgba(108,99,255,0.5);
        color: #ffffff;
        transform: translateY(-1px);
    }

    /* ── Sidebar styling ────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #12141e !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border-bottom: 2px solid rgba(108,99,255,0.25);
        padding-bottom: 0.4rem;
        font-weight: 600;
    }

    /* ── Inference info strip ───────────────────────────────────────────── */
    .inference-strip {
        background: rgba(26, 29, 41, 0.6);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: #8892a8;
        margin-top: 0.6rem;
    }
    .inference-strip code {
        background: rgba(108,99,255,0.15);
        color: #c4b5fd;
        padding: 0.15rem 0.4rem;
        border-radius: 5px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* ── Divider ────────────────────────────────────────────────────────── */
    .soft-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 1.2rem 0;
    }

    /* ── Streamlit metrics override ─────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(26, 29, 41, 0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        color: #8892a8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #e8e8ec !important;
    }

    /* ── Tab styling ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(26, 29, 41, 0.6);
        border-radius: 12px;
        padding: 5px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(108,99,255,0.2) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6C63FF 0%, #5A52D5 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 14px rgba(108,99,255,0.35) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(108,99,255,0.5) !important;
        transform: translateY(-1px);
    }
    .stButton > button:not([kind="primary"]) {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #c4b5fd !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: rgba(108,99,255,0.1) !important;
        border-color: rgba(108,99,255,0.3) !important;
    }

    /* ── Expanders ───────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(26, 29, 41, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        color: #e8e8ec !important;
    }

    /* ── Selectbox / Inputs ──────────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(26, 29, 41, 0.6) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e8e8ec !important;
    }

    /* ── Radio buttons ──────────────────────────────────────────────────── */
    .stRadio > div {
        gap: 0.3rem;
    }

    /* ── File uploader ──────────────────────────────────────────────────── */
    .stFileUploader > div {
        border: 2px dashed rgba(108,99,255,0.3) !important;
        border-radius: 14px !important;
        background: rgba(108,99,255,0.04) !important;
    }
    .stFileUploader > div:hover {
        border-color: rgba(108,99,255,0.5) !important;
        background: rgba(108,99,255,0.08) !important;
    }

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {width: 6px;}
    ::-webkit-scrollbar-track {background: transparent;}
    ::-webkit-scrollbar-thumb {
        background: rgba(108,99,255,0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {background: rgba(108,99,255,0.5);}

    /* ── Data frames ────────────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ── Progress bar ───────────────────────────────────────────────────── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF, #00D2FF) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "input"
ROI_DIR = ROOT / "data" / "roi"
OUTPUT_DIR = ROOT / "outputs" / "ui"
MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt"]
MEDIA_EXTS = config.IMAGE_EXTS | config.VIDEO_EXTS
LINE_COLOR = "#6C63FF"
DRAW_WIDTH = 900

# Auto-download demo sample files if not present (for Streamlit Cloud deployments)
if not (SAMPLE_DIR.exists() and any(p.suffix.lower() in MEDIA_EXTS for p in SAMPLE_DIR.glob("*"))):
    try:
        import subprocess
        subprocess.run(["python", str(ROOT / "tools" / "fetch_samples.py")], check=False)
    except Exception:
        pass

# Map class names → CSS color classes for KPI cards
CLASS_KPI_MAP = {
    "car": "kpi-car",
    "truck": "kpi-truck",
    "bus": "kpi-bus",
    "motorcycle": "kpi-motorcycle",
}


# ─── Helpers ────────────────────────────────────────────────────────────────── #
@st.cache_resource(show_spinner="Loading the YOLO model...")
def load_detector(model: str, device: str, conf: float, iou: float,
                  imgsz: int | str, class_ids: tuple[int, ...]) -> VehicleDetector:
    """Cached so the weights are not reloaded on every widget interaction."""
    return VehicleDetector(model_path=model, device=device, conf=conf,
                           iou=iou, imgsz=imgsz, class_ids=list(class_ids))


def first_frame(path: Path):
    """Representative frame, used for the ROI preview."""
    if source_kind(path) == "image":
        return cv2.imread(str(path))
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def bgr_to_rgb(frame):
    """OpenCV works in BGR; Streamlit expects RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_kpi_cards(counts: dict[str, int], total: int) -> None:
    """Render KPI cards with color-coded values per vehicle class."""
    cards_html = '<div class="kpi-grid">'
    # Total card first
    cards_html += (
        f'<div class="kpi-card kpi-total">'
        f'<div class="kpi-value">{total}</div>'
        f'<div class="kpi-label">Total Vehicles</div></div>'
    )
    for name, count in counts.items():
        css_class = CLASS_KPI_MAP.get(name, "kpi-default")
        icon = {"car": "🚗", "truck": "🚛", "bus": "🚌", "motorcycle": "🏍️"}.get(name, "🚘")
        cards_html += (
            f'<div class="kpi-card {css_class}">'
            f'<div class="kpi-value">{icon} {count}</div>'
            f'<div class="kpi-label">{name}</div></div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_status_bar(*items: tuple[str, str]) -> None:
    """Render a horizontal status bar with colored dots."""
    html = '<div class="status-bar">'
    for label, dot_class in items:
        html += (
            f'<span class="status-item">'
            f'<span class="status-dot {dot_class}"></span>{label}</span>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-icon">{icon}</span>'
        f'<h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────── #
with st.sidebar:
    # ── Sidebar logo / title ────────────────────────────────────────────── #
    st.markdown(
        '<div style="text-align:center; padding: 0.8rem 0 0.5rem 0;">'
        '<span style="font-size:2rem;">🚗</span>'
        '<h2 style="margin:0.2rem 0 0 0; font-size:1.15rem; '
        'background: linear-gradient(135deg, #c4b5fd, #6C63FF); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
        'font-weight:700;">Vehicle Counter</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border:none; border-top:1px solid rgba(255,255,255,0.08); margin:0.3rem 0 0.8rem 0;">', unsafe_allow_html=True)

    # ── Model ──────────────────────────────────────────────────────────── #
    st.markdown(
        '<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; '
        'color:#8892a8; font-weight:600; margin-bottom:0.3rem;">🤖 Model</p>',
        unsafe_allow_html=True,
    )
    model_name = st.selectbox(
        "Weights", MODELS, index=0,
        help="n = fastest, m = most accurate. Downloaded on first use.",
        label_visibility="collapsed",
    )

    # ── Device ─────────────────────────────────────────────────────────── #
    st.markdown(
        '<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; '
        'color:#8892a8; font-weight:600; margin-bottom:0.3rem;">💻 Device</p>',
        unsafe_allow_html=True,
    )
    device = st.selectbox(
        "Device", [config.pick_device(), "cpu", "mps"], index=0,
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border:none; border-top:1px solid rgba(255,255,255,0.08); margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── Detection Settings ─────────────────────────────────────────────── #
    st.markdown(
        '<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; '
        'color:#8892a8; font-weight:600; margin-bottom:0.3rem;">🎯 Detection</p>',
        unsafe_allow_html=True,
    )
    conf = st.slider(
        "Confidence threshold", 0.05, 0.95, config.DEFAULT_CONF, 0.05,
        help="Lower = more detections, but more false positives.",
    )
    iou = st.slider(
        "NMS IoU", 0.1, 0.9, config.DEFAULT_IOU, 0.05,
        help="How much overlap before two boxes merge.",
    )
    auto_imgsz = st.checkbox(
        "Auto inference size", value=True,
        help="Match source resolution instead of defaulting to 640px.",
    )
    imgsz: int | str = "auto"
    if not auto_imgsz:
        imgsz = st.select_slider(
            "Inference size", [640, 960, 1280, 1600, 1920], value=1280,
        )

    st.markdown('<hr style="border:none; border-top:1px solid rgba(255,255,255,0.08); margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── Classes ────────────────────────────────────────────────────────── #
    st.markdown(
        '<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; '
        'color:#8892a8; font-weight:600; margin-bottom:0.3rem;">🏷️ Classes</p>',
        unsafe_allow_html=True,
    )
    class_names = st.multiselect(
        "Count these classes",
        options=list(config.ALL_KNOWN_CLASSES.values()),
        default=list(config.VEHICLE_CLASSES.values()),
    )
    label_mode = st.radio(
        "Box captions", annotate.LABEL_MODES, index=0, horizontal=True,
        help="auto shrinks text to fit boxes.",
    )

    st.markdown('<hr style="border:none; border-top:1px solid rgba(255,255,255,0.08); margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── Video ──────────────────────────────────────────────────────────── #
    st.markdown(
        '<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; '
        'color:#8892a8; font-weight:600; margin-bottom:0.3rem;">🎬 Video</p>',
        unsafe_allow_html=True,
    )
    use_tracking = st.checkbox("Track vehicles between frames", value=True)
    min_track_hits = st.number_input(
        "Frames before a track counts", 1, 30, 3,
        help="Filters out transient tracker ghosts.",
    )
    stride = st.number_input("Process every Nth frame", 1, 10, 1)
    max_frames = st.number_input("Max frames (0 = all)", 0, 10_000, 0)

    # Hidden but needed downstream
    name_to_id = {v: k for k, v in config.ALL_KNOWN_CLASSES.items()}
    class_ids = tuple(sorted(name_to_id[n] for n in class_names))


# ─── Hero Banner ────────────────────────────────────────────────────────────── #
st.markdown(
    """
    <div class="hero">
        <h1>🚗 Parking Lot Vehicle Counter</h1>
        <p>Detect, track and count vehicles in images and video — powered by YOLO11</p>
        <div class="badge-row">
            <span class="badge">YOLO11</span>
            <span class="badge">ByteTrack</span>
            <span class="badge">OpenCV</span>
            <span class="badge">Auto Resolution</span>
            <span class="badge">ROI Filtering</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Source Selection ───────────────────────────────────────────────────────── #
st.markdown('<div class="glass-card-panel">', unsafe_allow_html=True)
section_header("📁", "Choose Source")

samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in MEDIA_EXTS)
has_samples = len(samples) > 0

# Source mode as interactive cards
source_mode = None
if has_samples:
    source_mode_choice = st.radio(
        "Source Selector",
        ["📁 Bundled Samples", "📤 Upload a File"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    source_mode = "Bundled sample" if "Bundled" in source_mode_choice else "Upload a file"
else:
    source_mode = "Upload a file"

source_path: Path | None = None
workdir = st.session_state.setdefault("_workdir", Path(tempfile.mkdtemp(prefix="vc_")))

if source_mode == "Bundled sample":
    picked = st.selectbox("Sample file", samples, format_func=lambda p: p.name)
    source_path = picked
else:
    uploaded = st.file_uploader(
        "Upload an image or a video",
        type=sorted(e.lstrip(".") for e in MEDIA_EXTS),
        label_visibility="collapsed",
    )
    if uploaded:
        source_path = workdir / uploaded.name
        source_path.write_bytes(uploaded.getbuffer())

st.markdown('</div>', unsafe_allow_html=True)

if source_path is None:
    st.markdown(
        '<div style="text-align:center; padding:2.5rem; background:rgba(26,29,41,0.6); '
        'border-radius:16px; border:2px dashed rgba(108,99,255,0.3); backdrop-filter:blur(20px);">'
        '<div style="font-size:2.5rem; margin-bottom:0.5rem;">🖼️</div>'
        '<p style="color:#a5b4c8; margin:0;">Select a bundled sample or upload a file to begin</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not class_ids:
    st.warning("⚠️ Pick at least one vehicle class in the sidebar.")
    st.stop()

kind = source_kind(source_path)
preview = first_frame(source_path)
if preview is None:
    st.error("❌ Could not read that file. Check the format and try again.")
    st.stop()
height, width = preview.shape[:2]


# ─── Region of Interest ─────────────────────────────────────────────────────── #
st.markdown('<div class="glass-card-panel">', unsafe_allow_html=True)
section_header("📐", "Region of Interest")

roi_files = sorted(ROI_DIR.glob("*.json"))
roi_options = ["Whole frame", "Draw polygon", "Rectangle"]
if roi_files:
    roi_options.append("Saved polygon")
roi_options.append("Upload JSON")

roi_mode = st.radio(
    "ROI", roi_options, horizontal=True, label_visibility="collapsed",
)

roi: ROI | None = None
roi_rule = "bottom"

if roi_mode == "Draw polygon":
    st.caption(
        "Click on the image to drop points. **3+ points** define a region. "
        "Points are normalized, so the same ROI works at any resolution."
    )
    points: list[tuple[float, float]] = st.session_state.setdefault("draw_points", [])

    canvas = annotate.draw_polygon_in_progress(
        preview.copy(), [(int(x * width), int(y * height)) for x, y in points]
    )
    click = streamlit_image_coordinates(
        bgr_to_rgb(canvas), width=DRAW_WIDTH, key="roi_canvas", cursor="crosshair",
    )

    if click and click.get("unix_time") != st.session_state.get("last_click_time"):
        st.session_state["last_click_time"] = click.get("unix_time")
        points.append((click["x"] / click["width"], click["y"] / click["height"]))
        st.rerun()

    btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 4])
    if btn_c1.button("↩ Undo", disabled=not points, width="stretch"):
        points.pop()
        st.rerun()
    if btn_c2.button("🗑 Clear", disabled=not points, width="stretch"):
        points.clear()
        st.rerun()
    btn_c3.markdown(
        f'<div style="padding-top:0.5rem; color:#64748b; font-size:0.85rem;">'
        f'<strong>{len(points)}</strong> point(s)'
        f'{" ✅" if len(points) >= 3 else " — need 3 minimum"}</div>',
        unsafe_allow_html=True,
    )

    if len(points) >= 3:
        roi = ROI(points=list(points), name="drawn", normalized=True)
        save_c1, save_c2 = st.columns([1, 3])
        roi_name = save_c2.text_input(
            "Save as", value="my_lot", label_visibility="collapsed",
            placeholder="region name",
        )
        if save_c1.button("💾 Save", width="stretch"):
            safe = "".join(c for c in roi_name if c.isalnum() or c in "-_") or "roi"
            saved = ROI(points=list(points), name=safe, normalized=True).save(
                ROI_DIR / f"{safe}.json"
            )
            st.success(
                f"Saved to `{saved.relative_to(ROOT)}` — "
                f"now available under **Saved polygon** and CLI `--roi`."
            )

elif roi_mode == "Rectangle":
    rect_c1, rect_c2 = st.columns(2)
    with rect_c1:
        left, right = st.slider("Horizontal", 0.0, 1.0, (0.0, 0.6), 0.01)
    with rect_c2:
        top, bottom = st.slider("Vertical", 0.0, 1.0, (0.15, 1.0), 0.01)
    if right > left and bottom > top:
        roi = ROI.from_rect(left, top, right, bottom, name="rectangle", normalized=True)
    else:
        st.warning("⚠️ Rectangle has no area — widen the range.")

elif roi_mode == "Saved polygon":
    picked_roi = st.selectbox("ROI file", roi_files, format_func=lambda p: p.name)
    roi = ROI.load(picked_roi)

elif roi_mode == "Upload JSON":
    roi_file = st.file_uploader("ROI JSON from roi_picker.py", type=["json"], key="roi")
    if roi_file:
        roi_path = workdir / "roi.json"
        roi_path.write_bytes(roi_file.getbuffer())
        roi = ROI.load(roi_path)

if roi is not None:
    roi_info_c1, roi_info_c2 = st.columns([3, 2])
    roi_info_c1.markdown(
        f'<div style="background:#eff6ff; border-radius:8px; padding:0.5rem 0.8rem; '
        f'font-size:0.85rem; color:#1e40af;">'
        f'📌 <strong>{roi.name}</strong> — {len(roi.points)} points</div>',
        unsafe_allow_html=True,
    )
    roi_rule = roi_info_c2.selectbox(
        "Containment rule", ROI_RULES, index=0,
        help="bottom = tire anchor (angled views); center = centroid; overlap = any box part.",
        label_visibility="collapsed",
    )

if roi_mode != "Draw polygon":
    st.image(
        bgr_to_rgb(annotate.draw_roi(preview.copy(), roi)),
        caption=f"{source_path.name} — {width}×{height}"
                + ("" if roi is None else f" — ROI '{roi.name}' shaded"),
        width="stretch",
    )

st.markdown('</div>', unsafe_allow_html=True)


# ─── Run Button ─────────────────────────────────────────────────────────────── #
run_c1, run_c2, run_c3 = st.columns([2, 2, 6])
with run_c1:
    run_btn = st.button("▶  Run Detection", type="primary", width="stretch")
with run_c2:
    stop_btn = st.button("⏹  Stop", width="stretch")

if stop_btn:
    st.info("Detection stopped.")
    st.stop()

if not run_btn:
    st.stop()


# ─── Status Bar ─────────────────────────────────────────────────────────────── #
resolved_imgsz = config.resolve_imgsz(imgsz, (height, width)) if imgsz == "auto" else int(imgsz)
device_label = config.pick_device(device) if device in (None, "", "auto") else device
render_status_bar(
    (f"Model: {Path(model_name).stem}", "dot-blue"),
    (f"Device: {device_label}", "dot-green"),
    (f"imgsz: {resolved_imgsz}", "dot-amber"),
    (f"Source: {kind}", "dot-blue"),
    (f"Resolution: {width}×{height}", "dot-amber"),
)


# ─── Detection ──────────────────────────────────────────────────────────────── #
detector = load_detector(model_name, device, conf, iou, imgsz, class_ids)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if kind == "image":
    with st.spinner("🔍 Detecting vehicles..."):
        result = process_image(
            source_path, detector, roi, roi_rule, OUTPUT_DIR,
            save_json=True, label_mode=label_mode,
        )

    # ── Results Panel ──────────────────────────────────────────────────── #
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    section_header("📊", "Detection Results")
    render_kpi_cards(result.counts, result.total)

    # Annotated image
    st.image(
        bgr_to_rgb(cv2.imread(str(result.output_image))),
        caption="Annotated output",
        width="stretch",
    )

    # Inference info
    st.markdown(
        f'<div class="inference-strip">'
        f'⏱️ Inference: <code>{result.elapsed_s * 1000:.0f} ms</code>'
        f' &nbsp;|&nbsp; Device: <code>{detector.device}</code>'
        f' &nbsp;|&nbsp; imgsz: <code>{detector.imgsz_for(preview)}</code>'
        f' &nbsp;|&nbsp; Saved: <code>{result.output_image.relative_to(ROOT)}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Details expander
    with st.expander("🔎 Raw detections (what YOLO returned)", expanded=False):
        st.dataframe(
            pd.DataFrame([
                {
                    "class": d.cls_name,
                    "confidence": f"{d.conf:.3f}",
                    "x1": round(d.xyxy[0]),
                    "y1": round(d.xyxy[1]),
                    "x2": round(d.xyxy[2]),
                    "y2": round(d.xyxy[3]),
                    "w": round(d.wh[0]),
                    "h": round(d.wh[1]),
                }
                for d in sorted(result.detections, key=lambda d: -d.conf)
            ]),
            width="stretch",
            hide_index=True,
        )

    # Downloads
    st.markdown('<div class="dl-grid">', unsafe_allow_html=True)
    dl_c1, dl_c2 = st.columns(2)
    dl_c1.download_button(
        "📥 Download Annotated Image",
        result.output_image.read_bytes(),
        file_name=result.output_image.name,
        mime="image/jpeg",
        width="stretch",
    )
    if result.summary_json:
        dl_c2.download_button(
            "📥 Download Counts JSON",
            result.summary_json.read_bytes(),
            file_name=result.summary_json.name,
            mime="application/json",
            width="stretch",
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # end result-panel

else:
    # ── Video Processing ───────────────────────────────────────────────── #
    bar = st.progress(0.0, text="🎬 Processing video...")

    def on_progress(done: int, total: int | None) -> None:
        if total:
            bar.progress(
                min(done / total, 1.0),
                text=f"🎬 Frame {done} of {total}",
            )

    result = process_video(
        source_path, detector, roi, roi_rule, OUTPUT_DIR,
        use_tracking=use_tracking, min_track_hits=int(min_track_hits),
        stride=int(stride), max_frames=int(max_frames) or None,
        save_json=True, label_mode=label_mode,
        progress_cb=on_progress, show_progress=False,
    )
    bar.empty()

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)

    # ── Occupancy Section ──────────────────────────────────────────────── #
    section_header("📊", "Occupancy (how full it gets)")
    occ_c1, occ_c2 = st.columns(2)
    occ_c1.metric("Peak vehicles in one frame", result.peak_occupancy)
    occ_c2.metric("Mean vehicles per frame", f"{result.mean_occupancy:.1f}")
    render_kpi_cards(result.peak_counts, result.peak_occupancy)

    # ── Unique Vehicles Section ────────────────────────────────────────── #
    if use_tracking:
        section_header("🔄", "Unique Vehicles (how many came through)")
        uniq_c1, uniq_c2 = st.columns(2)
        uniq_c1.metric(
            "Unique vehicles", result.unique_total,
            help="Distinct track IDs surviving the minimum-frames filter.",
        )
        uniq_c2.metric(
            "Raw track IDs", result.unique_raw,
            help="Before filtering. The gap is tracker ID churn.",
        )
        if result.unique_by_class:
            render_kpi_cards(result.unique_by_class, result.unique_total)

    # ── Video playback ─────────────────────────────────────────────────── #
    section_header("🎬", "Annotated Video")
    st.video(result.output_video.read_bytes())
    st.markdown(
        f'<div class="inference-strip">'
        f'⏱️ Processed <code>{result.frames_processed}</code> frames in '
        f'<code>{result.elapsed_s:.1f}s</code> '
        f'(<code>{result.fps_processing:.1f} FPS</code>) on '
        f'<code>{detector.device}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Occupancy chart ────────────────────────────────────────────────── #
    if result.per_frame:
        section_header("📈", "Occupancy Over Time")
        chart_df = pd.DataFrame(result.per_frame)[["time_s", "occupancy"]].set_index("time_s")
        chart_df.columns = ["vehicles in frame"]
        st.area_chart(chart_df, color="#3b82f6")

    # ── Downloads ──────────────────────────────────────────────────────── #
    section_header("📥", "Downloads")
    dl_c1, dl_c2 = st.columns(2)
    dl_c1.download_button(
        "📥 Download Annotated Video",
        result.output_video.read_bytes(),
        file_name=result.output_video.name,
        mime="video/mp4",
        width="stretch",
    )
    if result.per_frame_csv:
        dl_c2.download_button(
            "📥 Download Per-Frame CSV",
            result.per_frame_csv.read_bytes(),
            file_name=result.per_frame_csv.name,
            mime="text/csv",
            width="stretch",
        )
    if result.summary_json:
        st.download_button(
            "📥 Download Summary JSON",
            result.summary_json.read_bytes(),
            file_name=result.summary_json.name,
            mime="application/json",
            width="stretch",
        )

    st.markdown("</div>", unsafe_allow_html=True)  # end result-panel
