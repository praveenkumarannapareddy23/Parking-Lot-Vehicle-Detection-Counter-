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
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates

from vehicle_counter import annotate, config
from vehicle_counter.detector import VehicleDetector
from vehicle_counter.pipeline import process_image, process_video, source_kind
from vehicle_counter.roi import ROI, ROI_RULES

st.set_page_config(
    page_title="Vehicle Counter",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────── #
st.markdown(
    """
    <style>
    /* ── Fonts ──────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ─────────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-feature-settings: 'cv02', 'cv03', 'cv04';
    }
    ::selection { background: rgba(90, 116, 245, 0.35); color: #C7D2FE; }
    body, [data-testid="stAppViewContainer"] {
        background: #0A0A0B;
    }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 999999 !important;
        pointer-events: none;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto;
    }
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    /* Slightly larger base type: scales every rem-based size (body, widgets) */
    html {
        font-size: 17px;
    }

    /* ── Sidebar: wider, light, icon labels ─────────────────────────────── */
    section[data-testid="stSidebar"] {
        width: 380px !important;
        min-width: 380px !important;
        background: #131318 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 1px 0 0 rgba(0, 0, 0, 0.5);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #A3ADBB !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.4rem;
        font-weight: 600;
    }
    .side-label {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
        font-weight: 600;
        margin: 0.55rem 0 0.3rem 0;
    }
    .side-label svg { width: 13px; height: 13px; flex: none; }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.4rem 0 0.6rem 0;
    }
    .sidebar-brand .brand-tile {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px; height: 40px;
        border-radius: 10px;
        background: rgba(90, 116, 245, 0.16);
        border: 1px solid rgba(90, 116, 245, 0.5);
        color: #5A74F5;
        flex: none;
    }
    .sidebar-brand .brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F5F6F7;
        line-height: 1.2;
    }
    .sidebar-brand .brand-sub {
        font-size: 0.8rem;
        color: #8A93A3;
        margin-top: 0.1rem;
    }

    /* ── Sidebar toggle button (neutral, functional) ────────────────────── */
    button[data-testid="stHeaderIconButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #131318 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #A3ADBB !important;
        padding: 6px 10px !important;
        margin: 8px !important;
        cursor: pointer !important;
        transition:
            border-color 150ms cubic-bezier(0.23, 1, 0.32, 1),
            background-color 150ms cubic-bezier(0.23, 1, 0.32, 1) !important;
    }
    button[data-testid="stHeaderIconButton"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #A3ADBB !important;
        color: #A3ADBB !important;
    }

    /* ── App header (replaces the gradient hero) ────────────────────────── */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        background: #131318;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
    }
    .app-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 52px; height: 52px;
        border-radius: 13px;
        background: rgba(90, 116, 245, 0.16);
        border: 1px solid rgba(90, 116, 245, 0.5);
        color: #5A74F5;
        flex: none;
    }
    .app-title h1 {
        margin: 0 !important;
        font-size: 2.05rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
        color: #F5F6F7 !important;
        line-height: 1.15;
    }
    .app-title p {
        margin: 0 !important;
        font-size: 1.05rem;
        color: #8A93A3;
        font-weight: 400;
        margin-top: 0.2rem !important;
    }
    .badge-row {
        margin-left: auto;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .badge {
        display: inline-block;
        background: #141419;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 0.28rem 0.7rem;
        font-size: 0.72rem;
        color: #A3ADBB;
        font-weight: 500;
        white-space: nowrap;
    }
    @media (max-width: 860px) { .badge-row { display: none; } }

    /* ── Section headers ────────────────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1rem 0 0.7rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .section-header h3 {
        margin: 0 !important;
        font-size: 0.72rem !important;
        color: #A3ADBB !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }
    .section-icon { display: flex; color: #94A3B8; }
    .section-icon svg { width: 14px; height: 14px; }

    /* ── KPI cards: label over value, colored dot as the class key ──────── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.625rem;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: #131318;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    .kpi-card .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
        color: #F5F6F7;
    }
    .kpi-card .kpi-label {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 0.35rem;
        font-size: 0.66rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .kpi-dot { width: 7px; height: 7px; border-radius: 2px; flex: none; }
    .kpi-total {
        background: rgba(90, 116, 245, 0.16);
        border-color: rgba(90, 116, 245, 0.5);
    }
    .kpi-total .kpi-value { color: #C7D2FE; }
    .kpi-car .kpi-value { color: #4ADE80; }
    .kpi-truck .kpi-value { color: #F87171; }
    .kpi-bus .kpi-value { color: #60A5FA; }
    .kpi-motorcycle .kpi-value { color: #FB923C; }
    .kpi-default .kpi-value { color: #A78BFA; }

    /* ── Status chips ───────────────────────────────────────────────────── */
    .status-bar {
        background: #141419;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.5rem 0.9rem;
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 0.85rem;
        font-size: 0.8rem;
        color: #A3ADBB;
    }
    .status-bar .status-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-green { background: #4ADE80; }
    .dot-blue { background: #5A74F5; }
    .dot-amber { background: #FBBF24; }

    /* ── Radio pills (Choose Source & ROI) ──────────────────────────────── */
    div[data-testid="stRadio"] > div,
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.45rem !important;
        width: 100% !important;
        margin-bottom: 0.8rem !important;
    }
    div[data-testid="stRadio"] > div > label,
    div[role="radiogroup"] > label {
        flex: 1 1 auto !important;
        min-height: 38px !important;
        background: #131318 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 0.4rem 1rem !important;
        text-align: center !important;
        cursor: pointer !important;
        transition:
            border-color 420ms cubic-bezier(0.4, 0, 0.2, 1),
            background-color 420ms cubic-bezier(0.4, 0, 0.2, 1),
            color 420ms cubic-bezier(0.4, 0, 0.2, 1),
            transform 120ms cubic-bezier(0.23, 1, 0.32, 1) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: #A3ADBB !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        white-space: nowrap !important;
    }
    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked),
    div[role="radiogroup"] > label:has(input:checked) {
        border-color: #5A74F5 !important;
        background: rgba(90, 116, 245, 0.16) !important;
        color: #C7D2FE !important;
        font-weight: 600 !important;
    }
    div[role="radiogroup"] > label:active { transform: scale(0.98) !important; }
    div[role="radiogroup"] > label:focus-within {
        outline: 2px solid #5A74F5;
        outline-offset: 1px;
    }

    /* ── Section cards / panels (st.container(border=True)) ─────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #131318;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1.05rem 1.2rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }

    /* ── Run button area ────────────────────────────────────────────────── */
    .run-area {
        background: #141419;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.7rem 0;
    }

    /* ── Inference info strip ───────────────────────────────────────────── */
    .inference-strip {
        background: #141419;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.55rem 0.9rem;
        font-size: 0.8rem;
        color: #8A93A3;
        margin-top: 0.6rem;
    }
    .inference-strip code {
        background: rgba(90, 116, 245, 0.16);
        color: #C7D2FE;
        padding: 0.1rem 0.35rem;
        border-radius: 5px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* ── Streamlit metrics ──────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #131318;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.7rem 0.95rem;
    }
    [data-testid="stMetric"] label {
        font-size: 0.74rem !important;
        color: #94A3B8 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: #F5F6F7 !important;
        font-variant-numeric: tabular-nums;
    }

    /* ── Tab styling ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #141419;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(90, 116, 245, 0.16) !important;
        color: #C7D2FE !important;
    }

    /* ── Buttons: production indigo-blue ────────────────────────────────── */
    /* Covers both Streamlit markups: kind="primary" (older) and
       data-testid="stBaseButton-primary" (1.39+), so styling and hover
       always apply regardless of version. */
    .stButton > button,
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-tertiary"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        letter-spacing: 0.01em;
        /* Color fills in slowly and smoothly; press stays instant. */
        transition:
            transform 120ms cubic-bezier(0.23, 1, 0.32, 1),
            background-color 420ms cubic-bezier(0.4, 0, 0.2, 1),
            border-color 420ms cubic-bezier(0.4, 0, 0.2, 1),
            box-shadow 420ms cubic-bezier(0.4, 0, 0.2, 1),
            color 420ms cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"] {
        background: #5A74F5 !important;
        border: none !important;
        color: #FFFFFF !important;
        box-shadow: 0 1px 2px rgba(90, 116, 245, 0.35) !important;
    }
    .stButton > button:not([kind="primary"]),
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-tertiary"] {
        background: #131318 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        color: #A3ADBB !important;
    }
    .stButton > button:active,
    [data-testid="stBaseButton-primary"]:active,
    [data-testid="stBaseButton-secondary"]:active {
        transform: scale(0.98) !important;
    }
    .stButton > button:focus-visible,
    [data-testid="stBaseButton-primary"]:focus-visible,
    [data-testid="stBaseButton-secondary"]:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(90, 116, 245, 0.25) !important;
    }

    /* ── Selectbox / inputs ─────────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stMultiSelect > div > div {
        background: #131318 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 8px !important;
        color: #F5F6F7 !important;
        font-size: 0.85rem !important;
        transition: border-color 150ms cubic-bezier(0.23, 1, 0.32, 1),
                    box-shadow 150ms cubic-bezier(0.23, 1, 0.32, 1) !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #5A74F5 !important;
        box-shadow: 0 0 0 3px rgba(90, 116, 245, 0.15) !important;
    }

    /* ── File uploader ──────────────────────────────────────────────────── */
    .stFileUploader > div {
        border: 1px dashed rgba(90, 116, 245, 0.5) !important;
        border-radius: 10px !important;
        background: #141419 !important;
        transition: border-color 420ms cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* ── Expanders ──────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        background: #131318 !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #A3ADBB !important;
    }

    /* ── Scrollbar ──────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        border: 2px solid transparent;
        background-clip: content-box;
    }

    /* ── Data frames ────────────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ── Progress bar ───────────────────────────────────────────────────── */
    .stProgress > div > div > div {
        background: #5A74F5 !important;
        border-radius: 999px;
    }

    /* ── Hover: pointing devices only ───────────────────────────────────── */
    @media (hover: hover) and (pointer: fine) {
        /* Buttons keep a calm hover: border brightens only, no fill change. */
        .stButton button:hover,
        [data-testid="stDownloadButton"] button:hover {
            border-color: rgba(255, 255, 255, 0.22) !important;
        }
        /* Primary gets a slightly brighter border, no color fill. */
        .stButton button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover {
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        /* Radio pills: slow accent fill on hover, matching the buttons. */
        div[data-testid="stRadio"] > div > label:hover:not(:has(input:checked)),
        div[role="radiogroup"] > label:hover:not(:has(input:checked)) {
            background: rgba(90, 116, 245, 0.18) !important;
            border-color: rgba(90, 116, 245, 0.55) !important;
            color: #F5F6F7 !important;
        }
        .stFileUploader > div:hover {
            border-color: #5A74F5 !important;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
            background-clip: content-box;
        }
    }

    /* ── Reduced motion ─────────────────────────────────────────────────── */
    @media (prefers-reduced-motion: reduce) {
        * {
            transition-duration: 0.01ms !important;
            animation: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent


def pin_sidebar() -> None:
    """Keep the sidebar open unless the user clicks its own arrow.

    Streamlit collapses the sidebar when clicking anywhere in the main area.
    This injects a tiny script that watches the sidebar and immediately reopens
    it after any collapse that was not triggered by the sidebar's own
    collapse/expand buttons, so the menu stays pinned during normal use.
    """
    components.html(
        """
        <script>
        (function () {
          try {
            var doc = window.parent.document;
            if (doc.__vcSidebarPinned) return;
            doc.__vcSidebarPinned = true;

            var suppressUntil = 0;

            // The user pressed one of the sidebar's own arrows: allow it.
            doc.addEventListener('pointerdown', function (e) {
              var t = e.target;
              if (t && t.closest && t.closest(
                '[data-testid="stSidebarCollapseButton"], ' +
                '[data-testid="stSidebarCollapsedControl"]'
              )) {
                suppressUntil = Date.now() + 600;
              }
            }, true);

            // Anything else collapsed the sidebar: reopen it right away.
            var obs = new MutationObserver(function () {
              if (!doc.querySelector('[data-testid="stSidebar"]')
                  && Date.now() > suppressUntil) {
                var expand = doc.querySelector('[data-testid="stSidebarCollapsedControl"]');
                if (expand) expand.click();
              }
            });
            if (doc.body) {
              obs.observe(doc.body, { childList: true, subtree: true });
            }
          } catch (err) { /* parent document unavailable; keep default behavior */ }
        })();
        </script>
        """,
        height=0,
    )

pin_sidebar()

SAMPLE_DIR = ROOT / "data" / "input"
ROI_DIR = ROOT / "data" / "roi"
OUTPUT_DIR = ROOT / "outputs" / "ui"
MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt"]
MEDIA_EXTS = config.IMAGE_EXTS | config.VIDEO_EXTS
LINE_COLOR = "#5A74F5"
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

CLASS_DOT_COLORS = {
    "car": "#4ADE80",
    "truck": "#F87171",
    "bus": "#60A5FA",
    "motorcycle": "#FB923C",
}


# ─── Icons (inline SVG, feather-style) ──────────────────────────────────────── #
ICONS = {
    "logo": '<rect x="1" y="3" width="15" height="13"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
    "source": '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    "roi": '<path d="M6.13 1L6 16a2 2 0 0 0 2 2h15"/><path d="M1 6.13L16 6a2 2 0 0 1 2 2v15"/>',
    "chart": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "refresh": '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "video": '<polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>',
    "trend": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "model": '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>',
    "device": '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "detect": '<circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/>',
    "classes": '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.83z"/><line x1="7" y1="7" x2="7.01" y2="7"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
}


def icon(name: str, size: int = 15) -> str:
    """Inline SVG icon (stroke = currentColor so CSS colors it)."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


# ─── Helpers ────────────────────────────────────────────────────────────────── #
@st.cache_data(show_spinner=False)
def gpu_device_ids() -> list[str]:
    """CUDA GPU ids on this machine (['0', '1', ...]), probed once.

    Empty when torch is missing or no CUDA device is present. Cached so the
    torch import happens only on the first render.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


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
    """Render KPI cards: label over value, colored dot as the class key."""
    cards_html = '<div class="kpi-grid">'
    cards_html += (
        '<div class="kpi-card kpi-total">'
        '<div class="kpi-label"><span class="kpi-dot" style="background: #5A74F5;"></span>'
        'Total vehicles</div>'
        f'<div class="kpi-value">{total}</div></div>'
    )
    for name, count in counts.items():
        css_class = CLASS_KPI_MAP.get(name, "kpi-default")
        dot = CLASS_DOT_COLORS.get(name, "#7C3AED")
        cards_html += (
            f'<div class="kpi-card {css_class}">'
            f'<div class="kpi-label"><span class="kpi-dot" style="background: {dot};"></span>{name}</div>'
            f'<div class="kpi-value">{count}</div></div>'
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


def section_header(icon_name: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-icon">{icon(icon_name)}</span>'
        f'<h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────── #
with st.sidebar:
    # ── Sidebar brand ───────────────────────────────────────────────────── #
    st.markdown(
        f'<div class="sidebar-brand">'
        f'<div class="brand-tile">{icon("logo", 21)}</div>'
        f'<div>'
        f'<div class="brand-name">Vehicle Counter</div>'
        f'<div class="brand-sub">YOLO detection &amp; counting</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border:none; border-top:1px solid rgba(255, 255, 255, 0.08); margin:0.3rem 0 0.6rem 0;">', unsafe_allow_html=True)

    # ── Model ──────────────────────────────────────────────────────────── #
    st.markdown(f'<p class="side-label">{icon("model")} Model</p>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Weights", MODELS, index=0,
        help="n = fastest, m = most accurate. Downloaded on first use.",
        label_visibility="collapsed",
    )

    # ── Device ─────────────────────────────────────────────────────────── #
    st.markdown(f'<p class="side-label">{icon("device")} Device</p>', unsafe_allow_html=True)
    # Auto-detected device first, then every CUDA GPU, then cpu/mps —
    # deduplicated so nothing ever appears twice.
    _auto_device = config.pick_device()
    device = st.selectbox(
        "Device",
        list(dict.fromkeys([_auto_device, *gpu_device_ids(), "cpu", "mps"])),
        index=0,
        format_func=lambda d: f"{d} (GPU)" if d.isdigit() else d,
        help="Auto-detected best device first; GPU ids appear when CUDA is available.",
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border:none; border-top:1px solid rgba(255, 255, 255, 0.08); margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── Detection Settings ─────────────────────────────────────────────── #
    st.markdown(f'<p class="side-label">{icon("detect")} Detection</p>', unsafe_allow_html=True)
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

    st.markdown('<hr style="border:none; border-top:1px solid rgba(255, 255, 255, 0.08); margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── Classes ────────────────────────────────────────────────────────── #
    st.markdown(f'<p class="side-label">{icon("classes")} Classes</p>', unsafe_allow_html=True)
    class_names = st.multiselect(
        "Count these classes",
        options=list(config.ALL_KNOWN_CLASSES.values()),
        default=list(config.VEHICLE_CLASSES.values()),
    )
    label_mode = st.radio(
        "Box captions", annotate.LABEL_MODES, index=0, horizontal=True,
        help="auto shrinks text to fit boxes.",
    )

    st.markdown('<hr style="border:none; border-top:1px solid rgba(255, 255, 255, 0.08); margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── Video ──────────────────────────────────────────────────────────── #
    st.markdown(f'<p class="side-label">{icon("video")} Video</p>', unsafe_allow_html=True)
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


# ─── App Header ─────────────────────────────────────────────────────────────── #
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-logo">{icon("logo", 27)}</div>
        <div class="app-title">
            <h1>Parking Lot Vehicle Counter</h1>
            <p>Detect, track and count vehicles in images and video — powered by YOLO11</p>
        </div>
        <div class="badge-row">
            <span class="badge">YOLO11</span>
            <span class="badge">ByteTrack</span>
            <span class="badge">OpenCV</span>
            <span class="badge">ROI Filtering</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Source Selection ───────────────────────────────────────────────────────── #
with st.container(border=True):
    section_header("source", "Choose Source")

    samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in MEDIA_EXTS)
    has_samples = len(samples) > 0

    # Source mode as interactive cards
    source_mode = None
    if has_samples:
        source_mode_choice = st.radio(
            "Source Selector",
            ["Bundled Samples", "Upload a File"],
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

if source_path is None:
    st.markdown(
        f'<div style="text-align:center; padding:2.75rem 2rem; background:#141419; '
        f'border-radius:12px; border:1px dashed rgba(90, 116, 245, 0.5);">'
        f'<div style="display:flex; justify-content:center; color:#94A3B8; '
        f'margin-bottom:0.7rem;">{icon("image", 26)}</div>'
        f'<p style="color:#475569; margin:0; font-size:0.9rem;">'
        f'Select a bundled sample or upload a file to begin</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not class_ids:
    st.warning("Pick at least one vehicle class in the sidebar.")
    st.stop()

kind = source_kind(source_path)
preview = first_frame(source_path)
if preview is None:
    st.error("Could not read that file. Check the format and try again.")
    st.stop()
height, width = preview.shape[:2]


# ─── Region of Interest ─────────────────────────────────────────────────────── #
with st.container(border=True):
    section_header("roi", "Region of Interest")

    # Confirmation for a save that happened on the previous run.
    if st.session_state.pop("roi_saved_flash", None):
        st.success("ROI polygon saved — pick it under **Saved polygon** below.")

    ROI_DIR.mkdir(parents=True, exist_ok=True)
    roi_files = sorted(ROI_DIR.glob("*.json"))
    roi_options = ["Whole frame", "Draw polygon", "Rectangle"]
    if roi_files:
        roi_options.append("Saved polygon")

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
        if btn_c1.button("Undo", disabled=not points, width="stretch"):
            points.pop()
            st.rerun()
        if btn_c2.button("Clear", disabled=not points, width="stretch"):
            points.clear()
            st.rerun()
        btn_c3.markdown(
            f'<div style="padding-top:0.5rem; color:#8A93A3; font-size:0.85rem;">'
            f'<strong>{len(points)}</strong> point(s)'
            f'{" — ready" if len(points) >= 3 else " — need 3 minimum"}</div>',
            unsafe_allow_html=True,
        )

        if len(points) >= 3:
            roi = ROI(points=list(points), name="drawn", normalized=True)
            if st.button("Save Drawn ROI Polygon", width="stretch"):
                saved = ROI(points=list(points), name="drawn_roi", normalized=True).save(
                    ROI_DIR / "drawn_roi.json"
                )
                # Clear the canvas and rerun so the "Saved polygon" option and
                # its file list pick up the new file immediately.
                points.clear()
                st.session_state["roi_saved_flash"] = saved.relative_to(ROOT).as_posix()
                st.rerun()

    elif roi_mode == "Rectangle":
        rect_c1, rect_c2 = st.columns(2)
        with rect_c1:
            left, right = st.slider("Horizontal", 0.0, 1.0, (0.0, 0.6), 0.01)
        with rect_c2:
            top, bottom = st.slider("Vertical", 0.0, 1.0, (0.15, 1.0), 0.01)
        if right > left and bottom > top:
            roi = ROI.from_rect(left, top, right, bottom, name="rectangle", normalized=True)
        else:
            st.warning("Rectangle has no area — widen the range.")

    elif roi_mode == "Saved polygon":
        if not roi_files:
            st.info("No saved polygons yet — draw one first, then save it.")
        else:
            picked_roi = st.selectbox("ROI file", roi_files, format_func=lambda p: p.name)
            try:
                roi = ROI.load(picked_roi)
            except Exception as exc:
                st.error(
                    f"Could not load `{picked_roi.name}` — the file may be "
                    f"corrupted or use an old format. Redraw and save it again."
                )
                st.caption(f"Details: {exc}")

    if roi is not None:
        roi_info_c1, roi_info_c2 = st.columns([3, 2])
        roi_info_c1.markdown(
            f'<div style="display:flex; align-items:center; gap:0.45rem; '
            f'background:#141419; border:1px solid rgba(255, 255, 255, 0.08); border-radius:8px; '
            f'padding:0.5rem 0.8rem; font-size:0.85rem; color:#475569;">'
            f'<span style="width:7px; height:7px; border-radius:50%; background:#5A74F5;"></span>'
            f'<strong style="color:#F5F6F7;">{roi.name}</strong>'
            f'&ensp;·&ensp;{len(roi.points)} points</div>',
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


# ─── Run Button ─────────────────────────────────────────────────────────────── #
run_c1, run_c2, run_c3 = st.columns([2, 2, 6])
with run_c1:
    run_btn = st.button("Run Detection", type="primary", width="stretch")
with run_c2:
    stop_btn = st.button("Stop", width="stretch")

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
    with st.spinner("Detecting vehicles..."):
        result = process_image(
            source_path, detector, roi, roi_rule, OUTPUT_DIR,
            save_json=True, label_mode=label_mode,
        )

    # ── Results Panel ──────────────────────────────────────────────────── #
    with st.container(border=True):
        section_header("chart", "Detection Results")
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
            f'Inference: <code>{result.elapsed_s * 1000:.0f} ms</code>'
            f' &nbsp;|&nbsp; Device: <code>{detector.device}</code>'
            f' &nbsp;|&nbsp; imgsz: <code>{detector.imgsz_for(preview)}</code>'
            f' &nbsp;|&nbsp; Saved: <code>{result.output_image.relative_to(ROOT)}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Details expander
        with st.expander("Raw detections (what YOLO returned)", expanded=False):
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
        dl_c1, dl_c2 = st.columns(2)
        dl_c1.download_button(
            "Download Annotated Image",
            result.output_image.read_bytes(),
            file_name=result.output_image.name,
            mime="image/jpeg",
            width="stretch",
        )
        if result.summary_json:
            dl_c2.download_button(
                "Download Counts JSON",
                result.summary_json.read_bytes(),
                file_name=result.summary_json.name,
                mime="application/json",
                width="stretch",
            )

else:
    # ── Video Processing ───────────────────────────────────────────────── #
    bar = st.progress(0.0, text="Processing video...")

    def on_progress(done: int, total: int | None) -> None:
        if total:
            bar.progress(
                min(done / total, 1.0),
                text=f"Frame {done} of {total}",
            )

    result = process_video(
        source_path, detector, roi, roi_rule, OUTPUT_DIR,
        use_tracking=use_tracking, min_track_hits=int(min_track_hits),
        stride=int(stride), max_frames=int(max_frames) or None,
        save_json=True, label_mode=label_mode,
        progress_cb=on_progress, show_progress=False,
    )
    bar.empty()

    with st.container(border=True):
        # ── Occupancy Section ──────────────────────────────────────────── #
        section_header("chart", "Occupancy (how full it gets)")
        occ_c1, occ_c2 = st.columns(2)
        occ_c1.metric("Peak vehicles in one frame", result.peak_occupancy)
        occ_c2.metric("Mean vehicles per frame", f"{result.mean_occupancy:.1f}")
        render_kpi_cards(result.peak_counts, result.peak_occupancy)

        # ── Unique Vehicles Section ────────────────────────────────────── #
        if use_tracking:
            section_header("refresh", "Unique Vehicles (how many came through)")
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

        # ── Video playback ─────────────────────────────────────────────── #
        section_header("video", "Annotated Video")
        st.video(result.output_video.read_bytes())
        st.markdown(
            f'<div class="inference-strip">'
            f'Processed <code>{result.frames_processed}</code> frames in '
            f'<code>{result.elapsed_s:.1f}s</code> '
            f'(<code>{result.fps_processing:.1f} FPS</code>) on '
            f'<code>{detector.device}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Occupancy chart ────────────────────────────────────────────── #
        if result.per_frame:
            section_header("trend", "Occupancy Over Time")
            chart_df = pd.DataFrame(result.per_frame)[["time_s", "occupancy"]].set_index("time_s")
            chart_df.columns = ["vehicles in frame"]
            st.area_chart(chart_df, color="#5A74F5")

        # ── Downloads ──────────────────────────────────────────────────── #
        section_header("download", "Downloads")
        dl_c1, dl_c2 = st.columns(2)
        dl_c1.download_button(
            "Download Annotated Video",
            result.output_video.read_bytes(),
            file_name=result.output_video.name,
            mime="video/mp4",
            width="stretch",
        )
        if result.per_frame_csv:
            dl_c2.download_button(
                "Download Per-Frame CSV",
                result.per_frame_csv.read_bytes(),
                file_name=result.per_frame_csv.name,
                mime="text/csv",
                width="stretch",
            )
        if result.summary_json:
            st.download_button(
                "Download Summary JSON",
                result.summary_json.read_bytes(),
                file_name=result.summary_json.name,
                mime="application/json",
                width="stretch",
            )
