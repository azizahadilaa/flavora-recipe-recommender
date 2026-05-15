"""
╔══════════════════════════════════════════════════════════════════╗
║         🍽️ FLAVORA — Smart Recipe Recommendation Dashboard       ║
║         ETS Data Mining 2026 | Content-Based Filtering          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FLAVORA – Smart Recipe Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (luxurious food-theme, 3D cards)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════
   FLAVORA — PREMIUM REDESIGN v2
   Rich textures · Editorial type · 3-D depth
══════════════════════════════════════════ */

/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=DM+Sans:wght@300;400;500;600&family=Cormorant+Garamond:ital,wght@1,300;1,500&display=swap');

/* ── CSS Variables ── */
:root {
  --gold:      #D4A017;
  --gold-lt:   #F0C040;
  --amber:     #E07B39;
  --rust:      #B5451B;
  --cream:     #FFF8EC;
  --cream-dk:  #F5E0B0;
  --dark:      #110800;
  --dark2:     #1E1005;
  --forest:    #162808;
  --glass:     rgba(255,248,236,0.055);
  --glass-bdr: rgba(212,160,23,0.28);
  --shadow-lg: 0 20px 60px rgba(0,0,0,0.65);
  --shadow-md: 0 8px 28px rgba(0,0,0,0.45);
  --shadow-glow: 0 0 30px rgba(212,160,23,0.18);
}

/* ── Global ── */
* { box-sizing: border-box; }

.stApp {
  background:
    radial-gradient(ellipse 80% 60% at 15% 20%, rgba(60,30,0,0.55) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 85% 75%, rgba(22,40,8,0.55) 0%, transparent 55%),
    linear-gradient(160deg, #110800 0%, #1E1005 35%, #162808 70%, #0B1503 100%);
  background-attachment: fixed !important;
  min-height: 100vh;
}

/* Subtle grid overlay for depth */
.stApp::before {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(212,160,23,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(212,160,23,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, #0D0500 0%, #170D02 50%, #0E1A03 100%) !important;
  border-right: 1px solid rgba(212,160,23,0.22) !important;
  box-shadow: 6px 0 40px rgba(0,0,0,0.7) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #F0DEB0 !important; }

/* ── Sidebar Logo ── */
.sidebar-logo {
  text-align: center;
  padding: 2rem 1rem 1.2rem;
  background: linear-gradient(180deg, rgba(212,160,23,0.08) 0%, transparent 100%);
  border-bottom: 1px solid rgba(212,160,23,0.18);
  margin-bottom: 1.5rem;
  position: relative;
}
.sidebar-logo::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 30%; right: 30%;
  height: 2px;
  background: linear-gradient(90deg, transparent, #D4A017, transparent);
}
.sidebar-logo .logo-icon {
  font-size: 2.8rem;
  display: block;
  margin-bottom: 0.4rem;
  filter: drop-shadow(0 0 12px rgba(212,160,23,0.6));
}
.sidebar-logo h1 {
  font-family: 'Playfair Display', serif !important;
  font-size: 2rem !important;
  font-weight: 900 !important;
  letter-spacing: 5px !important;
  background: linear-gradient(135deg, #F0C040 0%, #FEFAE0 45%, #E07B39 100%);
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  margin: 0 !important;
  line-height: 1 !important;
}
.sidebar-logo p {
  font-family: 'Cormorant Garamond', serif !important;
  font-style: italic !important;
  font-size: 0.75rem !important;
  color: rgba(212,160,23,0.55) !important;
  letter-spacing: 2.5px !important;
  margin-top: 0.35rem !important;
  text-transform: uppercase !important;
}

/* ── Main block ── */
.main .block-container {
  padding: 0 2rem 4rem !important;
  max-width: 1440px !important;
}

/* ── Hero / Page Header ── */
.page-header {
  text-align: center;
  padding: 3.5rem 2rem 2rem;
  position: relative;
  overflow: hidden;
}
.page-header::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 70% 80% at 50% 50%, rgba(212,160,23,0.07) 0%, transparent 70%);
  pointer-events: none;
}
.page-header .eyebrow {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: rgba(212,160,23,0.65);
  margin-bottom: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.7rem;
}
.page-header .eyebrow::before,
.page-header .eyebrow::after {
  content: '';
  display: inline-block;
  width: 32px; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(212,160,23,0.5));
}
.page-header .eyebrow::after {
  background: linear-gradient(90deg, rgba(212,160,23,0.5), transparent);
}
.page-header h1 {
  font-family: 'Playfair Display', serif !important;
  font-size: 3.8rem !important;
  font-weight: 900 !important;
  line-height: 1.05 !important;
  margin: 0 0 0.8rem !important;
  background: linear-gradient(135deg, #D4A017 0%, #F5E6C8 40%, #E07B39 80%, #D4A017 100%);
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  filter: drop-shadow(0 4px 20px rgba(212,160,23,0.25));
}
.page-header p {
  font-family: 'DM Sans', sans-serif;
  font-size: 1rem;
  color: rgba(245,230,200,0.5);
  font-weight: 300;
  letter-spacing: 0.8px;
  margin: 0;
}
.page-header .divider {
  display: flex; align-items: center; justify-content: center;
  gap: 0.8rem; margin: 1.4rem auto 0; width: 220px;
}
.page-header .divider span {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(212,160,23,0.45));
}
.page-header .divider span:last-child {
  background: linear-gradient(90deg, rgba(212,160,23,0.45), transparent);
}
.page-header .divider .diamond {
  width: 6px; height: 6px;
  background: #D4A017;
  transform: rotate(45deg);
  flex: none;
}

/* ── Section Title ── */
.section-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.45rem;
  font-weight: 700;
  color: #D4A017;
  margin: 2.2rem 0 1.2rem;
  display: flex; align-items: center; gap: 0.8rem;
  letter-spacing: 0.3px;
}
.section-title::before {
  content: '';
  display: block; width: 4px; height: 1.4em;
  background: linear-gradient(180deg, #F0C040, #E07B39);
  border-radius: 2px; flex-shrink: 0;
}
.section-title::after {
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(212,160,23,0.3), transparent);
}

/* ══════════════════════════════════════════
   RECIPE CARD — Elevated 3-D
══════════════════════════════════════════ */
.recipe-card {
  background: linear-gradient(160deg, #FFFBF0 0%, #FFF2D4 55%, #FDEAB8 100%);
  border-radius: 22px;
  overflow: hidden;
  margin-bottom: 1.6rem;
  border: 1px solid rgba(212,160,23,0.5);
  position: relative;
  /* 3-D perspective lift */
  transform: perspective(900px) rotateX(3deg) translateZ(0);
  box-shadow:
    0 2px 0 rgba(255,255,255,0.85) inset,           /* top highlight */
    0 -2px 0 rgba(180,120,0,0.25) inset,             /* bottom shadow */
    0 14px 40px rgba(0,0,0,0.55),                    /* drop shadow */
    0 4px 12px rgba(212,160,23,0.22),                /* gold glow */
    0 1px 2px rgba(0,0,0,0.6);                       /* edge crisp */
  transition: transform 0.35s cubic-bezier(.22,.68,0,1.2),
              box-shadow 0.35s ease;
  cursor: pointer;
}
/* Gold accent bar on top */
.recipe-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px; z-index: 2;
  background: linear-gradient(90deg, #B8860B, #F0C040, #E07B39, #F0C040, #B8860B);
}
/* Subtle sheen sweep */
.recipe-card::after {
  content: '';
  position: absolute; top: 0; left: -80%; width: 60%; height: 100%;
  background: linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.12) 50%, transparent 60%);
  pointer-events: none;
  transition: left 0.6s ease;
}
.recipe-card:hover {
  transform: perspective(900px) rotateX(0deg) translateY(-10px) scale(1.015);
  box-shadow:
    0 2px 0 rgba(255,255,255,0.9) inset,
    0 -2px 0 rgba(180,120,0,0.2) inset,
    0 30px 70px rgba(0,0,0,0.6),
    0 8px 24px rgba(212,160,23,0.35),
    0 0 50px rgba(212,160,23,0.1);
}
.recipe-card:hover::after { left: 120%; }

/* Card Image */
.card-img {
  width: 100%; height: 210px;
  object-fit: cover; display: block;
  transition: transform 0.5s ease;
}
.recipe-card:hover .card-img { transform: scale(1.04); }
.card-img-placeholder {
  width: 100%; height: 210px;
  background: linear-gradient(135deg, #3D2005 0%, #1A2C05 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 4rem;
}

/* Ribbon on card image */
.card-img-wrapper { position: relative; overflow: hidden; }

/* Card Body */
.card-body {
  padding: 1.1rem 1.35rem 1.35rem;
  background: linear-gradient(180deg, #FFFBF0 0%, #FFF2D4 100%);
}
.card-category {
  display: inline-flex; align-items: center; gap: 0.3rem;
  background: linear-gradient(135deg, #C8940F, #E07B39);
  color: #fff;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.65rem; font-weight: 600;
  padding: 0.22rem 0.75rem;
  border-radius: 30px;
  letter-spacing: 0.8px;
  margin-bottom: 0.6rem;
  text-transform: uppercase;
  box-shadow: 0 2px 8px rgba(212,160,23,0.35);
}
.card-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.08rem; font-weight: 700;
  color: #2E1200;
  margin: 0 0 0.55rem;
  line-height: 1.28;
  letter-spacing: -0.2px;
}
.card-divider {
  height: 1px; margin: 0.55rem 0;
  background: linear-gradient(90deg, rgba(212,160,23,0.4), rgba(224,123,57,0.2), transparent);
}
.card-meta {
  display: flex; gap: 0; flex-wrap: wrap;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.78rem; font-weight: 600;
  color: #5A2E00;
}
.card-meta-item {
  display: flex; align-items: center; gap: 0.25rem;
  padding: 0.2rem 0.6rem;
  background: rgba(212,160,23,0.1);
  border-radius: 6px; margin: 0.2rem 0.25rem 0.2rem 0;
  border: 1px solid rgba(212,160,23,0.2);
  color: #5A2E00;
}
.stars { color: #C8940F; font-size: 0.9rem; letter-spacing: 1px; }
.star-score { color: #7A4A00; font-size: 0.8rem; font-weight: 600; margin-left: 0.2rem; }

/* ══════════════════════════════════════════
   KPI / METRIC BOXES
══════════════════════════════════════════ */
.metric-box {
  background: linear-gradient(145deg, rgba(255,253,245,0.07), rgba(30,16,2,0.6));
  border: 1px solid rgba(212,160,23,0.25);
  border-radius: 18px;
  padding: 1.3rem 1rem;
  text-align: center;
  backdrop-filter: blur(16px);
  position: relative; overflow: hidden;
  box-shadow:
    0 0 0 1px rgba(255,255,255,0.04) inset,
    var(--shadow-md),
    var(--shadow-glow);
  transition: transform 0.3s cubic-bezier(.22,.68,0,1.2), box-shadow 0.3s ease;
}
.metric-box::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(212,160,23,0.6), transparent);
}
.metric-box:hover {
  transform: translateY(-6px) scale(1.02);
  box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 40px rgba(212,160,23,0.2);
}
.metric-icon {
  font-size: 2rem; margin-bottom: 0.5rem;
  filter: drop-shadow(0 0 8px rgba(212,160,23,0.5));
}
.metric-value {
  font-family: 'Playfair Display', serif;
  font-size: 2rem; font-weight: 900;
  background: linear-gradient(135deg, #F0C040, #D4A017);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; line-height: 1;
}
.metric-label {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.68rem; font-weight: 600;
  color: rgba(245,230,200,0.5);
  margin-top: 0.35rem;
  text-transform: uppercase; letter-spacing: 1.2px;
}

/* ══════════════════════════════════════════
   INGREDIENT TAGS
══════════════════════════════════════════ */
.ingredient-tag {
  display: inline-flex; align-items: center; gap: 0.3rem;
  background: linear-gradient(135deg, rgba(212,160,23,0.12), rgba(224,123,57,0.08));
  border: 1px solid rgba(212,160,23,0.3);
  border-radius: 30px;
  padding: 0.28rem 0.85rem;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.76rem; font-weight: 500;
  color: #D4A017;
  margin: 0.22rem;
  transition: all 0.22s ease;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.ingredient-tag:hover {
  background: linear-gradient(135deg, rgba(212,160,23,0.28), rgba(224,123,57,0.18));
  transform: translateY(-2px) scale(1.04);
  box-shadow: 0 4px 12px rgba(212,160,23,0.25);
  color: #F0C040;
}

/* ══════════════════════════════════════════
   STEP CARDS
══════════════════════════════════════════ */
.step-card {
  background: linear-gradient(135deg, rgba(255,253,245,0.05), rgba(212,160,23,0.04));
  border-left: 3px solid #D4A017;
  border-radius: 0 14px 14px 0;
  padding: 0.85rem 1.15rem;
  margin: 0.55rem 0;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.87rem;
  color: rgba(245,230,200,0.82);
  line-height: 1.65;
  box-shadow: 0 2px 10px rgba(0,0,0,0.2);
  transition: border-color 0.2s, background 0.2s;
}
.step-card:hover {
  border-color: #F0C040;
  background: linear-gradient(135deg, rgba(255,253,245,0.09), rgba(212,160,23,0.07));
}
.step-number {
  font-family: 'Playfair Display', serif;
  font-size: 0.95rem; font-weight: 700;
  color: #D4A017; margin-bottom: 0.3rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.step-number::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(212,160,23,0.3), transparent);
}

/* ══════════════════════════════════════════
   DETAIL PANEL (expander inner)
══════════════════════════════════════════ */
.detail-panel {
  background: linear-gradient(145deg, rgba(255,253,245,0.05), rgba(20,12,2,0.85));
  border: 1px solid rgba(212,160,23,0.18);
  border-radius: 18px;
  padding: 1.4rem 1.6rem;
  margin-top: 0.8rem;
  backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.nutrition-header {
  font-family: 'Playfair Display', serif;
  font-size: 1.15rem; font-weight: 700;
  color: #D4A017;
  margin-bottom: 0.7rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.nutrition-header::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(212,160,23,0.3), transparent);
}

/* ══════════════════════════════════════════
   TABS
══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,248,236,0.03) !important;
  border-bottom: 1px solid rgba(212,160,23,0.18) !important;
  gap: 0.3rem !important;
  padding: 0 0.5rem !important;
  border-radius: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.88rem !important; font-weight: 500 !important;
  color: rgba(245,230,200,0.45) !important;
  padding: 0.7rem 1.4rem !important;
  border-radius: 8px 8px 0 0 !important;
  transition: color 0.2s, background 0.2s !important;
  letter-spacing: 0.3px !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: rgba(245,230,200,0.8) !important;
  background: rgba(212,160,23,0.07) !important;
}
.stTabs [aria-selected="true"] {
  color: #D4A017 !important;
  background: rgba(212,160,23,0.1) !important;
  border-bottom: 2px solid #D4A017 !important;
}

/* ══════════════════════════════════════════
   INPUTS & CONTROLS
══════════════════════════════════════════ */
.stTextInput > div > div > input {
  background: rgba(255,248,236,0.06) !important;
  border: 1px solid rgba(212,160,23,0.3) !important;
  border-radius: 12px !important;
  color: #F5E6C8 !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important;
  padding: 0.55rem 0.9rem !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2) inset !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
  border-color: #D4A017 !important;
  box-shadow: 0 0 0 3px rgba(212,160,23,0.15), 0 2px 8px rgba(0,0,0,0.2) inset !important;
}
.stTextInput > div > div > input::placeholder { color: rgba(245,230,200,0.3) !important; }

/* ── Button ── */
.stButton > button {
  background: linear-gradient(135deg, #C8940F 0%, #E07B39 100%) !important;
  color: #1A0A00 !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 700 !important; font-size: 0.88rem !important;
  border: none !important; border-radius: 14px !important;
  padding: 0.65rem 1.8rem !important;
  letter-spacing: 1.5px !important; text-transform: uppercase !important;
  box-shadow: 0 6px 22px rgba(212,160,23,0.4), 0 2px 4px rgba(0,0,0,0.4),
              0 1px 0 rgba(255,255,255,0.25) inset !important;
  transition: all 0.28s cubic-bezier(.22,.68,0,1.2) !important;
  position: relative !important; overflow: hidden !important;
}
.stButton > button:hover {
  transform: translateY(-3px) scale(1.02) !important;
  box-shadow: 0 12px 36px rgba(212,160,23,0.55), 0 4px 8px rgba(0,0,0,0.5) !important;
  background: linear-gradient(135deg, #D4A017 0%, #E8893F 100%) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.99) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: rgba(255,253,245,0.03) !important;
  border: 1px solid rgba(212,160,23,0.18) !important;
  border-radius: 14px !important;
  margin-top: 0.5rem !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 0.85rem !important;
  color: rgba(212,160,23,0.8) !important;
  padding: 0.7rem 1rem !important;
}
[data-testid="stExpander"] summary:hover {
  background: rgba(212,160,23,0.07) !important;
  color: #D4A017 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0D0500; }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #D4A017, #E07B39);
  border-radius: 3px;
}

/* ── Footer ── */
.footer {
  text-align: center; padding: 2.5rem 1rem;
  font-family: 'Cormorant Garamond', serif; font-style: italic;
  color: rgba(212,160,23,0.3); font-size: 0.85rem;
  border-top: 1px solid rgba(212,160,23,0.1); margin-top: 4rem;
  letter-spacing: 1.5px;
}

/* ── Typography overrides ── */
h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: #F5E6C8 !important; }
.stMarkdown p { color: rgba(245,230,200,0.75); font-family: 'DM Sans', sans-serif; line-height: 1.7; }
p, li { color: rgba(245,230,200,0.75); }

/* ── Plotly transparent bg ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .bg { background: transparent !important; }

/* ── Slider accent color ── */
[data-testid="stSlider"] [role="slider"] {
  background: #D4A017 !important;
  box-shadow: 0 0 10px rgba(212,160,23,0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def parse_instructions(raw):
    """Parse R-style c("...","...") instructions into a clean list."""
    if pd.isna(raw) or raw == "":
        return []
    raw = str(raw)
    # Remove leading c( and trailing )
    raw = re.sub(r'^c\(', '', raw.strip())
    raw = re.sub(r'\)$', '', raw.strip())
    # Extract quoted strings
    steps = re.findall(r'"(.*?)"(?:,\s*|$)', raw)
    if not steps:
        # Fallback: split by period
        steps = [s.strip() for s in raw.split('.') if len(s.strip()) > 5]
    return steps


def parse_ingredients(raw):
    """Return cleaned ingredient list."""
    if pd.isna(raw) or raw == "":
        return []
    parts = [p.strip() for p in str(raw).split(',')]
    return [p for p in parts if p]


def star_rating(rating):
    """Generate HTML star icons for a rating."""
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    stars = '★' * full + ('½' if half else '') + '☆' * empty
    return f'<span class="stars">{stars} {rating:.1f}</span>'


def format_time(minutes):
    if pd.isna(minutes) or minutes == 0:
        return "N/A"
    minutes = int(minutes)
    if minutes < 60:
        return f"{minutes} min"
    h, m = divmod(minutes, 60)
    return f"{h}h {m}m" if m else f"{h}h"


# ─────────────────────────────────────────────
#  DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('clean_recipes.csv')
    # Ensure correct types
    df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce').fillna(0)
    df['ProteinContent'] = pd.to_numeric(df['ProteinContent'], errors='coerce').fillna(0)
    df['FatContent'] = pd.to_numeric(df['FatContent'], errors='coerce').fillna(0)
    df['CarbohydrateContent'] = pd.to_numeric(df['CarbohydrateContent'], errors='coerce').fillna(0)
    df['AggregatedRating'] = pd.to_numeric(df['AggregatedRating'], errors='coerce').fillna(0)
    df['ReviewCount'] = pd.to_numeric(df['ReviewCount'], errors='coerce').fillna(0)
    df['TotalTime_Minutes'] = pd.to_numeric(df['TotalTime_Minutes'], errors='coerce').fillna(0)
    df['RecipeServings'] = pd.to_numeric(df['RecipeServings'], errors='coerce').fillna(1)
    df['combined_features'] = (
        df['Features'].fillna('') + ' ' +
        df['RecipeCategory'].fillna('') + ' ' +
        df['Keywords'].fillna('')
    )
    return df


@st.cache_resource(show_spinner=False)
def build_model(_df):
    """Build TF-IDF matrix only (sparse) — avoids the 1.48 GiB full cosine matrix."""
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(_df['combined_features'])
    return tfidf_matrix  # sparse matrix, hemat RAM!


def get_recommendations(df, tfidf_matrix, query, n=6,
                        max_cal=None, max_time=None,
                        min_rating=None, categories=None):
    """Optimized Content-based filtering: Memory-safe for large datasets."""
    
    # ── 1. Filter Tahap Awal (Agar pencarian hanya pada data yang relevan) ──
    mask = pd.Series([True] * len(df), index=df.index)
    if max_cal:
        mask &= df['Calories'] <= max_cal
    if max_time:
        mask &= (df['TotalTime_Minutes'] <= max_time) | (df['TotalTime_Minutes'] == 0)
    if min_rating:
        mask &= df['AggregatedRating'] >= min_rating
    if categories:
        mask &= df['RecipeCategory'].isin(categories)

    # Simpan index asli yang lolos filter
    filtered_indices = df.index[mask].tolist()
    if not filtered_indices:
        return pd.DataFrame()

    # ── 2. Cari Seed (Resep acuan berdasarkan input user) ──
    matches = df[df['Name'].str.contains(query, case=False, na=False)]
    also_ingr = df[df['RecipeIngredientParts'].str.contains(query, case=False, na=False)]
    also_kw = df[df['Keywords'].str.contains(query, case=False, na=False)]
    seeds = pd.concat([matches, also_ingr, also_kw]).drop_duplicates()

    # Jika tidak ada resep acuan yang ketemu, tampilkan hasil filter seadanya (head)
    if seeds.empty:
        return df.loc[filtered_indices].head(n)

    # Gunakan resep pertama yang paling cocok sebagai acuan perhitungan similarity
    seed_idx = seeds.index[0]
    
    # ── 3. Hitung Similarity Secara Hemat Memori ──
    # Ambil vektor resep acuan
    seed_vector = tfidf_matrix[seed_idx] 
    
    # HANYA ambil baris matriks yang lolos filter saja (Ini rahasianya biar gak Error RAM)
    filtered_matrix = tfidf_matrix[filtered_indices]
    
    # Hitung similarity hanya pada subset data yang sudah difilter
    from sklearn.metrics.pairwise import cosine_similarity
    sim_scores = cosine_similarity(seed_vector, filtered_matrix).flatten()

    # ── 4. Sorting & Hasil Akhir ──
    # Urutkan dari skor tertinggi (paling mirip)
    top_local_indices = sim_scores.argsort()[::-1]
    
    final_indices = []
    for i in top_local_indices:
        real_idx = filtered_indices[i]
        # Jangan masukkan resep acuan itu sendiri ke dalam rekomendasi
        if real_idx != seed_idx:
            final_indices.append(real_idx)
        if len(final_indices) >= n:
            break

    return df.loc[final_indices] if final_indices else df.loc[filtered_indices].head(n)

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#F5E6C8'),
    title_font=dict(family='Playfair Display', color='#D4A017', size=16),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#F5E6C8')),
    colorway=['#D4A017', '#E07B39', '#C0392B', '#2D5016', '#8B6914', '#F5C842'],
)

GOLD_COLORS = ['#D4A017', '#E07B39', '#C0392B', '#8B6914', '#F5C842',
               '#A0522D', '#CD853F', '#DEB887', '#B8860B', '#DAA520']


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🍽</span>
        <h1>FLAVORA</h1>
        <p>Smart Recipe Recommender</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Search & Filters")

    search_query = st.text_input(
        "Search recipe / ingredient",
        placeholder="e.g. chicken, pasta, garlic…",
        help="Enter a recipe name or ingredient"
    )

    st.markdown("---")
    st.markdown("**🍽 Recipe Filters**")

    max_calories = st.slider("Max Calories (kcal)", 50, 3000, 1500, 50)
    max_time = st.slider("Max Cook Time (minutes)", 5, 600, 120, 5)
    min_rating = st.slider("Minimum Rating ⭐", 1.0, 5.0, 3.5, 0.5)
    top_n = st.slider("Number of Recommendations", 3, 12, 6)

    st.markdown("---")
    st.markdown("**📂 Category Filter**")

    # Will populate after data load
    category_placeholder = st.empty()

    st.markdown("---")
    search_btn = st.button("✨ Find Recipes", use_container_width=True)

    st.markdown("""
    <div style='margin-top:2rem; padding-top:1rem; border-top:1px solid rgba(212,160,23,0.15);
                text-align:center; font-family:DM Sans; font-size:0.72rem;
                color:rgba(212,160,23,0.45);'>
        ETS Data Mining 2026<br>Content-Based Filtering<br>TF-IDF · Cosine Similarity
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────

with st.spinner("🍳 Preparing the kitchen…"):
    df = load_data()

with st.spinner("🧠 Building recommendation model…"):
    tfidf_matrix = build_model(df)

# Populate category filter after data load
all_categories = sorted(df['RecipeCategory'].dropna().unique().tolist())
selected_categories = category_placeholder.multiselect(
    "Filter by Category",
    options=all_categories,
    default=[],
    help="Leave empty to include all categories"
)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🏠 Discover", "📊 Analytics", "ℹ️ About"])


# ════════════════════════════════════════════════════════
#  TAB 1 — DISCOVER (RECOMMENDATIONS)
# ════════════════════════════════════════════════════════

with tab1:
    # ── Header ──
    st.markdown("""
    <div class="page-header">
        <div class="eyebrow">✦ AI-Powered · 14,000+ Recipes · Content-Based Filtering ✦</div>
        <h1>Discover Your Next<br>Favourite Dish</h1>
        <p>Tell us what you have — we'll find the perfect recipe for you</p>
        <div class="divider"><span></span><div class="diamond"></div><span></span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Global KPI Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("🍽️", f"{len(df):,}", "Total Recipes"),
        ("🗂️", str(df['RecipeCategory'].nunique()), "Categories"),
        ("⭐", f"{df['AggregatedRating'].mean():.1f}", "Avg Rating"),
        ("🔥", f"{int(df['Calories'].median())}", "Median Kcal"),
        ("⏱️", f"{int(df['TotalTime_Minutes'][df['TotalTime_Minutes']>0].median())} min", "Median Time"),
    ]
    for col, (icon, val, label) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendation Section ──
    if search_btn or search_query:
        if not search_query.strip():
            st.warning("⚠️ Please enter a recipe name or ingredient in the sidebar.")
        else:
            with st.spinner(f"🔍 Finding the best recipes for **{search_query}**…"):
                cats = selected_categories if selected_categories else None
                results = get_recommendations(
                    df, tfidf_matrix,
                    query=search_query,
                    n=top_n,
                    max_cal=max_calories,
                    max_time=max_time,
                    min_rating=min_rating,
                    categories=cats
                )

            if results.empty:
                st.error("😔 No recipes found matching your filters. Try relaxing the constraints.")
            else:
                st.markdown(f"""
                <div class="section-title">
                    ✨ Top {len(results)} Recommendations for "{search_query}"
                </div>
                """, unsafe_allow_html=True)

                # ── Recipe Cards Grid ──
                cols_per_row = 3
                rows = [results.iloc[i:i+cols_per_row] for i in range(0, len(results), cols_per_row)]

                for row_df in rows:
                    cols = st.columns(cols_per_row)
                    for col, (_, recipe) in zip(cols, row_df.iterrows()):
                        with col:
                            img_url = str(recipe.get('Images', '')).strip()
                            # 3D Card
                            if img_url and img_url.startswith('http'):
                                img_html = f'<img class="card-img" src="{img_url}" onerror="this.parentElement.innerHTML=\'<div class=card-img-placeholder>🍴</div>\'">'
                            else:
                                img_html = '<div class="card-img-placeholder">🍴</div>'

                            rating = recipe.get('AggregatedRating', 0)
                            stars_html = star_rating(rating)
                            cal = recipe.get('Calories', 0)
                            t = format_time(recipe.get('TotalTime_Minutes', 0))
                            category = recipe.get('RecipeCategory', 'Recipe')
                            servings = int(recipe.get('RecipeServings', 1))

                            st.markdown(f"""
                            <div class="recipe-card">
                                {img_html}
                                <div class="card-body">
                                    <div class="card-category">🍴 {category}</div>
                                    <div class="card-title">{recipe['Name']}</div>
                                    <div class="card-divider"></div>
                                    <div class="card-meta">
                                        <div class="card-meta-item">🔥 {cal:.0f} kcal</div>
                                        <div class="card-meta-item">⏱ {t}</div>
                                        <div class="card-meta-item">🍴 {servings} serv</div>
                                    </div>
                                    <div style="margin-top:0.6rem">
                                        <span class="stars">{'★' * int(rating)}{'☆' * (5 - int(rating))}</span>
                                        <span class="star-score">{rating:.1f}</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # ── Expandable Detail Panel ──
                            with st.expander(f"📋 View Full Recipe"):
                                # Image (larger)
                                if img_url and img_url.startswith('http'):
                                    st.image(img_url, use_container_width=True)

                                # ── INGREDIENTS ──
                                st.markdown("<div class='nutrition-header'>🥕 Ingredients</div>", unsafe_allow_html=True)
                                ingredients = parse_ingredients(recipe.get('RecipeIngredientParts', ''))
                                if ingredients:
                                    tags_html = ''.join(
                                        f'<span class="ingredient-tag">• {ing}</span>'
                                        for ing in ingredients
                                    )
                                    st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("_No ingredient data available._")

                                st.markdown("<br>", unsafe_allow_html=True)

                                # ── NUTRITION RADAR ──
                                st.markdown("<div class='nutrition-header'>📊 Nutrition Profile</div>", unsafe_allow_html=True)
                                nutrient_vals = [
                                    recipe.get('ProteinContent', 0),
                                    recipe.get('FatContent', 0),
                                    recipe.get('CarbohydrateContent', 0),
                                    recipe.get('Calories', 0) / 50,  # scaled
                                ]
                                nutrient_labels = ['Protein (g)', 'Fat (g)', 'Carbs (g)', 'Energy (×50kcal)']

                                radar_fig = go.Figure(go.Scatterpolar(
                                    r=nutrient_vals + [nutrient_vals[0]],
                                    theta=nutrient_labels + [nutrient_labels[0]],
                                    fill='toself',
                                    fillcolor='rgba(212,160,23,0.2)',
                                    line=dict(color='#D4A017', width=2.5),
                                    marker=dict(color='#E07B39', size=7)
                                ))
                                radar_fig.update_layout(
                                    **PLOTLY_LAYOUT,
                                    polar=dict(
                                        bgcolor='rgba(0,0,0,0)',
                                        radialaxis=dict(visible=True, gridcolor='rgba(212,160,23,0.15)',
                                                        tickfont=dict(color='#F5E6C8', size=9)),
                                        angularaxis=dict(gridcolor='rgba(212,160,23,0.15)',
                                                         tickfont=dict(color='#F5E6C8', size=10))
                                    ),
                                    height=280, margin=dict(t=20, b=20, l=20, r=20)
                                )
                                st.plotly_chart(radar_fig, use_container_width=True, key=f"radar_{recipe['Name']}")

                                # Macros bar
                                macro_fig = go.Figure()
                                macro_fig.add_trace(go.Bar(
                                    x=['Protein', 'Fat', 'Carbs'],
                                    y=[recipe.get('ProteinContent', 0),
                                       recipe.get('FatContent', 0),
                                       recipe.get('CarbohydrateContent', 0)],
                                    marker=dict(
                                        color=['#D4A017', '#E07B39', '#C0392B'],
                                        line=dict(color='rgba(0,0,0,0.3)', width=1)
                                    ),
                                    text=[f"{v:.1f}g" for v in [
                                        recipe.get('ProteinContent', 0),
                                        recipe.get('FatContent', 0),
                                        recipe.get('CarbohydrateContent', 0)
                                    ]],
                                    textposition='outside',
                                    textfont=dict(color='#F5E6C8', size=11)
                                ))
                                macro_fig.update_layout(
                                    **PLOTLY_LAYOUT,
                                    height=200,
                                    margin=dict(t=10, b=10, l=10, r=10),
                                    xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                                    yaxis=dict(gridcolor='rgba(212,160,23,0.1)', tickfont=dict(color='#F5E6C8'))
                                )
                                st.plotly_chart(macro_fig, use_container_width=True, key=f"macro_{recipe['Name']}")

                                # ── INSTRUCTIONS ──
                                st.markdown("<div class='nutrition-header'>👨‍🍳 Instructions</div>", unsafe_allow_html=True)
                                steps = parse_instructions(recipe.get('RecipeInstructions', ''))
                                if steps:
                                    for i, step in enumerate(steps, 1):
                                        st.markdown(f"""
                                        <div class="step-card">
                                            <div class="step-number">Step {i}</div>
                                            {step.strip()}
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("Instructions not available for this recipe.")

    else:
        # ── Default: show popular recipes ──
        st.markdown('<div class="section-title">⭐ Popular Recipes to Explore</div>', unsafe_allow_html=True)
        popular = df.nlargest(6, 'ReviewCount')

        cols_per_row = 3
        rows = [popular.iloc[i:i+cols_per_row] for i in range(0, len(popular), cols_per_row)]
        for row_df in rows:
            cols = st.columns(cols_per_row)
            for col, (_, recipe) in zip(cols, row_df.iterrows()):
                with col:
                    img_url = str(recipe.get('Images', '')).strip()
                    if img_url and img_url.startswith('http'):
                        img_html = f'<img class="card-img" src="{img_url}" onerror="this.parentElement.innerHTML=\'<div class=card-img-placeholder>🍴</div>\'">'
                    else:
                        img_html = '<div class="card-img-placeholder">🍴</div>'

                    rating = recipe.get('AggregatedRating', 0)
                    stars_html = star_rating(rating)
                    cal = recipe.get('Calories', 0)
                    t = format_time(recipe.get('TotalTime_Minutes', 0))
                    category = recipe.get('RecipeCategory', 'Recipe')
                    reviews = int(recipe.get('ReviewCount', 0))

                    st.markdown(f"""
                    <div class="recipe-card">
                        {img_html}
                        <div class="card-body">
                            <div class="card-category">🍴 {category}</div>
                            <div class="card-title">{recipe['Name']}</div>
                            <div class="card-divider"></div>
                            <div class="card-meta">
                                <div class="card-meta-item">🔥 {cal:.0f} kcal</div>
                                <div class="card-meta-item">⏱ {t}</div>
                                <div class="card-meta-item">💬 {reviews:,}</div>
                            </div>
                            <div style="margin-top:0.6rem">
                                <span class="stars">{'★' * int(rating)}{'☆' * (5 - int(rating))}</span>
                                <span class="star-score">{rating:.1f}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📋 View Full Recipe"):
                        if img_url and img_url.startswith('http'):
                            st.image(img_url, use_container_width=True)

                        st.markdown("<div class='nutrition-header'>🥕 Ingredients</div>", unsafe_allow_html=True)
                        ingredients = parse_ingredients(recipe.get('RecipeIngredientParts', ''))
                        if ingredients:
                            tags_html = ''.join(f'<span class="ingredient-tag">• {ing}</span>' for ing in ingredients)
                            st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("<div class='nutrition-header'>📊 Nutrition Profile</div>", unsafe_allow_html=True)

                        nutrient_vals = [
                            recipe.get('ProteinContent', 0),
                            recipe.get('FatContent', 0),
                            recipe.get('CarbohydrateContent', 0),
                            recipe.get('Calories', 0) / 50,
                        ]
                        nutrient_labels = ['Protein (g)', 'Fat (g)', 'Carbs (g)', 'Energy (×50kcal)']

                        radar_fig = go.Figure(go.Scatterpolar(
                            r=nutrient_vals + [nutrient_vals[0]],
                            theta=nutrient_labels + [nutrient_labels[0]],
                            fill='toself',
                            fillcolor='rgba(212,160,23,0.2)',
                            line=dict(color='#D4A017', width=2.5),
                            marker=dict(color='#E07B39', size=7)
                        ))
                        radar_fig.update_layout(
                            **PLOTLY_LAYOUT,
                            polar=dict(
                                bgcolor='rgba(0,0,0,0)',
                                radialaxis=dict(visible=True, gridcolor='rgba(212,160,23,0.15)',
                                                tickfont=dict(color='#F5E6C8', size=9)),
                                angularaxis=dict(gridcolor='rgba(212,160,23,0.15)',
                                                 tickfont=dict(color='#F5E6C8', size=10))
                            ),
                            height=260, margin=dict(t=20, b=20, l=20, r=20)
                        )
                        st.plotly_chart(radar_fig, use_container_width=True, key=f"pop_radar_{recipe['Name']}")

                        st.markdown("<div class='nutrition-header'>👨‍🍳 Instructions</div>", unsafe_allow_html=True)
                        steps = parse_instructions(recipe.get('RecipeInstructions', ''))
                        if steps:
                            for i, step in enumerate(steps, 1):
                                st.markdown(f"""
                                <div class="step-card">
                                    <div class="step-number">Step {i}</div>
                                    {step.strip()}
                                </div>
                                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="page-header">
        <div class="eyebrow">✦ Data Insights · Trends · Nutrition ✦</div>
        <h1>Market Insights<br>& Analytics</h1>
        <p>Explore popularity trends, nutritional patterns, and recipe distributions</p>
        <div class="divider"><span></span><div class="diamond"></div><span></span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Rating Distribution & Category Distribution ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">⭐ Rating Distribution</div>', unsafe_allow_html=True)
        rating_counts = df['AggregatedRating'].value_counts().sort_index()
        fig_rating = go.Figure(go.Bar(
            x=rating_counts.index.astype(str),
            y=rating_counts.values,
            marker=dict(
                color=rating_counts.values,
                colorscale=[[0, '#4A2500'], [0.5, '#D4A017'], [1, '#F5C842']],
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=rating_counts.values,
            textposition='outside',
            textfont=dict(color='#F5E6C8')
        ))
        fig_rating.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis=dict(title='Rating', gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='Count', gridcolor='rgba(212,160,23,0.1)'),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">🗂️ Top 10 Categories</div>', unsafe_allow_html=True)
        top_cats = df['RecipeCategory'].value_counts().head(10)
        fig_cat = go.Figure(go.Bar(
            x=top_cats.values,
            y=top_cats.index,
            orientation='h',
            marker=dict(
                color=GOLD_COLORS[:10],
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=top_cats.values,
            textposition='outside',
            textfont=dict(color='#F5E6C8')
        ))
        fig_cat.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            yaxis=dict(autorange='reversed', gridcolor='rgba(0,0,0,0)'),
            xaxis=dict(gridcolor='rgba(212,160,23,0.1)'),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Row 2: Calorie Distribution & Nutrition Comparison ──
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">🔥 Calorie Distribution</div>', unsafe_allow_html=True)
        cal_data = df[df['Calories'] < 3000]['Calories']
        fig_cal = go.Figure(go.Histogram(
            x=cal_data,
            nbinsx=50,
            marker=dict(
                color='rgba(212,160,23,0.75)',
                line=dict(color='rgba(212,160,23,0.3)', width=0.5)
            )
        ))
        fig_cal.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            xaxis=dict(title='Calories (kcal)', gridcolor='rgba(212,160,23,0.1)'),
            yaxis=dict(title='Frequency', gridcolor='rgba(212,160,23,0.1)'),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">🥗 Avg Macronutrients by Category</div>', unsafe_allow_html=True)
        macro_cat = df.groupby('RecipeCategory')[['ProteinContent', 'FatContent', 'CarbohydrateContent']].mean()
        top10 = df['RecipeCategory'].value_counts().head(8).index
        macro_cat = macro_cat.loc[macro_cat.index.isin(top10)]
        fig_macro = go.Figure()
        for nutrient, color in zip(['ProteinContent', 'FatContent', 'CarbohydrateContent'],
                                    ['#D4A017', '#E07B39', '#C0392B']):
            fig_macro.add_trace(go.Bar(
                name=nutrient.replace('Content', ''),
                x=macro_cat.index,
                y=macro_cat[nutrient],
                marker=dict(color=color, opacity=0.85)
            ))
        fig_macro.update_layout(
            **PLOTLY_LAYOUT,
            barmode='group',
            height=300,
            xaxis=dict(tickangle=-30, gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='grams', gridcolor='rgba(212,160,23,0.1)'),
            margin=dict(t=10, b=60, l=10, r=10)
        )
        st.plotly_chart(fig_macro, use_container_width=True)

    # ── Row 3: Top Rated Recipes (Bubble Chart) & Cook Time Distribution ──
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="section-title">🏆 Top Recipes by Reviews</div>', unsafe_allow_html=True)
        top_reviewed = df.nlargest(15, 'ReviewCount')[['Name', 'ReviewCount', 'AggregatedRating', 'Calories']].copy()
        top_reviewed['ShortName'] = top_reviewed['Name'].str[:28] + '…'
        fig_bubble = go.Figure(go.Scatter(
            x=top_reviewed['AggregatedRating'],
            y=top_reviewed['ReviewCount'],
            mode='markers+text',
            text=top_reviewed['ShortName'],
            textposition='top center',
            textfont=dict(size=8, color='#F5E6C8'),
            marker=dict(
                size=top_reviewed['Calories'] / 30,
                color=top_reviewed['ReviewCount'],
                colorscale=[[0, '#4A2500'], [0.5, '#D4A017'], [1, '#F5C842']],
                showscale=True,
                colorbar=dict(title=dict(text='Reviews', font=dict(color='#D4A017')), tickfont=dict(color='#F5E6C8')),
                line=dict(color='rgba(212,160,23,0.5)', width=1.5),
                opacity=0.85
            )
        ))
        fig_bubble.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            xaxis=dict(title='Rating', gridcolor='rgba(212,160,23,0.1)'),
            yaxis=dict(title='Review Count', gridcolor='rgba(212,160,23,0.1)'),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    with col_f:
        st.markdown('<div class="section-title">⏱️ Cook Time Distribution</div>', unsafe_allow_html=True)
        time_data = df[(df['TotalTime_Minutes'] > 0) & (df['TotalTime_Minutes'] < 300)]['TotalTime_Minutes']
        bins = [0, 15, 30, 60, 90, 120, 180, 300]
        labels = ['< 15m', '15-30m', '30-60m', '60-90m', '90-120m', '120-180m', '180m+']
        time_bins = pd.cut(time_data, bins=bins, labels=labels)
        time_counts = time_bins.value_counts().sort_index()
        fig_time = go.Figure(go.Bar(
            x=time_counts.index,
            y=time_counts.values,
            marker=dict(
                color=time_counts.values,
                colorscale=[[0, '#2D5016'], [0.5, '#D4A017'], [1, '#E07B39']],
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=time_counts.values,
            textposition='outside',
            textfont=dict(color='#F5E6C8')
        ))
        fig_time.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            xaxis=dict(title='Cook Time', gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='Count', gridcolor='rgba(212,160,23,0.1)'),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Row 4: Calories vs Rating Scatter & Pie Chart ──
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<div class="section-title">📈 Calories vs Rating</div>', unsafe_allow_html=True)
        scatter_raw = df[(df['Calories'] > 0) & (df['Calories'] < 3000) & (df['AggregatedRating'] > 0)].copy()
        scatter_df = scatter_raw.sample(min(1500, len(scatter_raw)), random_state=42)
        fig_scatter = go.Figure(go.Scatter(
            x=scatter_df['Calories'].tolist(),
            y=scatter_df['AggregatedRating'].tolist(),
            mode='markers',
            marker=dict(
                color=scatter_df['AggregatedRating'].tolist(),
                colorscale=[[0, '#4A2500'], [0.4, '#C0392B'], [0.7, '#D4A017'], [1, '#F5C842']],
                size=6,
                opacity=0.7,
                showscale=True,
                colorbar=dict(
                    title=dict(text='Rating', font=dict(color='#D4A017', size=11)),
                    tickfont=dict(color='#F5E6C8', size=10),
                    thickness=12,
                ),
                line=dict(color='rgba(0,0,0,0.15)', width=0.5)
            ),
            text=scatter_df['Name'].tolist(),
            hovertemplate='<b>%{text}</b><br>Calories: %{x:.0f} kcal<br>Rating: %{y}<extra></extra>'
        ))
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            height=340,
            xaxis=dict(title='Calories (kcal)', gridcolor='rgba(212,160,23,0.12)',
                       tickfont=dict(color='#F5E6C8')),
            yaxis=dict(title='Rating', gridcolor='rgba(212,160,23,0.12)',
                       tickfont=dict(color='#F5E6C8')),
            margin=dict(t=20, b=40, l=50, r=60)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_h:
        st.markdown('<div class="section-title">🥧 Category Share (Top 8)</div>', unsafe_allow_html=True)
        pie_data = df['RecipeCategory'].value_counts().head(8)
        fig_pie = go.Figure(go.Pie(
            labels=pie_data.index,
            values=pie_data.values,
            hole=0.45,
            marker=dict(
                colors=GOLD_COLORS[:8],
                line=dict(color='#1A0A00', width=2)
            ),
            textfont=dict(color='#F5E6C8', size=11),
            insidetextfont=dict(color='#1A0A00'),
        ))
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            annotations=[dict(text='Categories', x=0.5, y=0.5, font_size=13,
                               showarrow=False, font=dict(color='#D4A017'))],
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════════════════════

with tab3:
    st.markdown("""
    <div class="page-header">
        <div class="eyebrow">✦ ETS Data Mining 2026 ✦</div>
        <h1>About FLAVORA</h1>
        <p>Smart Recipe Recommendation System · TF-IDF × Cosine Similarity</p>
        <div class="divider"><span></span><div class="diamond"></div><span></span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        # ── Panel wrapper open ──
        st.markdown('<div class="detail-panel">', unsafe_allow_html=True)

        st.markdown('<div class="nutrition-header">🎯 Project Overview</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:rgba(245,230,200,0.85); line-height:1.9; font-family:\'DM Sans\'; font-size:0.95rem;">'
            'FLAVORA adalah sistem rekomendasi resep cerdas berbasis '
            '<strong style="color:#D4A017">Content-Based Filtering</strong> '
            'yang dikembangkan untuk membantu pengguna menemukan resep yang sesuai dengan bahan, preferensi nutrisi, '
            'dan gaya hidup mereka. Proyek ini merupakan bagian dari tugas ETS mata kuliah Data Mining 2026.'
            '</p>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="nutrition-header" style="margin-top:1.2rem">⚙️ Methodology</div>', unsafe_allow_html=True)

        for num, title, desc in [
            ("1", "Feature Engineering",
             "Menggabungkan kolom <em>RecipeIngredientParts</em>, <em>RecipeCategory</em>, dan <em>Keywords</em> "
             "menjadi satu representasi teks (combined_features) untuk setiap resep."),
            ("2", "TF-IDF Vectorization",
             "Mengubah teks combined_features menjadi vektor numerik menggunakan TF-IDF "
             "(Term Frequency–Inverse Document Frequency) dengan max 10,000 fitur dan stop-words bahasa Inggris."),
            ("3", "Cosine Similarity",
             "Menghitung kemiripan antar resep secara on-demand (1 baris seed × seluruh dataset) "
             "sehingga hemat memori namun tetap akurat."),
            ("4", "Filtered Recommendation",
             "Pengguna dapat memfilter berdasarkan kalori, waktu masak, rating, dan kategori. "
             "Sistem mengurutkan resep berdasarkan skor similarity tertinggi dalam subset yang difilter."),
        ]:
            st.markdown(
                f'<div class="step-card">'
                f'<div class="step-number">Step {num} — {title}</div>'
                f'{desc}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="nutrition-header" style="margin-top:1.2rem">📚 Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:rgba(245,230,200,0.85); font-family:\'DM Sans\'; line-height:1.8; font-size:0.93rem;">'
            'Dataset yang digunakan adalah <strong style="color:#D4A017">clean_recipes.csv</strong> '
            'hasil pre-processing dari dataset Food.com Recipes &amp; Interactions (Kaggle). '
            'Dataset berisi <strong style="color:#D4A017">14,103 resep</strong> dengan 15 fitur termasuk '
            'nama, bahan, instruksi, nutrisi, rating, dan gambar.'
            '</p>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)  # close detail-panel

    with col2:
        # ── Feature Summary panel ──
        st.markdown('<div class="detail-panel">', unsafe_allow_html=True)
        st.markdown('<div class="nutrition-header">📊 Feature Summary</div>', unsafe_allow_html=True)

        features_info = {
            "Name": "Nama resep",
            "Images": "URL foto resep",
            "RecipeInstructions": "Langkah memasak",
            "RecipeServings": "Jumlah porsi",
            "RecipeIngredientParts": "Daftar bahan",
            "Keywords": "Tag/kata kunci",
            "RecipeCategory": "Kategori masakan",
            "Calories": "Total kalori (kcal)",
            "ProteinContent": "Kandungan protein (g)",
            "FatContent": "Kandungan lemak (g)",
            "CarbohydrateContent": "Karbohidrat (g)",
            "AggregatedRating": "Rating (0–5)",
            "ReviewCount": "Jumlah ulasan",
            "Features": "Combined features text",
            "TotalTime_Minutes": "Waktu masak (menit)",
        }

        for feat, desc in features_info.items():
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:0.4rem 0;'
                f'border-bottom:1px solid rgba(212,160,23,0.1); font-family:\'DM Sans\'; font-size:0.82rem;">'
                f'<span style="color:#D4A017; font-weight:500">{feat}</span>'
                f'<span style="color:rgba(245,230,200,0.65); text-align:right; padding-left:0.5rem">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)  # close feature panel

        # ── Tech Stack panel ──
        st.markdown('<div class="detail-panel" style="margin-top:1rem">', unsafe_allow_html=True)
        st.markdown('<div class="nutrition-header">🛠️ Tech Stack</div>', unsafe_allow_html=True)

        for tool, purpose in [
            ("Python 3.x", "Core language"),
            ("Streamlit", "Dashboard framework"),
            ("Scikit-learn", "TF-IDF & Cosine Similarity"),
            ("Pandas / NumPy", "Data processing"),
            ("Plotly", "Interactive visualizations"),
        ]:
            st.markdown(
                f'<div class="ingredient-tag" style="margin:0.3rem 0.2rem; font-size:0.8rem">'
                f'<strong style="color:#E07B39">{tool}</strong> — {purpose}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)  # close tech stack panel


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🍽️ FLAVORA · Smart Recipe Recommendation System · ETS Data Mining 2026<br>
    Built with Streamlit · Content-Based Filtering · TF-IDF × Cosine Similarity
</div>
""", unsafe_allow_html=True)