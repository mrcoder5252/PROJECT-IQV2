"""
Project-IQ: Enterprise Academic Project Intelligence Platform
Dark-Themed Industry-Grade Streamlit Architecture

Features:
- Student Portal & Authenticated Admin Portal
- Conversational AI RAG Advisor (ChatGPT/Claude Style with Guardrails) & Novel Idea Generator
- Project Originality & Duplicate Idea Detector with Executive Synthesized Brief
- Standalone Research Gap Finder across Historical Project Limiting Factors
- Multi-Dimensional Engineering Quality Scorer (without Faculty Committee Jargon)
- Student Team Expertise Profiling & Domain Matcher
- Authenticated Admin Center: Ingestion, Privacy Audit Log & Corpus Telemetry
"""

from __future__ import annotations
import hmac
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import config
from src.logger import get_logger
from src.vector_store import ProjectIQVectorStore
from src.gap_analyzer import GapAnalyzer
from src.team_matcher import TeamMatcher, DOMAIN_TAXONOMY
from src.duplicate_detector import DuplicateDetector
from src.quality_scorer import QualityScorer
from src.rag_chat import RAGChatEngine
from src.clean import is_ner_available

logger = get_logger("project_iq.web")

# Page Configuration
st.set_page_config(
    page_title="Project-IQ — Academic Project Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — Sleek, Dark Glass Industry-Grade Engineering Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #F7F8FA;
    --surface: #FFFFFF;
    --border: #E4E7EC;
    --text-primary: #1A1F2B;
    --text-secondary: #667085;
    --accent: #4F46E5;
    --accent-hover: #4338CA;
    --success: #12B76A;
    --warning: #F79009;
    --critical: #F04438;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    color: var(--text-primary);
}

/* Background Layout & Structure */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
}
[data-testid="stHeader"] {
    background-color: rgba(247, 248, 250, 0.85) !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
}

/* High-Visibility Focus & Accessibility */
*:focus-visible {
    outline: 3px solid var(--accent) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 12px rgba(79, 70, 229, 0.3) !important;
    transition: outline-offset 120ms ease, box-shadow 120ms ease !important;
}
.sr-only {
    position: absolute; width: 1px; height: 1px; padding: 0;
    margin: -1px; overflow: hidden; clip: rect(0,0,0,0);
    white-space: nowrap; border: 0;
}

/* Text Hierarchy & Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
[data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
    line-height: 1.6;
}

/* Form Inputs & Widget Labels */
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
.stTextInput input, .stTextArea textarea {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06) !important;
    transition: all 0.15s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.3) !important;
}
div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] > div:hover {
    background-color: var(--bg) !important;
}
div[data-baseweb="popover"] ul {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 14px rgba(16, 24, 40, 0.08) !important;
}
div[data-baseweb="popover"] li {
    color: var(--text-primary) !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: var(--bg) !important;
    color: var(--accent) !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Module Hero Card */
.module-hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    position: relative;
    overflow: hidden;
    transition: all 200ms cubic-bezier(0.16, 1, 0.3, 1);
}
.module-title {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin: 0;
    color: var(--text-primary);
}
.module-desc {
    font-size: 0.92rem;
    color: var(--text-secondary);
    line-height: 1.55;
    margin: 0.35rem 0 0 0;
    font-weight: 400;
}
.module-meta-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.85rem;
    flex-wrap: wrap;
}
.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: #F2F4F7;
    color: var(--text-secondary);
    border: 1px solid var(--border);
}
.meta-pill-blue   { background: rgba(79, 70, 229, 0.1); color: #4F46E5; border-color: rgba(79, 70, 229, 0.2); }
.meta-pill-green  { background: rgba(18, 183, 106, 0.1); color: #12B76A; border-color: rgba(18, 183, 106, 0.2); }
.meta-pill-purple { background: rgba(139, 92, 246, 0.1); color: #8B5CF6; border-color: rgba(139, 92, 246, 0.2); }
.meta-pill-amber  { background: rgba(247, 144, 9, 0.1); color: #F79009; border-color: rgba(247, 144, 9, 0.2); }
.meta-pill-red    { background: rgba(240, 68, 56, 0.1); color: #F04438; border-color: rgba(240, 68, 56, 0.2); }

/* Stat Cards */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem 1.15rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    transition: all 200ms cubic-bezier(0.16, 1, 0.3, 1);
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
    border-color: var(--accent);
}
.stat-card .stat-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    margin-bottom: 0.75rem;
}
.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
    margin-bottom: 0.2rem;
    font-feature-settings: "tnum";
}
.stat-card .stat-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
}

/* Prompt Chips */
div:has(#prompt-chips-wrapper) + div[data-testid="stHorizontalBlock"] .stButton > button,
div:has(#prompt-chips-wrapper) + div[data-testid="stHorizontalBlock"] + div[data-testid="stHorizontalBlock"] .stButton > button,
div:has(#prompt-chips-wrapper) ~ div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] .stButton > button,
[data-testid="stVerticalBlock"]:has(#prompt-chips-wrapper) div[data-testid="stHorizontalBlock"] .stButton > button,
.prompt-chip-btn {
    min-height: 72px !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.75rem 0.95rem !important;
    text-align: left !important;
    color: var(--text-primary) !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06) !important;
    transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    white-space: pre-line !important;
    width: 100% !important;
}
div:has(#prompt-chips-wrapper) + div[data-testid="stHorizontalBlock"] .stButton > button p,
div:has(#prompt-chips-wrapper) + div[data-testid="stHorizontalBlock"] + div[data-testid="stHorizontalBlock"] .stButton > button p,
div:has(#prompt-chips-wrapper) ~ div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] .stButton > button p,
[data-testid="stVerticalBlock"]:has(#prompt-chips-wrapper) div[data-testid="stHorizontalBlock"] .stButton > button p {
    white-space: pre-line !important;
    text-align: left !important;
    color: var(--text-primary) !important;
    font-size: 0.84rem !important;
    line-height: 1.35 !important;
    margin: 0 !important;
}
div:has(#prompt-chips-wrapper) + div[data-testid="stHorizontalBlock"] .stButton > button:hover,
div:has(#prompt-chips-wrapper) + div[data-testid="stHorizontalBlock"] + div[data-testid="stHorizontalBlock"] .stButton > button:hover,
div:has(#prompt-chips-wrapper) ~ div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] .stButton > button:hover,
[data-testid="stVerticalBlock"]:has(#prompt-chips-wrapper) div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    border-color: var(--accent) !important;
    background-color: var(--bg) !important;
    box-shadow: 0 4px 12px rgba(16, 24, 40, 0.08) !important;
}

/* Badges & Tags */
.tag {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.15rem 0.2rem 0.15rem 0;
}
.tag-blue   { background: rgba(79,70,229,0.08); color: #4F46E5; border: 1px solid rgba(79,70,229,0.2); }
.tag-green  { background: rgba(18,183,106,0.08); color: #12B76A; border: 1px solid rgba(18,183,106,0.2); }
.tag-amber  { background: rgba(247,144,9,0.08); color: #F79009; border: 1px solid rgba(247,144,9,0.2); }
.tag-red    { background: rgba(240,68,56,0.08); color: #F04438; border: 1px solid rgba(240,68,56,0.2); }
.tag-purple { background: rgba(139,92,246,0.08); color: #8B5CF6; border: 1px solid rgba(139,92,246,0.2); }
.tag-slate  { background: #F2F4F7; color: #667085; border: 1px solid #E4E7EC; }

/* Citation Cards */
.citation-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.95rem 1.15rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    transition: all 200ms cubic-bezier(0.16, 1, 0.3, 1);
}
.citation-card:hover {
    box-shadow: 0 4px 12px rgba(16, 24, 40, 0.08);
}

/* Callout Box */
.callout-box {
    border-left: 4px solid var(--accent);
    background: rgba(79,70,229,0.06);
    padding: 0.9rem 1.1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: var(--text-primary);
    font-size: 0.91rem;
    line-height: 1.6;
}

/* Pulse Dot */
.pulse-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
}
.pulse-dot.green { background: #12B76A; box-shadow: 0 0 0 3px rgba(18, 183, 106, 0.25); }
.pulse-dot.blue  { background: #4F46E5; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.25); }
.pulse-dot.amber { background: #F79009; box-shadow: 0 0 0 3px rgba(247, 144, 9, 0.25); }

/* Sidebar Radio Navigation Items */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.8rem !important;
    margin-bottom: 0.25rem !important;
    cursor: pointer !important;
    transition: all 120ms cubic-bezier(0.16, 1, 0.3, 1) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: var(--bg) !important;
    transform: translateX(2px) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: rgba(79,70,229,0.06) !important;
    border-color: var(--accent) !important;
    border-left: 2px solid var(--accent) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: var(--accent) !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px !important;
    background: #F2F4F7 !important;
    padding: 4px !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #ffffff !important;
}

/* Button Styling */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.25rem !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06) !important;
    transition: all 150ms cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.2) !important;
}
.stButton > button:not([kind="primary"]) {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 150ms cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: var(--bg) !important;
}
.stButton > button:active {
    transform: scale(0.97) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06) !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stExpander"]:hover {
    border-color: var(--accent) !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1.1rem 1.25rem !important;
    margin-bottom: 0.9rem !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([aria-label*="user" i]) {
    background-color: rgba(79,70,229,0.06) !important;
    border-left: 4px solid var(--accent) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) {
    background-color: var(--surface) !important;
    border-left: 4px solid #8B5CF6 !important;
}

/* Animations */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stAppViewContainer"] > section > div {
    animation: fadeSlideUp 200ms cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.skeleton-loading {
    background: linear-gradient(90deg, #E4E7EC 25%, #F2F4F7 50%, #E4E7EC 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 6px;
}

@keyframes toastSlideIn {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stAlert"] {
    animation: toastSlideIn 300ms cubic-bezier(0.16, 1, 0.3, 1);
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)


def safe_html(text: Any) -> str:
    """Escapes text safely for direct interpolation into HTML templates."""
    if text is None:
        return ""
    return html.escape(str(text))


def render_hero_header(icon: str, title: str, subtitle: str, badges: Optional[List[tuple[str, str]]] = None):
    """Renders a consistent, polished gradient hero banner for each functional module."""
    badges_html = ""
    if badges:
        pills = [f'<span class="meta-pill {safe_html(c)}">{safe_html(t)}</span>' for t, c in badges]
        badges_html = f'<div class="module-meta-badges">{" ".join(pills)}</div>'

    st.markdown(
        f'<div class="module-hero">'
        f'<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.25rem;">'
        f'<span style="font-size: 1.75rem; line-height: 1;">{icon}</span>'
        f'<h1 class="module-title">{safe_html(title)}</h1>'
        f'</div>'
        f'<p class="module-desc">{safe_html(subtitle)}</p>'
        f'{badges_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def render_citation_card(src: dict) -> str:
    """Formats an institutional citation into a safe, polished dark HTML card."""
    idx = safe_html(src.get("index", 1))
    src_file = safe_html(src.get("source_file", "Archive Document"))
    dept = safe_html(src.get("department", "General"))
    group = safe_html(src.get("group_id", "Group"))
    sec = safe_html(src.get("section", "Passage"))
    aff = safe_html(src.get("similarity_score", "0.85"))
    excerpt = safe_html(src.get("excerpt", ""))
    return (
        f'<div class="citation-card">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.3rem;">'
        f'<span style="font-weight: 700; color: #4F46E5; font-size: 0.88rem;">[{idx}] {src_file}</span>'
        f'<div>'
        f'<span class="tag tag-blue">{dept} / {group}</span>'
        f'<span class="tag tag-purple">Section: {sec}</span>'
        f'<span class="tag tag-green">Affinity: {aff}</span>'
        f'</div>'
        f'</div>'
        f'<blockquote style="margin: 0; padding: 0.5rem 0.8rem; background: #F7F8FA; border-left: 3px solid #D0D5DD; border-radius: 4px; font-size: 0.84rem; color: #667085; font-style: italic;">'
        f'"{excerpt}"'
        f'</blockquote>'
        f'</div>'
    )


def apply_plotly_theme(fig, height=320):
    """Applies modern dark typography and styling to Plotly figures."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, -apple-system, sans-serif", color="#1A1F2B", size=12),
        margin=dict(t=25, b=20, l=20, r=20),
        height=height,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#1A1F2B"
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E4E7EC", zerolinecolor="#D0D5DD")
    fig.update_yaxes(showgrid=True, gridcolor="#E4E7EC", zerolinecolor="#D0D5DD")
    return fig


def extract_text_from_upload(uploaded_file) -> str:
    """Safely extracts text from uploaded draft (.txt, .pdf, .docx)."""
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".txt"):
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif fname.endswith(".pdf"):
            import fitz
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            text_parts = [page.get_text() for page in doc]
            return "\n".join(text_parts)
        elif fname.endswith(".docx"):
            import docx
            import io
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        logger.warning(f"Could not extract text from {fname}: {e}")
    return ""


@st.cache_resource
def get_services():
    store = ProjectIQVectorStore(db_path=config.chroma_dir)
    gap_analyzer = GapAnalyzer(vector_store=store)
    matcher = TeamMatcher()
    detector = DuplicateDetector(vector_store=store)
    scorer = QualityScorer()
    rag_chat = RAGChatEngine(vector_store=store)
    return store, gap_analyzer, matcher, detector, scorer, rag_chat


store, gap_analyzer, matcher, detector, scorer, rag_chat = get_services()


# Load summary telemetry data
@st.cache_data
def load_telemetry_data():
    chunks_file = config.chunks_file
    report_file = config.reports_dir / "cleaning_report.json"
    manifest_file = config.extracted_dir / "_manifest.json"

    chunks_data = []
    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        chunks_data.append(json.loads(line))
                    except Exception:
                        pass

    rep_data = {}
    if report_file.exists():
        try:
            rep_data = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    man_data = {}
    if manifest_file.exists():
        try:
            man_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    df_chunks = pd.DataFrame(chunks_data) if chunks_data else pd.DataFrame()
    return df_chunks, rep_data, man_data


df_chunks, rep_data, man_data = load_telemetry_data()
db_count = store.count()
total_redactions = sum(rep_data.get("total_redactions", {}).values()) if (rep_data and "total_redactions" in rep_data) else 0

# Check background Gemini API key and spaCy NER
if not os.getenv("GEMINI_API_KEY") and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"])

has_gemini = bool(os.getenv("GEMINI_API_KEY") or config.gemini_api_key)
ner_active = is_ner_available()

# -------------------------------------------------------------
# Sidebar: Custom Logo Branding & Portal Mode Switcher
# -------------------------------------------------------------
st.sidebar.markdown(
    '<header role="banner">'
    '<div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0 0.85rem 0;">'
    '<div style="width: 44px; height: 44px; border-radius: 12px; background: #4F46E5; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15); color: white;">🎓</div>'
    '<div>'
    '<div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(135deg, #1A1F2B 0%, #4F46E5 60%, #4F46E5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1;">Project-IQ</div>'
    '<div style="font-size: 0.7rem; color: #667085; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">Academic Intelligence</div>'
    '</div>'
    '</div>'
    '<div style="display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(79, 70, 229, 0.06); color: #4F46E5; border: 1px solid rgba(79, 70, 229, 0.15); border-radius: 20px; padding: 0.15rem 0.55rem; font-size: 0.68rem; font-weight: 700; margin-bottom: 0.85rem;">'
    '<span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #4F46E5;"></span> Project-IQ v2.0 • Academic Edition</div>'
    '</header>',
    unsafe_allow_html=True
)

# AI Engine Status in Sidebar
if has_gemini:
    st.sidebar.markdown(
        '<div style="font-size: 0.76rem; color: #10b981 !important; padding: 0.2rem 0; font-weight: 600; margin-bottom: 0.6rem;">'
        '<span class="pulse-dot green"></span> '
        '🟢 LLM Active (Gemini API)</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div style="font-size: 0.76rem; color: #f59e0b !important; padding: 0.2rem 0; font-weight: 600; margin-bottom: 0.6rem;">'
        '<span class="pulse-dot amber"></span> '
        '🟡 Fallback Mode (Local Synthesis)</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown('<div style="margin: 0.4rem 0 0.6rem 0; border-top: 1px solid #E4E7EC;"></div>', unsafe_allow_html=True)

# Portal Switcher (Student vs Admin)
portal_mode = st.sidebar.radio(
    "Portal Workspace",
    ["🎓 Student Portal", "🔐 Admin Portal"],
    index=0,
    help="Toggle between Student Project Intelligence Tools and Administrative Ingestion & Audit logs."
)

st.sidebar.markdown('<div style="margin: 0.4rem 0 0.6rem 0; border-top: 1px solid #E4E7EC;"></div>', unsafe_allow_html=True)

# Main Banner: Warning when Gemini API Key is not configured
if not has_gemini:
    st.warning("⚠️ **Notice: Gemini API key is not configured (`GEMINI_API_KEY`).** Project-IQ is operating in local fallback mode using rule-based synthesis.")

# -------------------------------------------------------------
# Student Portal Navigation & Views
# -------------------------------------------------------------
if portal_mode == "🎓 Student Portal":
    student_nav = st.sidebar.radio(
        "Student Intelligence Modules",
        [
            "💬 AI Project Advisor & Ideator",
            "🛡️ Originality & Duplicate Checker",
            "🔬 Research Gap Finder",
            "⚖️ Project Quality Scorer",
            "👥 Student Team Skill Matcher"
        ],
        help="Explore student tools to evaluate, refine, and originate engineering capstone projects."
    )

    # ---------------------------------------------------------
    # Student Module 1: AI Project Advisor & Ideator
    # ---------------------------------------------------------
    if student_nav == "💬 AI Project Advisor & Ideator":
        render_hero_header(
            icon="💬",
            title="AI Project Advisor & Novel Ideator",
            subtitle="Engage with the conversational engineering advisor or synthesize tailored capstone project proposals grounded in institutional research gaps.",
            badges=[
                ("🤖 Conversational LLM Guidance", "meta-pill-blue")
            ]
        )

        tab_chat, tab_ideator = st.tabs(["💬 AI Project Advisor", "💡 Novel Idea Generator"])

        # Tab 1: Conversational Chat
        with tab_chat:
            st.markdown(
                '<p style="font-size: 0.88rem; font-weight: 700; color: #344054; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.35rem;">'
                '<span>⚡</span> <span>Quick Research Inquiries (Click to ask):</span></p>',
                unsafe_allow_html=True
            )
            st.markdown('<div id="prompt-chips-wrapper"></div>', unsafe_allow_html=True)
            q1, q2, q3 = st.columns(3)
            q4, q5, q6 = st.columns(3)

            selected_prompt = None
            with q1:
                if st.button("🌾 Agriculture Hardware Limits\nSensor, MCU & battery constraints", key="chip_1", use_container_width=True):
                    selected_prompt = "What hardware, sensor, and battery limitations were reported in past Smart Agriculture IoT projects?"
            with q2:
                if st.button("🩺 Healthcare DL Architectures\nCNN, ResNet & medical datasets", key="chip_2", use_container_width=True):
                    selected_prompt = "Which deep learning architectures (e.g. CNNs, ResNet) and datasets were used in healthcare and medical diagnosis projects?"
            with q3:
                if st.button("🔐 Blockchain Latency & Gas\nTransaction throughput & costs", key="chip_3", use_container_width=True):
                    selected_prompt = "How did past blockchain and smart contract projects address transaction latency, security, and gas fee constraints?"
            with q4:
                if st.button("🤖 NLP Gaps in CS & DS\nTransformer & BERT limitations", key="chip_4", use_container_width=True):
                    selected_prompt = "What are the most common unaddressed research gaps and future scope items across NLP projects?"
            with q5:
                if st.button("📊 Compare CS vs DS Methods\nMethodologies & benchmark metrics", key="chip_5", use_container_width=True):
                    selected_prompt = "Compare the technical methodologies and evaluation metrics between Computer Science (CS) and Data Science (DS) projects."
            with q6:
                if st.button("💡 Novel Capstone Ideation\nSynthesize project from past gaps", key="chip_6", use_container_width=True):
                    selected_prompt = "Based on the limitations of past student projects, synthesize a novel capstone project proposal with technical stack and evaluation metrics."

            # Session State for Chat
            if "rag_chat_history" not in st.session_state:
                st.session_state.rag_chat_history = [
                    {
                        "role": "assistant",
                        "content": (
                            "👋 **Welcome to the Project-IQ AI Engineering Advisor!**\n\n"
                            f"I have synthesized intelligence across **{db_count:,} canonical project passages** from past Computer Science and Data Science cohorts.\n\n"
                            "Ask me **any technical question**—from architecture trade-offs (e.g. YOLO vs MobileNet on microcontrollers, SQL vs NoSQL for telemetry), "
                            "to benchmark targets, dataset constraints, and methodology structuring.\n\n"
                            "*Click one of the quick prompt cards above or type your question below!*"
                        ),
                        "engine": "System Greeting"
                    }
                ]

            if len(st.session_state.rag_chat_history) > 1:
                c_rst1, c_rst2 = st.columns([6, 1])
                with c_rst2:
                    if st.button("🗑️ Reset Chat", use_container_width=True):
                        st.session_state.rag_chat_history = [st.session_state.rag_chat_history[0]]
                        st.rerun()

            # Render Chat Messages (Clean, without raw document card leaks)
            for msg in st.session_state.rag_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("engine") and msg["engine"] != "System Greeting":
                        st.markdown(
                            f'<div style="display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.2rem 0.6rem; background: #E4E7EC; border: 1px solid #D0D5DD; border-radius: 6px; font-size: 0.74rem; color: #667085; margin-top: 0.4rem;">'
                            f'<span>⚙️</span> <span>Synthesized via <strong style="color: #4F46E5;">{safe_html(msg["engine"])}</strong></span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Chat Input
            user_input = st.chat_input("Ask about architectures, ML models, IoT hardware, database trade-offs...")
            active_query = selected_prompt or user_input

            if active_query:
                st.session_state.rag_chat_history.append({"role": "user", "content": active_query})
                with st.chat_message("user"):
                    st.markdown(active_query)

                with st.chat_message("assistant"):
                    with st.spinner("Synthesizing engineering advice across repository intelligence..."):
                        try:
                            result = rag_chat.ask(
                                question=active_query,
                                chat_history=[{"role": m["role"], "content": m["content"]} for m in st.session_state.rag_chat_history[:-1]],
                                n_results=6
                            )
                        except Exception as ex:
                            logger.error(f"RAG execution exception: {ex}")
                            result = {
                                "answer": f"⚠️ An error occurred during synthesis: {safe_html(str(ex))}. Please try rephrasing your technical inquiry.",
                                "engine_used": "Error Fallback"
                            }

                        st.markdown(result["answer"])
                        st.markdown(
                            f'<div style="display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.2rem 0.6rem; background: #E4E7EC; border: 1px solid #D0D5DD; border-radius: 6px; font-size: 0.74rem; color: #667085; margin-top: 0.4rem;">'
                            f'<span>⚙️</span> <span>Synthesized via <strong style="color: #4F46E5;">{safe_html(result["engine_used"])}</strong></span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        st.session_state.rag_chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "engine": result["engine_used"]
                        })
                        st.rerun()

        # Tab 2: Novel Idea Generator
        with tab_ideator:
            st.markdown("#### 💡 Customized Capstone Proposal Generator")
            st.markdown("Generate unique, viable capstone proposals tailored to your specific engineering domain, target stack, or problem interest.")

            col_ig1, col_ig2 = st.columns([1, 1])
            with col_ig1:
                idea_domain = st.selectbox("Focus Engineering Domain", ["All Domains"] + list(DOMAIN_TAXONOMY.keys()), key="idea_domain_sel")
                idea_interest = st.text_input("Specific Problem Focus / Interest (optional)", placeholder="e.g. Edge crop pathology, Federated privacy, Low-power IoT")
            with col_ig2:
                idea_stack = st.text_input("Target Tech Stack (optional)", placeholder="e.g. PyTorch, ONNX, FastAPI, Docker, Raspberry Pi")
                idea_count = st.slider("Number of Proposals to Generate", min_value=1, max_value=5, value=2, key="idea_cnt_slider")

            gen_idea_btn = st.button("🚀 Generate Tailored Novel Proposals", type="primary", use_container_width=True)

            if gen_idea_btn:
                with st.spinner("Mining unaddressed repository gaps and synthesizing novel proposals..."):
                    dom_arg = None if idea_domain == "All Domains" else idea_domain
                    if idea_interest.strip():
                        filter_query = f"{dom_arg or ''} {idea_interest.strip()} {idea_stack.strip()}".strip()
                    else:
                        filter_query = dom_arg

                    generated_proposals = gap_analyzer.generate_novel_projects(
                        domain_filter=filter_query,
                        num_proposals=idea_count
                    )
                    st.session_state["tailored_proposals"] = generated_proposals

            tailored_props = st.session_state.get("tailored_proposals", [])
            if tailored_props:
                st.markdown(f"### 📋 Tailored Novel Proposals ({len(tailored_props)})")
                for i, p in enumerate(tailored_props, 1):
                    feasibility = p.get('feasibility_score', 8)
                    f_pill = "tag-green" if feasibility >= 8 else ("tag-amber" if feasibility >= 6 else "tag-red")
                    st.markdown(
                        f'<div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 12px; padding: 1.25rem 1.4rem; margin-bottom: 1.25rem; box-shadow: 0 4px 14px rgba(16, 24, 40, 0.06);">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; flex-wrap: wrap; gap: 0.35rem;">'
                        f'<span style="font-size: 1.2rem; font-weight: 800; color: #1A1F2B;">{i}. {safe_html(p.get("title"))}</span>'
                        f'<div>'
                        f'<span class="tag tag-blue">Domain: {safe_html(p.get("domain"))}</span>'
                        f'<span class="tag {f_pill}">Feasibility: {feasibility}/10</span>'
                        f'</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    t_p1, t_p2, t_p3 = st.tabs(["📌 Problem & Innovation", "🛠️ Architecture & Tech Stack", "🔬 Addressed Gaps & Novelty Edge"])
                    with t_p1:
                        st.markdown(f"**Problem Statement:**\n\n{p.get('problem_statement')}")
                        st.markdown(
                            f'<div class="callout-box">'
                            f'<strong style="color: #4F46E5;">💡 Core Innovation:</strong> {safe_html(p.get("core_innovation"))}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with t_p2:
                        st.markdown(f"**Proposed Methodology:**\n\n{p.get('proposed_methodology')}")
                        stack = p.get("tech_stack", [])
                        st.markdown("**Recommended Stack:** " + " ".join(f'<span class="tag tag-slate">{safe_html(s)}</span>' for s in stack), unsafe_allow_html=True)
                        deliv = p.get("expected_deliverables", [])
                        if deliv:
                            st.markdown("**Expected Deliverables:**")
                            for d in deliv:
                                st.markdown(f"- ✅ {safe_html(d)}")
                    with t_p3:
                        citations = p.get("past_project_citations", [])
                        if citations:
                            for cite in citations:
                                st.markdown(
                                    f'<div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;">'
                                    f'<div style="font-size: 0.84rem; color: #344054;"><strong style="color: #f87171;">Tackled Limitation:</strong> {safe_html(cite.get("identified_gap"))}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("Direct synthesis based on institutional domain limitations.")

                st.download_button(
                    label="📥 Download Tailored Proposals JSON",
                    data=json.dumps(tailored_props, indent=2),
                    file_name="tailored_capstone_proposals.json",
                    mime="application/json"
                )

    # ---------------------------------------------------------
    # Student Module 2: Originality & Duplicate Checker
    # ---------------------------------------------------------
    elif student_nav == "🛡️ Originality & Duplicate Checker":
        render_hero_header(
            icon="🛡️",
            title="Project Originality & Redundancy Guard",
            subtitle="Screen proposed capstone concepts against historical institutional repositories to detect duplicate ideas, summarize conceptual overlaps in brief, and safeguard academic novelty.",
            badges=[
                ("🛡️ Redundancy Guard", "meta-pill-red")
            ]
        )

        col_in1, col_in2 = st.columns([1, 1])
        with col_in1:
            test_title = st.text_input(
                "Proposed Project Title",
                "Acupoint Therapy and Symptom Management Android Application",
                help="Tentative title of the proposed student project."
            )
        with col_in2:
            test_abstract = st.text_area(
                "Proposed Project Abstract / Summary",
                "An AI-powered Android mobile app developed with Firebase for musculoskeletal pain relief, guiding patients through questionnaires and recommending exercises.",
                height=100,
                help="Summary of the proposed system methodology, target users, and technology stack."
            )

        check_btn = st.button("🛡️ Assess Originality & Redundancy Risk", type="primary")

        if check_btn:
            with st.spinner("Analyzing semantic overlap and generating executive brief..."):
                dup_res = detector.check_originality(title=test_title, abstract=test_abstract, top_k=5)

            st.markdown('<div style="margin: 1.25rem 0 0.75rem 0; border-top: 1px solid #E4E7EC;"></div>', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.markdown("#### Verdict & Originality Score")
                max_pct = dup_res["max_similarity_pct"]
                orig_score = dup_res["originality_score"]

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=orig_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Originality Score (%)", 'font': {'size': 16, 'color': '#1A1F2B', 'family': 'Inter, sans-serif'}},
                    number={'font': {'size': 36, 'color': '#1A1F2B', 'family': 'Inter, sans-serif'}, 'valueformat': '.1f', 'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#667085', 'tickwidth': 1.5, 'tickfont': {'color': '#667085', 'size': 11}},
                        'bar': {'color': "#4F46E5", 'thickness': 0.28},
                        'bgcolor': "#FFFFFF",
                        'borderwidth': 1,
                        'bordercolor': "#D0D5DD",
                        'steps': [
                            {'range': [0, 35], 'color': "rgba(239, 68, 68, 0.25)"},
                            {'range': [35, 70], 'color': "rgba(245, 158, 11, 0.25)"},
                            {'range': [70, 100], 'color': "rgba(16, 185, 129, 0.25)"}
                        ],
                        'threshold': {
                            'line': {'color': "#ef4444", 'width': 3},
                            'thickness': 0.75,
                            'value': 35
                        }
                    }
                ))
                apply_plotly_theme(fig_gauge, height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)

                if dup_res["verdict_level"] == "danger":
                    st.markdown(
                        f'<div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-left: 5px solid #ef4444; border-radius: 8px; padding: 0.9rem; margin-bottom: 0.75rem;">'
                        f'<div style="font-weight: 800; color: #fca5a5; font-size: 0.96rem; display: flex; align-items: center; gap: 0.35rem;">'
                        f'<span>🔴</span> <span>HIGH REDUNDANCY DETECTED ({max_pct:.1f}% Historical Similarity)</span>'
                        f'</div>'
                        f'<p style="margin: 0.35rem 0 0 0; color: #fecaca; font-size: 0.85rem; line-height: 1.5;">{safe_html(dup_res["warning_message"])}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                elif dup_res["verdict_level"] == "warning":
                    st.markdown(
                        f'<div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-left: 5px solid #f59e0b; border-radius: 8px; padding: 0.9rem; margin-bottom: 0.75rem;">'
                        f'<div style="font-weight: 800; color: #fcd34d; font-size: 0.96rem; display: flex; align-items: center; gap: 0.35rem;">'
                        f'<span>🟡</span> <span>MODERATE OVERLAP ({max_pct:.1f}% Historical Similarity)</span>'
                        f'</div>'
                        f'<p style="margin: 0.35rem 0 0 0; color: #fef3c7; font-size: 0.85rem; line-height: 1.5;">{safe_html(dup_res["warning_message"])}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-left: 5px solid #10b981; border-radius: 8px; padding: 0.9rem; margin-bottom: 0.75rem;">'
                        f'<div style="font-weight: 800; color: #6ee7b7; font-size: 0.96rem; display: flex; align-items: center; gap: 0.35rem;">'
                        f'<span>🟢</span> <span>HIGH ORIGINALITY VERIFIED ({orig_score:.1f}% Novelty Rating)</span>'
                        f'</div>'
                        f'<p style="margin: 0.35rem 0 0 0; color: #d1fae5; font-size: 0.85rem; line-height: 1.5;">{safe_html(dup_res["warning_message"])}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with res_col2:
                st.markdown("#### 📋 Executive Synthesized Brief")
                st.markdown(
                    f'<div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 10px; padding: 1.1rem 1.25rem; margin-bottom: 1rem;">'
                    f'<h5 style="margin: 0 0 0.5rem 0; color: #4F46E5; font-weight: 700;">Conceptual Overlap Summary:</h5>'
                    f'<p style="margin: 0; font-size: 0.92rem; color: #344054; line-height: 1.6;">{safe_html(dup_res.get("executive_summary", ""))}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown("#### 💡 Actionable Differentiation Recommendations")
                recommendations = dup_res.get("differentiation_recommendations", [])
                for rec in recommendations:
                    st.markdown(
                        f'<div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-left: 4px solid #4F46E5; border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.6rem; font-size: 0.88rem; color: #344054;">'
                        f'👉 {safe_html(rec)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                '<div class="callout-box">'
                '<strong style="color: #4F46E5;">🛡️ Originality Screening:</strong> Enter your tentative project title and abstract summary, then click <strong>Assess Originality & Redundancy Risk</strong>. '
                'The system evaluates semantic overlap across the institutional repository and generates an executive brief with actionable technical recommendations to differentiate your capstone.'
                '</div>',
                unsafe_allow_html=True
            )

    # ---------------------------------------------------------
    # Student Module 3: Standalone Research Gap Finder
    # ---------------------------------------------------------
    elif student_nav == "🔬 Research Gap Finder":
        render_hero_header(
            icon="🔬",
            title="Institutional Research Gap Finder",
            subtitle="Explore unaddressed technical limitations, hardware bottlenecks, and future scope items mined across historical college projects.",
            badges=[
                ("🔬 Limitation Mining", "meta-pill-blue")
            ]
        )

        col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
        with col_f1:
            gap_domain_sel = st.selectbox("Filter Domain", ["All Domains"] + list(DOMAIN_TAXONOMY.keys()))
        with col_f2:
            gap_sec_sel = st.selectbox("Section Focus", ["ALL", "FUTURE SCOPE", "CONCLUSION", "RESULTS", "METHODOLOGY"])
        with col_f3:
            gap_search_term = st.text_input("Search Limitation Keywords", placeholder="e.g. latency, battery, accuracy, dataset, edge")

        with st.spinner("Extracting recurring limitations and research gaps..."):
            extracted_gaps = gap_analyzer.extract_project_gaps(top_k_per_section=15)

        # Apply Filters
        filtered_gaps = extracted_gaps
        if gap_sec_sel != "ALL":
            filtered_gaps = [g for g in filtered_gaps if g.get("section") == gap_sec_sel]
        if gap_search_term.strip():
            sterm = gap_search_term.strip().lower()
            filtered_gaps = [g for g in filtered_gaps if sterm in g.get("text", "").lower() or any(sterm in ind for ind in g.get("matched_indicators", []))]

        st.markdown(f"### 📊 Identified Research Gaps ({len(filtered_gaps)} Passages Mined)")

        # Limitation Indicators Overview
        all_indicators = []
        for g in filtered_gaps:
            all_indicators.extend(g.get("matched_indicators", []))

        if all_indicators:
            ind_counts = pd.Series(all_indicators).value_counts().head(8).reset_index()
            ind_counts.columns = ["Limitation Indicator", "Frequency"]
            fig_ind = px.bar(
                ind_counts,
                x="Frequency",
                y="Limitation Indicator",
                orientation="h",
                color_discrete_sequence=["#4F46E5"]
            )
            fig_ind.update_traces(marker_line_width=0)
            apply_plotly_theme(fig_ind, height=260)
            fig_ind.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_ind, use_container_width=True)
            st.caption("Figure: Most frequent technical bottlenecks and limitation markers identified across past reports.")

        # Render Gap Cards
        for idx, g in enumerate(filtered_gaps[:12], 1):
            indicators_html = " ".join(f'<span class="tag tag-amber">{safe_html(ind)}</span>' for ind in g.get("matched_indicators", []))
            clean_gap_text = re.sub(r'\[NAME\]|\[STUDENT_ID\]|\[PHONE\]|\[EMAIL\]', '', g.get("text", "")).strip()
            summary_gap = clean_gap_text[:280] + ("..." if len(clean_gap_text) > 280 else "")

            st.markdown(
                f'<div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.85rem; box-shadow: 0 2px 6px rgba(16, 24, 40, 0.06);">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem; flex-wrap: wrap; gap: 0.35rem;">'
                f'<span style="font-weight: 700; color: #4F46E5; font-size: 0.92rem;">#{idx} Section: {safe_html(g.get("section"))}</span>'
                f'<div>{indicators_html}</div>'
                f'</div>'
                f'<p style="margin: 0 0 0.6rem 0; font-size: 0.88rem; color: #344054; line-height: 1.55;">"{safe_html(summary_gap)}"</p>'
                f'<div style="background: #F7F8FA; border-left: 3px solid #12B76A; border-radius: 4px; padding: 0.5rem 0.8rem; font-size: 0.82rem; color: #067647;">'
                f'<strong>💡 Innovation Opportunity:</strong> Address this limitation by incorporating hardware-accelerated quantization, synthetic dataset expansion, or cross-domain validation.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ---------------------------------------------------------
    # Student Module 4: Project Quality Scorer
    # ---------------------------------------------------------
    elif student_nav == "⚖️ Project Quality Scorer":
        render_hero_header(
            icon="⚖️",
            title="Engineering Quality Scorer & Technical Evaluator",
            subtitle="Pre-submission quality evaluation measuring architecture depth, metric rigor, and technical feasibility.",
            badges=[
                ("⚖️ Multi-Dimensional Scoring", "meta-pill-blue")
            ]
        )

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            prop_title = st.text_input(
                "Project Topic / Title",
                "Edge-Optimized CNN for Real-Time Plant Pathology",
                help="Full tentative title of the proposed student project."
            )
            prop_domain = st.selectbox(
                "Primary Engineering Domain",
                ["Computer Vision & Healthcare", "Smart Agriculture & IoT", "Cloud & Distributed Systems", "Cybersecurity & Blockchain", "NLP & Intelligent Systems", "Data Science & Analytics"]
            )
            prop_prob = st.text_area(
                "Problem Statement",
                "Existing crop pathology models suffer from severe inference latency bottlenecks and high energy consumption on embedded edge devices.",
                height=90,
                help="Core technical problem statement being addressed."
            )
        with col_q2:
            prop_meth = st.text_area(
                "Proposed Methodology & Architecture",
                "1. Collect crop images. 2. Quantize ResNet model via ONNX runtime. 3. Deploy on Raspberry Pi with sub-40ms latency and 96.2% accuracy [1]. 4. Containerized verification pipeline.",
                height=110,
                help="Methodology with phases, target benchmarks, and pipeline stages."
            )
            prop_stack = st.text_input(
                "Tech Stack (comma-separated)",
                "PyTorch, ONNX, FastAPI, Docker, Raspberry Pi",
                help="Technologies planned for development, training, and deployment."
            )

        with st.expander("📄 Optional: Upload or Paste Draft Document / Synopsis", expanded=False):
            up_doc = st.file_uploader("Upload Draft Document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
            pasted_doc = st.text_area("Or Paste Document / Synopsis Text Here", height=80)

        score_btn = st.button("⚖️ Run Engineering Quality Evaluation", type="primary")

        if score_btn:
            doc_extracted = extract_text_from_upload(up_doc) if up_doc else ""
            combined_doc_text = f"{doc_extracted}\n{pasted_doc}".strip()

            with st.spinner("Analyzing methodology depth, metric rigor, and architecture feasibility..."):
                score_res = scorer.score_proposal(
                    title=prop_title,
                    problem_statement=prop_prob,
                    proposed_methodology=prop_meth,
                    tech_stack=prop_stack,
                    citations=["Ref [1] 2024"],
                    domain=prop_domain,
                    document_text=combined_doc_text
                )

            st.markdown('<div style="margin: 1.25rem 0 0.75rem 0; border-top: 1px solid #E4E7EC;"></div>', unsafe_allow_html=True)
            sc_col1, sc_col2 = st.columns([1, 1])

            with sc_col1:
                st.markdown("#### Overall Engineering Score")
                overall = score_res["overall_score"]
                stars = score_res["star_rating"]
                grade = score_res.get("grade", "Grade A")
                tier = score_res["readiness_tier"]

                st.markdown(
                    f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.85rem;">'
                    f'<div class="stat-card" style="padding: 1rem;">'
                    f'<div class="stat-value" style="font-size: 1.7rem; color: #4F46E5;">{overall:.1f}</div>'
                    f'<div class="stat-label">Composite Score / 100</div>'
                    f'<div style="font-size: 0.78rem; color: #4F46E5; margin-top: 0.3rem; font-weight: 700;">{safe_html(grade)}</div>'
                    f'</div>'
                    f'<div class="stat-card" style="padding: 1rem;">'
                    f'<div class="stat-value" style="font-size: 1.7rem; color: #f59e0b;">{stars} ⭐</div>'
                    f'<div class="stat-label">Technical Rating</div>'
                    f'<div style="font-size: 0.78rem; color: #667085; margin-top: 0.3rem;">Engineering Standard</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if overall >= 75:
                    st.success(f"🚀 **{tier}**")
                elif overall >= 50:
                    st.warning(f"⚠️ **{tier}**")
                else:
                    st.error(f"❌ **{tier}**")

                # Actionable Advice
                st.markdown("#### 💡 Actionable Improvements")
                for advice in score_res["actionable_advice"]:
                    st.markdown(
                        f'<div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-left: 4px solid #4F46E5; border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.6rem; font-size: 0.88rem; color: #344054;">'
                        f'👉 {safe_html(advice)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with sc_col2:
                st.markdown("#### 📊 Multi-Dimensional Breakdown")
                dims = score_res.get("dimensions", {})
                dim_names = list(dims.keys())
                dim_vals = list(dims.values())

                if dim_names:
                    # Radar Chart
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=dim_vals + [dim_vals[0]],
                        theta=dim_names + [dim_names[0]],
                        fill='toself',
                        fillcolor='rgba(79, 70, 229, 0.12)',
                        line=dict(color='#4F46E5', width=2)
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E4E7EC", tickfont=dict(color="#667085")),
                            angularaxis=dict(gridcolor="#E4E7EC", tickfont=dict(color="#1A1F2B"))
                        ),
                        margin=dict(t=25, b=25, l=30, r=30)
                    )
                    apply_plotly_theme(fig_radar, height=280)
                    st.plotly_chart(fig_radar, use_container_width=True)

                # Feature Attribution
                shaps = score_res.get("feature_contributions") or score_res.get("shap_attributions", {})
                if shaps:
                    df_shap = pd.DataFrame([
                        {"Dimension": k.replace("_", " ").title(), "Attribution": v}
                        for k, v in shaps.items()
                    ])
                    df_shap["Color"] = df_shap["Attribution"].apply(lambda x: "#10b981" if x >= 0 else "#ef4444")
                    fig_shap = px.bar(
                        df_shap,
                        x="Attribution",
                        y="Dimension",
                        orientation="h",
                        color="Color",
                        color_discrete_map="identity"
                    )
                    fig_shap.update_traces(marker_line_width=0)
                    apply_plotly_theme(fig_shap, height=200)
                    fig_shap.update_layout(
                        xaxis=dict(title="Weighted contribution vs. neutral baseline (rule-based)", showgrid=True, gridcolor="#E4E7EC"),
                        yaxis=dict(autorange="reversed", title="")
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.markdown(
                '<div class="callout-box">'
                '<strong style="color: #4F46E5;">⚖️ Pre-Submission Evaluation:</strong> Enter your proposal title, problem statement, methodology, and tech stack (or upload a draft document), then click <strong>Run Engineering Quality Evaluation</strong>. '
                'The system extracts quantitative rigor, architecture depth, and domain alignment to produce composite marks and radar dimensional breakdowns.'
                '</div>',
                unsafe_allow_html=True
            )

    # ---------------------------------------------------------
    # Student Module 5: Student Team Domain Matcher
    # ---------------------------------------------------------
    elif student_nav == "👥 Student Team Skill Matcher":
        render_hero_header(
            icon="👥",
            title="Student Team Expertise Profiler & Domain Matcher",
            subtitle="Evaluates student team skillsets against domain taxonomies to recommend optimal project directions and highlight skill gaps.",
            badges=[
                ("👥 7 Domain Taxonomies", "meta-pill-blue")
            ]
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            team_skills_str = st.text_area(
                "Team Programming Languages, Frameworks & Hardware Skills",
                "Python, PyTorch, OpenCV, Flask, Docker, CNN, Computer Vision",
                height=100,
                help="List skills separated by commas."
            )
        with col_m2:
            team_interests_str = st.text_area(
                "Target Project Interests / Elective Areas (optional)",
                "Healthcare Diagnostics, Medical Imaging, Automated Disease Classification",
                height=100,
                help="Optional elective interests or industry preferences."
            )

        match_btn = st.button("🎯 Match Team to Optimal Domains", type="primary")

        if match_btn:
            skills = [s.strip() for s in re.split(r"[,;\n]+", team_skills_str) if s.strip()]
            interests = [i.strip() for i in re.split(r"[,;\n]+", team_interests_str) if i.strip()]

            with st.spinner("Computing team skillset embeddings and domain affinity vectors..."):
                team_res = matcher.evaluate_team(skills=skills, interests=interests, top_k=4)

            st.markdown('<div style="margin: 1.25rem 0 0.75rem 0; border-top: 1px solid #E4E7EC;"></div>', unsafe_allow_html=True)
            st.markdown("### 🏆 Ranked Domain Recommendations")
            for i, d in enumerate(team_res["top_domains"], 1):
                readiness = d['team_readiness']
                r_pill = "tag-green" if readiness == "High" else ("tag-amber" if readiness == "Moderate" else "tag-red")
                matched_skills_html = " ".join(f'<span class="tag tag-blue">{safe_html(s)}</span>' for s in d['matched_skills']) if d['matched_skills'] else '<span style="color: #667085; font-size: 0.85rem;">Semantic affinity overlap</span>'
                learn_skills_html = " ".join(f'<span class="tag tag-amber">{safe_html(k)}</span>' for k in d['suggested_to_learn']) if d['suggested_to_learn'] else '<span style="color: #10b981; font-size: 0.85rem;">Comprehensive skillset covered</span>'

                st.markdown(
                    f'<div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 12px; padding: 1.25rem 1.35rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(16, 24, 40, 0.06);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.4rem;">'
                    f'<div>'
                    f'<span style="font-size: 1.15rem; font-weight: 800; color: #1A1F2B;">{i}. {safe_html(d["domain"])}</span>'
                    f'<p style="margin: 0.25rem 0 0 0; color: #667085; font-size: 0.88rem;">{safe_html(d["description"])}</p>'
                    f'</div>'
                    f'<div style="text-align: right;">'
                    f'<div style="font-size: 1.5rem; font-weight: 800; color: #4F46E5; line-height: 1;">{d["match_percentage"]}%</div>'
                    f'<div style="font-size: 0.72rem; color: #667085; font-weight: 600; text-transform: uppercase;">Match Rating</div>'
                    f'<div style="margin-top: 0.35rem;"><span class="tag {r_pill}">Readiness: {safe_html(readiness.upper())}</span></div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="margin-top: 0.75rem; border-top: 1px solid #E4E7EC; padding-top: 0.75rem;">'
                    f'<div style="margin-bottom: 0.5rem;"><strong style="font-size: 0.84rem; color: #344054;">Matched Competencies:</strong><br/>{matched_skills_html}</div>'
                    f'<div><strong style="font-size: 0.84rem; color: #344054;">💡 Recommended Skills to Learn:</strong><br/>{learn_skills_html}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# -------------------------------------------------------------
# Admin Portal Navigation & Views (Authenticated)
# -------------------------------------------------------------
else:
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    st.sidebar.markdown("#### 🔐 Admin Security Gate")
    if not st.session_state.admin_authenticated:
        configured_key = config.admin_passkey
        if not configured_key:
            st.sidebar.error("Admin portal is locked: set PROJECTIQ_ADMIN_PASSKEY in your environment or .env.")
        else:
            admin_key_input = st.sidebar.text_input("Enter Admin Passkey", type="password")
            if st.sidebar.button("Unlock Admin Portal", type="primary"):
                if hmac.compare_digest(admin_key_input.strip().encode(), configured_key.encode()):
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.sidebar.error("Invalid passkey. Access denied.")
    else:
        st.sidebar.success("✅ Admin Session Authenticated")
        if st.sidebar.button("Lock Admin Portal"):
            st.session_state.admin_authenticated = False
            st.rerun()

    if not st.session_state.admin_authenticated:
        render_hero_header(
            icon="🔐",
            title="Institutional Administration Portal",
            subtitle="Restricted environment for department coordinators, faculty evaluators, and repository administrators.",
            badges=[
                ("🔒 Restricted Access", "meta-pill-red")
            ]
        )
        st.markdown(
            '<div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 12px; padding: 2rem; text-align: center; max-width: 580px; margin: 2rem auto;">'
            '<div style="font-size: 2.5rem; margin-bottom: 1rem;">🔐</div>'
            '<h3 style="color: #1A1F2B; margin: 0 0 0.5rem 0;">Institutional Authentication Required</h3>'
            '<p style="color: #667085; font-size: 0.92rem; line-height: 1.6; margin-bottom: 1.5rem;">'
            'The Administration Center houses raw document ingestion, privacy redaction logs, and full corpus telemetry. '
            'Please enter the administrative passkey in the sidebar to authenticate.'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        admin_nav = st.sidebar.radio(
            "Admin Management Modules",
            [
                "📥 Data Ingestion & Processing Center",
                "🛡️ Privacy Sanitization Audit Log",
                "📊 Institutional Corpus Telemetry"
            ]
        )

        if not ner_active:
            st.warning(
                "⚠️ **Warning: spaCy NER model (`en_core_web_sm`) is not loaded.** "
                "Named Entity Recognition is degraded. Student and faculty name redaction is relying on fallback regex heuristics. "
                "Run `python -m spacy download en_core_web_sm` on the server to restore full NER capability."
            )

        # Admin Module 1: Ingestion & Document Processing
        if admin_nav == "📥 Data Ingestion & Processing Center":
            render_hero_header(
                icon="📥",
                title="Data Ingestion & Multi-Format Processing Center",
                subtitle="Upload new project documents, organize into institutional hierarchy, and trigger vector pipeline indexing.",
                badges=[
                    ("📤 Multi-Format Parser", "meta-pill-blue")
                ]
            )

            st.markdown("#### Upload Academic Project Files")
            st.markdown("Supported formats: `.pdf` (Blackbooks), `.pptx` (Presentations), `.docx` (Synopses, Research Papers).")

            target_dept = st.selectbox("Assign to Department", ["CS_1", "CS_2", "DS_1", "DS_2", "General"])
            target_group = st.text_input("Assign to Group ID", "Group 14")
            target_category = st.selectbox("Document Category", ["Black Book", "Review 4", "synopsis", "Research Paper", "Abstract"])

            uploaded_files = st.file_uploader(
                "Choose files to ingest into repository",
                type=["pdf", "pptx", "docx", "doc"],
                accept_multiple_files=True
            )

            if uploaded_files:
                clean_group = re.sub(r'[^a-zA-Z0-9_\-\s]', '', target_group).strip() or "Group_14"
                upload_dest = config.raw_dir / target_dept / clean_group / target_category
                upload_dest.mkdir(parents=True, exist_ok=True)
                for f in uploaded_files:
                    clean_filename = Path(f.name).name
                    save_path = upload_dest / clean_filename
                    save_path.write_bytes(f.getvalue())
                    st.success(f"Saved: `{clean_filename}` to `{save_path}`")
                st.info("Files successfully staged. Execute `python run_pipeline.py` to index new records into ChromaDB.")

        # Admin Module 2: Privacy Sanitization Audit Log
        elif admin_nav == "🛡️ Privacy Sanitization Audit Log":
            render_hero_header(
                icon="🛡️",
                title="Privacy Sanitization & PII Masking Audit Log",
                subtitle="Inspect privacy redaction metrics, masked data breakdowns, and zero-leakage security protocols.",
                badges=[
                    ("🛡️ spaCy NER + Regex Masking", "meta-pill-green")
                ]
            )

            if rep_data:
                redactions = rep_data.get("total_redactions", {})
                st.markdown(
                    '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.85rem; margin-bottom: 1.25rem;">'
                    f'<div class="stat-card" style="padding: 1rem;">'
                    f'<div class="stat-value" style="font-size: 1.5rem; color: #4F46E5;">{redactions.get("local_name", 0):,}</div>'
                    f'<div class="stat-label">Local Names</div>'
                    f'<div style="font-size: 0.72rem; color: #667085; margin-top: 0.2rem;">Masked with [NAME]</div>'
                    f'</div>'
                    f'<div class="stat-card" style="padding: 1rem;">'
                    f'<div class="stat-value" style="font-size: 1.5rem; color: #60a5fa;">{redactions.get("phone", 0):,}</div>'
                    f'<div class="stat-label">Phone Numbers</div>'
                    f'<div style="font-size: 0.72rem; color: #667085; margin-top: 0.2rem;">Masked with [PHONE]</div>'
                    f'</div>'
                    f'<div class="stat-card" style="padding: 1rem;">'
                    f'<div class="stat-value" style="font-size: 1.5rem; color: #c084fc;">{redactions.get("student_id", 0):,}</div>'
                    f'<div class="stat-label">PRNs / Roll IDs</div>'
                    f'<div style="font-size: 0.72rem; color: #667085; margin-top: 0.2rem;">Masked with [STUDENT_ID]</div>'
                    f'</div>'
                    f'<div class="stat-card" style="padding: 1rem;">'
                    f'<div class="stat-value" style="font-size: 1.5rem; color: #34d399;">{redactions.get("email", 0):,}</div>'
                    f'<div class="stat-label">Personal Emails</div>'
                    f'<div style="font-size: 0.72rem; color: #667085; margin-top: 0.2rem;">Masked with [EMAIL]</div>'
                    f'</div>'
                    f'<div class="stat-card" style="padding: 1rem;">'
                    f'<div class="stat-value" style="font-size: 1.5rem; color: #f87171;">{redactions.get("credential", 0):,}</div>'
                    f'<div class="stat-label">Credentials</div>'
                    f'<div style="font-size: 0.72rem; color: #667085; margin-top: 0.2rem;">Masked with [CREDENTIAL]</div>'
                    f'</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown("""
                <div style="background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 10px; padding: 1.25rem 1.4rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #4F46E5; font-weight: 700;">Sanitization Protocols Applied:</h5>
                <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.88rem; color: #344054; line-height: 1.6;">
                <li><strong>Administrative Front-Matter:</strong> Certificates of completion, candidate declarations, acknowledgments, and plagiarism reports are automatically filtered prior to vectorization.</li>
                <li><strong>PII Masking:</strong> Student phone numbers, email addresses, institutional roll/PRN numbers, and local guide/student names (discovered via context cues + spaCy Named Entity Recognition) are replaced with deterministic tags (<code>[NAME]</code>, <code>[STUDENT_ID]</code>, <code>[PHONE]</code>, <code>[EMAIL]</code>).</li>
                <li><strong>Scholarly Preservation:</strong> Academic literature citations (e.g. <code>[1]</code>, <code>[2]</code>, <code>Vaswani et al. (2017)</code>) in Literature Review and Reference sections are preserved intact for accurate RAG synthesis.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

        # Admin Module 3: Institutional Corpus Telemetry
        elif admin_nav == "📊 Institutional Corpus Telemetry":
            render_hero_header(
                icon="📊",
                title="Institutional Corpus Telemetry & Search",
                subtitle="Corpus telemetry, passage token distributions, and document search across engineering cohorts.",
                badges=[
                    (f"📊 {db_count:,} Passages", "meta-pill-blue")
                ]
            )

            # KPI Row
            st.markdown(
                '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">'
                f'<div class="stat-card">'
                f'<div class="stat-icon" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa;">📄</div>'
                f'<div class="stat-value">{man_data.get("total_files", 0)}</div>'
                f'<div class="stat-label">Raw Documents</div>'
                f'</div>'
                f'<div class="stat-card">'
                f'<div class="stat-icon" style="background: rgba(16, 185, 129, 0.2); color: #34d399;">🧹</div>'
                f'<div class="stat-value">{rep_data.get("cleaned_documents", 0)}</div>'
                f'<div class="stat-label">Sanitized Docs</div>'
                f'</div>'
                f'<div class="stat-card">'
                f'<div class="stat-icon" style="background: rgba(139, 92, 246, 0.2); color: #c084fc;">🧬</div>'
                f'<div class="stat-value">{db_count:,}</div>'
                f'<div class="stat-label">Vector Chunks</div>'
                f'</div>'
                f'<div class="stat-card">'
                f'<div class="stat-icon" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24;">🛡️</div>'
                f'<div class="stat-value">{total_redactions:,}</div>'
                f'<div class="stat-label">PII Redactions</div>'
                f'</div>'
                '</div>',
                unsafe_allow_html=True
            )

            if not df_chunks.empty:
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("#### 📊 Chunks by Department")
                    dept_counts = df_chunks["department"].value_counts().reset_index()
                    dept_counts.columns = ["Department", "Chunks"]
                    fig_dept = px.bar(
                        dept_counts,
                        x="Department",
                        y="Chunks",
                        color="Department",
                        color_discrete_sequence=["#4F46E5", "#6366F1", "#818CF8", "#A5B4FC", "#C7D2FE"],
                        text="Chunks"
                    )
                    fig_dept.update_traces(textposition='outside', marker_line_width=0)
                    apply_plotly_theme(fig_dept, height=300)
                    fig_dept.update_layout(showlegend=False)
                    st.plotly_chart(fig_dept, use_container_width=True)

                with col_right:
                    st.markdown("#### 📑 Document Category Distribution")
                    cat_counts = df_chunks["category"].value_counts().reset_index()
                    cat_counts.columns = ["Category", "Chunks"]
                    fig_cat = px.pie(
                        cat_counts,
                        names="Category",
                        values="Chunks",
                        hole=0.52,
                        color_discrete_sequence=["#4F46E5", "#6366F1", "#818CF8", "#A5B4FC", "#C7D2FE"]
                    )
                    fig_cat.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
                    apply_plotly_theme(fig_cat, height=300)
                    fig_cat.update_layout(showlegend=False)
                    st.plotly_chart(fig_cat, use_container_width=True)

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("#### 📏 Token Size Compliance (Hard Max $\\le 520$)")
                    fig_tok = px.histogram(
                        df_chunks,
                        x="token_count",
                        nbins=30,
                        color_discrete_sequence=["#4F46E5"]
                    )
                    fig_tok.add_vline(x=520, line_dash="dash", line_color="#ef4444", annotation_text="Hard Max (520 tokens)", annotation_font_color="#ef4444")
                    apply_plotly_theme(fig_tok, height=280)
                    st.plotly_chart(fig_tok, use_container_width=True)

                with col_t2:
                    st.markdown("#### 🔬 Section Boundary Breakdown")
                    sec_counts = df_chunks["section"].value_counts().head(8).reset_index()
                    sec_counts.columns = ["Section", "Chunks"]
                    fig_sec = px.bar(
                        sec_counts,
                        x="Chunks",
                        y="Section",
                        orientation="h",
                        color_discrete_sequence=["#8b5cf6"]
                    )
                    fig_sec.update_traces(marker_line_width=0)
                    apply_plotly_theme(fig_sec, height=280)
                    fig_sec.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_sec, use_container_width=True)

            # Passage Search (Admin Archive Exploration)
            with st.expander("🔎 Deep Passage Archive Search", expanded=False):
                s_col1, s_col2 = st.columns([3, 1])
                with s_col1:
                    a_search = st.text_input("Passage Search Query", placeholder="e.g. CNN accuracy plant disease latency")
                with s_col2:
                    a_top_k = st.slider("Top Chunks", min_value=3, max_value=12, value=5)

                if a_search:
                    search_res = store.query_similar(query_text=a_search, n_results=a_top_k)
                    for r in search_res:
                        st.markdown(render_citation_card({
                            "source_file": r["metadata"].get("source_file"),
                            "department": r["metadata"].get("department"),
                            "group_id": r["metadata"].get("group_id"),
                            "section": r["metadata"].get("section"),
                            "similarity_score": f"{r.get('similarity', 0.0):.2f}",
                            "excerpt": r.get("text", "")[:280]
                        }), unsafe_allow_html=True)
