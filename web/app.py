"""
Project-IQ: Enterprise Academic Project Intelligence Platform
Immersive Edition — per-module identity, motion design, oversized typography
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
import streamlit.components.v1 as components

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.svg"

from src.config import config
from src.logger import get_logger
from src.vector_store import ProjectIQVectorStore
from src.gap_analyzer import GapAnalyzer, GAP_INDICATORS
from src.team_matcher import TeamMatcher, DOMAIN_TAXONOMY
from src.duplicate_detector import DuplicateDetector
from src.quality_scorer import QualityScorer
from src.rag_chat import RAGChatEngine
from src.clean import is_ner_available

logger = get_logger("project_iq.web")

# Page Configuration
st.set_page_config(
    page_title="Project-IQ — Academic Intelligence Platform",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# Global Design System — Motion, Typography, Per-Module Identity
# =============================================================


def icon_svg(name: str, size: int = 24, color: str = "currentColor", stroke_width: float = 2) -> str:
    """Returns unique, animated, high-fidelity SVG emblems and icons for every module."""
    icons = {
        # Brand Logo (Geometric Crest with Floating Academic Cap)
        "brand_crest": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" fill="none">
            <rect width="48" height="48" rx="14" fill="url(#brand-grad)"/>
            <path d="M24 12L38 19L24 26L10 19L24 12Z" fill="white" fill-opacity="0.95"/>
            <path d="M14 22V30C14 34.5 18.5 37 24 37C29.5 37 34 34.5 34 30V22" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="38" cy="27" r="2.5" fill="#FCD34D"/>
            <path d="M38 29.5V34" stroke="#FCD34D" stroke-width="2" stroke-linecap="round"/>
            <defs>
                <linearGradient id="brand-grad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#4F46E5"/>
                    <stop offset="1" stop-color="#7C3AED"/>
                </linearGradient>
            </defs>
        </svg>""",

        # Module 1: AI Advisor & Ideator (Neural Synapse Core with radiating rings)
        "chat_advisor": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="20" fill="url(#m1-bg)" fill-opacity="0.15"/>
            <circle cx="24" cy="24" r="16" stroke="url(#m1-glow)" stroke-width="2" stroke-dasharray="3 3"/>
            <rect x="13" y="14" width="22" height="17" rx="6" fill="url(#m1-grad)"/>
            <circle cx="19" cy="22.5" r="2" fill="white"/>
            <circle cx="29" cy="22.5" r="2" fill="white"/>
            <path d="M22 26C22.8 27.2 25.2 27.2 26 26" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <path d="M19 31L16 35V31H19Z" fill="url(#m1-grad)"/>
            <path d="M24 7V11M37 24H41M7 24H11" stroke="#6366F1" stroke-width="2.5" stroke-linecap="round"/>
            <defs>
                <linearGradient id="m1-grad" x1="13" y1="14" x2="35" y2="35" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#4F46E5"/>
                    <stop offset="1" stop-color="#8B5CF6"/>
                </linearGradient>
                <linearGradient id="m1-glow" x1="8" y1="8" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#6366F1"/>
                    <stop offset="1" stop-color="#A855F7"/>
                </linearGradient>
                <radialGradient id="m1-bg" cx="50%" cy="50%" r="50%">
                    <stop stop-color="#6366F1"/>
                    <stop offset="1" stop-color="transparent"/>
                </radialGradient>
            </defs>
        </svg>""",

        # Module 2: Originality & Duplicate Guard (Radar Scanning Shield)
        "shield_check": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" fill="none">
            <path d="M24 6L39 12V24C39 33.5 32.5 40 24 43C15.5 40 9 33.5 9 24V12L24 6Z" fill="url(#m2-grad)"/>
            <path d="M24 11V37M13 24H35" stroke="white" stroke-opacity="0.3" stroke-width="1.5"/>
            <circle cx="24" cy="24" r="9" stroke="white" stroke-opacity="0.4" stroke-width="1.5"/>
            <path d="M18 24L22 28L30 19" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
            <defs>
                <linearGradient id="m2-grad" x1="9" y1="6" x2="39" y2="43" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#D97706"/>
                    <stop offset="1" stop-color="#E11D48"/>
                </linearGradient>
            </defs>
        </svg>""",

        # Module 3: Research Gap Finder (Frontier Astrolabe & Compass)
        "compass_gap": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="18" fill="url(#m3-grad)"/>
            <circle cx="24" cy="24" r="13" stroke="white" stroke-opacity="0.3" stroke-width="2"/>
            <polygon points="31,17 26,26 17,31 22,22" fill="white"/>
            <circle cx="24" cy="24" r="2.5" fill="#047857"/>
            <circle cx="36" cy="12" r="2" fill="#34D399"/>
            <defs>
                <linearGradient id="m3-grad" x1="6" y1="6" x2="42" y2="42" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#059669"/>
                    <stop offset="1" stop-color="#0D9488"/>
                </linearGradient>
            </defs>
        </svg>""",

        # Module 4: Project Quality Scorer (Diamond Precision Gauge)
        "gauge_score": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" fill="none">
            <rect width="48" height="48" rx="14" fill="url(#m4-grad)"/>
            <path d="M12 33A15 15 0 1 1 36 33" stroke="white" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="4 2"/>
            <path d="M24 27L31 19" stroke="#FEF08A" stroke-width="3" stroke-linecap="round"/>
            <circle cx="24" cy="27" r="4" fill="white"/>
            <defs>
                <linearGradient id="m4-grad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#0284C7"/>
                    <stop offset="1" stop-color="#2563EB"/>
                </linearGradient>
            </defs>
        </svg>""",

        # Module 5: Team Skill Matcher (Synergy Tri-Node Constellation)
        "team_users": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" fill="none">
            <rect width="48" height="48" rx="14" fill="url(#m5-grad)"/>
            <line x1="24" y1="14" x2="15" y2="31" stroke="white" stroke-opacity="0.5" stroke-width="2"/>
            <line x1="24" y1="14" x2="33" y2="31" stroke="white" stroke-opacity="0.5" stroke-width="2"/>
            <line x1="15" y1="31" x2="33" y2="31" stroke="white" stroke-opacity="0.5" stroke-width="2"/>
            <circle cx="24" cy="14" r="5" fill="white"/>
            <circle cx="15" cy="31" r="5" fill="white"/>
            <circle cx="33" cy="31" r="5" fill="white"/>
            <defs>
                <linearGradient id="m5-grad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#9333EA"/>
                    <stop offset="1" stop-color="#C026D3"/>
                </linearGradient>
            </defs>
        </svg>""",

        # Admin Vault (Quantum Security Gate)
        "vault_lock": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" fill="none">
            <rect width="48" height="48" rx="14" fill="url(#admin-grad)"/>
            <rect x="14" y="20" width="20" height="16" rx="4" fill="white"/>
            <path d="M18 20V15C18 11.7 20.7 9 24 9C27.3 9 30 11.7 30 15V20" stroke="white" stroke-width="3" stroke-linecap="round"/>
            <circle cx="24" cy="27" r="2.5" fill="#1E3A8A"/>
            <path d="M24 29.5V32" stroke="#1E3A8A" stroke-width="2" stroke-linecap="round"/>
            <defs>
                <linearGradient id="admin-grad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#1D4ED8"/>
                    <stop offset="1" stop-color="#4338CA"/>
                </linearGradient>
            </defs>
        </svg>""",

        # General utility icons
        "zap": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        "sparkles": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
        "target": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "search": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
        "bar_chart": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/></svg>',
        "database": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        "activity_pulse": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
        "file_text": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>',
        "cpu_chip": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>',
        "check_circle": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "academic_cap": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>',
        "info": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
        "database_sync": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>'
    }
    return icons.get(name, icons["sparkles"])

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --a1: #6366F1;
    --a2: #A855F7;
    --bg: #0D111A;
    --panel: #151B28;
    --panel-card: #182030;
    --line: rgba(255, 255, 255, 0.08);
    --line-glow: rgba(99, 102, 241, 0.4);
    --txt: #FFFFFF;
    --txt-muted: #94A3B8;
}

/* ---------- Canvas Background & Base Styling ---------- */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}

[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(1200px circle at 15% -8%, rgba(99, 102, 241, 0.16) 0%, transparent 60%),
        radial-gradient(1000px circle at 88% 12%, rgba(168, 85, 247, 0.13) 0%, transparent 55%),
        radial-gradient(900px circle at 50% 95%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
        #0D111A !important;
    color: var(--txt) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ---------- Sidebar Styling ---------- */
[data-testid="stSidebar"] {
    background: 
        radial-gradient(380px circle at 0% 0%, rgba(99,102,241,0.18), transparent 65%),
        radial-gradient(420px circle at 100% 100%, rgba(168,85,247,0.15), transparent 60%),
        linear-gradient(180deg, #131825 0%, #0D111A 100%) !important;
    border-right: 1px solid var(--line) !important;
    box-shadow: 4px 0 28px rgba(0, 0, 0, 0.4) !important;
}

/* Brand typography in sidebar */
.side-brand-word {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    line-height: 1.15;
    background: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 80%, #C4B5FD 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.025em;
}

.side-chip {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 0.86rem;
    font-weight: 700;
    padding: 0.6rem 1rem;
    border-radius: 14px;
    margin-bottom: 1.25rem;
    letter-spacing: 0.02em;
}
.side-chip .chip-dot {
    width: 10px; height: 10px; border-radius: 50%;
    flex-shrink: 0;
    animation: live-pulse 2s ease-in-out infinite;
}
.side-chip.green { background: rgba(16,185,129,0.15); color: #34D399 !important; border: 1px solid rgba(16,185,129,0.4); }
.side-chip.green .chip-dot { background: #10B981; box-shadow: 0 0 12px rgba(16,185,129,0.9); }
.side-chip.amber { background: rgba(245,158,11,0.15); color: #FBBF24 !important; border: 1px solid rgba(245,158,11,0.4); }
.side-chip.amber .chip-dot { background: #F59E0B; box-shadow: 0 0 12px rgba(245,158,11,0.9); }

@keyframes live-pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.45); opacity: 0.65; }
}

.side-divider {
    border: none;
    height: 1px;
    margin: 1.3rem 0;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.45), rgba(168,85,247,0.3), transparent);
}

.sidebar-section-title {
    font-size: 0.76rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #64748B !important;
    margin: 1.25rem 0 0.5rem 0 !important;
}

/* Sidebar navigation buttons */
[data-testid="stSidebar"] .stButton > button {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    padding: 0.78rem 1.05rem !important;
    min-height: 48px !important;
    border-radius: 14px !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    font-size: 1.02rem !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.45rem !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    text-align: left !important;
    box-shadow: none !important;
    cursor: pointer !important;
    position: relative !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
    color: #FFFFFF !important;
    transform: translateX(4px) !important;
}

[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    color: #CBD5E1 !important;
    line-height: 1.35 !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, color-mix(in srgb, var(--a1) 85%, #151B28) 0%, color-mix(in srgb, var(--a2) 75%, #151B28) 100%) !important;
    border: 1px solid color-mix(in srgb, var(--a1) 80%, #FFFFFF) !important;
    color: #FFFFFF !important;
    font-weight: 750 !important;
    box-shadow: 0 6px 20px color-mix(in srgb, var(--a1) 35%, transparent) !important;
    transform: translateX(4px) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[kind="primary"] [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    font-weight: 750 !important;
}

.side-foot {
    background: linear-gradient(160deg, rgba(99,102,241,0.08) 0%, rgba(21,27,40,0.8) 100%);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    font-size: 0.84rem !important;
    color: #94A3B8 !important;
    line-height: 1.6;
    margin-top: 1.6rem;
}
.side-foot b { color: #FFFFFF !important; font-weight: 700; }

/* ---------- Main Container & Module Transition ---------- */
.main .block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1440px !important;
    padding: 2rem 2.8rem 5rem 2.8rem !important;
}

@keyframes fullWindowReveal {
    from { opacity: 0.85; }
    to { opacity: 1; }
}

.main-content-animated {
    animation: fullWindowReveal 0.05s ease-out both !important;
}

/* ---------- Typography Hierarchy (Balanced, Modern, Clear) ---------- */
h1 {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.035em !important;
    color: #FFFFFF !important;
    line-height: 1.15 !important;
}
h2 {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    color: #FFFFFF !important;
    line-height: 1.25 !important;
}
h3 {
    font-size: 1.38rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.015em !important;
    color: #F8FAFC !important;
    line-height: 1.3 !important;
}
h4 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #E2E8F0 !important;
    line-height: 1.35 !important;
}

/* Body text, Paragraphs, Lists */
p,
li,
.stMarkdown p,
[data-testid="stMarkdownContainer"] > p,
[data-testid="stMarkdownContainer"] > ul > li,
[data-testid="stMarkdownContainer"] > ol > li {
    font-size: 1.05rem !important;
    line-height: 1.68 !important;
    color: #CBD5E1 !important;
}

/* Form Widget Labels */
[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] p {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.35rem !important;
    line-height: 1.4 !important;
}

/* Captions / Microcopy */
.stCaption, small {
    font-size: 0.88rem !important;
    color: #94A3B8 !important;
    line-height: 1.5 !important;
}

/* ---------- Hero Module Banner ---------- */
.piq-hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 2.8rem 3rem 2.5rem 3rem;
    margin-bottom: 2.2rem;
    background: linear-gradient(135deg, color-mix(in srgb, var(--a1) 22%, #151B28) 0%, #151B28 55%, #0D111A 100%);
    border: 1px solid color-mix(in srgb, var(--a1) 35%, transparent);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.08);
}

.piq-hero::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--a1), var(--a2));
}

.piq-hero .hero-orb {
    display: none !important;
}

.piq-hero .hero-icon-box {
    width: 64px !important;
    height: 64px !important;
    min-width: 64px !important;
    min-height: 64px !important;
    border-radius: 18px !important;
    background: linear-gradient(135deg, color-mix(in srgb, var(--a1) 35%, #151B28), #182030) !important;
    border: 1.5px solid color-mix(in srgb, var(--a1) 55%, transparent) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 10px 24px color-mix(in srgb, var(--a1) 35%, transparent) !important;
    flex-shrink: 0 !important;
}

.piq-hero .hero-title,
.piq-hero h1,
h1.hero-title,
[data-testid="stMarkdownContainer"] h1.hero-title,
[data-testid="stMainBlockContainer"] .piq-hero h1 {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.035em !important;
    line-height: 1.15 !important;
    margin: 0 !important;
    padding: 0 !important;
    color: #FFFFFF !important;
    text-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
}

.piq-hero .hero-underline {
    width: 90px;
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--a1), var(--a2));
    margin-top: 0.9rem;
    box-shadow: 0 0 16px color-mix(in srgb, var(--a1) 60%, transparent);
}

.piq-hero .hero-desc {
    font-size: 1.12rem !important;
    color: #CBD5E1 !important;
    margin-top: 1rem !important;
    line-height: 1.65 !important;
    max-width: 60rem;
    font-weight: 450;
}

.piq-hero .hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.80rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: color-mix(in srgb, var(--a1) 70%, #FFF) !important;
    background: color-mix(in srgb, var(--a1) 16%, transparent);
    border: 1px solid color-mix(in srgb, var(--a1) 32%, transparent);
    padding: 0.4rem 1rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

/* ---------- Section Headers ---------- */
.piq-section {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.025em;
    margin: 2.2rem 0 1.2rem 0 !important;
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.piq-section .sec-dot {
    width: 14px; height: 14px; border-radius: 4px;
    background: linear-gradient(135deg, var(--a1), var(--a2));
    box-shadow: 0 0 14px color-mix(in srgb, var(--a1) 75%, transparent);
    animation: dot-pulse 2.2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes dot-pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.35); opacity: 0.75; }
}

/* ---------- Cards (Deep Velvet Titanium) ---------- */
.piq-card {
    background: #182030 !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    padding: 1.75rem 1.9rem !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3) !important;
    transition: transform 0.28s cubic-bezier(.22,1,.36,1), border-color 0.28s ease, box-shadow 0.28s ease !important;
    height: 100%;
}
.piq-card:hover {
    transform: translateY(-4px) scale(1.01) !important;
    border-color: color-mix(in srgb, var(--a1) 50%, transparent) !important;
    box-shadow: 0 18px 40px color-mix(in srgb, var(--a1) 20%, rgba(0,0,0,0.45)) !important;
}

.piq-card .card-kicker {
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: color-mix(in srgb, var(--a1) 70%, #FFF) !important;
    margin-bottom: 0.5rem;
}

.piq-card .card-title {
    font-size: 1.28rem !important;
    font-weight: 750 !important;
    color: #FFFFFF !important;
    margin-bottom: 0.6rem;
    letter-spacing: -0.015em;
    line-height: 1.35;
}

.piq-card .card-body {
    font-size: 0.98rem !important;
    color: #CBD5E1 !important;
    line-height: 1.62 !important;
}

/* ---------- Stat Cards ---------- */
.stat-card {
    background: #182030 !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    padding: 1.75rem 1.85rem !important;
    text-align: center !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3) !important;
    transition: transform 0.28s ease, box-shadow 0.28s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
.stat-card::after {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, var(--a1), var(--a2));
}
.stat-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 18px 40px color-mix(in srgb, var(--a1) 20%, rgba(0,0,0,0.45)) !important;
}

.stat-card .stat-value {
    font-size: 2.75rem !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    line-height: 1.1;
    background: linear-gradient(135deg, #FFFFFF 30%, color-mix(in srgb, var(--a1) 70%, #FFF));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}
.stat-card .stat-label {
    font-size: 0.82rem !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 750;
    margin-top: 0.5rem;
}

/* ---------- Progress Bars ---------- */
.pbar-track {
    background: #0E131E;
    border: 1px solid var(--line);
    border-radius: 999px;
    height: 14px;
    overflow: hidden;
}
.pbar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--a1), var(--a2));
    width: 0;
    animation: pbar-grow 1.2s cubic-bezier(.22,1,.36,1) forwards;
    box-shadow: 0 0 14px color-mix(in srgb, var(--a1) 50%, transparent);
}
@keyframes pbar-grow { to { width: var(--w, 0%); } }

.pbar-row { margin-bottom: 1.2rem; }
.pbar-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.45rem;
}
.pbar-name { font-weight: 750; font-size: 1.02rem; color: #FFFFFF !important; }
.pbar-pct {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.98rem;
    color: color-mix(in srgb, var(--a1) 70%, #FFF) !important;
}

/* ---------- Metadata Pills ---------- */
.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 9999px;
    font-size: 0.84rem;
    font-weight: 750;
    letter-spacing: 0.02em;
}
.meta-pill-blue   { background: rgba(99,102,241,0.18);  color: #A5B4FC !important; border: 1px solid rgba(99,102,241,0.4); }
.meta-pill-green  { background: rgba(16,185,129,0.18);  color: #6EE7B7 !important; border: 1px solid rgba(16,185,129,0.4); }
.meta-pill-amber  { background: rgba(245,158,11,0.18);  color: #FCD34D !important; border: 1px solid rgba(245,158,11,0.4); }
.meta-pill-red    { background: rgba(239,68,68,0.18);   color: #FCA5A5 !important; border: 1px solid rgba(239,68,68,0.4); }
.meta-pill-cyan   { background: rgba(6,182,212,0.18);   color: #67E8F9 !important; border: 1px solid rgba(6,182,212,0.4); }
.meta-pill-purple { background: rgba(168,85,247,0.18);  color: #D8B4FE !important; border: 1px solid rgba(168,85,247,0.4); }
.meta-pill-slate  { background: rgba(148,163,184,0.16); color: #CBD5E1 !important; border: 1px solid rgba(148,163,184,0.35); }

/* ---------- Buttons & Prompts ---------- */
.stButton > button {
    background-color: #182030 !important;
    color: #F8FAFC !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    font-weight: 650 !important;
    font-size: 1.02rem !important;
    line-height: 1.45 !important;
    transition: all 0.24s cubic-bezier(.22,1,.36,1) !important;
    width: 100% !important;
    min-height: 76px !important;
    text-align: left !important;
    white-space: normal !important;
    word-break: break-word !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2) !important;
}

.stButton > button:hover {
    border-color: var(--a1) !important;
    box-shadow: 0 10px 26px color-mix(in srgb, var(--a1) 28%, transparent) !important;
    transform: translateY(-3px) scale(1.008) !important;
    color: #FFFFFF !important;
    background-color: #1F2A3F !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--a1) 0%, var(--a2) 100%) !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.08rem !important;
    border: none !important;
    min-height: 3.2rem !important;
    padding: 0.85rem 2rem !important;
    text-align: center !important;
    letter-spacing: 0.02em;
    box-shadow: 0 10px 28px color-mix(in srgb, var(--a1) 45%, transparent) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 16px 38px color-mix(in srgb, var(--a1) 60%, transparent) !important;
    transform: translateY(-3px) scale(1.015) !important;
}

/* ---------- Inputs, Textareas, Selectboxes ---------- */
.stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
    background-color: #0E131E !important;
    color: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    font-size: 1.02rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.18) !important;
    transition: border-color 0.22s ease, box-shadow 0.22s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, div[data-baseweb="select"] > div:focus {
    border-color: var(--a1) !important;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--a1) 25%, transparent) !important;
}

/* ---------- Tabs Modernization ---------- */
[data-baseweb="tab-list"] {
    gap: 0.6rem !important;
    background-color: #0E131E !important;
    padding: 0.45rem !important;
    border-radius: 16px !important;
    border: 1px solid var(--line) !important;
}
[data-baseweb="tab"] {
    border-radius: 12px !important;
    padding: 0.7rem 1.6rem !important;
    font-weight: 650 !important;
    color: #94A3B8 !important;
    font-size: 1.02rem !important;
    transition: all 0.2s ease !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, color-mix(in srgb, var(--a1) 40%, #182030), #182030) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25) !important;
    border: 1px solid color-mix(in srgb, var(--a1) 45%, transparent) !important;
    font-weight: 750 !important;
}

/* ---------- Chat Messages ---------- */
[data-testid="stChatMessage"] {
    background-color: #151B28 !important;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    padding: 1.25rem 1.45rem !important;
    margin-bottom: 1rem !important;
}
[data-testid="stChatMessage"] p {
    font-size: 1.04rem !important;
    line-height: 1.65 !important;
    color: #E2E8F0 !important;
}

/* ---------- Code & Monospace Blocks ---------- */
code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
}
</style>
""", unsafe_allow_html=True)




def safe_html(text: Any) -> str:
    """Escapes text safely for direct interpolation into HTML templates."""
    if text is None:
        return ""
    return html.escape(str(text))


MODULE_ACCENTS = {
    "advisor":     ("#4F46E5", "#7C3AED"),   # Royal Indigo → Electric Violet
    "originality": ("#D97706", "#E11D48"),   # Sunset Amber → Rose
    "gaps":        ("#059669", "#0D9488"),   # Vibrant Emerald → Teal
    "quality":     ("#0284C7", "#2563EB"),   # Electric Cyan → Blue
    "team":        ("#9333EA", "#C026D3"),   # Purple → Fuchsia
    "admin":       ("#1D4ED8", "#4338CA"),   # Sapphire → Deep Indigo
}



def set_module_theme(theme: str):
    """Swaps the page accent variables so every element adopts the module identity."""
    a1, a2 = MODULE_ACCENTS.get(theme, MODULE_ACCENTS["advisor"])
    st.markdown(f"<style>:root {{ --a1: {a1}; --a2: {a2}; }}</style>", unsafe_allow_html=True)


def render_hero_header(
    icon_name: str,
    title: str,
    subtitle: str,
    badges: Optional[List[tuple[str, str]]] = None,
    eyebrow: str = "",
):
    """Renders an animated hero banner with oversized gradient typography and clean SVG icon."""
    badges_html = ""
    if badges:
        pills = [f'<span class="meta-pill {safe_html(c)}" style="animation-delay:{i * 0.08:.2f}s">{safe_html(t)}</span>'
                 for i, (t, c) in enumerate(badges)]
        badges_html = f'<div style="display: flex; gap: 0.6rem; margin-top: 1.25rem; flex-wrap: wrap;">{" ".join(pills)}</div>'

    eyebrow_html = f'<div class="hero-eyebrow">{safe_html(eyebrow)}</div>' if eyebrow else ""
    icon_markup = icon_svg(icon_name, size=40, color="var(--a1)", stroke_width=2.2)

    st.markdown(
        f'<div class="piq-hero">'
        f'<div style="position: relative; z-index: 1;">'
        f'{eyebrow_html}'
        f'<div style="display: flex; align-items: center; gap: 1.25rem; margin-bottom: 0.4rem;">'
        f'<div class="hero-icon-box">{icon_markup}</div>'
        f'<h1 class="hero-title">{safe_html(title)}</h1>'
        f'</div>'
        f'<div class="hero-underline"></div>'
        f'<p class="hero-desc">{safe_html(subtitle)}</p>'
        f'{badges_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def section_header(text: str, icon_name: str = ""):
    icon_html = f'<span style="color:var(--a1); display:inline-flex; align-items:center;">{icon_svg(icon_name, size=22, color="var(--a1)")}</span>' if icon_name else '<span class="sec-dot"></span>'
    st.markdown(
        f'<div class="piq-section">{icon_html}<span>{safe_html(text)}</span></div>',
        unsafe_allow_html=True
    )


def stat_card(value: str, label: str, delay: float = 0.0):
    st.markdown(
        f'<div class="stat-card fade-up" style="animation-delay:{delay:.2f}s">'
        f'<div class="stat-value">{safe_html(value)}</div>'
        f'<div class="stat-label">{safe_html(label)}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def animated_bar(name: str, pct: float, delay: float = 0.0, suffix: str = "%"):
    pct = max(0.0, min(100.0, float(pct)))
    st.markdown(
        f'<div class="pbar-row fade-up" style="animation-delay:{delay:.2f}s">'
        f'<div class="pbar-head"><span class="pbar-name">{safe_html(name)}</span>'
        f'<span class="pbar-pct">{pct:g}{safe_html(suffix)}</span></div>'
        f'<div class="pbar-track"><div class="pbar-fill" style="--w: {pct:.1f}%; animation-delay: {delay:.2f}s;"></div></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def info_card(kicker: str, title: str, body: str, delay: float = 0.0):
    st.markdown(
        f'<div class="piq-card fade-up" style="animation-delay:{delay:.2f}s">'
        f'<div class="card-kicker">{safe_html(kicker)}</div>'
        f'<div class="card-title">{safe_html(title)}</div>'
        f'<div class="card-body">{safe_html(body)}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def verdict_pill(level: str) -> tuple[str, str]:
    return {
        "success": ("High Originality", "meta-pill-green"),
        "warning": ("Moderate Overlap", "meta-pill-amber"),
        "danger":  ("High Overlap", "meta-pill-red"),
        "info":    ("Baseline Evaluation", "meta-pill-blue"),
    }.get(level, ("Report Analysis", "meta-pill-blue"))


def score_gauge(value: float, label: str, color_hex: str, max_val: float = 100.0) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 42, "color": "#FFFFFF", "family": "Plus Jakarta Sans"}},
        title={"text": label, "font": {"size": 42, "color": "#94A3B8", "family": "Plus Jakarta Sans"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#64748B", "tickfont": {"color": "#94A3B8", "size": 12}},
            "bar": {"color": color_hex, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_val * 0.5], "color": "#182030"},
                {"range": [max_val * 0.5, max_val * 0.75], "color": "#202B40"},
                {"range": [max_val * 0.75, max_val], "color": "#293650"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 4},
                "thickness": 0.85,
                "value": value,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#FFFFFF"),
        height=350,
        margin=dict(t=40, b=10, l=30, r=30),
        uirevision=True,
    )
    return fig


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
        f'<div class="piq-card" style="margin-bottom: 0.9rem; padding: 1.25rem 1.45rem !important;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">'
        f'<span style="font-weight: 750; color: #FFFFFF; font-size: 1.05rem;">[{idx}] {src_file}</span>'
        f'<div style="display: flex; gap: 0.4rem; flex-wrap: wrap;">'
        f'<span class="meta-pill meta-pill-blue">{dept} / {group}</span>'
        f'<span class="meta-pill meta-pill-purple">Section: {sec}</span>'
        f'<span class="meta-pill meta-pill-green">Affinity: {aff}</span>'
        f'</div>'
        f'</div>'
        f'<blockquote style="margin: 0.4rem 0 0 0; padding: 0.65rem 0.95rem; background: #0E131E; border-left: 3px solid var(--a1); border-radius: 8px; font-size: 0.96rem; color: #CBD5E1; line-height: 1.6;">'
        f'"{excerpt}"'
        f'</blockquote>'
        f'</div>'
    )


def apply_plotly_theme(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#FFFFFF", size=13),
        margin=dict(t=30, b=30, l=30, r=30),
        height=height,
        xaxis=dict(gridcolor="#1F2937", zerolinecolor="#374151"),
        yaxis=dict(gridcolor="#1F2937", zerolinecolor="#374151"),
    )


def extract_text_from_upload(uploaded_file) -> str:
    """Extracts text from uploaded draft (.txt, .pdf, .docx)."""
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".txt"):
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif fname.endswith(".pdf"):
            import fitz
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif fname.endswith(".docx"):
            import docx, io
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

    rep_data = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
    man_data = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}

    return pd.DataFrame(chunks_data) if chunks_data else pd.DataFrame(), rep_data, man_data


df_chunks, rep_data, man_data = load_telemetry_data()
@st.cache_data
def get_cached_db_count():
    try:
        return store.count()
    except Exception:
        return 0

db_count = get_cached_db_count()
total_redactions = sum(rep_data.get("total_redactions", {}).values()) if (rep_data and "total_redactions" in rep_data) else 0

# Check background Gemini API key and spaCy NER
if not os.getenv("GEMINI_API_KEY"):
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        # Streamlit raises StreamlitSecretNotFoundError if secrets.toml does not exist
        pass

has_gemini = bool(os.getenv("GEMINI_API_KEY") or config.gemini_api_key)
ner_active = is_ner_available()

# -------------------------------------------------------------
# Sidebar: Logo & Real Button Navigation
# -------------------------------------------------------------
st.sidebar.markdown(
    '<div style="display: flex; align-items: center; gap: 0.85rem; padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 1rem;">'
    f'<div style="width: 50px; height: 50px; border-radius: 16px; background: linear-gradient(135deg, #6366F1, #8B5CF6); display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 24px rgba(99, 102, 241, 0.45); color: white;">{icon_svg("academic_cap", size=26, color="#FFFFFF", stroke_width=2.2)}</div>'
    '<div>'
    '<div class="side-brand-word">Project-IQ</div>'
    '<div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 2px;">Academic Intelligence</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# Initialize Session State Navigation
if "portal_mode" not in st.session_state:
    st.session_state.portal_mode = "Student Workspace"

if "student_nav" not in st.session_state:
    st.session_state.student_nav = "AI Project Advisor & Novel Ideator"

if "admin_nav" not in st.session_state:
    st.session_state.admin_nav = "Institutional Corpus Telemetry"

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = True


def set_portal_mode(pm: str):
    st.session_state.portal_mode = pm


def set_student_module(m: str):
    st.session_state.student_nav = m


def set_admin_module(m: str):
    st.session_state.admin_nav = m


# Workspace Selection Buttons
st.sidebar.markdown('<div class="sidebar-section-title">PORTAL WORKSPACE</div>', unsafe_allow_html=True)
is_std = (st.session_state.portal_mode == "Student Workspace")
st.sidebar.button(
    "Student Workspace",
    key="btn_ws_student",
    type="primary" if is_std else "secondary",
    use_container_width=True,
    on_click=set_portal_mode,
    args=("Student Workspace",)
)

is_adm = (st.session_state.portal_mode in ("Institutional Admin Center", "Institutional Admin Vault"))
st.sidebar.button(
    "Institutional Admin Center",
    key="btn_ws_admin",
    type="primary" if is_adm else "secondary",
    use_container_width=True,
    on_click=set_portal_mode,
    args=("Institutional Admin Center",)
)

portal_mode = st.session_state.portal_mode

# Module Navigation Buttons (with FULL Titles on Buttons)
if portal_mode == "Student Workspace":
    st.sidebar.markdown('<div class="sidebar-section-title">INTELLIGENCE MODULES</div>', unsafe_allow_html=True)
    
    MODULES = [
        "AI Project Advisor & Novel Ideator",
        "Originality & Duplicate Checker",
        "Research Gap Finder",
        "Project Quality Scorer",
        "Student Team Skill Matcher"
    ]
    
    for mod in MODULES:
        is_active = (st.session_state.student_nav == mod)
        st.sidebar.button(
            mod,
            key=f"nav_btn_{mod}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            on_click=set_student_module,
            args=(mod,)
        )

    student_nav = st.session_state.student_nav
else:
    student_nav = "Institutional Admin Center"
    st.sidebar.markdown('<div class="sidebar-section-title">ADMIN SECURITY GATE</div>', unsafe_allow_html=True)
    
    if not st.session_state.admin_authenticated:
        configured_key = config.admin_passkey or "admin"
        admin_key_input = st.sidebar.text_input("Enter Admin Passkey", type="password")
        if st.sidebar.button("Unlock Admin Portal", type="primary", use_container_width=True):
            if hmac.compare_digest(admin_key_input.strip().encode(), configured_key.encode()):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.sidebar.error("Invalid passkey. Access denied.")
    else:
        st.sidebar.markdown(
            '<div class="side-chip green">'
            '<div class="chip-dot"></div>'
            '<span>Admin Session Authenticated</span>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.sidebar.button("Lock Admin Portal", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()

        st.sidebar.markdown('<div class="sidebar-section-title">ADMIN MANAGEMENT MODULES</div>', unsafe_allow_html=True)
        ADMIN_MODULES = [
            "Data Ingestion & Processing Center",
            "Privacy Sanitization Audit Log",
            "Institutional Corpus Telemetry"
        ]
        for amod in ADMIN_MODULES:
            is_active = (st.session_state.admin_nav == amod)
            st.sidebar.button(
                amod,
                key=f"admin_nav_btn_{amod}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                on_click=set_admin_module,
                args=(amod,)
            )

# Fast instant transition without iframe reflow latency

if portal_mode == "Student Workspace":
    # =========================================================
    # =========================================================
    # Module 1: AI Project Advisor & Ideator — INDIGO / VIOLET
    # =========================================================
    if student_nav == "AI Project Advisor & Novel Ideator":
        set_module_theme("advisor")
        render_hero_header(
            icon_name="chat_advisor",
            title="AI Project Advisor & Novel Ideator",
            subtitle="Engage with the conversational advisor or synthesize tailored capstone project proposals grounded in institutional research gaps.",
            badges=[("Conversational LLM Active", "meta-pill-blue"), (f"{db_count:,} Indexed Passages", "meta-pill-purple")],
            eyebrow="Module 01 · Conversational Intelligence"
        )

        tab_chat, tab_ideator = st.tabs(["Conversational Advisor", "Novel Project Ideator"])

        with tab_chat:
            section_header("Quick Research Inquiries", icon_name="zap")
            st.caption("Click any card below to instantly ask the advisor:")

            q1, q2, q3 = st.columns(3)
            q4, q5, q6 = st.columns(3)

            selected_prompt = None
            with q1:
                if st.button("Agriculture IoT Architectures\nHardware, sensor & battery constraints", key="chip_1"):
                    selected_prompt = "What hardware, sensor, and battery limitations were reported in past Smart Agriculture IoT projects?"
            with q2:
                if st.button("Healthcare Deep Learning\nCNN, ResNet & clinical dataset constraints", key="chip_2"):
                    selected_prompt = "Which deep learning architectures (e.g. CNNs, ResNet) and datasets were used in healthcare and medical diagnosis projects?"
            with q3:
                if st.button("Blockchain Smart Contracts\nTransaction latency, throughput & gas constraints", key="chip_3"):
                    selected_prompt = "How did past blockchain and smart contract projects address transaction latency, security, and gas fee constraints?"
            with q4:
                if st.button("NLP & Transformer Gaps\nAttention mechanisms & benchmark limits", key="chip_4"):
                    selected_prompt = "What are the most common unaddressed research gaps and future scope items across NLP projects?"
            with q5:
                if st.button("CS vs DS Methodologies\nBenchmark metrics & evaluation protocols", key="chip_5"):
                    selected_prompt = "Compare the technical methodologies and evaluation metrics between Computer Science (CS) and Data Science (DS) projects."
            with q6:
                if st.button("Novel Capstone Synthesis\nSynthesize project scope from historical gaps", key="chip_6"):
                    selected_prompt = "Based on the limitations of past student projects, synthesize a novel capstone project proposal with technical stack and evaluation metrics."

            if "rag_chat_history" not in st.session_state:
                st.session_state.rag_chat_history = [{
                    "role": "assistant",
                    "content": f"**Welcome to the Project-IQ AI Engineering Advisor.**\n\nI have indexed **{db_count:,} project passages** across CS and DS departments. Ask me any technical question or select a prompt card above!"
                }]

            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
            section_header("Active Advisor Dialogue", icon_name="chat_advisor")

            for msg in st.session_state.rag_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Ask about architectures, ML models, IoT hardware, database trade-offs...")
            active_query = selected_prompt or user_input

            if active_query:
                st.session_state.rag_chat_history.append({"role": "user", "content": active_query})
                with st.chat_message("user"):
                    st.markdown(active_query)

                with st.chat_message("assistant"):
                    with st.spinner("Synthesizing response..."):
                        result = rag_chat.ask(
                            question=active_query,
                            chat_history=[{"role": m["role"], "content": m["content"]} for m in st.session_state.rag_chat_history[:-1]],
                            n_results=6
                        )
                        st.markdown(result["answer"])
                        st.session_state.rag_chat_history.append({"role": "assistant", "content": result["answer"]})
                        st.rerun()

        with tab_ideator:
            section_header("Tailored Capstone Proposal Generator", icon_name="sparkles")
            col_ig1, col_ig2 = st.columns(2)
            with col_ig1:
                idea_domain = st.selectbox("Focus Engineering Domain", ["All Domains"] + list(DOMAIN_TAXONOMY.keys()))
                idea_interest = st.text_input("Specific Problem Focus (optional)", placeholder="e.g. Edge crop pathology, Federated privacy")
            with col_ig2:
                idea_stack = st.text_input("Target Tech Stack (optional)", placeholder="e.g. PyTorch, ONNX, FastAPI, Raspberry Pi")
                idea_count = st.slider("Proposals to Generate", 1, 5, 2)

            if st.button("Synthesize Tailored Proposals", type="primary"):
                with st.spinner("Mining gaps and generating proposals..."):
                    dom_arg = None if idea_domain == "All Domains" else idea_domain
                    filter_q = f"{dom_arg or ''} {idea_interest.strip()} {idea_stack.strip()}".strip() or dom_arg
                    st.session_state["tailored_proposals"] = gap_analyzer.generate_novel_projects(domain_filter=filter_q, num_proposals=idea_count)

            tailored_props = st.session_state.get("tailored_proposals", [])
            if tailored_props:
                st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
            for i, p in enumerate(tailored_props, 1):
                st.markdown(
                    f'<div class="piq-card fade-up" style="animation-delay:{i * 0.1:.2f}s; margin-bottom: 1rem;">'
                    f'<div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem;">'
                    f'<div style="width:44px; height:44px; border-radius:12px; background:linear-gradient(135deg, var(--a1), var(--a2)); display:flex; align-items:center; justify-content:center; font-weight:800; font-size:1.2rem; color:#FFF; flex-shrink:0;">{i}</div>'
                    f'<div class="card-title" style="margin:0; font-size:1.15rem !important;">{safe_html(p.get("title"))}</div>'
                    f'</div>'
                    f'<div class="card-body"><b style="color:#A5B4FC;">Problem:</b> {safe_html(p.get("problem_statement"))}<br><br>'
                    f'<b style="color:#C4B5FD;">Innovation:</b> {safe_html(p.get("core_innovation"))}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # =========================================================
    # Module 2: Originality Checker — AMBER / RED (Radar identity)
    # =========================================================
    elif student_nav == "Originality & Duplicate Checker":
        set_module_theme("originality")
        render_hero_header(
            icon_name="shield_check",
            title="Project Originality & Redundancy Guard",
            subtitle="Screen proposed capstones against historical repositories to detect duplicate ideas and safeguard academic novelty.",
            badges=[("Semantic Similarity Engine", "meta-pill-amber"), ("Live Institutional Archive", "meta-pill-red")],
            eyebrow="Module 02 · Plausibility Radar"
        )

        col_in1, col_in2 = st.columns([1, 1], gap="large")
        with col_in1:
            section_header("Propose Your Capstone Concept", icon_name="file_text")
            test_title = st.text_input("Proposed Project Title", "Acupoint Therapy Android Application")
            test_abstract = st.text_area("Proposed Project Summary", "An Android mobile app developed with Firebase for pain relief guiding patients through questionnaires.", height=130)
        with col_in2:
            section_header("Scan Configuration & Thresholds", icon_name="compass_gap")
            st.markdown(
                '<div class="piq-card fade-up">'
                '<div class="card-kicker">How it works</div>'
                '<div class="card-body">'
                'Your proposal is embedded and compared against every indexed historical project report. '
                'A <b style="color:#FCD34D;">semantic similarity</b> score above the duplicate threshold flags high redundancy, '
                'while a low score confirms strong academic originality.<br><br>'
                'The scanner also harvests technical themes from matched reports to generate '
                '<b style="color:#FCA5A5;">differentiation strategies</b> tailored to your risk profile.'
                '</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("")
        if st.button("Assess Originality Risk", type="primary", use_container_width=True):
            with st.spinner("Scanning institutional archive for semantic overlap..."):
                dup_res = detector.check_originality(title=test_title, abstract=test_abstract, top_k=5)

            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)

            level = dup_res.get("verdict_level", "info")
            pill_text, pill_class = verdict_pill(level)
            accent = {"success": "#10B981", "warning": "#F59E0B", "danger": "#EF4444"}.get(level, "#6366F1")

            gauge_col, detail_col = st.columns([1, 1.35], gap="large")

            with gauge_col:
                st.plotly_chart(
                    score_gauge(dup_res["originality_score"], "Originality Score", accent),
                    use_container_width=True
                )
                st.markdown(
                    f'<div style="text-align:center; margin-top:-0.5rem;">'
                    f'<span class="meta-pill {pill_class}" style="font-size:0.95rem; padding:0.45rem 1.2rem;">{pill_text}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with detail_col:
                section_header("Executive Summary", icon_name="file_text")
                st.info(dup_res.get("executive_summary", ""))
                c1, c2, c3 = st.columns(3)
                with c1:
                    stat_card(f"{dup_res['originality_score']:.1f}%", "Originality", 0.0)
                with c2:
                    stat_card(f"{dup_res.get('max_similarity_pct', 0):.1f}%", "Max Overlap", 0.1)
                with c3:
                    stat_card(str(len(dup_res.get("matching_projects", []))), "Closest Matches", 0.2)

            recs = dup_res.get("differentiation_recommendations") or []
            if recs:
                section_header("Differentiation Playbook", icon_name="compass_gap")
                rc1, rc2 = st.columns(2)
                for i, rec in enumerate(recs):
                    title_part, _, body_part = rec.partition(": ")
                    with (rc1 if i % 2 == 0 else rc2):
                        info_card(f"Strategy {i + 1}", title_part or f"Recommendation {i + 1}", body_part or rec, delay=i * 0.08)

            matches = dup_res.get("matching_projects") or []
            if matches:
                section_header("Closest Historical Archive Matches", icon_name="database")
                for i, m in enumerate(matches):
                    sim = float(m.get("similarity_pct", 0) or 0)
                    with st.expander(
                        f"#{i + 1} · {m.get('project_name', 'Unknown')}  —  {sim:.1f}% similarity"
                    ):
                        p1, p2, p3 = st.columns(3)
                        p1.markdown(f"**Department:** {m.get('department', 'General')}")
                        p2.markdown(f"**Section:** {m.get('section', 'General')}")
                        p3.markdown(f"**Category:** {m.get('category', 'Report')}")
                        animated_bar("Semantic Similarity", sim, delay=0.05)
                        st.markdown(f"> {m.get('excerpt', '')}")

    # =========================================================
    # Module 3: Research Gap Finder — EMERALD / TEAL
    # =========================================================
    elif student_nav == "Research Gap Finder":
        set_module_theme("gaps")
        render_hero_header(
            icon_name="compass_gap",
            title="Institutional Research Gap Finder",
            subtitle="Explore unaddressed technical limitations and future scope items mined across historical project reports.",
            badges=[("Innovation Frontier Lens", "meta-pill-green"), ("Limitation Mining", "meta-pill-cyan")],
            eyebrow="Module 03 · Discovery Optics"
        )

        col_search, col_cfg1, col_cfg2 = st.columns([3, 2, 1], gap="large")
        with col_search:
            section_header("Search Domain or Topic", icon_name="search")
            gap_topic = st.text_input(
                "Type anything to find gaps",
                placeholder="e.g. IoT in agriculture, blockchain security, plant disease detection",
                help="Type a topic, technology or problem area — gaps are mined from reports most similar to your query."
            )
        with col_cfg1:
            section_header("Extraction Depth Parameters", icon_name="gauge_score")
            top_k = st.slider("Max gaps per report section", 3, 20, 10,
                              help="Higher values retrieve more limitation statements per report section.")
        with col_cfg2:
            section_header("Corpus Snapshot", icon_name="bar_chart")
            stat_card(f"{db_count:,}", "Indexed Passages", 0.0)

        if st.button("Mine Research Gaps", type="primary", use_container_width=True):
            topic = (gap_topic or "").strip()
            with st.spinner(f"Mining gaps for “{topic}”..." if topic else "Mining limitations & future-scope statements across the archive..."):
                extracted_gaps = []

                if topic:
                    # Topic-driven semantic search: pull the most relevant passages, then keep
                    # sentences that state a limitation / future-scope indicator.
                    results = store.query_similar(query_text=topic, n_results=25, min_similarity=0.12)
                    for r in results:
                        meta = r.get("metadata") or {}
                        text = r.get("text") or ""
                        sim_pct = round(float(r.get("similarity", 0) or 0) * 100, 1)
                        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                        hits = [s for s in sentences if GAP_INDICATORS.search(s)]
                        for s in hits[:2]:
                            extracted_gaps.append({
                                "section": f"{meta.get('section') or 'General'}",
                                "text": s,
                                "similarity_pct": sim_pct,
                                "source_file": meta.get("source_file", "Unknown"),
                            })
                    # Fallback: no indicator sentences → show closest related passages instead
                    if not extracted_gaps:
                        for r in results[:8]:
                            meta = r.get("metadata") or {}
                            extracted_gaps.append({
                                "section": f"{meta.get('section') or 'Related'}",
                                "text": r.get("text") or "",
                                "similarity_pct": round(float(r.get("similarity", 0) or 0) * 100, 1),
                                "source_file": meta.get("source_file", "Unknown"),
                            })
                else:
                    extracted_gaps = gap_analyzer.extract_project_gaps(top_k_per_section=top_k)

            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
            top_col1, top_col2 = st.columns([1, 2], gap="large")
            with top_col1:
                stat_card(str(len(extracted_gaps)), "Gaps Discovered", 0.0)
            with top_col2:
                if topic:
                    st.markdown(
                        f'<div class="piq-card fade-up" style="animation-delay:0.1s;">'
                        f'<div class="card-kicker">Topic search · {safe_html(topic)}</div>'
                        f'<div class="card-body">Gaps below are <b style="color:#6EE7B7;">limitation or future-scope statements</b> '
                        f'semantically closest to your typed topic, with a relevance match score per card. '
                        f'Leave the search box empty next time to sweep the whole archive instead.</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="piq-card fade-up" style="animation-delay:0.1s;">'
                        '<div class="card-kicker">Reading guide</div>'
                        '<div class="card-body">Each card below is a <b style="color:#6EE7B7;">limitation or future-scope statement</b> '
                        'extracted from historical project reports. Cluster these by theme to spot under-explored '
                        'problem spaces for your next capstone — or type a topic above for a focused search.</div></div>',
                        unsafe_allow_html=True
                    )

            gaps_to_show = extracted_gaps[:12]
            cols = st.columns(2)
            for i, g in enumerate(gaps_to_show):
                text = (g.get("text") or "").strip()
                if not text:
                    continue
                snippet = text[:280] + ("…" if len(text) > 280 else "")
                sim = g.get("similarity_pct")
                kicker = str(g.get("section") or "General")
                if sim is not None:
                    kicker = f"{kicker} · {sim:g}% match"
                with cols[i % 2]:
                    info_card(
                        kicker=kicker,
                        title=f"Gap #{i + 1}",
                        body=snippet,
                        delay=(i % 2) * 0.08 + (i // 2) * 0.12,
                    )
                if i % 2 == 1:
                    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)

            if len(extracted_gaps) > 12:
                st.caption(f"Showing 12 of {len(extracted_gaps)} extracted gaps — narrow your topic or increase depth for more.")

    # =========================================================
    # Module 4: Quality Scorer — CYAN / BLUE (Blueprint identity)
    # =========================================================
    elif student_nav == "Project Quality Scorer":
        set_module_theme("quality")
        render_hero_header(
            icon_name="gauge_score",
            title="Engineering Quality Scorer",
            subtitle="Pre-submission quality evaluation measuring architecture depth, metric rigor, and technical feasibility.",
            badges=[("Heuristic Rubric v2", "meta-pill-cyan"), ("Readiness Assessment", "meta-pill-blue")],
            eyebrow="Module 04 · Blueprint Review"
        )

        section_header("Proposal Draft Input", icon_name="file_text")
        p_title = st.text_input("Project Topic", "Edge-Optimized CNN for Real-Time Plant Pathology")
        p_prob = st.text_area("Problem Statement", "Existing crop pathology models suffer from latency issues on edge devices.", height=110)
        p_meth = st.text_area("Proposed Methodology", "Quantize ResNet model via ONNX runtime and deploy on Raspberry Pi with sub-40ms latency.", height=110)
        p_stack = st.text_input("Tech Stack", "PyTorch, ONNX, Raspberry Pi")

        if st.button("Run Quality Evaluation", type="primary", use_container_width=True):
            with st.spinner("Evaluating architecture depth, metrics, and feasibility..."):
                score_res = scorer.score_proposal(
                    title=p_title,
                    problem_statement=p_prob,
                    proposed_methodology=p_meth,
                    tech_stack=p_stack,
                )

            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)

            overall = score_res["overall_score"]
            gauge_col, grade_col = st.columns([1, 1.2], gap="large")
            with gauge_col:
                st.plotly_chart(
                    score_gauge(overall, "Overall Quality", "#22D3EE"),
                    use_container_width=True
                )
            with grade_col:
                section_header("Diagnostic Verdict", icon_name="check_circle")
                st.markdown(f"### {safe_html(score_res['grade'])}")
                tier = score_res.get("readiness_tier", "")
                tier_pill = "meta-pill-green" if overall >= 75 else ("meta-pill-amber" if overall >= 50 else "meta-pill-red")
                score_stars = round(float(score_res.get("star_rating", 0) or 0), 1)
                stars = f"Rating: {score_stars} / 5.0"
                st.markdown(
                    f'<span class="meta-pill {tier_pill}" style="font-size:0.95rem; padding:0.45rem 1.2rem;">{safe_html(tier)}</span>'
                    f'<div style="font-size:1.6rem; margin-top:0.8rem; color:#FCD34D; letter-spacing:0.15em;">{stars}</div>',
                    unsafe_allow_html=True
                )

            section_header("Evaluation Dimension Breakdown", icon_name="bar_chart")
            dim_col1, dim_col2 = st.columns(2, gap="large")
            dims = score_res.get("dimensions", {})
            with dim_col1:
                for i, (name, val) in enumerate(list(dims.items())[:3]):
                    animated_bar(name, float(val), delay=i * 0.1)
            with dim_col2:
                for i, (name, val) in enumerate(list(dims.items())[3:], start=3):
                    animated_bar(name, float(val), delay=i * 0.1)

            advice = score_res.get("actionable_advice") or []
            if advice:
                section_header("Actionable Recommendations", icon_name="check_circle")
                ac1, ac2 = st.columns(2)
                for i, tip in enumerate(advice):
                    with (ac1 if i % 2 == 0 else ac2):
                        info_card(f"Tip {i + 1}", "Refinement", tip, delay=i * 0.08)

    # =========================================================
    # Module 5: Skill Matcher — FUCHSIA / PURPLE
    # =========================================================
    elif student_nav == "Student Team Skill Matcher":
        set_module_theme("team")
        render_hero_header(
            icon_name="team_users",
            title="Student Team Expertise Profiler",
            subtitle="Evaluates student team skillsets against domain taxonomies to recommend optimal project directions.",
            badges=[("Skill Taxonomy Mapping", "meta-pill-purple"), ("Domain Fit Scoring", "meta-pill-red")],
            eyebrow="Module 05 · Team Fusion"
        )

        col_in1, col_in2 = st.columns([1, 1], gap="large")
        with col_in1:
            section_header("Team Technical Competencies", icon_name="cpu_chip")
            skills_str = st.text_area(
                "Team Skills (comma-separated)",
                "Python, PyTorch, OpenCV, Docker",
                height=120,
                help="List every technical skill the team collectively possesses."
            )
        with col_in2:
            section_header("Domain Taxonomy Alignment", icon_name="compass_gap")
            st.markdown(
                '<div class="piq-card fade-up">'
                '<div class="card-kicker">How matching works</div>'
                '<div class="card-body">Your skills are embedded and compared against '
                f'<b style="color:#D8B4FE;">{len(DOMAIN_TAXONOMY)} engineering domain profiles</b>. '
                'Each domain scores a blend of <b>keyword coverage (60%)</b> and '
                '<b>semantic fit (40%)</b>, producing a readiness signal for your team.<br><br>'
                'Top matches include <b>suggested skills to learn</b> to close the gap toward a perfect fit.'
                '</div></div>',
                unsafe_allow_html=True
            )

        if st.button("Match Team Domains", type="primary", use_container_width=True):
            skills = [s.strip() for s in skills_str.split(",") if s.strip()]
            with st.spinner("Fusing skill profiles against the domain taxonomy..."):
                team_res = matcher.evaluate_team(skills=skills, top_k=3)

            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
            section_header("Optimal Domain Alignments", icon_name="target")

            top_domains = team_res.get("top_domains", [])
            for i, d in enumerate(top_domains):
                pct = float(d.get("match_percentage", 0) or 0)
                readiness = d.get("team_readiness", "Low")
                ready_pill = {"High": "meta-pill-green", "Moderate": "meta-pill-amber"}.get(readiness, "meta-pill-red")
                st.markdown(
                    f'<div class="piq-card fade-up" style="animation-delay:{i * 0.12:.2f}s; margin-bottom: 1.1rem;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem;">'
                    f'<div class="card-title" style="font-size:1.2rem !important; margin:0;">{safe_html(d.get("domain"))}</div>'
                    f'<div style="display:flex; gap:0.5rem; align-items:center;">'
                    f'<span class="meta-pill {ready_pill}">Readiness: {safe_html(readiness)}</span>'
                    f'<span class="meta-pill meta-pill-purple" style="font-size:1rem;">{pct:g}% Match</span>'
                    f'</div></div>'
                    f'<div class="card-body" style="margin:0.5rem 0 0.9rem 0;">{safe_html(d.get("description", ""))}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                animated_bar(f"{d.get('domain')} fit", pct, delay=i * 0.12 + 0.15)

                matched = d.get("matched_skills") or []
                to_learn = d.get("suggested_to_learn") or []
                sc1, sc2 = st.columns(2, gap="large")
                with sc1:
                    st.markdown("**Verified Technical Skills**")
                    if matched:
                        st.markdown(" ".join(
                            f'<span class="meta-pill meta-pill-green" style="margin:0.15rem 0.2rem 0.15rem 0;">{safe_html(k)}</span>'
                            for k in matched
                        ), unsafe_allow_html=True)
                    else:
                        st.caption("No direct keyword matches — rely on semantic fit.")
                with sc2:
                    st.markdown("**Recommended Expansion Skills**")
                    if to_learn:
                        st.markdown(" ".join(
                            f'<span class="meta-pill meta-pill-amber" style="margin:0.15rem 0.2rem 0.15rem 0;">{safe_html(k)}</span>'
                            for k in to_learn
                        ), unsafe_allow_html=True)
                    else:
                        st.caption("Team already covers the core skill set.")
                st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)

# -------------------------------------------------------------
# Admin Portal (Institutional Admin Center)
# -------------------------------------------------------------
else:
    set_module_theme("admin")

    if not st.session_state.admin_authenticated:
        render_hero_header(
            icon_name="vault_lock",
            title="Institutional Admin Center",
            subtitle="Restricted environment for department coordinators, faculty evaluators, and repository administrators.",
            badges=[("Restricted Access", "meta-pill-red")],
            eyebrow="Administration · Security Gate"
        )
        st.markdown(
            '<div class="piq-card" style="text-align: center; max-width: 580px; margin: 2rem auto; padding: 2.5rem !important;">'
            '<div style="width: 64px; height: 64px; margin: 0 auto 1.2rem auto; border-radius: 18px; background: rgba(99,102,241,0.15); display: flex; align-items: center; justify-content: center; color: var(--a1);">'
            f'{icon_svg("vault_lock", size=36, color="var(--a1)")}'
            '</div>'
            '<h3 style="color: #FFFFFF; margin: 0 0 0.6rem 0;">Institutional Authentication Required</h3>'
            '<p style="color: #94A3B8; font-size: 1.02rem; line-height: 1.65; margin-bottom: 0;">'
            'The Administration Center houses raw document ingestion, privacy redaction logs, and full corpus telemetry. '
            'Please enter the administrative passkey in the sidebar to authenticate.'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        admin_nav = st.session_state.admin_nav

        if not ner_active:
            st.warning(
                "Warning: spaCy NER model (en_core_web_sm) is not loaded. "
                "Named Entity Recognition is degraded. Student and faculty name redaction is relying on fallback regex heuristics. "
                "Run `python -m spacy download en_core_web_sm` on the server to restore full NER capability."
            )

        # ---------------------------------------------------------
        # Admin Module 1: Data Ingestion & Processing Center
        # ---------------------------------------------------------
        if admin_nav == "Data Ingestion & Processing Center":
            render_hero_header(
                icon_name="database_sync",
                title="Data Ingestion & Multi-Format Processing Center",
                subtitle="Upload new project documents, organize into institutional hierarchy, and trigger vector pipeline indexing.",
                badges=[("Multi-Format Parser", "meta-pill-blue")],
                eyebrow="Admin Center · Ingestion & Processing"
            )

            section_header("Upload Academic Project Files", icon_name="database_sync")
            st.caption("Supported formats: .pdf (Blackbooks), .pptx (Presentations), .docx (Synopses, Research Papers).")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                target_dept = st.selectbox("Assign to Department", ["CS_1", "CS_2", "DS_1", "DS_2", "General"])
                target_group = st.text_input("Assign to Group ID", "Group 14")
            with col_d2:
                target_category = st.selectbox("Document Category", ["Black Book", "Review 4", "synopsis", "Research Paper", "Abstract"])

            uploaded_files = st.file_uploader(
                "Choose files to ingest into repository",
                type=["pdf", "pptx", "docx", "doc"],
                accept_multiple_files=True
            )

            if uploaded_files:
                clean_group = re.sub(r'[^a-zA-Z0-9_\s-]', '', target_group).strip() or "Group_14"
                upload_dest = config.raw_dir / target_dept / clean_group / target_category
                upload_dest.mkdir(parents=True, exist_ok=True)
                for f in uploaded_files:
                    clean_filename = Path(f.name).name
                    save_path = upload_dest / clean_filename
                    save_path.write_bytes(f.getvalue())
                    st.success(f"Saved: `{clean_filename}` to `{save_path}`")
                st.info("Files successfully staged. Execute `python run_pipeline.py` to index new records into ChromaDB.")

        # ---------------------------------------------------------
        # Admin Module 2: Privacy Sanitization Audit Log
        # ---------------------------------------------------------
        elif admin_nav == "Privacy Sanitization Audit Log":
            render_hero_header(
                icon_name="shield_check",
                title="Privacy Sanitization & PII Masking Audit Log",
                subtitle="Inspect privacy redaction metrics, masked data breakdowns, and zero-leakage security protocols.",
                badges=[("spaCy NER + Regex Masking", "meta-pill-green")],
                eyebrow="Admin Center · Privacy Compliance"
            )

            redactions = rep_data.get("total_redactions", {})
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                stat_card(f"{redactions.get('local_name', 0):,}", "Local Names")
            with c2:
                stat_card(f"{redactions.get('phone', 0):,}", "Phone Numbers")
            with c3:
                stat_card(f"{redactions.get('student_id', 0):,}", "PRNs / Roll IDs")
            with c4:
                stat_card(f"{redactions.get('email', 0):,}", "Personal Emails")
            with c5:
                stat_card(f"{redactions.get('credential', 0):,}", "Credentials")

            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
            section_header("Sanitization Protocols Applied", icon_name="shield_check")

            st.markdown("""
            <div class="piq-card" style="padding: 1.8rem 2rem !important;">
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 1.02rem; color: #CBD5E1; line-height: 1.75;">
                <li style="margin-bottom: 0.6rem;"><strong style="color: #FFFFFF;">Administrative Front-Matter:</strong> Certificates of completion, candidate declarations, acknowledgments, and plagiarism reports are automatically filtered prior to vectorization.</li>
                <li style="margin-bottom: 0.6rem;"><strong style="color: #FFFFFF;">PII Masking:</strong> Student phone numbers, email addresses, institutional roll/PRN numbers, and local guide/student names (discovered via context cues + spaCy Named Entity Recognition) are replaced with deterministic tags (<code>[NAME]</code>, <code>[STUDENT_ID]</code>, <code>[PHONE]</code>, <code>[EMAIL]</code>).</li>
                <li><strong style="color: #FFFFFF;">Scholarly Preservation:</strong> Academic literature citations (e.g. <code>[1]</code>, <code>[2]</code>, <code>Vaswani et al. (2017)</code>) in Literature Review and Reference sections are preserved intact for accurate RAG synthesis.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # Admin Module 3: Institutional Corpus Telemetry & Search
        # ---------------------------------------------------------
        elif admin_nav == "Institutional Corpus Telemetry":
            render_hero_header(
                icon_name="bar_chart",
                title="Institutional Corpus Telemetry & Search",
                subtitle="Corpus telemetry, passage token distributions, and document search across engineering cohorts.",
                badges=[(f"{db_count:,} Passages", "meta-pill-blue")],
                eyebrow="Admin Center · Telemetry & Search"
            )

            # 4 KPI Stat Cards
            raw_docs_count = man_data.get("total_files", 250) if man_data else 250
            sanitized_docs_count = rep_data.get("cleaned_documents", 242) if rep_data else 242
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                stat_card(f"{raw_docs_count:,}", "Raw Documents")
            with k2:
                stat_card(f"{sanitized_docs_count:,}", "Sanitized Docs")
            with k3:
                stat_card(f"{len(df_chunks):,}", "Vector Chunks")
            with k4:
                stat_card(f"{total_redactions:,}", "PII Redactions")

            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)

            if not df_chunks.empty:
                # Row 1: Chunks by Dept + Category Distribution
                col_left, col_right = st.columns(2)
                with col_left:
                    section_header("Chunks by Department", icon_name="bar_chart")
                    if "department" in df_chunks.columns:
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
                        apply_plotly_theme(fig_dept, height=340)
                        fig_dept.update_layout(showlegend=False)
                        st.plotly_chart(fig_dept, use_container_width=True)

                with col_right:
                    section_header("Document Category Distribution", icon_name="bar_chart")
                    if "category" in df_chunks.columns:
                        cat_counts = df_chunks["category"].value_counts().reset_index()
                        cat_counts.columns = ["Category", "Chunks"]
                        fig_cat = px.pie(
                            cat_counts,
                            names="Category",
                            values="Chunks",
                            hole=0.52,
                            color_discrete_sequence=["#4F46E5", "#6366F1", "#818CF8", "#A5B4FC", "#C7D2FE"]
                        )
                        fig_cat.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0D111A', width=2)))
                        apply_plotly_theme(fig_cat, height=340)
                        fig_cat.update_layout(showlegend=False)
                        st.plotly_chart(fig_cat, use_container_width=True)

                # Row 2: Token Size Compliance + Section Boundary Breakdown
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    section_header("Token Size Compliance (Hard Max <= 520)", icon_name="gauge_score")
                    if "token_count" in df_chunks.columns:
                        fig_tok = px.histogram(
                            df_chunks,
                            x="token_count",
                            nbins=35,
                            color_discrete_sequence=["#4F46E5"]
                        )
                        fig_tok.add_vline(x=520, line_dash="dash", line_color="#EF4444", annotation_text="Hard Max (520 tokens)", annotation_font_color="#EF4444")
                        apply_plotly_theme(fig_tok, height=300)
                        st.plotly_chart(fig_tok, use_container_width=True)

                with col_t2:
                    section_header("Section Boundary Breakdown", icon_name="compass_gap")
                    if "section" in df_chunks.columns:
                        sec_counts = df_chunks["section"].value_counts().head(8).reset_index()
                        sec_counts.columns = ["Section", "Chunks"]
                        fig_sec = px.bar(
                            sec_counts,
                            x="Chunks",
                            y="Section",
                            orientation="h",
                            color_discrete_sequence=["#8B5CF6"]
                        )
                        fig_sec.update_traces(marker_line_width=0)
                        apply_plotly_theme(fig_sec, height=300)
                        fig_sec.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_sec, use_container_width=True)

            # Deep Passage Archive Search
            with st.expander("Deep Passage Archive Search", expanded=False):
                s_col1, s_col2 = st.columns([3, 1])
                with s_col1:
                    a_search = st.text_input("Passage Search Query", placeholder="e.g. CNN accuracy plant disease latency")
                with s_col2:
                    a_top_k = st.slider("Top Chunks", min_value=3, max_value=12, value=5)

                if a_search:
                    search_res = store.query_similar(query_text=a_search, n_results=a_top_k)
                    if search_res:
                        for i, r in enumerate(search_res):
                            st.markdown(render_citation_card({
                                "index": i + 1,
                                "source_file": r["metadata"].get("source_file"),
                                "department": r["metadata"].get("department"),
                                "group_id": r["metadata"].get("group_id"),
                                "section": r["metadata"].get("section"),
                                "similarity_score": f"{r.get('similarity', 0.0):.2f}",
                                "excerpt": r.get("text", "")[:280]
                            }), unsafe_allow_html=True)
                    else:
                        st.info("No matching passages found for this query.")

# -------------------------------------------------------------
# Sidebar Footer (Minimal & Clean)
# -------------------------------------------------------------
st.sidebar.markdown(
    '<div style="text-align: center; padding: 2rem 0 1rem 0; font-size: 0.72rem; color: #64748B; letter-spacing: 0.1em; text-transform: uppercase;">'
    f'Project-IQ · {db_count:,} Passages Indexed'
    '</div>',
    unsafe_allow_html=True
)
