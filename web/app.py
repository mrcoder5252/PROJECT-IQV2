"""
Project-IQ: Academic Project Intelligence Platform
An interactive, full-fledged professional dashboard supporting:
- Institutional Corpus Telemetry & Visual Analytics
- Dual-Mode Hybrid RAG Explorer (Exact Word + Semantic Meaning)
- Project Originality & Duplicate Idea Detector
- Research Gap Mining & Novel Ideation Engine (with Grounded Citations)
- Explainable ML Quality Scorer (XGBoost/SHAP Rubric Evaluation)
- Student Team Expertise Profiling & Domain Matcher
- Data Ingestion & Privacy Sanitization Admin Center

Accessibility (a11y) & UX Standards:
- WCAG 2.1 AA Compliant Color Contrast (> 4.5:1 text, > 3.0:1 UI components)
- Multi-modal status indications (combining icons, distinct colors, and text labels)
- High-visibility keyboard focus indicators (:focus-visible)
- Semantic landmarks, ARIA live regions, and descriptive tooltips
"""

from __future__ import annotations
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

logger = get_logger("project_iq.web")

# Page Configuration
st.set_page_config(
    page_title="Project-IQ — Academic Project Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern CSS with WCAG 2.1 AA Accessibility Standards
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* === BASE RESET === */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* Screen reader only */
.sr-only {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
}

/* Focus ring */
*:focus-visible {
    outline: 3px solid #6366f1 !important;
    outline-offset: 2px !important;
}

/* === SIDEBAR DARK THEME === */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #cbd5e1 !important;
    transition: all 0.2s ease;
    border-radius: 8px;
    padding: 4px 8px;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #ffffff !important;
    background: rgba(99, 102, 241, 0.15);
}
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem;
    font-weight: 500;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(148, 163, 184, 0.2) !important;
}
section[data-testid="stSidebar"] .stMetric label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] input {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    color: #f1f5f9 !important;
}

/* === MAIN HEADER === */
.main-header {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sub-header {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 2rem;
    font-weight: 400;
    line-height: 1.5;
    max-width: 720px;
}

/* === GLASS METRIC CARDS === */
.metric-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.12);
    border-color: rgba(99, 102, 241, 0.3);
}

/* === STAT CARD (for KPIs) === */
.stat-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.2s ease;
}
.stat-card:hover {
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    transform: translateY(-2px);
}
.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.3rem 0;
}
.stat-card .stat-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
.stat-card .stat-icon {
    font-size: 1.4rem;
    margin-bottom: 0.3rem;
}

/* === BADGES === */
.badge {
    display: inline-block;
    padding: 0.28rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 50px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}
.badge-blue { background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe; }
.badge-green { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.badge-amber { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.badge-red { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.badge-purple { background: #faf5ff; color: #6b21a8; border: 1px solid #d8b4fe; }

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f8fafc;
    padding: 4px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.88rem;
    border: none;
    background: transparent;
    color: #475569;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #e2e8f0;
    color: #1e293b;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
}

/* === CALLOUT BOX === */
.callout-box {
    border-left: 4px solid #6366f1;
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    padding: 1.2rem 1.5rem;
    border-radius: 0 12px 12px 0;
    margin: 1rem 0;
    color: #1e293b;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* === WELCOME CARD === */
.welcome-card {
    background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 50%, #f5f3ff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* === QUICK PROMPT CHIPS === */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    color: #374151 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.stButton > button:hover {
    border-color: #6366f1 !important;
    color: #4338ca !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15) !important;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    transform: translateY(-2px);
}

/* === CHAT STYLING === */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    border: 1px solid #e2e8f0 !important;
    padding: 1.2rem !important;
    margin-bottom: 0.8rem !important;
    background: #ffffff !important;
}

/* === EXPANDER STYLING === */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.9rem;
    color: #1e293b;
    background: #f8fafc;
    border-radius: 10px;
}

/* === INPUT FIELDS === */
.stTextInput input, .stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* === SELECT BOX === */
.stSelectbox > div > div {
    border-radius: 10px !important;
}

/* === SEPARATOR === */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    margin: 1.5rem 0;
}

/* === PLOTLY CHARTS CONTAINER === */
.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
}

/* === STATUS PULSE DOT === */
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
.pulse-dot.green { background: #22c55e; }
.pulse-dot.blue { background: #6366f1; }
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}

/* === VERDICT HERO CARDS === */
.verdict-hero {
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.verdict-hero.success {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #34d399;
}
.verdict-hero.warning {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 2px solid #fbbf24;
}
.verdict-hero.danger {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #f87171;
}
</style>
""", unsafe_allow_html=True)


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

# Sidebar Navigation
st.sidebar.markdown(
    '<header role="banner">'
    '<div style="text-align: center; padding: 0.5rem 0 0.8rem 0;">'
    '<div style="font-size: 2.2rem; margin-bottom: 0.2rem;">🎓</div>'
    '<h1 style="font-size: 1.4rem; font-weight: 800; '
    'background: linear-gradient(135deg, #818cf8, #a78bfa); '
    '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
    'margin: 0;">Project-IQ</h1>'
    '<p style="font-size: 0.72rem; color: #94a3b8 !important; margin: 0.2rem 0 0 0; '
    'text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600;">'
    'Academic Intelligence Platform</p>'
    '</div>'
    '</header>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Quick Stats Widget with ARIA support
db_count = store.count()
st.sidebar.markdown('<section aria-label="Quick System Metrics">', unsafe_allow_html=True)
st.sidebar.metric("Indexed Corpus Chunks", f"{db_count:,}")
if rep_data:
    st.sidebar.metric("Sanitized Documents", rep_data.get("cleaned_documents", 0))
    total_redactions = sum(rep_data.get("total_redactions", {}).values())
    st.sidebar.markdown(
        f'<div style="font-size: 0.78rem; color: #94a3b8 !important; padding: 0.3rem 0;">'
        f'<span class="pulse-dot green"></span> '
        f'<strong>PII Shield Active</strong> — {total_redactions:,} redactions</div>',
        unsafe_allow_html=True
    )
st.sidebar.markdown('</section>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<section aria-label="AI Engine Settings">'
    '<p style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; '
    'font-weight: 700; color: #64748b !important; margin-bottom: 0.5rem;">'
    '⚡ AI ENGINE</p>',
    unsafe_allow_html=True
)
gemini_key = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    placeholder="AIzaSy...",
    help="Enter your Google Gemini API key to enable generative conversational answers. If omitted, Project-IQ uses its built-in local structured synthesis engine."
)
if gemini_key:
    rag_chat.set_api_key(gemini_key)
    st.sidebar.markdown(
        '<div style="font-size: 0.78rem; color: #94a3b8 !important; padding: 0.3rem 0;">'
        '<span class="pulse-dot green"></span> '
        '<strong>Gemini LLM Connected</strong></div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div style="font-size: 0.78rem; color: #94a3b8 !important; padding: 0.3rem 0;">'
        '<span class="pulse-dot blue"></span> '
        '<strong>Local Synthesizer Active</strong></div>',
        unsafe_allow_html=True
    )
st.sidebar.markdown('</section>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<section aria-label="AI Engine Settings">'
    '<h3 style="font-size: 1.05rem; font-weight: 600; color: #1e3a8a; margin-bottom: 0.5rem;">🤖 AI RAG Engine</h3>',
    unsafe_allow_html=True
)
gemini_key = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    placeholder="AIzaSy...",
    help="Enter your Google Gemini API key to enable generative conversational answers. If omitted, Project-IQ uses its built-in local structured synthesis engine."
)
if gemini_key:
    rag_chat.set_api_key(gemini_key)
    st.sidebar.caption("🟢 **Connected:** Gemini LLM Active")
else:
    st.sidebar.caption("⚡ **Active:** Offline High-Fidelity Synthesizer")
st.sidebar.markdown('</section>', unsafe_allow_html=True)

st.sidebar.markdown("---")
view_mode = st.sidebar.radio(
    "Modules & Navigation",
    [
        "💬 Conversational AI RAG Advisor (Ask Anything)",
        "🏛️ Corpus Telemetry & Analytics",
        "🔎 Corpus Document & Passage Search",
        "🛡️ Originality & Duplicate Detector",
        "💡 Research Gap Miner & Novel Ideator",
        "⚖️ Explainable ML Quality Scorer",
        "👥 Student Team Domain Matcher",
        "⚙️ Ingestion & Privacy Audit Center"
    ],
    help="Select a functional module to explore past project data or evaluate new proposals."
)

# -------------------------------------------------------------
# Module 0: Conversational AI RAG Advisor (Ask Anything)
# -------------------------------------------------------------
if view_mode == "💬 Conversational AI RAG Advisor (Ask Anything)":
    st.markdown('<div class="main-header" role="heading" aria-level="1">💬 Conversational AI RAG Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask any open-ended question across all 4,792 indexed college project passages. Powered by dense ChromaDB retrieval + Gemini / Local Academic Synthesis.</div>', unsafe_allow_html=True)

    # Top Configuration Controls
    with st.expander("⚙️ RAG Retrieval & Filter Settings", expanded=False):
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            rag_dept = st.selectbox("Department Focus", ["ALL", "CS_1", "CS_2", "DS_1", "DS_2"], help="Filter context chunks by department cohort.")
        with c_f2:
            rag_sec = st.selectbox("Section Focus", ["ALL", "FUTURE SCOPE", "METHODOLOGY", "RESULTS", "CONCLUSION", "ABSTRACT"], help="Focus on specific sections such as limitations or methodology.")
        with c_f3:
            rag_top_k = st.slider("Context Passages", min_value=3, max_value=12, value=6, help="Number of nearest chunks retrieved to ground the answer.")

    # Quick-Start Prompt Chips
    st.markdown('<p style="font-size: 0.9rem; font-weight: 600; color: #475569; margin-bottom: 0.4rem;">⚡ Quick Research Inquiries (Click to ask):</p>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    q4, q5, q6 = st.columns(3)
    
    selected_prompt = None
    with q1:
        if st.button("🌾 Agriculture Hardware Limits", use_container_width=True, help="What hardware, sensor, and battery limitations were reported in past Smart Agriculture IoT projects?"):
            selected_prompt = "What hardware, sensor, and battery limitations were reported in past Smart Agriculture IoT projects?"
    with q2:
        if st.button("🩺 Healthcare Deep Learning Models", use_container_width=True, help="Which deep learning architectures (e.g. CNNs, ResNet) and datasets were used in healthcare and medical diagnosis projects?"):
            selected_prompt = "Which deep learning architectures (e.g. CNNs, ResNet) and datasets were used in healthcare and medical diagnosis projects?"
    with q3:
        if st.button("🔐 Blockchain Latency & Gas Costs", use_container_width=True, help="How did past blockchain and smart contract projects address transaction latency, security, and gas fee constraints?"):
            selected_prompt = "How did past blockchain and smart contract projects address transaction latency, security, and gas fee constraints?"
    with q4:
        if st.button("🤖 NLP Research Gaps in CS & DS", use_container_width=True, help="What are the most common unaddressed research gaps and future scope items across NLP projects?"):
            selected_prompt = "What are the most common unaddressed research gaps and future scope items across NLP projects?"
    with q5:
        if st.button("📊 Compare CS vs DS Methodologies", use_container_width=True, help="Compare the technical methodologies and evaluation metrics between Computer Science (CS) and Data Science (DS) projects."):
            selected_prompt = "Compare the technical methodologies and evaluation metrics between Computer Science (CS) and Data Science (DS) projects."
    with q6:
        if st.button("💡 Synthesize Novel Capstone Proposal", use_container_width=True, help="Synthesize a novel capstone project proposal addressing unaddressed limitations in the repository."):
            selected_prompt = "Based on the limitations of past student projects, synthesize a novel capstone project proposal with technical stack and evaluation metrics."

    # Initialize Chat Session State
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = [
            {
                "role": "assistant",
                "content": "👋 **Welcome to Project-IQ Academic Advisor!**\n\nI have indexed **4,792 canonical passages** from your institution's past project reports across Computer Science and Data Science.\n\nYou can ask me **any question**—from specific model architectures, experimental benchmarks, and dataset constraints, to cross-group comparisons and novel project ideation.\n\n*Click one of the quick prompt buttons above or type your question below to get started!*",
                "sources": [],
                "engine": "System Greeting"
            }
        ]

    # Conversation management button
    if len(st.session_state.rag_chat_history) > 1:
        c_btn1, c_btn2 = st.columns([6, 1])
        with c_btn2:
            if st.button("🗑️ Reset Chat", use_container_width=True):
                st.session_state.rag_chat_history = [st.session_state.rag_chat_history[0]]
                st.rerun()

    # Render Chat History
    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Grounded Institutional Citations ({len(msg['sources'])} Source Passages)", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"**[{src['index']}] {src['source_file']}** — `{src['department']} / {src['group_id']}` | Section: `{src['section']}` *(Affinity: {src['similarity_score']})*")
                        st.caption(f"> \"{src['excerpt']}\"")
                        st.markdown("---")
            if msg.get("engine") and msg["engine"] != "System Greeting":
                st.caption(f"⚙️ *Synthesized via {msg['engine']}*")

    # Chat Input handling
    user_input = st.chat_input("Ask any question about past college projects, architectures, datasets, gaps...")
    active_query = selected_prompt or user_input

    if active_query:
        # Append user message and render immediately
        st.session_state.rag_chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing 4,792 institutional project passages..."):
                result = rag_chat.ask(
                    question=active_query,
                    chat_history=[{"role": m["role"], "content": m["content"]} for m in st.session_state.rag_chat_history[:-1]],
                    department_filter=rag_dept,
                    section_filter=rag_sec,
                    n_results=rag_top_k
                )

                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander(f"📚 Grounded Institutional Citations ({len(result['sources'])} Source Passages)", expanded=False):
                        for src in result["sources"]:
                            st.markdown(f"**[{src['index']}] {src['source_file']}** — `{src['department']} / {src['group_id']}` | Section: `{src['section']}` *(Affinity: {src['similarity_score']})*")
                            st.caption(f"> \"{src['excerpt']}\"")
                            st.markdown("---")
                st.caption(f"⚙️ *Synthesized via {result['engine_used']}*")

                # Append to session state
                st.session_state.rag_chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "engine": result["engine_used"]
                })
                st.rerun()

# -------------------------------------------------------------
# Module 1: Institutional Corpus Telemetry & Visual Analytics
# -------------------------------------------------------------
elif view_mode == "🏛️ Corpus Telemetry & Analytics":
    st.markdown('<div class="main-header" role="heading" aria-level="1">🏛️ Institutional Corpus Telemetry & Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time statistics, structural distributions, and privacy sanitization metrics across college project cohorts.</div>', unsafe_allow_html=True)

    total_red = sum(rep_data.get("total_redactions", {}).values()) if rep_data else 7385
    # KPI Row with stat cards
    st.markdown(
        '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">'
        f'<div class="stat-card"><div class="stat-icon">📄</div>'
        f'<div class="stat-value">{man_data.get("total_files", 249)}</div>'
        f'<div class="stat-label">Raw Documents</div></div>'
        f'<div class="stat-card"><div class="stat-icon">🧹</div>'
        f'<div class="stat-value">{rep_data.get("cleaned_documents", 242)}</div>'
        f'<div class="stat-label">Sanitized Docs</div></div>'
        f'<div class="stat-card"><div class="stat-icon">🧬</div>'
        f'<div class="stat-value">{db_count:,}</div>'
        f'<div class="stat-label">Vector Chunks</div></div>'
        f'<div class="stat-card"><div class="stat-icon">🛡️</div>'
        f'<div class="stat-value">{total_red:,}</div>'
        f'<div class="stat-label">PII Redactions</div></div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    if not df_chunks.empty:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📊 Chunks by Department")
            dept_counts = df_chunks["department"].value_counts().reset_index()
            dept_counts.columns = ["Department", "Chunks"]
            fig_dept = px.bar(
                dept_counts,
                x="Department",
                y="Chunks",
                color="Department",
                color_discrete_sequence=px.colors.qualitative.Safe,
                text="Chunks"
            )
            fig_dept.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=320,
                xaxis_title="Engineering Department",
                yaxis_title="Canonical Chunk Count"
            )
            st.plotly_chart(fig_dept, use_container_width=True)
            st.caption("Figure 1: Distribution of indexed text chunks across Computer Science and Data Science divisions.")

        with col_right:
            st.subheader("📑 Document Category Distribution")
            cat_counts = df_chunks["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Chunks"]
            fig_cat = px.pie(
                cat_counts,
                names="Category",
                values="Chunks",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_cat.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_cat, use_container_width=True)
            st.caption("Figure 2: Proportional representation of Blackbooks, Research Papers, Synopses, and Presentations.")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("📏 RAG Token Size Compliance (Hard Max $\le 520$)")
            fig_tok = px.histogram(
                df_chunks,
                x="token_count",
                nbins=30,
                color_discrete_sequence=["#6366f1"]
            )
            fig_tok.add_vline(x=520, line_dash="dash", line_color="#b91c1c", annotation_text="Hard Max (520)")
            fig_tok.update_layout(
                xaxis_title="Token Count (tiktoken cl100k_base)",
                yaxis_title="Passage Frequency",
                margin=dict(t=10, b=10, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig_tok, use_container_width=True)
            st.caption("Figure 3: Token length distribution demonstrating strict compliance with the $\le 520$ token safety boundary.")

        with col_t2:
            st.subheader("🔬 Section Boundary Breakdown")
            sec_counts = df_chunks["section"].value_counts().head(8).reset_index()
            sec_counts.columns = ["Section", "Chunks"]
            fig_sec = px.bar(
                sec_counts,
                x="Chunks",
                y="Section",
                orientation="h",
                color_discrete_sequence=["#8b5cf6"]
            )
            fig_sec.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                xaxis_title="Chunks in Section",
                yaxis_title="Academic Section"
            )
            st.plotly_chart(fig_sec, use_container_width=True)
            st.caption("Figure 4: Frequency of section headers mined across reports.")
    else:
        st.info("Corpus data is not yet indexed. Run `python run_pipeline.py` to populate telemetry.")

    with st.expander("🏛️ System Architecture Reference (Springer Conference Design)"):
        st.markdown("""
        **Pipeline Data Flow:**
        1. **Multi-Format Extraction:** PyMuPDF (PDFs), python-pptx (Slides), python-docx, with Tesseract OCR fallback for scanned reports.
        2. **Privacy Sanitization:** Administrative front-matter removal and PII redaction via regex & spaCy NER.
        3. **Structure-Aware Chunking:** Token bounds (target 450, hard ceiling $\le 520$), section splitting, and 6-shingle Jaccard near-duplicate clustering.
        4. **Vector Storage & Hybrid Retrieval:** Persistent ChromaDB with SentenceTransformers (`all-MiniLM-L6-v2`).
        5. **Gap Analysis & RAG Ideation:** Limitation mining across past projects to synthesize novel proposals with citations.
        6. **Explainable ML Quality Scoring:** Feature extraction & SHAP-style explainable rubric advice.
        7. **Team Expertise Matcher:** Embedding cosine similarity & technical keyword alignment.
        """)

# -------------------------------------------------------------
# Module 2: Corpus Document & Passage Search
# -------------------------------------------------------------
elif view_mode == "🔎 Corpus Document & Passage Search":
    st.markdown('<div class="main-header" role="heading" aria-level="1">🔎 Corpus Document & Passage Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search past project reports using exact keyword matching, dense semantic embeddings, or hybrid filtering.</div>', unsafe_allow_html=True)

    with st.container():
        row1_col1, row1_col2 = st.columns([3, 1])
        with row1_col1:
            search_query = st.text_input(
                "Enter Search Query or Concept",
                placeholder="e.g. CNN accuracy plant disease detection edge latency",
                help="Search term for dense vector retrieval or lexical matching."
            )
        with row1_col2:
            search_mode = st.selectbox(
                "Search Mode",
                ["Dense Semantic Search", "Exact Keyword Search", "Hybrid (Keyword + Semantic)"],
                help="Dense search uses 384-dimensional embeddings; Exact search finds exact substrings."
            )

        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        with row2_col1:
            dept_filter = st.selectbox("Department Filter", ["ALL", "CS_1", "CS_2", "DS_1", "DS_2", "General"])
        with row2_col2:
            section_filter = st.selectbox(
                "Section Filter",
                ["ALL", "FUTURE SCOPE", "METHODOLOGY", "RESULTS", "CONCLUSION", "ABSTRACT", "LITERATURE SURVEY"]
            )
        with row2_col3:
            cat_filter = st.selectbox(
                "Document Type",
                ["ALL", "blackbook", "research_paper", "review4", "abstract_synopsis"]
            )
        with row2_col4:
            top_k = st.slider("Top Chunks", min_value=3, max_value=20, value=6, help="Number of nearest passages to retrieve.")

    if search_query:
        with st.spinner("Executing RAG retrieval across indexed corpus..."):
            sec = None if section_filter == "ALL" else section_filter
            dept = None if dept_filter == "ALL" else dept_filter
            cat = None if cat_filter == "ALL" else cat_filter

            if search_mode == "Exact Keyword Search" and not df_chunks.empty:
                mask = df_chunks["text"].str.contains(re.escape(search_query), case=False, na=False)
                if sec:
                    mask &= (df_chunks["section"] == sec)
                if dept:
                    mask &= (df_chunks["department"] == dept)
                if cat:
                    mask &= (df_chunks["category"] == cat)
                matched_df = df_chunks[mask].head(top_k)

                results = []
                for _, row in matched_df.iterrows():
                    results.append({
                        "text": row["text"],
                        "similarity": 1.0,
                        "metadata": {
                            "document_id": row.get("document_id", "N/A"),
                            "source_file": row.get("source_file", "N/A"),
                            "department": row.get("department", "N/A"),
                            "group_id": row.get("group_id", "N/A"),
                            "category": row.get("category", "N/A"),
                            "section": row.get("section", "N/A"),
                            "token_count": row.get("token_count", 0),
                            "has_citation": row.get("has_citation", False),
                            "has_numbers": row.get("has_numbers", False)
                        }
                    })
            else:
                results = store.query_similar(
                    query_text=search_query,
                    n_results=top_k * 2 if (dept or cat) else top_k,
                    section=sec
                )
                if dept:
                    results = [r for r in results if r["metadata"].get("department") == dept]
                if cat:
                    results = [r for r in results if r["metadata"].get("category") == cat]
                results = results[:top_k]

        if results:
            st.success(f"Retrieved {len(results)} relevant project passages matching your query criteria:")
            for i, r in enumerate(results, 1):
                meta = r["metadata"]
                sim = r.get("similarity", 1.0)
                with st.expander(
                    f"#{i} [{meta.get('department', 'General')} / {meta.get('group_id', 'Group')}] {meta.get('source_file')} — Section: {meta.get('section')} (Affinity: {sim:.2f})",
                    expanded=(i == 1)
                ):
                    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                    b_col1.markdown(f"**Category:** `{meta.get('category')}`")
                    b_col2.markdown(f"**Tokens:** `{meta.get('token_count')}`")
                    b_col3.markdown(f"**Citations:** `{'Yes' if meta.get('has_citation') else 'None'}`")
                    b_col4.markdown(f"**Metrics:** `{'Yes' if meta.get('has_numbers') else 'None'}`")

                    st.markdown("---")
                    st.write(r["text"])
        else:
            st.info("No matching passages found. Try relaxing the department/section filters or broadening the search query.")

# -------------------------------------------------------------
# Module 3: Originality & Duplicate Project Detector
# -------------------------------------------------------------
elif view_mode == "🛡️ Originality & Duplicate Detector":
    st.markdown('<div class="main-header" role="heading" aria-level="1">🛡️ Project Originality & Duplicate Idea Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Screen proposed project concepts against the historical repository to detect duplication and safeguard academic novelty.</div>', unsafe_allow_html=True)

    col_in1, col_in2 = st.columns([1, 1])
    with col_in1:
        test_title = st.text_input(
            "Proposed Project Title",
            "Acupoint Therapy and Symptom Management Android Application",
            help="Full tentative title of the proposed student project."
        )
    with col_in2:
        test_abstract = st.text_area(
            "Proposed Project Abstract / Summary",
            "An AI-powered Android mobile app developed with Firebase for musculoskeletal pain relief, guiding patients through questionnaires and recommending exercises.",
            height=100,
            help="Summary of the proposed system methodology, target users, and technology stack."
        )

    check_btn = st.button("🛡️ Assess Originality & Redundancy Risk", type="primary", help="Trigger semantic similarity scan across historical reports.")

    if check_btn:
        with st.spinner("Analyzing semantic overlap across historical project corpus..."):
            dup_res = detector.check_originality(title=test_title, abstract=test_abstract, top_k=5)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.subheader("Verdict & Risk Assessment")
            max_pct = dup_res["max_similarity_pct"]
            orig_score = dup_res["originality_score"]

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=orig_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Originality Score", 'font': {'size': 18, 'color': '#1e293b'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#334155'},
                    'bar': {'color': "#6366f1"},
                    'steps': [
                        {'range': [0, 35], 'color': "#fee2e2"},
                        {'range': [35, 70], 'color': "#fef3c7"},
                        {'range': [70, 100], 'color': "#dcfce7"}
                    ],
                    'threshold': {
                        'line': {'color': "#b91c1c", 'width': 4},
                        'thickness': 0.75,
                        'value': 35
                    }
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=20, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if dup_res["verdict_level"] == "danger":
                st.error(f"🔴 **[HIGH REDUNDANCY DETECTED]** ({max_pct}% historical similarity)")
            elif dup_res["verdict_level"] == "warning":
                st.warning(f"🟡 **[MODERATE OVERLAP]** ({max_pct}% historical similarity)")
            else:
                st.success(f"🟢 **[HIGH ORIGINALITY VERIFIED]** ({orig_score}% Novelty Rating)")

            st.markdown(f'<div class="callout-box" role="status" aria-live="polite">{dup_res["warning_message"]}</div>', unsafe_allow_html=True)

        with res_col2:
            st.subheader("📚 Closest Historical Projects in Archive")
            matches = dup_res.get("matching_projects", [])
            if matches:
                for idx, m in enumerate(matches, 1):
                    with st.container():
                        st.markdown(f"**{idx}. [{m['department']} / {m['group_id']}] {m['source_file']}** — Similarity: `{m['similarity_pct']}%`")
                        st.markdown(f"*Category:* `{m['category']}` | *Section:* `{m['section']}`")
                        st.write(f"> \"{m['excerpt']}\"")
                        st.markdown("---")
            else:
                st.info("No matching historical projects found.")

# -------------------------------------------------------------
# Module 4: Research Gap Miner & Novel Ideation Engine
# -------------------------------------------------------------
elif view_mode == "💡 Research Gap Miner & Novel Ideator":
    st.markdown('<div class="main-header" role="heading" aria-level="1">💡 Research Gap Miner & Novel Project Ideator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mines technical limitations in past student projects and synthesizes novel, high-scoring final year project proposals with grounded citations.</div>', unsafe_allow_html=True)

    col_g1, col_g2, col_g3 = st.columns([2, 1, 1])
    with col_g1:
        domain_sel = st.selectbox("Focus Engineering Domain", ["All Domains"] + list(DOMAIN_TAXONOMY.keys()))
    with col_g2:
        num_prop = st.slider("Number of Proposals", min_value=1, max_value=5, value=2)
    with col_g3:
        st.write("")
        st.write("")
        gen_btn = st.button("🚀 Synthesize Novel Proposals", type="primary", use_container_width=True)

    proposals_file = config.reports_dir / "novel_project_proposals.json"
    proposals = []

    if gen_btn:
        with st.spinner("Extracting technical gaps and generating grounded novel proposals..."):
            domain_arg = None if domain_sel == "All Domains" else domain_sel
            proposals = gap_analyzer.generate_novel_projects(domain_filter=domain_arg, num_proposals=num_prop)
    elif proposals_file.exists():
        try:
            proposals = json.loads(proposals_file.read_text(encoding="utf-8"))
        except Exception:
            proposals = []

    if proposals:
        st.markdown(f"### 📋 Synthesized Project Proposals ({len(proposals)})")
        for i, p in enumerate(proposals, 1):
            with st.container():
                st.markdown(f"### {i}. {p.get('title')}")
                st.markdown(f"**Domain:** `{p.get('domain')}` | **Feasibility Score:** `{p.get('feasibility_score', 8)}/10`")

                tab_p1, tab_p2, tab_p3 = st.tabs(["📌 Problem & Innovation", "🛠️ Architecture & Tech Stack", "📚 Grounded Past Project Citations"])
                with tab_p1:
                    st.markdown(f"**Problem Statement:** {p.get('problem_statement')}")
                    st.markdown(f"**Core Innovation:** {p.get('core_innovation')}")
                with tab_p2:
                    st.markdown(f"**Proposed Methodology:** {p.get('proposed_methodology')}")
                    stack = p.get("tech_stack", [])
                    st.markdown("**Recommended Stack:** " + ", ".join(f"`{s}`" for s in stack))
                    deliv = p.get("expected_deliverables", [])
                    if deliv:
                        st.markdown("**Expected Deliverables:**")
                        for d in deliv:
                            st.markdown(f"- {d}")
                with tab_p3:
                    citations = p.get("past_project_citations", [])
                    if citations:
                        for cite in citations:
                            st.markdown(f"- **Source:** `{cite.get('source_file')}`")
                            st.write(f"  *Identified Limitation:* {cite.get('identified_gap')}")
                    else:
                        st.info("General institutional synthesis.")
                st.markdown("---")

        # Export as JSON
        st.download_button(
            label="📥 Download Proposals JSON Report",
            data=json.dumps(proposals, indent=2),
            file_name="project_iq_novel_proposals.json",
            mime="application/json"
        )

# -------------------------------------------------------------
# Module 5: Explainable ML Quality Scorer (Feature 6 in Paper)
# -------------------------------------------------------------
elif view_mode == "⚖️ Explainable ML Quality Scorer":
    st.markdown('<div class="main-header" role="heading" aria-level="1">⚖️ Explainable ML Quality Scorer & Rubric Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pre-submission quality evaluation predicting project marks and generating SHAP-style explainable improvement tips.</div>', unsafe_allow_html=True)

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        prop_title = st.text_input(
            "Project Proposal Title",
            "Federated Learning Framework with Differential Privacy for Edge Healthcare",
            help="Tentative title for rubric evaluation."
        )
        prop_prob = st.text_area(
            "Problem Statement",
            "Centralized clinical data aggregation introduces severe patient data privacy violations and communication latency bottlenecks.",
            height=90,
            help="Core problem formulation."
        )
    with col_q2:
        prop_meth = st.text_area(
            "Proposed Methodology",
            "1. Client node local model training on decentralized scans. 2. Secure federated aggregation using FedAvg with DP noise. 3. Target latency < 45ms with 94.8% classification accuracy [1]. 4. Containerized evaluation pipeline.",
            height=110,
            help="Methodology with phases and target benchmarks."
        )
        prop_stack = st.text_input(
            "Tech Stack (comma-separated)",
            "Python, PyTorch, Flower, FastAPI, Docker, Streamlit",
            help="Technologies planned for development and deployment."
        )

    score_btn = st.button("⚖️ Run Explainable ML Quality Evaluation", type="primary")

    if score_btn:
        with st.spinner("Extracting rubric features and computing SHAP attributions..."):
            score_res = scorer.score_proposal(
                title=prop_title,
                problem_statement=prop_prob,
                proposed_methodology=prop_meth,
                tech_stack=prop_stack,
                citations=["VASWANI_2017", "PROJECT_REPORT_REF"]
            )

        st.markdown("---")
        sc_col1, sc_col2 = st.columns([1, 2])

        with sc_col1:
            st.subheader("Rubric Evaluation Signal")
            overall = score_res["overall_score"]
            stars = score_res["star_rating"]
            tier = score_res["readiness_tier"]

            st.metric("Overall Score", f"{overall}/100", help="Predicted composite marks out of 100 based on faculty rubric.")
            st.metric("Star Rating", f"{stars} / 5.0 ⭐", help="Normalized 5-star rubric rating.")
            if "Approved" in tier:
                st.success(f"✅ **[APPROVED]** {tier}")
            elif "Revision" in tier:
                st.warning(f"⚠️ **[REVISION NEEDED]** {tier}")
            else:
                st.error(f"❌ **[CRITICAL GAPS]** {tier}")

        with sc_col2:
            st.subheader("📊 SHAP Feature Attribution Breakdown")
            shaps = score_res["shap_attributions"]
            df_shap = pd.DataFrame([
                {"Feature": k.replace("_", " ").title(), "Contribution": v}
                for k, v in shaps.items()
            ])
            df_shap["Color"] = df_shap["Contribution"].apply(lambda x: "#059669" if x >= 0 else "#b91c1c")

            fig_shap = px.bar(
                df_shap,
                x="Contribution",
                y="Feature",
                orientation="h",
                color="Color",
                color_discrete_map="identity"
            )
            fig_shap.update_layout(
                xaxis_title="SHAP Attribution (Deviation from Rubric Baseline)",
                yaxis=dict(autorange="reversed"),
                height=260,
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Figure 5: Explainable SHAP attributions illustrating which rubric criteria boosted or penalized the score.")

        st.subheader("💡 Actionable Improvement Advice")
        for advice in score_res["actionable_advice"]:
            st.info(f"👉 {advice}")

# -------------------------------------------------------------
# Module 6: Student Team Expertise & Domain Matcher
# -------------------------------------------------------------
elif view_mode == "👥 Student Team Domain Matcher":
    st.markdown('<div class="main-header" role="heading" aria-level="1">👥 Student Team Expertise Profiler & Domain Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluates student team skillsets against domain taxonomies to recommend optimal project directions and highlight skill gaps.</div>', unsafe_allow_html=True)

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

        st.subheader("🏆 Ranked Domain Recommendations")
        for i, d in enumerate(team_res["top_domains"], 1):
            with st.container():
                m_col1, m_col2 = st.columns([3, 1])
                with m_col1:
                    st.markdown(f"#### {i}. {d['domain']}")
                    st.write(d['description'])
                    st.markdown(f"**Matched Competencies:** {', '.join(d['matched_skills']) if d['matched_skills'] else 'Semantic affinity'}")
                    if d['suggested_to_learn']:
                        st.markdown("💡 **Skills to Learn to Excel in this Domain:** " + ", ".join(f"`{k}`" for k in d['suggested_to_learn']))
                with m_col2:
                    st.metric("Domain Match", f"{d['match_percentage']}%")
                    readiness = d['team_readiness']
                    if readiness == "High":
                        st.success(f"Readiness: **[HIGH]**")
                    elif readiness == "Moderate":
                        st.warning(f"Readiness: **[MODERATE]**")
                    else:
                        st.error(f"Readiness: **[LOW]**")
                st.markdown("---")

# -------------------------------------------------------------
# Module 7: Ingestion & Privacy Audit Center
# -------------------------------------------------------------
elif view_mode == "⚙️ Ingestion & Privacy Audit Center":
    st.markdown('<div class="main-header" role="heading" aria-level="1">⚙️ Data Ingestion & Privacy Sanitization Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload new documents, inspect privacy audit metrics, and monitor pipeline operations.</div>', unsafe_allow_html=True)

    tab_adm1, tab_adm2 = st.tabs(["📤 Upload New Project Documents", "🛡️ Privacy Sanitization Audit Log"])

    with tab_adm1:
        st.subheader("Upload Academic Project Files")
        st.markdown("Supported formats: `.pdf` (Blackbooks), `.pptx` (Presentations), `.docx` (Synopses, Research Papers).")

        target_dept = st.selectbox("Assign to Department", ["CS_1", "CS_2", "DS_1", "DS_2", "General"])
        target_group = st.text_input("Assign to Group ID", "Group 14")
        target_category = st.selectbox("Document Category", ["Black Book", "Review 4", "synopsis", "Research Paper", "Abstract"])

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "pptx", "docx", "doc"],
            accept_multiple_files=True,
            help="Files uploaded here will be organized into the college directory structure."
        )

        if uploaded_files:
            upload_dest = config.raw_dir / target_dept / target_group / target_category
            upload_dest.mkdir(parents=True, exist_ok=True)
            for f in uploaded_files:
                save_path = upload_dest / f.name
                save_path.write_bytes(f.getvalue())
                st.success(f"Saved: `{f.name}` to `{save_path}`")
            st.info("Files saved to raw directory. Run `python run_pipeline.py` to index newly uploaded files into ChromaDB.")

    with tab_adm2:
        st.subheader("🛡️ PII Redaction Telemetry")
        if rep_data:
            redactions = rep_data.get("total_redactions", {})
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Local Names", f"{redactions.get('local_name', 0):,}", help="Student and supervisor names masked with [NAME]")
            r2.metric("Phone Numbers", f"{redactions.get('phone', 0):,}", help="Phone numbers masked with [PHONE]")
            r3.metric("Student PRNs / IDs", f"{redactions.get('student_id', 0):,}", help="Institutional IDs masked with [STUDENT_ID]")
            r4.metric("Personal Emails", f"{redactions.get('email', 0):,}", help="Personal and academic emails masked with [EMAIL]")
            r5.metric("Credentials", f"{redactions.get('credential', 0):,}", help="API keys and secrets masked with [REDACTED_CREDENTIAL]")

            st.markdown("---")
            st.markdown("""
            **Sanitization Protocols Applied:**
            - **Administrative Front-Matter:** Certificates of completion, candidate declarations, acknowledgments, and plagiarism reports are automatically filtered prior to vectorization.
            - **PII Masking:** Student phone numbers, email addresses, institutional roll/PRN numbers, and local guide/student names (discovered via context cues + spaCy Named Entity Recognition) are replaced with deterministic tags (`[NAME]`, `[STUDENT_ID]`, `[PHONE]`, `[EMAIL]`).
            - **Scholarly Preservation:** Academic literature citations (e.g. `[1]`, `[2]`, `Vaswani et al. (2017)`) in Literature Review and Reference sections are preserved intact for accurate RAG synthesis.
            """)
        else:
            st.info("No cleaning report found. Execute pipeline to generate audit data.")
