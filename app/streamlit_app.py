"""
AI Music Composition System — Main Streamlit Application
=========================================================
Premium dark-themed multi-page web app for real-time AI music generation.

Run
---
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Add project root to path ───────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# ── Page Config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="AI Music Composer | Transformer Networks",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks",
        "Report a Bug": "https://github.com/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks/issues",
        "About": "AI Music Composition System using Transformer Networks — by Rohit Birdawade",
    },
)

# ── Premium Dark Theme CSS ─────────────────────────────────────────────────
PREMIUM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card:      #1a1a2e;
    --bg-glass:     rgba(26, 26, 46, 0.6);
    --accent-purple:#8b5cf6;
    --accent-blue:  #3b82f6;
    --accent-teal:  #14b8a6;
    --accent-pink:  #ec4899;
    --text-primary: #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:   #475569;
    --border:       rgba(139, 92, 246, 0.2);
    --glow-purple:  0 0 20px rgba(139, 92, 246, 0.3);
    --glow-blue:    0 0 20px rgba(59, 130, 246, 0.3);
    --gradient-main: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 50%, #14b8a6 100%);
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Hide default Streamlit elements ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--accent-purple); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Sidebar nav items ── */
.stRadio > div { gap: 4px !important; }
.stRadio label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
}
.stRadio label:hover {
    border-color: var(--accent-purple) !important;
    background: rgba(139, 92, 246, 0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--gradient-main) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
    box-shadow: var(--glow-purple) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Sliders ── */
.stSlider > div > div > div {
    background: var(--gradient-main) !important;
}
.stSlider > div > div > div > div {
    background: white !important;
    border: 2px solid var(--accent-purple) !important;
    box-shadow: var(--glow-purple) !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Metric boxes ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: var(--glow-purple) !important;
}
[data-testid="metric-container"] label { color: var(--text-secondary) !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-purple) !important;
    font-weight: 700 !important;
}

/* ── Info / Success / Warning boxes ── */
.stInfo    { background: rgba(59, 130, 246, 0.1) !important; border-color: var(--accent-blue) !important; border-radius: 12px !important; }
.stSuccess { background: rgba(20, 184, 166, 0.1) !important; border-color: var(--accent-teal) !important; border-radius: 12px !important; }
.stWarning { background: rgba(245, 158, 11, 0.1) !important; border-color: #f59e0b !important; border-radius: 12px !important; }
.stError   { background: rgba(239, 68, 68, 0.1) !important; border-color: #ef4444 !important; border-radius: 12px !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: var(--gradient-main) !important;
    border-radius: 10px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--gradient-main) !important;
    color: white !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 24px 0 !important;
}

/* ── Custom Card ── */
.music-card {
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px;
    margin: 12px 0;
    box-shadow: var(--glow-purple);
    transition: all 0.3s ease;
}
.music-card:hover {
    border-color: var(--accent-purple);
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.2);
}

/* ── Hero Section ── */
.hero-title {
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.15rem;
    font-weight: 400;
    line-height: 1.6;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: var(--accent-purple);
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 2px;
}

/* ── Animated gradient border ── */
@keyframes borderGlow {
    0%   { box-shadow: 0 0 5px rgba(139, 92, 246, 0.3); }
    50%  { box-shadow: 0 0 20px rgba(59, 130, 246, 0.5), 0 0 40px rgba(139, 92, 246, 0.2); }
    100% { box-shadow: 0 0 5px rgba(139, 92, 246, 0.3); }
}
.glow-border { animation: borderGlow 3s ease-in-out infinite; }

/* ── Piano keys visual ── */
.piano-container {
    display: flex;
    gap: 2px;
    justify-content: center;
    padding: 12px;
    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border);
}
.piano-key {
    width: 28px;
    height: 90px;
    background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 0 0 6px 6px;
    border: 1px solid #cbd5e1;
    transition: all 0.15s ease;
    cursor: pointer;
}
.piano-key.black {
    width: 18px;
    height: 58px;
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    margin: 0 -9px;
    z-index: 1;
    border-radius: 0 0 4px 4px;
}
.piano-key.active {
    background: var(--gradient-main) !important;
    box-shadow: var(--glow-purple);
}

/* ── Note visualization ── */
.note-pill {
    display: inline-block;
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 100px;
    padding: 4px 10px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-purple);
    margin: 2px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-purple) !important;
    background: rgba(139, 92, 246, 0.05) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--accent-teal) !important;
    color: var(--accent-teal) !important;
    box-shadow: 0 0 20px rgba(20, 184, 166, 0.2) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(20, 184, 166, 0.1) !important;
    transform: translateY(-2px) !important;
}
</style>
"""


def main():
    # Inject global CSS
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # ── Sidebar Navigation ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 16px 0 24px;'>
            <div style='font-size: 2.5rem; margin-bottom: 8px;'>🎵</div>
            <div style='font-size: 1.1rem; font-weight: 700; 
                        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                AI Music Composer
            </div>
            <div style='font-size: 0.75rem; color: #64748b; margin-top: 4px;'>
                Transformer Networks v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Navigation**")
        page = st.radio(
            "Go to",
            ["🎹 Generate Music", "📤 Upload Style (RAG)", "📊 Visualize", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Quick status
        st.markdown("**System Status**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mode", "Demo", help="Running in demo mode — no GPU needed")
        with col2:
            st.metric("API", "✅", help="Flask API available at localhost:5000")

        st.markdown("""
        <div style='margin-top: 16px; padding: 12px; background: rgba(139,92,246,0.1); 
                    border: 1px solid rgba(139,92,246,0.3); border-radius: 10px;
                    font-size: 12px; color: #94a3b8;'>
            💡 <strong style='color: #c4b5fd;'>Demo Mode Active</strong><br>
            Generates music using music theory heuristics. 
            Train a model for AI-powered generation.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-size: 11px; color: #475569; text-align: center;'>
            Built by <strong style='color: #8b5cf6;'>Rohit Birdawade</strong><br>
            <a href='https://github.com/rohitbirdawade007' 
               style='color: #3b82f6; text-decoration: none;'>
               🐙 GitHub
            </a>
        </div>
        """, unsafe_allow_html=True)

    # ── Route to page ──────────────────────────────────────────────────────
    if page == "🎹 Generate Music":
        from app.pages.generate import show_generate_page
        show_generate_page()
    elif page == "📤 Upload Style (RAG)":
        from app.pages.upload_style import show_upload_page
        show_upload_page()
    elif page == "📊 Visualize":
        from app.pages.visualize import show_visualize_page
        show_visualize_page()
    elif page == "ℹ️ About":
        from app.pages.about import show_about_page
        show_about_page()


if __name__ == "__main__":
    main()
