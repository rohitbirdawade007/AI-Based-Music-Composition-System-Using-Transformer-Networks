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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

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

h1, h2, h3, h4, h5, h6, .hero-title {
    font-family: 'Outfit', -apple-system, sans-serif !important;
    font-weight: 700 !important;
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
        <div style='text-align:center; padding: 12px 0 20px;'>
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 10px rgba(139, 92, 246, 0.6)); margin-bottom: 8px;">
                <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm1 14.5h-2v-9h2v9zm4 2h-2v-13h2v13zm-8-4H7v-5h2v5z" fill="url(#grad)" />
                <defs>
                    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#8b5cf6" />
                        <stop offset="100%" stop-color="#14b8a6" />
                    </linearGradient>
                </defs>
            </svg>
            <div style='font-family: "Outfit", sans-serif; font-size: 1.3rem; font-weight: 800; letter-spacing: -0.5px;
                        background: linear-gradient(135deg, #8b5cf6, #3b82f6 50%, #14b8a6);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;'>
                SONIC NEURALIS
            </div>
            <div style='font-size: 0.65rem; color: #475569; margin-top: 4px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;'>
                Enterprise Composer
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

        # Enterprise System Monitor
        st.markdown("""
        <div style="background: rgba(26, 26, 46, 0.4); border: 1px solid rgba(139, 92, 246, 0.15); border-radius: 12px; padding: 14px; margin-top: 8px;">
            <div style="font-size: 10px; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <span>System Status</span>
                <span style="color: #14b8a6; display: flex; align-items: center; gap: 4px; font-weight: 600;">
                    <span style="display: inline-block; width: 6px; height: 6px; background: #14b8a6; border-radius: 50%;"></span>
                    Active
                </span>
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px; display: flex; justify-content: space-between;">
                <span>Generation Engine</span>
                <span style="font-weight: 600; color: #8b5cf6;">Demo (Heuristics)</span>
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px; display: flex; justify-content: space-between;">
                <span>REST API Server</span>
                <span style="font-weight: 600; color: #14b8a6;">Online</span>
            </div>
            <hr style="margin: 8px 0; border: none; height: 1px; background: rgba(139, 92, 246, 0.15);" />
            <!-- CPU Bar -->
            <div style="margin-bottom: 6px;">
                <div style="font-size: 10px; color: #94a3b8; display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>CPU Load</span>
                    <span>14%</span>
                </div>
                <div style="background: rgba(255,255,255,0.05); height: 4px; border-radius: 2px; overflow: hidden;">
                    <div style="background: #3b82f6; width: 14%; height: 100%;"></div>
                </div>
            </div>
            <!-- VRAM Bar -->
            <div>
                <div style="font-size: 10px; color: #94a3b8; display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>GPU Memory</span>
                    <span>0.0 / 16 GB</span>
                </div>
                <div style="background: rgba(255,255,255,0.05); height: 4px; border-radius: 2px; overflow: hidden;">
                    <div style="background: #475569; width: 0%; height: 100%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # User Section
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; background: rgba(26,26,46,0.3); border: 1px solid rgba(139,92,246,0.1); border-radius: 12px; padding: 10px; margin-top: 16px;">
            <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #8b5cf6, #3b82f6); display: flex; justify-content: center; align-items: center; font-weight: 700; color: white; font-size: 11px; border: 1px solid rgba(255,255,255,0.2);">
                RB
            </div>
            <div style="flex: 1; overflow: hidden;">
                <div style="font-size: 11px; font-weight: 600; color: #f1f5f9; white-space: nowrap; text-overflow: ellipsis; text-align: left;">Rohit Birdawade</div>
                <div style="font-size: 9px; color: #94a3b8; white-space: nowrap; text-overflow: ellipsis; text-align: left;">Lead AI Engineer</div>
            </div>
            <a href="https://github.com/rohitbirdawade007" target="_blank" style="text-decoration: none; font-size: 1.1rem; color: #8b5cf6; display: flex; align-items: center;">
                🐙
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
