"""About Page — Project information and architecture showcase."""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st


def show_about_page():
    st.markdown("""
    <div style='padding: 8px 0 32px;'>
        <div class='hero-title'>ℹ️ About This Project</div>
        <p class='hero-subtitle'>
            AI-Based Music Composition System Using Transformer Networks — 
            an end-to-end Generative AI project demonstrating practical implementation 
            of deep learning for creative applications.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Project Banner ─────────────────────────────────────────────────────
    st.markdown("""
    <div class='music-card glow-border' style='text-align: center; padding: 40px;'>
        <div style='font-size: 4rem; margin-bottom: 16px;'>🎵</div>
        <h1 style='background: linear-gradient(135deg, #8b5cf6, #3b82f6, #14b8a6); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 1.8rem; font-weight: 800; margin: 0 0 12px;'>
            AI Music Composition System
        </h1>
        <p style='color: #94a3b8; font-size: 1rem; max-width: 600px; margin: 0 auto 20px;'>
            Using Transformer Networks, RAG, and Music Theory Constraints to generate 
            original musical compositions from symbolic MIDI data.
        </p>
        <div style='display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;'>
            <a href='https://github.com/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks'
               target='_blank' style='text-decoration: none;'>
                <div style='background: rgba(139,92,246,0.2); border: 1px solid rgba(139,92,246,0.5);
                            border-radius: 100px; padding: 8px 20px; color: #c4b5fd; font-weight: 600;
                            display: flex; align-items: center; gap: 8px;'>
                    🐙 GitHub Repository
                </div>
            </a>
            <a href='https://colab.research.google.com/github/rohitbirdawade007/AI-Based-Music-Composition-System-Using-Transformer-Networks/blob/main/notebooks/02_model_training.ipynb'
               target='_blank' style='text-decoration: none;'>
                <div style='background: rgba(249,115,22,0.2); border: 1px solid rgba(249,115,22,0.5);
                            border-radius: 100px; padding: 8px 20px; color: #fb923c; font-weight: 600;
                            display: flex; align-items: center; gap: 8px;'>
                    ☁️ Open in Colab
                </div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture Overview ──────────────────────────────────────────────
    st.markdown("## 🏗️ System Architecture")
    st.markdown("""
    <div class='music-card'>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;'>
        <div style='background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3); 
                    border-radius: 14px; padding: 20px;'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>📥</div>
            <strong style='color: #c4b5fd;'>1. Data Pipeline</strong>
            <p style='color: #94a3b8; font-size: 13px; margin: 8px 0 0;'>
                MIDI parsing → Note extraction → Tokenization → 
                Sequence windowing → Train/Val/Test split
            </p>
        </div>
        <div style='background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); 
                    border-radius: 14px; padding: 20px;'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>🤖</div>
            <strong style='color: #93c5fd;'>2. Transformer Model</strong>
            <p style='color: #94a3b8; font-size: 13px; margin: 8px 0 0;'>
                6-layer Encoder-Decoder · 8 attention heads · 
                256-dim embeddings · Causal masking
            </p>
        </div>
        <div style='background: rgba(20,184,166,0.1); border: 1px solid rgba(20,184,166,0.3); 
                    border-radius: 14px; padding: 20px;'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>🎼</div>
            <strong style='color: #5eead4;'>3. Theory Engine</strong>
            <p style='color: #94a3b8; font-size: 13px; margin: 8px 0 0;'>
                Key/Scale validation · Chord progressions · 
                Interval consonance · Rhythm consistency
            </p>
        </div>
        <div style='background: rgba(236,72,153,0.1); border: 1px solid rgba(236,72,153,0.3); 
                    border-radius: 14px; padding: 20px;'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>🔍</div>
            <strong style='color: #f9a8d4;'>4. RAG Pipeline</strong>
            <p style='color: #94a3b8; font-size: 13px; margin: 8px 0 0;'>
                Feature extraction → 128-D embeddings → 
                FAISS index → Top-K retrieval → Style guidance
            </p>
        </div>
        <div style='background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); 
                    border-radius: 14px; padding: 20px;'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>⚡</div>
            <strong style='color: #fcd34d;'>5. Inference Engine</strong>
            <p style='color: #94a3b8; font-size: 13px; margin: 8px 0 0;'>
                Greedy · Temperature · Top-K · Top-P · Beam Search · Demo mode
            </p>
        </div>
        <div style='background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); 
                    border-radius: 14px; padding: 20px;'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>🌐</div>
            <strong style='color: #6ee7b7;'>6. Web Application</strong>
            <p style='color: #94a3b8; font-size: 13px; margin: 8px 0 0;'>
                Streamlit UI · Flask REST API · 
                Piano roll viz · MIDI download
            </p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tech Stack ─────────────────────────────────────────────────────────
    st.markdown("## 🛠️ Technology Stack")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='music-card'>
            <strong style='color: #c4b5fd;'>🤖 Deep Learning</strong>
            <ul style='color: #94a3b8; font-size: 13px; margin-top: 8px; padding-left: 16px;'>
                <li>PyTorch 2.0+</li>
                <li>Custom Transformer Encoder-Decoder</li>
                <li>Multi-Head Attention</li>
                <li>Sinusoidal/Learnable PE</li>
                <li>AMP Mixed Precision</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='music-card'>
            <strong style='color: #93c5fd;'>🎵 Music Processing</strong>
            <ul style='color: #94a3b8; font-size: 13px; margin-top: 8px; padding-left: 16px;'>
                <li>pretty_midi / mido</li>
                <li>music21</li>
                <li>midiutil</li>
                <li>Custom MusicTokenizer (512 vocab)</li>
                <li>Music Theory Engine</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='music-card'>
            <strong style='color: #5eead4;'>🔍 RAG & Retrieval</strong>
            <ul style='color: #94a3b8; font-size: 13px; margin-top: 8px; padding-left: 16px;'>
                <li>FAISS vector database</li>
                <li>128-D feature embeddings</li>
                <li>Pitch/Rhythm/Velocity features</li>
                <li>Top-K similarity search</li>
                <li>Style-aware generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("""
        <div class='music-card'>
            <strong style='color: #fcd34d;'>🌐 Web & API</strong>
            <ul style='color: #94a3b8; font-size: 13px; margin-top: 8px; padding-left: 16px;'>
                <li>Streamlit (Frontend)</li>
                <li>Flask REST API (Backend)</li>
                <li>Flask-CORS</li>
                <li>SQLite (History)</li>
                <li>Streamlit Community Cloud</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class='music-card'>
            <strong style='color: #6ee7b7;'>📊 Visualization</strong>
            <ul style='color: #94a3b8; font-size: 13px; margin-top: 8px; padding-left: 16px;'>
                <li>Matplotlib (Piano Roll)</li>
                <li>Plotly (Interactive)</li>
                <li>Seaborn (Heatmaps)</li>
                <li>Attention weight viz</li>
                <li>Training metrics dash</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class='music-card'>
            <strong style='color: #f9a8d4;'>⚙️ DevOps</strong>
            <ul style='color: #94a3b8; font-size: 13px; margin-top: 8px; padding-left: 16px;'>
                <li>Git + GitHub</li>
                <li>GitHub Actions CI</li>
                <li>pytest + coverage</li>
                <li>flake8 + black</li>
                <li>Google Colab Notebooks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Features ───────────────────────────────────────────────────────
    st.markdown("## ✨ Key Features")
    features = [
        ("🎵 AI Music Generation", "Generate original melodies from seed notes using Transformer Networks with autoregressive decoding"),
        ("🎼 Music Theory Engine", "Validate and improve output using key signatures, scales, chord progressions, and interval rules"),
        ("🔍 RAG-Based Style Retrieval", "Upload MIDI files to guide generation style via FAISS vector similarity search"),
        ("🎛️ Interactive Controls", "Fine-tune genre, mood, key, mode, tempo, length, sampling strategy, and more"),
        ("📊 Rich Visualizations", "Piano roll, pitch histograms, attention heatmaps, and training dashboards"),
        ("⬇️ MIDI Download", "Download generated compositions as standard MIDI files compatible with all DAWs"),
        ("🔌 REST API", "Flask API for programmatic access — integrate with any application"),
        ("☁️ Google Colab Ready", "Train the model on GPU using the included Colab notebook"),
    ]

    for i in range(0, len(features), 2):
        c1, c2 = st.columns(2)
        for col, (title, desc) in zip([c1, c2], features[i:i+2]):
            col.markdown(f"""
            <div class='music-card' style='padding: 20px;'>
                <strong style='color: #c4b5fd; font-size: 1rem;'>{title}</strong>
                <p style='color: #94a3b8; font-size: 13px; margin: 8px 0 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Author ─────────────────────────────────────────────────────────────
    st.markdown("## 👨‍💻 About the Author")
    st.markdown("""
    <div class='music-card' style='display: flex; align-items: center; gap: 24px; flex-wrap: wrap;'>
        <div style='font-size: 4rem;'>👨‍💻</div>
        <div>
            <h3 style='color: #c4b5fd; margin: 0 0 8px;'>Rohit Birdawade</h3>
            <p style='color: #94a3b8; margin: 0 0 12px; font-size: 14px;'>
                AI/ML Engineer · Generative AI Specialist · Full-Stack Developer
            </p>
            <div style='display: flex; gap: 12px; flex-wrap: wrap;'>
                <a href='https://github.com/rohitbirdawade007' target='_blank' style='text-decoration: none;'>
                    <div style='background: rgba(139,92,246,0.2); border: 1px solid rgba(139,92,246,0.4);
                                border-radius: 100px; padding: 6px 16px; color: #c4b5fd; font-size: 13px;'>
                        🐙 GitHub
                    </div>
                </a>
            </div>
        </div>
        <div style='flex: 1; min-width: 250px;'>
            <p style='color: #94a3b8; font-size: 14px; line-height: 1.7;'>
                This project demonstrates end-to-end AI engineering skills spanning 
                Transformer architecture implementation, symbolic music processing, 
                MIDI tokenization, RAG pipelines, vector databases, and full-stack 
                web application development.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
