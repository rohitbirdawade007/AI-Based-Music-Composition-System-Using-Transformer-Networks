"""Upload Style (RAG) Page — Upload MIDI for style-guided generation."""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st


def show_upload_page():
    st.markdown("""
    <div style='padding: 8px 0 32px;'>
        <div class='hero-title'>📤 Style Upload (RAG)</div>
        <p class='hero-subtitle'>
            Upload your own MIDI files to guide the AI's composition style. 
            The system extracts musical features, creates embeddings, and stores them 
            in a FAISS vector database for style-aware retrieval.
        </p>
        <div>
            <span class='badge'>🔍 FAISS Vector DB</span>
            <span class='badge'>📊 Feature Extraction</span>
            <span class='badge'>🎯 Style-Aware</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── RAG Pipeline Diagram ───────────────────────────────────────────────
    with st.expander("🔍 How RAG Works for Music Generation", expanded=False):
        st.markdown("""
        <div class='music-card'>
        <div style='display: flex; align-items: center; gap: 16px; flex-wrap: wrap; 
                    justify-content: center; text-align: center;'>
            <div style='background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.4); 
                        border-radius: 12px; padding: 16px; min-width: 120px;'>
                <div style='font-size: 1.8rem;'>🎵</div>
                <div style='font-size: 12px; color: #94a3b8; margin-top: 4px;'>Upload MIDI</div>
            </div>
            <div style='font-size: 1.5rem; color: #8b5cf6;'>→</div>
            <div style='background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4); 
                        border-radius: 12px; padding: 16px; min-width: 120px;'>
                <div style='font-size: 1.8rem;'>📊</div>
                <div style='font-size: 12px; color: #94a3b8; margin-top: 4px;'>Extract Features</div>
            </div>
            <div style='font-size: 1.5rem; color: #8b5cf6;'>→</div>
            <div style='background: rgba(20,184,166,0.15); border: 1px solid rgba(20,184,166,0.4); 
                        border-radius: 12px; padding: 16px; min-width: 120px;'>
                <div style='font-size: 1.8rem;'>🔢</div>
                <div style='font-size: 12px; color: #94a3b8; margin-top: 4px;'>128-D Embedding</div>
            </div>
            <div style='font-size: 1.5rem; color: #8b5cf6;'>→</div>
            <div style='background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.4); 
                        border-radius: 12px; padding: 16px; min-width: 120px;'>
                <div style='font-size: 1.8rem;'>🗄️</div>
                <div style='font-size: 12px; color: #94a3b8; margin-top: 4px;'>FAISS Index</div>
            </div>
            <div style='font-size: 1.5rem; color: #8b5cf6;'>→</div>
            <div style='background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.4); 
                        border-radius: 12px; padding: 16px; min-width: 120px;'>
                <div style='font-size: 1.8rem;'>🎼</div>
                <div style='font-size: 12px; color: #94a3b8; margin-top: 4px;'>Style-Guided Gen</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Extracted Features (48 dimensions):**
        - **Pitch Histogram** (12 dims): Distribution of pitch classes in the MIDI
        - **Rhythm Patterns** (16 dims): Note duration distribution and rhythmic density
        - **Velocity Profile** (8 dims): Dynamic range and velocity distribution
        - **Interval Histogram** (12 dims): Melodic interval frequency
        
        These are projected to a 128-dimensional embedding and stored in FAISS for fast similarity search.
        During generation, the top-5 most similar patterns are retrieved and used to guide the Transformer.
        """)

    st.markdown("---")

    # ── Upload Area ────────────────────────────────────────────────────────
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("### 📁 Upload MIDI Files")
        uploaded_files = st.file_uploader(
            "Drag and drop MIDI files here",
            type=["mid", "midi"],
            accept_multiple_files=True,
            key="midi_uploader",
            help="Upload .mid or .midi files to add to the style database",
        )

        if uploaded_files:
            for midi_file in uploaded_files:
                with st.spinner(f"Processing {midi_file.name}..."):
                    try:
                        midi_bytes = midi_file.read()
                        _process_uploaded_midi(midi_bytes, midi_file.name)
                    except Exception as e:
                        st.error(f"Error processing {midi_file.name}: {e}")

        st.markdown("""
        <div style='margin-top: 16px; padding: 16px; 
                    background: rgba(59,130,246,0.08); 
                    border: 1px solid rgba(59,130,246,0.3); 
                    border-radius: 12px; font-size: 13px; color: #94a3b8;'>
            <strong style='color: #60a5fa;'>📌 Supported formats:</strong> .mid, .midi<br>
            <strong style='color: #60a5fa;'>📌 Max file size:</strong> 16 MB<br>
            <strong style='color: #60a5fa;'>📌 Max files:</strong> 100 in index<br>
            <strong style='color: #60a5fa;'>📌 Privacy:</strong> Files are processed locally, not sent externally
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🗄️ Style Database")
        _show_rag_database_stats()

    st.markdown("---")

    # ── Test retrieval ─────────────────────────────────────────────────────
    st.markdown("### 🔍 Test Style Retrieval")
    col3, col4 = st.columns([1, 1])
    with col3:
        test_key = st.selectbox("Query Key", ["C", "G", "D", "A", "F"], key="rag_test_key")
        test_mode = st.selectbox("Query Mode", ["major", "minor", "blues"], key="rag_test_mode")
    with col4:
        test_tempo = st.slider("Query Tempo", 60, 180, 120, key="rag_test_tempo")

    if st.button("🔍 Find Similar Styles", key="rag_search_btn"):
        _show_retrieval_results(test_key, test_mode, test_tempo)


def _process_uploaded_midi(midi_bytes: bytes, filename: str):
    """Process and index an uploaded MIDI file."""
    try:
        retriever = _get_global_retriever()
        
        # Check for duplicates
        if any(m.get("file_name") == filename for m in retriever._metadata):
            st.info(f"ℹ️ {filename} is already indexed in style database.")
            return

        from src.rag.embedder import MusicEmbedder
        embedder = MusicEmbedder()
        
        retriever.add_midi(midi_bytes, filename, metadata={"filename": filename})

        # Show success with features
        st.markdown(f"""
        <div style='background: rgba(20,184,166,0.1); border: 1px solid rgba(20,184,166,0.4); 
                    border-radius: 12px; padding: 16px; margin: 8px 0;'>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span style='font-size: 1.4rem;'>✅</span>
                <div>
                    <strong style='color: #14b8a6;'>{filename}</strong><br>
                    <span style='color: #94a3b8; font-size: 13px;'>
                        Indexed · Embedding dim: 128 · FAISS updated
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except ImportError:
        st.warning(
            "⚠️ RAG components need additional packages. "
            "Install with: `pip install faiss-cpu`"
        )
    except Exception as e:
        st.error(f"Processing failed: {e}")


def _get_global_retriever():
    """Get or create a global retriever from session state and auto-load samples."""
    if "rag_retriever" not in st.session_state:
        from src.rag.retriever import MusicRetriever
        retriever = MusicRetriever()
        
        # Auto-index sample files if index is empty
        try:
            sample_dir = Path("data/sample")
            if sample_dir.exists():
                for midi_path in sample_dir.glob("*.mid"):
                    if not any(m.get("file_name") == midi_path.name for m in retriever._metadata):
                        midi_bytes = midi_path.read_bytes()
                        retriever.add_midi(midi_bytes, midi_path.name, metadata={"filename": midi_path.name})
        except Exception as e:
            st.warning(f"Could not load sample MIDI files: {e}")
            
        st.session_state.rag_retriever = retriever
    return st.session_state.rag_retriever


def _show_rag_database_stats():
    """Display RAG database statistics."""
    retriever = _get_global_retriever()
    stats = retriever.get_stats()

    if retriever.size == 0:
        st.markdown("""
        <div class='music-card' style='text-align: center; padding: 40px 20px;'>
            <div style='font-size: 3rem; margin-bottom: 12px;'>🗄️</div>
            <div style='color: #64748b;'>No styles indexed yet.<br>Upload MIDI files to begin.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    m1, m2 = st.columns(2)
    m1.metric("Indexed Files", stats.get("unique_files", 0))
    m2.metric("Index Size", f"{stats.get('total_entries', 0)} vectors")

    files = [m.get("file_name", "Unknown") for m in retriever._metadata]
    if files:
        st.markdown("**Indexed files (latest):**")
        for f in files[-5:]:
            st.markdown(f"- 📄 {f}")


def _create_query_embedding(key: str, mode: str, tempo: int) -> np.ndarray:
    """Create a query embedding vector by synthesizing a scale-aligned melody."""
    from src.data.midi_parser import ParsedNote
    from src.theory.music_theory_engine import MusicTheoryEngine
    from src.rag.embedder import MusicEmbedder
    
    # 1. Get scale pitches
    engine = MusicTheoryEngine(key=key, mode=mode)
    scale_pitches = engine.get_scale_pitches()
    
    # 2. Build synthetic note sequence (quarter-note melody stepping up/down)
    notes = []
    beat_dur = 60.0 / tempo
    
    pitches_sequence = []
    for step in range(16):
        pitch_class = scale_pitches[step % len(scale_pitches)]
        octave = 5 if step % 2 == 0 else 4
        pitch = 12 * octave + pitch_class
        pitches_sequence.append(pitch)
        
    start_time = 0.0
    for p in pitches_sequence:
        notes.append(ParsedNote(
            pitch=p,
            start_time=start_time,
            end_time=start_time + beat_dur,
            duration=beat_dur,
            velocity=80,
            channel=0
        ))
        start_time += beat_dur
        
    # 3. Embed synthetic notes
    embedder = MusicEmbedder()
    return embedder.embed(notes)


def _show_retrieval_results(key, mode, tempo):
    """Show similar style retrieval results."""
    retriever = _get_global_retriever()
    if retriever.size == 0:
        st.info("No styles indexed yet. Upload MIDI files first.")
        return

    try:
        import numpy as np
        query_emb = _create_query_embedding(key, mode, tempo)
        results = retriever.search(query_emb, top_k=3)

        if not results:
            st.info("No similar styles found in the database.")
            return

        st.markdown("**Top matching styles:**")
        for i, r in enumerate(results):
            score = r.get("score", 0.0)
            col = ["#8b5cf6", "#3b82f6", "#14b8a6"][i % 3]
            st.markdown(f"""
            <div style='background: rgba(139,92,246,0.08); border: 1px solid rgba(139,92,246,0.3);
                        border-radius: 12px; padding: 12px; margin: 6px 0;
                        display: flex; align-items: center; gap: 12px;'>
                <div style='font-size: 1.5rem;'>🎵</div>
                <div style='flex: 1;'>
                    <strong style='color: {col};'>#{i+1} {r.get("file_name", "Unknown")}</strong><br>
                    <div style='color: #64748b; font-size: 12px;'>
                        Similarity: {score:.3f} · Rank: {r.get("rank", i+1)}
                    </div>
                </div>
                <div style='background: {col}22; border: 1px solid {col}44; 
                            border-radius: 8px; padding: 4px 12px;
                            color: {col}; font-size: 12px; font-weight: 600;'>
                    {max(0.0, score):.0%} match
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Retrieval failed: {e}")
        import traceback
        st.code(traceback.format_exc())
