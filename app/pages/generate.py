"""Generate Music Page — Main music generation UI."""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st


def show_generate_page():
    # ── Page Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding: 8px 0 32px;'>
        <div class='hero-title'>🎹 Generate Music</div>
        <p class='hero-subtitle'>
            Configure your composition parameters and let the AI create original music 
            using Transformer Networks and music theory constraints.
        </p>
        <div>
            <span class='badge'>🤖 AI-Powered</span>
            <span class='badge'>🎼 Theory Engine</span>
            <span class='badge'>🔍 RAG-Ready</span>
            <span class='badge'>⚡ Real-time</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Layout ────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        st.markdown("### ⚙️ Generation Parameters")

        with st.container():
            # ── Genre & Mood ───────────────────────────────────────────────
            st.markdown("**🎭 Style**")
            col1, col2 = st.columns(2)
            with col1:
                genre = st.selectbox(
                    "Genre",
                    ["Classical", "Jazz", "Pop", "Blues", "Electronic", "Ambient", "Folk"],
                    index=0,
                    key="genre_select",
                )
            with col2:
                mood = st.selectbox(
                    "Mood",
                    ["Calm", "Happy", "Dramatic", "Mysterious", "Energetic", "Melancholic", "Romantic"],
                    index=0,
                    key="mood_select",
                )

            # ── Key & Mode ─────────────────────────────────────────────────
            st.markdown("**🎵 Key & Scale**")
            col3, col4 = st.columns(2)
            with col3:
                key = st.selectbox(
                    "Key",
                    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
                    index=0,
                    key="key_select",
                )
            with col4:
                mode = st.selectbox(
                    "Mode / Scale",
                    ["major", "minor", "dorian", "blues", "pentatonic_major", "pentatonic_minor",
                     "lydian", "mixolydian", "phrygian"],
                    index=0,
                    key="mode_select",
                )

            # ── Tempo & Length ─────────────────────────────────────────────
            st.markdown("**🥁 Tempo & Length**")
            tempo = st.slider(
                "Tempo (BPM)",
                min_value=40, max_value=220, value=120, step=5,
                key="tempo_slider",
                help="Beats per minute. 60-80=slow, 120=medium, 140+=fast"
            )
            num_notes = st.slider(
                "Number of Notes",
                min_value=16, max_value=256, value=64, step=8,
                key="notes_slider",
            )

            # ── Instrument ─────────────────────────────────────────────────
            instrument = st.selectbox(
                "🎸 Instrument",
                ["Piano", "Guitar", "Strings", "Flute", "Violin", "Organ", "Synth"],
                key="instrument_select",
            )

            # ── Advanced Options ───────────────────────────────────────────
            with st.expander("🔬 Advanced Sampling Options"):
                temperature = st.slider(
                    "Temperature",
                    min_value=0.1, max_value=2.0, value=1.0, step=0.05,
                    help="Higher = more random, Lower = more deterministic",
                    key="temp_slider",
                )
                col5, col6 = st.columns(2)
                with col5:
                    top_k = st.slider("Top-K", min_value=1, max_value=100, value=50, key="topk_slider")
                with col6:
                    top_p = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05, key="topp_slider")

                use_beam = st.checkbox("Use Beam Search", value=False, key="beam_cb")
                if use_beam:
                    beam_width = st.slider("Beam Width", 2, 8, 4, key="beam_slider")
                else:
                    beam_width = 4

                apply_theory = st.checkbox("Apply Music Theory Constraints", value=True, key="theory_cb",
                                           help="Snap notes to scale, validate chord progressions")

            # ── Seed Notes ────────────────────────────────────────────────
            with st.expander("🌱 Seed Melody (Optional)"):
                st.markdown(
                    "<small style='color: #64748b;'>Enter MIDI note numbers (0-127) separated by commas. "
                    "C4=60, E4=64, G4=67</small>",
                    unsafe_allow_html=True,
                )
                seed_input = st.text_input(
                    "Seed notes", placeholder="60, 64, 67, 72", key="seed_input"
                )
                seed_notes = []
                if seed_input.strip():
                    try:
                        seed_notes = [int(x.strip()) for x in seed_input.split(",") if x.strip()]
                        seed_notes = [max(0, min(127, n)) for n in seed_notes]
                        st.success(f"✓ {len(seed_notes)} seed notes loaded")
                    except ValueError:
                        st.error("Invalid format. Use comma-separated integers.")

        # ── Generate Button ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button("🎵 Generate Music", key="generate_btn", use_container_width=True)

    # ── Output Panel ───────────────────────────────────────────────────────
    with right_col:
        st.markdown("### 🎼 Generated Composition")

        # Init session state
        if "generation_result" not in st.session_state:
            st.session_state.generation_result = None
        if "generation_count" not in st.session_state:
            st.session_state.generation_count = 0

        if generate_clicked:
            with st.spinner("🎵 Composing your music..."):
                try:
                    from src.inference.generator import MusicGenerator
                    generator = MusicGenerator(demo_mode=True)
                    t0 = time.time()
                    result = generator.generate(
                        seed_pitches=seed_notes,
                        genre=genre,
                        mood=mood,
                        key=key,
                        mode=mode,
                        tempo=tempo,
                        num_notes=num_notes,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        use_beam=use_beam,
                        beam_width=beam_width,
                        apply_theory=apply_theory,
                    )
                    st.session_state.generation_result = result
                    st.session_state.generation_count += 1
                    st.success(f"✅ Generated {len(result.pitches)} notes in {result.generation_time*1000:.0f}ms")
                except Exception as e:
                    st.error(f"Generation error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        result = st.session_state.generation_result

        if result is None:
            # Placeholder
            st.markdown("""
            <div class='music-card' style='text-align: center; padding: 60px 20px;'>
                <div style='font-size: 4rem; margin-bottom: 16px;'>🎵</div>
                <div style='color: #64748b; font-size: 1.1rem;'>
                    Configure parameters and click<br>
                    <strong style='color: #8b5cf6;'>Generate Music</strong> to start composing
                </div>
                <div style='margin-top: 20px; color: #475569; font-size: 0.85rem;'>
                    Powered by Transformer Networks + Music Theory Engine
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ── Metrics Row ────────────────────────────────────────────────
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Notes", len(result.pitches))
            mc2.metric("Theory Score", f"{result.theory_score:.2f}")
            mc3.metric("Tempo", f"{result.tempo} BPM")
            mc4.metric("Method", result.method.capitalize())

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Tabs for output ────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs(["🔊 Play Audio", "🎹 Piano Roll", "🎵 Note Sequence", "📥 Download"])

            with tab1:
                try:
                    from src.utils.audio import AudioUtils
                    audio_util = AudioUtils()
                    player_html = audio_util.generate_audio_html(
                        result.midi_bytes,
                        title=f"{genre} · {mood} · {key} {mode}"
                    )
                    st.components.v1.html(player_html, height=380, scrolling=False)
                except Exception as e:
                    st.error(f"Could not load audio player: {e}")

            with tab2:
                try:
                    from src.utils.visualization import MusicVisualizer
                    viz = MusicVisualizer()
                    fig = viz.plot_piano_roll(
                        result.pitches,
                        result.durations,
                        result.velocities,
                        title=f"{genre} · {mood} · {key} {mode} · {tempo} BPM",
                    )
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Visualization unavailable: {e}")
                    # Fallback: text representation
                    _show_text_piano_roll(result.pitches)

            with tab3:
                _show_note_sequence(result.pitches, result.durations, result.velocities, key, mode)

            with tab4:
                _show_download_section(result)

            # ── Theory Validation ──────────────────────────────────────────
            with st.expander("🎼 Music Theory Analysis"):
                _show_theory_analysis(result, key, mode)

    # ── Generation History ─────────────────────────────────────────────────
    if st.session_state.generation_count > 0:
        st.markdown("---")
        st.markdown(f"### 📋 Session History ({st.session_state.generation_count} compositions)")
        if "history" not in st.session_state:
            st.session_state.history = []
        if generate_clicked and result:
            st.session_state.history.insert(0, {
                "genre": genre, "mood": mood, "key": key, "mode": mode,
                "notes": len(result.pitches), "score": round(result.theory_score, 3),
                "time": time.strftime("%H:%M:%S"),
            })
            st.session_state.history = st.session_state.history[:10]

        if st.session_state.history:
            import pandas as pd
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "score": st.column_config.ProgressColumn("Theory Score", min_value=0, max_value=1),
                }
            )


# ── Helper rendering functions ─────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_to_name(pitch: int) -> str:
    if pitch <= 0:
        return "REST"
    octave = pitch // 12 - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


def _show_text_piano_roll(pitches):
    # Simple ASCII-art piano roll fallback
    if not pitches:
        return
    min_p = max(min(pitches) - 2, 21)
    max_p = min(max(pitches) + 2, 108)
    lines = []
    for p in range(max_p, min_p - 1, -1):
        name = f"{_pitch_to_name(p):4s}"
        row = "".join("█" if n == p else "·" for n in pitches[:60])
        lines.append(f"`{name}│{row}`")
    st.markdown("\n".join(lines[:20]))


def _show_note_sequence(pitches, durations, velocities, key, mode):
    if not pitches:
        return

    note_pills = " ".join(
        f"<span class='note-pill'>{_pitch_to_name(p)}</span>"
        for p in pitches[:48]
    )
    st.markdown(
        f"<div style='line-height: 2.2;'>{note_pills}"
        + (f" <span style='color:#475569; font-size:11px;'>+ {len(pitches)-48} more...</span>" if len(pitches) > 48 else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**First 16 notes:**")
    import pandas as pd
    rows = []
    for i, (p, d, v) in enumerate(zip(pitches[:16], durations[:16], velocities[:16])):
        rows.append({
            "#": i + 1,
            "Note": _pitch_to_name(p),
            "MIDI": p,
            "Duration (s)": round(d, 3),
            "Velocity": v,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _show_download_section(result):
    st.markdown("""
    <div class='music-card' style='text-align: center;'>
        <div style='font-size: 2rem; margin-bottom: 12px;'>📥</div>
        <div style='color: #94a3b8; margin-bottom: 20px;'>
            Download your generated composition as a MIDI file
        </div>
    </div>
    """, unsafe_allow_html=True)

    filename = f"ai_music_{result.genre.lower()}_{result.key}{result.mode}_{result.tempo}bpm.mid"

    st.download_button(
        label="⬇️ Download MIDI File",
        data=result.midi_bytes,
        file_name=filename,
        mime="audio/midi",
        use_container_width=True,
        key="midi_download_btn",
    )

    st.markdown("""
    <div style='text-align: center; margin-top: 12px; color: #475569; font-size: 12px;'>
        Open the downloaded .mid file with GarageBand, MuseScore, Logic Pro,<br>
        Ableton Live, or any DAW / media player that supports MIDI.
    </div>
    """, unsafe_allow_html=True)

    # Show MIDI info
    st.markdown("---")
    st.markdown("**File Information**")
    col1, col2 = st.columns(2)
    col1.markdown(f"- **Format:** MIDI Type 0")
    col1.markdown(f"- **Tempo:** {result.tempo} BPM")
    col1.markdown(f"- **Key:** {result.key} {result.mode}")
    col2.markdown(f"- **Notes:** {len(result.pitches)}")
    col2.markdown(f"- **Genre:** {result.genre}")
    col2.markdown(f"- **Size:** {len(result.midi_bytes):,} bytes")


def _show_theory_analysis(result, key, mode):
    try:
        from src.theory.music_theory_engine import MusicTheoryEngine
        engine = MusicTheoryEngine(key=key, mode=mode)
        validation = engine.validate_melody(result.pitches, result.durations)

        score_color = "#14b8a6" if validation.score > 0.7 else "#f59e0b" if validation.score > 0.4 else "#ef4444"
        st.markdown(
            f"<div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>"
            f"<div style='font-size: 2rem;'>🎼</div>"
            f"<div><strong style='color: {score_color}; font-size: 1.3rem;'>"
            f"Score: {validation.score:.2f}/1.00</strong><br>"
            f"<span style='color: #64748b; font-size: 0.85rem;'>"
            f"{'Excellent' if validation.score > 0.8 else 'Good' if validation.score > 0.6 else 'Fair'} musical quality</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("In-Scale Ratio", f"{validation.in_scale_ratio:.0%}")
        c2.metric("Consonance", f"{validation.consonance_score:.0%}")
        c3.metric("Range Score", f"{validation.range_score:.0%}")

        if validation.violations:
            st.warning("**Violations detected:**")
            for v in validation.violations:
                st.markdown(f"- ⚠️ {v}")

        if validation.suggestions:
            st.info("**Suggestions:**")
            for s in validation.suggestions:
                st.markdown(f"- 💡 {s}")

        if validation.is_valid:
            st.success("✅ All music theory checks passed!")

    except Exception as e:
        st.warning(f"Theory analysis unavailable: {e}")
