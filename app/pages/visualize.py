"""Visualize Page — Music analysis and visualization dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st


def show_visualize_page():
    st.markdown("""
    <div style='padding: 8px 0 32px;'>
        <div class='hero-title'>📊 Visualizations</div>
        <p class='hero-subtitle'>
            Explore interactive visualizations of generated music — piano rolls, 
            pitch distributions, attention heatmaps, and training metrics.
        </p>
        <div>
            <span class='badge'>🎹 Piano Roll</span>
            <span class='badge'>🔥 Attention Maps</span>
            <span class='badge'>📈 Metrics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check if we have a recent result
    result = st.session_state.get("generation_result")

    if result is None:
        st.info("💡 Generate some music first on the **Generate Music** page to see visualizations here.")
        # Show demo visualizations with random data
        st.markdown("### Demo Visualizations (random data)")
        _show_demo_visualizations()
        return

    # ── Real result visualizations ─────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎹 Piano Roll", "📊 Pitch Distribution", "🔥 Attention Heatmap", "📈 Training Metrics"]
    )

    with tab1:
        _show_piano_roll_tab(result)

    with tab2:
        _show_pitch_distribution_tab(result)

    with tab3:
        _show_attention_heatmap_tab()

    with tab4:
        _show_training_metrics_tab()


def _show_piano_roll_tab(result):
    st.markdown("### 🎹 Piano Roll Visualization")
    st.markdown(
        "<p style='color: #64748b;'>Visual representation of note pitch (y-axis) vs. time (x-axis). "
        "Bar height represents velocity (loudness).</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        colormap = st.selectbox("Colormap", ["viridis", "plasma", "magma", "coolwarm", "inferno"],
                                key="piano_cmap")
        show_velocity = st.checkbox("Show Velocity", value=True, key="piano_vel")

    with col1:
        try:
            from src.utils.visualization import MusicVisualizer
            viz = MusicVisualizer()
            fig = viz.plot_piano_roll(
                result.pitches, result.durations,
                result.velocities if show_velocity else None,
                title=f"{result.genre} · {result.key} {result.mode} · {result.tempo} BPM",
                colormap=colormap,
            )
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Piano roll unavailable: {e}")
            _fallback_piano_roll(result.pitches)


def _show_pitch_distribution_tab(result):
    st.markdown("### 📊 Pitch Distribution Analysis")
    st.markdown(
        "<p style='color: #64748b;'>Distribution of pitch classes (C, D, E...) in the generated melody "
        "overlaid with the target scale.</p>",
        unsafe_allow_html=True,
    )
    try:
        from src.utils.visualization import MusicVisualizer
        viz = MusicVisualizer()
        fig = viz.plot_pitch_histogram(result.pitches, key=result.key, mode=result.mode)
        st.pyplot(fig, use_container_width=True)

        # Additional: note distribution over full MIDI range
        st.markdown("### Note Range Distribution")
        fig2 = viz.plot_note_distribution(result.pitches)
        st.pyplot(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Pitch distribution unavailable: {e}")
        _fallback_pitch_chart(result.pitches)


def _show_attention_heatmap_tab():
    st.markdown("### 🔥 Attention Heatmap")
    st.markdown(
        "<p style='color: #64748b;'>Cross-attention weights from the Transformer decoder — shows which "
        "source tokens the model attends to when generating each target token.</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "🔮 Attention heatmaps are available when the Transformer model is loaded with a trained checkpoint. "
        "Below is a demo heatmap with synthetic data."
    )

    try:
        import numpy as np
        from src.utils.visualization import MusicVisualizer
        viz = MusicVisualizer()

        # Synthetic attention weights for demo
        q_len, k_len = 16, 16
        rng = np.random.default_rng(42)
        # Make it look like realistic attention (peaked)
        weights = rng.dirichlet([0.5] * k_len, size=(8, q_len))
        weights = weights.reshape(8, q_len, k_len)

        fig = viz.plot_attention_heatmap(
            weights,
            src_tokens=[f"src_{i}" for i in range(k_len)],
            tgt_tokens=[f"tgt_{i}" for i in range(q_len)],
            title="Demo: Cross-Attention Weights (Layer 6, All 8 Heads)",
        )
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Attention heatmap unavailable: {e}")


def _show_training_metrics_tab():
    st.markdown("### 📈 Training Metrics Dashboard")
    st.markdown(
        "<p style='color: #64748b;'>Track model training progress — loss, accuracy, and learning rate curves.</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "📋 Training metrics are populated from checkpoint logs during actual model training. "
        "Below is a simulated example."
    )

    try:
        import numpy as np
        from src.utils.visualization import MusicVisualizer
        viz = MusicVisualizer()

        # Simulated training curves
        epochs = 50
        rng = np.random.default_rng(7)
        train_losses = 5.0 * np.exp(-0.06 * np.arange(epochs)) + rng.normal(0, 0.1, epochs).clip(0)
        val_losses = 5.2 * np.exp(-0.055 * np.arange(epochs)) + rng.normal(0, 0.15, epochs).clip(0)
        train_acc = 1 - np.exp(-0.07 * np.arange(epochs)) + rng.normal(0, 0.01, epochs).clip(0, 1)
        val_acc = 0.97 * (1 - np.exp(-0.065 * np.arange(epochs))) + rng.normal(0, 0.015, epochs).clip(0, 1)

        fig = viz.plot_training_metrics(train_losses, val_losses, train_acc, val_acc)
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Training metrics unavailable: {e}")
        _fallback_training_chart()


def _show_demo_visualizations():
    """Show demo visualizations with random data when no result is available."""
    import numpy as np
    rng = np.random.default_rng(42)

    try:
        from src.theory.music_theory_engine import MusicTheoryEngine
        from src.utils.visualization import MusicVisualizer

        engine = MusicTheoryEngine(key="C", mode="major")
        scale = engine.get_scale_pitches()
        pitches = rng.choice(scale, size=32).tolist()
        durations = rng.choice([0.25, 0.5, 0.5, 1.0], size=32).tolist()
        velocities = rng.integers(50, 90, size=32).tolist()

        viz = MusicVisualizer()
        col1, col2 = st.columns(2)

        with col1:
            fig = viz.plot_piano_roll(pitches, durations, velocities, title="Demo Piano Roll (C Major)")
            st.pyplot(fig, use_container_width=True)

        with col2:
            fig2 = viz.plot_pitch_histogram(pitches, key="C", mode="major")
            st.pyplot(fig2, use_container_width=True)

    except Exception as e:
        st.warning(f"Demo visualization failed: {e}")


def _fallback_piano_roll(pitches):
    """Fallback Plotly-based piano roll."""
    try:
        import plotly.graph_objects as go
        NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        fig = go.Figure()
        for i, p in enumerate(pitches):
            fig.add_trace(go.Bar(x=[i], y=[1], base=[p - 0.4], name=NOTE_NAMES[p % 12],
                                  marker_color="#8b5cf6", showlegend=False, width=0.8))
        fig.update_layout(
            paper_bgcolor="#0a0a0f", plot_bgcolor="#12121a",
            font_color="#94a3b8", height=400,
            xaxis_title="Note Index", yaxis_title="MIDI Pitch",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.code(" ".join([f"MIDI:{p}" for p in pitches[:20]]))


def _fallback_pitch_chart(pitches):
    """Fallback Plotly pitch histogram."""
    try:
        import plotly.graph_objects as go
        NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        counts = [0] * 12
        for p in pitches:
            counts[p % 12] += 1
        fig = go.Figure(go.Bar(
            x=NOTE_NAMES, y=counts,
            marker=dict(color="#8b5cf6", line=dict(color="#6d28d9", width=1)),
        ))
        fig.update_layout(
            paper_bgcolor="#0a0a0f", plot_bgcolor="#12121a",
            font_color="#94a3b8", height=300,
            title="Pitch Class Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _fallback_training_chart():
    """Fallback Plotly training chart."""
    try:
        import numpy as np
        import plotly.graph_objects as go
        epochs = list(range(50))
        losses = [5 * 0.94 ** e + 0.1 for e in epochs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=losses, name="Train Loss", line=dict(color="#8b5cf6")))
        fig.update_layout(paper_bgcolor="#0a0a0f", plot_bgcolor="#12121a", font_color="#94a3b8")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
