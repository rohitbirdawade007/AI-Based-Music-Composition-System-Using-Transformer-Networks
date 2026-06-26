"""
Music Visualization Module
===========================
Provides the ``MusicVisualizer`` class with a suite of publication-quality
plots for analysing and presenting AI-generated music.

All plots use a dark "GitHub-inspired" colour scheme:
  - Background  : #0d1117
  - Grid / panel: #21262d
  - Foreground  : #c9d1d9
  - Accent      : #58a6ff  (blue)

Dependencies:
  - matplotlib  (pip install matplotlib)
  - numpy

Usage:
    from src.utils.visualization import MusicVisualizer

    vis = MusicVisualizer()
    fig = vis.plot_piano_roll(pitches, durations, velocities)
    fig.savefig("piano_roll.png", dpi=150)
"""

from __future__ import annotations

import base64
import io
import logging
from typing import List, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# Lazy matplotlib import — avoids issues in headless / test environments
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for servers)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark-theme colour palette
# ---------------------------------------------------------------------------

_DARK = {
    "bg":       "#0d1117",   # figure / axes background
    "panel":    "#21262d",   # secondary backgrounds, grid lines
    "fg":       "#c9d1d9",   # default text, tick labels
    "accent":   "#58a6ff",   # primary accent (blue)
    "accent2":  "#3fb950",   # secondary accent (green)
    "accent3":  "#f78166",   # tertiary accent (red/orange)
    "accent4":  "#d2a8ff",   # quaternary accent (purple)
    "grid":     "#30363d",   # subtle grid lines
    "border":   "#30363d",   # axes spines
}

# MIDI note name lookup
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

# Scale intervals for in-scale colouring
_SCALE_INTERVALS = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "blues":      [0, 3, 5, 6, 7, 10],
    "pentatonic": [0, 2, 4, 7, 9],
}

_NOTE_TO_ROOT = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def _midi_note_name(pitch: int) -> str:
    """Return the note name string for a MIDI pitch (e.g. 60 → 'C4')."""
    octave = (pitch // 12) - 1
    note = _NOTE_NAMES[pitch % 12]
    return f"{note}{octave}"


def _apply_dark_theme(ax: "plt.Axes", fig: "Figure") -> None:
    """Apply the GitHub dark-theme style to *fig* and *ax*."""
    fig.patch.set_facecolor(_DARK["bg"])
    ax.set_facecolor(_DARK["panel"])
    ax.tick_params(colors=_DARK["fg"], labelsize=9)
    ax.xaxis.label.set_color(_DARK["fg"])
    ax.yaxis.label.set_color(_DARK["fg"])
    ax.title.set_color(_DARK["fg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_DARK["border"])
    ax.grid(True, color=_DARK["grid"], linewidth=0.5, alpha=0.7)


def _apply_dark_theme_all_axes(fig: "Figure") -> None:
    """Apply dark theme to every axes object in *fig*."""
    fig.patch.set_facecolor(_DARK["bg"])
    for ax in fig.get_axes():
        _apply_dark_theme(ax, fig)


def _check_matplotlib(method_name: str) -> bool:
    """Log a warning and return False if matplotlib is unavailable."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning(
            "matplotlib is not installed; %s() is unavailable. "
            "Install with: pip install matplotlib",
            method_name,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# MusicVisualizer
# ---------------------------------------------------------------------------

class MusicVisualizer:
    """
    A collection of visualization helpers for music generation analysis.

    All methods return :class:`matplotlib.figure.Figure` objects that can be
    saved with ``fig.savefig(...)``, displayed interactively with
    ``plt.show()``, or converted to base64 for embedding in web pages via
    :meth:`figure_to_base64`.

    Parameters
    ----------
    dpi : int
        Default resolution for all figures (dots per inch).
    """

    def __init__(self, dpi: int = 120) -> None:
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Piano Roll
    # ------------------------------------------------------------------

    def plot_piano_roll(
        self,
        pitches: List[int],
        durations: List[float],
        velocities: Optional[List[int]] = None,
        title: str = "Piano Roll",
        figsize: tuple = (14, 6),
        colormap: str = "viridis",
    ) -> "Figure":
        """
        Render a piano-roll visualisation of a sequence of notes.

        Each note is drawn as a horizontal bar whose:
          - x-position and width correspond to onset time and duration
          - y-position corresponds to pitch
          - colour encodes velocity (if provided) via *colormap*

        Rests (``pitch == -1``) are skipped.

        Parameters
        ----------
        pitches : List[int]
            MIDI pitches (0-127); ``-1`` means rest.
        durations : List[float]
            Duration in seconds for each event.
        velocities : List[int], optional
            MIDI velocities (0-127).  Defaults to a uniform value of 80.
        title : str
            Plot title.
        figsize : tuple
            Matplotlib figure size ``(width, height)`` in inches.
        colormap : str
            Matplotlib colormap name for velocity colouring.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _check_matplotlib("plot_piano_roll"):
            return None  # type: ignore[return-value]

        if velocities is None:
            velocities = [80] * len(pitches)

        # Filter out rests
        notes = [
            (p, d, v) for p, d, v in zip(pitches, durations, velocities)
            if p >= 0
        ]

        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

        if not notes:
            ax.text(0.5, 0.5, "No notes to display",
                    ha="center", va="center", color=_DARK["fg"],
                    transform=ax.transAxes, fontsize=14)
            _apply_dark_theme(ax, fig)
            ax.set_title(title, color=_DARK["fg"], pad=12)
            fig.tight_layout()
            return fig

        cmap = plt.get_cmap(colormap)
        vel_array = np.array([v for _, _, v in notes], dtype=float)
        norm = Normalize(vmin=vel_array.min(), vmax=vel_array.max())

        onset = 0.0
        min_pitch = max(min(p for p, _, _ in notes) - 3, 0)
        max_pitch = min(max(p for p, _, _ in notes) + 3, 127)

        for pitch, duration, velocity in notes:
            colour = cmap(norm(velocity))
            bar = mpatches.FancyBboxPatch(
                xy=(onset, pitch - 0.4),
                width=duration,
                height=0.8,
                boxstyle="round,pad=0.02",
                linewidth=0.4,
                edgecolor=_DARK["bg"],
                facecolor=colour,
                alpha=0.92,
            )
            ax.add_patch(bar)
            onset += duration

        total_time = sum(d for _, d, _ in notes)

        # Y-axis: MIDI pitch labels (only every C note for clarity)
        c_pitches = [p for p in range(min_pitch, max_pitch + 1) if p % 12 == 0]
        ax.set_yticks(c_pitches)
        ax.set_yticklabels([_midi_note_name(p) for p in c_pitches],
                           color=_DARK["fg"], fontsize=8)
        ax.set_ylim(min_pitch - 0.5, max_pitch + 0.5)
        ax.set_xlim(0, total_time * 1.01)

        ax.set_xlabel("Time (seconds)", color=_DARK["fg"])
        ax.set_ylabel("Pitch", color=_DARK["fg"])
        ax.set_title(title, color=_DARK["fg"], pad=12, fontsize=13, fontweight="bold")

        # Colourbar for velocity
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
        cbar.set_label("Velocity", color=_DARK["fg"], fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=_DARK["fg"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_DARK["fg"])
        cbar.outline.set_edgecolor(_DARK["border"])

        _apply_dark_theme(ax, fig)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Attention Heatmap
    # ------------------------------------------------------------------

    def plot_attention_heatmap(
        self,
        attention_weights: "np.ndarray",
        src_tokens: Optional[List[str]] = None,
        tgt_tokens: Optional[List[str]] = None,
        title: str = "Attention Weights",
    ) -> "Figure":
        """
        Visualise multi-head attention weights as a grid of heatmaps.

        Parameters
        ----------
        attention_weights : numpy.ndarray
            Shape ``(num_heads, q_len, k_len)`` — attention weight matrix.
        src_tokens : List[str], optional
            Labels for the key/source axis (x-axis of each head).
        tgt_tokens : List[str], optional
            Labels for the query/target axis (y-axis of each head).
        title : str
            Overall figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _check_matplotlib("plot_attention_heatmap"):
            return None  # type: ignore[return-value]

        weights = np.asarray(attention_weights)

        if weights.ndim == 2:
            # Single-head: add a head dimension
            weights = weights[np.newaxis, ...]

        num_heads, q_len, k_len = weights.shape

        # Lay out heads in a grid
        ncols = min(4, num_heads)
        nrows = math.ceil(num_heads / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            dpi=self.dpi,
            squeeze=False,
        )

        for head_idx in range(num_heads):
            row, col = divmod(head_idx, ncols)
            ax = axes[row][col]

            im = ax.imshow(
                weights[head_idx],
                cmap="Blues",
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=weights[head_idx].max() or 1.0,
            )

            ax.set_title(f"Head {head_idx + 1}", color=_DARK["fg"],
                         fontsize=9, pad=6)

            # Source (key) labels on x-axis
            if src_tokens and len(src_tokens) == k_len:
                ax.set_xticks(range(k_len))
                ax.set_xticklabels(src_tokens, rotation=45, ha="right",
                                   fontsize=7, color=_DARK["fg"])
            else:
                ax.set_xticks(range(0, k_len, max(1, k_len // 10)))
                ax.tick_params(axis="x", colors=_DARK["fg"])

            # Target (query) labels on y-axis
            if tgt_tokens and len(tgt_tokens) == q_len:
                ax.set_yticks(range(q_len))
                ax.set_yticklabels(tgt_tokens, fontsize=7, color=_DARK["fg"])
            else:
                ax.set_yticks(range(0, q_len, max(1, q_len // 10)))
                ax.tick_params(axis="y", colors=_DARK["fg"])

            for spine in ax.spines.values():
                spine.set_edgecolor(_DARK["border"])

            ax.set_facecolor(_DARK["panel"])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused axes
        for idx in range(num_heads, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.patch.set_facecolor(_DARK["bg"])
        fig.suptitle(title, color=_DARK["fg"], fontsize=14,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Pitch Histogram
    # ------------------------------------------------------------------

    def plot_pitch_histogram(
        self,
        pitches: List[int],
        key: str = "C",
        mode: str = "major",
    ) -> "Figure":
        """
        Plot a histogram of pitch-class usage, highlighting in-scale notes.

        Parameters
        ----------
        pitches : List[int]
            MIDI pitch values; rests (``-1``) are excluded.
        key : str
            Root note of the key for in-scale colouring.
        mode : str
            Scale mode.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _check_matplotlib("plot_pitch_histogram"):
            return None  # type: ignore[return-value]

        valid_pitches = [p for p in pitches if 0 <= p <= 127]
        if not valid_pitches:
            logger.warning("plot_pitch_histogram: no valid pitches provided.")
            return _empty_figure("No pitch data", self.dpi)

        root = _NOTE_TO_ROOT.get(key, 0)
        scale_set = set(_SCALE_INTERVALS.get(mode, _SCALE_INTERVALS["major"]))

        pitch_classes = [p % 12 for p in valid_pitches]
        counts = np.bincount(pitch_classes, minlength=12)

        colours = []
        for pc in range(12):
            offset = (pc - root) % 12
            if offset == 0:
                colours.append(_DARK["accent3"])   # root — special highlight
            elif offset in scale_set:
                colours.append(_DARK["accent"])    # in scale
            else:
                colours.append(_DARK["panel"])     # out of scale

        fig, ax = plt.subplots(figsize=(10, 5), dpi=self.dpi)

        bars = ax.bar(
            range(12), counts, color=colours,
            edgecolor=_DARK["bg"], linewidth=0.6, width=0.85,
        )

        # Value labels on top of each bar
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    str(count),
                    ha="center", va="bottom",
                    fontsize=8, color=_DARK["fg"],
                )

        ax.set_xticks(range(12))
        ax.set_xticklabels(_NOTE_NAMES, color=_DARK["fg"], fontsize=9)
        ax.set_ylabel("Count", color=_DARK["fg"])
        ax.set_xlabel("Pitch Class", color=_DARK["fg"])
        ax.set_title(
            f"Pitch-Class Histogram  |  Key: {key} {mode}",
            color=_DARK["fg"], pad=12, fontsize=13, fontweight="bold",
        )

        # Legend
        legend_patches = [
            mpatches.Patch(color=_DARK["accent3"], label=f"Root ({key})"),
            mpatches.Patch(color=_DARK["accent"],  label="In scale"),
            mpatches.Patch(color=_DARK["panel"],   label="Out of scale"),
        ]
        ax.legend(handles=legend_patches, facecolor=_DARK["bg"],
                  edgecolor=_DARK["border"], labelcolor=_DARK["fg"],
                  fontsize=9, loc="upper right")

        _apply_dark_theme(ax, fig)
        ax.set_facecolor(_DARK["bg"])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Training Metrics
    # ------------------------------------------------------------------

    def plot_training_metrics(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: Optional[List[float]] = None,
        val_accuracies: Optional[List[float]] = None,
    ) -> "Figure":
        """
        Plot training and validation loss curves (and optionally accuracy).

        Parameters
        ----------
        train_losses : List[float]
            Per-epoch training loss values.
        val_losses : List[float]
            Per-epoch validation loss values.
        train_accuracies : List[float], optional
            Per-epoch training accuracy (0-1 or 0-100).
        val_accuracies : List[float], optional
            Per-epoch validation accuracy.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _check_matplotlib("plot_training_metrics"):
            return None  # type: ignore[return-value]

        has_acc = train_accuracies is not None and val_accuracies is not None
        nrows = 2 if has_acc else 1
        fig, axes = plt.subplots(nrows, 1,
                                 figsize=(10, 5 * nrows), dpi=self.dpi,
                                 sharex=True)

        if nrows == 1:
            axes = [axes]

        epochs = range(1, len(train_losses) + 1)

        # ---- Loss ----
        ax_loss = axes[0]
        ax_loss.plot(epochs, train_losses, color=_DARK["accent"],
                     linewidth=2, label="Train Loss", marker="o",
                     markersize=4, alpha=0.9)
        ax_loss.plot(epochs, val_losses, color=_DARK["accent3"],
                     linewidth=2, label="Val Loss", marker="s",
                     markersize=4, linestyle="--", alpha=0.9)

        # Shade the gap between train and val loss
        ax_loss.fill_between(epochs, train_losses, val_losses,
                             alpha=0.08, color=_DARK["accent4"])

        # Mark the best validation epoch
        if val_losses:
            best_epoch = int(np.argmin(val_losses)) + 1
            best_loss = min(val_losses)
            ax_loss.axvline(best_epoch, color=_DARK["accent2"],
                            linewidth=1.2, linestyle=":", alpha=0.8,
                            label=f"Best val (ep {best_epoch})")
            ax_loss.scatter([best_epoch], [best_loss],
                            color=_DARK["accent2"], s=60, zorder=5)

        ax_loss.set_ylabel("Loss", color=_DARK["fg"])
        ax_loss.set_title("Training Metrics", color=_DARK["fg"],
                          pad=12, fontsize=13, fontweight="bold")
        ax_loss.legend(facecolor=_DARK["bg"], edgecolor=_DARK["border"],
                       labelcolor=_DARK["fg"], fontsize=9)
        _apply_dark_theme(ax_loss, fig)

        # ---- Accuracy (optional) ----
        if has_acc:
            ax_acc = axes[1]
            acc_epochs = range(1, len(train_accuracies) + 1)
            ax_acc.plot(acc_epochs, train_accuracies, color=_DARK["accent"],
                        linewidth=2, label="Train Acc", marker="o",
                        markersize=4, alpha=0.9)
            ax_acc.plot(acc_epochs, val_accuracies, color=_DARK["accent3"],
                        linewidth=2, label="Val Acc", marker="s",
                        markersize=4, linestyle="--", alpha=0.9)
            ax_acc.set_ylabel("Accuracy", color=_DARK["fg"])
            ax_acc.set_xlabel("Epoch", color=_DARK["fg"])
            ax_acc.set_ylim(0, max(max(train_accuracies), max(val_accuracies)) * 1.05)
            ax_acc.legend(facecolor=_DARK["bg"], edgecolor=_DARK["border"],
                          labelcolor=_DARK["fg"], fontsize=9)
            _apply_dark_theme(ax_acc, fig)
        else:
            axes[0].set_xlabel("Epoch", color=_DARK["fg"])

        fig.tight_layout(h_pad=0.5)
        return fig

    # ------------------------------------------------------------------
    # Note Distribution
    # ------------------------------------------------------------------

    def plot_note_distribution(
        self,
        pitches: List[int],
    ) -> "Figure":
        """
        Plot the full MIDI pitch distribution as a bar chart, annotated with
        octave markers and note names on prominent pitches.

        Parameters
        ----------
        pitches : List[int]
            MIDI pitch values; rests (``-1``) are excluded.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _check_matplotlib("plot_note_distribution"):
            return None  # type: ignore[return-value]

        valid = [p for p in pitches if 0 <= p <= 127]
        if not valid:
            return _empty_figure("No note data", self.dpi)

        counts = np.bincount(valid, minlength=128)
        min_p, max_p = min(valid), max(valid)
        pitch_range = range(min_p, max_p + 1)

        # Colour each bar by octave
        octave_colours = [
            "#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
            "#ffa657", "#79c0ff", "#56d364", "#ff7b72",
        ]
        bar_colours = [octave_colours[(p // 12) % len(octave_colours)]
                       for p in pitch_range]

        fig, ax = plt.subplots(figsize=(14, 5), dpi=self.dpi)
        ax.bar(list(pitch_range), [counts[p] for p in pitch_range],
               color=bar_colours, edgecolor=_DARK["bg"], linewidth=0.3)

        # Octave boundary lines
        for octave in range(11):
            boundary = octave * 12
            if min_p <= boundary <= max_p:
                ax.axvline(boundary - 0.5, color=_DARK["fg"],
                           linewidth=0.4, alpha=0.3)
                ax.text(boundary, ax.get_ylim()[1] * 0.98,
                        f"C{octave - 1}", ha="center", va="top",
                        fontsize=7, color=_DARK["fg"], alpha=0.6)

        # Annotate the top-5 most frequent pitches
        top_indices = np.argsort(counts[min_p:max_p + 1])[::-1][:5]
        for idx in top_indices:
            pitch = min_p + idx
            cnt = counts[pitch]
            if cnt > 0:
                ax.text(pitch, cnt + max(counts) * 0.005,
                        _midi_note_name(pitch),
                        ha="center", va="bottom", fontsize=7.5,
                        color=_DARK["fg"], fontweight="bold")

        ax.set_xlabel("MIDI Pitch", color=_DARK["fg"])
        ax.set_ylabel("Frequency", color=_DARK["fg"])
        ax.set_title("Note Distribution (MIDI Pitch)",
                     color=_DARK["fg"], pad=12, fontsize=13, fontweight="bold")
        ax.set_xlim(min_p - 1, max_p + 1)
        _apply_dark_theme(ax, fig)
        ax.set_facecolor(_DARK["bg"])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def figure_to_base64(self, fig: "Figure", fmt: str = "png") -> str:
        """
        Convert a matplotlib Figure to a base64-encoded string.

        Useful for embedding plots directly in HTML without writing to disk.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to encode.
        fmt : str
            Image format (``'png'``, ``'svg'``, ``'jpeg'``).

        Returns
        -------
        str
            Data URI string: ``data:image/<fmt>;base64,<data>``.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/{fmt};base64,{encoded}"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _empty_figure(message: str, dpi: int) -> "Figure":
    """Return a styled empty figure with a centred *message*."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color=_DARK["fg"], fontsize=14, transform=ax.transAxes)
    ax.axis("off")
    _apply_dark_theme(ax, fig)
    fig.tight_layout()
    return fig


# Re-export for import convenience
import math  # noqa: E402 (needed inside plot_attention_heatmap)
