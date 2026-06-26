"""Utils Package.

General-purpose utilities shared across the AI Music Composition project.

Modules:
    visualization: Piano-roll and other plot helpers (matplotlib / pretty_midi).
    audio:         Audio rendering, playback, and format-conversion helpers.
"""

from .visualization import MusicVisualizer
from .audio import AudioUtils

__all__ = ["MusicVisualizer", "AudioUtils"]
