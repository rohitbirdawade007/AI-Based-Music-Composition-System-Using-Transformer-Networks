"""Data Package.

This package provides tools for parsing, preprocessing, and managing
MIDI music data for the AI Music Composition pipeline.

Modules:
    midi_parser:  Low-level MIDI file parsing via the ``mido`` library.
    preprocessor: Feature-engineering and data-augmentation helpers.
    dataset:      PyTorch Dataset / DataModule wrappers for model training.
"""

from .midi_parser import MidiParser, ParsedNote
from .preprocessor import MusicPreprocessor
from .dataset import MusicDataset

__all__ = ["MidiParser", "ParsedNote", "MusicPreprocessor", "MusicDataset"]
