"""Inference Package.

This package exposes the high-level music generation API used at
serving time.  It combines the trained model with the RAG retriever
so that callers only need to interact with a single :class:`MusicGenerator`
object.

Modules:
    generator: End-to-end generation pipeline (model + RAG + post-processing).
"""

from .generator import MusicGenerator

__all__ = ["MusicGenerator"]
