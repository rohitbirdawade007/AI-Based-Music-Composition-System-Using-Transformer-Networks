"""RAG (Retrieval-Augmented Generation) Package.

This package implements a music-specific retrieval pipeline that
embeds MIDI files into a compact feature space and retrieves the most
similar pieces at inference time to guide generation.

Modules:
    embedder:  Pure feature-engineering embedder (no GPU required).
    retriever: FAISS-backed (or NumPy fallback) nearest-neighbour index.
"""

from .embedder import MusicEmbedder
from .retriever import MusicRetriever

__all__ = ["MusicEmbedder", "MusicRetriever"]
