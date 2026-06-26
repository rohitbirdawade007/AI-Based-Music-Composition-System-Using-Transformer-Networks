"""Music Retriever Module.

Provides :class:`MusicRetriever`, a nearest-neighbour index for music
embeddings backed by **FAISS** when available or by a pure-NumPy cosine
similarity fallback when it is not.

Architecture
------------
* Each MIDI file is embedded by :class:`~src.rag.embedder.MusicEmbedder`
  and added to an inner-product (cosine) index.
* Metadata (file name, arbitrary user-supplied dict) is stored in a JSON
  sidecar file alongside the FAISS index so that search results carry
  rich context back to the caller.
* The index and sidecar are saved/loaded atomically to/from a directory.

Backend selection
-----------------
* If ``faiss`` (``faiss-cpu`` or ``faiss-gpu``) is importable the
  ``IndexFlatIP`` (exact inner-product / cosine) index is used.
* Otherwise a :class:`_NumpyIndex` that performs brute-force cosine
  similarity over a ``(N, dim)`` float32 array is used transparently.

Typical usage::

    from src.rag.embedder  import MusicEmbedder
    from src.rag.retriever import MusicRetriever

    retriever = MusicRetriever(embedding_dim=128)

    # Index some MIDI files
    for path in midi_paths:
        retriever.add_midi(path.read_bytes(), file_name=path.name)

    retriever.build_index("data/rag_index")

    # At query time
    embedder = MusicEmbedder()
    query_vec = embedder.embed_midi_file("query.mid")
    results   = retriever.search(query_vec, top_k=5)
    for r in results:
        print(r["score"], r["file_name"])
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Tolerate missing sibling packages during isolated testing
try:
    from src.rag.embedder import MusicEmbedder
except ImportError:
    try:
        from .embedder import MusicEmbedder  # type: ignore[no-redef]
    except ImportError:
        MusicEmbedder = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FAISS (optional)
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
    logger.debug("FAISS backend available (version %s)", faiss.__version__)
except ImportError:
    _FAISS_AVAILABLE = False
    logger.info(
        "faiss not installed — using NumPy cosine similarity fallback.  "
        "Install faiss-cpu with:  pip install faiss-cpu"
    )

# File names within an index directory
_INDEX_FILE = "music.index"
_META_FILE = "metadata.json"
_VECTORS_FILE = "vectors.npy"


# ---------------------------------------------------------------------------
# NumPy fallback index
# ---------------------------------------------------------------------------


class _NumpyIndex:
    """Minimal brute-force cosine similarity index using NumPy.

    Vectors are L2-normalised on insertion so that inner-product equals
    cosine similarity.

    Args:
        dim: Embedding dimensionality.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None  # (N, dim) float32

    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Number of vectors in the index."""
        if self._vectors is None:
            return 0
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        """Add L2-normalised copies of *vectors* to the index.

        Args:
            vectors: ``(n, dim)`` float32 array.
        """
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        vectors = vectors / norms
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.concatenate([self._vectors, vectors], axis=0)

    def search(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-*k* most similar vectors.

        Args:
            query: ``(1, dim)`` or ``(dim,)`` float32 array.
            k:     Number of results.

        Returns:
            ``(scores, indices)`` — both ``(1, k)`` float32 / int64 arrays
            matching the FAISS convention.
        """
        if self._vectors is None or self.ntotal == 0:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)

        q = query.reshape(1, -1).astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        sims = (self._vectors @ q.T).flatten()  # (N,)
        k = min(k, len(sims))
        top_idx = np.argsort(sims)[::-1][:k]
        top_scores = sims[top_idx]
        return top_scores.reshape(1, -1), top_idx.reshape(1, -1).astype(np.int64)

    def save(self, path: str | Path) -> None:
        """Persist vectors to a ``.npy`` file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        if self._vectors is not None:
            np.save(str(path), self._vectors)
        else:
            np.save(str(path), np.zeros((0, self.dim), dtype=np.float32))
        logger.debug("NumPy index saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load vectors from a ``.npy`` file.

        Args:
            path: Input file path (must exist).
        """
        path = Path(path)
        self._vectors = np.load(str(path))
        logger.debug("NumPy index loaded from %s (%d vectors)", path, self.ntotal)


# ---------------------------------------------------------------------------
# MusicRetriever
# ---------------------------------------------------------------------------


class MusicRetriever:
    """Nearest-neighbour retriever for MIDI music embeddings.

    Uses FAISS for fast approximate search when available, or falls back to
    exact NumPy cosine similarity otherwise.

    Args:
        embedding_dim: Dimensionality of the embedding vectors.  Must match
                       the output dimension of the :class:`MusicEmbedder`
                       being used.  Defaults to ``128``.
        index_path:    Optional path to an existing saved index directory.
                       If provided the index is loaded immediately.
        embedder:      Optional pre-configured :class:`MusicEmbedder`
                       instance.  A default one is created if not supplied.

    Attributes:
        embedding_dim (int): Embedding dimensionality.
        backend (str):       ``"faiss"`` or ``"numpy"``.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        index_path: Optional[str | Path] = None,
        embedder: Optional["MusicEmbedder"] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self._embedder: Optional["MusicEmbedder"] = embedder
        self._metadata: List[Dict[str, Any]] = []  # parallel to index rows

        # Initialise backend
        if _FAISS_AVAILABLE:
            self.backend = "faiss"
            # Inner-product index (cosine after L2 normalisation)
            self._index = faiss.IndexFlatIP(embedding_dim)  # type: ignore[attr-defined]
        else:
            self.backend = "numpy"
            self._index = _NumpyIndex(embedding_dim)

        logger.info(
            "MusicRetriever initialised (dim=%d, backend=%s)",
            embedding_dim,
            self.backend,
        )

        if index_path is not None:
            self.load_index(index_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return int(self._index.ntotal)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_midi(
        self,
        midi_bytes: bytes,
        file_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Embed a MIDI file and add it to the index.

        Args:
            midi_bytes: Raw bytes of a MIDI file.
            file_name:  Human-readable name (e.g. the original filename).
            metadata:   Optional dict of extra information to store alongside
                        this entry (e.g. ``{"composer": "Bach", "key": "Dm"}``).

        Returns:
            A unique string ID for this entry (UUID4).

        Raises:
            ImportError: If MIDI parsing dependencies are unavailable.
            ValueError:  If *midi_bytes* is not valid MIDI data.
        """
        entry_id = str(uuid.uuid4())
        embedder = self._get_embedder()
        vec = embedder.embed_midi_bytes(midi_bytes).astype(np.float32)  # (128,)
        vec_2d = vec.reshape(1, -1)

        # FAISS IndexFlatIP expects L2-normalised vectors for cosine similarity
        if self.backend == "faiss":
            faiss.normalize_L2(vec_2d)  # type: ignore[attr-defined]
            self._index.add(vec_2d)
        else:
            self._index.add(vec_2d)  # _NumpyIndex normalises internally

        entry: Dict[str, Any] = {
            "id": entry_id,
            "file_name": file_name,
            "index_pos": self.size - 1,
        }
        if metadata:
            entry.update(metadata)
        self._metadata.append(entry)

        logger.debug("Added MIDI '%s' (id=%s) at index pos %d", file_name, entry_id, self.size - 1)
        return entry_id

    def add_embedding(
        self,
        embedding: np.ndarray,
        file_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a pre-computed embedding vector to the index.

        Use this when you have already computed the embedding externally.

        Args:
            embedding: ``(embedding_dim,)`` float array.
            file_name: Human-readable identifier.
            metadata:  Optional extra information dict.

        Returns:
            A unique string ID for this entry.

        Raises:
            ValueError: If *embedding* has the wrong shape.
        """
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {embedding.shape[-1]}"
            )
        entry_id = str(uuid.uuid4())
        vec = embedding.astype(np.float32).reshape(1, -1)

        if self.backend == "faiss":
            faiss.normalize_L2(vec)  # type: ignore[attr-defined]
            self._index.add(vec)
        else:
            self._index.add(vec)

        entry: Dict[str, Any] = {
            "id": entry_id,
            "file_name": file_name,
            "index_pos": self.size - 1,
        }
        if metadata:
            entry.update(metadata)
        self._metadata.append(entry)
        return entry_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find the *top_k* most similar entries to *query_embedding*.

        Args:
            query_embedding: ``(embedding_dim,)`` float array.  Will be
                             L2-normalised internally.
            top_k:           Number of results to return.  If the index
                             contains fewer entries, all are returned.

        Returns:
            List of result dicts, each containing:

            * ``"score"``     – cosine similarity in ``[-1, 1]`` (higher = more similar).
            * ``"rank"``      – 1-based rank.
            * ``"id"``        – unique entry ID.
            * ``"file_name"`` – name supplied at insertion time.
            * All other keys from the *metadata* dict supplied at insertion.

        Raises:
            RuntimeError: If the index is empty.
        """
        if self.size == 0:
            logger.warning("search() called on empty index — returning empty results")
            return []

        top_k = min(top_k, self.size)
        vec = query_embedding.astype(np.float32).reshape(1, -1)

        if self.backend == "faiss":
            faiss.normalize_L2(vec)  # type: ignore[attr-defined]
            scores_2d, indices_2d = self._index.search(vec, top_k)
        else:
            scores_2d, indices_2d = self._index.search(vec, top_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(
            zip(scores_2d[0], indices_2d[0]), start=1
        ):
            idx = int(idx)
            if idx < 0 or idx >= len(self._metadata):
                continue  # FAISS may return –1 for unfilled slots
            entry = dict(self._metadata[idx])
            entry["score"] = float(score)
            entry["rank"] = rank
            results.append(entry)

        logger.debug("search top_%d: %s", top_k, [r["file_name"] for r in results])
        return results

    def search_midi_file(
        self,
        path: str | Path,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Convenience: embed a MIDI file and search the index.

        Args:
            path:  Path to the query MIDI file.
            top_k: Number of results.

        Returns:
            See :meth:`search`.
        """
        embedder = self._get_embedder()
        vec = embedder.embed_midi_file(path)
        return self.search(vec, top_k=top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def build_index(self, directory: str | Path) -> None:
        """Save the index and metadata to *directory*.

        The directory is created if it does not exist.  Two files are written:

        * ``music.index`` (FAISS binary) or ``vectors.npy`` (NumPy fallback).
        * ``metadata.json`` — JSON array of metadata dicts.

        Args:
            directory: Target directory path.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.backend == "faiss":
            index_file = str(directory / _INDEX_FILE)
            faiss.write_index(self._index, index_file)  # type: ignore[attr-defined]
            logger.info("FAISS index saved to %s", index_file)
        else:
            vec_file = directory / _VECTORS_FILE
            self._index.save(vec_file)
            logger.info("NumPy vectors saved to %s", vec_file)

        meta_file = directory / _META_FILE
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, default=str)
        logger.info("Metadata saved to %s (%d entries)", meta_file, len(self._metadata))

    def load_index(self, directory: str | Path) -> None:
        """Load a previously saved index from *directory*.

        Args:
            directory: Directory containing ``music.index`` (or
                       ``vectors.npy``) and ``metadata.json``.

        Raises:
            FileNotFoundError: If expected files are missing.
        """
        directory = Path(directory)
        meta_file = directory / _META_FILE
        if not meta_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {directory}")

        with meta_file.open("r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        if self.backend == "faiss":
            index_file = directory / _INDEX_FILE
            if not index_file.exists():
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")
            self._index = faiss.read_index(str(index_file))  # type: ignore[attr-defined]
            logger.info("FAISS index loaded from %s (%d vectors)", index_file, self.size)
        else:
            vec_file = directory / _VECTORS_FILE
            if not vec_file.exists():
                raise FileNotFoundError(f"NumPy vectors file not found: {vec_file}")
            self._index.load(vec_file)
            logger.info("NumPy index loaded from %s (%d vectors)", vec_file, self.size)

        logger.info("Loaded %d metadata entries from %s", len(self._metadata), meta_file)

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the current index state.

        Returns:
            Dict with keys:

            * ``"backend"``       – ``"faiss"`` or ``"numpy"``.
            * ``"total_entries"`` – number of indexed items.
            * ``"embedding_dim"`` – vector dimensionality.
            * ``"faiss_available"`` – bool.
            * ``"unique_files"``  – number of distinct ``file_name`` values.
        """
        file_names = [m.get("file_name", "") for m in self._metadata]
        return {
            "backend": self.backend,
            "total_entries": self.size,
            "embedding_dim": self.embedding_dim,
            "faiss_available": _FAISS_AVAILABLE,
            "unique_files": len(set(file_names)),
        }

    def remove_all(self) -> None:
        """Clear the index and all stored metadata.

        .. warning::
            This operation is irreversible unless you have previously called
            :meth:`build_index` to persist the index to disk.
        """
        if self.backend == "faiss":
            self._index = faiss.IndexFlatIP(self.embedding_dim)  # type: ignore[attr-defined]
        else:
            self._index = _NumpyIndex(self.embedding_dim)
        self._metadata = []
        logger.info("Index cleared")

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"MusicRetriever(backend={self.backend!r}, "
            f"size={self.size}, dim={self.embedding_dim})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embedder(self) -> "MusicEmbedder":
        """Return the shared :class:`MusicEmbedder`, creating one if needed.

        Returns:
            A :class:`~src.rag.embedder.MusicEmbedder` instance.

        Raises:
            ImportError: If the embedder module could not be imported.
        """
        if self._embedder is None:
            if MusicEmbedder is None:
                raise ImportError(
                    "Could not import MusicEmbedder.  Make sure the src.rag "
                    "package is on the Python path."
                )
            self._embedder = MusicEmbedder(embed_dim=self.embedding_dim)
        return self._embedder
