"""Music Embedder Module.

Provides :class:`MusicEmbedder`, a *pure feature-engineering* embedder that
converts a sequence of :class:`~src.data.midi_parser.ParsedNote` objects into
a fixed-size 128-dimensional embedding vector.

No neural network or GPU is required — the features are hand-crafted
statistics derived directly from the note events.  The 48-dimensional
intermediate feature vector is projected to 128 dimensions using a random but
fixed orthogonal linear projection, making the embeddings suitable for
nearest-neighbour search.

Feature breakdown (48 dims total before projection)
----------------------------------------------------
* **Pitch histogram**   (12 dims) – normalised count per pitch class (C … B).
* **Rhythm pattern**    (16 dims) – normalised note-onset counts per 16th-note
  slot within a bar (assuming 4/4 time).
* **Velocity profile**  (8 dims)  – histogram of velocities in 8 equal-width
  bins over ``[0, 128)``.
* **Interval histogram** (12 dims) – normalised count of melodic intervals
  (semitone differences between consecutive pitches), clamped to ±6 semitones.

After extraction the 48-d vector is L2-normalised and then multiplied by a
fixed ``(48, 128)`` projection matrix (seeded from ``numpy`` with seed 0) to
produce the final 128-dimensional embedding.

Typical usage::

    from src.rag.embedder import MusicEmbedder

    embedder = MusicEmbedder()
    features = embedder.embed_midi_file("bach_invention.mid")
    print(features.shape)   # (128,)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Tolerate missing sibling packages during isolated testing
try:
    from src.data.midi_parser import MidiParser, ParsedNote
except ImportError:
    try:
        from ..data.midi_parser import MidiParser, ParsedNote  # type: ignore[no-redef]
    except ImportError:
        MidiParser = None  # type: ignore[assignment,misc]
        ParsedNote = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PITCH_HIST_DIM: int = 12   # one bin per pitch class
_RHYTHM_DIM: int = 16       # 16 sixteenth-note slots per bar (4/4)
_VELOCITY_BINS: int = 8     # velocity histogram bins
_INTERVAL_DIM: int = 12     # intervals –6 … +6 semitones (+ 0 = 12 buckets? no → see below)
_RAW_DIM: int = _PITCH_HIST_DIM + _RHYTHM_DIM + _VELOCITY_BINS + _INTERVAL_DIM  # 48
_EMBED_DIM: int = 128       # final embedding dimension


def _build_projection_matrix(in_dim: int, out_dim: int, seed: int = 0) -> np.ndarray:
    """Build a fixed random projection matrix using QR decomposition.

    The columns are orthonormal when ``in_dim <= out_dim``; otherwise the
    rows are orthonormal.  Using a fixed seed ensures reproducibility.

    Args:
        in_dim:  Input dimensionality.
        out_dim: Output dimensionality.
        seed:    Random seed.  Defaults to ``0``.

    Returns:
        ``(in_dim, out_dim)`` float64 matrix.
    """
    rng = np.random.default_rng(seed)
    # Random Gaussian matrix
    G = rng.standard_normal((in_dim, out_dim))
    # QR decomposition for approximate orthogonality
    Q, _ = np.linalg.qr(G if in_dim >= out_dim else G.T)
    if in_dim < out_dim:
        Q = Q.T
    # Q shape: (in_dim, out_dim)  when in_dim >= out_dim
    #          (out_dim, in_dim)  → transposed to (in_dim, out_dim) above
    return Q.astype(np.float64)


# Module-level fixed projection matrix (avoids recomputation)
_PROJECTION: np.ndarray = _build_projection_matrix(_RAW_DIM, _EMBED_DIM)


# ---------------------------------------------------------------------------
# MusicEmbedder
# ---------------------------------------------------------------------------


class MusicEmbedder:
    """Convert MIDI note sequences into fixed-size 128-dimensional embeddings.

    The embedder is stateless after construction: the same projection matrix
    is used for every call.  It is therefore safe to share a single instance
    across threads / processes.

    Args:
        embed_dim:    Output embedding dimension.  Defaults to ``128``.
        bar_duration: Assumed bar duration in **seconds** for computing the
                      rhythm pattern feature.  Defaults to ``2.0`` (= 120 BPM,
                      4/4).  This is used only when MIDI tempo information is
                      unavailable.
        proj_seed:    Seed for the random projection matrix.  Change this only
                      if you need a different projection.

    Examples::

        embedder = MusicEmbedder()
        vec = embedder.embed_midi_file("moonlight_sonata.mid")
        print(vec.shape)   # (128,)
        print(np.linalg.norm(vec))  # ≈ 1.0
    """

    def __init__(
        self,
        embed_dim: int = _EMBED_DIM,
        bar_duration: float = 2.0,
        proj_seed: int = 0,
    ) -> None:
        self.embed_dim = embed_dim
        self.bar_duration = bar_duration
        # Rebuild projection only if non-default parameters are given
        if embed_dim != _EMBED_DIM or proj_seed != 0:
            self._projection = _build_projection_matrix(_RAW_DIM, embed_dim, proj_seed)
        else:
            self._projection = _PROJECTION

        logger.debug(
            "MusicEmbedder ready: raw_dim=%d → embed_dim=%d", _RAW_DIM, embed_dim
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, notes: Sequence[ParsedNote]) -> np.ndarray:  # type: ignore[override]
        """Extract a 48-dimensional hand-crafted feature vector from *notes*.

        The vector is the concatenation of:

        1. **Pitch histogram** (12 d) — fraction of notes per pitch class.
        2. **Rhythm pattern** (16 d) — fraction of note onsets per 16th-note
           slot within a 4/4 bar.
        3. **Velocity profile** (8 d) — fraction of notes per velocity bucket.
        4. **Interval histogram** (12 d) — fraction of melodic intervals per
           semitone offset bucket (–6 to +5, wrapping extremes).

        Args:
            notes: Sequence of :class:`~src.data.midi_parser.ParsedNote`
                   objects.  If empty, returns a zero vector.

        Returns:
            ``(48,)`` float64 NumPy array (L2-normalised within each
            sub-feature block).
        """
        if not notes:
            logger.debug("extract_features called with empty note list")
            return np.zeros(_RAW_DIM, dtype=np.float64)

        pitches = np.array([n.pitch for n in notes], dtype=np.int64)
        starts = np.array([n.start_time for n in notes], dtype=np.float64)
        velocities = np.array([n.velocity for n in notes], dtype=np.float64)

        pitch_hist = self._pitch_histogram(pitches)         # (12,)
        rhythm = self._rhythm_pattern(starts)               # (16,)
        vel_prof = self._velocity_profile(velocities)       # (8,)
        interval_hist = self._interval_histogram(pitches)   # (12,)

        features = np.concatenate([pitch_hist, rhythm, vel_prof, interval_hist])
        return features  # (48,)

    def embed(self, notes: Sequence[ParsedNote]) -> np.ndarray:  # type: ignore[override]
        """Embed a note sequence into a 128-dimensional L2-normalised vector.

        This calls :meth:`extract_features` and then applies the fixed linear
        projection followed by L2 normalisation.

        Args:
            notes: Sequence of :class:`~src.data.midi_parser.ParsedNote`
                   objects.

        Returns:
            ``(embed_dim,)`` float64 NumPy array, L2-normalised.
        """
        raw = self.extract_features(notes)           # (48,)
        raw_norm = self._l2_normalize(raw)
        projected = raw_norm @ self._projection      # (128,)
        return self._l2_normalize(projected)

    def embed_midi_file(self, path: str | Path) -> np.ndarray:
        """Parse a MIDI file and embed it.

        Args:
            path: Filesystem path to a MIDI file.

        Returns:
            ``(embed_dim,)`` float64 embedding vector.

        Raises:
            ImportError:      If ``mido`` is not installed.
            FileNotFoundError: If *path* does not exist.
        """
        parser = self._get_parser()
        notes = parser.parse_file(path)
        return self.embed(notes)

    def embed_midi_bytes(self, midi_bytes: bytes) -> np.ndarray:
        """Parse MIDI bytes and embed them.

        Args:
            midi_bytes: Raw bytes of a MIDI file.

        Returns:
            ``(embed_dim,)`` float64 embedding vector.

        Raises:
            ImportError: If ``mido`` is not installed.
            ValueError:  If *midi_bytes* is not valid MIDI data.
        """
        parser = self._get_parser()
        notes = parser.parse_from_bytes(midi_bytes)
        return self.embed(notes)

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def _pitch_histogram(self, pitches: np.ndarray) -> np.ndarray:
        """Compute a normalised 12-bin pitch-class histogram.

        Args:
            pitches: 1-D integer array of MIDI pitch numbers.

        Returns:
            ``(12,)`` float64 array summing to 1 (or all zeros if empty).
        """
        hist = np.zeros(12, dtype=np.float64)
        for p in pitches:
            hist[int(p) % 12] += 1.0
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _rhythm_pattern(self, starts: np.ndarray) -> np.ndarray:
        """Compute a 16-bin onset histogram over one 4/4 bar.

        Each note's start time is mapped to one of 16 equally-spaced slots
        within a bar of duration :attr:`bar_duration`.  The histogram is
        normalised so that it sums to 1.

        Args:
            starts: 1-D float array of note start times in seconds.

        Returns:
            ``(16,)`` float64 array.
        """
        hist = np.zeros(_RHYTHM_DIM, dtype=np.float64)
        for s in starts:
            pos = (s % self.bar_duration) / self.bar_duration  # [0, 1)
            slot = int(pos * _RHYTHM_DIM) % _RHYTHM_DIM
            hist[slot] += 1.0
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _velocity_profile(self, velocities: np.ndarray) -> np.ndarray:
        """Compute a normalised 8-bin velocity histogram over ``[0, 128)``.

        Args:
            velocities: 1-D float array of MIDI velocity values.

        Returns:
            ``(8,)`` float64 array.
        """
        bins = np.linspace(0, 128, _VELOCITY_BINS + 1)
        hist, _ = np.histogram(velocities, bins=bins)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _interval_histogram(self, pitches: np.ndarray) -> np.ndarray:
        """Compute a normalised 12-bin melodic interval histogram.

        Intervals are the semitone differences between consecutive pitches.
        Values outside ``[–6, +5]`` are clamped to the nearest bin.

        Bin layout::

            index 0  →  interval –6 semitones (or lower)
            index 1  →  interval –5
            ...
            index 6  →  interval  0  (repeated pitch)
            ...
            index 11 →  interval +5  (or higher)

        Args:
            pitches: 1-D integer array of MIDI pitches (ordered by time).

        Returns:
            ``(12,)`` float64 array.
        """
        hist = np.zeros(_INTERVAL_DIM, dtype=np.float64)
        if len(pitches) < 2:
            return hist
        intervals = np.diff(pitches.astype(np.int64))  # (N-1,)
        # Map interval to bin: offset by 6 so that –6 → 0, 0 → 6, +5 → 11
        for iv in intervals:
            bin_idx = int(np.clip(iv + 6, 0, _INTERVAL_DIM - 1))
            hist[bin_idx] += 1.0
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """Return the L2-normalised version of *vec*.

        If the norm is effectively zero (< 1e-12) the original vector is
        returned unchanged to avoid NaN values.

        Args:
            vec: 1-D float array.

        Returns:
            L2-normalised 1-D float array.
        """
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return vec
        return vec / norm

    @staticmethod
    def _get_parser() -> "MidiParser":
        """Return a :class:`MidiParser` instance, raising if unavailable.

        Returns:
            A :class:`~src.data.midi_parser.MidiParser` instance.

        Raises:
            ImportError: If the ``midi_parser`` module could not be imported.
        """
        if MidiParser is None:
            raise ImportError(
                "Could not import MidiParser.  Make sure the src.data package "
                "is on the Python path."
            )
        return MidiParser()
