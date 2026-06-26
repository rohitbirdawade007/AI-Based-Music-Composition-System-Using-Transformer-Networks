"""Music Preprocessor Module.

Provides :class:`MusicPreprocessor`, a pipeline that converts raw
:class:`~src.data.midi_parser.ParsedNote` sequences into numpy arrays suitable
for training deep-learning models.

Key capabilities:

* **Sequence windowing** – sliding windows of fixed length with configurable
  stride for auto-regressive training targets.
* **Velocity normalisation** – scale MIDI velocities to ``[0, 1]``.
* **Pitch transposition** – data augmentation by chromatic shifting.
* **Time stretching** – proportional scaling of all note timestamps.
* **Vocabulary statistics** – pitch-frequency histograms for tokenisation.
* **Serialisation** – save / load processed sequences to / from ``.npz`` files.

All public methods are designed to be composable and side-effect-free (they
return new objects rather than mutating their inputs).

Typical usage::

    from src.data.midi_parser import MidiParser
    from src.data.preprocessor import MusicPreprocessor

    parser = MidiParser()
    notes  = parser.parse_file("song.mid")

    pre = MusicPreprocessor(seq_len=64, stride=32)
    seqs = pre.create_sequences(notes)
    seqs = pre.normalize_velocity(seqs)
    pre.save_processed(seqs, "data/processed/song.npz")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Local import – tolerate missing package during isolated testing
try:
    from .midi_parser import ParsedNote
except ImportError:  # pragma: no cover
    from src.data.midi_parser import ParsedNote  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

NoteArray = np.ndarray  # shape (N, 5) – [pitch, start, end, velocity, channel]


# ---------------------------------------------------------------------------
# MusicPreprocessor
# ---------------------------------------------------------------------------


class MusicPreprocessor:
    """Preprocessing pipeline for MIDI note sequences.

    Args:
        seq_len: Number of notes per training window.  Defaults to ``64``.
        stride:  Step size between consecutive windows.  Defaults to ``seq_len``
                 (non-overlapping).  Set to a smaller value for dense sampling.
        pitch_range: ``(min_pitch, max_pitch)`` tuple used to clip pitches
                     during normalisation.  Defaults to ``(21, 108)`` —
                     standard 88-key piano range.
        velocity_range: ``(min_vel, max_vel)`` used for velocity scaling.
                        Defaults to ``(1, 127)``.

    Attributes:
        seq_len (int): Window length in notes.
        stride  (int): Stride between windows in notes.
    """

    # Default piano range (A0–C8)
    _PITCH_MIN: int = 21
    _PITCH_MAX: int = 108

    def __init__(
        self,
        seq_len: int = 64,
        stride: Optional[int] = None,
        pitch_range: Tuple[int, int] = (21, 108),
        velocity_range: Tuple[int, int] = (1, 127),
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be ≥ 1, got {seq_len}")
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        if self.stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {self.stride}")
        self.pitch_range = pitch_range
        self.velocity_range = velocity_range

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        notes: Sequence[ParsedNote],
        seq_len: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> List[NoteArray]:
        """Split a note sequence into overlapping fixed-length windows.

        Each window is represented as a ``(seq_len, 5)`` float32 NumPy array
        with columns ``[pitch, start_time, end_time, velocity, channel]``.

        Args:
            notes:   Ordered sequence of :class:`ParsedNote` objects.
            seq_len: Override instance ``seq_len`` for this call.
            stride:  Override instance ``stride`` for this call.

        Returns:
            List of ``(seq_len, 5)`` float32 arrays.  May be empty if
            ``len(notes) < seq_len``.

        Examples::

            seqs = preprocessor.create_sequences(notes, seq_len=32, stride=16)
        """
        sl = seq_len if seq_len is not None else self.seq_len
        st = stride if stride is not None else self.stride

        note_array = self._notes_to_array(notes)  # (N, 5)
        n = len(note_array)

        if n < sl:
            logger.warning(
                "Sequence length %d is shorter than window size %d – returning empty list",
                n,
                sl,
            )
            return []

        sequences: List[NoteArray] = []
        for start in range(0, n - sl + 1, st):
            window = note_array[start : start + sl].copy()
            sequences.append(window)

        logger.debug(
            "Created %d windows (sl=%d, stride=%d) from %d notes",
            len(sequences),
            sl,
            st,
            n,
        )
        return sequences

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_velocity(
        self,
        sequences: List[NoteArray],
    ) -> List[NoteArray]:
        """Normalise the velocity column of each sequence to ``[0, 1]``.

        Velocities are clipped to :attr:`velocity_range` before scaling so
        that extreme outliers do not compress the dynamic range.

        Args:
            sequences: List of ``(seq_len, 5)`` float32 arrays as returned
                       by :meth:`create_sequences`.

        Returns:
            New list of arrays with the velocity column (index 3) scaled to
            the unit interval.
        """
        lo, hi = self.velocity_range
        span = float(hi - lo) if hi > lo else 1.0
        result: List[NoteArray] = []
        for seq in sequences:
            s = seq.copy()
            s[:, 3] = np.clip(s[:, 3], lo, hi)
            s[:, 3] = (s[:, 3] - lo) / span
            result.append(s)
        return result

    def denormalize_velocity(
        self,
        sequences: List[NoteArray],
    ) -> List[NoteArray]:
        """Inverse of :meth:`normalize_velocity`.

        Args:
            sequences: List of arrays whose velocity column is in ``[0, 1]``.

        Returns:
            List of arrays with velocity restored to MIDI scale.
        """
        lo, hi = self.velocity_range
        span = float(hi - lo) if hi > lo else 1.0
        result: List[NoteArray] = []
        for seq in sequences:
            s = seq.copy()
            s[:, 3] = np.clip(s[:, 3], 0.0, 1.0) * span + lo
            result.append(s)
        return result

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------

    def transpose_augmentation(
        self,
        notes: Sequence[ParsedNote],
        semitones: int,
    ) -> List[ParsedNote]:
        """Transpose every note by *semitones* semitones.

        Pitches are clamped to the valid MIDI range ``[0, 127]``.

        Args:
            notes:    Source note sequence.
            semitones: Chromatic shift (may be negative).

        Returns:
            New list of :class:`ParsedNote` objects with adjusted pitches.
        """
        return [n.transpose(semitones) for n in notes]

    def augment_multi_transpose(
        self,
        notes: Sequence[ParsedNote],
        shifts: Sequence[int],
    ) -> List[List[ParsedNote]]:
        """Apply :meth:`transpose_augmentation` for each value in *shifts*.

        Useful for generating multiple augmented copies of a single piece.

        Args:
            notes:  Source note sequence.
            shifts: Iterable of semitone offsets, e.g. ``[-2, -1, 1, 2]``.

        Returns:
            List of transposed note lists, one per shift value.
        """
        return [self.transpose_augmentation(notes, s) for s in shifts]

    def time_stretch(
        self,
        notes: Sequence[ParsedNote],
        factor: float,
    ) -> List[ParsedNote]:
        """Scale all note timestamps by *factor*.

        Args:
            notes:  Source note sequence.
            factor: Positive float.  ``> 1`` → slower; ``< 1`` → faster.

        Returns:
            New list of :class:`ParsedNote` objects with scaled timestamps.

        Raises:
            ValueError: If *factor* is not positive.
        """
        if factor <= 0:
            raise ValueError(f"time_stretch factor must be positive, got {factor!r}")
        return [n.time_stretch(factor) for n in notes]

    # ------------------------------------------------------------------
    # Vocabulary / statistics
    # ------------------------------------------------------------------

    def build_vocab_stats(
        self,
        all_notes: Sequence[Sequence[ParsedNote]],
    ) -> Dict[str, Any]:
        """Compute pitch-frequency and other statistics across a corpus.

        Args:
            all_notes: An iterable of note sequences (one per piece or
                       per track).

        Returns:
            Dictionary with the following keys:

            * ``pitch_counts``  – ``{pitch: count}`` mapping.
            * ``pitch_freq``    – ``{pitch: relative_frequency}``.
            * ``total_notes``   – total note count across all sequences.
            * ``unique_pitches``– number of distinct pitches observed.
            * ``mean_velocity`` – mean velocity across all notes.
            * ``mean_duration`` – mean note duration (seconds).
        """
        pitch_counts: Dict[int, int] = {}
        velocities: List[float] = []
        durations: List[float] = []

        for seq in all_notes:
            for note in seq:
                pitch_counts[note.pitch] = pitch_counts.get(note.pitch, 0) + 1
                velocities.append(float(note.velocity))
                durations.append(note.duration)

        total = sum(pitch_counts.values())
        pitch_freq = (
            {p: c / total for p, c in pitch_counts.items()} if total > 0 else {}
        )

        stats: Dict[str, Any] = {
            "pitch_counts": pitch_counts,
            "pitch_freq": pitch_freq,
            "total_notes": total,
            "unique_pitches": len(pitch_counts),
            "mean_velocity": float(np.mean(velocities)) if velocities else 0.0,
            "mean_duration": float(np.mean(durations)) if durations else 0.0,
        }
        logger.info(
            "Vocab stats: %d total notes, %d unique pitches",
            total,
            len(pitch_counts),
        )
        return stats

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_processed(
        self,
        sequences: List[NoteArray],
        path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a list of processed sequence arrays to a compressed NumPy file.

        The sequences are stacked into a single ``(num_windows, seq_len, 5)``
        array and saved with ``np.savez_compressed``.  Optional *metadata* is
        pickled alongside.

        Args:
            sequences: List of ``(seq_len, 5)`` arrays.
            path:      Output path (should end with ``.npz``).
            metadata:  Optional dict of extra information (tempos, file names,
                       etc.) stored as a pickled byte array inside the archive.

        Raises:
            ValueError: If *sequences* is empty.
        """
        if not sequences:
            raise ValueError("sequences list is empty – nothing to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        stacked = np.stack(sequences, axis=0).astype(np.float32)  # (W, L, 5)

        save_kwargs: Dict[str, Any] = {"sequences": stacked}
        if metadata is not None:
            save_kwargs["metadata_bytes"] = np.frombuffer(
                pickle.dumps(metadata), dtype=np.uint8
            )

        np.savez_compressed(str(path), **save_kwargs)
        logger.info("Saved %d sequences to %s", len(sequences), path)

    def load_processed(
        self,
        path: str | Path,
    ) -> Tuple[List[NoteArray], Optional[Dict[str, Any]]]:
        """Load sequences previously saved with :meth:`save_processed`.

        Args:
            path: Path to the ``.npz`` file.

        Returns:
            A ``(sequences, metadata)`` tuple.  *sequences* is a list of
            ``(seq_len, 5)`` float32 arrays.  *metadata* is the original dict
            if it was saved, otherwise *None*.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Processed file not found: {path}")

        archive = np.load(str(path), allow_pickle=False)
        stacked: np.ndarray = archive["sequences"]  # (W, L, 5)
        sequences = [stacked[i] for i in range(stacked.shape[0])]

        metadata: Optional[Dict[str, Any]] = None
        if "metadata_bytes" in archive:
            metadata = pickle.loads(archive["metadata_bytes"].tobytes())

        logger.info("Loaded %d sequences from %s", len(sequences), path)
        return sequences, metadata

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def sequences_to_token_ids(
        self,
        sequences: List[NoteArray],
        vocab: Optional[Dict[int, int]] = None,
    ) -> List[np.ndarray]:
        """Convert pitch columns to integer token IDs.

        If *vocab* is provided it must be a ``{midi_pitch: token_id}`` dict.
        Otherwise pitches are used directly as token IDs.

        Args:
            sequences: List of ``(seq_len, 5)`` arrays.
            vocab:     Optional pitch → token ID mapping.

        Returns:
            List of ``(seq_len,)`` int64 arrays containing token IDs.
        """
        result: List[np.ndarray] = []
        for seq in sequences:
            pitches = seq[:, 0].astype(np.int64)
            if vocab is not None:
                pitches = np.array(
                    [vocab.get(int(p), 0) for p in pitches], dtype=np.int64
                )
            result.append(pitches)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _notes_to_array(notes: Sequence[ParsedNote]) -> NoteArray:
        """Convert a list of :class:`ParsedNote` to a ``(N, 5)`` float32 array.

        Column order: ``[pitch, start_time, end_time, velocity, channel]``.

        Args:
            notes: Sequence of :class:`ParsedNote` objects.

        Returns:
            ``(N, 5)`` float32 NumPy array.
        """
        if not notes:
            return np.zeros((0, 5), dtype=np.float32)
        rows = [
            [n.pitch, n.start_time, n.end_time, n.velocity, n.channel]
            for n in notes
        ]
        return np.array(rows, dtype=np.float32)
