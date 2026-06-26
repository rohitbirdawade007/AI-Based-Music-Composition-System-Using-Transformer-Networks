"""
Music Tokenizer
===============
Converts symbolic music events (note pitch, duration, velocity, tempo)
into integer token sequences suitable for Transformer training/inference.

Vocabulary Layout (512 tokens):
  [0]       PAD
  [1]       BOS  (Begin of Sequence)
  [2]       EOS  (End of Sequence)
  [3]       REST
  [4]       UNK
  [5–131]   NOTE_ON  for MIDI pitches 0–127  (offset 5)
  [132–258] NOTE_OFF for MIDI pitches 0–127  (offset 132)
  [259–290] DURATION bins 0–31              (offset 259)
  [291–306] VELOCITY bins 0–15              (offset 291)
  [307–314] TEMPO bins 0–7                  (offset 307)
  [315–511] Reserved / future use
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Special token IDs ──────────────────────────────────────────────────────
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
REST_ID = 3
UNK_ID = 4

# ── Offsets ────────────────────────────────────────────────────────────────
NOTE_ON_OFFSET = 5          # 5  … 131
NOTE_OFF_OFFSET = 132       # 132 … 258
DURATION_OFFSET = 259       # 259 … 290  (32 bins)
VELOCITY_OFFSET = 291       # 291 … 306  (16 bins)
TEMPO_OFFSET = 307          # 307 … 314  (8 bins)

VOCAB_SIZE = 512

# ── Duration quantisation (in seconds, 32 log-spaced bins) ─────────────────
_DUR_MIN = 0.05   # 50 ms
_DUR_MAX = 4.0    # 4 s
DURATION_BINS = np.logspace(math.log10(_DUR_MIN), math.log10(_DUR_MAX), 32)

# ── Velocity quantisation (MIDI 0–127 → 16 bins) ──────────────────────────
VELOCITY_BINS = np.linspace(0, 127, 16, endpoint=False).astype(int)

# ── Tempo quantisation (BPM 40–240 → 8 bins) ──────────────────────────────
TEMPO_BINS = np.linspace(40, 240, 8, endpoint=False).astype(int)


def _quantise(value: float, bins: np.ndarray) -> int:
    """Return the index of the nearest bin for *value*."""
    idx = int(np.searchsorted(bins, value, side="right")) - 1
    return max(0, min(idx, len(bins) - 1))


class NoteEvent:
    """A single symbolic music event."""

    __slots__ = ("pitch", "duration", "velocity", "start_time", "is_rest")

    def __init__(
        self,
        pitch: int,
        duration: float,
        velocity: int = 80,
        start_time: float = 0.0,
        is_rest: bool = False,
    ) -> None:
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity
        self.start_time = start_time
        self.is_rest = is_rest

    def __repr__(self) -> str:
        if self.is_rest:
            return f"REST(dur={self.duration:.3f})"
        return (
            f"Note(pitch={self.pitch}, dur={self.duration:.3f}, "
            f"vel={self.velocity}, t={self.start_time:.3f})"
        )


class MusicTokenizer:
    """
    Tokenise / detokenise sequences of :class:`NoteEvent` objects.

    Example
    -------
    >>> tok = MusicTokenizer()
    >>> events = [NoteEvent(60, 0.5, 80), NoteEvent(64, 0.25, 70)]
    >>> ids = tok.encode(events, add_bos=True, add_eos=True)
    >>> decoded = tok.decode(ids)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        duration_bins: Optional[np.ndarray] = None,
        velocity_bins: Optional[np.ndarray] = None,
        tempo_bins: Optional[np.ndarray] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.duration_bins = duration_bins if duration_bins is not None else DURATION_BINS
        self.velocity_bins = velocity_bins if velocity_bins is not None else VELOCITY_BINS
        self.tempo_bins = tempo_bins if tempo_bins is not None else TEMPO_BINS

        # Reverse-lookup tables
        self._duration_values = self.duration_bins.tolist()
        self._velocity_values = [int(v) for v in self.velocity_bins]
        self._tempo_values = [int(t) for t in self.tempo_bins]

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def rest_id(self) -> int:
        return REST_ID

    # ── Token → ID helpers ─────────────────────────────────────────────────
    def note_on_id(self, pitch: int) -> int:
        assert 0 <= pitch <= 127, f"Invalid pitch {pitch}"
        return NOTE_ON_OFFSET + pitch

    def note_off_id(self, pitch: int) -> int:
        assert 0 <= pitch <= 127, f"Invalid pitch {pitch}"
        return NOTE_OFF_OFFSET + pitch

    def duration_id(self, duration_sec: float) -> int:
        bin_idx = _quantise(duration_sec, self.duration_bins)
        return DURATION_OFFSET + bin_idx

    def velocity_id(self, velocity: int) -> int:
        bin_idx = _quantise(velocity, self.velocity_bins)
        return VELOCITY_OFFSET + bin_idx

    def tempo_id(self, bpm: float) -> int:
        bin_idx = _quantise(bpm, self.tempo_bins)
        return TEMPO_OFFSET + bin_idx

    # ── Encoding ───────────────────────────────────────────────────────────
    def encode_event(self, event: NoteEvent, include_velocity: bool = True) -> List[int]:
        """Encode a single NoteEvent into a list of token IDs."""
        if event.is_rest:
            return [REST_ID, self.duration_id(event.duration)]

        tokens = [
            self.note_on_id(event.pitch),
            self.duration_id(event.duration),
        ]
        if include_velocity:
            tokens.append(self.velocity_id(event.velocity))
        return tokens

    def encode(
        self,
        events: List[NoteEvent],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
        include_velocity: bool = True,
    ) -> List[int]:
        """Encode a sequence of NoteEvents into token IDs."""
        ids: List[int] = []
        if add_bos:
            ids.append(BOS_ID)

        for event in events:
            ids.extend(self.encode_event(event, include_velocity))

        if add_eos:
            ids.append(EOS_ID)

        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
                if add_eos:
                    ids[-1] = EOS_ID
            else:
                ids += [PAD_ID] * (max_length - len(ids))

        return ids

    # ── Decoding ───────────────────────────────────────────────────────────
    def decode(self, ids: List[int]) -> List[NoteEvent]:
        """Decode a list of token IDs back into NoteEvents."""
        events: List[NoteEvent] = []
        i = 0
        current_time = 0.0

        while i < len(ids):
            tok = ids[i]

            # Skip specials
            if tok in (PAD_ID, BOS_ID, EOS_ID):
                i += 1
                continue

            # REST
            if tok == REST_ID:
                dur = self._decode_duration(ids[i + 1]) if i + 1 < len(ids) else 0.25
                events.append(NoteEvent(0, dur, is_rest=True, start_time=current_time))
                current_time += dur
                i += 2
                continue

            # NOTE_ON
            if NOTE_ON_OFFSET <= tok < NOTE_OFF_OFFSET:
                pitch = tok - NOTE_ON_OFFSET
                dur = 0.25
                vel = 80
                if i + 1 < len(ids) and DURATION_OFFSET <= ids[i + 1] < VELOCITY_OFFSET:
                    dur = self._decode_duration(ids[i + 1])
                    i += 1
                if i + 1 < len(ids) and VELOCITY_OFFSET <= ids[i + 1] < TEMPO_OFFSET:
                    vel = self._decode_velocity(ids[i + 1])
                    i += 1
                events.append(NoteEvent(pitch, dur, vel, start_time=current_time))
                current_time += dur
                i += 1
                continue

            i += 1

        return events

    def _decode_duration(self, tok: int) -> float:
        idx = tok - DURATION_OFFSET
        if 0 <= idx < len(self._duration_values):
            return self._duration_values[idx]
        return 0.25

    def _decode_velocity(self, tok: int) -> int:
        idx = tok - VELOCITY_OFFSET
        if 0 <= idx < len(self._velocity_values):
            return self._velocity_values[idx]
        return 80

    # ── Utilities ──────────────────────────────────────────────────────────
    def pad_sequence(
        self, sequences: List[List[int]], max_length: Optional[int] = None
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Pad a batch of sequences and return (padded, attention_masks)."""
        if max_length is None:
            max_length = max(len(s) for s in sequences)

        padded, masks = [], []
        for seq in sequences:
            pad_len = max_length - len(seq)
            padded.append(seq + [PAD_ID] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)

        return padded, masks

    def token_type(self, tok_id: int) -> str:
        """Return a human-readable label for a token ID."""
        if tok_id == PAD_ID:
            return "PAD"
        if tok_id == BOS_ID:
            return "BOS"
        if tok_id == EOS_ID:
            return "EOS"
        if tok_id == REST_ID:
            return "REST"
        if NOTE_ON_OFFSET <= tok_id < NOTE_OFF_OFFSET:
            return f"NOTE_ON_{tok_id - NOTE_ON_OFFSET}"
        if NOTE_OFF_OFFSET <= tok_id < DURATION_OFFSET:
            return f"NOTE_OFF_{tok_id - NOTE_OFF_OFFSET}"
        if DURATION_OFFSET <= tok_id < VELOCITY_OFFSET:
            return f"DUR_{tok_id - DURATION_OFFSET}"
        if VELOCITY_OFFSET <= tok_id < TEMPO_OFFSET:
            return f"VEL_{tok_id - VELOCITY_OFFSET}"
        if TEMPO_OFFSET <= tok_id < VOCAB_SIZE:
            return f"TEMPO_{tok_id - TEMPO_OFFSET}"
        return "UNK"

    def save(self, path: Union[str, Path]) -> None:
        """Persist tokenizer config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "vocab_size": self.vocab_size,
            "duration_bins": self.duration_bins.tolist(),
            "velocity_bins": self.velocity_bins.tolist(),
            "tempo_bins": self.tempo_bins.tolist(),
        }
        path.write_text(json.dumps(config, indent=2))
        logger.info("Tokenizer saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MusicTokenizer":
        """Load tokenizer config from JSON."""
        config = json.loads(Path(path).read_text())
        return cls(
            vocab_size=config["vocab_size"],
            duration_bins=np.array(config["duration_bins"]),
            velocity_bins=np.array(config["velocity_bins"]),
            tempo_bins=np.array(config["tempo_bins"]),
        )

    def __repr__(self) -> str:
        return (
            f"MusicTokenizer(vocab_size={self.vocab_size}, "
            f"duration_bins={len(self.duration_bins)}, "
            f"velocity_bins={len(self.velocity_bins)})"
        )
