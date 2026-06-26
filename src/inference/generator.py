"""
Music Generator Module
======================
Provides the MusicGenerator class for AI-driven music composition.
Supports a DEMO MODE (no trained model required) using Markov-chain-inspired
scale/rhythm heuristics, plus full model-based decoding strategies
(greedy, temperature sampling, top-k, top-p, beam search).

Dependencies:
  - midiutil  (pip install midiutil)
  - numpy
  - torch      (optional; only required for model-based generation)

Usage (demo):
    from src.inference.generator import MusicGenerator, cli_generate
    gen = MusicGenerator(demo_mode=True)
    result = gen.generate(genre='Jazz', mood='Energetic', num_notes=32)
    with open('output.mid', 'wb') as f:
        f.write(result.midi_bytes)

Usage (CLI):
    python -m src.inference.generator --genre Classical --mood Calm --num_notes 64
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch import — gracefully degrade when not installed
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    F = None      # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional midiutil import
# ---------------------------------------------------------------------------
try:
    from midiutil import MIDIFile  # type: ignore

    MIDIUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    MIDIUTIL_AVAILABLE = False
    MIDIFile = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: Scales & Modes
# ---------------------------------------------------------------------------

# Semitone intervals from root for common scales
SCALE_INTERVALS: Dict[str, List[int]] = {
    "major":        [0, 2, 4, 5, 7, 9, 11],
    "minor":        [0, 2, 3, 5, 7, 8, 10],
    "dorian":       [0, 2, 3, 5, 7, 9, 10],
    "phrygian":     [0, 1, 3, 5, 7, 8, 10],
    "lydian":       [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":   [0, 2, 4, 5, 7, 9, 10],
    "locrian":      [0, 1, 3, 5, 6, 8, 10],
    "pentatonic":   [0, 2, 4, 7, 9],
    "blues":        [0, 3, 5, 6, 7, 10],
    "chromatic":    list(range(12)),
}

# Map note name → MIDI root pitch in octave 4 (middle octave)
NOTE_TO_MIDI_ROOT: Dict[str, int] = {
    "C": 60, "C#": 61, "Db": 61,
    "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66,
    "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70,
    "Bb": 70, "B": 71,
}

MIDI_NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

# ---------------------------------------------------------------------------
# Genre / Mood profiles for demo mode
# ---------------------------------------------------------------------------

@dataclass
class _GenreMoodProfile:
    """Encapsulates generation parameters for a genre+mood combination."""
    scale: str
    octave_range: Tuple[int, int]          # (low_octave, high_octave) — MIDI octave numbers
    duration_range: Tuple[float, float]    # (min_seconds, max_seconds)
    velocity_range: Tuple[int, int]        # (min_vel, max_vel)
    rest_probability: float                # probability of inserting a rest
    syncopation: bool                      # whether to add syncopated rhythms
    pitch_drift: int                       # max semitone drift for "bends"
    chord_probability: float               # probability of repeating a note as a chord tone


# Lookup: (genre_lower, mood_lower) → profile
# Falls back to a default profile when no exact match found.
_GENRE_MOOD_PROFILES: Dict[Tuple[str, str], _GenreMoodProfile] = {
    ("classical", "calm"): _GenreMoodProfile(
        scale="major", octave_range=(4, 6), duration_range=(0.5, 1.0),
        velocity_range=(50, 70), rest_probability=0.05, syncopation=False,
        pitch_drift=0, chord_probability=0.15,
    ),
    ("classical", "energetic"): _GenreMoodProfile(
        scale="major", octave_range=(4, 6), duration_range=(0.25, 0.5),
        velocity_range=(70, 100), rest_probability=0.03, syncopation=False,
        pitch_drift=0, chord_probability=0.2,
    ),
    ("classical", "melancholic"): _GenreMoodProfile(
        scale="minor", octave_range=(3, 5), duration_range=(0.5, 1.5),
        velocity_range=(40, 65), rest_probability=0.08, syncopation=False,
        pitch_drift=0, chord_probability=0.1,
    ),
    ("jazz", "calm"): _GenreMoodProfile(
        scale="dorian", octave_range=(4, 6), duration_range=(0.25, 0.75),
        velocity_range=(55, 80), rest_probability=0.1, syncopation=True,
        pitch_drift=1, chord_probability=0.25,
    ),
    ("jazz", "energetic"): _GenreMoodProfile(
        scale="dorian", octave_range=(4, 6), duration_range=(0.125, 0.5),
        velocity_range=(70, 105), rest_probability=0.12, syncopation=True,
        pitch_drift=1, chord_probability=0.3,
    ),
    ("blues", "calm"): _GenreMoodProfile(
        scale="blues", octave_range=(3, 5), duration_range=(0.375, 1.0),
        velocity_range=(55, 80), rest_probability=0.1, syncopation=True,
        pitch_drift=2, chord_probability=0.2,
    ),
    ("blues", "energetic"): _GenreMoodProfile(
        scale="blues", octave_range=(3, 5), duration_range=(0.125, 0.375),
        velocity_range=(75, 110), rest_probability=0.08, syncopation=True,
        pitch_drift=2, chord_probability=0.25,
    ),
    ("electronic", "energetic"): _GenreMoodProfile(
        scale="pentatonic", octave_range=(4, 7), duration_range=(0.125, 0.25),
        velocity_range=(90, 127), rest_probability=0.05, syncopation=True,
        pitch_drift=0, chord_probability=0.3,
    ),
    ("electronic", "calm"): _GenreMoodProfile(
        scale="pentatonic", octave_range=(4, 6), duration_range=(0.25, 0.75),
        velocity_range=(60, 85), rest_probability=0.08, syncopation=False,
        pitch_drift=0, chord_probability=0.2,
    ),
    ("ambient", "calm"): _GenreMoodProfile(
        scale="major", octave_range=(3, 6), duration_range=(1.0, 4.0),
        velocity_range=(30, 55), rest_probability=0.15, syncopation=False,
        pitch_drift=0, chord_probability=0.1,
    ),
    ("ambient", "melancholic"): _GenreMoodProfile(
        scale="minor", octave_range=(3, 6), duration_range=(1.5, 5.0),
        velocity_range=(25, 50), rest_probability=0.2, syncopation=False,
        pitch_drift=0, chord_probability=0.1,
    ),
    ("pop", "calm"): _GenreMoodProfile(
        scale="major", octave_range=(4, 6), duration_range=(0.25, 0.5),
        velocity_range=(60, 80), rest_probability=0.05, syncopation=False,
        pitch_drift=0, chord_probability=0.2,
    ),
    ("pop", "energetic"): _GenreMoodProfile(
        scale="major", octave_range=(4, 6), duration_range=(0.125, 0.375),
        velocity_range=(80, 110), rest_probability=0.04, syncopation=True,
        pitch_drift=0, chord_probability=0.25,
    ),
    ("rock", "energetic"): _GenreMoodProfile(
        scale="minor", octave_range=(3, 6), duration_range=(0.125, 0.375),
        velocity_range=(85, 120), rest_probability=0.05, syncopation=True,
        pitch_drift=1, chord_probability=0.3,
    ),
    ("folk", "calm"): _GenreMoodProfile(
        scale="major", octave_range=(4, 5), duration_range=(0.25, 0.75),
        velocity_range=(55, 75), rest_probability=0.08, syncopation=False,
        pitch_drift=0, chord_probability=0.1,
    ),
}

_DEFAULT_PROFILE = _GenreMoodProfile(
    scale="major", octave_range=(4, 6), duration_range=(0.25, 0.75),
    velocity_range=(60, 90), rest_probability=0.07, syncopation=False,
    pitch_drift=0, chord_probability=0.15,
)


def _get_profile(genre: str, mood: str) -> _GenreMoodProfile:
    """Return the best-matching genre/mood profile, falling back gracefully."""
    key = (genre.lower(), mood.lower())
    if key in _GENRE_MOOD_PROFILES:
        return _GENRE_MOOD_PROFILES[key]
    # Try genre-only fallback (first match)
    for (g, _m), profile in _GENRE_MOOD_PROFILES.items():
        if g == genre.lower():
            return profile
    return _DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# GenerationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Container for a single music generation output.

    Attributes
    ----------
    pitches : List[int]
        MIDI pitch numbers (0-127). A value of -1 indicates a rest.
    durations : List[float]
        Duration of each note/rest in seconds.
    velocities : List[int]
        MIDI velocity (0-127) for each note.
    tempo : float
        Beats per minute.
    key : str
        Musical key root (e.g., ``'C'``, ``'F#'``).
    mode : str
        Scale mode used (e.g., ``'major'``, ``'dorian'``).
    genre : str
        Genre label (e.g., ``'Jazz'``).
    mood : str
        Mood label (e.g., ``'Calm'``).
    midi_bytes : bytes
        Binary MIDI file content (can be written directly to a ``.mid`` file).
    theory_score : float
        Heuristic score [0, 1] measuring conformance to music theory rules.
        1.0 means all notes are scale-conformant.
    generation_time : float
        Wall-clock time taken for generation in seconds.
    method : str
        Generation method: ``'demo'``, ``'greedy'``, ``'sampling'``, or ``'beam'``.
    metadata : Dict[str, Any]
        Arbitrary extra information from the generation process.
    """
    pitches: List[int]
    durations: List[float]
    velocities: List[int]
    tempo: float
    key: str
    mode: str
    genre: str
    mood: str
    midi_bytes: bytes
    theory_score: float
    generation_time: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_notes(self) -> int:
        """Number of pitched notes (excluding rests)."""
        return sum(1 for p in self.pitches if p >= 0)

    @property
    def total_duration(self) -> float:
        """Total duration of the piece in seconds."""
        return sum(self.durations)

    def save_midi(self, path: str) -> None:
        """Write the MIDI bytes to *path*."""
        with open(path, "wb") as fh:
            fh.write(self.midi_bytes)
        logger.info("MIDI saved to %s", path)

    def __repr__(self) -> str:
        return (
            f"GenerationResult(notes={self.num_notes}, "
            f"duration={self.total_duration:.1f}s, "
            f"key={self.key} {self.mode}, "
            f"genre={self.genre}, mood={self.mood}, "
            f"method={self.method}, theory_score={self.theory_score:.3f})"
        )


# ---------------------------------------------------------------------------
# Stub / placeholder for MusicTheoryEngine
# ---------------------------------------------------------------------------

class _FallbackTheoryEngine:
    """
    Minimal theory engine used when the real MusicTheoryEngine is unavailable.
    Provides scale note sets and a simple next-note suggestion method.
    """

    def get_scale_pitches(self, key: str, mode: str, octave_range: Tuple[int, int]) -> List[int]:
        """Return all MIDI pitches within *octave_range* that belong to the scale."""
        root = NOTE_TO_MIDI_ROOT.get(key, 60)
        intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
        pitches: List[int] = []
        for octave in range(octave_range[0], octave_range[1] + 1):
            base = (octave - 4) * 12 + root  # shift from octave-4 root
            for interval in intervals:
                pitch = base + interval
                if 0 <= pitch <= 127:
                    pitches.append(pitch)
        return sorted(pitches)

    def suggest_next_note(
        self,
        current_pitch: int,
        scale_pitches: List[int],
        step_weight: float = 0.6,
    ) -> int:
        """
        Suggest the next MIDI pitch using weighted proximity within the scale.

        Parameters
        ----------
        current_pitch : int
            The last played pitch.
        scale_pitches : List[int]
            Allowed pitches in the current scale.
        step_weight : float
            Proportion of weight given to step-wise motion vs. leaps.

        Returns
        -------
        int
            A MIDI pitch from *scale_pitches*.
        """
        if not scale_pitches:
            return current_pitch

        # Compute distance-weighted probabilities
        distances = [abs(p - current_pitch) for p in scale_pitches]
        max_dist = max(distances) or 1
        # Closer notes get higher weight (step-wise preference)
        weights = [(1 - d / max_dist) * step_weight + (1 - step_weight)
                   for d in distances]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(scale_pitches, weights=probs, k=1)[0]

    def score_sequence(self, pitches: List[int], key: str, mode: str) -> float:
        """Return fraction of pitches that lie within the given scale."""
        if not pitches:
            return 1.0
        root = NOTE_TO_MIDI_ROOT.get(key, 60) % 12
        intervals = set(SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"]))
        in_scale = sum(1 for p in pitches if p >= 0 and (p - root) % 12 in intervals)
        total = sum(1 for p in pitches if p >= 0)
        return in_scale / total if total else 1.0


# ---------------------------------------------------------------------------
# MusicGenerator
# ---------------------------------------------------------------------------

class MusicGenerator:
    """
    AI Music Generator supporting both demo (model-free) and full model modes.

    In **demo mode** (``demo_mode=True``), generation uses a Markov-chain-inspired
    approach with genre/mood heuristics — no trained neural network required.

    In **model mode**, the generator wraps a trained sequence-to-sequence model
    and supports greedy decoding, temperature sampling, top-k, top-p, and beam search.

    Parameters
    ----------
    model : optional
        A PyTorch nn.Module with a ``forward(src, tgt)`` signature.
        Not required in demo mode.
    tokenizer : optional
        A tokenizer/vocabulary object with ``encode`` and ``decode`` methods.
    theory_engine : optional
        An instance of ``MusicTheoryEngine``.  Falls back to a built-in stub.
    config : dict, optional
        Configuration overrides (e.g. ``{'max_len': 256}``).
    demo_mode : bool
        If ``True`` (default), always use the heuristic generator regardless of
        whether a model is supplied.
    """

    # Default configuration
    _DEFAULT_CONFIG: Dict[str, Any] = {
        "max_len": 256,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "device": "cpu",
    }

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        theory_engine: Any = None,
        config: Optional[Dict[str, Any]] = None,
        demo_mode: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.theory_engine: Any = theory_engine or _FallbackTheoryEngine()
        self.config: Dict[str, Any] = {**self._DEFAULT_CONFIG, **(config or {})}
        self.demo_mode = demo_mode or (model is None)

        # Warn if torch is missing but model-mode was requested
        if not self.demo_mode and not TORCH_AVAILABLE:
            logger.warning(
                "PyTorch is not installed; falling back to demo mode."
            )
            self.demo_mode = True

        if not MIDIUTIL_AVAILABLE:
            logger.warning(
                "midiutil is not installed.  MIDI output will return empty bytes. "
                "Install with: pip install midiutil"
            )

        logger.info(
            "MusicGenerator initialised | demo_mode=%s | torch=%s | midiutil=%s",
            self.demo_mode, TORCH_AVAILABLE, MIDIUTIL_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        seed_pitches: Optional[List[int]] = None,
        genre: str = "Classical",
        mood: str = "Calm",
        key: str = "C",
        mode: str = "major",
        tempo: float = 120.0,
        num_notes: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_beam: bool = False,
        beam_width: int = 4,
        apply_theory: bool = True,
        rag_context: Optional[Any] = None,
    ) -> GenerationResult:
        """
        Generate a musical sequence.

        Parameters
        ----------
        seed_pitches : List[int], optional
            Initial MIDI pitches to condition generation on.
        genre : str
            Musical genre (e.g., ``'Classical'``, ``'Jazz'``, ``'Blues'``).
        mood : str
            Emotional mood (e.g., ``'Calm'``, ``'Energetic'``, ``'Melancholic'``).
        key : str
            Root note of the key (e.g., ``'C'``, ``'F#'``).
        mode : str
            Scale mode.  Overridden by genre/mood profile in demo mode.
        tempo : float
            BPM for playback and MIDI output.
        num_notes : int
            Target number of notes to generate (rests excluded from count).
        temperature : float
            Sampling temperature; higher → more random.
        top_k : int
            Top-K cutoff for sampling (model mode only).
        top_p : float
            Nucleus probability threshold (model mode only).
        use_beam : bool
            Use beam search when in model mode.
        beam_width : int
            Number of beams for beam search.
        apply_theory : bool
            Apply theory-guided corrections after generation.
        rag_context : optional
            Retrieval-augmented generation context (reserved for future use).

        Returns
        -------
        GenerationResult
            Complete generation output including MIDI bytes.
        """
        start_time = time.perf_counter()
        seed_pitches = seed_pitches or []

        if self.demo_mode:
            pitches, durations, velocities = self._demo_generate(
                seed_pitches=seed_pitches,
                genre=genre,
                mood=mood,
                key=key,
                mode=mode,
                num_notes=num_notes,
                temperature=temperature,
            )
            method = "demo"
        else:
            pitches, durations, velocities = self._model_generate(
                seed_pitches=seed_pitches,
                key=key,
                mode=mode,
                num_notes=num_notes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_beam=use_beam,
                beam_width=beam_width,
            )
            method = "beam" if use_beam else "sampling"

        # Apply theory-guided corrections
        if apply_theory and hasattr(self.theory_engine, "correct_sequence"):
            pitches = self.theory_engine.correct_sequence(pitches, key=key, mode=mode)

        # Compute theory conformance score
        theory_score = self.theory_engine.score_sequence(pitches, key=key, mode=mode)

        # Build MIDI bytes
        midi_bytes = self.pitches_to_midi(
            pitches=pitches,
            durations=durations,
            velocities=velocities,
            tempo=tempo,
        )

        generation_time = time.perf_counter() - start_time

        result = GenerationResult(
            pitches=pitches,
            durations=durations,
            velocities=velocities,
            tempo=tempo,
            key=key,
            mode=mode,
            genre=genre,
            mood=mood,
            midi_bytes=midi_bytes,
            theory_score=theory_score,
            generation_time=generation_time,
            method=method,
            metadata={
                "seed_pitches": seed_pitches,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "use_beam": use_beam,
                "beam_width": beam_width,
                "rag_context_provided": rag_context is not None,
            },
        )

        logger.info("Generated: %s", result)
        return result

    # ------------------------------------------------------------------
    # Demo-mode generation
    # ------------------------------------------------------------------

    def _demo_generate(
        self,
        seed_pitches: List[int],
        genre: str,
        mood: str,
        key: str,
        mode: str,
        num_notes: int,
        temperature: float,
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Heuristic Markov-chain-like generation using genre/mood profiles.

        The algorithm:
          1. Resolve genre/mood → scale, duration range, velocity range.
          2. Build the full set of valid pitches in the scale.
          3. Iteratively suggest the next pitch using the theory engine,
             then apply temperature-weighted random selection.
          4. Sample duration and velocity from the profile's ranges,
             adding rhythmic variation (syncopation, dotted rhythms).
          5. Optionally add rests according to the rest probability.
        """
        profile = _get_profile(genre, mood)

        # The profile may override the user-requested mode
        effective_mode = profile.scale

        # Build scale pitch set
        scale_pitches = self.theory_engine.get_scale_pitches(
            key=key,
            mode=effective_mode,
            octave_range=profile.octave_range,
        )

        if not scale_pitches:
            # Absolute fallback: chromatic middle octave
            scale_pitches = list(range(60, 72))

        pitches: List[int] = []
        durations: List[float] = []
        velocities: List[int] = []

        # Seed the Markov chain
        if seed_pitches:
            current_pitch = seed_pitches[-1]
        else:
            # Start near the middle of the scale
            mid_idx = len(scale_pitches) // 2
            current_pitch = scale_pitches[mid_idx]

        notes_generated = 0
        max_iterations = num_notes * 4  # guard against infinite loop with rests

        rng = random.Random()  # local RNG for reproducibility seeding if needed

        for _iteration in range(max_iterations):
            if notes_generated >= num_notes:
                break

            # ---- Decide rest ----
            if rng.random() < profile.rest_probability:
                pitches.append(-1)  # rest sentinel
                rest_dur = rng.uniform(*profile.duration_range)
                durations.append(round(rest_dur, 4))
                velocities.append(0)
                continue

            # ---- Suggest next pitch via theory engine ----
            suggested = self.theory_engine.suggest_next_note(
                current_pitch=current_pitch,
                scale_pitches=scale_pitches,
            )

            # Apply temperature-weighted selection over scale candidates
            candidate_pitch = self._temperature_select(
                current=current_pitch,
                candidates=scale_pitches,
                temperature=temperature,
                rng=rng,
            )

            # Blend suggestion and sampled candidate
            final_pitch = suggested if rng.random() < 0.5 else candidate_pitch

            # Apply pitch drift (simulates blues bends / vibrato)
            if profile.pitch_drift > 0:
                drift = rng.randint(-profile.pitch_drift, profile.pitch_drift)
                final_pitch = max(0, min(127, final_pitch + drift))

            pitches.append(final_pitch)
            current_pitch = final_pitch

            # ---- Sample duration ----
            dur = self._sample_duration(profile=profile, rng=rng)
            # Syncopation: occasionally halve or double the duration
            if profile.syncopation and rng.random() < 0.25:
                dur = dur * rng.choice([0.5, 1.5])
            durations.append(round(max(0.0625, dur), 4))

            # ---- Sample velocity ----
            vel = rng.randint(*profile.velocity_range)
            # Accent first beat of each bar (every 4th note approx.)
            if notes_generated % 4 == 0 and rng.random() < 0.6:
                vel = min(127, vel + rng.randint(8, 15))
            velocities.append(vel)

            notes_generated += 1

        return pitches, durations, velocities

    @staticmethod
    def _temperature_select(
        current: int,
        candidates: List[int],
        temperature: float,
        rng: random.Random,
    ) -> int:
        """
        Select a pitch from *candidates* using a temperature-softmax over
        proximity scores (inverse distance from *current*).

        Parameters
        ----------
        current : int
            Current MIDI pitch.
        candidates : List[int]
            Pool of valid pitches.
        temperature : float
            Sampling temperature.  Values close to 0 → greedy (nearest neighbour);
            values > 1 → more random.
        rng : random.Random
            Random number generator instance.
        """
        if temperature <= 0 or len(candidates) == 1:
            # Greedy: nearest candidate
            return min(candidates, key=lambda p: abs(p - current))

        # Compute negative-distance logits
        logits = [-abs(p - current) / max(temperature, 1e-8) for p in candidates]
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]
        return rng.choices(candidates, weights=probs, k=1)[0]

    @staticmethod
    def _sample_duration(profile: _GenreMoodProfile, rng: random.Random) -> float:
        """
        Sample a note duration with occasional dotted-note variation.

        Returns seconds as a float.
        """
        base_dur = rng.uniform(*profile.duration_range)
        # 15% chance of dotted note (×1.5)
        if rng.random() < 0.15:
            base_dur *= 1.5
        # 10% chance of triplet note (×2/3)
        elif rng.random() < 0.10:
            base_dur *= 2 / 3
        return base_dur

    # ------------------------------------------------------------------
    # Model-based generation
    # ------------------------------------------------------------------

    def _model_generate(
        self,
        seed_pitches: List[int],
        key: str,
        mode: str,
        num_notes: int,
        temperature: float,
        top_k: int,
        top_p: float,
        use_beam: bool,
        beam_width: int,
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Run the neural model to generate a token sequence and decode to pitches.

        Falls back to demo mode if no model or tokenizer is present.
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("No model/tokenizer — delegating to demo generation.")
            return self._demo_generate(
                seed_pitches=seed_pitches, genre="Classical", mood="Calm",
                key=key, mode=mode, num_notes=num_notes, temperature=temperature,
            )

        device = self.config.get("device", "cpu")
        max_len = self.config.get("max_len", 256)

        # Encode seed
        if seed_pitches:
            src_tokens = self.tokenizer.encode(seed_pitches)
        else:
            src_tokens = [self.config["bos_token_id"]]

        src = torch.tensor([src_tokens], dtype=torch.long, device=device)

        self.model.eval()
        with torch.no_grad():
            if use_beam:
                token_ids = self._beam_search(src, max_len=max_len, beam_width=beam_width)
            else:
                token_ids = self._greedy_decode(src, max_len=min(max_len, num_notes * 2))

        # Decode tokens to (pitch, duration, velocity) triples
        pitches, durations, velocities = self.tokenizer.decode(token_ids)
        return pitches[:num_notes], durations[:num_notes], velocities[:num_notes]

    # ------------------------------------------------------------------
    # Decoding strategies
    # ------------------------------------------------------------------

    def _greedy_decode(self, src: "torch.Tensor", max_len: int) -> List[int]:
        """
        Greedy autoregressive decoding: always picks the argmax token.

        Parameters
        ----------
        src : torch.Tensor
            Shape ``(1, src_len)`` — encoded source sequence.
        max_len : int
            Maximum number of tokens to generate.

        Returns
        -------
        List[int]
            Generated token IDs (excluding BOS/EOS).
        """
        if not TORCH_AVAILABLE:
            return []

        eos_id = self.config["eos_token_id"]
        bos_id = self.config["bos_token_id"]
        device = src.device

        tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        generated: List[int] = []

        for _ in range(max_len):
            logits = self.model(src, tgt)  # (1, tgt_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            next_token = int(next_token_logits.argmax(dim=-1).item())

            if next_token == eos_id:
                break

            generated.append(next_token)
            tgt = torch.cat(
                [tgt, torch.tensor([[next_token]], dtype=torch.long, device=device)],
                dim=1,
            )

        return generated

    @staticmethod
    def _temperature_sample(logits: "torch.Tensor", temperature: float) -> int:
        """
        Sample a token from *logits* after applying temperature scaling.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(vocab_size,)`` — raw unnormalised log-probabilities.
        temperature : float
            Scaling factor.  > 1 flattens distribution; < 1 sharpens it.

        Returns
        -------
        int
            Sampled token ID.
        """
        if not TORCH_AVAILABLE:
            return 0
        scaled = logits / max(temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def _top_k_sample(logits: "torch.Tensor", k: int, temperature: float) -> int:
        """
        Top-K sampling: zero out all but the *k* highest-probability tokens,
        then sample with temperature.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(vocab_size,)``.
        k : int
            Number of top candidates to keep.
        temperature : float
            Sampling temperature applied before softmax.

        Returns
        -------
        int
            Sampled token ID.
        """
        if not TORCH_AVAILABLE:
            return 0
        k = min(k, logits.size(-1))
        top_values, _ = torch.topk(logits, k)
        # Mask values below the k-th largest
        threshold = top_values[..., -1, None]
        filtered = logits.masked_fill(logits < threshold, float("-inf"))
        scaled = filtered / max(temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def _top_p_sample(logits: "torch.Tensor", p: float, temperature: float) -> int:
        """
        Nucleus (Top-P) sampling: keep the smallest set of tokens whose
        cumulative probability exceeds *p*.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(vocab_size,)``.
        p : float
            Nucleus probability threshold (0 < p ≤ 1).
        temperature : float
            Sampling temperature.

        Returns
        -------
        int
            Sampled token ID.
        """
        if not TORCH_AVAILABLE:
            return 0
        scaled = logits / max(temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens once cumulative probability exceeds p
        sorted_to_remove = cumulative_probs - sorted_probs > p
        sorted_probs[sorted_to_remove] = 0.0
        sorted_probs /= sorted_probs.sum()
        sampled_idx = int(torch.multinomial(sorted_probs, num_samples=1).item())
        return int(sorted_indices[sampled_idx].item())

    def _beam_search(
        self,
        src: "torch.Tensor",
        max_len: int,
        beam_width: int,
    ) -> List[int]:
        """
        Beam search decoding over the model's output distribution.

        Maintains *beam_width* candidate sequences simultaneously and returns
        the one with the highest accumulated log-probability.

        Parameters
        ----------
        src : torch.Tensor
            Shape ``(1, src_len)``.
        max_len : int
            Maximum generation length.
        beam_width : int
            Number of beams.

        Returns
        -------
        List[int]
            Best token sequence (excluding BOS/EOS).
        """
        if not TORCH_AVAILABLE:
            return []

        device = src.device
        bos_id = self.config["bos_token_id"]
        eos_id = self.config["eos_token_id"]

        # Each beam: (log_prob, token_ids_tensor)
        beams: List[Tuple[float, "torch.Tensor"]] = [
            (0.0, torch.tensor([[bos_id]], dtype=torch.long, device=device))
        ]
        completed: List[Tuple[float, List[int]]] = []

        for _step in range(max_len):
            if not beams:
                break

            all_candidates: List[Tuple[float, "torch.Tensor"]] = []

            for log_prob, tgt in beams:
                logits = self.model(src, tgt)          # (1, tgt_len, vocab_size)
                next_logits = logits[0, -1, :]         # (vocab_size,)
                log_probs = F.log_softmax(next_logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    token = int(top_indices[i].item())
                    new_log_prob = log_prob + float(top_log_probs[i].item())
                    new_tgt = torch.cat(
                        [tgt, torch.tensor([[token]], dtype=torch.long, device=device)],
                        dim=1,
                    )
                    if token == eos_id:
                        seq = new_tgt[0, 1:-1].tolist()  # strip BOS + EOS
                        completed.append((new_log_prob, seq))
                    else:
                        all_candidates.append((new_log_prob, new_tgt))

            # Keep top beam_width beams
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_width]

        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]

        # Return best incomplete beam
        if beams:
            best_tgt = beams[0][1]
            return best_tgt[0, 1:].tolist()  # strip BOS
        return []

    # ------------------------------------------------------------------
    # MIDI utilities
    # ------------------------------------------------------------------

    def tokens_to_midi(
        self,
        tokens: List[int],
        tempo: float = 120.0,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Convert a list of raw token IDs to MIDI bytes.

        Token IDs are interpreted directly as MIDI pitches (valid range 0-127).
        All notes receive a uniform duration of one quarter note.

        Parameters
        ----------
        tokens : List[int]
            Token IDs to convert.
        tempo : float
            Playback tempo in BPM.
        output_path : str, optional
            If provided, also write the MIDI to this file path.

        Returns
        -------
        bytes
            Binary MIDI file content.
        """
        valid_pitches = [t for t in tokens if 0 <= t <= 127]
        durations = [0.5] * len(valid_pitches)
        velocities = [80] * len(valid_pitches)
        midi_bytes = self.pitches_to_midi(valid_pitches, durations, velocities, tempo)

        if output_path:
            with open(output_path, "wb") as fh:
                fh.write(midi_bytes)
            logger.info("MIDI written to %s", output_path)

        return midi_bytes

    def pitches_to_midi(
        self,
        pitches: List[int],
        durations: List[float],
        velocities: List[int],
        tempo: float = 120.0,
    ) -> bytes:
        """
        Convert aligned pitch/duration/velocity lists to a MIDI file in memory.

        Parameters
        ----------
        pitches : List[int]
            MIDI pitches (0-127); ``-1`` is treated as a rest.
        durations : List[float]
            Duration of each note/rest in seconds.
        velocities : List[int]
            MIDI velocity (0-127) for each note.
        tempo : float
            Tempo in BPM.

        Returns
        -------
        bytes
            Binary MIDI file content suitable for writing to a ``.mid`` file.
        """
        if not MIDIUTIL_AVAILABLE:
            logger.error("midiutil is not installed; returning empty MIDI bytes.")
            return b""

        beats_per_second = tempo / 60.0

        midi = MIDIFile(1)           # one track
        midi.addTempo(0, 0, tempo)   # track 0, time 0
        midi.addProgramChange(0, 0, 0, 0)  # channel 0, program 0 (Acoustic Grand Piano)

        current_time_beats = 0.0

        for pitch, duration_sec, velocity in zip(pitches, durations, velocities):
            duration_beats = duration_sec * beats_per_second

            if pitch >= 0:  # note (not rest)
                midi.addNote(
                    track=0,
                    channel=0,
                    pitch=int(np.clip(pitch, 0, 127)),
                    time=current_time_beats,
                    duration=max(duration_beats, 0.0625),
                    volume=int(np.clip(velocity, 0, 127)),
                )

            current_time_beats += duration_beats

        # Write to in-memory buffer
        buf = BytesIO()
        midi.writeFile(buf)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_generate() -> None:
    """
    Command-line interface for quick music generation.

    Example
    -------
    .. code-block:: bash

        python -m src.inference.generator \\
            --genre Jazz --mood Energetic \\
            --key D --num_notes 48 --tempo 140 \\
            --temperature 0.9 --output my_song.mid
    """
    parser = argparse.ArgumentParser(
        description="AI Music Generator — Demo CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genre", default="Classical",
                        help="Musical genre (Classical/Jazz/Blues/Electronic/Ambient/Pop/Rock/Folk)")
    parser.add_argument("--mood", default="Calm",
                        help="Mood (Calm/Energetic/Melancholic)")
    parser.add_argument("--key", default="C",
                        help="Musical key root note (C/D/E/F/G/A/B with optional #/b)")
    parser.add_argument("--mode", default="major",
                        help="Scale mode (major/minor/dorian/blues/pentatonic/…)")
    parser.add_argument("--tempo", type=float, default=120.0,
                        help="Tempo in BPM")
    parser.add_argument("--num_notes", type=int, default=64,
                        help="Number of notes to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--output", default="generated.mid",
                        help="Output MIDI file path")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    generator = MusicGenerator(demo_mode=True)

    print(f"\n🎵  Generating {args.num_notes} notes | Genre: {args.genre} | "
          f"Mood: {args.mood} | Key: {args.key} {args.mode} | Tempo: {args.tempo} BPM")

    result = generator.generate(
        genre=args.genre,
        mood=args.mood,
        key=args.key,
        mode=args.mode,
        tempo=args.tempo,
        num_notes=args.num_notes,
        temperature=args.temperature,
    )

    result.save_midi(args.output)

    print(f"\n✅  Done!")
    print(f"   Notes generated : {result.num_notes}")
    print(f"   Total duration  : {result.total_duration:.2f}s")
    print(f"   Theory score    : {result.theory_score:.3f}")
    print(f"   Generation time : {result.generation_time*1000:.1f}ms")
    print(f"   MIDI saved to   : {args.output}\n")


if __name__ == "__main__":
    cli_generate()
