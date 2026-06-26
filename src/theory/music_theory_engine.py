"""
Music Theory Constraint Engine
================================
Validates and improves generated note sequences using music theory rules:
  - Key signature detection and note filtering
  - Scale membership validation (Major, Minor, Modes, Pentatonic, Blues)
  - Basic chord progression checking
  - Melody range and interval validation
  - Rhythm consistency scoring
  - Harmonic consistency improvement
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Scale Definitions (semitones from root) ────────────────────────────────
SCALES: Dict[str, List[int]] = {
    "major":           [0, 2, 4, 5, 7, 9, 11],
    "minor":           [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":  [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":   [0, 2, 3, 5, 7, 9, 11],
    "dorian":          [0, 2, 3, 5, 7, 9, 10],
    "phrygian":        [0, 1, 3, 5, 7, 8, 10],
    "lydian":          [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":      [0, 2, 4, 5, 7, 9, 10],
    "locrian":         [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_major":[0, 2, 4, 7, 9],
    "pentatonic_minor":[0, 3, 5, 7, 10],
    "blues":           [0, 3, 5, 6, 7, 10],
    "chromatic":       list(range(12)),
}

# ── Note names ────────────────────────────────────────────────────────────
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_SEMITONE = {n: i for i, n in enumerate(NOTE_NAMES)}

# ── Common chord progressions (in scale degrees, 0-indexed) ───────────────
CHORD_PROGRESSIONS: Dict[str, List[List[int]]] = {
    "major": [
        [0, 3, 4],         # I - IV - V
        [0, 5, 3, 4],      # I - vi - IV - V
        [0, 4, 5, 3],      # I - V - vi - IV
        [1, 4, 0],         # ii - V - I (jazz)
        [0, 3, 5, 4],      # I - IV - vi - V
    ],
    "minor": [
        [0, 3, 4],         # i - iv - v
        [0, 6, 3, 4],      # i - VII - iv - V
        [0, 3, 6, 4],      # i - iv - VII - V
        [5, 1, 4, 0],      # VI - III - VII - i
    ],
}

# ── Interval consonance (in semitones) ────────────────────────────────────
CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9, 12}
DISSONANT_INTERVALS = {1, 2, 6, 10, 11}


@dataclass
class TheoryValidationResult:
    """Result of a music theory validation check."""
    is_valid: bool = True
    score: float = 1.0               # 0.0 to 1.0 — higher is better
    violations: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    in_scale_ratio: float = 1.0
    consonance_score: float = 1.0
    range_score: float = 1.0
    rhythm_score: float = 1.0


class MusicTheoryEngine:
    """
    Music theory constraint engine for validating and improving melodies.

    Usage
    -----
    >>> engine = MusicTheoryEngine(key="C", mode="major")
    >>> pitches = [60, 62, 64, 65, 67]
    >>> result = engine.validate_melody(pitches)
    >>> improved = engine.apply_constraints(pitches)
    """

    def __init__(
        self,
        key: str = "C",
        mode: str = "major",
        melody_range_min: int = 40,
        melody_range_max: int = 84,
    ) -> None:
        self.key = key
        self.mode = mode
        self.melody_range_min = melody_range_min
        self.melody_range_max = melody_range_max

        self._root_semitone = NOTE_TO_SEMITONE.get(key, 0)
        self._scale_intervals = SCALES.get(mode, SCALES["major"])
        self._scale_pitches: Set[int] = self._build_scale_set()

    # ── Internal helpers ──────────────────────────────────────────────────
    def _build_scale_set(self) -> Set[int]:
        """Build the full set of MIDI pitches belonging to this key/mode."""
        pitches = set()
        for octave in range(11):
            for interval in self._scale_intervals:
                p = octave * 12 + self._root_semitone + interval
                if 0 <= p <= 127:
                    pitches.add(p)
        return pitches

    def _in_scale(self, pitch: int) -> bool:
        return (pitch - self._root_semitone) % 12 in self._scale_intervals

    def _nearest_scale_pitch(self, pitch: int) -> int:
        """Return the nearest pitch that is in the current scale."""
        if self._in_scale(pitch):
            return pitch
        for delta in range(1, 13):
            for candidate in [pitch + delta, pitch - delta]:
                if 0 <= candidate <= 127 and self._in_scale(candidate):
                    return candidate
        return pitch

    # ── Validation ────────────────────────────────────────────────────────
    def validate_key_membership(self, pitches: List[int]) -> float:
        """Return ratio of pitches that belong to the current key/mode."""
        if not pitches:
            return 1.0
        in_scale = sum(1 for p in pitches if self._in_scale(p))
        return in_scale / len(pitches)

    def validate_range(self, pitches: List[int]) -> float:
        """Return ratio of pitches within the acceptable melody range."""
        if not pitches:
            return 1.0
        in_range = sum(
            1 for p in pitches
            if self.melody_range_min <= p <= self.melody_range_max
        )
        return in_range / len(pitches)

    def validate_intervals(self, pitches: List[int]) -> float:
        """Return consonance score based on melodic intervals."""
        if len(pitches) < 2:
            return 1.0
        scores = []
        for a, b in zip(pitches, pitches[1:]):
            interval = abs(a - b) % 12
            scores.append(1.0 if interval in CONSONANT_INTERVALS else 0.0)
        return sum(scores) / len(scores)

    def validate_rhythm(self, durations: List[float]) -> float:
        """
        Score rhythm consistency.
        Reward sequences that stick to regular subdivisions of a beat.
        """
        if not durations:
            return 1.0
        # Common rhythmic values (in quarter-note fractions)
        valid_durs = {0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}
        regular = sum(
            1 for d in durations
            if any(abs(d - v) < 0.05 for v in valid_durs)
        )
        return regular / len(durations)

    def validate_chord_progression(self, pitches: List[int], window: int = 4) -> float:
        """
        Check if the first note of each window follows a common chord progression.
        Returns a rough score in [0, 1].
        """
        if len(pitches) < window:
            return 1.0

        mode_key = "major" if self.mode in ("major", "lydian", "mixolydian") else "minor"
        progressions = CHORD_PROGRESSIONS.get(mode_key, [])
        if not progressions:
            return 1.0

        # Chord roots for each window
        chord_roots = []
        for i in range(0, len(pitches) - window + 1, window):
            chunk = pitches[i:i + window]
            root = min(chunk) % 12  # simplified: lowest note as root
            scale_degree = (root - self._root_semitone) % 12
            chord_roots.append(scale_degree)

        if len(chord_roots) < 2:
            return 1.0

        # Check how many transitions appear in known progressions
        valid_transitions = 0
        total_transitions = 0
        for prog in progressions:
            prog_set = set(zip(prog, prog[1:]))
            for a, b in zip(chord_roots, chord_roots[1:]):
                total_transitions += 1
                if (a, b) in prog_set or (a % len(prog), b % len(prog)) in prog_set:
                    valid_transitions += 1

        return min(1.0, valid_transitions / max(total_transitions, 1))

    def validate_melody(
        self,
        pitches: List[int],
        durations: Optional[List[float]] = None,
    ) -> TheoryValidationResult:
        """
        Run all validation checks and return a composite result.
        """
        result = TheoryValidationResult()
        violations: List[str] = []
        suggestions: List[str] = []

        # Key membership
        in_scale_ratio = self.validate_key_membership(pitches)
        result.in_scale_ratio = in_scale_ratio
        if in_scale_ratio < 0.7:
            violations.append(
                f"Only {in_scale_ratio:.0%} of notes are in {self.key} {self.mode}"
            )
            suggestions.append(f"Snap non-scale notes to nearest {self.key} {self.mode} pitch")

        # Range
        range_score = self.validate_range(pitches)
        result.range_score = range_score
        if range_score < 0.9:
            violations.append("Some notes fall outside the standard melody range")
            suggestions.append("Transpose extreme notes to stay within E2–C6")

        # Consonance
        consonance = self.validate_intervals(pitches)
        result.consonance_score = consonance
        if consonance < 0.6:
            violations.append("Many dissonant melodic intervals detected")
            suggestions.append("Reduce large leaps; prefer stepwise motion")

        # Rhythm
        if durations:
            rhythm_score = self.validate_rhythm(durations)
            result.rhythm_score = rhythm_score
            if rhythm_score < 0.7:
                violations.append("Rhythm is irregular or non-standard")
                suggestions.append("Quantise durations to common note values")
        
        result.violations = violations
        result.suggestions = suggestions
        result.is_valid = len(violations) == 0

        # Composite score
        weights = [0.4, 0.2, 0.3, 0.1]
        scores = [
            result.in_scale_ratio,
            result.range_score,
            result.consonance_score,
            result.rhythm_score,
        ]
        result.score = sum(w * s for w, s in zip(weights, scores))
        return result

    # ── Constraint Application ────────────────────────────────────────────
    def snap_to_scale(self, pitches: List[int]) -> List[int]:
        """Move each pitch to the nearest in-scale pitch."""
        return [self._nearest_scale_pitch(p) for p in pitches]

    def clamp_range(self, pitches: List[int]) -> List[int]:
        """Clamp pitches to the allowed melody range (via octave transposition)."""
        result = []
        for p in pitches:
            while p < self.melody_range_min:
                p += 12
            while p > self.melody_range_max:
                p -= 12
            result.append(p)
        return result

    def smooth_leaps(self, pitches: List[int], max_leap: int = 7) -> List[int]:
        """Reduce large melodic leaps by stepping through intermediate scale pitches."""
        if len(pitches) < 2:
            return pitches
        result = [pitches[0]]
        for prev, curr in zip(pitches, pitches[1:]):
            leap = abs(curr - prev)
            if leap > max_leap:
                # Fill with a midpoint from the scale
                mid = (prev + curr) // 2
                mid = self._nearest_scale_pitch(mid)
                result.append(mid)
            result.append(curr)
        return result

    def apply_constraints(
        self,
        pitches: List[int],
        snap_scale: bool = True,
        clamp: bool = True,
        smooth: bool = True,
    ) -> List[int]:
        """Apply all enabled constraints to improve a melody."""
        if snap_scale:
            pitches = self.snap_to_scale(pitches)
        if clamp:
            pitches = self.clamp_range(pitches)
        if smooth:
            pitches = self.smooth_leaps(pitches)
        return pitches

    # ── Scale information ─────────────────────────────────────────────────
    def get_scale_pitches(self, octave_range: Tuple[int, int] = (3, 6)) -> List[int]:
        """Return all scale pitches in the given octave range (MIDI numbers)."""
        pitches = []
        for octave in range(*octave_range):
            for interval in self._scale_intervals:
                p = (octave + 1) * 12 + self._root_semitone + interval
                if 0 <= p <= 127:
                    pitches.append(p)
        return sorted(pitches)

    def get_chord_pitches(self, degree: int = 0) -> List[int]:
        """Return the triad chord pitches for a given scale degree (0-indexed)."""
        intervals = self._scale_intervals
        root_interval = intervals[degree % len(intervals)]
        third = intervals[(degree + 2) % len(intervals)]
        fifth = intervals[(degree + 4) % len(intervals)]

        # Build triad from C4 (MIDI 60)
        base = 60 + self._root_semitone
        chord = []
        for iv in [root_interval, third, fifth]:
            p = base + iv
            while p < 60:
                p += 12
            chord.append(p)
        return chord

    def suggest_next_note(
        self, previous_pitches: List[int], temperature: float = 1.0
    ) -> int:
        """
        Suggest the next note based on scale membership and step-motion preference.
        """
        scale_pitches = self.get_scale_pitches()
        if not previous_pitches:
            return random.choice(scale_pitches)

        last = previous_pitches[-1]
        # Prefer notes within a 5th of the last note
        candidates = [p for p in scale_pitches if abs(p - last) <= 7]
        if not candidates:
            candidates = scale_pitches

        # Weight by proximity
        weights = [1.0 / (abs(p - last) + 1) for p in candidates]
        # Apply temperature
        weights = [w ** (1.0 / max(temperature, 0.01)) for w in weights]
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(candidates, weights=weights, k=1)[0]

    def __repr__(self) -> str:
        return f"MusicTheoryEngine(key={self.key!r}, mode={self.mode!r})"
