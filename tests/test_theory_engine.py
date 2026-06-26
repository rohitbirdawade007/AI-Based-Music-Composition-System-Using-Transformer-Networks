"""
Tests for MusicTheoryEngine
"""
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.theory.music_theory_engine import MusicTheoryEngine, SCALES


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def cmajor():
    return MusicTheoryEngine(key="C", mode="major")


@pytest.fixture
def aminor():
    return MusicTheoryEngine(key="A", mode="minor")


@pytest.fixture
def cblues():
    return MusicTheoryEngine(key="C", mode="blues")


# ── Construction ──────────────────────────────────────────────────────────
class TestInit:
    def test_default_key_mode(self, cmajor):
        assert cmajor.key == "C"
        assert cmajor.mode == "major"

    def test_custom_key_mode(self):
        engine = MusicTheoryEngine(key="G", mode="dorian")
        assert engine.key == "G"
        assert engine.mode == "dorian"

    def test_repr(self, cmajor):
        r = repr(cmajor)
        assert "C" in r
        assert "major" in r

    def test_all_modes_construct(self):
        for mode in SCALES.keys():
            eng = MusicTheoryEngine(key="C", mode=mode)
            assert eng.mode == mode


# ── Scale membership ──────────────────────────────────────────────────────
class TestScaleMembership:
    def test_c_major_pitches(self, cmajor):
        # C4=60, D4=62, E4=64, F4=65, G4=67, A4=69, B4=71 — all in C major
        c_major_notes = [60, 62, 64, 65, 67, 69, 71]
        for pitch in c_major_notes:
            assert cmajor._in_scale(pitch), f"Expected {pitch} to be in C major"

    def test_c_major_non_members(self, cmajor):
        # C# D# F# G# A# — not in C major
        non_scale = [61, 63, 66, 68, 70]
        for pitch in non_scale:
            assert not cmajor._in_scale(pitch), f"Expected {pitch} NOT in C major"

    def test_a_minor_relative_to_c_major(self, aminor):
        # A minor has same notes as C major (A B C D E F G)
        a_minor_roots = [69, 71, 60, 62, 64, 65, 67]
        for pitch in a_minor_roots:
            assert aminor._in_scale(pitch)

    def test_get_scale_pitches_returns_sorted(self, cmajor):
        pitches = cmajor.get_scale_pitches()
        assert pitches == sorted(pitches)

    def test_get_scale_pitches_all_in_scale(self, cmajor):
        pitches = cmajor.get_scale_pitches()
        for p in pitches:
            assert cmajor._in_scale(p)

    def test_get_scale_pitches_octave_range(self, cmajor):
        pitches = cmajor.get_scale_pitches(octave_range=(4, 5))
        assert all(48 <= p <= 72 for p in pitches)


# ── Validation ────────────────────────────────────────────────────────────
class TestValidation:
    def test_perfect_scale_ratio(self, cmajor):
        c_major = [60, 62, 64, 65, 67, 69, 71]
        ratio = cmajor.validate_key_membership(c_major)
        assert ratio == 1.0

    def test_zero_scale_ratio(self, cmajor):
        non_scale = [61, 63, 66, 68, 70]
        ratio = cmajor.validate_key_membership(non_scale)
        assert ratio == 0.0

    def test_empty_pitches_ratio(self, cmajor):
        assert cmajor.validate_key_membership([]) == 1.0

    def test_range_validation_in_range(self, cmajor):
        pitches = list(range(48, 80))
        score = cmajor.validate_range(pitches)
        assert score == 1.0

    def test_range_validation_out_of_range(self, cmajor):
        pitches = [20, 110]  # very low and very high
        score = cmajor.validate_range(pitches)
        assert score < 1.0

    def test_interval_consonance_unison(self, cmajor):
        # All unison intervals (0 semitones) are consonant
        pitches = [60, 60, 60, 60]
        score = cmajor.validate_intervals(pitches)
        assert score == 1.0

    def test_interval_consonance_thirds(self, cmajor):
        # Third intervals (3, 4 semitones) — consonant
        pitches = [60, 64, 60, 63]  # major and minor thirds
        score = cmajor.validate_intervals(pitches)
        assert score > 0.5

    def test_rhythm_regular_durations(self, cmajor):
        durations = [0.25, 0.5, 0.25, 0.5, 1.0]
        score = cmajor.validate_rhythm(durations)
        assert score > 0.8

    def test_rhythm_irregular_durations(self, cmajor):
        durations = [0.13, 0.37, 0.62, 0.91]  # non-standard
        score = cmajor.validate_rhythm(durations)
        assert score < 0.5

    def test_validate_melody_good(self, cmajor):
        pitches = [60, 62, 64, 65, 67, 69, 71]  # C major scale
        result = cmajor.validate_melody(pitches)
        assert result.in_scale_ratio == 1.0
        assert result.score > 0.6

    def test_validate_melody_bad(self, cmajor):
        pitches = [61, 63, 66, 68, 70]  # all non-scale
        result = cmajor.validate_melody(pitches)
        assert result.in_scale_ratio == 0.0
        assert result.score < 0.5
        assert not result.is_valid
        assert len(result.violations) > 0

    def test_validate_melody_with_durations(self, cmajor):
        pitches = [60, 62, 64, 65]
        durations = [0.5, 0.5, 0.25, 0.25]
        result = cmajor.validate_melody(pitches, durations)
        assert result.rhythm_score > 0.5


# ── Constraint application ────────────────────────────────────────────────
class TestConstraints:
    def test_snap_to_scale(self, cmajor):
        # C# (61) should snap to C (60) or D (62)
        snapped = cmajor.snap_to_scale([61])
        assert snapped[0] in cmajor.get_scale_pitches()

    def test_snap_to_scale_all_in_scale(self, cmajor):
        non_scale = [61, 63, 66, 68, 70]
        snapped = cmajor.snap_to_scale(non_scale)
        for p in snapped:
            assert cmajor._in_scale(p)

    def test_clamp_range_low(self, cmajor):
        low_pitches = [20, 25, 30]
        clamped = cmajor.clamp_range(low_pitches)
        for p in clamped:
            assert p >= cmajor.melody_range_min

    def test_clamp_range_high(self, cmajor):
        high_pitches = [90, 100, 110]
        clamped = cmajor.clamp_range(high_pitches)
        for p in clamped:
            assert p <= cmajor.melody_range_max

    def test_smooth_leaps(self, cmajor):
        # A large leap should be smoothed
        pitches = [60, 84]  # 24 semitone leap
        smoothed = cmajor.smooth_leaps(pitches, max_leap=7)
        assert len(smoothed) > 2  # intermediate note inserted

    def test_apply_constraints_reduces_violations(self, cmajor):
        non_scale = [61, 63, 66, 68, 70]
        improved = cmajor.apply_constraints(non_scale)
        ratio_before = cmajor.validate_key_membership(non_scale)
        ratio_after = cmajor.validate_key_membership(improved)
        assert ratio_after > ratio_before

    def test_apply_constraints_empty(self, cmajor):
        result = cmajor.apply_constraints([])
        assert result == []


# ── Chord functionality ───────────────────────────────────────────────────
class TestChords:
    def test_get_chord_pitches_tonic(self, cmajor):
        chord = cmajor.get_chord_pitches(degree=0)
        assert len(chord) == 3  # triad

    def test_suggest_next_note_returns_scale_pitch(self, cmajor):
        scale = cmajor.get_scale_pitches()
        suggestion = cmajor.suggest_next_note([60, 64])
        assert suggestion in scale

    def test_suggest_next_note_empty_history(self, cmajor):
        scale = cmajor.get_scale_pitches()
        suggestion = cmajor.suggest_next_note([])
        assert suggestion in scale

    def test_suggest_next_note_temperature_zero(self, cmajor):
        # Temperature 0 → deterministic
        results = [cmajor.suggest_next_note([60], temperature=0.01) for _ in range(5)]
        # Most should be the same with very low temperature
        assert len(set(results)) <= 3  # some variance ok due to random.choices


# ── Chord progression ─────────────────────────────────────────────────────
class TestChordProgression:
    def test_progression_score_range(self, cmajor):
        pitches = [60, 65, 67, 62]  # I-IV-V-ii like
        score = cmajor.validate_chord_progression(pitches, window=2)
        assert 0.0 <= score <= 1.0

    def test_progression_short_melody(self, cmajor):
        # Should return 1.0 for very short melodies
        score = cmajor.validate_chord_progression([60], window=4)
        assert score == 1.0
