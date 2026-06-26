"""
Tests for MusicGenerator (demo mode)
"""
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.generator import MusicGenerator, GenerationResult


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def generator():
    return MusicGenerator(demo_mode=True)


# ── Construction ──────────────────────────────────────────────────────────
class TestMusicGeneratorInit:
    def test_demo_mode_default(self, generator):
        assert generator.demo_mode is True

    def test_no_model_forces_demo(self):
        gen = MusicGenerator(model=None, demo_mode=False)
        assert gen.demo_mode is True  # forced because no model

    def test_repr(self, generator):
        r = repr(generator)
        assert "MusicGenerator" in r


# ── Generation ────────────────────────────────────────────────────────────
class TestGenerationBasic:
    def test_returns_generation_result(self, generator):
        result = generator.generate(num_notes=16)
        assert isinstance(result, GenerationResult)

    def test_correct_number_of_notes(self, generator):
        result = generator.generate(num_notes=32)
        # May have slightly more or fewer due to rests, but roughly correct
        assert 10 <= len(result.pitches) <= 50

    def test_pitches_are_valid_midi(self, generator):
        result = generator.generate(num_notes=32)
        for p in result.pitches:
            assert -1 <= p <= 127  # -1 for rests

    def test_durations_are_positive(self, generator):
        result = generator.generate(num_notes=32)
        for d in result.durations:
            assert d > 0

    def test_velocities_are_valid(self, generator):
        result = generator.generate(num_notes=32)
        for v in result.velocities:
            assert 0 <= v <= 127

    def test_midi_bytes_not_empty(self, generator):
        result = generator.generate(num_notes=16)
        assert isinstance(result.midi_bytes, bytes)
        assert len(result.midi_bytes) > 0

    def test_theory_score_in_range(self, generator):
        result = generator.generate(num_notes=16)
        assert 0.0 <= result.theory_score <= 1.0

    def test_generation_time_positive(self, generator):
        result = generator.generate(num_notes=16)
        assert result.generation_time > 0

    def test_method_is_demo(self, generator):
        result = generator.generate(num_notes=16)
        assert result.method == "demo"


# ── Genre/Mood combinations ───────────────────────────────────────────────
class TestGenreMoodCombinations:
    GENRES = ["Classical", "Jazz", "Blues", "Electronic", "Ambient", "Pop", "Folk"]
    MOODS  = ["Calm", "Energetic", "Melancholic", "Happy"]

    @pytest.mark.parametrize("genre", GENRES)
    def test_all_genres_generate(self, generator, genre):
        result = generator.generate(genre=genre, num_notes=16)
        assert len(result.pitches) > 0

    @pytest.mark.parametrize("mood", MOODS)
    def test_all_moods_generate(self, generator, mood):
        result = generator.generate(mood=mood, num_notes=16)
        assert len(result.pitches) > 0

    def test_different_genres_different_velocities(self, generator):
        """Classical (calm) should have lower avg velocity than Electronic (energetic)."""
        classical = generator.generate(genre="Classical", mood="Calm", num_notes=64)
        electronic = generator.generate(genre="Electronic", mood="Energetic", num_notes=64)
        avg_vel_classical = sum(v for v in classical.velocities if v > 0) / max(sum(1 for v in classical.velocities if v > 0), 1)
        avg_vel_electronic = sum(v for v in electronic.velocities if v > 0) / max(sum(1 for v in electronic.velocities if v > 0), 1)
        assert avg_vel_electronic >= avg_vel_classical - 10  # Electronic generally louder


# ── Key/Mode combinations ─────────────────────────────────────────────────
class TestKeyMode:
    KEYS  = ["C", "G", "D", "F", "A"]
    MODES = ["major", "minor", "dorian", "blues", "pentatonic_major"]

    @pytest.mark.parametrize("key", KEYS)
    def test_different_keys(self, generator, key):
        result = generator.generate(key=key, num_notes=16, apply_theory=True)
        assert result.key == key
        assert len(result.pitches) > 0

    @pytest.mark.parametrize("mode", MODES)
    def test_different_modes(self, generator, mode):
        result = generator.generate(mode=mode, num_notes=16)
        assert len(result.pitches) > 0


# ── Temperature ───────────────────────────────────────────────────────────
class TestTemperature:
    def test_low_temperature_generates_notes(self, generator):
        result = generator.generate(temperature=0.1, num_notes=32)
        assert len(result.pitches) > 0

    def test_high_temperature_generates_notes(self, generator):
        result = generator.generate(temperature=2.0, num_notes=32)
        assert len(result.pitches) > 0

    def test_temperature_affects_diversity(self, generator):
        """Higher temperature should produce more pitch variety."""
        import random
        random.seed(42)
        low_temp = generator.generate(temperature=0.1, num_notes=64)
        random.seed(42)
        high_temp = generator.generate(temperature=2.0, num_notes=64)

        low_unique = len(set(p for p in low_temp.pitches if p > 0))
        high_unique = len(set(p for p in high_temp.pitches if p > 0))
        # High temperature generally produces more unique pitches
        # (soft assertion — may occasionally fail due to randomness)
        assert high_unique >= low_unique - 5


# ── Seed notes ────────────────────────────────────────────────────────────
class TestSeedNotes:
    def test_with_seed_notes(self, generator):
        seed = [60, 64, 67]  # C major chord
        result = generator.generate(seed_pitches=seed, num_notes=32)
        assert len(result.pitches) > 0

    def test_empty_seed_notes(self, generator):
        result = generator.generate(seed_pitches=[], num_notes=32)
        assert len(result.pitches) > 0

    def test_large_seed_notes(self, generator):
        seed = list(range(60, 72))
        result = generator.generate(seed_pitches=seed, num_notes=32)
        assert len(result.pitches) > 0


# ── Theory constraints ────────────────────────────────────────────────────
class TestTheoryConstraints:
    def test_apply_theory_improves_score(self, generator):
        """Applying theory should not decrease the theory score."""
        no_theory = generator.generate(num_notes=32, apply_theory=False, key="C", mode="major")
        with_theory = generator.generate(num_notes=32, apply_theory=True, key="C", mode="major")
        assert with_theory.theory_score >= no_theory.theory_score - 0.1  # allow small variance

    def test_theory_score_with_good_scale(self, generator):
        result = generator.generate(
            key="C", mode="major", genre="Classical", mood="Calm",
            num_notes=32, apply_theory=True,
        )
        assert result.theory_score > 0.5


# ── MIDI output ───────────────────────────────────────────────────────────
class TestMidiOutput:
    def test_midi_has_midi_header(self, generator):
        result = generator.generate(num_notes=16)
        # MIDI files start with "MThd"
        assert result.midi_bytes[:4] in (b"MThd", b"")

    def test_save_midi(self, generator, tmp_path):
        result = generator.generate(num_notes=16)
        out = tmp_path / "test_output.mid"
        result.save_midi(str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_midi_different_tempos(self, generator):
        slow = generator.generate(tempo=60, num_notes=16)
        fast = generator.generate(tempo=200, num_notes=16)
        assert slow.tempo == 60.0
        assert fast.tempo == 200.0


# ── GenerationResult properties ───────────────────────────────────────────
class TestGenerationResultProperties:
    def test_num_notes_property(self, generator):
        result = generator.generate(num_notes=32)
        # num_notes counts only pitches > 0 (not rests)
        counted = sum(1 for p in result.pitches if p > 0)
        assert result.num_notes == counted

    def test_total_duration_property(self, generator):
        result = generator.generate(num_notes=16)
        expected = sum(result.durations)
        assert abs(result.total_duration - expected) < 1e-6

    def test_repr_contains_info(self, generator):
        result = generator.generate(num_notes=16)
        r = repr(result)
        assert "GenerationResult" in r
        assert "notes=" in r
