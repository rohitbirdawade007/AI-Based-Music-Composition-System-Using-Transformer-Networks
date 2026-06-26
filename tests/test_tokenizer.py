"""
Tests for MusicTokenizer
"""
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizer.music_tokenizer import (
    MusicTokenizer,
    NoteEvent,
    PAD_ID, BOS_ID, EOS_ID, REST_ID,
    NOTE_ON_OFFSET, DURATION_OFFSET, VELOCITY_OFFSET,
)


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def tokenizer():
    return MusicTokenizer()


@pytest.fixture
def sample_events():
    return [
        NoteEvent(pitch=60, duration=0.5, velocity=80),
        NoteEvent(pitch=64, duration=0.25, velocity=70),
        NoteEvent(pitch=67, duration=1.0, velocity=90),
    ]


# ── Basic construction ────────────────────────────────────────────────────
class TestMusicTokenizerInit:
    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 512

    def test_special_token_ids(self, tokenizer):
        assert tokenizer.pad_id == PAD_ID
        assert tokenizer.bos_id == BOS_ID
        assert tokenizer.eos_id == EOS_ID
        assert tokenizer.rest_id == REST_ID

    def test_repr(self, tokenizer):
        r = repr(tokenizer)
        assert "MusicTokenizer" in r
        assert "512" in r


# ── Note event creation ────────────────────────────────────────────────────
class TestNoteEvent:
    def test_basic_creation(self):
        e = NoteEvent(pitch=60, duration=0.5, velocity=80)
        assert e.pitch == 60
        assert e.duration == 0.5
        assert e.velocity == 80
        assert not e.is_rest

    def test_rest_creation(self):
        e = NoteEvent(pitch=0, duration=0.5, is_rest=True)
        assert e.is_rest

    def test_repr_note(self):
        e = NoteEvent(60, 0.5, 80)
        assert "60" in repr(e)

    def test_repr_rest(self):
        e = NoteEvent(0, 0.5, is_rest=True)
        assert "REST" in repr(e)


# ── Token ID generation ───────────────────────────────────────────────────
class TestTokenIds:
    def test_note_on_id_c4(self, tokenizer):
        assert tokenizer.note_on_id(60) == NOTE_ON_OFFSET + 60

    def test_note_on_id_range(self, tokenizer):
        for pitch in [0, 60, 127]:
            tok_id = tokenizer.note_on_id(pitch)
            assert NOTE_ON_OFFSET <= tok_id < NOTE_ON_OFFSET + 128

    def test_note_on_id_invalid(self, tokenizer):
        with pytest.raises(AssertionError):
            tokenizer.note_on_id(128)
        with pytest.raises(AssertionError):
            tokenizer.note_on_id(-1)

    def test_duration_id_short(self, tokenizer):
        tok = tokenizer.duration_id(0.1)
        assert DURATION_OFFSET <= tok < VELOCITY_OFFSET

    def test_duration_id_long(self, tokenizer):
        tok = tokenizer.duration_id(3.0)
        assert DURATION_OFFSET <= tok < VELOCITY_OFFSET

    def test_velocity_id(self, tokenizer):
        tok = tokenizer.velocity_id(80)
        assert VELOCITY_OFFSET <= tok

    def test_token_type_labels(self, tokenizer):
        assert tokenizer.token_type(PAD_ID) == "PAD"
        assert tokenizer.token_type(BOS_ID) == "BOS"
        assert tokenizer.token_type(EOS_ID) == "EOS"
        assert tokenizer.token_type(REST_ID) == "REST"
        assert "NOTE_ON" in tokenizer.token_type(NOTE_ON_OFFSET + 60)
        assert "DUR" in tokenizer.token_type(DURATION_OFFSET + 5)
        assert "VEL" in tokenizer.token_type(VELOCITY_OFFSET + 3)


# ── Encoding ──────────────────────────────────────────────────────────────
class TestEncoding:
    def test_encode_with_bos_eos(self, tokenizer, sample_events):
        ids = tokenizer.encode(sample_events, add_bos=True, add_eos=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID

    def test_encode_without_bos_eos(self, tokenizer, sample_events):
        ids = tokenizer.encode(sample_events, add_bos=False, add_eos=False)
        assert ids[0] != BOS_ID
        assert ids[-1] != EOS_ID

    def test_encode_empty(self, tokenizer):
        ids = tokenizer.encode([], add_bos=True, add_eos=True)
        assert ids == [BOS_ID, EOS_ID]

    def test_encode_max_length_truncation(self, tokenizer, sample_events):
        ids = tokenizer.encode(sample_events, max_length=5)
        assert len(ids) == 5

    def test_encode_max_length_padding(self, tokenizer):
        events = [NoteEvent(60, 0.5, 80)]
        ids = tokenizer.encode(events, max_length=20, add_bos=True, add_eos=True)
        assert len(ids) == 20
        assert ids[-1] == PAD_ID  # last pad token

    def test_encode_rest_event(self, tokenizer):
        events = [NoteEvent(0, 0.5, is_rest=True)]
        ids = tokenizer.encode(events, add_bos=False, add_eos=False)
        assert REST_ID in ids

    def test_encode_note_on_in_ids(self, tokenizer):
        events = [NoteEvent(60, 0.5, 80)]
        ids = tokenizer.encode(events, add_bos=False, add_eos=False)
        assert tokenizer.note_on_id(60) in ids


# ── Decoding ──────────────────────────────────────────────────────────────
class TestDecoding:
    def test_roundtrip(self, tokenizer):
        original = [NoteEvent(60, 0.5, 80), NoteEvent(64, 0.25, 70)]
        ids = tokenizer.encode(original, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids)
        pitches = [e.pitch for e in decoded if not e.is_rest]
        assert 60 in pitches
        assert 64 in pitches

    def test_decode_skips_specials(self, tokenizer):
        ids = [BOS_ID, PAD_ID, EOS_ID]
        events = tokenizer.decode(ids)
        assert len(events) == 0

    def test_decode_rest(self, tokenizer):
        ids = [REST_ID, DURATION_OFFSET + 5]
        events = tokenizer.decode(ids)
        assert len(events) == 1
        assert events[0].is_rest


# ── Padding ───────────────────────────────────────────────────────────────
class TestPadding:
    def test_pad_sequence_equal_lengths(self, tokenizer):
        seqs = [[1, 2, 3], [4, 5, 6]]
        padded, masks = tokenizer.pad_sequence(seqs)
        assert all(len(p) == 3 for p in padded)
        assert all(len(m) == 3 for m in masks)

    def test_pad_sequence_variable_lengths(self, tokenizer):
        seqs = [[1, 2], [3, 4, 5, 6]]
        padded, masks = tokenizer.pad_sequence(seqs)
        assert all(len(p) == 4 for p in padded)
        assert padded[0][-1] == PAD_ID
        assert masks[0][-1] == 0
        assert masks[1][-1] == 1

    def test_pad_sequence_custom_max(self, tokenizer):
        seqs = [[1, 2], [3, 4, 5]]
        padded, _ = tokenizer.pad_sequence(seqs, max_length=6)
        assert all(len(p) == 6 for p in padded)


# ── Save / Load ───────────────────────────────────────────────────────────
class TestSaveLoad:
    def test_save_and_load(self, tokenizer, tmp_path):
        import numpy as np
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(save_path)
        loaded = MusicTokenizer.load(save_path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert np.allclose(loaded.duration_bins, tokenizer.duration_bins)

    def test_loaded_tokenizer_identical_encoding(self, tokenizer, tmp_path):
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(save_path)
        loaded = MusicTokenizer.load(save_path)
        events = [NoteEvent(60, 0.5, 80)]
        assert tokenizer.encode(events) == loaded.encode(events)
