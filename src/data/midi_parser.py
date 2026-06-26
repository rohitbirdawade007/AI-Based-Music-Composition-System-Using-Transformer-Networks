"""MIDI Parser Module.

This module provides :class:`ParsedNote` and :class:`MidiParser` for reading
and writing MIDI files.  It depends on the ``mido`` library when available but
degrades gracefully when it is not installed (parse operations will raise a
clear :class:`ImportError`; simple MIDI generation falls back to a minimal
hand-rolled SMF writer).

Typical usage::

    parser = MidiParser()
    notes  = parser.parse_file("my_song.mid")
    bpm    = parser.extract_tempo_from_file("my_song.mid")
    parser.notes_to_midi(notes, tempo=120.0, output_path="out.mid")
"""

from __future__ import annotations

import io
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: mido
# ---------------------------------------------------------------------------
try:
    import mido  # type: ignore

    _MIDO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MIDO_AVAILABLE = False
    logger.warning(
        "mido is not installed.  Install it with `pip install mido` to enable "
        "full MIDI parsing.  Minimal MIDI *generation* is still available."
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ParsedNote:
    """A single musical note extracted from a MIDI file.

    Attributes:
        pitch:      MIDI pitch number (0–127, middle-C = 60).
        start_time: Absolute start time of the note in **seconds**.
        end_time:   Absolute end time of the note in **seconds**.
        duration:   Convenience property equal to ``end_time - start_time``.
        velocity:   MIDI velocity (0–127).
        channel:    MIDI channel on which the note was played (0–15).
    """

    pitch: int
    start_time: float
    end_time: float
    velocity: int
    channel: int = 0

    # ------------------------------------------------------------------
    # Derived / computed
    # ------------------------------------------------------------------

    @property
    def duration(self) -> float:
        """Duration of the note in seconds."""
        return max(0.0, self.end_time - self.start_time)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def transpose(self, semitones: int) -> "ParsedNote":
        """Return a new :class:`ParsedNote` transposed by *semitones*.

        Args:
            semitones: Number of semitones to shift (may be negative).

        Returns:
            A new :class:`ParsedNote` with the adjusted pitch, clamped to
            the valid MIDI range [0, 127].
        """
        new_pitch = max(0, min(127, self.pitch + semitones))
        return ParsedNote(
            pitch=new_pitch,
            start_time=self.start_time,
            end_time=self.end_time,
            velocity=self.velocity,
            channel=self.channel,
        )

    def time_stretch(self, factor: float) -> "ParsedNote":
        """Return a new :class:`ParsedNote` with timing scaled by *factor*.

        Args:
            factor: Positive float.  Values > 1 slow the note down;
                    values < 1 speed it up.

        Returns:
            A new :class:`ParsedNote` with scaled ``start_time`` and
            ``end_time``.

        Raises:
            ValueError: If *factor* is not positive.
        """
        if factor <= 0:
            raise ValueError(f"time_stretch factor must be positive, got {factor!r}")
        return ParsedNote(
            pitch=self.pitch,
            start_time=self.start_time * factor,
            end_time=self.end_time * factor,
            velocity=self.velocity,
            channel=self.channel,
        )

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"ParsedNote(pitch={self.pitch}, "
            f"start={self.start_time:.3f}s, "
            f"end={self.end_time:.3f}s, "
            f"vel={self.velocity}, ch={self.channel})"
        )


# ---------------------------------------------------------------------------
# MidiParser
# ---------------------------------------------------------------------------


class MidiParser:
    """Parse MIDI files into :class:`ParsedNote` sequences and generate MIDI.

    This class wraps the ``mido`` library for reading arbitrary MIDI data
    and provides a minimal pure-Python writer for generating MIDI output
    even when ``mido`` is not installed.

    Args:
        default_tempo: Fallback tempo in BPM used when no tempo event is
            found in the MIDI stream.  Defaults to ``120.0``.
        quantize_ticks: If *True* (default), quantize note timings to the
            MIDI tick grid during generation.

    Examples::

        parser = MidiParser()
        notes = parser.parse_file("beethoven.mid")
        print(f"Parsed {len(notes)} notes, first: {notes[0]}")
    """

    # MIDI standard: default microseconds per beat (= 120 BPM)
    _DEFAULT_TEMPO_US: int = 500_000  # µs / beat  →  120 BPM

    def __init__(
        self,
        default_tempo: float = 120.0,
        quantize_ticks: bool = True,
    ) -> None:
        self.default_tempo = default_tempo
        self.quantize_ticks = quantize_ticks

    # ------------------------------------------------------------------
    # Public parsing API
    # ------------------------------------------------------------------

    def parse_file(self, path: str | Path) -> List[ParsedNote]:
        """Parse a MIDI file from disk.

        Args:
            path: Filesystem path to a ``.mid`` / ``.midi`` file.

        Returns:
            Chronologically sorted list of :class:`ParsedNote` objects.

        Raises:
            ImportError:  If ``mido`` is not installed.
            FileNotFoundError: If *path* does not exist.
            ValueError:   If the file cannot be parsed as MIDI.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MIDI file not found: {path}")
        logger.info("Parsing MIDI file: %s", path)
        return self.parse_from_bytes(path.read_bytes())

    def parse_from_bytes(self, midi_bytes: bytes) -> List[ParsedNote]:
        """Parse MIDI data from a raw byte string.

        Args:
            midi_bytes: Raw bytes of a MIDI file (e.g., read from disk or
                fetched from a database / network).

        Returns:
            Chronologically sorted list of :class:`ParsedNote` objects.

        Raises:
            ImportError: If ``mido`` is not installed.
            ValueError:  If *midi_bytes* is not valid MIDI data.
        """
        _require_mido()
        try:
            mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        except Exception as exc:
            raise ValueError(f"Could not parse MIDI bytes: {exc}") from exc
        return self._extract_notes(mid)

    # ------------------------------------------------------------------
    # Tempo / time-signature helpers
    # ------------------------------------------------------------------

    def extract_tempo(self, mid: "mido.MidiFile") -> float:  # type: ignore[name-defined]
        """Extract the first tempo marking from a *mido* :class:`MidiFile`.

        Args:
            mid: A parsed ``mido.MidiFile`` object.

        Returns:
            Tempo in BPM (beats per minute).  Falls back to
            :attr:`default_tempo` if no tempo event is present.
        """
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    bpm = mido.tempo2bpm(msg.tempo)  # type: ignore[attr-defined]
                    logger.debug("Found tempo event: %.2f BPM", bpm)
                    return float(bpm)
        logger.debug("No tempo event found; using default %.2f BPM", self.default_tempo)
        return self.default_tempo

    def extract_tempo_from_file(self, path: str | Path) -> float:
        """Convenience wrapper: open a file and return its tempo in BPM.

        Args:
            path: Path to the MIDI file.

        Returns:
            Tempo in BPM.
        """
        _require_mido()
        mid = mido.MidiFile(str(path))  # type: ignore[attr-defined]
        return self.extract_tempo(mid)

    def extract_time_signature(
        self,
        mid: "mido.MidiFile",  # type: ignore[name-defined]
    ) -> Tuple[int, int]:
        """Extract the first time-signature event from a *mido* MidiFile.

        Args:
            mid: A parsed ``mido.MidiFile`` object.

        Returns:
            A ``(numerator, denominator)`` tuple, e.g. ``(4, 4)`` for common
            time.  Falls back to ``(4, 4)`` if no event is found.
        """
        for track in mid.tracks:
            for msg in track:
                if msg.type == "time_signature":
                    logger.debug(
                        "Found time signature: %d/%d", msg.numerator, msg.denominator
                    )
                    return (msg.numerator, msg.denominator)
        logger.debug("No time-signature event found; defaulting to 4/4")
        return (4, 4)

    # ------------------------------------------------------------------
    # MIDI generation
    # ------------------------------------------------------------------

    def notes_to_midi(
        self,
        notes: Sequence[ParsedNote],
        tempo: float = 120.0,
        output_path: Optional[str | Path] = None,
        ticks_per_beat: int = 480,
    ) -> bytes:
        """Serialise a sequence of :class:`ParsedNote` objects to MIDI bytes.

        The notes are written to a single-track (type-0) MIDI file.  If
        ``mido`` is available it is used for reliability; otherwise a
        minimal hand-rolled SMF writer is used.

        Args:
            notes:         Iterable of :class:`ParsedNote` objects.
            tempo:         Target tempo in BPM.  Defaults to ``120.0``.
            output_path:   Optional path to write the MIDI file to disk.
                           When *None* only the bytes are returned.
            ticks_per_beat: MIDI ticks per quarter-note resolution.

        Returns:
            Raw MIDI bytes of the generated file.
        """
        notes = list(notes)
        if not notes:
            logger.warning("notes_to_midi called with empty note list")

        if _MIDO_AVAILABLE:
            midi_bytes = self._notes_to_midi_mido(notes, tempo, ticks_per_beat)
        else:
            midi_bytes = self._notes_to_midi_fallback(notes, tempo, ticks_per_beat)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(midi_bytes)
            logger.info("MIDI saved to %s", output_path)

        return midi_bytes

    def generate_from_tuples(
        self,
        note_tuples: Sequence[Tuple[int, float, int]],
        tempo: float = 120.0,
        output_path: Optional[str | Path] = None,
        ticks_per_beat: int = 480,
        channel: int = 0,
    ) -> bytes:
        """Generate a MIDI file from a list of ``(pitch, duration, velocity)`` tuples.

        This is a convenience wrapper around :meth:`notes_to_midi` that builds
        :class:`ParsedNote` objects sequentially (notes are placed back-to-back
        with no gaps).

        Args:
            note_tuples:   Sequence of ``(pitch, duration_seconds, velocity)``.
            tempo:         Target tempo in BPM.
            output_path:   Optional output path.
            ticks_per_beat: MIDI ticks per quarter-note.
            channel:       MIDI channel for all notes (default 0).

        Returns:
            Raw MIDI bytes.

        Examples::

            parser = MidiParser()
            # C major scale, each note 0.5 s at velocity 80
            scale = [(60+i, 0.5, 80) for i in [0, 2, 4, 5, 7, 9, 11, 12]]
            parser.generate_from_tuples(scale, output_path="scale.mid")
        """
        cursor = 0.0
        parsed: List[ParsedNote] = []
        for pitch, dur, vel in note_tuples:
            parsed.append(
                ParsedNote(
                    pitch=int(pitch),
                    start_time=cursor,
                    end_time=cursor + float(dur),
                    velocity=int(vel),
                    channel=channel,
                )
            )
            cursor += float(dur)
        return self.notes_to_midi(
            parsed, tempo=tempo, output_path=output_path, ticks_per_beat=ticks_per_beat
        )

    # ------------------------------------------------------------------
    # Internal helpers – note extraction
    # ------------------------------------------------------------------

    def _extract_notes(self, mid: "mido.MidiFile") -> List[ParsedNote]:  # type: ignore[name-defined]
        """Convert a *mido* MidiFile into a sorted list of ParsedNotes.

        This method merges all tracks into a single absolute-time event
        stream, then matches note-on / note-off pairs to build duration
        information.

        Args:
            mid: A parsed ``mido.MidiFile`` object.

        Returns:
            Chronologically sorted list of :class:`ParsedNote` objects.
        """
        # -- Collect all messages with absolute times in seconds ----------
        events: List[Tuple[float, "mido.Message"]] = []  # type: ignore[name-defined]
        for track in mid.tracks:
            abs_time_s = 0.0
            tempo_us = self._DEFAULT_TEMPO_US
            for msg in track:
                # Convert delta ticks → seconds
                delta_s = mido.tick2second(  # type: ignore[attr-defined]
                    msg.time, mid.ticks_per_beat, tempo_us
                )
                abs_time_s += delta_s
                if msg.type == "set_tempo":
                    tempo_us = msg.tempo
                events.append((abs_time_s, msg))

        events.sort(key=lambda x: x[0])

        # -- Match note-on / note-off pairs --------------------------------
        # pending[(channel, pitch)] = (start_time, velocity)
        pending: dict[Tuple[int, int], Tuple[float, int]] = {}
        parsed: List[ParsedNote] = []

        for abs_time, msg in events:
            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                pending[key] = (abs_time, msg.velocity)

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if key in pending:
                    start_time, velocity = pending.pop(key)
                    parsed.append(
                        ParsedNote(
                            pitch=msg.note,
                            start_time=start_time,
                            end_time=abs_time,
                            velocity=velocity,
                            channel=msg.channel,
                        )
                    )

        # Close any notes that were never explicitly closed
        for (channel, pitch), (start_time, velocity) in pending.items():
            logger.debug(
                "Unclosed note ch=%d pitch=%d at %.3fs — forcing end",
                channel,
                pitch,
                start_time,
            )
            parsed.append(
                ParsedNote(
                    pitch=pitch,
                    start_time=start_time,
                    end_time=start_time + 0.25,  # default quarter-second duration
                    velocity=velocity,
                    channel=channel,
                )
            )

        parsed.sort(key=lambda n: (n.start_time, n.pitch))
        logger.info("Extracted %d notes from MIDI", len(parsed))
        return parsed

    # ------------------------------------------------------------------
    # Internal helpers – MIDI generation (mido path)
    # ------------------------------------------------------------------

    def _notes_to_midi_mido(
        self,
        notes: List[ParsedNote],
        tempo: float,
        ticks_per_beat: int,
    ) -> bytes:
        """Generate MIDI bytes using ``mido`` (preferred path).

        Args:
            notes:         List of :class:`ParsedNote` objects.
            tempo:         Tempo in BPM.
            ticks_per_beat: Ticks per quarter-note.

        Returns:
            Raw MIDI bytes.
        """
        tempo_us = int(mido.bpm2tempo(tempo))  # type: ignore[attr-defined]
        mid = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)  # type: ignore[attr-defined]
        track = mido.MidiTrack()  # type: ignore[attr-defined]
        mid.tracks.append(track)

        track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))  # type: ignore[attr-defined]

        # Build a list of (abs_tick, type, pitch, velocity, channel) events
        raw_events: List[Tuple[int, str, int, int, int]] = []
        for note in notes:
            start_tick = int(
                mido.second2tick(note.start_time, ticks_per_beat, tempo_us)  # type: ignore[attr-defined]
            )
            end_tick = int(
                mido.second2tick(note.end_time, ticks_per_beat, tempo_us)  # type: ignore[attr-defined]
            )
            raw_events.append((start_tick, "note_on", note.pitch, note.velocity, note.channel))
            raw_events.append((end_tick, "note_off", note.pitch, 0, note.channel))

        raw_events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

        prev_tick = 0
        for abs_tick, etype, pitch, vel, ch in raw_events:
            delta = abs_tick - prev_tick
            prev_tick = abs_tick
            track.append(
                mido.Message(etype, note=pitch, velocity=vel, channel=ch, time=delta)  # type: ignore[attr-defined]
            )

        buf = io.BytesIO()
        mid.save(file=buf)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Internal helpers – MIDI generation (pure-Python fallback)
    # ------------------------------------------------------------------

    def _notes_to_midi_fallback(
        self,
        notes: List[ParsedNote],
        tempo: float,
        ticks_per_beat: int,
    ) -> bytes:
        """Generate MIDI bytes without ``mido`` (minimal SMF writer).

        Produces a Type-0 (single-track) Standard MIDI File.

        Args:
            notes:         List of :class:`ParsedNote` objects.
            tempo:         Tempo in BPM.
            ticks_per_beat: Ticks per quarter-note.

        Returns:
            Raw MIDI bytes.
        """
        tempo_us = int(60_000_000 / tempo)
        seconds_per_tick = (tempo_us / 1_000_000) / ticks_per_beat

        def seconds_to_ticks(s: float) -> int:
            return max(0, int(round(s / seconds_per_tick)))

        def encode_vlq(value: int) -> bytes:
            """Encode an integer as a MIDI variable-length quantity."""
            buf: List[int] = []
            buf.append(value & 0x7F)
            value >>= 7
            while value:
                buf.append((value & 0x7F) | 0x80)
                value >>= 7
            buf.reverse()
            return bytes(buf)

        # Tempo meta event: FF 51 03 tttttt
        tempo_event = (
            b"\x00"  # delta = 0
            b"\xff\x51\x03"
            + struct.pack(">I", tempo_us)[1:]  # 3 bytes big-endian
        )

        raw_events: List[Tuple[int, bytes]] = []
        for note in notes:
            st = seconds_to_ticks(note.start_time)
            et = seconds_to_ticks(note.end_time)
            # note-on
            raw_events.append(
                (st, bytes([0x90 | (note.channel & 0x0F), note.pitch & 0x7F, note.velocity & 0x7F]))
            )
            # note-off
            raw_events.append(
                (et, bytes([0x80 | (note.channel & 0x0F), note.pitch & 0x7F, 0x00]))
            )

        raw_events.sort(key=lambda e: e[0])

        track_data = bytearray(tempo_event)
        prev_tick = 0
        for abs_tick, msg_bytes in raw_events:
            delta = abs_tick - prev_tick
            prev_tick = abs_tick
            track_data += encode_vlq(delta) + msg_bytes

        # End-of-track meta event
        track_data += b"\x00\xff\x2f\x00"

        # SMF header chunk
        header = struct.pack(">4sI3H", b"MThd", 6, 0, 1, ticks_per_beat)
        # Track chunk
        track_chunk = (
            b"MTrk"
            + struct.pack(">I", len(track_data))
            + bytes(track_data)
        )

        return header + track_chunk


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _require_mido() -> None:
    """Raise :class:`ImportError` if ``mido`` is not installed."""
    if not _MIDO_AVAILABLE:
        raise ImportError(
            "The 'mido' library is required for MIDI parsing. "
            "Install it with:  pip install mido"
        )
