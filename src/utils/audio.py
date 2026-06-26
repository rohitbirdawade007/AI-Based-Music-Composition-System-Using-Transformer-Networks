"""
Audio Utilities Module
======================
Provides the ``AudioUtils`` class with helpers for converting MIDI data into
portable HTML players, base64 data URIs, and ABC music notation.

No audio playback libraries are required at the Python level; the HTML player
uses a JavaScript-based MIDI synthesizer (MIDI.js-inspired inline approach)
that runs entirely in the browser.

Dependencies (all optional — graceful degradation):
  - mido  (pip install mido)  — for MIDI introspection
  - numpy

Usage:
    from src.utils.audio import AudioUtils

    util = AudioUtils()
    html = util.generate_audio_html(midi_bytes, title="My Composition")
    with open("player.html", "w") as f:
        f.write(html)
"""

from __future__ import annotations

import base64
import html as html_lib
import logging
import math
import struct
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional mido import for MIDI introspection
# ---------------------------------------------------------------------------
try:
    import mido  # type: ignore
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    mido = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Note / scale helpers
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

_ABC_DURATIONS: List[Tuple[float, str]] = [
    # (seconds_per_beat_fraction, abc_length_token)  — ordered longest-first
    (4.0,   "4"),
    (3.0,   "3"),
    (2.0,   "2"),
    (1.0,   ""),
    (0.75,  "3/4"),
    (0.5,   "/2"),
    (0.375, "3/8"),
    (0.25,  "/4"),
    (0.125, "/8"),
]

_ABC_PITCH_MAP = {
    # Octave 4 (lower-case in ABC)
    60: "c", 61: "^c", 62: "d", 63: "^d", 64: "e", 65: "f",
    66: "^f", 67: "g", 68: "^g", 69: "a", 70: "^a", 71: "b",
    # Octave 5 (lower-case with ')
    72: "c'", 73: "^c'", 74: "d'", 75: "^d'", 76: "e'", 77: "f'",
    78: "^f'", 79: "g'", 80: "^g'", 81: "a'", 82: "^a'", 83: "b'",
    # Octave 3 (upper-case)
    48: "C,", 49: "^C,", 50: "D,", 51: "^D,", 52: "E,", 53: "F,",
    54: "^F,", 55: "G,", 56: "^G,", 57: "A,", 58: "^A,", 59: "B,",
    # Octave 6
    84: 'c"', 85: '^c"', 86: 'd"', 87: '^d"', 88: 'e"', 89: 'f"',
}

# ---------------------------------------------------------------------------
# AudioUtils
# ---------------------------------------------------------------------------

class AudioUtils:
    """
    Audio and MIDI utility helpers.

    Methods are designed to be stateless (no instance variables required)
    so the class can be used as a namespace.  An instance is still preferred
    so that future configuration (e.g. MIDI channel, instrument) can be added.
    """

    # ------------------------------------------------------------------
    # Base64 encoding
    # ------------------------------------------------------------------

    @staticmethod
    def midi_bytes_to_base64(midi_bytes: bytes) -> str:
        """
        Encode raw MIDI bytes as a base64 string suitable for a data URI.

        Parameters
        ----------
        midi_bytes : bytes
            Binary MIDI file content.

        Returns
        -------
        str
            Base64-encoded string (no line breaks, no padding stripped).

        Example
        -------
        >>> b64 = AudioUtils.midi_bytes_to_base64(midi_bytes)
        >>> data_uri = f"data:audio/midi;base64,{b64}"
        """
        if not midi_bytes:
            logger.warning("midi_bytes_to_base64: received empty bytes.")
            return ""
        return base64.b64encode(midi_bytes).decode("ascii")

    # ------------------------------------------------------------------
    # HTML Player
    # ------------------------------------------------------------------

    @staticmethod
    def create_html_player(midi_bytes: bytes, title: str = "Generated Music") -> str:
        """
        Generate a minimal self-contained HTML snippet with a MIDI download
        link and an embedded ``<audio>`` element.

        Because browser support for MIDI audio via ``<audio>`` tags is limited,
        this method produces:
          1. A guaranteed-working **download link** for the MIDI file.
          2. A human-friendly notice about browser MIDI support.
          3. A placeholder for the richer ``generate_audio_html`` player.

        For a full interactive player, use :meth:`generate_audio_html`.

        Parameters
        ----------
        midi_bytes : bytes
            Binary MIDI file content.
        title : str
            Title displayed in the player.

        Returns
        -------
        str
            HTML string with an embedded data URI download link.
        """
        b64 = AudioUtils.midi_bytes_to_base64(midi_bytes)
        safe_title = html_lib.escape(title)
        data_uri = f"data:audio/midi;base64,{b64}"

        return f"""
<div style="
    font-family: 'Segoe UI', sans-serif;
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:16px; max-width:480px;
    color:#c9d1d9;">
  <h3 style="margin:0 0 10px; color:#58a6ff;">🎵 {safe_title}</h3>
  <p style="font-size:13px; color:#8b949e; margin:0 0 12px;">
    MIDI files require a browser plugin or dedicated player.
    Use the download button below to open in your preferred MIDI player.
  </p>
  <a href="{data_uri}" download="{safe_title.replace(' ','_')}.mid"
     style="display:inline-block; background:#238636; color:#fff;
            border:none; border-radius:6px; padding:8px 16px;
            font-size:13px; text-decoration:none; cursor:pointer;">
    ⬇ Download MIDI
  </a>
</div>
""".strip()

    def generate_audio_html(
        self,
        midi_bytes: bytes,
        title: str = "Generated Music",
    ) -> str:
        """
        Generate a complete, self-contained HTML page with an interactive
        MIDI player.

        The player includes:
          - **Play / Pause / Stop** buttons with SVG icons
          - A **progress bar** that advances as notes play
          - A **mini piano visualisation** that highlights active notes
          - A **download button** that always works (data URI)
          - A JavaScript synthesiser using the Web Audio API (``OscillatorNode``)
            that can render basic MIDI note-on/note-off events

        Browser Compatibility
        ---------------------
        Uses the Web Audio API (supported in Chrome, Firefox, Safari, Edge ≥ 79).
        The MIDI is parsed inline from the embedded base64 data; no server-side
        component or plugin is required.

        Parameters
        ----------
        midi_bytes : bytes
            Binary MIDI file content.
        title : str
            Title displayed in the player header.

        Returns
        -------
        str
            A complete ``<!DOCTYPE html>`` HTML document as a string.
        """
        b64 = self.midi_bytes_to_base64(midi_bytes)
        safe_title = html_lib.escape(title)

        # Extract basic MIDI info for display
        info = self.get_midi_info(midi_bytes)
        duration_str = f"{info.get('duration_seconds', 0):.1f}s"
        num_notes_str = str(info.get("num_notes", "?"))
        tempo_str = str(info.get("tempo", 120))

        return _HTML_PLAYER_TEMPLATE.format(
            title=safe_title,
            midi_b64=b64,
            duration=duration_str,
            num_notes=num_notes_str,
            tempo=tempo_str,
        )

    # ------------------------------------------------------------------
    # MIDI info extraction
    # ------------------------------------------------------------------

    @staticmethod
    def get_midi_info(midi_bytes: bytes) -> Dict[str, object]:
        """
        Extract summary information from a binary MIDI file.

        Uses *mido* when available; otherwise falls back to a minimal
        pure-Python MIDI header parser.

        Parameters
        ----------
        midi_bytes : bytes
            Binary MIDI file content.

        Returns
        -------
        dict
            Keys:
            - ``num_notes``       (int)   — total note-on events
            - ``duration_seconds``(float) — approximate total duration
            - ``tempo``           (int)   — BPM (from first tempo event)
            - ``time_signature``  (str)   — e.g. ``'4/4'``
            - ``num_tracks``      (int)   — number of MIDI tracks
        """
        if not midi_bytes:
            return {"num_notes": 0, "duration_seconds": 0.0,
                    "tempo": 120, "time_signature": "4/4", "num_tracks": 0}

        if MIDO_AVAILABLE:
            return _parse_midi_mido(midi_bytes)
        else:
            return _parse_midi_minimal(midi_bytes)

    # ------------------------------------------------------------------
    # ABC Notation
    # ------------------------------------------------------------------

    @staticmethod
    def notes_to_abc_notation(
        pitches: List[int],
        durations: List[float],
        key: str = "C",
        title: str = "Generated Piece",
        tempo: int = 120,
    ) -> str:
        """
        Convert pitch/duration lists to `ABC music notation
        <https://abcnotation.com/>`_.

        ABC notation is a plain-text music format widely supported by
        notation software (MuseScore, abcjs, etc.).

        Parameters
        ----------
        pitches : List[int]
            MIDI pitch values; ``-1`` is a rest.
        durations : List[float]
            Duration in seconds for each event.
        key : str
            Root note (e.g. ``'C'``, ``'G'``).
        title : str
            Piece title embedded in the ABC header.
        tempo : int
            BPM used for the Q: (tempo) header.

        Returns
        -------
        str
            Complete ABC notation string.
        """
        lines: List[str] = [
            "X:1",
            f"T:{title}",
            f"M:4/4",
            f"L:1/8",
            f"Q:1/4={tempo}",
            f"K:{key}",
        ]

        body_tokens: List[str] = []
        beats_per_second = tempo / 60.0
        bar_counter = 0
        beat_accumulator = 0.0

        for pitch, dur in zip(pitches, durations):
            dur_beats = dur * beats_per_second
            abc_token = _duration_to_abc(dur_beats)

            if pitch == -1:
                # Rest
                body_tokens.append(f"z{abc_token}")
            else:
                note_str = _ABC_PITCH_MAP.get(pitch, _fallback_abc_pitch(pitch))
                body_tokens.append(f"{note_str}{abc_token}")

            beat_accumulator += dur_beats
            if beat_accumulator >= 4.0:
                # End of a 4/4 bar
                body_tokens.append("|")
                beat_accumulator = beat_accumulator % 4.0
                bar_counter += 1
                if bar_counter % 4 == 0:
                    body_tokens.append("\n")

        if body_tokens and body_tokens[-1] != "|":
            body_tokens.append("|]")
        elif body_tokens:
            body_tokens[-1] = "|]"

        # Wrap lines at ~70 characters
        body = " ".join(body_tokens)
        lines.append(body)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _duration_to_abc(beats: float) -> str:
    """Convert a beat duration to an ABC notation length token."""
    # Use 1/8 note as base length (L:1/8) → beats * 2 = ABC units
    abc_units = beats * 2
    rounded = round(abc_units * 4) / 4  # round to nearest sixteenth

    if rounded <= 0:
        return "/4"

    # Represent as integer or simple fraction
    numerator = int(round(rounded * 4))
    denominator = 4

    # Simplify
    from math import gcd as _gcd
    g = _gcd(numerator, denominator)
    numerator //= g
    denominator //= g

    if denominator == 1:
        return str(numerator) if numerator != 1 else ""
    return f"{numerator}/{denominator}" if numerator != 1 else f"/{denominator}"


def _fallback_abc_pitch(pitch: int) -> str:
    """Generate an ABC pitch token for pitches not in the lookup table."""
    note_name = _NOTE_NAMES[pitch % 12]
    octave = (pitch // 12) - 1
    # Middle octave (4) → lower-case; above → add ', below → upper-case with ,
    if octave >= 5:
        token = note_name.lower() + "'" * (octave - 5)
    elif octave == 4:
        token = note_name.lower()
    elif octave == 3:
        token = note_name.upper()
    else:
        token = note_name.upper() + "," * (3 - octave)
    return token


def _parse_midi_mido(midi_bytes: bytes) -> Dict[str, object]:
    """Parse MIDI using the *mido* library."""
    import io as _io
    mid = mido.MidiFile(file=_io.BytesIO(midi_bytes))

    tempo_us = 500_000  # default 120 BPM
    time_sig_num, time_sig_den = 4, 4
    num_notes = 0

    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
            elif msg.type == "time_signature":
                time_sig_num = msg.numerator
                time_sig_den = msg.denominator
            elif msg.type == "note_on" and msg.velocity > 0:
                num_notes += 1

    bpm = int(round(60_000_000 / tempo_us)) if tempo_us else 120
    duration_seconds = mid.length

    return {
        "num_notes": num_notes,
        "duration_seconds": round(duration_seconds, 2),
        "tempo": bpm,
        "time_signature": f"{time_sig_num}/{time_sig_den}",
        "num_tracks": len(mid.tracks),
    }


def _parse_midi_minimal(midi_bytes: bytes) -> Dict[str, object]:
    """
    Minimal pure-Python MIDI header parser (no dependencies).

    Reads the MIDI header chunk to extract track count and time division,
    then scans for tempo (FF 51 03) and time-signature (FF 58 04) meta events.
    """
    try:
        if midi_bytes[:4] != b"MThd":
            raise ValueError("Not a valid MIDI file.")

        _length = struct.unpack(">I", midi_bytes[4:8])[0]
        _fmt = struct.unpack(">H", midi_bytes[8:10])[0]
        num_tracks = struct.unpack(">H", midi_bytes[10:12])[0]
        ticks_per_beat = struct.unpack(">H", midi_bytes[12:14])[0]

        tempo_us = 500_000
        time_sig_num, time_sig_den = 4, 4
        note_on_count = 0

        i = 14
        while i < len(midi_bytes) - 8:
            if midi_bytes[i:i+4] == b"MTrk":
                track_length = struct.unpack(">I", midi_bytes[i+4:i+8])[0]
                track_data = midi_bytes[i+8: i+8+track_length]
                # Count note-on (0x9?) events and find meta events
                j = 0
                while j < len(track_data):
                    # Read variable-length delta time
                    _delta, j = _read_var_len(track_data, j)
                    if j >= len(track_data):
                        break
                    status = track_data[j]
                    if status == 0xFF:  # Meta event
                        if j + 2 >= len(track_data):
                            break
                        meta_type = track_data[j+1]
                        meta_len, j2 = _read_var_len(track_data, j+2)
                        meta_data = track_data[j2: j2+meta_len]
                        if meta_type == 0x51 and meta_len == 3:
                            tempo_us = (meta_data[0] << 16 | meta_data[1] << 8
                                        | meta_data[2])
                        elif meta_type == 0x58 and meta_len >= 2:
                            time_sig_num = meta_data[0]
                            time_sig_den = 2 ** meta_data[1]
                        j = j2 + meta_len
                    elif 0x90 <= status <= 0x9F:  # Note on
                        if j + 2 < len(track_data) and track_data[j+2] > 0:
                            note_on_count += 1
                        j += 3
                    elif 0x80 <= status <= 0x8F:  # Note off
                        j += 3
                    elif 0xA0 <= status <= 0xBF:  # Polyphonic/Control
                        j += 3
                    elif 0xC0 <= status <= 0xDF:  # Program/Channel pressure
                        j += 2
                    elif 0xE0 <= status <= 0xEF:  # Pitch bend
                        j += 3
                    else:
                        j += 1
                i += 8 + track_length
            else:
                i += 1

        bpm = int(round(60_000_000 / tempo_us)) if tempo_us else 120
        # Rough duration estimate (note count * default duration)
        seconds_per_beat = tempo_us / 1_000_000
        approx_duration = note_on_count * 0.5 * seconds_per_beat

        return {
            "num_notes": note_on_count,
            "duration_seconds": round(approx_duration, 2),
            "tempo": bpm,
            "time_signature": f"{time_sig_num}/{time_sig_den}",
            "num_tracks": num_tracks,
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("Minimal MIDI parser error: %s", exc)
        return {"num_notes": 0, "duration_seconds": 0.0,
                "tempo": 120, "time_signature": "4/4", "num_tracks": 0}


def _read_var_len(data: bytes, pos: int) -> Tuple[int, int]:
    """Read a MIDI variable-length quantity starting at *pos*."""
    result = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        result = (result << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return result, pos


# ---------------------------------------------------------------------------
# HTML Player Template
# ---------------------------------------------------------------------------

_HTML_PLAYER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    /* ---- Reset & Base ---- */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
      padding: 32px 16px;
    }}

    /* ---- Player Card ---- */
    .player-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      width: 100%;
      max-width: 620px;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }}

    /* ---- Header ---- */
    .player-header {{
      background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
      padding: 20px 24px 16px;
      border-bottom: 1px solid #30363d;
    }}
    .player-header h1 {{
      font-size: 1.1rem;
      font-weight: 600;
      color: #58a6ff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .player-header .meta {{
      margin-top: 6px;
      font-size: 12px;
      color: #8b949e;
      display: flex;
      gap: 16px;
    }}
    .meta span {{ display: flex; align-items: center; gap: 4px; }}

    /* ---- Piano Visualiser ---- */
    .piano-section {{
      padding: 16px 24px 8px;
      border-bottom: 1px solid #21262d;
    }}
    .piano-container {{
      position: relative;
      width: 100%;
      height: 56px;
      overflow-x: auto;
      overflow-y: hidden;
    }}
    .piano {{
      display: flex;
      position: relative;
      height: 56px;
      user-select: none;
    }}
    .key-w {{
      width: 22px;
      height: 56px;
      background: #e8e8e8;
      border: 1px solid #555;
      border-top: none;
      border-radius: 0 0 4px 4px;
      position: relative;
      flex-shrink: 0;
      transition: background 0.05s;
    }}
    .key-w.active {{ background: #58a6ff; }}
    .key-b {{
      width: 14px;
      height: 34px;
      background: #1a1a2e;
      border-radius: 0 0 3px 3px;
      position: absolute;
      z-index: 2;
      border: 1px solid #000;
      top: 0;
      transition: background 0.05s;
    }}
    .key-b.active {{ background: #388bfd; }}

    /* ---- Progress / Transport ---- */
    .transport {{
      padding: 16px 24px;
    }}
    .progress-wrap {{
      background: #21262d;
      border-radius: 4px;
      height: 6px;
      width: 100%;
      margin-bottom: 14px;
      cursor: pointer;
      position: relative;
    }}
    .progress-bar {{
      background: linear-gradient(90deg, #388bfd, #58a6ff);
      height: 100%;
      border-radius: 4px;
      width: 0%;
      transition: width 0.1s linear;
    }}
    .time-display {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: #8b949e;
      margin-bottom: 14px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .btn {{
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 8px;
      color: #c9d1d9;
      cursor: pointer;
      padding: 8px 14px;
      font-size: 13px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s, border-color 0.15s;
      white-space: nowrap;
    }}
    .btn:hover {{ background: #30363d; border-color: #58a6ff; color: #58a6ff; }}
    .btn:active {{ transform: scale(0.97); }}
    .btn.primary {{
      background: #1f6feb;
      border-color: #388bfd;
      color: #fff;
    }}
    .btn.primary:hover {{ background: #388bfd; }}
    .btn.danger {{ border-color: #f85149; color: #f85149; }}
    .btn.danger:hover {{ background: #3d1a19; border-color: #f85149; }}
    .btn-spacer {{ flex: 1; }}

    /* ---- Status ---- */
    .status-bar {{
      padding: 8px 24px 16px;
      font-size: 12px;
      color: #8b949e;
      min-height: 36px;
    }}
    .status-bar .notice {{
      background: #1c2128;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 6px 12px;
      display: none;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }}
    .status-bar .notice.show {{ display: flex; }}
    .notice-icon {{ font-size: 14px; }}
  </style>
</head>
<body>
<div class="player-card">

  <!-- Header -->
  <div class="player-header">
    <h1>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2">
        <path d="M9 18V5l12-2v13"/>
        <circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
      </svg>
      {title}
    </h1>
    <div class="meta">
      <span>🎵 {num_notes} notes</span>
      <span>⏱ {duration}</span>
      <span>♩ {tempo} BPM</span>
    </div>
  </div>

  <!-- Mini Piano -->
  <div class="piano-section">
    <div class="piano-container">
      <div class="piano" id="piano"></div>
    </div>
  </div>

  <!-- Transport Controls -->
  <div class="transport">
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar" id="progressBar"></div>
    </div>
    <div class="time-display">
      <span id="timeCurrent">0:00</span>
      <span id="timeTotal">0:00</span>
    </div>
    <div class="controls">
      <button class="btn primary" id="btnPlay" onclick="playerPlay()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5,3 19,12 5,21"/>
        </svg>
        Play
      </button>
      <button class="btn" id="btnPause" onclick="playerPause()" style="display:none">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
        </svg>
        Pause
      </button>
      <button class="btn danger" onclick="playerStop()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
        </svg>
        Stop
      </button>
      <div class="btn-spacer"></div>
      <a class="btn" id="btnDownload"
         href="data:audio/midi;base64,{midi_b64}"
         download="{title}.mid">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download MIDI
      </a>
    </div>
  </div>

  <!-- Status -->
  <div class="status-bar">
    <div class="notice" id="noticeNoAudio">
      <span class="notice-icon">ℹ️</span>
      <span>
        Web Audio playback uses a basic sine-wave synthesiser.
        For best results, download the MIDI file and open it in a DAW or media player.
      </span>
    </div>
  </div>

</div>

<script>
// ============================================================
// Embedded MIDI data (base64)
// ============================================================
const MIDI_B64 = "{midi_b64}";

// ============================================================
// Minimal MIDI parser (handles Type-0 and Type-1 MIDI files)
// ============================================================
function base64ToBytes(b64) {{
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}}

function readUint32BE(arr, pos) {{
  return ((arr[pos] << 24) | (arr[pos+1] << 16) | (arr[pos+2] << 8) | arr[pos+3]) >>> 0;
}}

function readUint16BE(arr, pos) {{
  return (arr[pos] << 8) | arr[pos+1];
}}

function readVarLen(arr, pos) {{
  let value = 0, b;
  do {{
    b = arr[pos++];
    value = (value << 7) | (b & 0x7F);
  }} while (b & 0x80);
  return {{ value, pos }};
}}

function parseMidi(bytes) {{
  if (String.fromCharCode(...bytes.slice(0,4)) !== 'MThd')
    return null;
  const format = readUint16BE(bytes, 8);
  const numTracks = readUint16BE(bytes, 10);
  const ticksPerBeat = readUint16BE(bytes, 12);

  let pos = 14;
  const tracks = [];

  for (let t = 0; t < numTracks; t++) {{
    if (pos + 8 > bytes.length) break;
    const hdr = String.fromCharCode(...bytes.slice(pos, pos+4));
    if (hdr !== 'MTrk') {{ pos += 1; t--; continue; }}
    const trackLen = readUint32BE(bytes, pos+4);
    const trackEnd = pos + 8 + trackLen;
    const events = [];
    let trkPos = pos + 8;
    let absoluteTick = 0;
    let runningStatus = 0;

    while (trkPos < trackEnd) {{
      const dt = readVarLen(bytes, trkPos);
      trkPos = dt.pos;
      absoluteTick += dt.value;

      let status = bytes[trkPos];
      if (status & 0x80) {{
        runningStatus = status;
        trkPos++;
      }} else {{
        status = runningStatus;
      }}

      const type = status >> 4;
      const ch = status & 0x0F;

      if (status === 0xFF) {{
        // Meta event
        const metaType = bytes[trkPos++];
        const metaLen = readVarLen(bytes, trkPos);
        trkPos = metaLen.pos;
        const metaData = bytes.slice(trkPos, trkPos + metaLen.value);
        trkPos += metaLen.value;
        if (metaType === 0x51) {{
          const tempoUs = (metaData[0] << 16) | (metaData[1] << 8) | metaData[2];
          events.push({{ tick: absoluteTick, type: 'tempo', tempoUs }});
        }} else if (metaType === 0x2F) {{
          break; // End of track
        }}
      }} else if (type === 0x9) {{
        const note = bytes[trkPos++];
        const vel = bytes[trkPos++];
        events.push({{ tick: absoluteTick, type: vel > 0 ? 'noteOn' : 'noteOff', ch, note, vel }});
      }} else if (type === 0x8) {{
        const note = bytes[trkPos++];
        const vel = bytes[trkPos++];
        events.push({{ tick: absoluteTick, type: 'noteOff', ch, note, vel }});
      }} else if (type === 0xA) {{
        trkPos += 2;
      }} else if (type === 0xB) {{
        trkPos += 2;
      }} else if (type === 0xC || type === 0xD) {{
        trkPos += 1;
      }} else if (type === 0xE) {{
        trkPos += 2;
      }} else {{
        trkPos += 1;
      }}
    }}
    tracks.push(events);
    pos = trackEnd;
  }}
  return {{ format, numTracks, ticksPerBeat, tracks }};
}}

// Merge and sort all events from all tracks by tick
function mergeEvents(midi) {{
  let allEvents = [];
  for (const track of midi.tracks) {{
    for (const ev of track) {{
      allEvents.push(ev);
    }}
  }}
  allEvents.sort((a, b) => a.tick - b.tick);
  return allEvents;
}}

// Convert ticks to seconds with tempo map
function buildTimeline(events, ticksPerBeat) {{
  let tempoUs = 500000; // 120 BPM default
  let lastTick = 0;
  let lastTime = 0;
  const timeline = [];

  for (const ev of events) {{
    const deltaTick = ev.tick - lastTick;
    const deltaTime = (deltaTick / ticksPerBeat) * (tempoUs / 1e6);
    const time = lastTime + deltaTime;

    if (ev.type === 'tempo') {{
      lastTick = ev.tick;
      lastTime = time;
      tempoUs = ev.tempoUs;
    }}

    timeline.push({{ ...ev, time }});
  }}
  return timeline;
}}

// ============================================================
// Web Audio Synthesiser
// ============================================================
let audioCtx = null;
let scheduledSources = [];
let playerState = 'stopped'; // 'stopped' | 'playing' | 'paused'
let startOffset = 0;
let startAudioTime = 0;
let totalDuration = 0;
let animFrameId = null;
let midiTimeline = [];
let ticksPerBeat = 480;

const NOTE_FREQ_A4 = 440.0;
function midiToFreq(note) {{
  return NOTE_FREQ_A4 * Math.pow(2, (note - 69) / 12);
}}

function getOrCreateAudioCtx() {{
  if (!audioCtx) {{
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }}
  if (audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}}

function schedulePlayback(offset) {{
  const ctx = getOrCreateAudioCtx();
  const now = ctx.currentTime;
  startAudioTime = now - offset;

  // Cancel any already-scheduled sources
  scheduledSources.forEach(n => {{ try {{ n.stop(0); }} catch(e){{}} }});
  scheduledSources = [];
  activeKeys.clear();

  for (const ev of midiTimeline) {{
    if (ev.time < offset - 0.01) continue;
    if (ev.type !== 'noteOn') continue;

    // Find the matching noteOff
    let duration = 0.5; // default
    for (const off of midiTimeline) {{
      if (off.type === 'noteOff' && off.note === ev.note
          && off.ch === ev.ch && off.time > ev.time) {{
        duration = off.time - ev.time;
        break;
      }}
    }}

    const startAt = now + (ev.time - offset);
    if (startAt < now - 0.01) continue;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = midiToFreq(ev.note);
    gain.gain.setValueAtTime(0, startAt);
    gain.gain.linearRampToValueAtTime((ev.vel / 127) * 0.25, startAt + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, startAt + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(startAt);
    osc.stop(startAt + duration + 0.02);
    scheduledSources.push(osc);

    // Schedule piano key highlight
    setTimeout(() => {{
      highlightKey(ev.note, true);
      setTimeout(() => highlightKey(ev.note, false), duration * 1000);
    }}, Math.max(0, (startAt - ctx.currentTime) * 1000));
  }}
}}

function playerPlay() {{
  if (!midiTimeline.length) return;
  if (playerState === 'paused') {{
    schedulePlayback(startOffset);
    playerState = 'playing';
    updateUI();
    animateProgress();
    return;
  }}
  if (playerState === 'playing') return;

  startOffset = 0;
  schedulePlayback(0);
  playerState = 'playing';
  updateUI();
  animateProgress();
  document.getElementById('noticeNoAudio').classList.add('show');
}}

function playerPause() {{
  if (playerState !== 'playing') return;
  const ctx = getOrCreateAudioCtx();
  startOffset += ctx.currentTime - startAudioTime;
  scheduledSources.forEach(n => {{ try {{ n.stop(0); }} catch(e){{}} }});
  scheduledSources = [];
  cancelAnimationFrame(animFrameId);
  playerState = 'paused';
  updateUI();
}}

function playerStop() {{
  scheduledSources.forEach(n => {{ try {{ n.stop(0); }} catch(e){{}} }});
  scheduledSources = [];
  cancelAnimationFrame(animFrameId);
  startOffset = 0;
  playerState = 'stopped';
  activeKeys.clear();
  renderPiano();
  document.getElementById('progressBar').style.width = '0%';
  document.getElementById('timeCurrent').textContent = '0:00';
  updateUI();
}}

function animateProgress() {{
  const ctx = getOrCreateAudioCtx();
  const elapsed = startOffset + (ctx.currentTime - startAudioTime);
  const pct = Math.min(100, (elapsed / totalDuration) * 100);
  document.getElementById('progressBar').style.width = pct + '%';
  document.getElementById('timeCurrent').textContent = formatTime(elapsed);

  if (elapsed >= totalDuration) {{
    playerStop();
    return;
  }}
  animFrameId = requestAnimationFrame(animateProgress);
}}

function updateUI() {{
  const isPlaying = playerState === 'playing';
  const isPaused = playerState === 'paused';
  document.getElementById('btnPlay').style.display = (isPlaying) ? 'none' : 'flex';
  document.getElementById('btnPause').style.display = (isPlaying) ? 'flex' : 'none';
}}

function formatTime(secs) {{
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return m + ':' + String(s).padStart(2, '0');
}}

// ============================================================
// Piano Keyboard Rendering
// ============================================================
const activeKeys = new Set();
const WHITE_NOTES = [0,2,4,5,7,9,11]; // C D E F G A B
const BLACK_NOTES = [1,3,6,8,10];     // C# D# F# G# A#
const BLACK_OFFSET = [1,2,null,4,5,6]; // offset within white-key group

function buildPiano() {{
  const container = document.getElementById('piano');
  container.innerHTML = '';
  const NUM_OCTAVES = 4;
  const START_OCTAVE = 3;

  let whiteIdx = 0;
  for (let oct = START_OCTAVE; oct < START_OCTAVE + NUM_OCTAVES; oct++) {{
    for (let n = 0; n < 7; n++) {{
      const pc = WHITE_NOTES[n];
      const midi = (oct + 1) * 12 + pc;
      const key = document.createElement('div');
      key.className = 'key-w';
      key.dataset.midi = midi;
      container.appendChild(key);
    }}
    // Add black keys
    const whites = container.querySelectorAll('.key-w');
    const groupStart = (oct - START_OCTAVE) * 7;
    const blackOffsets = [0.75, 1.75, null, 3.75, 4.75, 5.75]; // fractions of white key width
    for (let b = 0; b < 6; b++) {{
      if (blackOffsets[b] === null) continue;
      const pc = BLACK_NOTES[b < 2 ? b : b - 1];
      const midi = (oct + 1) * 12 + [1,3,6,8,10][b < 2 ? b : b - 1];
      const key = document.createElement('div');
      key.className = 'key-b';
      key.dataset.midi = midi;
      const leftPx = (groupStart + blackOffsets[b]) * 22 + 8;
      key.style.left = leftPx + 'px';
      container.appendChild(key);
    }}
  }}
}}

function highlightKey(midiNote, on) {{
  if (on) activeKeys.add(midiNote);
  else activeKeys.delete(midiNote);
  const el = document.querySelector(`[data-midi="${{midiNote}}"]`);
  if (el) {{
    if (on) el.classList.add('active');
    else el.classList.remove('active');
  }}
}}

function renderPiano() {{
  document.querySelectorAll('.key-w, .key-b').forEach(k => {{
    k.classList.toggle('active', activeKeys.has(parseInt(k.dataset.midi)));
  }});
}}

// ============================================================
// Initialisation
// ============================================================
function init() {{
  buildPiano();

  try {{
    const bytes = base64ToBytes(MIDI_B64);
    const midi = parseMidi(bytes);
    if (midi) {{
      const events = mergeEvents(midi);
      midiTimeline = buildTimeline(events, midi.ticksPerBeat);
      totalDuration = Math.max(...midiTimeline.map(e => e.time));
      document.getElementById('timeTotal').textContent = formatTime(totalDuration);
    }}
  }} catch(e) {{
    console.warn('MIDI parse error:', e);
  }}
}}
init();
</script>
</body>
</html>"""
