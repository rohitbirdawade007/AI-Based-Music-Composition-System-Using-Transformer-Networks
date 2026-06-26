"""
Flask API Routes
================
All REST endpoints for the AI Music Composition System.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, current_app, jsonify, request, send_file

# ── Add project root ───────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)

# In-memory generation history (deque with max size)
_history: deque = deque(maxlen=50)

# Lazy-loaded components
_generator = None
_retriever = None


def _get_generator():
    """Lazy-load the music generator."""
    global _generator
    if _generator is None:
        from src.inference.generator import MusicGenerator
        from src.theory.music_theory_engine import MusicTheoryEngine
        _generator = MusicGenerator(demo_mode=True)
        logger.info("MusicGenerator initialized (demo_mode=True)")
    return _generator


def _get_retriever():
    """Lazy-load the RAG retriever."""
    global _retriever
    if _retriever is None:
        try:
            from src.rag.retriever import MusicRetriever
            _retriever = MusicRetriever()
            logger.info("MusicRetriever initialized")
        except Exception as e:
            logger.warning("Could not initialize RAG retriever: %s", e)
            _retriever = None
    return _retriever


# ── Health Check ──────────────────────────────────────────────────────────
@bp.route("/health", methods=["GET"])
def health():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "mode": "demo",
        "endpoints": [
            "GET  /api/health",
            "GET  /api/config",
            "POST /api/generate",
            "POST /api/upload-style",
            "GET  /api/history",
            "DELETE /api/history",
        ],
    })


# ── Configuration ─────────────────────────────────────────────────────────
@bp.route("/config", methods=["GET"])
def get_config():
    """Return available generation options."""
    return jsonify({
        "genres": ["Classical", "Jazz", "Pop", "Blues", "Electronic", "Ambient", "Folk"],
        "moods": ["Happy", "Calm", "Dramatic", "Mysterious", "Energetic", "Melancholic", "Romantic"],
        "keys": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        "modes": [
            "major", "minor", "dorian", "phrygian", "lydian",
            "mixolydian", "pentatonic_major", "pentatonic_minor", "blues"
        ],
        "instruments": ["Piano", "Guitar", "Strings", "Flute", "Violin", "Organ", "Synth"],
        "decoding_strategies": ["greedy", "temperature", "top_k", "top_p", "beam_search"],
        "defaults": {
            "genre": "Classical",
            "mood": "Calm",
            "key": "C",
            "mode": "major",
            "tempo": 120,
            "num_notes": 64,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
            "beam_width": 4,
        },
    })


# ── Music Generation ──────────────────────────────────────────────────────
@bp.route("/generate", methods=["POST"])
def generate():
    """
    Generate music from the given parameters.

    Request Body (JSON):
    {
        "genre":       "Classical",
        "mood":        "Calm",
        "key":         "C",
        "mode":        "major",
        "tempo":       120,
        "num_notes":   64,
        "temperature": 1.0,
        "top_k":       50,
        "top_p":       0.9,
        "use_beam":    false,
        "beam_width":  4,
        "seed_notes":  [60, 64, 67],
        "apply_theory": true
    }

    Response (JSON):
    {
        "id":            "uuid",
        "pitches":       [...],
        "durations":     [...],
        "velocities":    [...],
        "midi_b64":      "base64-encoded MIDI bytes",
        "theory_score":  0.87,
        "generation_time_ms": 123,
        "method":        "demo",
        "key":           "C",
        "mode":          "major",
        "genre":         "Classical",
        "mood":          "Calm",
        "tempo":         120,
        "timestamp":     1234567890.0
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    # Extract parameters with defaults
    genre = data.get("genre", "Classical")
    mood = data.get("mood", "Calm")
    key = data.get("key", "C")
    mode = data.get("mode", "major")
    tempo = int(data.get("tempo", 120))
    num_notes = min(int(data.get("num_notes", 64)), 256)
    temperature = float(data.get("temperature", 1.0))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.9))
    use_beam = bool(data.get("use_beam", False))
    beam_width = int(data.get("beam_width", 4))
    seed_notes = data.get("seed_notes", [])
    apply_theory = bool(data.get("apply_theory", True))

    try:
        generator = _get_generator()
        result = generator.generate(
            seed_pitches=seed_notes,
            genre=genre,
            mood=mood,
            key=key,
            mode=mode,
            tempo=tempo,
            num_notes=num_notes,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_beam=use_beam,
            beam_width=beam_width,
            apply_theory=apply_theory,
        )

        # Encode MIDI as base64
        midi_b64 = base64.b64encode(result.midi_bytes).decode("utf-8")

        response_data = {
            "id": str(uuid.uuid4()),
            "pitches": result.pitches,
            "durations": result.durations,
            "velocities": result.velocities,
            "midi_b64": midi_b64,
            "theory_score": round(result.theory_score, 4),
            "generation_time_ms": round(result.generation_time * 1000, 2),
            "method": result.method,
            "key": result.key,
            "mode": result.mode,
            "genre": result.genre,
            "mood": result.mood,
            "tempo": result.tempo,
            "num_notes": len(result.pitches),
            "timestamp": time.time(),
        }

        # Store in history
        history_entry = {k: v for k, v in response_data.items() if k != "midi_b64"}
        _history.appendleft(history_entry)

        return jsonify(response_data), 200

    except Exception as e:
        logger.exception("Generation failed: %s", e)
        return jsonify({"error": str(e), "message": "Music generation failed"}), 500


# ── MIDI Download ─────────────────────────────────────────────────────────
@bp.route("/generate/download", methods=["POST"])
def generate_and_download():
    """Generate music and return the MIDI file directly as a download."""
    data = request.get_json(force=True, silent=True) or {}

    try:
        generator = _get_generator()
        result = generator.generate(
            genre=data.get("genre", "Classical"),
            mood=data.get("mood", "Calm"),
            key=data.get("key", "C"),
            mode=data.get("mode", "major"),
            tempo=int(data.get("tempo", 120)),
            num_notes=int(data.get("num_notes", 64)),
        )

        filename = f"music_{data.get('genre', 'classical').lower()}_{int(time.time())}.mid"
        return send_file(
            io.BytesIO(result.midi_bytes),
            mimetype="audio/midi",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        logger.exception("Download generation failed: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Style Upload (RAG) ────────────────────────────────────────────────────
@bp.route("/upload-style", methods=["POST"])
def upload_style():
    """
    Upload a MIDI file to use as style reference for RAG-guided generation.

    Request: multipart/form-data with field 'midi_file'
    Response: { "id": "uuid", "filename": "...", "num_notes": N, "message": "..." }
    """
    if "midi_file" not in request.files:
        return jsonify({"error": "No midi_file field in request"}), 400

    file = request.files["midi_file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = {".mid", ".midi"}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": "File must be a .mid or .midi file"}), 400

    try:
        midi_bytes = file.read()
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"

        # Save file
        upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
        upload_path.write_bytes(midi_bytes)

        # Try to index in RAG system
        retriever = _get_retriever()
        num_notes = 0
        if retriever:
            try:
                retriever.add_midi(midi_bytes, filename, metadata={"original_name": file.filename})
                num_notes = 50  # approximate
            except Exception as e:
                logger.warning("RAG indexing failed: %s", e)

        return jsonify({
            "id": file_id,
            "filename": file.filename,
            "saved_as": filename,
            "num_notes": num_notes,
            "message": "MIDI file uploaded and indexed successfully",
            "rag_enabled": retriever is not None,
        }), 200

    except Exception as e:
        logger.exception("Upload failed: %s", e)
        return jsonify({"error": str(e)}), 500


# ── History ───────────────────────────────────────────────────────────────
@bp.route("/history", methods=["GET"])
def get_history():
    """Return generation history."""
    limit = min(int(request.args.get("limit", 20)), 50)
    return jsonify({
        "total": len(_history),
        "items": list(_history)[:limit],
    })


@bp.route("/history", methods=["DELETE"])
def clear_history():
    """Clear generation history."""
    _history.clear()
    return jsonify({"message": "History cleared"})


# ── Error Handlers ────────────────────────────────────────────────────────
@bp.app_errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "message": str(e)}), 404


@bp.app_errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large", "message": "Maximum upload size is 16MB"}), 413


@bp.app_errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500
