"""
Flask REST API
==============
Provides REST endpoints for the AI Music Composition System:
  POST /api/generate        – generate music
  POST /api/upload-style    – upload MIDI for RAG
  GET  /api/history         – generation history
  GET  /api/health          – health check
  GET  /api/config          – available options

Run
---
  python api/app.py
  # or
  flask --app api/app.py run --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── Add project root to path ───────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    CORS(app, origins="*")

    # Config
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
    app.config["UPLOAD_FOLDER"] = str(project_root / "data" / "uploads")
    app.config["OUTPUT_FOLDER"] = str(project_root / "output")
    app.config["MAX_HISTORY"] = 50

    # Create necessary directories
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)

    # Register routes
    from api.routes import bp
    app.register_blueprint(bp, url_prefix="/api")

    logger.info("Flask app created. API available at http://localhost:5000/api/")
    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Flask server on %s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)
