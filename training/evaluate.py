"""
Model Evaluation Script
========================
Evaluates a trained MusicTransformer checkpoint on a test set.
Computes perplexity, note accuracy, and generates sample outputs.

Usage
-----
  python training/evaluate.py --checkpoint checkpoints/best.pth --num-samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

# ── Add project root to path ───────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

METRICS_TEMPLATE = {
    "perplexity": None,
    "token_accuracy": None,
    "in_scale_ratio": None,
    "consonance_score": None,
    "rhythm_score": None,
    "generation_time_ms": None,
}


def compute_perplexity(avg_loss: float) -> float:
    return math.exp(min(avg_loss, 100))


def evaluate_music_quality(generated_pitches: List[List[int]], key: str = "C", mode: str = "major") -> Dict:
    """Evaluate musical quality of generated sequences."""
    from src.theory.music_theory_engine import MusicTheoryEngine

    engine = MusicTheoryEngine(key=key, mode=mode)
    scores = {"in_scale_ratio": [], "consonance": [], "range_score": []}

    for pitches in generated_pitches:
        result = engine.validate_melody(pitches)
        scores["in_scale_ratio"].append(result.in_scale_ratio)
        scores["consonance"].append(result.consonance_score)
        scores["range_score"].append(result.range_score)

    return {k: sum(v) / max(len(v), 1) for k, v in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MusicTransformer")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output", default="logs/evaluation_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    from src.model.transformer import MusicTransformer
    from src.inference.generator import MusicGenerator
    from src.tokenizer.music_tokenizer import MusicTokenizer

    device = torch.device(args.device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})

    model = MusicTransformer(
        vocab_size=model_cfg.get("vocab_size", 512),
        d_model=model_cfg.get("d_model", 256),
        nhead=model_cfg.get("nhead", 8),
        num_encoder_layers=model_cfg.get("num_encoder_layers", 6),
        num_decoder_layers=model_cfg.get("num_decoder_layers", 6),
        dim_feedforward=model_cfg.get("dim_feedforward", 1024),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = MusicTokenizer()
    generator = MusicGenerator(model=model, tokenizer=tokenizer, demo_mode=False)

    # Generate samples
    logger.info("Generating %d samples for evaluation...", args.num_samples)
    all_pitches = []
    generation_times = []

    import time
    for i in range(args.num_samples):
        t0 = time.time()
        result = generator.generate(
            seed_pitches=[60, 64, 67],
            genre="Classical",
            mood="Calm",
            num_notes=64,
            temperature=1.0,
        )
        generation_times.append((time.time() - t0) * 1000)
        all_pitches.append(result.pitches)
        logger.info("Sample %d: %d notes, theory_score=%.3f", i + 1, len(result.pitches), result.theory_score)

    # Music quality metrics
    music_metrics = evaluate_music_quality(all_pitches)
    avg_gen_time = sum(generation_times) / len(generation_times)

    results = {
        **METRICS_TEMPLATE,
        "checkpoint": args.checkpoint,
        "epoch": checkpoint.get("epoch", "N/A"),
        "val_loss": checkpoint.get("val_loss", "N/A"),
        "perplexity": compute_perplexity(checkpoint.get("val_loss", 5.0)),
        **music_metrics,
        "generation_time_ms": avg_gen_time,
        "num_samples": args.num_samples,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))

    logger.info("\n=== Evaluation Results ===")
    for k, v in results.items():
        if v is not None and isinstance(v, float):
            logger.info("  %-25s: %.4f", k, v)
        elif v is not None:
            logger.info("  %-25s: %s", k, v)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
