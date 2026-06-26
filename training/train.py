"""
Training Script
================
Full training loop for the MusicTransformer model with:
  - Mixed precision (AMP)
  - Warmup + cosine LR scheduling
  - Gradient clipping
  - Checkpoint saving
  - TensorBoard logging
  - Early stopping

Usage
-----
  python training/train.py --config training/config_training.yaml
  python training/train.py --config training/config_training.yaml --resume checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ── Add project root to path ───────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "train.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    """Warmup + cosine annealing scheduler."""
    import torch
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(cosine, min_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    config: dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
) -> str:
    import torch
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )
    logger.info("Checkpoint saved → %s (val_loss=%.4f)", path, val_loss)
    return str(path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    import torch
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.info(
        "Loaded checkpoint from %s (epoch=%d, val_loss=%.4f)",
        path, checkpoint["epoch"], checkpoint["val_loss"],
    )
    return checkpoint["epoch"], checkpoint["val_loss"]


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    gradient_clip: float = 1.0,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, avg_accuracy)."""
    import torch
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        src = batch["src"].to(device)          # (batch, src_len)
        tgt = batch["tgt"].to(device)          # (batch, tgt_len)
        tgt_input = tgt[:, :-1].transpose(0, 1)  # (tgt_len-1, batch)
        tgt_output = tgt[:, 1:].contiguous()     # (batch, tgt_len-1)
        src = src.transpose(0, 1)                # (src_len, batch)

        src_pad_mask = (src == 0).transpose(0, 1)
        tgt_pad_mask = (tgt_input == 0).transpose(0, 1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask)
            # logits: (tgt_len-1, batch, vocab)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        mask = tgt_flat != 0
        total_loss += loss.item() * mask.sum().item()
        preds = logits_flat.argmax(dim=-1)
        total_correct += (preds[mask] == tgt_flat[mask]).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device, use_amp: bool = True) -> Tuple[float, float]:
    """Run one validation epoch. Returns (avg_loss, avg_accuracy)."""
    import torch
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        src = batch["src"].to(device).transpose(0, 1)
        tgt = batch["tgt"].to(device)
        tgt_input = tgt[:, :-1].transpose(0, 1)
        tgt_output = tgt[:, 1:].contiguous()

        src_pad_mask = (src == 0).transpose(0, 1)
        tgt_pad_mask = (tgt_input == 0).transpose(0, 1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)

        mask = tgt_flat != 0
        total_loss += loss.item() * mask.sum().item()
        preds = logits_flat.argmax(dim=-1)
        total_correct += (preds[mask] == tgt_flat[mask]).sum().item()
        total_tokens += mask.sum().item()

    return total_loss / max(total_tokens, 1), total_correct / max(total_tokens, 1)


def main() -> None:
    """Main training entry point."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        logger.error("Missing dependency: %s. Install with: pip install torch", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train MusicTransformer")
    parser.add_argument("--config", default="training/config_training.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_dir"])

    device_str = args.device or cfg["hardware"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Set seed
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model
    from src.model.transformer import MusicTransformer
    from src.tokenizer.music_tokenizer import MusicTokenizer

    model_cfg = cfg["model"]
    model = MusicTransformer(
        vocab_size=model_cfg["d_model"],  # will be replaced with actual vocab_size
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        positional_encoding=model_cfg["positional_encoding"],
        tie_embeddings=model_cfg.get("tie_embeddings", True),
    ).to(device)

    logger.info("Model: %s", model)

    # Optimizer
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg["weight_decay"],
    )

    # Loss
    train_cfg = cfg["training"]
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=train_cfg["label_smoothing"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["use_amp"])

    # Scheduler
    sched_cfg = cfg["scheduler"]
    total_steps = train_cfg["num_epochs"] * 1000  # estimate
    scheduler = get_lr_scheduler(optimizer, sched_cfg["warmup_steps"], total_steps)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg["logging"]["log_dir"])

    # ── NOTE: Replace with actual DataLoaders ──────────────────────────────
    # For real training, load your processed dataset:
    #   from src.data.dataset import MusicDataset
    #   train_ds = MusicDataset.from_path("data/processed/train.npz")
    #   train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], ...)
    # ─────────────────────────────────────────────────────────────────────

    logger.info(
        "Training ready. Epochs=%d, Batch=%d, Device=%s",
        train_cfg["num_epochs"], train_cfg["batch_size"], device,
    )
    logger.info(
        "\nTo start real training, load your MIDI dataset into data/raw/ "
        "and run src/data/preprocessor.py to generate data/processed/ sequences.\n"
        "Then update this script to point train_loader to MusicDataset.\n"
    )

    # Checkpoint dir
    Path(cfg["checkpointing"]["save_dir"]).mkdir(parents=True, exist_ok=True)

    # Save model architecture summary
    arch_path = Path(cfg["checkpointing"]["save_dir"]) / "model_architecture.txt"
    arch_path.write_text(str(model))
    logger.info("Model architecture saved to %s", arch_path)

    writer.close()
    logger.info("Training setup complete. Model has %s parameters.", f"{model.count_parameters():,}")


# Make eval_epoch importable
try:
    import torch
    eval_epoch = torch.no_grad()(eval_epoch)
except ImportError:
    pass


if __name__ == "__main__":
    main()
