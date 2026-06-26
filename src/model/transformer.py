"""
Music Transformer
=================
Custom Transformer Encoder-Decoder architecture for music generation.
Supports both training (teacher-forcing) and autoregressive inference.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import get_positional_encoding
from .attention import create_causal_mask, create_padding_mask


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder layer with pre-layer normalization."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = src
        src = self.norm1(src)
        src, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = residual + self.drop1(src)

        # Pre-norm feed-forward
        residual = src
        src = self.norm2(src)
        src = self.ff(src)
        src = residual + self.drop2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder layer with pre-layer normalization."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        # Store latest cross-attention weights for visualization
        self._cross_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Masked self-attention
        residual = tgt
        tgt = self.norm1(tgt)
        tgt, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = residual + self.drop1(tgt)

        # Cross-attention over encoder memory
        residual = tgt
        tgt = self.norm2(tgt)
        tgt, cross_weights = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
        )
        self._cross_attn_weights = cross_weights
        tgt = residual + self.drop2(tgt)

        # Feed-forward
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = self.ff(tgt)
        tgt = residual + self.drop3(tgt)
        return tgt


class MusicTransformer(nn.Module):
    """
    Transformer Encoder-Decoder for symbolic music generation.

    Parameters
    ----------
    vocab_size:          Size of the token vocabulary
    d_model:             Embedding / hidden dimension
    nhead:               Number of attention heads
    num_encoder_layers:  Depth of encoder stack
    num_decoder_layers:  Depth of decoder stack
    dim_feedforward:     FFN inner dimension
    dropout:             Dropout probability
    max_seq_len:         Maximum sequence length
    pad_id:              Padding token ID
    positional_encoding: "sinusoidal" or "learnable"
    tie_embeddings:      Share encoder / decoder / output embeddings
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_id: int = 0,
        positional_encoding: str = "sinusoidal",
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.pad_id = pad_id

        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        self.src_pos_enc = get_positional_encoding(
            positional_encoding, d_model, dropout, max_seq_len
        )
        self.tgt_pos_enc = get_positional_encoding(
            positional_encoding, d_model, dropout, max_seq_len
        )

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tie embeddings
        if tie_embeddings:
            self.output_proj.weight = self.tgt_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.src_embedding.weight, std=0.02)
        nn.init.normal_(self.tgt_embedding.weight, std=0.02)

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run encoder on source sequence.

        Args:
            src:                  (seq_len, batch)
            src_key_padding_mask: (batch, seq_len) — True at padding positions

        Returns:
            memory: (seq_len, batch, d_model)
        """
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_pos_enc(x)

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run decoder on target sequence.

        Args:
            tgt:                    (tgt_len, batch)
            memory:                 (src_len, batch, d_model)
            tgt_mask:               (tgt_len, tgt_len) causal mask
            tgt_key_padding_mask:   (batch, tgt_len)
            memory_key_padding_mask:(batch, src_len)

        Returns:
            (tgt_len, batch, d_model)
        """
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pos_enc(x)

        for layer in self.decoder_layers:
            x = layer(
                x, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return self.decoder_norm(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass (teacher-forcing).

        Args:
            src:                  (src_len, batch) — seed / condition sequence
            tgt:                  (tgt_len, batch) — target sequence (shifted right)
            src_key_padding_mask: (batch, src_len)
            tgt_key_padding_mask: (batch, tgt_len)

        Returns:
            logits: (tgt_len, batch, vocab_size)
        """
        tgt_len = tgt.size(0)
        tgt_mask = create_causal_mask(tgt_len, device=tgt.device)

        memory = self.encode(src, src_key_padding_mask)
        decoded = self.decode(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_proj(decoded)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_cross_attention_weights(self) -> Dict[int, Optional[torch.Tensor]]:
        """Return cross-attention weights from all decoder layers."""
        return {
            i: layer._cross_attn_weights
            for i, layer in enumerate(self.decoder_layers)
        }

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"MusicTransformer(\n"
            f"  vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"nhead={self.nhead}\n"
            f"  encoder_layers={len(self.encoder_layers)}, "
            f"decoder_layers={len(self.decoder_layers)}\n"
            f"  parameters={params:,}\n"
            f")"
        )
