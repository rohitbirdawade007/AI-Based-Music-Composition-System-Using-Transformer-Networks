"""
Positional Encoding
===================
Implements both sinusoidal (fixed) and learnable positional encodings
for Transformer models.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)           # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                        # (d_model/2,)

        pe = torch.zeros(max_len, 1, d_model)                   # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)

        Returns:
            Tensor of same shape with positional encoding added.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding.
    Each position gets its own embedding learned during training.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device)      # (seq_len,)
        pos_enc = self.pe(positions).unsqueeze(1)                # (seq_len, 1, d_model)
        x = x + pos_enc
        return self.dropout(x)


def get_positional_encoding(
    enc_type: str, d_model: int, dropout: float = 0.1, max_len: int = 5000
) -> nn.Module:
    """Factory function for positional encodings."""
    if enc_type == "sinusoidal":
        return PositionalEncoding(d_model, dropout, max_len)
    elif enc_type == "learnable":
        return LearnablePositionalEncoding(d_model, dropout, max_len)
    else:
        raise ValueError(f"Unknown positional encoding type: {enc_type!r}")
