"""
Multi-Head Attention with Weight Extraction
===========================================
Wraps PyTorch's MultiheadAttention to also return attention weights
for visualization purposes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWithWeights(nn.Module):
    """
    Multi-Head Self/Cross-Attention that exposes attention weight matrices.

    Identical to ``nn.MultiheadAttention`` but always returns attention weights
    even when ``need_weights=False`` was intended (for visualization).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query:            (seq_q, batch, embed_dim)  or  (batch, seq_q, embed_dim)
            key:              (seq_k, batch, embed_dim)
            value:            (seq_v, batch, embed_dim)
            key_padding_mask: (batch, seq_k) — True where key positions are padded
            attn_mask:        (seq_q, seq_k) — additive causal / custom mask
            need_weights:     if True, return attention weights

        Returns:
            output:       (seq_q, batch, embed_dim)
            attn_weights: (batch, num_heads, seq_q, seq_k) or None
        """
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        seq_q, batch_size, _ = query.shape
        seq_k = key.shape[0]

        # Project
        Q = self.q_proj(query)  # (seq_q, batch, embed_dim)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape → (batch * num_heads, seq, head_dim)
        def reshape(t: torch.Tensor, seq: int) -> torch.Tensor:
            return (
                t.view(seq, batch_size, self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
                .reshape(batch_size * self.num_heads, seq, self.head_dim)
            )

        Q = reshape(Q, seq_q)
        K = reshape(K, seq_k)
        V = reshape(V, seq_k)

        # Scaled dot-product
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch*heads, q, k)

        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask

        if key_padding_mask is not None:
            # (batch, 1, 1, seq_k)
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)
            kpm = kpm.expand(batch_size, self.num_heads, seq_q, seq_k)
            kpm = kpm.reshape(batch_size * self.num_heads, seq_q, seq_k)
            attn_logits = attn_logits.masked_fill(kpm, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Weighted sum
        output = torch.bmm(attn_weights_dropped, V)  # (batch*heads, q, head_dim)
        output = (
            output.view(batch_size, self.num_heads, seq_q, self.head_dim)
            .permute(2, 0, 1, 3)
            .reshape(seq_q, batch_size, self.embed_dim)
        )
        output = self.out_proj(output)

        if self.batch_first:
            output = output.transpose(0, 1)

        if need_weights:
            # Average over heads for visualization
            attn_w = attn_weights.view(batch_size, self.num_heads, seq_q, seq_k)
            return output, attn_w
        return output, None


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (upper-triangular) mask for autoregressive generation.

    Returns:
        (seq_len, seq_len) float tensor with 0 at valid positions and -inf elsewhere.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


def create_padding_mask(token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    Create a key-padding mask from token IDs.

    Returns:
        (batch, seq_len) bool tensor, True where tokens are padding.
    """
    return token_ids == pad_id
