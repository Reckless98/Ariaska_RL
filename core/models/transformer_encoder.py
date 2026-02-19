"""core/models/transformer_encoder.py — Phase 41: Transformer-based state encoder.

Optional replacement for the MLP-based state encoder.  Uses a small
TransformerEncoder over a rolling window of recent states to capture
temporal patterns in the attack sequence.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

logger = logging.getLogger("ariaska.models.transformer_encoder")

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class TransformerEncoderConfig:
    """Configuration for transformer state encoder."""
    window_size: int = 4
    state_dim: int = 512
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class PositionalEncoding(nn.Module if _HAS_TORCH else object):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 32) -> None:
        if _HAS_TORCH:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Add positional encoding.  x shape: (seq, batch, d_model)."""
        return x + self.pe[: x.size(0)]


class StateWindowBuffer:
    """Rolling window buffer for state sequences."""

    def __init__(self, window_size: int = 4, state_dim: int = 512) -> None:
        self.window_size = window_size
        self.state_dim = state_dim
        self._buffer: Deque["torch.Tensor"] = deque(maxlen=window_size)

    def add(self, state: "torch.Tensor") -> None:
        """Add a state to the window."""
        if state.dim() > 1:
            state = state.flatten()[:self.state_dim]
        self._buffer.append(state.detach().clone())

    def get_window(self) -> "torch.Tensor":
        """Get current window as (window_size, state_dim) tensor, zero-padded."""
        if not _HAS_TORCH:
            raise RuntimeError("torch required")
        if not self._buffer:
            return torch.zeros(self.window_size, self.state_dim)
        states = list(self._buffer)
        pad_count = self.window_size - len(states)
        if pad_count > 0:
            zeros = [torch.zeros(self.state_dim)] * pad_count
            states = zeros + states
        return torch.stack(states)

    def get_mask(self) -> "torch.Tensor":
        """Get padding mask: True for padded positions."""
        if not _HAS_TORCH:
            raise RuntimeError("torch required")
        n_valid = len(self._buffer)
        mask = torch.ones(self.window_size, dtype=torch.bool)
        mask[-n_valid:] = False
        return mask

    def reset(self) -> None:
        """Clear the window."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Number of states in buffer."""
        return len(self._buffer)


class StateTransformerEncoder(nn.Module if _HAS_TORCH else object):
    """Transformer-based state encoder.

    Input:  (B, window_size, state_dim)
    Output: (B, state_dim)
    """

    def __init__(self, config: Optional[TransformerEncoderConfig] = None) -> None:
        if _HAS_TORCH:
            super().__init__()
        self.config = config or TransformerEncoderConfig()

        if not _HAS_TORCH:
            return

        self.input_proj = nn.Linear(self.config.state_dim, self.config.d_model)
        self.pos_enc = PositionalEncoding(self.config.d_model, max_len=32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_layers
        )
        self.output_proj = nn.Linear(self.config.d_model, self.config.state_dim)

    def forward(
        self,
        x: "torch.Tensor",
        src_key_padding_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Forward pass.

        Args:
            x: (B, W, state_dim) state window.
            src_key_padding_mask: (B, W) True for padded positions.

        Returns:
            (B, state_dim) encoded state.
        """
        # (B, W, D) -> (W, B, d_model)
        h = self.input_proj(x)
        h = h.permute(1, 0, 2)
        h = self.pos_enc(h)

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Mean pool over sequence dim -> (B, d_model)
        if src_key_padding_mask is not None:
            # Mask out padding
            mask_expanded = (~src_key_padding_mask).float().T.unsqueeze(-1)
            h = (h * mask_expanded).sum(dim=0) / mask_expanded.sum(dim=0).clamp(min=1.0)
        else:
            h = h.mean(dim=0)

        return self.output_proj(h)
