"""core/algorithms/attention_ppo.py — Phase 48: Attention-Enhanced PPO Wrapper.

Upgrades PPOActorCritic with deeper attention mechanisms:
  1. Temporal Attention: attends over recent action history (last 16 states)
  2. Cross-Feature Attention: attends between different feature groups
     (phase, ports, services, flags, temporal → fused representation)
  3. Gated Fusion: learned gating between attention-enhanced and
     standard backbone features
  4. Relative Position Encoding for temporal attention

Works as a drop-in wrapper around PPOActorCritic — does NOT modify
the original class. Instead, processes state tensors before and after
the backbone, and fuses the results.

Leverages existing building blocks from core/models/advanced_networks.py:
  - MultiHeadAttention, ResidualBlock, NoisyLinear
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ariaska.algorithms.attention_ppo")

# State feature group slicing (from state_encoder.py, STATE_DIM=512)
_FEATURE_GROUPS: Dict[str, Tuple[int, int]] = {
    "phase":    (0, 12),     # Phase one-hot + progress
    "flags":    (12, 27),    # State flags
    "ports":    (27, 47),    # Top 20 port indicators
    "services": (47, 59),   # Service type presence
    "numeric":  (59, 71),   # Numeric features
    "history":  (71, 81),   # Action history encoding
    "llm":      (81, 86),   # LLM/Mentor features
    "temporal": (86, 91),   # Temporal features
    "reserved": (91, 512),  # Reserved (zero-padded)
}

# Active feature groups (skip reserved)
_ACTIVE_GROUPS = ["phase", "flags", "ports", "services", "numeric", "history", "llm", "temporal"]


@dataclass
class AttentionPPOConfig:
    """Configuration for attention-enhanced PPO.

    Args:
        enabled: Master switch.
        state_dim: Input state dimension (must match STATE_DIM=512).
        attention_dim: Internal attention dimension.
        num_heads: Number of attention heads.
        temporal_window: Number of recent states to attend over.
        cross_feature_heads: Heads for cross-feature attention.
        gate_init_bias: Initial bias for fusion gate (negative = prefer backbone).
        dropout: Dropout rate.
        use_relative_pos: Use relative position encoding in temporal attention.
        use_cross_feature: Enable cross-feature attention.
        use_temporal: Enable temporal attention.
    """
    enabled: bool = True
    state_dim: int = 512
    attention_dim: int = 128
    num_heads: int = 4
    temporal_window: int = 16
    cross_feature_heads: int = 4
    gate_init_bias: float = -2.0    # sigmoid(-2) ≈ 0.12 → mostly backbone initially
    dropout: float = 0.1
    use_relative_pos: bool = True
    use_cross_feature: bool = True
    use_temporal: bool = True


class RelativePositionEncoding(nn.Module):
    """Relative position encoding for temporal attention.

    Adds learned relative position biases to attention scores,
    allowing the model to differentiate between recent and older states.
    """

    def __init__(self, max_len: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        # Learned bias table: (2*max_len - 1, num_heads)
        self.bias_table = nn.Parameter(
            torch.zeros(2 * max_len - 1, num_heads)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get position bias matrix.

        Args:
            seq_len: Sequence length.

        Returns:
            (num_heads, seq_len, seq_len) bias tensor.
        """
        # Relative positions: [-seq_len+1, ..., 0, ..., seq_len-1]
        positions = torch.arange(seq_len, device=self.bias_table.device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        relative = relative + self.max_len - 1  # Shift to [0, 2*max_len-2]
        relative = relative.clamp(0, 2 * self.max_len - 2)

        bias = self.bias_table[relative]  # (seq, seq, heads)
        return bias.permute(2, 0, 1)  # (heads, seq, seq)


class TemporalAttention(nn.Module):
    """Self-attention over a window of recent state observations.

    Learns which past states are relevant for the current decision,
    enabling the agent to capture temporal patterns (e.g., stagnation,
    phase transitions, discovery chains).
    """

    def __init__(self, config: AttentionPPOConfig) -> None:
        super().__init__()
        self.config = config
        d = config.attention_dim

        self.state_proj = nn.Linear(config.state_dim, d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self.num_heads = config.num_heads
        self.d_k = d // config.num_heads

        if config.use_relative_pos:
            self.pos_enc = RelativePositionEncoding(
                max_len=config.temporal_window,
                num_heads=config.num_heads,
            )
        else:
            self.pos_enc = None

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """Attend over state history.

        Args:
            state_history: (B, T, state_dim) tensor where T = temporal_window.

        Returns:
            (B, attention_dim) summary of attended history.
        """
        B, T, _ = state_history.shape

        x = self.state_proj(state_history)  # (B, T, D)

        Q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.pos_enc is not None:
            pos_bias = self.pos_enc(T)  # (heads, T, T)
            scores = scores + pos_bias.unsqueeze(0)  # (B, heads, T, T)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, D)
        out = self.out_proj(out)

        # Take last position (most recent) as summary
        summary = self.norm(out[:, -1, :])  # (B, D)
        return summary


class CrossFeatureAttention(nn.Module):
    """Cross-attention between different feature groups in the state vector.

    Instead of treating the 512-dim state as flat input, splits it into
    semantic feature groups (phase, ports, services, flags, etc.) and
    uses cross-attention to learn inter-group relationships.

    This captures how port information relates to phase decisions,
    how service types influence exploit selection, etc.
    """

    def __init__(self, config: AttentionPPOConfig) -> None:
        super().__init__()
        self.config = config
        d = config.attention_dim
        n_groups = len(_ACTIVE_GROUPS)

        # Project each feature group to attention_dim
        self.group_projections = nn.ModuleDict()
        for name in _ACTIVE_GROUPS:
            start, end = _FEATURE_GROUPS[name]
            group_dim = end - start
            self.group_projections[name] = nn.Linear(group_dim, d)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.cross_feature_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d * n_groups, d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Cross-attend between feature groups.

        Args:
            state: (B, state_dim) tensor.

        Returns:
            (B, attention_dim) cross-attended features.
        """
        B = state.shape[0]
        group_features: List[torch.Tensor] = []

        for name in _ACTIVE_GROUPS:
            start, end = _FEATURE_GROUPS[name]
            group = state[:, start:end]  # (B, group_dim)
            proj = self.group_projections[name](group)  # (B, D)
            group_features.append(proj)

        # Stack as sequence: (B, n_groups, D)
        x = torch.stack(group_features, dim=1)

        # Self-attention across groups
        attended, _ = self.cross_attn(x, x, x)  # (B, n_groups, D)
        attended = self.norm(attended + x)  # Residual

        # Flatten and project
        flat = attended.view(B, -1)  # (B, n_groups * D)
        out = self.output_proj(flat)  # (B, D)
        return self.output_norm(out)


class AttentionEnhancedPPO(nn.Module):
    """Attention-enhanced wrapper for PPO state processing.

    Adds temporal and cross-feature attention as supplementary
    pathways that fuse with the standard PPO backbone via learned gating.

    Does NOT replace PPOActorCritic — augments it by providing
    enhanced state features that can be concatenated to the backbone
    output before the actor/critic heads.

    Usage::

        attn_ppo = AttentionEnhancedPPO(config)

        # During forward:
        attn_features = attn_ppo(state, state_history)
        # Concatenate with backbone features, then pass to heads
    """

    def __init__(self, config: Optional[AttentionPPOConfig] = None) -> None:
        super().__init__()
        self.config = config or AttentionPPOConfig()
        d = self.config.attention_dim

        # Temporal attention over state history
        if self.config.use_temporal:
            self.temporal_attn = TemporalAttention(self.config)
        else:
            self.temporal_attn = None

        # Cross-feature attention within current state
        if self.config.use_cross_feature:
            self.cross_feature_attn = CrossFeatureAttention(self.config)
        else:
            self.cross_feature_attn = None

        # Count number of active attention features
        n_features = 0
        if self.temporal_attn is not None:
            n_features += d
        if self.cross_feature_attn is not None:
            n_features += d

        # Fusion: combine attention features into single vector
        self.fusion = nn.Sequential(
            nn.Linear(max(n_features, 1), d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        # Gate: learned blending weight (sigmoid output)
        # Initialized to prefer backbone (gate_init_bias < 0)
        self.gate = nn.Linear(d, d)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, self.config.gate_init_bias)

        self.output_dim = d
        self._state_history: deque[torch.Tensor] = deque(
            maxlen=self.config.temporal_window
        )

    def push_state(self, state: torch.Tensor) -> None:
        """Add current state to temporal history buffer.

        Call this at each step BEFORE forward() so temporal
        attention has access to recent history.

        Args:
            state: (state_dim,) or (1, state_dim) tensor.
        """
        if state.dim() > 1:
            state = state.squeeze(0)
        self._state_history.append(state.detach().cpu())

    def get_state_history(self, state: torch.Tensor) -> torch.Tensor:
        """Get padded state history tensor.

        Args:
            state: Current state (B, state_dim).

        Returns:
            (B, T, state_dim) tensor with history, zero-padded if needed.
        """
        T = self.config.temporal_window
        device = state.device
        B = state.shape[0]
        state_dim = state.shape[-1]

        if len(self._state_history) == 0:
            # No history yet, use current state repeated
            return state.unsqueeze(1).expand(B, T, state_dim)

        # Stack history
        history = list(self._state_history)
        while len(history) < T:
            history.insert(0, torch.zeros(state_dim))

        history = history[-T:]  # Trim to window
        history_tensor = torch.stack(history).to(device)  # (T, state_dim)

        # Broadcast for batch
        return history_tensor.unsqueeze(0).expand(B, T, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-enhanced features and fusion gate.

        Args:
            state: (B, state_dim) current state.
            state_history: Optional (B, T, state_dim) history. If None,
                          uses internal buffer.

        Returns:
            (attention_features, gate_values) — both (B, attention_dim).
            gate_values in [0, 1]: higher = more attention, lower = more backbone.
        """
        features: List[torch.Tensor] = []

        # Temporal attention
        if self.temporal_attn is not None:
            if state_history is None:
                state_history = self.get_state_history(state)
            temporal_feat = self.temporal_attn(state_history)  # (B, D)
            features.append(temporal_feat)

        # Cross-feature attention
        if self.cross_feature_attn is not None:
            cross_feat = self.cross_feature_attn(state)  # (B, D)
            features.append(cross_feat)

        if not features:
            # No attention modules active, return zeros
            B = state.shape[0]
            d = self.config.attention_dim
            zero = torch.zeros(B, d, device=state.device)
            return zero, torch.zeros_like(zero)

        # Concatenate and fuse
        combined = torch.cat(features, dim=-1)  # (B, n_features * D)
        fused = self.fusion(combined)  # (B, D)

        # Compute gate
        gate_logits = self.gate(fused)  # (B, D)
        gate_values = torch.sigmoid(gate_logits)  # (B, D) in [0, 1]

        return fused, gate_values

    def reset_history(self) -> None:
        """Clear temporal state history (call at episode boundaries)."""
        self._state_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return module statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "total_params": total_params,
            "attention_dim": self.config.attention_dim,
            "temporal_window": self.config.temporal_window,
            "num_heads": self.config.num_heads,
            "use_temporal": self.config.use_temporal,
            "use_cross_feature": self.config.use_cross_feature,
            "history_length": len(self._state_history),
        }
