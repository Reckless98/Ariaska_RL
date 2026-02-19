"""core/algorithms/progressive_net.py — Phase 41: Progressive network expansion.

Net2Net-style widening: when explained variance plateaus, expand hidden
dimensions to increase model capacity while preserving learned features.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.algorithms.progressive_net")

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class ExpansionConfig:
    """Configuration for progressive expansion."""
    enabled: bool = False
    milestones: List[int] = field(default_factory=lambda: [200, 500])
    target_dims: List[List[int]] = field(
        default_factory=lambda: [[512, 512, 256], [768, 768, 384]]
    )
    plateau_window: int = 50
    plateau_threshold: float = 0.01
    noise_scale: float = 0.01


class ProgressiveExpander:
    """Monitors training and expands network capacity when plateaued.

    Uses Net2Net widening: copies existing weights into a larger layer,
    adds small noise to break symmetry.
    """

    def __init__(self, config: Optional[ExpansionConfig] = None) -> None:
        self.config = config or ExpansionConfig()
        self._ev_history: Deque[float] = deque(maxlen=self.config.plateau_window)
        self._expansions_done: int = 0

    def should_expand(
        self,
        episode: int,
        explained_variance: float,
    ) -> bool:
        """Check if network should be expanded.

        Args:
            episode: Current episode number.
            explained_variance: Current explained variance from PPO.

        Returns:
            True if expansion should occur.
        """
        if not self.config.enabled:
            return False
        if self._expansions_done >= len(self.config.milestones):
            return False

        self._ev_history.append(explained_variance)

        # Check milestone
        next_milestone = self.config.milestones[self._expansions_done]
        if episode < next_milestone:
            return False

        # Check plateau
        if len(self._ev_history) < self.config.plateau_window:
            return False

        ev_list = list(self._ev_history)
        ev_range = max(ev_list) - min(ev_list)
        return ev_range < self.config.plateau_threshold

    def get_target_dims(self) -> Optional[List[int]]:
        """Get the target hidden dimensions for the next expansion."""
        idx = self._expansions_done
        if idx < len(self.config.target_dims):
            return self.config.target_dims[idx]
        return None

    def record_expansion(self) -> None:
        """Record that an expansion was performed."""
        self._expansions_done += 1
        self._ev_history.clear()
        logger.info(
            "Network expansion #%d complete", self._expansions_done
        )

    @staticmethod
    def widen_layer(
        old_linear: Any,
        new_in: int,
        new_out: int,
        noise_scale: float = 0.01,
    ) -> Any:
        """Net2Net widen a linear layer.

        Copies old weights into the top-left of a new larger layer
        and adds small noise to break symmetry.

        Args:
            old_linear: Existing nn.Linear layer.
            new_in: New input dimension.
            new_out: New output dimension.
            noise_scale: Scale of symmetry-breaking noise.

        Returns:
            New nn.Linear with preserved + expanded weights.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch required")

        old_in = old_linear.in_features
        old_out = old_linear.out_features
        new_layer = nn.Linear(new_in, new_out)

        with torch.no_grad():
            new_layer.weight.zero_()
            new_layer.bias.zero_()

            # Copy old weights
            copy_in = min(old_in, new_in)
            copy_out = min(old_out, new_out)
            new_layer.weight[:copy_out, :copy_in] = old_linear.weight[:copy_out, :copy_in]
            new_layer.bias[:copy_out] = old_linear.bias[:copy_out]

            # Add noise to new dimensions
            if new_out > old_out:
                new_layer.weight[old_out:, :copy_in] = (
                    old_linear.weight[:min(new_out - old_out, old_out), :copy_in]
                    + torch.randn(min(new_out - old_out, old_out), copy_in) * noise_scale
                )
            if new_in > old_in:
                new_layer.weight[:copy_out, old_in:] = (
                    torch.randn(copy_out, new_in - old_in) * noise_scale
                )

        return new_layer
