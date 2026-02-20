"""core/metrics/chain_scorer.py — Phase 42: Kill-chain momentum scorer.

Computes a momentum bonus based on consecutive successful steps through
the kill chain. Used by SmartRewardCalculator to amplify rewards when
the agent maintains forward progress through attack phases.

Momentum formula:
    bonus = base_bonus * (1 + chain_length * growth_rate)
    capped at max_bonus

Author: Phase 42 Contract
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.metrics.chain_scorer")


@dataclass
class ChainConfig:
    """Configuration for chain scoring."""
    base_bonus: float = 1.0
    growth_rate: float = 0.15
    max_bonus: float = 5.0
    decay_on_fail: float = 0.5  # multiply chain by this on failure
    min_chain_for_bonus: int = 2  # minimum chain length to earn bonus


class ChainScorer:
    """Phase 42: Kill-chain momentum scorer.

    Tracks consecutive productive steps and computes a momentum bonus.
    "Productive" means the step produced at least one discovery or
    advanced the attack phase.

    Methods:
        record_step(): Record outcome of a step
        get_momentum(): Current momentum bonus multiplier
        get_chain_length(): Current chain length
        summary(): Stats for telemetry
        reset(): Clear for new episode
    """

    def __init__(self, config: Optional[ChainConfig] = None) -> None:
        self.config = config or ChainConfig()
        self._chain_length: int = 0
        self._max_chain: int = 0
        self._total_productive: int = 0
        self._total_steps: int = 0
        self._chain_history: List[int] = []  # chain lengths at breaks
        logger.info("ChainScorer initialized")

    def record_step(self, productive: bool) -> float:
        """Record the outcome of a step.

        Args:
            productive: True if the step produced a discovery or phase advance.

        Returns:
            Current momentum bonus (0.0 if chain too short).
        """
        self._total_steps += 1
        if productive:
            self._chain_length += 1
            self._total_productive += 1
            self._max_chain = max(self._max_chain, self._chain_length)
        else:
            if self._chain_length > 0:
                self._chain_history.append(self._chain_length)
            self._chain_length = int(
                self._chain_length * self.config.decay_on_fail
            )

        return self.get_momentum()

    def get_momentum(self) -> float:
        """Get current momentum bonus.

        Returns:
            Momentum bonus value (0.0 if chain below minimum).
        """
        if self._chain_length < self.config.min_chain_for_bonus:
            return 0.0
        bonus = self.config.base_bonus * (
            1.0 + self._chain_length * self.config.growth_rate
        )
        return min(bonus, self.config.max_bonus)

    def get_chain_length(self) -> int:
        """Get current chain length.

        Returns:
            Number of consecutive productive steps.
        """
        return self._chain_length

    def summary(self) -> Dict[str, Any]:
        """Stats for telemetry/dashboard.

        Returns:
            Dict with chain stats.
        """
        return {
            "chain_length": self._chain_length,
            "max_chain": self._max_chain,
            "total_productive": self._total_productive,
            "total_steps": self._total_steps,
            "productivity_rate": (
                self._total_productive / max(self._total_steps, 1)
            ),
            "current_momentum": self.get_momentum(),
            "chain_breaks": len(self._chain_history),
            "avg_chain": (
                sum(self._chain_history) / max(len(self._chain_history), 1)
                if self._chain_history else 0.0
            ),
        }

    def reset(self) -> None:
        """Clear for new episode."""
        if self._chain_length > 0:
            self._chain_history.append(self._chain_length)
        self._chain_length = 0
        self._max_chain = 0
        self._total_productive = 0
        self._total_steps = 0
        self._chain_history.clear()
        logger.debug("ChainScorer reset")
