"""core/training/self_play.py — Phase 41: Asymmetric Red vs Blue self-play.

Supplementary adversarial training: Red tries to exploit, Blue tries
to detect.  Produces shaped rewards for both agents.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("ariaska.training.self_play")


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
    enabled: bool = False
    frequency: int = 5
    red_success_reward: float = 10.0
    red_detected_penalty: float = -5.0
    blue_detect_reward: float = 8.0
    blue_miss_penalty: float = -3.0
    elo_k_factor: float = 32.0
    elo_initial: float = 1000.0


@dataclass
class SelfPlayResult:
    """Result of a self-play round."""
    red_reward: float = 0.0
    blue_reward: float = 0.0
    red_succeeded: bool = False
    blue_detected: bool = False
    episode: int = 0


class EloTracker:
    """Simple Elo rating tracker for Red vs Blue."""

    def __init__(self, k_factor: float = 32.0, initial: float = 1000.0) -> None:
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {
            "red": initial,
            "blue": initial,
        }

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Compute expected score for player A vs B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner: str, loser: str) -> None:
        """Update ratings after a match.

        Args:
            winner: Agent id that won ("red" or "blue").
            loser: Agent id that lost.
        """
        if winner not in self.ratings or loser not in self.ratings:
            return
        e_w = self.expected_score(self.ratings[winner], self.ratings[loser])
        e_l = 1.0 - e_w
        self.ratings[winner] += self.k_factor * (1.0 - e_w)
        self.ratings[loser] += self.k_factor * (0.0 - e_l)

    def get_ratings(self) -> Dict[str, float]:
        """Return current ratings."""
        return dict(self.ratings)


class SelfPlayManager:
    """Manages adversarial self-play rounds between Red and Blue agents."""

    def __init__(self, config: Optional[SelfPlayConfig] = None) -> None:
        self.config = config or SelfPlayConfig()
        self.elo = EloTracker(
            k_factor=self.config.elo_k_factor,
            initial=self.config.elo_initial,
        )
        self.total_rounds: int = 0
        self.red_wins: int = 0
        self.blue_wins: int = 0

    def should_run_self_play(self, episode: int) -> bool:
        """Check if self-play should run this episode."""
        if not self.config.enabled:
            return False
        return episode > 0 and episode % self.config.frequency == 0

    def compute_adversarial_rewards(
        self,
        red_action: str,
        blue_detected: bool,
        red_success: bool,
    ) -> Tuple[float, float]:
        """Compute rewards for both Red and Blue.

        Args:
            red_action: Command Red attempted.
            blue_detected: Whether Blue detected the attack.
            red_success: Whether Red's attack succeeded.

        Returns:
            (red_reward, blue_reward) tuple.
        """
        red_reward = 0.0
        blue_reward = 0.0

        if red_success and not blue_detected:
            # Red wins: succeeded undetected
            red_reward = self.config.red_success_reward
            blue_reward = self.config.blue_miss_penalty
            self.elo.update("red", "blue")
            self.red_wins += 1
        elif red_success and blue_detected:
            # Draw: Red succeeded but was detected
            red_reward = self.config.red_success_reward * 0.3
            blue_reward = self.config.blue_detect_reward * 0.5
        elif not red_success and blue_detected:
            # Blue wins: Red failed and was detected
            red_reward = self.config.red_detected_penalty
            blue_reward = self.config.blue_detect_reward
            self.elo.update("blue", "red")
            self.blue_wins += 1
        else:
            # Both failed
            red_reward = -1.0
            blue_reward = -1.0

        self.total_rounds += 1
        return red_reward, blue_reward

    def get_stats(self) -> Dict[str, Any]:
        """Return self-play statistics."""
        return {
            "total_rounds": self.total_rounds,
            "red_wins": self.red_wins,
            "blue_wins": self.blue_wins,
            "elo_ratings": self.elo.get_ratings(),
            "red_win_rate": self.red_wins / max(1, self.total_rounds),
        }
