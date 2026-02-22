"""core/algorithms/self_play.py — Phase 48: Improved Self-Play Training.

ELO-rated opponent pool with curriculum difficulty scaling for
adversarial training between Red (attacker) and Blue (defender) agents.

Features:
  - ELO rating system for opponent snapshots
  - Opponent pool with diverse skill levels
  - Curriculum difficulty: match against progressively harder opponents
  - Win-rate tracking with moving averages
  - Best-response computation (play against strongest opponent)
  - Adversarial scenario generation for Blue agent
  - Frozen-opponent evaluation rounds
"""
from __future__ import annotations

import copy
import logging
import math
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.algorithms.self_play")

# Default ELO parameters
_INITIAL_ELO = 1200.0
_K_FACTOR = 32.0
_ELO_SCALE = 400.0


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score of A in a match against B (ELO formula)."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / _ELO_SCALE))


def _update_elo(
    rating_a: float, rating_b: float,
    score_a: float, k: float = _K_FACTOR,
) -> Tuple[float, float]:
    """Update ELO ratings after a match.

    Args:
        rating_a: Current ELO of player A.
        rating_b: Current ELO of player B.
        score_a: Actual score for A (1.0=win, 0.5=draw, 0.0=loss).
        k: K-factor (volatility).

    Returns:
        (new_rating_a, new_rating_b)
    """
    expected_a = _expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)
    return new_a, new_b


@dataclass
class OpponentSnapshot:
    """A frozen snapshot of an agent's network weights + metadata."""
    snapshot_id: str
    weights: Dict[str, Any]
    elo: float = _INITIAL_ELO
    wins: int = 0
    losses: int = 0
    draws: int = 0
    created_at_step: int = 0
    agent_role: str = "red"

    @property
    def total_matches(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        total = self.total_matches
        if total == 0:
            return 0.0
        return self.wins / total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata (excluding weights for logging)."""
        return {
            "snapshot_id": self.snapshot_id,
            "elo": self.elo,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "created_at_step": self.created_at_step,
            "agent_role": self.agent_role,
        }


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training.

    Args:
        enabled: Master switch.
        pool_size: Maximum opponent snapshots to keep.
        snapshot_interval: Save snapshot every N episodes.
        match_episodes: Episodes per self-play match.
        curriculum: Enable curriculum difficulty scaling.
        best_response_ratio: Fraction of matches against best opponent.
        evaluation_interval: Frozen evaluation every N meta-steps.
        min_pool_size: Minimum opponents before starting self-play.
        elo_k_factor: ELO K-factor (higher = more volatile ratings).
        win_rate_ema_alpha: EMA alpha for win-rate tracking.
    """
    enabled: bool = True
    pool_size: int = 20
    snapshot_interval: int = 5
    match_episodes: int = 3
    curriculum: bool = True
    best_response_ratio: float = 0.2
    evaluation_interval: int = 10
    min_pool_size: int = 3
    elo_k_factor: float = _K_FACTOR
    win_rate_ema_alpha: float = 0.1


class OpponentPool:
    """Thread-safe pool of opponent snapshots with ELO ratings."""

    def __init__(self, max_size: int = 20) -> None:
        self._pool: List[OpponentSnapshot] = []
        self._max_size = max_size
        self._lock = threading.Lock()
        self._next_id = 0

    def add_snapshot(
        self,
        weights: Dict[str, Any],
        step: int,
        role: str = "red",
        initial_elo: float = _INITIAL_ELO,
    ) -> str:
        """Add a new opponent snapshot to the pool.

        If pool is full, removes the weakest opponent (lowest ELO).
        Returns the snapshot_id.
        """
        with self._lock:
            snapshot_id = f"opponent_{self._next_id:04d}"
            self._next_id += 1

            snapshot = OpponentSnapshot(
                snapshot_id=snapshot_id,
                weights=copy.deepcopy(weights),
                elo=initial_elo,
                created_at_step=step,
                agent_role=role,
            )
            self._pool.append(snapshot)

            # Evict weakest if over capacity
            if len(self._pool) > self._max_size:
                self._pool.sort(key=lambda s: s.elo, reverse=True)
                removed = self._pool.pop()
                logger.debug("Evicted opponent %s (ELO=%.0f)", removed.snapshot_id, removed.elo)

            return snapshot_id

    def select_opponent(
        self,
        current_elo: float,
        maturity: float = 0.0,
        best_response: bool = False,
    ) -> Optional[OpponentSnapshot]:
        """Select an opponent from the pool.

        Args:
            current_elo: Current agent's ELO rating.
            maturity: Agent maturity [0, 1]. Higher → harder opponents.
            best_response: If True, always select the strongest opponent.

        Returns:
            Selected opponent snapshot, or None if pool is empty.
        """
        with self._lock:
            if not self._pool:
                return None

            if best_response:
                return max(self._pool, key=lambda s: s.elo)

            # Curriculum: prefer opponents close to or slightly above agent's level
            weights: List[float] = []
            for opp in self._pool:
                elo_diff = opp.elo - current_elo
                # Gaussian-like preference centered on slightly-harder opponents
                target_diff = 50 + maturity * 200  # Range: [50, 250] ELO above current
                distance = abs(elo_diff - target_diff)
                weight = math.exp(-distance ** 2 / (2 * 150 ** 2))
                weights.append(max(0.01, weight))

            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            chosen = random.choices(self._pool, weights=probs, k=1)[0]
            return chosen

    def update_elo(
        self,
        snapshot_id: str,
        opponent_elo: float,
        won: bool,
        k: float = _K_FACTOR,
    ) -> float:
        """Update ELO rating for a snapshot after a match.

        Returns:
            New ELO rating for the snapshot.
        """
        with self._lock:
            for s in self._pool:
                if s.snapshot_id == snapshot_id:
                    score = 1.0 if won else 0.0
                    new_elo, _ = _update_elo(s.elo, opponent_elo, score, k)
                    s.elo = new_elo
                    if won:
                        s.wins += 1
                    else:
                        s.losses += 1
                    return new_elo
            return opponent_elo

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._pool)

    def get_rankings(self) -> List[Dict[str, Any]]:
        """Return all opponents sorted by ELO."""
        with self._lock:
            sorted_pool = sorted(self._pool, key=lambda s: s.elo, reverse=True)
            return [s.to_dict() for s in sorted_pool]

    def get_best(self) -> Optional[OpponentSnapshot]:
        """Return strongest opponent."""
        with self._lock:
            if not self._pool:
                return None
            return max(self._pool, key=lambda s: s.elo)

    def get_stats(self) -> Dict[str, Any]:
        """Pool statistics."""
        with self._lock:
            if not self._pool:
                return {"size": 0, "avg_elo": 0.0, "max_elo": 0.0, "min_elo": 0.0}
            elos = [s.elo for s in self._pool]
            return {
                "size": len(self._pool),
                "avg_elo": sum(elos) / len(elos),
                "max_elo": max(elos),
                "min_elo": min(elos),
                "total_matches": sum(s.total_matches for s in self._pool),
            }


class MatchResult:
    """Result of a self-play match between two agents."""

    def __init__(
        self,
        agent_score: float,
        opponent_score: float,
        agent_elo_before: float,
        opponent_elo_before: float,
        agent_elo_after: float,
        opponent_elo_after: float,
        agent_discoveries: int = 0,
        opponent_discoveries: int = 0,
        episodes_played: int = 1,
        scenario: str = "",
    ) -> None:
        self.agent_score = agent_score
        self.opponent_score = opponent_score
        self.agent_elo_before = agent_elo_before
        self.opponent_elo_before = opponent_elo_before
        self.agent_elo_after = agent_elo_after
        self.opponent_elo_after = opponent_elo_after
        self.agent_discoveries = agent_discoveries
        self.opponent_discoveries = opponent_discoveries
        self.episodes_played = episodes_played
        self.scenario = scenario

    @property
    def agent_won(self) -> bool:
        return self.agent_score > self.opponent_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_score": self.agent_score,
            "opponent_score": self.opponent_score,
            "agent_won": self.agent_won,
            "agent_elo": f"{self.agent_elo_before:.0f} → {self.agent_elo_after:.0f}",
            "opponent_elo": f"{self.opponent_elo_before:.0f} → {self.opponent_elo_after:.0f}",
            "episodes": self.episodes_played,
            "scenario": self.scenario,
        }


class SelfPlayTrainer:
    """Improved self-play training system with ELO-rated opponent pool.

    Usage::

        sp = SelfPlayTrainer(config)

        # snapshot current agent periodically
        if episode % config.snapshot_interval == 0:
            sp.save_snapshot(agent.network.state_dict(), episode)

        # run a self-play match
        if sp.should_play(episode):
            result = sp.play_match(
                agent_model=agent.network,
                match_fn=my_match_function,
                agent_elo=current_elo,
            )
    """

    def __init__(self, config: Optional[SelfPlayConfig] = None) -> None:
        self.config = config or SelfPlayConfig()
        self.pool = OpponentPool(max_size=self.config.pool_size)
        self.agent_elo: float = _INITIAL_ELO
        self._match_count: int = 0
        self._win_rate_ema: float = 0.5
        self._match_history: deque[MatchResult] = deque(maxlen=100)

    def save_snapshot(
        self,
        weights: Dict[str, Any],
        step: int,
        role: str = "red",
    ) -> str:
        """Save current agent weights as a new opponent snapshot."""
        initial_elo = self.agent_elo  # New opponents start at agent's current level
        snapshot_id = self.pool.add_snapshot(
            weights=weights,
            step=step,
            role=role,
            initial_elo=initial_elo,
        )
        logger.info("Saved snapshot %s at step %d (ELO=%.0f)", snapshot_id, step, initial_elo)
        return snapshot_id

    def should_play(self, episode: int) -> bool:
        """Check if self-play should occur this episode."""
        if not self.config.enabled:
            return False
        if self.pool.size < self.config.min_pool_size:
            return False
        return episode % self.config.snapshot_interval == 0

    def should_snapshot(self, episode: int) -> bool:
        """Check if we should save a snapshot this episode."""
        if not self.config.enabled:
            return False
        return episode % self.config.snapshot_interval == 0

    def select_opponent(self, maturity: float = 0.0) -> Optional[OpponentSnapshot]:
        """Select opponent using curriculum + best-response mix."""
        use_best = random.random() < self.config.best_response_ratio
        return self.pool.select_opponent(
            current_elo=self.agent_elo,
            maturity=maturity,
            best_response=use_best,
        )

    def play_match(
        self,
        agent_model: Any,
        match_fn: Callable[[Any, Dict[str, Any], str], Tuple[float, float, int]],
        maturity: float = 0.0,
        scenario: str = "generic_linux",
    ) -> Optional[MatchResult]:
        """Execute a self-play match against a pool opponent.

        Args:
            agent_model: Current agent's network (nn.Module).
            match_fn: Callable(agent_model, opponent_weights, scenario) →
                      (agent_score, opponent_score, discoveries).
            maturity: Agent maturity for curriculum opponent selection.
            scenario: Scenario name for the match.

        Returns:
            MatchResult with ELO updates, or None if no opponent available.
        """
        opponent = self.select_opponent(maturity=maturity)
        if opponent is None:
            return None

        agent_elo_before = self.agent_elo
        opp_elo_before = opponent.elo

        # Run the match
        agent_score, opp_score, agent_discoveries = match_fn(
            agent_model, opponent.weights, scenario
        )

        # Determine winner
        if agent_score > opp_score:
            match_score = 1.0
        elif agent_score < opp_score:
            match_score = 0.0
        else:
            match_score = 0.5

        # Update ELO ratings
        new_agent_elo, _ = _update_elo(
            self.agent_elo, opponent.elo, match_score, self.config.elo_k_factor
        )
        self.agent_elo = new_agent_elo
        self.pool.update_elo(
            opponent.snapshot_id, agent_elo_before,
            won=(match_score < 0.5), k=self.config.elo_k_factor,
        )

        # Update win rate EMA
        alpha = self.config.win_rate_ema_alpha
        self._win_rate_ema = alpha * match_score + (1 - alpha) * self._win_rate_ema

        result = MatchResult(
            agent_score=agent_score,
            opponent_score=opp_score,
            agent_elo_before=agent_elo_before,
            opponent_elo_before=opp_elo_before,
            agent_elo_after=self.agent_elo,
            opponent_elo_after=opponent.elo,
            agent_discoveries=agent_discoveries,
            episodes_played=self.config.match_episodes,
            scenario=scenario,
        )
        self._match_history.append(result)
        self._match_count += 1

        logger.info(
            "Self-play match %d: %s vs %s → agent=%.1f opp=%.1f "
            "(ELO: %.0f→%.0f, win_rate=%.2f)",
            self._match_count, "agent", opponent.snapshot_id,
            agent_score, opp_score,
            agent_elo_before, self.agent_elo,
            self._win_rate_ema,
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return self-play statistics."""
        return {
            "match_count": self._match_count,
            "agent_elo": self.agent_elo,
            "win_rate_ema": self._win_rate_ema,
            "pool": self.pool.get_stats(),
            "recent_matches": [m.to_dict() for m in list(self._match_history)[-5:]],
        }

    def get_rankings(self) -> List[Dict[str, Any]]:
        """Get full opponent pool rankings."""
        rankings = self.pool.get_rankings()
        # Insert current agent
        rankings.insert(0, {
            "snapshot_id": "CURRENT_AGENT",
            "elo": self.agent_elo,
            "wins": sum(1 for m in self._match_history if m.agent_won),
            "losses": sum(1 for m in self._match_history if not m.agent_won),
            "win_rate": self._win_rate_ema,
        })
        return rankings
