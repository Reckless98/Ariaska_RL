"""core/algorithms/auto_curriculum.py — Phase 49: Automatic Curriculum Learning.

Dynamically orders training scenarios by difficulty based on agent
performance, implementing a competence-based curriculum that maximises
the learning gradient at every stage.

Approach:
  - Track per-scenario win-rate, avg reward, and steps-to-completion.
  - Competence score = f(win_rate, avg_reward_norm, speed).
  - Sample scenarios from the "zone of proximal development" — not too
    easy (bored), not too hard (stuck).
  - Uses a beta distribution (α,β) per scenario updated online.

Key concepts:
  - **Mastery threshold**: competence >= 0.85 → scenario mastered
  - **ZPD window**: sample most from competence ∈ [0.30, 0.75]
  - **Re-test**: 10% chance of revisiting mastered scenarios to check retention

Architecture:
    ScenarioPool → CurriculumScheduler → next_scenario()
                                        ↳ update(scenario, result)
                                        ↳ curriculum_stats()
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.algorithms.auto_curriculum")


@dataclass
class CurriculumConfig:
    """Configuration for automatic curriculum learning.

    Args:
        enabled: Master switch.
        mastery_threshold: Competence score to consider a scenario mastered.
        zpd_low: Lower bound of zone of proximal development.
        zpd_high: Upper bound of zone of proximal development.
        retest_prob: Probability of revisiting mastered scenarios.
        ema_alpha: Exponential moving average decay for reward tracking.
        min_attempts: Minimum attempts before trusting competence estimates.
        difficulty_bins: Number of difficulty bins for stratified sampling.
        anti_stagnation_boost: Extra sampling weight for stagnating scenarios.
    """
    enabled: bool = True
    mastery_threshold: float = 0.85
    zpd_low: float = 0.30
    zpd_high: float = 0.75
    retest_prob: float = 0.10
    ema_alpha: float = 0.15
    min_attempts: int = 3
    difficulty_bins: int = 5
    anti_stagnation_boost: float = 2.0


@dataclass
class ScenarioStats:
    """Statistics tracked per scenario."""
    name: str
    difficulty: float = 0.5  # prior difficulty estimate [0, 1]
    attempts: int = 0
    successes: int = 0
    total_reward: float = 0.0
    ema_reward: float = 0.0
    best_reward: float = float("-inf")
    avg_steps: float = 0.0
    # Beta distribution parameters for Bayesian competence
    alpha: float = 1.0  # successes + 1
    beta: float = 1.0   # failures + 1
    # Streak tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_competence: float = 0.5

    @property
    def win_rate(self) -> float:
        """Observed win rate."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def competence(self) -> float:
        """Bayesian competence estimate = mean of Beta(alpha, beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Uncertainty = variance of Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


class CurriculumScheduler:
    """Auto-curriculum that maximises learning by sampling from the ZPD.

    Usage::

        scheduler = CurriculumScheduler(config)
        scheduler.register_scenario("easy_box", difficulty=0.2)
        scheduler.register_scenario("hard_box", difficulty=0.8)

        scenario = scheduler.next_scenario()
        # ... run training episode on scenario ...
        scheduler.update(scenario.name, reward=42.0, success=True, steps=50)

        stats = scheduler.curriculum_stats()
    """

    def __init__(self, config: Optional[CurriculumConfig] = None) -> None:
        self.config = config or CurriculumConfig()
        self._scenarios: Dict[str, ScenarioStats] = {}
        self._history: List[Dict[str, Any]] = []
        self._total_episodes: int = 0

    def register_scenario(
        self,
        name: str,
        difficulty: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a training scenario with initial difficulty estimate."""
        if name in self._scenarios:
            return
        self._scenarios[name] = ScenarioStats(name=name, difficulty=difficulty)
        logger.debug(f"Registered scenario '{name}' with difficulty={difficulty:.2f}")

    def register_scenarios(self, scenarios: List[Tuple[str, float]]) -> None:
        """Batch-register scenarios: [(name, difficulty), ...]."""
        for name, diff in scenarios:
            self.register_scenario(name, diff)

    def next_scenario(self, rng: Optional[random.Random] = None) -> ScenarioStats:
        """Select next scenario using ZPD-based curriculum.

        Sampling strategy:
          1. With `retest_prob`: pick a mastered scenario (retention check).
          2. Otherwise: weight scenarios by ZPD proximity.
          3. Boost under-explored scenarios (< min_attempts).
          4. Boost stagnating scenarios (consecutive failures).

        Returns:
            ScenarioStats for the selected scenario.

        Raises:
            RuntimeError: If no scenarios are registered.
        """
        if not self._scenarios:
            raise RuntimeError("No scenarios registered")

        rng = rng or random.Random()
        scenarios = list(self._scenarios.values())

        # Retest mastered scenarios occasionally
        mastered = [s for s in scenarios if s.competence >= self.config.mastery_threshold]
        if mastered and rng.random() < self.config.retest_prob:
            chosen = rng.choice(mastered)
            logger.debug(f"Retest mastered scenario: '{chosen.name}'")
            return chosen

        # Compute sampling weights based on ZPD
        weights = []
        for s in scenarios:
            w = self._zpd_weight(s)
            weights.append(w)

        # Normalise
        total_w = sum(weights)
        if total_w <= 0:
            # Fall back to uniform
            return rng.choice(scenarios)

        probs = [w / total_w for w in weights]
        chosen = rng.choices(scenarios, weights=probs, k=1)[0]
        logger.debug(
            f"Selected scenario '{chosen.name}' "
            f"(competence={chosen.competence:.3f}, "
            f"difficulty={chosen.difficulty:.2f})"
        )
        return chosen

    def _zpd_weight(self, scenario: ScenarioStats) -> float:
        """Compute sampling weight for a scenario based on ZPD proximity."""
        c = scenario.competence
        cfg = self.config

        # Under-explored: high weight to gather data
        if scenario.attempts < cfg.min_attempts:
            return 3.0

        # In ZPD sweet spot: highest weight
        if cfg.zpd_low <= c <= cfg.zpd_high:
            # Peak at center of ZPD
            center = (cfg.zpd_low + cfg.zpd_high) / 2
            dist = abs(c - center)
            max_dist = (cfg.zpd_high - cfg.zpd_low) / 2
            w = 3.0 - 2.0 * (dist / max_dist)  # 3.0 at center, 1.0 at edges
        elif c < cfg.zpd_low:
            # Too hard: lower weight but not zero
            w = 0.5
        else:
            # Near mastery: moderate weight (some practice still useful)
            w = 0.3

        # Stagnation boost: repeated failures → try again with fresh context
        if scenario.consecutive_failures >= 3:
            w *= cfg.anti_stagnation_boost

        # Uncertainty boost: uncertain estimates need more data
        if scenario.uncertainty > 0.05:
            w *= 1.5

        return max(w, 0.01)

    def update(
        self,
        scenario_name: str,
        reward: float,
        success: bool,
        steps: int,
        max_reward: float = 100.0,
    ) -> Dict[str, Any]:
        """Update scenario stats after an episode.

        Args:
            scenario_name: Name of the scenario just trained on.
            reward: Total episode reward.
            success: Whether the episode was considered successful.
            steps: Number of steps taken.
            max_reward: Maximum possible reward (for normalisation).

        Returns:
            Dict with competence delta and curriculum state.
        """
        if scenario_name not in self._scenarios:
            self.register_scenario(scenario_name)

        s = self._scenarios[scenario_name]
        old_competence = s.competence

        # Update counts
        s.attempts += 1
        if success:
            s.successes += 1
            s.alpha += 1.0
            s.consecutive_successes += 1
            s.consecutive_failures = 0
        else:
            s.beta += 1.0
            s.consecutive_failures += 1
            s.consecutive_successes = 0

        # Update reward tracking
        s.total_reward += reward
        if s.ema_reward == 0.0:
            s.ema_reward = reward
        else:
            s.ema_reward = (
                self.config.ema_alpha * reward
                + (1 - self.config.ema_alpha) * s.ema_reward
            )
        s.best_reward = max(s.best_reward, reward)

        # Update average steps
        if s.avg_steps == 0.0:
            s.avg_steps = float(steps)
        else:
            s.avg_steps = (
                self.config.ema_alpha * steps
                + (1 - self.config.ema_alpha) * s.avg_steps
            )

        # Adjust difficulty estimate based on performance
        norm_reward = min(1.0, max(0.0, reward / max(max_reward, 1e-8)))
        s.difficulty = (
            0.9 * s.difficulty + 0.1 * (1.0 - norm_reward)
        )

        s.last_competence = s.competence
        self._total_episodes += 1

        # Record history
        record = {
            "episode": self._total_episodes,
            "scenario": scenario_name,
            "reward": reward,
            "success": success,
            "steps": steps,
            "competence": s.competence,
            "delta": s.competence - old_competence,
        }
        self._history.append(record)

        logger.info(
            f"[CURRICULUM] '{scenario_name}' "
            f"competence={s.competence:.3f} (Δ{s.competence - old_competence:+.3f}) "
            f"win_rate={s.win_rate:.2f} attempts={s.attempts}"
        )
        return record

    def get_mastery_report(self) -> Dict[str, Any]:
        """Return mastery report across all scenarios."""
        scenarios = list(self._scenarios.values())
        mastered = [s for s in scenarios if s.competence >= self.config.mastery_threshold]
        in_zpd = [
            s for s in scenarios
            if self.config.zpd_low <= s.competence <= self.config.zpd_high
        ]
        struggling = [s for s in scenarios if s.competence < self.config.zpd_low]

        return {
            "total_scenarios": len(scenarios),
            "mastered": len(mastered),
            "in_zpd": len(in_zpd),
            "struggling": len(struggling),
            "mastered_names": [s.name for s in mastered],
            "struggling_names": [s.name for s in struggling],
            "overall_competence": (
                sum(s.competence for s in scenarios) / max(len(scenarios), 1)
            ),
            "total_episodes": self._total_episodes,
        }

    def curriculum_stats(self) -> Dict[str, Any]:
        """Return full curriculum statistics."""
        report = self.get_mastery_report()
        per_scenario = {}
        for name, s in self._scenarios.items():
            per_scenario[name] = {
                "competence": round(s.competence, 4),
                "win_rate": round(s.win_rate, 4),
                "attempts": s.attempts,
                "difficulty": round(s.difficulty, 4),
                "ema_reward": round(s.ema_reward, 2),
                "best_reward": round(s.best_reward, 2) if s.best_reward > float("-inf") else 0.0,
                "consecutive_failures": s.consecutive_failures,
                "uncertainty": round(s.uncertainty, 6),
            }
        report["per_scenario"] = per_scenario
        return report

    def reset(self) -> None:
        """Reset all stats but keep registered scenarios."""
        for s in self._scenarios.values():
            s.attempts = 0
            s.successes = 0
            s.total_reward = 0.0
            s.ema_reward = 0.0
            s.best_reward = float("-inf")
            s.avg_steps = 0.0
            s.alpha = 1.0
            s.beta = 1.0
            s.consecutive_failures = 0
            s.consecutive_successes = 0
            s.last_competence = 0.5
        self._history.clear()
        self._total_episodes = 0
        logger.info("[CURRICULUM] Reset all scenario stats")
