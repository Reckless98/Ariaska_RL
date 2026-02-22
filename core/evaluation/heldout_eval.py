"""core/evaluation/heldout_eval.py — C09: Held-Out Evaluation.

Periodically evaluates PPO policy on held-out scenarios without
training updates, producing reward-invariant metrics for tracking
genuine learning progress.

Key design:
- Runs every N episodes (configurable, default=5)
- Uses CyberEnvironment in eval mode (no training signal)
- Collects metrics: unique_commands, diversity_ratio, phase_reached,
  total_discoveries, steps_to_first_exploit
- Stores rolling eval history for trend analysis
- Gated by FF_HELDOUT_EVAL feature flag
"""
from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.evaluation.heldout_eval")


@dataclass
class EvalScenario:
    """A held-out evaluation scenario."""
    name: str
    target_ip: str = "192.168.1.100"
    max_steps: int = 50
    seed: int = 42
    phase_weights: Dict[str, float] = field(default_factory=dict)
    expected_difficulty: float = 0.5


# ── Built-in eval scenarios ──────────────────────────────────────

EVAL_SCENARIOS: List[EvalScenario] = [
    EvalScenario(
        name="eval_web_basic",
        max_steps=40,
        seed=100,
        phase_weights={"RECON": 0.3, "ENUMERATION": 0.3, "EXPLOITATION": 0.3, "PRIVILEGE_ESCALATION": 0.1},
        expected_difficulty=0.3,
    ),
    EvalScenario(
        name="eval_ssh_pivot",
        max_steps=50,
        seed=200,
        phase_weights={"RECON": 0.2, "ENUMERATION": 0.2, "EXPLOITATION": 0.3, "LATERAL_MOVEMENT": 0.2, "PRIVILEGE_ESCALATION": 0.1},
        expected_difficulty=0.5,
    ),
    EvalScenario(
        name="eval_full_chain",
        max_steps=60,
        seed=300,
        phase_weights={"RECON": 0.15, "ENUMERATION": 0.2, "EXPLOITATION": 0.25, "PRIVILEGE_ESCALATION": 0.2, "EXFILTRATION": 0.2},
        expected_difficulty=0.7,
    ),
]


@dataclass
class EvalMetrics:
    """Result of a single evaluation run."""
    scenario_name: str
    unique_commands: int = 0
    total_steps: int = 0
    diversity_ratio: float = 0.0
    total_discoveries: int = 0
    highest_phase: str = "RECON"
    steps_to_first_exploit: int = -1  # -1 = never reached
    episode_reward: float = 0.0
    success: bool = False  # reached EXPLOITATION or beyond
    phase_progression: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "unique_commands": self.unique_commands,
            "total_steps": self.total_steps,
            "diversity_ratio": self.diversity_ratio,
            "total_discoveries": self.total_discoveries,
            "highest_phase": self.highest_phase,
            "steps_to_first_exploit": self.steps_to_first_exploit,
            "episode_reward": self.episode_reward,
            "success": self.success,
            "phase_progression": self.phase_progression,
        }


@dataclass
class EvalConfig:
    """Configuration for held-out evaluation."""
    eval_interval: int = 5              # Run every N episodes
    scenarios: List[EvalScenario] = field(default_factory=lambda: list(EVAL_SCENARIOS))
    max_history: int = 100              # Rolling window of eval results
    deterministic: bool = True          # Use fixed seeds for reproducibility
    log_results: bool = True            # Log eval metrics to logger


class HeldOutEvaluator:
    """Periodic held-out evaluation for PPO policies.

    Usage in SmartOrchestrator::

        evaluator = HeldOutEvaluator()

        # At episode boundary:
        if evaluator.should_eval(episode):
            results = evaluator.evaluate(
                policy_fn=lambda state: coach.ppo_agent.select_action(state),
                env_step_fn=lambda action: env.step(action),
                env_reset_fn=lambda: env.reset(),
                episode=episode,
            )
    """

    def __init__(self, config: Optional[EvalConfig] = None) -> None:
        self.config = config or EvalConfig()
        self._history: deque[Dict[str, Any]] = deque(maxlen=self.config.max_history)
        self._eval_count: int = 0
        self._best_score: float = 0.0
        self._best_episode: int = -1

    def should_eval(self, episode: int) -> bool:
        """Check if evaluation should run at this episode."""
        if episode < 1:
            return False
        return episode % self.config.eval_interval == 0

    def evaluate(
        self,
        policy_fn: Callable[[Any], Tuple[int, Any, Any]],
        env_step_fn: Callable[[Any], Tuple[Any, float, bool, Dict[str, Any]]],
        env_reset_fn: Callable[[], Any],
        episode: int,
        scenarios: Optional[List[EvalScenario]] = None,
    ) -> List[EvalMetrics]:
        """Run evaluation across held-out scenarios.

        Args:
            policy_fn: Callable(state_tensor) -> (action_idx, log_prob, value).
                       Should be PPO's select_action in eval mode.
            env_step_fn: Callable(action) -> (state, reward, done, info).
            env_reset_fn: Callable() -> initial_state.
            episode: Current training episode number.
            scenarios: Override scenarios (default: config.scenarios).

        Returns:
            List of EvalMetrics, one per scenario.
        """
        scenarios = scenarios or self.config.scenarios
        results: List[EvalMetrics] = []

        for scenario in scenarios:
            if self.config.deterministic:
                random.seed(scenario.seed)

            metrics = self._run_scenario(
                scenario=scenario,
                policy_fn=policy_fn,
                env_step_fn=env_step_fn,
                env_reset_fn=env_reset_fn,
            )
            results.append(metrics)

            if self.config.log_results:
                logger.info(
                    "[EVAL] ep=%d scenario=%s | cmds=%d div=%.2f "
                    "disc=%d phase=%s exploit_step=%d reward=%.1f",
                    episode, metrics.scenario_name,
                    metrics.unique_commands, metrics.diversity_ratio,
                    metrics.total_discoveries, metrics.highest_phase,
                    metrics.steps_to_first_exploit, metrics.episode_reward,
                )

        # Aggregate and store
        agg = self._aggregate(results, episode)
        self._history.append(agg)
        self._eval_count += 1

        # Track best
        score = agg["mean_discoveries"] + agg["mean_diversity"] * 10
        if score > self._best_score:
            self._best_score = score
            self._best_episode = episode

        return results

    def _run_scenario(
        self,
        scenario: EvalScenario,
        policy_fn: Callable,
        env_step_fn: Callable,
        env_reset_fn: Callable,
    ) -> EvalMetrics:
        """Run a single evaluation scenario."""
        metrics = EvalMetrics(scenario_name=scenario.name)
        used_commands: set = set()
        phase_progression: List[str] = []

        try:
            state = env_reset_fn()
        except Exception as e:
            logger.debug(f"[EVAL] env reset failed for {scenario.name}: {e}")
            return metrics

        current_phase = "RECON"
        total_reward = 0.0

        for step in range(scenario.max_steps):
            try:
                action_result = policy_fn(state)
                if isinstance(action_result, tuple) and len(action_result) >= 1:
                    action = action_result[0]
                else:
                    action = action_result

                state, reward, done, info = env_step_fn(action)
                total_reward += reward

                # Track command
                command = info.get("command", f"action_{action}")
                used_commands.add(command)

                # Track discoveries
                discoveries = info.get("discoveries", [])
                metrics.total_discoveries += len(discoveries)

                # Track phase
                new_phase = info.get("phase", current_phase)
                if new_phase != current_phase:
                    phase_progression.append(new_phase)
                    current_phase = new_phase

                # Track first exploit
                if (metrics.steps_to_first_exploit < 0
                        and current_phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION",
                                              "LATERAL_MOVEMENT", "POST_EXPLOITATION",
                                              "EXFILTRATION", "CLOSEOUT")):
                    metrics.steps_to_first_exploit = step

                if done:
                    break

            except Exception as e:
                logger.debug(f"[EVAL] step {step} error in {scenario.name}: {e}")
                break

        metrics.unique_commands = len(used_commands)
        metrics.total_steps = step + 1 if 'step' in dir() else 0
        metrics.diversity_ratio = (
            metrics.unique_commands / max(metrics.total_steps, 1)
        )
        metrics.highest_phase = current_phase
        metrics.episode_reward = total_reward
        metrics.success = current_phase not in ("RECON", "ENUMERATION")
        metrics.phase_progression = phase_progression

        return metrics

    def _aggregate(self, results: List[EvalMetrics], episode: int) -> Dict[str, Any]:
        """Aggregate metrics across scenarios."""
        n = max(len(results), 1)
        return {
            "episode": episode,
            "eval_count": self._eval_count + 1,
            "n_scenarios": len(results),
            "mean_unique_commands": sum(r.unique_commands for r in results) / n,
            "mean_diversity": sum(r.diversity_ratio for r in results) / n,
            "mean_discoveries": sum(r.total_discoveries for r in results) / n,
            "mean_reward": sum(r.episode_reward for r in results) / n,
            "success_rate": sum(1 for r in results if r.success) / n,
            "scenarios": [r.to_dict() for r in results],
        }

    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """Get trend metrics over recent evaluations.

        Returns improvement/regression signals for key metrics.
        """
        if len(self._history) < 2:
            return {"status": "insufficient_data", "evals_done": len(self._history)}

        recent = list(self._history)[-window:]
        half = len(recent) // 2
        first_half = recent[:half] if half > 0 else recent[:1]
        second_half = recent[half:] if half > 0 else recent[1:]

        def _mean(records: List[Dict], key: str) -> float:
            vals = [r.get(key, 0.0) for r in records]
            return sum(vals) / max(len(vals), 1)

        trend = {}
        for key in ("mean_discoveries", "mean_diversity", "mean_reward", "success_rate"):
            v1 = _mean(first_half, key)
            v2 = _mean(second_half, key)
            trend[key] = {
                "early": round(v1, 3),
                "recent": round(v2, 3),
                "delta": round(v2 - v1, 3),
                "improving": v2 > v1,
            }

        trend["evals_done"] = len(self._history)
        trend["best_score"] = self._best_score
        trend["best_episode"] = self._best_episode
        return trend

    def get_stats(self) -> Dict[str, Any]:
        """Return evaluator statistics."""
        return {
            "eval_count": self._eval_count,
            "history_size": len(self._history),
            "best_score": self._best_score,
            "best_episode": self._best_episode,
            "config": {
                "eval_interval": self.config.eval_interval,
                "n_scenarios": len(self.config.scenarios),
                "deterministic": self.config.deterministic,
            },
        }
