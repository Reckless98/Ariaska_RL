"""core/algorithms/reptile_meta.py — Phase 48: Reptile Meta-Learning.

First-order meta-learning for fast adaptation to new target scenarios.

Algorithm (Nichol et al., 2018):
    1. snapshot_weights = model.state_dict()
    2. for each meta-step:
         a. sample random scenario from scenario pool
         b. run K inner PPO updates on that scenario
         c. interpolate: weights = snapshot + ε * (weights - snapshot)
    3. model.load_state_dict(weights)

Benefits over MAML:
  - 1st-order only (no 2nd-order Hessian → same VRAM as normal training)
  - Stable with PPO's clipped surrogate (MAML is fragile with PPO)
  - Only ~2-5% worse than MAML on few-shot benchmarks
  - Compatible with PPOAgent, DDQNMacro, SACAgent

Scenario pool: 27 distinct target profiles (generic_linux, htb_web_easy,
ctf_web, etc.) each with unique command distributions and phase requirements.
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.algorithms.reptile_meta")

# ── Scenario definitions ────────────────────────────────────────

SCENARIO_PROFILES: Dict[str, Dict[str, Any]] = {
    "generic_linux": {
        "phase_weights": {"RECON": 0.3, "ENUMERATION": 0.3, "EXPLOITATION": 0.25, "PRIVILEGE_ESCALATION": 0.15},
        "typical_ports": [22, 80, 443, 8080],
        "difficulty": 0.3,
    },
    "generic_windows": {
        "phase_weights": {"RECON": 0.25, "ENUMERATION": 0.35, "EXPLOITATION": 0.25, "PRIVILEGE_ESCALATION": 0.15},
        "typical_ports": [135, 139, 445, 3389],
        "difficulty": 0.4,
    },
    "htb_web_easy": {
        "phase_weights": {"RECON": 0.2, "ENUMERATION": 0.4, "EXPLOITATION": 0.3, "PRIVILEGE_ESCALATION": 0.1},
        "typical_ports": [80, 443, 8080, 8443],
        "difficulty": 0.3,
    },
    "htb_web_medium": {
        "phase_weights": {"RECON": 0.15, "ENUMERATION": 0.35, "EXPLOITATION": 0.35, "PRIVILEGE_ESCALATION": 0.15},
        "typical_ports": [80, 443, 8080, 3000],
        "difficulty": 0.5,
    },
    "htb_web_hard": {
        "phase_weights": {"RECON": 0.1, "ENUMERATION": 0.3, "EXPLOITATION": 0.35, "PRIVILEGE_ESCALATION": 0.25},
        "typical_ports": [80, 443, 8080, 5000, 9200],
        "difficulty": 0.7,
    },
    "htb_ad_windows": {
        "phase_weights": {"RECON": 0.2, "ENUMERATION": 0.3, "EXPLOITATION": 0.2, "LATERAL_MOVEMENT": 0.2, "PRIVILEGE_ESCALATION": 0.1},
        "typical_ports": [88, 135, 139, 389, 445, 636, 3268, 5985],
        "difficulty": 0.8,
    },
    "htb_privesc_linux": {
        "phase_weights": {"RECON": 0.1, "ENUMERATION": 0.2, "EXPLOITATION": 0.3, "PRIVILEGE_ESCALATION": 0.4},
        "typical_ports": [22, 80],
        "difficulty": 0.5,
    },
    "htb_privesc_windows": {
        "phase_weights": {"RECON": 0.1, "ENUMERATION": 0.2, "EXPLOITATION": 0.25, "PRIVILEGE_ESCALATION": 0.45},
        "typical_ports": [135, 445, 3389, 5985],
        "difficulty": 0.6,
    },
    "ctf_web": {
        "phase_weights": {"RECON": 0.15, "ENUMERATION": 0.35, "EXPLOITATION": 0.4, "EXFILTRATION": 0.1},
        "typical_ports": [80, 443, 8080],
        "difficulty": 0.4,
    },
    "ctf_pwn": {
        "phase_weights": {"RECON": 0.1, "ENUMERATION": 0.2, "EXPLOITATION": 0.5, "PRIVILEGE_ESCALATION": 0.2},
        "typical_ports": [1337, 9999, 31337],
        "difficulty": 0.6,
    },
    "ctf_crypto": {
        "phase_weights": {"RECON": 0.1, "ENUMERATION": 0.3, "EXPLOITATION": 0.4, "POST_EXPLOITATION": 0.2},
        "typical_ports": [80, 443],
        "difficulty": 0.5,
    },
    "ms2_easy": {
        "phase_weights": {"RECON": 0.25, "ENUMERATION": 0.3, "EXPLOITATION": 0.35, "PRIVILEGE_ESCALATION": 0.1},
        "typical_ports": [21, 22, 23, 25, 80, 139, 445, 3306, 5432, 6667, 8180],
        "difficulty": 0.2,
    },
    "ms3_medium": {
        "phase_weights": {"RECON": 0.2, "ENUMERATION": 0.3, "EXPLOITATION": 0.3, "PRIVILEGE_ESCALATION": 0.2},
        "typical_ports": [21, 22, 80, 445, 3306, 8080, 8585],
        "difficulty": 0.4,
    },
    "iot_embedded": {
        "phase_weights": {"RECON": 0.3, "ENUMERATION": 0.3, "EXPLOITATION": 0.3, "POST_EXPLOITATION": 0.1},
        "typical_ports": [23, 80, 443, 8443, 554, 1883],
        "difficulty": 0.5,
    },
    "cloud_aws": {
        "phase_weights": {"RECON": 0.2, "ENUMERATION": 0.35, "EXPLOITATION": 0.25, "LATERAL_MOVEMENT": 0.1, "EXFILTRATION": 0.1},
        "typical_ports": [80, 443, 22, 5432, 6379],
        "difficulty": 0.7,
    },
    "thick_client": {
        "phase_weights": {"RECON": 0.15, "ENUMERATION": 0.35, "EXPLOITATION": 0.35, "POST_EXPLOITATION": 0.15},
        "typical_ports": [80, 443, 1433, 5432, 8080, 27017],
        "difficulty": 0.5,
    },
    "api_only": {
        "phase_weights": {"RECON": 0.15, "ENUMERATION": 0.4, "EXPLOITATION": 0.35, "POST_EXPLOITATION": 0.1},
        "typical_ports": [80, 443, 8080, 3000, 5000],
        "difficulty": 0.4,
    },
    "lateral_network": {
        "phase_weights": {"RECON": 0.15, "ENUMERATION": 0.2, "EXPLOITATION": 0.2, "LATERAL_MOVEMENT": 0.3, "POST_EXPLOITATION": 0.15},
        "typical_ports": [22, 80, 135, 445, 3389, 5985],
        "difficulty": 0.8,
    },
}


@dataclass
class ReptileConfig:
    """Configuration for Reptile meta-learning.

    Args:
        enabled: Master switch for Reptile.
        outer_lr: Meta-learning rate (ε). Controls interpolation strength.
        inner_steps: Number of PPO updates per inner loop (K).
        scenarios_per_step: How many scenarios to sample per meta-step.
        outer_steps: Total meta-optimization steps per call.
        warmup_steps: Skip Reptile for first N training steps.
        anneal_outer_lr: Cosine-anneal outer_lr to this minimum.
        scenario_pool: List of scenario names to draw from.
        cosine_anneal_steps: Steps over which to anneal outer_lr.
    """
    enabled: bool = True
    outer_lr: float = 0.1
    inner_steps: int = 5
    scenarios_per_step: int = 3
    outer_steps: int = 1
    warmup_steps: int = 100
    anneal_outer_lr: float = 0.01
    scenario_pool: List[str] = field(default_factory=lambda: list(SCENARIO_PROFILES.keys()))
    cosine_anneal_steps: int = 5000


class ScenarioSampler:
    """Samples scenarios with curriculum-aware weighting.

    Harder scenarios are weighted lower initially and ramp up
    as the agent matures (success_rate increases).
    """

    def __init__(self, pool: List[str], curriculum: bool = True) -> None:
        self.pool = pool
        self.curriculum = curriculum
        self._success_rates: Dict[str, float] = {s: 0.0 for s in pool}
        self._attempt_counts: Dict[str, int] = {s: 0 for s in pool}

    def sample(self, n: int, maturity: float = 0.0) -> List[str]:
        """Sample n scenarios with difficulty-aware weighting.

        Args:
            n: Number of scenarios to sample.
            maturity: Agent maturity signal [0, 1]. Higher = more hard scenarios.
        """
        if not self.curriculum or maturity >= 0.8:
            return random.sample(self.pool, min(n, len(self.pool)))

        weights: List[float] = []
        for s in self.pool:
            difficulty = SCENARIO_PROFILES.get(s, {}).get("difficulty", 0.5)
            # Lower maturity → prefer easier scenarios
            weight = max(0.05, 1.0 - difficulty * (1.0 - maturity))
            weights.append(weight)

        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        return random.choices(self.pool, weights=probs, k=min(n, len(self.pool)))

    def record_result(self, scenario: str, success: bool) -> None:
        """Record scenario attempt result for adaptive weighting."""
        if scenario in self._attempt_counts:
            self._attempt_counts[scenario] += 1
            old_rate = self._success_rates[scenario]
            total = self._attempt_counts[scenario]
            self._success_rates[scenario] = old_rate + (1.0 if success else 0.0 - old_rate) / total

    def get_stats(self) -> Dict[str, Any]:
        """Return sampling statistics."""
        return {
            "pool_size": len(self.pool),
            "attempt_counts": dict(self._attempt_counts),
            "success_rates": dict(self._success_rates),
        }


class ReptileMeta:
    """Reptile first-order meta-learning for RL agents.

    Works with any model that exposes ``state_dict()`` / ``load_state_dict()``:
    PPOAgent, DDQNMacro, SACAgent.

    Usage::

        reptile = ReptileMeta(config)

        # During training:
        if reptile.should_run(global_step):
            stats = reptile.meta_step(
                model=ppo_agent.network,
                inner_train_fn=my_inner_loop,
                global_step=global_step,
            )
    """

    def __init__(self, config: Optional[ReptileConfig] = None) -> None:
        self.config = config or ReptileConfig()
        self.sampler = ScenarioSampler(
            pool=self.config.scenario_pool,
            curriculum=True,
        )
        self._meta_steps_done: int = 0
        self._total_inner_steps: int = 0
        self._scenario_results: List[Dict[str, Any]] = []

    # ── Annealing ──────────────────────────────────────────────
    def _current_outer_lr(self, global_step: int) -> float:
        """Cosine-annealed outer learning rate."""
        import math
        if self.config.cosine_anneal_steps <= 0:
            return self.config.outer_lr
        progress = min(1.0, global_step / self.config.cosine_anneal_steps)
        lr_range = self.config.outer_lr - self.config.anneal_outer_lr
        return self.config.anneal_outer_lr + 0.5 * lr_range * (1 + math.cos(math.pi * progress))

    # ── Main API ───────────────────────────────────────────────
    def should_run(self, global_step: int) -> bool:
        """Check if Reptile should run at this step."""
        if not self.config.enabled:
            return False
        if global_step < self.config.warmup_steps:
            return False
        return True

    def meta_step(
        self,
        model: Any,
        inner_train_fn: Callable[[Any, str, int], Dict[str, Any]],
        global_step: int,
        maturity: float = 0.0,
    ) -> Dict[str, Any]:
        """Execute one Reptile meta-optimization step.

        Args:
            model: nn.Module with state_dict()/load_state_dict() (e.g. PPOActorCritic).
            inner_train_fn: Callable(model, scenario_name, inner_steps) → {loss, reward, ...}.
                            Should run K PPO updates on the given scenario and return metrics.
            global_step: Current global training step (for annealing).
            maturity: Agent maturity signal [0, 1] for curriculum sampling.

        Returns:
            Dict with meta-step statistics.
        """
        import torch

        outer_lr = self._current_outer_lr(global_step)

        # 1. Snapshot current weights
        snapshot = copy.deepcopy(model.state_dict())

        # 2. Sample scenarios
        scenarios = self.sampler.sample(
            self.config.scenarios_per_step, maturity=maturity
        )

        # 3. Accumulate weight deltas across scenarios
        accumulated_delta: Dict[str, Any] = {}
        scenario_metrics: List[Dict[str, Any]] = []

        for scenario_name in scenarios:
            # Restore snapshot before each inner loop
            model.load_state_dict(copy.deepcopy(snapshot))

            # Run inner training loop
            metrics = inner_train_fn(model, scenario_name, self.config.inner_steps)
            scenario_metrics.append({
                "scenario": scenario_name,
                **metrics,
            })

            # Compute delta: (trained_weights - snapshot)
            trained_state = model.state_dict()
            for key in snapshot:
                if key not in trained_state:
                    continue
                delta = trained_state[key].float() - snapshot[key].float()
                if key not in accumulated_delta:
                    accumulated_delta[key] = delta.clone()
                else:
                    accumulated_delta[key] += delta

            self._total_inner_steps += self.config.inner_steps

        # 4. Reptile interpolation: weights = snapshot + ε * mean(deltas)
        n_scenarios = len(scenarios)
        final_state = copy.deepcopy(snapshot)
        for key in accumulated_delta:
            mean_delta = accumulated_delta[key] / max(n_scenarios, 1)
            final_state[key] = snapshot[key].float() + outer_lr * mean_delta
            # Cast back to original dtype
            final_state[key] = final_state[key].to(snapshot[key].dtype)

        model.load_state_dict(final_state)
        self._meta_steps_done += 1

        # Record scenario results
        for sm in scenario_metrics:
            success = sm.get("reward", 0) > 0 or sm.get("success", False)
            self.sampler.record_result(sm["scenario"], success)

        stats = {
            "meta_step": self._meta_steps_done,
            "outer_lr": outer_lr,
            "scenarios": [s["scenario"] for s in scenario_metrics],
            "inner_steps": self.config.inner_steps,
            "total_inner_steps": self._total_inner_steps,
            "scenario_metrics": scenario_metrics,
            "n_params_updated": len(accumulated_delta),
        }
        logger.info(
            "Reptile meta-step %d: lr=%.4f, scenarios=%s, inner=%d",
            self._meta_steps_done, outer_lr,
            [s["scenario"] for s in scenario_metrics],
            self.config.inner_steps,
        )
        return stats

    def get_scenario_profile(self, name: str) -> Dict[str, Any]:
        """Get scenario profile by name."""
        return SCENARIO_PROFILES.get(name, {})

    def get_stats(self) -> Dict[str, Any]:
        """Return meta-learning statistics."""
        return {
            "meta_steps_done": self._meta_steps_done,
            "total_inner_steps": self._total_inner_steps,
            "outer_lr_current": self._current_outer_lr(
                self._meta_steps_done * self.config.inner_steps
            ),
            "sampler": self.sampler.get_stats(),
            "config": {
                "outer_lr": self.config.outer_lr,
                "inner_steps": self.config.inner_steps,
                "scenarios_per_step": self.config.scenarios_per_step,
                "cosine_anneal_steps": self.config.cosine_anneal_steps,
            },
        }
