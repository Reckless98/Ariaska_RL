"""
core/decision/maturity_controller.py — Phase 2+4: Global Maturity Controller.

Single authority for ALL annealing schedules.  Replaces the fragmented
3-writer entropy control and independent mentor/BC/KL scheduling.

Controlled parameters:
  - PPO entropy_coef
  - SAC alpha floor
  - Mentor influence rate
  - BC distillation coefficient
  - KL teacher coefficient
  - LLM prior alpha
  - RND intrinsic scale
  - Reptile trigger window
  - Evaluation cadence

M ∈ [0, 1] is the single maturity signal driving everything.

Author: Phase 2+4 — Entropy Sovereignty + Schedule Coupling
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("ariaska.decision.maturity")


@dataclass
class MaturityConfig:
    """Configuration for GlobalMaturityController."""

    # Entropy schedule
    entropy_coef_max: float = 0.08
    entropy_coef_min: float = 0.003

    # Mentor rate schedule
    mentor_rate_max: float = 0.80
    mentor_rate_min: float = 0.10

    # BC coefficient schedule
    bc_coef_max: float = 0.10
    bc_coef_min: float = 0.01

    # KL teacher coefficient schedule
    kl_coef_max: float = 0.15
    kl_coef_min: float = 0.01

    # LLM prior alpha schedule
    prior_alpha_max: float = 0.50
    prior_alpha_min: float = 0.02

    # RND intrinsic scale
    rnd_scale_max: float = 1.0
    rnd_scale_min: float = 0.1

    # SAC alpha floor
    sac_alpha_floor_max: float = 0.2
    sac_alpha_floor_min: float = 0.05

    # Reptile window (active when maturity < threshold)
    reptile_maturity_ceiling: float = 0.6

    # Eval cadence (episodes between evals)
    eval_cadence_early: int = 5
    eval_cadence_late: int = 20

    # Maturity signal weights
    w_success_rate: float = 0.4
    w_skill_coverage: float = 0.3
    w_discovery_efficiency: float = 0.2
    w_stagnation: float = 0.1


@dataclass
class MaturityState:
    """Current state of all controlled parameters."""
    maturity: float = 0.0
    entropy_coef: float = 0.08
    mentor_rate: float = 0.80
    bc_coef: float = 0.10
    kl_coef: float = 0.15
    prior_alpha: float = 0.50
    rnd_scale: float = 1.0
    sac_alpha_floor: float = 0.2
    reptile_active: bool = True
    eval_cadence: int = 5
    update_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maturity": round(self.maturity, 4),
            "entropy_coef": round(self.entropy_coef, 5),
            "mentor_rate": round(self.mentor_rate, 4),
            "bc_coef": round(self.bc_coef, 4),
            "kl_coef": round(self.kl_coef, 4),
            "prior_alpha": round(self.prior_alpha, 4),
            "rnd_scale": round(self.rnd_scale, 4),
            "sac_alpha_floor": round(self.sac_alpha_floor, 4),
            "reptile_active": self.reptile_active,
            "eval_cadence": self.eval_cadence,
            "update_count": self.update_count,
        }


class GlobalMaturityController:
    """Single authority for all annealing schedules.

    Replaces:
      - PPO internal cosine entropy schedule (ppo_agent.py L1574)
      - SmartCoach entropy multiplier mutation (smart_coach.py L3177)
      - ScheduleCoupler entropy/mentor/BC/KL writes (schedule_coupler.py)
      - Independent mentor rate computation in decide() (smart_coach.py L3485)

    All controlled parameters are computed from ONE maturity signal M.
    The maturity signal itself is computed from reward-invariant metrics.

    INVARIANT: This is the ONLY module that writes to:
      - ppo_agent.entropy_coef
      - ppo_agent.config.entropy_coef
      - ppo_agent._entropy_adaptive_multiplier

    Any other module attempting to write these should raise an error
    (enforced by IntegrityCheck).
    """

    def __init__(self, config: Optional[MaturityConfig] = None) -> None:
        self.config = config or MaturityConfig()
        self.state = MaturityState()
        self._exploration_boost: float = 0.0
        self._boost_reason: str = ""
        self._boost_ttl: int = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def update(
        self,
        episode: int,
        metrics: Dict[str, float],
    ) -> MaturityState:
        """Recompute ALL schedules from current metrics.

        Called ONCE per episode at episode start.  Every parameter is
        derived from the single maturity signal M.

        Args:
            episode: Current episode number.
            metrics: Reward-invariant metrics dict with keys:
                success_rate, skill_coverage, discovery_efficiency,
                stagnation_rate.

        Returns:
            Updated MaturityState.
        """
        cfg = self.config

        # ── Compute single maturity signal M ∈ [0, 1] ────────────────
        success_rate = metrics.get("success_rate", 0.0)
        skill_coverage = metrics.get("skill_coverage", 0.0)
        disc_efficiency = metrics.get("discovery_efficiency", 0.0)
        stagnation_rate = metrics.get("stagnation_rate", 0.0)

        maturity = (
            cfg.w_success_rate * success_rate
            + cfg.w_skill_coverage * skill_coverage
            + cfg.w_discovery_efficiency * disc_efficiency
            + cfg.w_stagnation * (1.0 - stagnation_rate)
        )
        maturity = max(0.0, min(1.0, maturity))
        self.state.maturity = maturity

        # ── Derive all schedules from M ──────────────────────────────

        # Entropy: high when immature, low when mature (cosine)
        entropy = self._cosine_anneal(
            cfg.entropy_coef_max, cfg.entropy_coef_min, maturity,
        )
        # Apply exploration boost if requested
        if self._boost_ttl > 0:
            entropy *= (1.0 + self._exploration_boost)
            self._boost_ttl -= 1
            if self._boost_ttl == 0:
                self._exploration_boost = 0.0
                self._boost_reason = ""

        self.state.entropy_coef = max(cfg.entropy_coef_min, entropy)

        # Mentor rate: high when immature, low when mature
        self.state.mentor_rate = self._cosine_anneal(
            cfg.mentor_rate_max, cfg.mentor_rate_min, maturity,
        )

        # BC coef: high early (learn from mentor), low late
        self.state.bc_coef = self._cosine_anneal(
            cfg.bc_coef_max, cfg.bc_coef_min, maturity,
        )

        # KL teacher coef: same pattern
        self.state.kl_coef = self._cosine_anneal(
            cfg.kl_coef_max, cfg.kl_coef_min, maturity,
        )

        # LLM prior alpha: same pattern
        self.state.prior_alpha = self._cosine_anneal(
            cfg.prior_alpha_max, cfg.prior_alpha_min, maturity,
        )

        # RND scale: slight decay as exploration becomes less needed
        self.state.rnd_scale = self._cosine_anneal(
            cfg.rnd_scale_max, cfg.rnd_scale_min, maturity,
        )

        # SAC alpha floor
        self.state.sac_alpha_floor = self._cosine_anneal(
            cfg.sac_alpha_floor_max, cfg.sac_alpha_floor_min, maturity,
        )

        # Reptile: active only when immature
        self.state.reptile_active = maturity < cfg.reptile_maturity_ceiling

        # Eval cadence: more frequent early, less frequent late
        self.state.eval_cadence = int(
            cfg.eval_cadence_early + (cfg.eval_cadence_late - cfg.eval_cadence_early) * maturity
        )

        self.state.update_count += 1

        logger.info(
            f"[MATURITY] ep={episode} M={maturity:.3f} "
            f"entropy={self.state.entropy_coef:.4f} "
            f"mentor={self.state.mentor_rate:.3f} "
            f"bc={self.state.bc_coef:.3f} "
            f"rnd={self.state.rnd_scale:.3f} "
            f"reptile={'ON' if self.state.reptile_active else 'OFF'}"
        )

        return self.state

    # ------------------------------------------------------------------
    # Apply to algorithm instances
    # ------------------------------------------------------------------

    def apply_to_ppo(self, ppo_agent: Any) -> None:
        """Write entropy_coef to PPO. This is the ONLY write point.

        Replaces:
          - ppo_agent.py L1574 (internal cosine schedule)
          - smart_coach.py L3177 (multiplier mutation)
          - schedule_coupler.py L229 (base reduction)
        """
        if ppo_agent is None:
            return
        # Set both config and live value
        if hasattr(ppo_agent, 'config') and ppo_agent.config is not None:
            ppo_agent.config.entropy_coef = self.state.entropy_coef
        ppo_agent.entropy_coef = self.state.entropy_coef
        # Lock the adaptive multiplier to 1.0 — maturity controller owns entropy
        ppo_agent._entropy_adaptive_multiplier = 1.0
        # Prevent PPO internal methods from overwriting entropy_coef
        ppo_agent._entropy_locked = True

        logger.debug(
            f"[MATURITY] PPO entropy set to {self.state.entropy_coef:.5f} "
            f"(M={self.state.maturity:.3f})"
        )

    def apply_to_sac(self, sac_agent: Any) -> None:
        """Apply alpha floor to SAC."""
        if sac_agent is None:
            return
        if hasattr(sac_agent, 'alpha'):
            sac_agent.alpha = max(sac_agent.alpha, self.state.sac_alpha_floor)

    def apply_to_bridge(self, llm_bridge: Any) -> None:
        """Apply prior_alpha and kl_coef to LLMPolicyBridge.

        P3: Also sets _mc_prior_alpha so the bridge's _compute_anneal_alpha
        uses MC's value directly instead of recomputing (no double-decay).
        """
        if llm_bridge is None:
            return
        # P3: Set authority signal — bridge reads this in _compute_anneal_alpha
        llm_bridge._mc_prior_alpha = self.state.prior_alpha
        if hasattr(llm_bridge, 'config'):
            bridge_cfg = llm_bridge.config
            if hasattr(bridge_cfg, 'prior_alpha_init'):
                bridge_cfg.prior_alpha_init = self.state.prior_alpha
            if hasattr(bridge_cfg, 'kl_teacher_coef'):
                bridge_cfg.kl_teacher_coef = self.state.kl_coef

    def apply_to_mentor_policy(self, mentor_policy: Any) -> None:
        """Apply mentor_rate to MentorPolicy."""
        if mentor_policy is None:
            return
        if hasattr(mentor_policy, 'set_maturity'):
            mentor_policy.set_maturity(self.state.maturity)

    def apply_all(
        self,
        ppo_agent: Any = None,
        sac_agent: Any = None,
        llm_bridge: Any = None,
        mentor_policy: Any = None,
    ) -> None:
        """Apply all scheduled values to their target modules."""
        self.apply_to_ppo(ppo_agent)
        self.apply_to_sac(sac_agent)
        self.apply_to_bridge(llm_bridge)
        self.apply_to_mentor_policy(mentor_policy)

    # ------------------------------------------------------------------
    # Exploration boost (replaces SmartCoach stagnation entropy mutation)
    # ------------------------------------------------------------------

    def request_exploration_boost(
        self,
        reason: str,
        magnitude: float = 0.5,
        duration_episodes: int = 3,
    ) -> None:
        """Request temporary entropy boost (replaces SmartCoach L3177 mutation).

        Instead of SmartCoach directly mutating PPO's _entropy_adaptive_multiplier,
        it sends a boost REQUEST to the maturity controller. The controller
        decides the actual boost magnitude and duration.

        Args:
            reason: Why the boost is requested.
            magnitude: Requested boost factor (0.0-1.0, added to entropy).
            duration_episodes: How many episodes the boost lasts.
        """
        self._exploration_boost = max(0.0, min(1.0, magnitude))
        self._boost_reason = reason
        self._boost_ttl = max(1, duration_episodes)
        logger.info(
            f"[MATURITY] Exploration boost requested: "
            f"magnitude={self._exploration_boost:.2f} "
            f"duration={self._boost_ttl} reason={reason}"
        )

    # ------------------------------------------------------------------
    # Maturity-gated queries
    # ------------------------------------------------------------------

    def should_call_mentor(self, agent_name: str) -> bool:
        """Whether mentor should be called this step.

        Uses maturity-derived mentor_rate as probability, with per-agent
        scaling (Red/Orion get higher rates).
        """
        import random
        rate = self.state.mentor_rate

        # Per-agent scaling (same as original but from single source)
        if agent_name == "RedAgent":
            rate = min(1.0, rate * 1.2)
        elif agent_name == "OrionAgent":
            rate = min(1.0, rate * 1.1)
        else:
            rate = rate * 0.7

        return random.random() < rate

    def should_run_reptile(self) -> bool:
        """Whether Reptile meta-step should run this episode."""
        return self.state.reptile_active

    def should_eval(self, episode: int) -> bool:
        """Whether held-out evaluation should run this episode."""
        if self.state.eval_cadence <= 0:
            return False
        return episode % self.state.eval_cadence == 0

    @property
    def maturity(self) -> float:
        return self.state.maturity

    @property
    def mentor_rate(self) -> float:
        return self.state.mentor_rate

    @property
    def entropy_coef(self) -> float:
        return self.state.entropy_coef

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_anneal(high: float, low: float, progress: float) -> float:
        """Cosine annealing from high → low as progress goes 0 → 1.

        cos(0)=1 → high, cos(π)=-1 → low.
        """
        return low + 0.5 * (high - low) * (1.0 + math.cos(math.pi * progress))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "exploration_boost": self._exploration_boost,
            "boost_reason": self._boost_reason,
            "boost_ttl": self._boost_ttl,
        }
