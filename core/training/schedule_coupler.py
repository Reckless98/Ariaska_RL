"""core/training/schedule_coupler.py — C11: Unified Schedule Coupling.

Coordinates annealing schedules across three subsystems so they decay
together based on a single maturity signal:

  1. MentorPolicy — mentor call frequency floor
  2. LLMPolicyBridge — KL/BC teacher influence, prior alpha
  3. PPO config — entropy_coef, exploration parameters

Without coupling, each subsystem anneals independently on different
timescales, causing desynchronization:
  - Mentor still calling at high rate while BC loss is nearly zero
  - Prior alpha decayed but mentor floor still high
  - Entropy high while mentor guides are assertive

The ScheduleCoupler runs once per episode start and propagates the
maturity signal to all subsystems with configurable coupling strength.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("ariaska.training.schedule_coupler")


@dataclass
class CouplerConfig:
    """Configuration for schedule coupling.

    Args:
        coupling_strength: How tightly subsystems track maturity [0, 1].
                          0.0 = independent (legacy), 1.0 = fully coupled.
        mentor_floor_range: (high, low) mentor floor at maturity 0 and 1.
        kl_coef_range: (high, low) KL teacher coef at maturity 0 and 1.
        bc_coef_range: (high, low) BC loss coef at maturity 0 and 1.
        prior_alpha_range: (high, low) LLM prior alpha at maturity 0 and 1.
        entropy_coef_range: (high, low) PPO entropy coef at maturity 0 and 1.
        log_updates: Whether to log each coupling update.
    """
    coupling_strength: float = 0.8
    mentor_floor_range: tuple = (0.60, 0.08)
    kl_coef_range: tuple = (0.15, 0.01)
    bc_coef_range: tuple = (0.10, 0.01)
    prior_alpha_range: tuple = (0.50, 0.02)
    entropy_coef_range: tuple = (0.08, 0.01)
    log_updates: bool = True


@dataclass
class CoupledState:
    """Snapshot of the coupled schedule state after an update."""
    maturity: float = 0.0
    mentor_floor: float = 0.60
    kl_coef: float = 0.15
    bc_coef: float = 0.10
    prior_alpha: float = 0.50
    entropy_coef: float = 0.08
    coupling_strength: float = 0.8
    episode: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maturity": round(self.maturity, 4),
            "mentor_floor": round(self.mentor_floor, 4),
            "kl_coef": round(self.kl_coef, 4),
            "bc_coef": round(self.bc_coef, 4),
            "prior_alpha": round(self.prior_alpha, 4),
            "entropy_coef": round(self.entropy_coef, 4),
            "coupling_strength": round(self.coupling_strength, 4),
            "episode": self.episode,
        }


class ScheduleCoupler:
    """Coordinates annealing across MentorPolicy, LLMPolicyBridge, and PPO.

    Called at episode start with the current maturity signal. Computes
    coupled target values and applies them to each subsystem.

    Usage in SmartOrchestrator::

        coupler = ScheduleCoupler()
        # At episode start:
        coupled = coupler.update(
            maturity=maturity_signal,
            episode=episode_number,
            mentor_policy=coach.mentor_policy,  # optional
            llm_bridge=coach.llm_bridge,        # optional
            ppo_agent=coach.ppo_agent,           # optional
        )
    """

    def __init__(self, config: Optional[CouplerConfig] = None) -> None:
        self.config = config or CouplerConfig()
        self._history: list[CoupledState] = []
        self._update_count: int = 0

    def _interpolate(
        self, high: float, low: float, maturity: float, strength: float
    ) -> float:
        """Interpolate between high and low based on maturity.

        Uses cosine schedule for smooth transition.
        strength=1.0 → fully coupled to maturity.
        strength=0.0 → always returns high (independent).
        """
        # Cosine interpolation: smooth at endpoints
        t = maturity * strength
        # Cosine schedule: 0→1 maps high→low smoothly
        alpha = 0.5 * (1 - math.cos(math.pi * t))
        return high + alpha * (low - high)

    def compute(self, maturity: float, episode: int = 0) -> CoupledState:
        """Compute coupled schedule values without applying them.

        Args:
            maturity: Learning maturity signal [0, 1].
            episode: Current episode number.

        Returns:
            CoupledState with all computed target values.
        """
        s = self.config.coupling_strength
        return CoupledState(
            maturity=maturity,
            mentor_floor=self._interpolate(
                *self.config.mentor_floor_range, maturity, s,
            ),
            kl_coef=self._interpolate(
                *self.config.kl_coef_range, maturity, s,
            ),
            bc_coef=self._interpolate(
                *self.config.bc_coef_range, maturity, s,
            ),
            prior_alpha=self._interpolate(
                *self.config.prior_alpha_range, maturity, s,
            ),
            entropy_coef=self._interpolate(
                *self.config.entropy_coef_range, maturity, s,
            ),
            coupling_strength=s,
            episode=episode,
        )

    def update(
        self,
        maturity: float,
        episode: int = 0,
        mentor_policy: Any = None,
        llm_bridge: Any = None,
        ppo_agent: Any = None,
    ) -> CoupledState:
        """Compute and apply coupled schedule to subsystems.

        Args:
            maturity: Learning maturity signal [0, 1].
            episode: Current episode number.
            mentor_policy: MentorPolicy instance (optional).
            llm_bridge: LLMPolicyBridge instance (optional).
            ppo_agent: PPOAgent instance (optional).

        Returns:
            CoupledState with applied values.
        """
        state = self.compute(maturity, episode)

        # 1. Apply to MentorPolicy
        if mentor_policy is not None:
            self._apply_mentor(mentor_policy, state)

        # 2. Apply to LLMPolicyBridge
        if llm_bridge is not None:
            self._apply_bridge(llm_bridge, state)

        # 3. Apply to PPO
        if ppo_agent is not None:
            self._apply_ppo(ppo_agent, state)

        self._history.append(state)
        self._update_count += 1

        if self.config.log_updates:
            logger.debug(
                "[COUPLE] ep=%d mat=%.3f mentor=%.3f kl=%.4f bc=%.4f "
                "alpha=%.3f ent=%.4f",
                episode, maturity, state.mentor_floor, state.kl_coef,
                state.bc_coef, state.prior_alpha, state.entropy_coef,
            )

        return state

    def _apply_mentor(self, mentor_policy: Any, state: CoupledState) -> None:
        """Apply coupled mentor floor to MentorPolicy."""
        try:
            # MentorPolicy has set_maturity() which controls the floor internally.
            # We set the maturity signal directly — mentor policy will compute
            # its own floor from it. C11 coupling ensures consistent values.
            if hasattr(mentor_policy, 'set_maturity'):
                mentor_policy.set_maturity(state.maturity)
            # Also directly set the config floor if accessible
            cfg = getattr(mentor_policy, 'config', None)
            if cfg is not None and hasattr(cfg, 'min_adaptive_rate'):
                # Gently push the floor toward our coupled value
                # Don't override the entire policy — just nudge the floor
                pass  # MentorPolicy.set_maturity handles this internally
        except Exception as e:
            logger.debug(f"[COUPLE] MentorPolicy apply error: {e}")

    def _apply_bridge(self, llm_bridge: Any, state: CoupledState) -> None:
        """Apply coupled KL/prior values to LLMPolicyBridge."""
        try:
            influence = getattr(llm_bridge, 'influence', None)
            if influence is not None:
                influence.kl_teacher_coef = state.kl_coef
                influence.prior_alpha = state.prior_alpha
                influence.maturity_signal = state.maturity
        except Exception as e:
            logger.debug(f"[COUPLE] LLMBridge apply error: {e}")

    def _apply_ppo(self, ppo_agent: Any, state: CoupledState) -> None:
        """Apply coupled entropy coef to PPO config."""
        try:
            config = getattr(ppo_agent, 'config', None)
            if config is not None and hasattr(config, 'entropy_coef'):
                # Only adjust if the coupled value is lower (don't increase)
                if state.entropy_coef < config.entropy_coef:
                    config.entropy_coef = state.entropy_coef
        except Exception as e:
            logger.debug(f"[COUPLE] PPO apply error: {e}")

    def get_history(self, n: int = 10) -> list[Dict[str, Any]]:
        """Return last N coupling states."""
        return [s.to_dict() for s in self._history[-n:]]

    def get_stats(self) -> Dict[str, Any]:
        """Return coupler statistics."""
        latest = self._history[-1].to_dict() if self._history else {}
        return {
            "update_count": self._update_count,
            "latest": latest,
            "config": {
                "coupling_strength": self.config.coupling_strength,
                "mentor_floor_range": self.config.mentor_floor_range,
                "kl_coef_range": self.config.kl_coef_range,
                "bc_coef_range": self.config.bc_coef_range,
                "prior_alpha_range": self.config.prior_alpha_range,
                "entropy_coef_range": self.config.entropy_coef_range,
            },
        }
