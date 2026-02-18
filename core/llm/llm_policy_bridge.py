#!/usr/bin/env python3
"""
core/llm/llm_policy_bridge.py — ARIASKA LLM ↔ Policy Bridge v1.0 (Phase 37)

Level 5 GPT ↔ RL Integration Bridge.

This module is the central nervous system connecting LLM intelligence
to the PPO neural network. It converts unstructured LLM outputs into
mathematically precise tensors that directly influence:

  1. **Policy logits** — via action prior vector injection
  2. **State representation** — via LLM feature concatenation
  3. **Value estimation** — via confidence-augmented targets
  4. **Gradient updates** — via KL teacher distillation loss
  5. **Exploration** — via curriculum-driven dynamic shaping

All influence decays over time via the teacher anneal schedule,
allowing the policy to internalize LLM knowledge and become autonomous.

Architecture:
    ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
    │  MicroChain   │────▶│                 │────▶│ PPO Policy   │
    │  PhaseGuide   │     │ LLMPolicyBridge │     │ select_action│
    │  MentorTrace  │────▶│                 │────▶│ + logit prior│
    │  SmartMentor  │     │ • prior vector  │     ├──────────────┤
    └──────────────┘     │ • llm features  │────▶│ PPO Critic   │
                          │ • teacher dist  │     │ + value reg  │
                          │ • anneal alpha  │     ├──────────────┤
                          │ • curriculum    │────▶│ PPO update() │
                          └─────────────────┘     │ + KL teacher │
                                                  │ + ranking    │
                                                  │ + value_reg  │
                                                  └──────────────┘

Author: Filip Volf — Phase 37 Level 5 Integration
"""

import math
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("ariaska.llm_policy_bridge")

# ── Constants ────────────────────────────────────────────────────────────
LLM_FEATURE_DIM = 256          # Additional dims appended to state
ENHANCED_STATE_DIM = 512 + LLM_FEATURE_DIM  # 768 total

# Anneal schedule constants
PRIOR_ALPHA_INIT = 0.50        # Initial LLM prior weight (50% influence)
PRIOR_ALPHA_MIN = 0.02         # Minimum (never fully remove — safety net)
KL_TEACHER_COEF_INIT = 0.15    # Initial KL regularization weight
KL_TEACHER_COEF_MIN = 0.01     # Minimum KL weight
VALUE_REG_COEF = 0.10          # Value alignment regularization weight
RANKING_LOSS_COEF = 0.05       # Ranking margin loss weight

# Maturity thresholds for anneal acceleration
MATURITY_FAST_DECAY = 0.7      # Above this → decay 2x faster
MATURITY_PLATEAU = 0.9         # Above this → near-minimum influence


@dataclass
class LLMInfluenceState:
    """Tracks the current state of LLM influence on the policy.

    All values are updated each step and logged to the dashboard.
    """
    # Current weights (all decay over time)
    prior_alpha: float = PRIOR_ALPHA_INIT
    kl_teacher_coef: float = KL_TEACHER_COEF_INIT
    value_reg_coef: float = VALUE_REG_COEF
    ranking_loss_coef: float = RANKING_LOSS_COEF

    # Anneal progress
    teacher_anneal_pct: float = 1.0  # 1.0 = full influence, 0.0 = none
    maturity_signal: float = 0.0     # 0.0 = novice, 1.0 = master

    # Maturity inputs (rolling)
    success_rate: float = 0.0
    reward_velocity: float = 0.0
    discovery_efficiency: float = 0.0
    exploit_success_rate: float = 0.0

    # Per-step LLM outputs (latest)
    llm_feature_dim: int = LLM_FEATURE_DIM
    enhanced_state_dim: int = ENHANCED_STATE_DIM

    # Loss tracking (last update)
    last_bc_loss: float = 0.0
    last_kl_teacher_loss: float = 0.0
    last_ranking_loss: float = 0.0
    last_value_reg_loss: float = 0.0

    # Counters
    total_steps: int = 0
    total_episodes: int = 0


@dataclass
class LLMGuidancePacket:
    """Structured LLM guidance for a single step.

    Produced by LLMPolicyBridge.compute_guidance() and consumed by
    PPO's select_action() and update().
    """
    # Action prior: preference distribution over action_dim
    action_prior: Optional[Any] = None  # torch.Tensor (action_dim,)

    # LLM feature vector to concatenate with state
    llm_features: Optional[Any] = None  # torch.Tensor (LLM_FEATURE_DIM,)

    # Teacher soft distribution for KL loss
    teacher_distribution: Optional[Any] = None  # torch.Tensor (action_dim,)

    # Confidence and phase signals for value shaping
    llm_confidence: float = 0.5
    phase_progress: float = 0.0
    exploitation_likelihood: float = 0.0
    anomaly_score: float = 0.0
    exploration_bias: float = 0.5
    risk_estimate: float = 0.5

    # Current anneal weight
    prior_alpha: float = PRIOR_ALPHA_INIT

    # Raw source data  (for logging/debugging)
    source: str = "none"
    timestamp: float = 0.0


class LLMPolicyBridge:
    """Central bridge connecting LLM intelligence to PPO policy networks.

    Responsibilities:
      1. Convert LLM outputs → action prior vectors
      2. Compute LLM feature embeddings for enhanced state
      3. Generate teacher distributions for KL distillation
      4. Manage dynamic anneal schedule (LLM influence decay)
      5. Track maturity signals for curriculum shaping
      6. Provide ablation toggle (FF_LLM_POLICY_BRIDGE)

    Thread-safe: all state mutations are in-place on single-owner objects.
    """

    def __init__(
        self,
        action_dim: int = 5,
        llm_feature_dim: int = LLM_FEATURE_DIM,
        prior_alpha_init: float = PRIOR_ALPHA_INIT,
        prior_alpha_min: float = PRIOR_ALPHA_MIN,
        total_anneal_steps: int = 3000,
    ):
        self.action_dim = action_dim
        self.llm_feature_dim = llm_feature_dim
        self.prior_alpha_init = prior_alpha_init
        self.prior_alpha_min = prior_alpha_min
        self.total_anneal_steps = total_anneal_steps

        self.state = LLMInfluenceState()
        self._step_count = 0
        self._episode_count = 0

        # Rolling maturity trackers
        self._recent_rewards: List[float] = []
        self._recent_successes: List[bool] = []
        self._recent_discoveries: List[int] = []
        self._recent_exploit_results: List[bool] = []
        self._max_window = 50

        # Ablation flag
        self._enabled = True

        logger.info(
            "LLMPolicyBridge initialized: action_dim=%d, llm_feat=%d, "
            "alpha_init=%.2f, anneal_steps=%d",
            action_dim, llm_feature_dim, prior_alpha_init, total_anneal_steps,
        )

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def set_enabled(self, enabled: bool) -> None:
        """Toggle LLM integration on/off for ablation testing."""
        self._enabled = enabled
        if not enabled:
            logger.warning("LLMPolicyBridge DISABLED — ablation mode")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def compute_guidance(
        self,
        state_dict: Dict[str, Any],
        micro_chain_result: Optional[Any] = None,
        phase_guide_result: Optional[Any] = None,
        mentor_trace: Optional[Any] = None,
        mentor_confidence: float = 0.5,
        mentor_top_actions: Optional[List[int]] = None,
        phase: str = "RECON",
        step: int = 0,
        episode: int = 0,
    ) -> LLMGuidancePacket:
        """Compute full LLM guidance packet for this step.

        This is the main entry point called by SmartCoach before
        PPO.select_action(). It produces all tensors needed for
        Level 5 integration.

        Args:
            state_dict: Environment state dictionary
            micro_chain_result: MicroChainResult from MicroChain.score()
            phase_guide_result: PhaseGuidedResult from PhaseGuidedLLM.guide()
            mentor_trace: MentorTrace from latest mentor call
            mentor_confidence: Mentor's confidence in its recommendation
            mentor_top_actions: Action indices the mentor prefers
            phase: Current attack phase name
            step: Current step number
            episode: Current episode number

        Returns:
            LLMGuidancePacket with all integration tensors
        """
        import torch

        if not self._enabled:
            return self._empty_packet()

        self._step_count += 1
        self.state.total_steps = self._step_count

        # ── 1. Compute anneal weight ─────────────────────────────────
        alpha = self._compute_anneal_alpha()
        self.state.prior_alpha = alpha
        self.state.teacher_anneal_pct = alpha / self.prior_alpha_init

        # ── 2. Build action prior vector ─────────────────────────────
        prior = self._build_action_prior(
            mentor_top_actions=mentor_top_actions,
            mentor_confidence=mentor_confidence,
            micro_chain_result=micro_chain_result,
            phase=phase,
        )

        # ── 3. Build LLM feature vector ─────────────────────────────
        llm_features = self._build_llm_features(
            state_dict=state_dict,
            micro_chain_result=micro_chain_result,
            phase_guide_result=phase_guide_result,
            mentor_trace=mentor_trace,
            mentor_confidence=mentor_confidence,
            phase=phase,
            step=step,
        )

        # ── 4. Build teacher distribution ────────────────────────────
        teacher_dist = self._build_teacher_distribution(
            mentor_top_actions=mentor_top_actions,
            mentor_confidence=mentor_confidence,
            micro_chain_result=micro_chain_result,
        )

        # ── 5. Extract scalar signals ────────────────────────────────
        phase_progress = self._phase_to_progress(phase)
        exploit_likelihood = self._compute_exploit_likelihood(
            state_dict, micro_chain_result, phase
        )
        anomaly = self._compute_anomaly_score(state_dict, step)
        exploration_bias = self._compute_exploration_bias()
        risk = self._compute_risk_estimate(state_dict, phase)

        return LLMGuidancePacket(
            action_prior=prior,
            llm_features=llm_features,
            teacher_distribution=teacher_dist,
            llm_confidence=mentor_confidence,
            phase_progress=phase_progress,
            exploitation_likelihood=exploit_likelihood,
            anomaly_score=anomaly,
            exploration_bias=exploration_bias,
            risk_estimate=risk,
            prior_alpha=alpha,
            source="bridge",
            timestamp=time.time(),
        )

    def record_step_outcome(
        self,
        reward: float,
        discoveries: int = 0,
        exploit_success: bool = False,
    ) -> None:
        """Record step outcome for maturity signal computation.

        Called after each step to update the rolling maturity trackers
        that drive the anneal schedule.
        """
        self._recent_rewards.append(reward)
        self._recent_successes.append(reward > 0)
        self._recent_discoveries.append(discoveries)
        self._recent_exploit_results.append(exploit_success)

        # Trim to window
        if len(self._recent_rewards) > self._max_window:
            self._recent_rewards = self._recent_rewards[-self._max_window:]
            self._recent_successes = self._recent_successes[-self._max_window:]
            self._recent_discoveries = self._recent_discoveries[-self._max_window:]
            self._recent_exploit_results = self._recent_exploit_results[-self._max_window:]

        # Update maturity signals
        self._update_maturity()

    def record_episode_end(self) -> None:
        """Called at episode boundary to update episode-level stats."""
        self._episode_count += 1
        self.state.total_episodes = self._episode_count

    def get_influence_snapshot(self) -> Dict[str, Any]:
        """Return current influence state for dashboard display."""
        s = self.state
        return {
            "prior_alpha": round(s.prior_alpha, 4),
            "kl_teacher_coef": round(s.kl_teacher_coef, 4),
            "value_reg_coef": round(s.value_reg_coef, 4),
            "ranking_loss_coef": round(s.ranking_loss_coef, 4),
            "teacher_anneal_pct": round(s.teacher_anneal_pct, 3),
            "maturity_signal": round(s.maturity_signal, 3),
            "success_rate": round(s.success_rate, 3),
            "reward_velocity": round(s.reward_velocity, 3),
            "discovery_efficiency": round(s.discovery_efficiency, 3),
            "exploit_success_rate": round(s.exploit_success_rate, 3),
            "llm_feature_dim": s.llm_feature_dim,
            "enhanced_state_dim": s.enhanced_state_dim,
            "bc_loss": round(s.last_bc_loss, 6),
            "kl_teacher_loss": round(s.last_kl_teacher_loss, 6),
            "ranking_loss": round(s.last_ranking_loss, 6),
            "value_reg_loss": round(s.last_value_reg_loss, 6),
            "enabled": self._enabled,
            "total_steps": s.total_steps,
        }

    # =====================================================================
    # INTERNAL: Anneal Schedule
    # =====================================================================

    def _compute_anneal_alpha(self) -> float:
        """Dynamic teacher anneal based on maturity signal.

        Schedule:
            Step 0        → alpha = 0.50 (60% LLM influence)
            Mid training  → alpha = 0.25 (30% influence)
            Late training → alpha = 0.10 (10% influence)
            Mastery       → alpha = 0.02 (5% influence, safety net)

        Maturity accelerates decay:
            Low maturity (struggling) → slow decay
            High maturity (succeeding) → fast decay
        """
        # Base schedule: cosine decay from init to min
        progress = min(1.0, self._step_count / max(self.total_anneal_steps, 1))

        # Cosine decay (smooth, stays high longer)
        base_alpha = self.prior_alpha_min + 0.5 * (
            self.prior_alpha_init - self.prior_alpha_min
        ) * (1.0 + math.cos(math.pi * progress))

        # Maturity acceleration: if agent is succeeding, decay faster
        maturity = self.state.maturity_signal
        if maturity > MATURITY_PLATEAU:
            # Near mastery — accelerate to minimum
            base_alpha = max(self.prior_alpha_min, base_alpha * 0.3)
        elif maturity > MATURITY_FAST_DECAY:
            # Getting good — moderate acceleration
            accel = 1.0 + (maturity - MATURITY_FAST_DECAY) * 3.0
            base_alpha = max(self.prior_alpha_min, base_alpha / accel)

        # Struggling boost: if agent is failing, slow the decay
        if self.state.success_rate < 0.2 and self._step_count > 50:
            base_alpha = min(self.prior_alpha_init * 0.8, base_alpha * 1.5)

        # Also decay the auxiliary loss coefficients proportionally
        decay_factor = base_alpha / self.prior_alpha_init
        self.state.kl_teacher_coef = max(
            KL_TEACHER_COEF_MIN,
            KL_TEACHER_COEF_INIT * decay_factor,
        )
        self.state.value_reg_coef = VALUE_REG_COEF * max(0.1, decay_factor)
        self.state.ranking_loss_coef = RANKING_LOSS_COEF * max(0.1, decay_factor)

        return base_alpha

    def _update_maturity(self) -> None:
        """Update the maturity signal from rolling performance metrics.

        maturity = 0.4 * success_rate
                 + 0.3 * reward_velocity (normalized)
                 + 0.2 * discovery_efficiency
                 + 0.1 * exploit_success_rate
        """
        n = len(self._recent_rewards)
        if n < 5:
            return

        # Success rate
        sr = sum(1 for s in self._recent_successes if s) / max(n, 1)
        self.state.success_rate = sr

        # Reward velocity (are rewards trending up?)
        if n >= 10:
            first_half = self._recent_rewards[:n // 2]
            second_half = self._recent_rewards[n // 2:]
            rv = (
                (sum(second_half) / len(second_half))
                - (sum(first_half) / len(first_half))
            )
            # Normalize to [0, 1] — positive velocity = good
            self.state.reward_velocity = min(1.0, max(0.0, (rv + 5.0) / 10.0))
        else:
            self.state.reward_velocity = 0.5

        # Discovery efficiency (discoveries per step)
        total_disc = sum(self._recent_discoveries[-20:])
        steps = min(20, n)
        self.state.discovery_efficiency = min(1.0, total_disc / max(steps, 1))

        # Exploit success rate
        exploit_results = self._recent_exploit_results[-20:]
        if exploit_results:
            self.state.exploit_success_rate = sum(
                1 for e in exploit_results if e
            ) / len(exploit_results)

        # Composite maturity signal
        self.state.maturity_signal = min(1.0, max(0.0,
            0.4 * self.state.success_rate
            + 0.3 * self.state.reward_velocity
            + 0.2 * self.state.discovery_efficiency
            + 0.1 * self.state.exploit_success_rate
        ))

    # =====================================================================
    # INTERNAL: Action Prior
    # =====================================================================

    def _build_action_prior(
        self,
        mentor_top_actions: Optional[List[int]],
        mentor_confidence: float,
        micro_chain_result: Optional[Any],
        phase: str,
    ) -> Any:
        """Build action prior vector from LLM guidance.

        Converts mentor/MicroChain action preferences into a soft
        probability-like vector over action_dim. This gets added to
        PPO's raw logits (before softmax) weighted by alpha.

        Returns:
            torch.Tensor of shape (action_dim,) or None
        """
        import torch

        prior = torch.zeros(self.action_dim, dtype=torch.float32)

        if mentor_top_actions:
            # Assign decaying preference to mentor's ranked actions
            for rank, action_idx in enumerate(mentor_top_actions[:3]):
                if 0 <= action_idx < self.action_dim:
                    # Top action gets highest boost, decreasing for lower ranks
                    boost = mentor_confidence * (1.0 - rank * 0.3)
                    prior[action_idx] += boost * 2.0  # Scale to logit space

        # MicroChain candidate influence
        if micro_chain_result is not None:
            mc_score = getattr(micro_chain_result, 'score', 0.0)
            mc_phase_fit = getattr(micro_chain_result, 'phase_fit', 0.0)
            selected = getattr(micro_chain_result, 'selected', None)
            if selected:
                sel_score = getattr(selected, 'score', mc_score)
                # Boost the most phase-appropriate action group
                phase_group = self._phase_to_group(phase)
                if 0 <= phase_group < self.action_dim:
                    prior[phase_group] += sel_score * mc_phase_fit * 1.5

        # Phase-based prior: gently bias toward phase-appropriate actions
        phase_group = self._phase_to_group(phase)
        if 0 <= phase_group < self.action_dim:
            prior[phase_group] += 0.3  # Small constant phase preference

        # Normalize: ensure prior doesn't dominate logits
        prior_max = prior.abs().max()
        if prior_max > 3.0:
            prior = prior * (3.0 / prior_max)

        return prior

    def _build_teacher_distribution(
        self,
        mentor_top_actions: Optional[List[int]],
        mentor_confidence: float,
        micro_chain_result: Optional[Any],
    ) -> Any:
        """Build soft teacher distribution for KL distillation.

        Returns a proper probability distribution (sums to 1) over
        the action space, representing the teacher's belief about
        optimal action selection.

        Returns:
            torch.Tensor of shape (action_dim,) — soft probabilities
        """
        import torch

        # Start with uniform
        logits = torch.zeros(self.action_dim, dtype=torch.float32)

        if mentor_top_actions:
            for rank, action_idx in enumerate(mentor_top_actions[:3]):
                if 0 <= action_idx < self.action_dim:
                    logits[action_idx] += mentor_confidence * (3.0 - rank)

        if micro_chain_result is not None:
            selected = getattr(micro_chain_result, 'selected', None)
            if selected:
                candidates = getattr(micro_chain_result, 'candidates', [])
                for cand in candidates[:3]:
                    # Map candidate to action group via phase
                    score = getattr(cand, 'score', 0.5)
                    phase_fit = getattr(cand, 'phase_fit', 0.5)
                    # Spread across action dim proportionally
                    for i in range(self.action_dim):
                        logits[i] += score * phase_fit * 0.3 / self.action_dim

        # If no guidance → return None (skip KL loss)
        if logits.abs().max() < 0.01:
            return None

        # Softmax to get proper probabilities
        teacher_probs = torch.softmax(logits / max(0.1, 1.0 - mentor_confidence + 0.3), dim=-1)
        return teacher_probs

    # =====================================================================
    # INTERNAL: LLM Feature Vector
    # =====================================================================

    def _build_llm_features(
        self,
        state_dict: Dict[str, Any],
        micro_chain_result: Optional[Any],
        phase_guide_result: Optional[Any],
        mentor_trace: Optional[Any],
        mentor_confidence: float,
        phase: str,
        step: int,
    ) -> Any:
        """Build LLM feature vector for enhanced state representation.

        Encodes structured LLM signals into a fixed-size tensor that
        gets concatenated with the base 512-dim state vector, producing
        a 768-dim enhanced state.

        Feature layout (256 dims):
            [0-4]    Phase guidance signals (5)
            [5-9]    MicroChain scores (5)
            [10-14]  Mentor signals (5)
            [15-19]  Anneal & maturity (5)
            [20-24]  Exploration signals (5)
            [25-29]  Risk & anomaly (5)
            [30-34]  Action prior summary (5)
            [35-39]  Temporal features (5)
            [40-255] Reserved zero-padded (216)

        Returns:
            torch.Tensor of shape (LLM_FEATURE_DIM,)
        """
        import torch

        features = torch.zeros(self.llm_feature_dim, dtype=torch.float32)

        # [0-4] Phase guidance signals
        if phase_guide_result is not None:
            features[0] = getattr(phase_guide_result, 'confidence', 0.5)
            features[1] = self._phase_to_progress(phase)
            features[2] = getattr(phase_guide_result, 'stay_score', 0.5)
            features[3] = getattr(phase_guide_result, 'move_score', 0.5)
            features[4] = float(getattr(phase_guide_result, 'escalated', False))
        else:
            features[1] = self._phase_to_progress(phase)

        # [5-9] MicroChain scores
        if micro_chain_result is not None:
            selected = getattr(micro_chain_result, 'selected', None)
            if selected:
                features[5] = getattr(selected, 'score', 0.0)
                features[6] = getattr(selected, 'phase_fit', 0.0)
                features[7] = getattr(selected, 'evidence_support', 0.0)
                features[8] = getattr(selected, 'novelty', 0.0)
            features[9] = float(getattr(micro_chain_result, 'escalated', False))

        # [10-14] Mentor signals
        features[10] = mentor_confidence
        if mentor_trace is not None:
            features[11] = getattr(mentor_trace, 'confidence', 0.5)
            features[12] = float(getattr(mentor_trace, 'mentor_was_correct', False))
            features[13] = float(getattr(mentor_trace, 'produced_discovery', False))
            features[14] = getattr(mentor_trace, 'actual_reward', 0.0) / 50.0  # normalize

        # [15-19] Anneal & maturity
        features[15] = self.state.prior_alpha
        features[16] = self.state.teacher_anneal_pct
        features[17] = self.state.maturity_signal
        features[18] = self.state.success_rate
        features[19] = self.state.reward_velocity

        # [20-24] Exploration signals
        features[20] = self._compute_exploration_bias()
        features[21] = self.state.discovery_efficiency
        features[22] = self.state.exploit_success_rate
        features[23] = min(1.0, step / 500.0)  # step progress
        features[24] = min(1.0, self._episode_count / 100.0)  # episode progress

        # [25-29] Risk & anomaly
        features[25] = self._compute_risk_estimate(state_dict, phase)
        features[26] = self._compute_anomaly_score(state_dict, step)
        features[27] = self._compute_exploit_likelihood(
            state_dict, micro_chain_result, phase
        )
        detection_risk = state_dict.get("detection_risk", 0.0)
        features[28] = min(1.0, float(detection_risk))
        features[29] = float(state_dict.get("blue_team_alert_level", 0.0)) / 10.0

        # [30-34] Action prior summary (compressed from action_dim)
        prior = self._build_action_prior(None, mentor_confidence, micro_chain_result, phase)
        if prior is not None:
            features[30] = prior.max().item()
            features[31] = prior.min().item()
            features[32] = prior.mean().item()
            features[33] = prior.std().item()
            features[34] = float(prior.argmax().item()) / max(self.action_dim - 1, 1)

        # [35-39] Temporal features
        features[35] = min(1.0, len(self._recent_rewards) / self._max_window)
        if self._recent_rewards:
            features[36] = max(-1.0, min(1.0, self._recent_rewards[-1] / 50.0))
        features[37] = float(len(self._recent_discoveries) > 0 and self._recent_discoveries[-1] > 0)
        features[38] = min(1.0, self._step_count / max(self.total_anneal_steps, 1))
        features[39] = float(self._step_count % 10) / 10.0  # cycle position

        return features

    # =====================================================================
    # INTERNAL: Helper Functions
    # =====================================================================

    def _empty_packet(self) -> LLMGuidancePacket:
        """Return an empty guidance packet (for ablation mode)."""
        import torch
        return LLMGuidancePacket(
            action_prior=torch.zeros(self.action_dim, dtype=torch.float32),
            llm_features=torch.zeros(self.llm_feature_dim, dtype=torch.float32),
            teacher_distribution=None,
            prior_alpha=0.0,
            source="disabled",
        )

    def _phase_to_progress(self, phase: str) -> float:
        """Convert phase name to [0, 1] progress value."""
        phase_order = {
            "RECON": 0.0, "ENUMERATION": 0.125,
            "EXPLOITATION": 0.25, "PRIVILEGE_ESCALATION": 0.375,
            "LATERAL_MOVEMENT": 0.5, "POST_EXPLOITATION": 0.625,
            "EXFILTRATION": 0.75, "CLOSEOUT": 1.0,
        }
        return phase_order.get(phase.upper(), 0.0)

    def _phase_to_group(self, phase: str) -> int:
        """Map phase to action group index (0-4)."""
        groups = {
            "RECON": 0, "ENUMERATION": 0,
            "EXPLOITATION": 1, "PRIVILEGE_ESCALATION": 2,
            "LATERAL_MOVEMENT": 3, "POST_EXPLOITATION": 3,
            "EXFILTRATION": 4, "CLOSEOUT": 4,
        }
        return groups.get(phase.upper(), 0)

    def _compute_exploit_likelihood(
        self,
        state_dict: Dict[str, Any],
        micro_chain_result: Optional[Any],
        phase: str,
    ) -> float:
        """Estimate how close the agent is to exploitation success."""
        score = 0.0
        # Have ports → 0.2
        ports = state_dict.get("ports_discovered", state_dict.get("ports", set()))
        if ports:
            score += 0.2
        # Have services → 0.2
        services = state_dict.get("services", set())
        if services:
            score += 0.2
        # Have credentials → 0.3
        creds = state_dict.get("credentials", set())
        if creds:
            score += 0.3
        # Already in exploit phase → 0.3
        if phase.upper() in ("EXPLOITATION", "PRIVILEGE_ESCALATION"):
            score += 0.3
        return min(1.0, score)

    def _compute_anomaly_score(
        self, state_dict: Dict[str, Any], step: int,
    ) -> float:
        """Detect stagnation or anomalous patterns."""
        if len(self._recent_rewards) < 5:
            return 0.0
        # Stagnation: many consecutive zero/negative rewards
        recent = self._recent_rewards[-10:]
        neg_count = sum(1 for r in recent if r <= 0)
        return min(1.0, neg_count / max(len(recent), 1))

    def _compute_exploration_bias(self) -> float:
        """How much the agent should explore vs exploit."""
        # Early: explore more (high bias)
        # Late / high maturity: exploit more (low bias)
        base = max(0.1, 1.0 - self.state.maturity_signal)
        # Boost if stagnating
        if self.state.success_rate < 0.2:
            base = min(1.0, base * 1.5)
        return base

    def _compute_risk_estimate(
        self, state_dict: Dict[str, Any], phase: str,
    ) -> float:
        """Estimate current risk level."""
        risk = 0.0
        risk += float(state_dict.get("detection_risk", 0.0)) * 0.4
        risk += float(state_dict.get("blue_team_alert_level", 0.0)) / 10.0 * 0.3
        if phase.upper() in ("EXPLOITATION", "PRIVILEGE_ESCALATION"):
            risk += 0.3
        return min(1.0, risk)

    # =====================================================================
    # CURRICULUM SHAPING
    # =====================================================================

    def get_curriculum_adjustments(self) -> Dict[str, float]:
        """Return dynamic curriculum adjustments based on maturity.

        Used by SmartCoach to dynamically adjust:
          - exploration_rate
          - burst_budget
          - command_diversity_pressure
          - anti_repeat_threshold

        Returns:
            Dict of adjustment factors
        """
        m = self.state.maturity_signal

        return {
            # Exploration: high when immature, low when mastered
            "exploration_rate": max(0.05, 0.8 * (1.0 - m)),
            # Burst budget: allow more bursts when struggling
            "burst_budget_multiplier": 1.0 + (1.0 - m) * 0.5,
            # Diversity pressure: high early, relaxed later
            "diversity_pressure": max(0.1, 0.9 * (1.0 - m * 0.7)),
            # Anti-repeat: strict early, relaxed when focused
            "anti_repeat_threshold": max(2, int(5 * (1.0 - m * 0.5))),
            # Mentor query rate multiplier
            "mentor_rate_multiplier": max(0.1, 1.0 - m * 0.8),
        }
