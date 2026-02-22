"""
Phase 50 — DecisionPacket: Unified decision envelope.

Single data structure that flows orchestrator → coach → post-step,
carrying per-step signals from every algorithm module:

  Orchestrator populates: episode context, state, RND intrinsic, coherence
  Coach populates:        PPO proposal, SAC proposal, DDQN macro, CognitionNode vote
  Post-step populates:    reward breakdown, source attribution, grad norms

This replaces the ad-hoc dict-passing between SmartOrchestrator and SmartCoach,
providing ONE object for transition storage, reward composition, and attribution.

No top-level cross-module imports — only stdlib + typing + dataclasses.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.decision_packet")


# ---------------------------------------------------------------------------
# Sub-structures (lightweight, no cross-module deps)
# ---------------------------------------------------------------------------

@dataclass
class PPOProposal:
    """PPO policy network output for this step."""
    action_idx: int = -1
    log_prob: float = 0.0
    value: float = 0.0
    entropy: float = 0.0
    command: str = ""
    template_name: str = ""
    confidence: float = 0.0
    # Phase-gated head info (HRL-lite)
    head_group: str = ""  # "recon", "exploit", "post_exploit"

    @property
    def valid(self) -> bool:
        return self.action_idx >= 0


@dataclass
class SACProposal:
    """SAC policy output for this step (when enabled)."""
    action_idx: int = -1
    log_prob: float = 0.0
    q_value: float = 0.0
    command: str = ""
    template_name: str = ""
    confidence: float = 0.0
    alpha: float = 0.0  # Current entropy temperature

    @property
    def valid(self) -> bool:
        return self.action_idx >= 0


@dataclass
class DDQNMacro:
    """DDQN macro-action intent for this step."""
    macro_idx: int = -1
    q_value: float = 0.0
    macro_name: str = ""
    confidence: float = 0.0
    steps_remaining: int = 0  # Steps left in current macro

    @property
    def valid(self) -> bool:
        return self.macro_idx >= 0


@dataclass
class CognitionVote:
    """CognitionNode multi-brain fusion result (when enabled)."""
    fused_action_idx: int = -1
    gate_confidence: float = 0.0
    ppo_weight: float = 0.0
    sac_weight: float = 0.0
    ddqn_weight: float = 0.0
    command: str = ""
    template_name: str = ""
    reasoning: str = ""

    @property
    def valid(self) -> bool:
        return self.fused_action_idx >= 0


@dataclass
class RNDSignal:
    """RND intrinsic motivation signal for this step."""
    intrinsic_reward: float = 0.0
    novelty_score: float = 0.0
    phase_decay: float = 1.0  # Phase-aware decay applied

    @property
    def valid(self) -> bool:
        return self.intrinsic_reward != 0.0


@dataclass
class RewardComposition:
    """Unified reward breakdown — ONE composition point for all reward sources."""
    extrinsic: float = 0.0       # Environment reward
    intrinsic_rnd: float = 0.0   # RND curiosity bonus
    discovery_bonus: float = 0.0  # Discovery-type bonuses
    phase_bonus: float = 0.0      # Phase progression reward
    repeat_penalty: float = 0.0   # Anti-repeat penalty (negative)
    mentor_bonus: float = 0.0     # Mentor imitation bonus
    sil_bonus: float = 0.0        # Self-imitation learning bonus
    bc_bonus: float = 0.0         # Behavioral cloning bonus
    coherence_bonus: float = 0.0  # Coherence-based reward shaping

    @property
    def total(self) -> float:
        return (
            self.extrinsic
            + self.intrinsic_rnd
            + self.discovery_bonus
            + self.phase_bonus
            + self.repeat_penalty
            + self.mentor_bonus
            + self.sil_bonus
            + self.bc_bonus
            + self.coherence_bonus
        )


@dataclass
class SourceAttribution:
    """Tracks which pipeline stage produced the final command."""
    # The winning source
    source: str = "unknown"
    # Full pipeline trace: each stage that ran, what it proposed, why it won/lost
    pipeline_trace: List[Dict[str, Any]] = field(default_factory=list)
    # Win-rate EMA per source (populated post-step)
    source_win_rates: Dict[str, float] = field(default_factory=dict)

    def record_stage(
        self,
        stage: str,
        proposed: str = "",
        accepted: bool = False,
        reason: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Record a pipeline stage evaluation."""
        self.pipeline_trace.append({
            "stage": stage,
            "proposed": proposed[:120],
            "accepted": accepted,
            "reason": reason,
            "confidence": round(confidence, 4),
            "ts": time.monotonic(),
        })
        if accepted:
            self.source = stage


@dataclass
class GradNorms:
    """Per-loss gradient norms from the most recent PPO update (populated post-update)."""
    policy_grad_norm: float = 0.0
    value_grad_norm: float = 0.0
    entropy_grad_norm: float = 0.0
    bc_grad_norm: float = 0.0
    sil_grad_norm: float = 0.0
    kl_teacher_grad_norm: float = 0.0
    ranking_grad_norm: float = 0.0
    total_grad_norm: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "policy": self.policy_grad_norm,
            "value": self.value_grad_norm,
            "entropy": self.entropy_grad_norm,
            "bc": self.bc_grad_norm,
            "sil": self.sil_grad_norm,
            "kl_teacher": self.kl_teacher_grad_norm,
            "ranking": self.ranking_grad_norm,
            "total": self.total_grad_norm,
        }


# ---------------------------------------------------------------------------
# Main schema
# ---------------------------------------------------------------------------

@dataclass
class DecisionPacket:
    """
    Unified decision envelope — the SINGLE data structure that flows through
    the entire orchestrator → coach → post-step pipeline.

    Lifecycle:
    1. Orchestrator creates DecisionPacket with episode context + RND signal
    2. Coach.decide() receives it, populates PPO/SAC/DDQN/CognitionNode proposals
    3. Coach selects winning command, sets source attribution
    4. Post-step: reward composition populated, transitions stored from packet fields
    5. End-of-episode: grad norms populated from PPO.update()

    All algorithm modules read/write to this ONE object. No more scattered dicts.
    """

    # ── Episode context (set by orchestrator) ─────────────────────────
    episode: int = 0
    step: int = 0
    agent_name: str = ""
    phase: str = "recon"
    target_ip: str = ""

    # Raw state dict (enriched with discovery board)
    state: Dict[str, Any] = field(default_factory=dict)

    # State tensor (512-dim, set by coach/orchestrator when encoding)
    state_tensor: Optional[Any] = None  # torch.Tensor, optional import

    # Attack context reference (set by orchestrator — typed as Any to avoid import)
    attack_context: Optional[Any] = None

    # ── Algorithm proposals (set by coach during decide()) ────────────
    ppo: PPOProposal = field(default_factory=PPOProposal)
    sac: SACProposal = field(default_factory=SACProposal)
    ddqn: DDQNMacro = field(default_factory=DDQNMacro)
    cognition: CognitionVote = field(default_factory=CognitionVote)

    # ── Intrinsic motivation (set by orchestrator before coach) ───────
    rnd: RNDSignal = field(default_factory=RNDSignal)

    # ── Coherence signal (set by orchestrator) ────────────────────────
    coherence: float = 0.5
    macro_confidence: float = 0.0

    # ── Final decision (set by coach after pipeline runs) ─────────────
    chosen_command: str = ""
    chosen_template: str = ""
    chosen_confidence: float = 0.5

    # ── Source attribution (set by coach pipeline) ────────────────────
    attribution: SourceAttribution = field(default_factory=SourceAttribution)

    # ── Reward composition (set post-step by orchestrator/coach) ──────
    reward: RewardComposition = field(default_factory=RewardComposition)

    # ── Gradient norms (set post-update by PPO) ───────────────────────
    grad_norms: GradNorms = field(default_factory=GradNorms)

    # ── Metadata ──────────────────────────────────────────────────────
    timestamp: float = field(default_factory=time.time)
    decision_trace: List[Dict[str, Any]] = field(default_factory=list)

    # ── Flags ─────────────────────────────────────────────────────────
    mentor_called: bool = False
    evidence_gate_passed: bool = True
    forced_novel: bool = False
    repeat_blocked: bool = False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """Shortcut to attribution source."""
        return self.attribution.source

    @property
    def total_reward(self) -> float:
        """Shortcut to composed reward total."""
        return self.reward.total

    @property
    def has_ppo(self) -> bool:
        return self.ppo.valid

    @property
    def has_sac(self) -> bool:
        return self.sac.valid

    @property
    def has_ddqn(self) -> bool:
        return self.ddqn.valid

    @property
    def has_cognition(self) -> bool:
        return self.cognition.valid

    @property
    def has_rnd(self) -> bool:
        return self.rnd.valid

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for tracing/logging."""
        return {
            "episode": self.episode,
            "step": self.step,
            "agent": self.agent_name,
            "phase": self.phase,
            "target": self.target_ip,
            "chosen_command": self.chosen_command[:200],
            "chosen_template": self.chosen_template,
            "confidence": round(self.chosen_confidence, 4),
            "source": self.source,
            "ppo": {
                "action": self.ppo.action_idx,
                "log_prob": round(self.ppo.log_prob, 4),
                "value": round(self.ppo.value, 4),
                "command": self.ppo.command[:120],
            } if self.has_ppo else None,
            "sac": {
                "action": self.sac.action_idx,
                "q_value": round(self.sac.q_value, 4),
                "command": self.sac.command[:120],
            } if self.has_sac else None,
            "ddqn": {
                "macro_idx": self.ddqn.macro_idx,
                "macro_name": self.ddqn.macro_name,
                "q_value": round(self.ddqn.q_value, 4),
            } if self.has_ddqn else None,
            "cognition": {
                "fused_action": self.cognition.fused_action_idx,
                "gate_conf": round(self.cognition.gate_confidence, 4),
                "weights": {
                    "ppo": round(self.cognition.ppo_weight, 3),
                    "sac": round(self.cognition.sac_weight, 3),
                    "ddqn": round(self.cognition.ddqn_weight, 3),
                },
            } if self.has_cognition else None,
            "rnd": {
                "intrinsic": round(self.rnd.intrinsic_reward, 4),
                "novelty": round(self.rnd.novelty_score, 4),
            } if self.has_rnd else None,
            "reward": {
                "total": round(self.reward.total, 4),
                "extrinsic": round(self.reward.extrinsic, 4),
                "intrinsic_rnd": round(self.reward.intrinsic_rnd, 4),
                "discovery": round(self.reward.discovery_bonus, 4),
                "phase": round(self.reward.phase_bonus, 4),
            },
            "coherence": round(self.coherence, 4),
            "mentor_called": self.mentor_called,
            "evidence_gate_passed": self.evidence_gate_passed,
            "forced_novel": self.forced_novel,
            "attribution_trace_len": len(self.attribution.pipeline_trace),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_step_context(
        cls,
        step_ctx: Any,
        *,
        rnd_intrinsic: float = 0.0,
        coherence: float = 0.5,
        macro_confidence: float = 0.0,
    ) -> "DecisionPacket":
        """
        Create a DecisionPacket from an existing SmartStepContext.

        This is the primary factory for orchestrator usage — preserves full
        backwards compatibility with existing SmartStepContext construction.
        """
        phase = "recon"
        target_ip = ""
        attack_context = None
        if hasattr(step_ctx, "attack_context") and step_ctx.attack_context is not None:
            attack_context = step_ctx.attack_context
            if hasattr(attack_context, "current_phase"):
                phase = attack_context.current_phase.name.lower()
            if hasattr(attack_context, "target"):
                target_ip = attack_context.target

        return cls(
            episode=getattr(step_ctx, "episode", 0),
            step=getattr(step_ctx, "step", 0),
            agent_name=getattr(step_ctx, "agent_name", ""),
            phase=phase,
            target_ip=target_ip,
            state=getattr(step_ctx, "state", {}),
            attack_context=attack_context,
            rnd=RNDSignal(intrinsic_reward=rnd_intrinsic),
            coherence=coherence,
            macro_confidence=macro_confidence,
        )

    def to_step_context(self) -> Dict[str, Any]:
        """
        Extract a SmartStepContext-compatible dict for legacy code paths.

        Used when old code expects step_ctx fields directly.
        """
        return {
            "episode": self.episode,
            "step": self.step,
            "agent_name": self.agent_name,
            "attack_context": self.attack_context,
            "state": self.state,
        }
