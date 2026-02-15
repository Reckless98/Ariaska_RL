#!/usr/bin/env python3
"""
core/algorithms/cognition_node.py — ARIASKA CognitionNode v1.0

Unified multi-brain decision fusion node.  Orchestrates:
  1. DDQN Macro  — strategic intent (which family of commands?)
  2. PPO Micro   — tactical action (which exact command + params?)
  3. RND Curiosity — intrinsic novelty bonus (is this state new?)
  4. SAC Explorer — entropy-regularised alternative (MaxEnt exploration)
  5. EMA Critics  — exponential-moving-average value baselines
  6. Self-Imitation Learning (SIL) — replay only high-return transitions

All algorithms run in parallel on every step.  A learned *confidence gate*
fuses their outputs into a single command without overlapping or conflicting.

Design invariants:
  • Each sub-brain produces an (action_index, confidence, metadata) triple.
  • The CognitionNode never invents commands — it only selects from the
    approved CommandRegistry via CommandActionMapper.
  • Outputs are deterministic given the same RNG seed and state.
  • Memory produced here feeds directly into the agent's memory sync system.

Phase 8:  Multi-brain integration for humanlike reasoning.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.algorithms.ppo_agent import PPOAgent
    from core.algorithms.sac_agent import SACAgent
    from core.algorithms.ddqn_macro import DDQNMacro
    from core.algorithms.rnd_curiosity import RNDCuriosity

logger = logging.getLogger("ariaska.algorithms.cognition")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BrainVote:
    """Single sub-brain's suggestion for an action."""
    brain_name: str          # "ddqn", "ppo", "sac", "rnd", "sil"
    action_idx: int          # Index into CommandActionMapper.commands
    confidence: float        # 0–1  (higher = more certain)
    q_value: float = 0.0     # Raw Q or value estimate
    entropy: float = 0.0     # Policy entropy (lower = more exploitative)
    novelty: float = 0.0     # RND intrinsic bonus (0 if not RND)
    log_prob: float = 0.0    # Log probability of selected action
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BrainVote({self.brain_name}: a={self.action_idx}, "
            f"c={self.confidence:.2f}, q={self.q_value:.2f})"
        )


@dataclass
class CognitionResult:
    """Fused output of the CognitionNode for one timestep."""
    # Winning action
    action_idx: int
    winning_brain: str       # Which brain's action was selected
    confidence: float        # Fused confidence

    # All votes (for telemetry / JSONL logging)
    votes: List[BrainVote] = field(default_factory=list)

    # Intrinsic reward from RND
    rnd_bonus: float = 0.0

    # EMA baseline values
    ema_value_fast: float = 0.0
    ema_value_slow: float = 0.0

    # Macro intent from DDQN (if any)
    macro_intent: Optional[str] = None
    macro_confidence: float = 0.0

    # SIL replay triggered
    sil_triggered: bool = False
    sil_bonus: float = 0.0

    # PPO trajectory data (for later reward pairing)
    ppo_log_prob: float = 0.0
    ppo_value: float = 0.0
    ppo_state: Optional[torch.Tensor] = None
    ppo_action: int = -1

    # SAC trajectory data
    sac_state: Optional[torch.Tensor] = None
    sac_action: int = -1

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_telemetry(self) -> Dict[str, Any]:
        """Serialise for JSONL watcher."""
        return {
            "action_idx": self.action_idx,
            "winning_brain": self.winning_brain,
            "confidence": round(self.confidence, 4),
            "rnd_bonus": round(self.rnd_bonus, 4),
            "ema_fast": round(self.ema_value_fast, 4),
            "ema_slow": round(self.ema_value_slow, 4),
            "macro_intent": self.macro_intent,
            "macro_confidence": round(self.macro_confidence, 4),
            "sil_triggered": self.sil_triggered,
            "votes": [
                {
                    "brain": v.brain_name,
                    "action": v.action_idx,
                    "conf": round(v.confidence, 3),
                    "q": round(v.q_value, 3),
                    "novelty": round(v.novelty, 3),
                }
                for v in self.votes
            ],
        }


@dataclass
class CognitionConfig:
    """Tunable parameters for the CognitionNode."""

    # ── Confidence gating thresholds ──
    # DDQN macro overrides PPO only when its confidence exceeds this
    # AND either PPO is uncertain or RND novelty is high.
    macro_override_threshold: float = 0.70
    ppo_uncertain_threshold: float = 0.40   # PPO entropy above this = uncertain
    rnd_novelty_high: float = 2.0           # RND bonus above this = very novel

    # ── EMA smoothing ──
    ema_fast_alpha: float = 0.15   # Fast EMA (tracks recent rewards)
    ema_slow_alpha: float = 0.03   # Slow EMA (long-term baseline)

    # ── Self-imitation learning ──
    sil_threshold_pct: float = 0.80  # Replay only transitions above this percentile
    sil_max_buffer: int = 500        # Max golden transitions
    sil_bonus: float = 2.0           # Bonus reward for SIL-replayed transitions

    # ── Brain fusion weights (learned via gate head; initial biases) ──
    initial_ppo_weight: float = 0.50
    initial_sac_weight: float = 0.20
    initial_ddqn_weight: float = 0.20
    initial_rnd_weight: float = 0.10

    # ── Adaptive entropy ──
    entropy_low_bound: float = 0.5   # Below this → too exploitative
    entropy_high_bound: float = 2.0  # Above this → too exploratory


class ConfidenceGate(nn.Module):
    """Learned confidence-gating head.

    Takes concatenated features from all sub-brains and outputs a
    soft weight distribution over brains + a fused confidence scalar.

    Input:  [ppo_conf, ppo_entropy, sac_conf, sac_entropy,
             ddqn_conf, rnd_novelty, ema_fast, ema_slow,
             phase_onehot(8)]  → 16 dims
    Output: [brain_weights(4), fused_confidence(1)] → 5 dims
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, num_brains: int = 4):
        super().__init__()
        self.num_brains = num_brains
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.weight_head = nn.Linear(hidden_dim, num_brains)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (brain_weights [softmax], fused_confidence [sigmoid])."""
        h = self.net(x)
        weights = F.softmax(self.weight_head(h), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(h))
        return weights, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Main CognitionNode
# ─────────────────────────────────────────────────────────────────────────────

class CognitionNode:
    """
    Multi-brain decision fusion node.

    Coordinates DDQN (macro), PPO (micro), SAC (explorer), RND (curiosity),
    EMA critics, and self-imitation replay into a single coherent action
    selection per timestep.

    Usage:
        node = CognitionNode(config, ppo=ppo, sac=sac, ddqn=ddqn, rnd=rnd)
        result = node.think(state_tensor, action_mask, phase_name)
        # result.action_idx → pass to CommandActionMapper
        # After reward: node.observe(result, reward, next_state, done)
    """

    PHASE_INDICES = {
        "RECON": 0, "ENUMERATION": 1, "EXPLOITATION": 2,
        "PRIVILEGE_ESCALATION": 3, "LATERAL_MOVEMENT": 4,
        "POST_EXPLOITATION": 5, "EXFILTRATION": 6, "CLOSEOUT": 7,
    }

    def __init__(
        self,
        config: Optional[CognitionConfig] = None,
        ppo: Optional["PPOAgent"] = None,
        sac: Optional["SACAgent"] = None,
        ddqn: Optional["DDQNMacro"] = None,
        rnd: Optional["RNDCuriosity"] = None,
        device: str = "cpu",
    ):
        self.config = config or CognitionConfig()
        self.ppo = ppo
        self.sac = sac
        self.ddqn = ddqn
        self.rnd = rnd
        self.device = torch.device(device)

        # ── Confidence gate ──
        self.gate = ConfidenceGate(input_dim=16, hidden_dim=32, num_brains=4).to(self.device)
        self.gate_optimizer = torch.optim.Adam(self.gate.parameters(), lr=1e-3)

        # ── EMA critics ──
        self._ema_fast = 0.0
        self._ema_slow = 0.0

        # ── Self-imitation buffer ──
        self._sil_buffer: List[Dict[str, Any]] = []
        self._sil_reward_threshold = 0.0  # Dynamic: updated to percentile

        # ── Metrics ──
        self._step_count = 0
        self._brain_win_counts: Dict[str, int] = {
            "ppo": 0, "sac": 0, "ddqn": 0, "rnd": 0, "sil": 0, "gate": 0,
        }
        self._total_rnd_bonus = 0.0
        self._total_sil_replays = 0

        logger.info(
            f"CognitionNode initialized: "
            f"PPO={'✓' if ppo else '✗'} "
            f"SAC={'✓' if sac else '✗'} "
            f"DDQN={'✓' if ddqn else '✗'} "
            f"RND={'✓' if rnd else '✗'} "
            f"gate_params={sum(p.numel() for p in self.gate.parameters()):,}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Core: think()
    # ─────────────────────────────────────────────────────────────────────

    def think(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        phase: str = "RECON",
        macro_allowed_indices: Optional[Set[int]] = None,
    ) -> CognitionResult:
        """
        Run all sub-brains, fuse their votes, and select a single action.

        Args:
            state: (512,) encoded state tensor.
            action_mask: (action_dim,) bool mask of legal actions.
            phase: Current attack phase name.
            macro_allowed_indices: If DDQN picked a macro, indices of
                commands in that macro's command set (already filtered
                by action_mask).

        Returns:
            CognitionResult with fused action and full telemetry.
        """
        self._step_count += 1
        votes: List[BrainVote] = []
        result = CognitionResult(action_idx=-1, winning_brain="none", confidence=0.0)

        s = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)

        # ── 1. DDQN Macro ──
        ddqn_vote = self._vote_ddqn(s, phase)
        if ddqn_vote is not None:
            votes.append(ddqn_vote)
            result.macro_intent = ddqn_vote.metadata.get("macro_name")
            result.macro_confidence = ddqn_vote.confidence

        # ── 2. PPO Micro ──
        ppo_vote = self._vote_ppo(s, action_mask, macro_allowed_indices)
        if ppo_vote is not None:
            votes.append(ppo_vote)
            result.ppo_log_prob = ppo_vote.log_prob
            result.ppo_value = ppo_vote.q_value
            result.ppo_state = state
            result.ppo_action = ppo_vote.action_idx

        # ── 3. SAC Explorer ──
        sac_vote = self._vote_sac(s, action_mask, macro_allowed_indices)
        if sac_vote is not None:
            votes.append(sac_vote)
            result.sac_state = state
            result.sac_action = sac_vote.action_idx

        # ── 4. RND Curiosity ──
        rnd_bonus = 0.0
        if self.rnd is not None:
            rnd_bonus = self.rnd.compute_intrinsic_reward(state, phase)
            result.rnd_bonus = rnd_bonus
            self._total_rnd_bonus += rnd_bonus

            # If RND signals very high novelty, create a pseudo-vote
            # that biases toward the least-visited action
            if rnd_bonus > self.config.rnd_novelty_high and ppo_vote is not None:
                rnd_vote = BrainVote(
                    brain_name="rnd",
                    action_idx=ppo_vote.action_idx,  # Same as PPO but boosted
                    confidence=min(0.9, rnd_bonus / 5.0),
                    q_value=rnd_bonus,
                    novelty=rnd_bonus,
                )
                votes.append(rnd_vote)

        # ── 5. EMA Baselines ──
        # Updated later via observe(); here we just read current values
        result.ema_value_fast = self._ema_fast
        result.ema_value_slow = self._ema_slow

        # ── 6. Confidence Gate Fusion ──
        if not votes:
            # No brains available — random legal action
            legal = action_mask.nonzero(as_tuple=True)[0]
            if len(legal) > 0:
                idx = legal[torch.randint(len(legal), (1,)).item()].item()
                result.action_idx = idx
                result.winning_brain = "random"
                result.confidence = 0.1
            return result

        result = self._fuse_votes(votes, state, phase, action_mask, result)
        result.votes = votes

        # ── 7. SIL check ──
        sil_result = self._check_sil(state, action_mask, phase)
        if sil_result is not None:
            result.sil_triggered = True
            result.sil_bonus = self.config.sil_bonus
            self._total_sil_replays += 1

        # Track winner
        self._brain_win_counts[result.winning_brain] = (
            self._brain_win_counts.get(result.winning_brain, 0) + 1
        )

        return result

    # ─────────────────────────────────────────────────────────────────────
    # observe() — post-step learning
    # ─────────────────────────────────────────────────────────────────────

    def observe(
        self,
        cognition_result: CognitionResult,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """
        After environment returns reward, update all sub-brains.

        Args:
            cognition_result: The CognitionResult from think().
            reward: Extrinsic reward from environment.
            next_state: Next state tensor (512,).
            done: Whether episode ended.
        """
        total_reward = reward + cognition_result.rnd_bonus + cognition_result.sil_bonus

        # ── EMA update ──
        self._ema_fast += self.config.ema_fast_alpha * (total_reward - self._ema_fast)
        self._ema_slow += self.config.ema_slow_alpha * (total_reward - self._ema_slow)

        # ── PPO trajectory ──
        if self.ppo is not None and cognition_result.ppo_state is not None:
            try:
                self.ppo.store_transition(
                    state=cognition_result.ppo_state,
                    action=cognition_result.ppo_action,
                    log_prob=cognition_result.ppo_log_prob,
                    reward=total_reward,
                    value=cognition_result.ppo_value,
                    done=done,
                )
            except Exception as e:
                logger.debug(f"PPO store failed: {e}")

        # ── SAC replay ──
        if self.sac is not None and cognition_result.sac_state is not None:
            try:
                self.sac.store_transition(
                    state=cognition_result.sac_state,
                    action=cognition_result.sac_action,
                    reward=total_reward,
                    next_state=next_state,
                    done=done,
                )
            except Exception as e:
                logger.debug(f"SAC store failed: {e}")

        # ── DDQN macro ──
        if self.ddqn is not None and cognition_result.macro_intent is not None:
            # DDQN transition stored externally by SmartCoach (it owns the state)
            pass

        # ── RND update predictor ──
        if self.rnd is not None:
            try:
                self.rnd.update(next_state)
            except Exception as e:
                logger.debug(f"RND update failed: {e}")

        # ── SIL buffer ──
        if total_reward > self._sil_reward_threshold:
            self._sil_buffer.append({
                "state": cognition_result.ppo_state,
                "action": cognition_result.action_idx,
                "reward": total_reward,
                "log_prob": cognition_result.ppo_log_prob,
            })
            if len(self._sil_buffer) > self.config.sil_max_buffer:
                # Evict lowest-reward entry
                self._sil_buffer.sort(key=lambda x: x["reward"])
                self._sil_buffer.pop(0)
            # Update threshold to percentile
            if len(self._sil_buffer) >= 10:
                rewards = sorted(x["reward"] for x in self._sil_buffer)
                idx = int(len(rewards) * self.config.sil_threshold_pct)
                self._sil_reward_threshold = rewards[min(idx, len(rewards) - 1)]

        # ── Gate self-training ──
        # The gate learns from actual outcomes: if the winning brain's action
        # led to positive reward, reinforce its weight; else suppress.
        self._train_gate(cognition_result, total_reward)

        # ── Phase 9.1: Mirror to HybridMemory ──
        # All transitions from all brains flow into the shared 3-tier memory
        # for cross-algorithm replay (DDQN, SAC, SIL can all sample from it).
        try:
            from core.memory.hybrid_memory import get_hybrid_memory
            hm = get_hybrid_memory()
            if hm is not None and cognition_result.ppo_state is not None:
                _state_np = cognition_result.ppo_state.detach().cpu().numpy().flatten()
                _next_np = next_state.detach().cpu().numpy().flatten() if next_state is not None else _state_np
                hm.store_transition(
                    state=_state_np,
                    action=cognition_result.action_idx,
                    reward=total_reward,
                    next_state=_next_np,
                    done=done,
                    metadata={
                        "source": f"cognition:{cognition_result.winning_brain}",
                        "rnd_bonus": cognition_result.rnd_bonus,
                        "sil_bonus": cognition_result.sil_bonus,
                    },
                    priority=abs(total_reward) + cognition_result.rnd_bonus + 0.01,
                )
        except Exception:
            pass  # HybridMemory not available — continue

    # ─────────────────────────────────────────────────────────────────────
    # Episode lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset per-episode state (keep learned weights)."""
        self._step_count = 0
        # Don't reset EMA — they are cross-episode
        # Don't reset SIL buffer — it's long-term memory
        logger.debug("CognitionNode episode reset")

    def end_episode(self) -> Dict[str, Any]:
        """Return episode-level metrics and trigger sub-brain updates."""
        metrics = {
            "brain_wins": dict(self._brain_win_counts),
            "total_rnd_bonus": round(self._total_rnd_bonus, 2),
            "sil_buffer_size": len(self._sil_buffer),
            "sil_threshold": round(self._sil_reward_threshold, 2),
            "sil_replays": self._total_sil_replays,
            "ema_fast": round(self._ema_fast, 2),
            "ema_slow": round(self._ema_slow, 2),
        }

        # PPO update happens externally in SmartCoach.end_episode_ppo()
        # SAC update if enough buffer
        if self.sac is not None:
            try:
                sac_loss = self.sac.update()
                if sac_loss is not None:
                    metrics["sac_loss"] = round(sac_loss, 4)
            except Exception as e:
                logger.debug(f"SAC end-episode update failed: {e}")

        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # Private: sub-brain vote methods
    # ─────────────────────────────────────────────────────────────────────

    def _vote_ddqn(
        self, state: torch.Tensor, phase: str,
    ) -> Optional[BrainVote]:
        """Get DDQN macro-level vote."""
        if self.ddqn is None:
            return None
        try:
            macro, q_values, confidence = self.ddqn.select_macro(
                state.squeeze(0), phase,
            )
            return BrainVote(
                brain_name="ddqn",
                action_idx=macro.value,  # Macro index, not command index
                confidence=confidence,
                q_value=float(q_values.max()) if q_values is not None else 0.0,
                metadata={"macro_name": macro.name, "q_values": q_values},
            )
        except Exception as e:
            logger.debug(f"DDQN vote failed: {e}")
            return None

    def _vote_ppo(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        macro_indices: Optional[Set[int]] = None,
    ) -> Optional[BrainVote]:
        """Get PPO micro-level vote with optional macro constraint."""
        if self.ppo is None:
            return None
        try:
            # Apply macro constraint to mask if available
            effective_mask = mask.clone()
            if macro_indices is not None and len(macro_indices) > 0:
                macro_mask = torch.zeros_like(mask)
                for idx in macro_indices:
                    if idx < len(macro_mask):
                        macro_mask[idx] = True
                combined = effective_mask & macro_mask
                if combined.sum() >= 2:
                    effective_mask = combined

            # PPO select_action with mask
            with torch.no_grad():
                logits = self.ppo.network.actor(state.unsqueeze(0) if state.dim() == 1 else state)
                # Apply mask
                logits = logits.squeeze(0)
                logits[~effective_mask] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                value = self.ppo.network.critic(state.unsqueeze(0) if state.dim() == 1 else state)

            # Confidence = 1 - normalised entropy
            max_entropy = math.log(max(effective_mask.sum().item(), 2))
            conf = max(0.0, 1.0 - entropy.item() / max(max_entropy, 1e-6))

            return BrainVote(
                brain_name="ppo",
                action_idx=action.item(),
                confidence=conf,
                q_value=value.item(),
                entropy=entropy.item(),
                log_prob=log_prob.item(),
            )
        except Exception as e:
            logger.debug(f"PPO vote failed: {e}")
            return None

    def _vote_sac(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        macro_indices: Optional[Set[int]] = None,
    ) -> Optional[BrainVote]:
        """Get SAC vote (entropy-regularised exploration)."""
        if self.sac is None:
            return None
        try:
            effective_mask = mask.clone()
            if macro_indices is not None and len(macro_indices) > 0:
                macro_mask = torch.zeros_like(mask)
                for idx in macro_indices:
                    if idx < len(macro_mask):
                        macro_mask[idx] = True
                combined = effective_mask & macro_mask
                if combined.sum() >= 2:
                    effective_mask = combined

            action, log_prob = self.sac.select_action(state, mask=effective_mask)
            # SAC confidence from alpha temperature: lower alpha → more confident
            alpha = self.sac.alpha if hasattr(self.sac, "alpha") else 0.2
            conf = max(0.1, 1.0 - float(alpha))

            return BrainVote(
                brain_name="sac",
                action_idx=action,
                confidence=conf,
                q_value=0.0,  # SAC doesn't expose Q directly here
                entropy=float(alpha),
                log_prob=float(log_prob) if log_prob is not None else 0.0,
            )
        except Exception as e:
            logger.debug(f"SAC vote failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Private: fusion
    # ─────────────────────────────────────────────────────────────────────

    def _fuse_votes(
        self,
        votes: List[BrainVote],
        state: torch.Tensor,
        phase: str,
        mask: torch.Tensor,
        result: CognitionResult,
    ) -> CognitionResult:
        """
        Fuse sub-brain votes via learned confidence gate.

        Rules:
          1. If DDQN macro confidence > threshold AND (PPO uncertain OR RND high):
             → macro overrides PPO's command selection space.
          2. The gate network produces soft weights over [ppo, sac, ddqn, rnd].
          3. Weighted-majority vote: highest combined score wins.
          4. If gate is untrained (early steps), use heuristic fusion.
        """
        # Find votes by brain
        ppo_v = next((v for v in votes if v.brain_name == "ppo"), None)
        sac_v = next((v for v in votes if v.brain_name == "sac"), None)
        ddqn_v = next((v for v in votes if v.brain_name == "ddqn"), None)
        rnd_v = next((v for v in votes if v.brain_name == "rnd"), None)

        # ── Build gate input ──
        phase_idx = self.PHASE_INDICES.get(phase, 0)
        phase_onehot = torch.zeros(8, device=self.device)
        if phase_idx < 8:
            phase_onehot[phase_idx] = 1.0

        gate_input = torch.tensor([
            ppo_v.confidence if ppo_v else 0.0,
            ppo_v.entropy if ppo_v else 2.0,
            sac_v.confidence if sac_v else 0.0,
            sac_v.entropy if sac_v else 2.0,
            ddqn_v.confidence if ddqn_v else 0.0,
            rnd_v.novelty if rnd_v else 0.0,
            self._ema_fast,
            self._ema_slow,
        ], dtype=torch.float32, device=self.device)
        gate_input = torch.cat([gate_input, phase_onehot])

        # ── Adaptive gating ──
        try:
            with torch.no_grad():
                weights, fused_conf = self.gate(gate_input.unsqueeze(0))
            weights = weights.squeeze(0)  # (4,) → [ppo, sac, ddqn, rnd]
            fused_conf = fused_conf.item()
        except Exception:
            # Fallback: heuristic weights
            weights = torch.tensor([
                self.config.initial_ppo_weight,
                self.config.initial_sac_weight,
                self.config.initial_ddqn_weight,
                self.config.initial_rnd_weight,
            ], device=self.device)
            fused_conf = 0.5

        # ── Macro override logic ──
        # DDQN overrides if confident AND PPO uncertain / RND novel
        ppo_uncertain = (ppo_v is None or ppo_v.confidence < self.config.ppo_uncertain_threshold)
        rnd_high = (rnd_v is not None and rnd_v.novelty > self.config.rnd_novelty_high)

        if (ddqn_v is not None
                and ddqn_v.confidence > self.config.macro_override_threshold
                and (ppo_uncertain or rnd_high)):
            # Macro override: scale DDQN weight up
            weights[2] = weights[2] * 2.0
            weights = weights / weights.sum()
            logger.debug(
                f"[COGNITION] Macro override: DDQN conf={ddqn_v.confidence:.2f}, "
                f"PPO uncertain={ppo_uncertain}, RND high={rnd_high}"
            )

        # ── Weighted score per action ──
        # Map brain → weight_index: ppo=0, sac=1, ddqn=2, rnd=3
        brain_weight_map = {"ppo": 0, "sac": 1, "ddqn": 2, "rnd": 3}
        action_scores: Dict[int, float] = {}

        for vote in votes:
            if vote.brain_name not in brain_weight_map:
                continue
            w_idx = brain_weight_map[vote.brain_name]
            w = weights[w_idx].item()
            score = w * vote.confidence
            action_scores[vote.action_idx] = action_scores.get(vote.action_idx, 0.0) + score

        # Pick highest scoring legal action
        best_action = -1
        best_score = -1.0
        best_brain = "none"

        for act_idx, score in sorted(action_scores.items(), key=lambda x: -x[1]):
            if act_idx < len(mask) and mask[act_idx]:
                best_action = act_idx
                best_score = score
                # Find which brain contributed most
                for vote in votes:
                    if vote.action_idx == act_idx:
                        best_brain = vote.brain_name
                        break
                break

        if best_action < 0:
            # Fallback: pick highest-confidence vote with legal action
            for vote in sorted(votes, key=lambda v: -v.confidence):
                if vote.action_idx < len(mask) and mask[vote.action_idx]:
                    best_action = vote.action_idx
                    best_brain = vote.brain_name
                    best_score = vote.confidence
                    break

        if best_action < 0:
            # Last resort: random legal
            legal = mask.nonzero(as_tuple=True)[0]
            if len(legal) > 0:
                best_action = legal[torch.randint(len(legal), (1,)).item()].item()
                best_brain = "random"
                best_score = 0.1

        result.action_idx = best_action
        result.winning_brain = best_brain
        result.confidence = fused_conf

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Private: SIL
    # ─────────────────────────────────────────────────────────────────────

    def _check_sil(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        phase: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if self-imitation replay should inject a golden transition."""
        if not self._sil_buffer or self.ppo is None:
            return None

        # Only trigger SIL periodically (every 5 steps)
        if self._step_count % 5 != 0:
            return None

        # Pick best transition from buffer
        best = max(self._sil_buffer, key=lambda x: x["reward"])
        if best["state"] is not None:
            try:
                self.ppo.store_transition(
                    state=best["state"],
                    action=best["action"],
                    log_prob=best["log_prob"],
                    reward=best["reward"] * 0.5,  # Discount replayed reward
                    value=0.0,
                    done=False,
                )
                return best
            except Exception:
                return None
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Private: gate training
    # ─────────────────────────────────────────────────────────────────────

    def _train_gate(self, result: CognitionResult, reward: float) -> None:
        """
        Train the confidence gate using the actual reward signal.

        Loss: cross-entropy encouraging the gate to assign higher weight
        to the brain that was selected when reward is positive,
        and lower when negative.
        """
        if not result.votes:
            return

        brain_weight_map = {"ppo": 0, "sac": 1, "ddqn": 2, "rnd": 3}
        winner_idx = brain_weight_map.get(result.winning_brain, 0)

        # Build gate input (same as in _fuse_votes)
        ppo_v = next((v for v in result.votes if v.brain_name == "ppo"), None)
        sac_v = next((v for v in result.votes if v.brain_name == "sac"), None)
        ddqn_v = next((v for v in result.votes if v.brain_name == "ddqn"), None)
        rnd_v = next((v for v in result.votes if v.brain_name == "rnd"), None)

        phase_idx = self.PHASE_INDICES.get(result.macro_intent or "RECON", 0)
        phase_onehot = torch.zeros(8, device=self.device)
        if phase_idx < 8:
            phase_onehot[phase_idx] = 1.0

        gate_input = torch.tensor([
            ppo_v.confidence if ppo_v else 0.0,
            ppo_v.entropy if ppo_v else 2.0,
            sac_v.confidence if sac_v else 0.0,
            sac_v.entropy if sac_v else 2.0,
            ddqn_v.confidence if ddqn_v else 0.0,
            rnd_v.novelty if rnd_v else 0.0,
            self._ema_fast,
            self._ema_slow,
        ], dtype=torch.float32, device=self.device)
        gate_input = torch.cat([gate_input, phase_onehot]).unsqueeze(0)

        # Weighted label: if reward > ema_slow, reinforce winner; else suppress
        target = torch.zeros(4, device=self.device)
        if reward > self._ema_slow:
            target[winner_idx] = 1.0
        else:
            # Distribute weight to other brains
            for i in range(4):
                target[i] = 0.0 if i == winner_idx else 1.0 / 3.0

        weights, conf = self.gate(gate_input)
        loss = F.cross_entropy(weights, target.unsqueeze(0))

        # Confidence loss: should predict high conf when reward > baseline
        conf_target = torch.tensor(
            [[1.0 if reward > self._ema_slow else 0.0]],
            device=self.device,
        )
        conf_loss = F.binary_cross_entropy(conf, conf_target)

        total_loss = loss + 0.5 * conf_loss

        self.gate_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gate.parameters(), 1.0)
        self.gate_optimizer.step()

    # ─────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregated metrics for telemetry."""
        total = max(sum(self._brain_win_counts.values()), 1)
        return {
            "brain_distribution": {
                k: round(v / total, 3) for k, v in self._brain_win_counts.items()
            },
            "total_steps": self._step_count,
            "ema_fast": round(self._ema_fast, 3),
            "ema_slow": round(self._ema_slow, 3),
            "sil_buffer": len(self._sil_buffer),
            "sil_threshold": round(self._sil_reward_threshold, 2),
            "total_rnd_bonus": round(self._total_rnd_bonus, 2),
        }
