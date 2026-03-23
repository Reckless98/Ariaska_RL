"""
core/decision/decision_core.py — Phase 1: Single Decision Authority.

DecisionCore is the ONLY action authority in the system.
All candidate sources (PPO, SAC, DDQN, Mentor, MicroChain, PhaseGuided,
Playbook, Skill, Registry, CognitionNode, Reflex, Codex) produce
Advisory objects. DecisionCore scores them and picks ONE winner.

Design principles:
  - NO cascade ladder — all advisories scored in parallel
  - NO silent bypass — every source path logged
  - NO priority override tree — hard constraints are FILTERS, not sources
  - Full attribution on every decision

Call chain:
  SmartOrchestrator._run_step()
    → SmartCoach.decide()
      → SmartCoach._collect_advisories()
      → DecisionCore.arbitrate(advisories, constraints, maturity)
      → SmartDecisionResult

Author: Phase 1 — Authority Unification
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.decision.core")


# ---------------------------------------------------------------------------
# Advisory — input from any candidate source
# ---------------------------------------------------------------------------

@dataclass
class Advisory:
    """A single advisory input from any candidate source.

    Every source (PPO, SAC, mentor, playbook, etc.) produces an Advisory.
    DecisionCore scores all advisories and selects the winner.
    """

    source: str = ""
    """Source identifier: 'ppo', 'sac', 'mentor', 'playbook', 'micro_chain',
    'phase_guided', 'codex_meta', 'cognition', 'hypothesis', 'skill',
    'registry', 'reflex', 'followup'."""

    command: str = ""
    """The proposed command string."""

    template_name: str = ""
    """CommandTemplate name for registry matching."""

    confidence: float = 0.5
    """Source's self-reported confidence [0, 1]."""

    phase_fit: float = 0.5
    """How well this command fits the current phase [0, 1]."""

    novelty: float = 0.0
    """RND-based novelty score [0, 1]. 0 = seen many times, 1 = completely novel."""

    expected_value: float = 0.0
    """Expected reward if this action is taken."""

    logits: Optional[Any] = None
    """Raw logits/Q-values from neural source (torch.Tensor)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Source-specific metadata (reasoning, model_used, etc.)."""

    # --- Computed by DecisionCore ---
    arbitration_score: float = 0.0
    """Final score after weighting. Set by DecisionCore."""


@dataclass
class HardConstraint:
    """A hard constraint that FILTERS advisories (never chooses).

    Hard constraints remove invalid advisories before scoring.
    They are NOT decision sources — they enforce structural correctness.
    """

    name: str = ""
    """Constraint identifier: 'phase_ladder', 'anti_repeat', 'evidence_gate',
    'closeout', 'command_lockout'."""

    blocked_commands: Set[str] = field(default_factory=set)
    """Commands blocked by this constraint."""

    blocked_templates: Set[str] = field(default_factory=set)
    """Templates blocked by this constraint."""

    required_template: Optional[str] = None
    """If set, ONLY this template is allowed (closeout, forced actions)."""

    required_command: Optional[str] = None
    """If set, ONLY this command is allowed."""

    reason: str = ""
    """Human-readable reason for the constraint."""


@dataclass
class ArbitrationResult:
    """Output of DecisionCore.arbitrate()."""

    command: str = ""
    """Winning command string."""

    template_name: str = ""
    """Winning template name."""

    source: str = "none"
    """Which advisory source won."""

    confidence: float = 0.0
    """Winning advisory's confidence."""

    was_override: bool = False
    """True if a hard constraint forced the selection (closeout, etc.)."""

    override_reason: str = ""
    """If was_override, why."""

    # --- Full attribution ---

    arbitration_weights: Dict[str, float] = field(default_factory=dict)
    """Weight given to each source: {source: weight}."""

    all_scores: Dict[str, float] = field(default_factory=dict)
    """Final scores of all advisories: {source: score}."""

    trace: List[Dict[str, Any]] = field(default_factory=list)
    """Full arbitration trace for debugging/metrics."""

    advisories_received: int = 0
    """How many advisories were submitted."""

    advisories_after_filter: int = 0
    """How many survived hard constraints."""

    winning_advisory: Optional[Advisory] = None
    """Reference to the winning Advisory object."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Extra metadata (mentor_reasoning, etc.)."""


# ---------------------------------------------------------------------------
# Source weight configuration
# ---------------------------------------------------------------------------

# Default source priors — base weights for each source.
# These are multiplied by the advisory's own confidence + phase_fit + novelty.
# Updated at runtime from SourceWinRateTracker.
DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = {
    "ppo": 1.0,           # Primary RL learner — highest base weight
    "sac": 0.3,           # Advisory only — Q-value signal
    "mentor": 0.85,       # LLM guidance — high but decays with maturity
    "playbook": 0.65,     # Curriculum-guided — anneals over time
    "micro_chain": 0.75,  # 3-stage LLM scoring
    "phase_guided": 0.70, # Structured LLM guidance
    "codex_meta": 0.80,   # Full codex reasoning (expensive)
    "cognition": 0.60,    # Multi-brain fusion
    "hypothesis": 0.55,   # Hypothesis-driven
    "skill": 0.50,        # Skill library replay
    "registry": 0.35,     # Generic registry match
    "reflex": 0.90,       # Hardcoded safety responses
    "followup": 0.30,     # Web path curl probes — LOW priority, must not override LLM/PPO
    "arbitrator": 0.70,   # Legacy arbitrator (if still used)
    "strategic_advisor": 0.85,  # Cloud LLM strategic plans — high priority
}


# ---------------------------------------------------------------------------
# DecisionCore — THE single action authority
# ---------------------------------------------------------------------------

class DecisionCore:
    """Single decision authority for all agent actions.

    Replaces the 12-level priority cascade in SmartCoach.decide() with
    weighted arbitration. All candidate sources produce Advisory objects;
    DecisionCore scores them and selects ONE winner.

    Hard constraints (phase ladder, anti-repeat, evidence gate) are applied
    as filters BEFORE scoring — they remove invalid options but never choose.

    The scoring function is:
        score = (
            source_weight
            × confidence
            × (0.5 + 0.3 × phase_fit + 0.2 × novelty)
            × maturity_scaling[source]
        )

    Where maturity_scaling is:
        PPO: increases with maturity (1.0 → 2.0)
        Mentor/LLM: decreases with maturity (1.0 → 0.2)
        Registry/Skill: stable (1.0)
    """

    def __init__(
        self,
        source_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._source_weights = dict(source_weights or DEFAULT_SOURCE_WEIGHTS)
        self._decision_count: int = 0
        self._source_hit_count: Dict[str, int] = {}
        self._last_result: Optional[ArbitrationResult] = None

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def arbitrate(
        self,
        advisories: List[Advisory],
        constraints: Optional[List[HardConstraint]] = None,
        maturity: float = 0.0,
        rnd_novelty: float = 0.0,
        source_win_rates: Optional[Dict[str, float]] = None,
        stagnation_steps: int = 0,
    ) -> ArbitrationResult:
        """Score all advisories, apply constraints, select winner.

        Args:
            advisories: All candidate actions from all sources.
            constraints: Hard constraints to filter invalid candidates.
            maturity: Global maturity signal M ∈ [0, 1].
            rnd_novelty: Current step's RND novelty score.
            source_win_rates: EMA win-rate per source (from SourceWinRateTracker).
            stagnation_steps: Steps since last discovery (0 = no stagnation).

        Returns:
            ArbitrationResult with winner, weights, scores, and full trace.
        """
        if constraints is None:
            constraints = []
        self._decision_count += 1
        result = ArbitrationResult()
        result.advisories_received = len(advisories)
        trace: List[Dict[str, Any]] = []

        if not advisories:
            result.source = "none"
            result.trace = [{"event": "no_advisories", "step": self._decision_count}]
            logger.warning(f"[DC] Step {self._decision_count}: No advisories received")
            return result

        # ── Step 1: Apply hard constraints (filters) ─────────────────
        filtered = list(advisories)
        forced_advisory: Optional[Advisory] = None

        for constraint in constraints:
            if constraint.required_command:
                # Closeout/forced — inject as mandatory advisory
                forced_advisory = Advisory(
                    source=f"override_{constraint.name}",
                    command=constraint.required_command,
                    template_name=constraint.required_template or "",
                    confidence=1.0,
                    phase_fit=1.0,
                    metadata={"override_reason": constraint.reason},
                )
                trace.append({
                    "event": "hard_override",
                    "constraint": constraint.name,
                    "command": constraint.required_command[:80],
                    "reason": constraint.reason,
                })
                break  # Required command overrides everything

            # Filter out blocked commands/templates
            before = len(filtered)
            filtered = [
                a for a in filtered
                if a.command not in constraint.blocked_commands
                and a.template_name not in constraint.blocked_templates
            ]
            removed = before - len(filtered)
            if removed > 0:
                trace.append({
                    "event": "constraint_filter",
                    "constraint": constraint.name,
                    "removed": removed,
                    "reason": constraint.reason,
                })

        if forced_advisory is not None:
            result.command = forced_advisory.command
            result.template_name = forced_advisory.template_name
            result.source = forced_advisory.source
            result.confidence = 1.0
            result.was_override = True
            result.override_reason = forced_advisory.metadata.get("override_reason", "")
            result.advisories_after_filter = 1
            result.winning_advisory = forced_advisory
            result.trace = trace
            self._record_hit(forced_advisory.source)
            logger.debug(
                f"[DC] Step {self._decision_count}: OVERRIDE by "
                f"{forced_advisory.source} → {forced_advisory.command[:60]}"
            )
            self._last_result = result
            return result

        result.advisories_after_filter = len(filtered)

        if not filtered:
            # All advisories filtered — this is a problem
            trace.append({"event": "all_filtered", "original": len(advisories)})
            result.source = "none"
            result.trace = trace
            logger.warning(
                f"[DC] Step {self._decision_count}: All {len(advisories)} "
                f"advisories filtered by constraints"
            )
            self._last_result = result
            return result

        # ── Step 2: Score each advisory ──────────────────────────────
        win_rates = source_win_rates or {}
        for advisory in filtered:
            score = self._score_advisory(
                advisory, maturity, rnd_novelty, win_rates, stagnation_steps
            )
            advisory.arbitration_score = score
            result.all_scores[advisory.source] = score
            trace.append({
                "event": "scored",
                "source": advisory.source,
                "command": advisory.command[:60],
                "confidence": round(advisory.confidence, 3),
                "phase_fit": round(advisory.phase_fit, 3),
                "novelty": round(advisory.novelty, 3),
                "score": round(score, 4),
            })

        # ── Step 3: Select winner (highest score) ────────────────────
        winner = max(filtered, key=lambda a: a.arbitration_score)

        result.command = winner.command
        result.template_name = winner.template_name
        result.source = winner.source
        result.confidence = winner.confidence
        result.winning_advisory = winner
        result.metadata = dict(winner.metadata)

        # Record arbitration weights (normalized scores)
        total_score = sum(a.arbitration_score for a in filtered) or 1.0
        result.arbitration_weights = {
            a.source: round(a.arbitration_score / total_score, 4)
            for a in filtered
        }

        trace.append({
            "event": "winner",
            "source": winner.source,
            "command": winner.command[:80],
            "score": round(winner.arbitration_score, 4),
            "margin": round(
                winner.arbitration_score - (
                    sorted(filtered, key=lambda a: a.arbitration_score)[-2].arbitration_score
                    if len(filtered) > 1 else 0.0
                ), 4,
            ),
        })

        result.trace = trace
        self._record_hit(winner.source)

        logger.debug(
            f"[DC] Step {self._decision_count}: {winner.source} wins "
            f"(score={winner.arbitration_score:.3f}, "
            f"cmd={winner.command[:50]}, "
            f"n_advisories={len(filtered)}/{len(advisories)})"
        )

        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_advisory(
        self,
        advisory: Advisory,
        maturity: float,
        rnd_novelty: float,
        win_rates: Dict[str, float],
        stagnation_steps: int = 0,
    ) -> float:
        """Compute arbitration score for a single advisory.

        Score = source_weight × confidence × quality × maturity_factor
               × win_rate_factor × stagnation_factor

        Where:
            quality = 0.5 + 0.3 × phase_fit + 0.2 × novelty
            maturity_factor = per-source scaling from maturity signal
            win_rate_factor = boost for historically successful sources
            stagnation_factor = penalizes PPO / boosts LLM when stagnating
        """
        source = advisory.source
        base_weight = self._source_weights.get(source, 0.5)

        # Quality combines phase fit and novelty
        quality = 0.5 + 0.3 * advisory.phase_fit + 0.2 * max(advisory.novelty, rnd_novelty * 0.5)

        # Maturity scaling: RL sources grow, LLM sources decay
        mat_factor = self._maturity_factor(source, maturity)

        # Win-rate boost: historically successful sources get a small bonus
        wr = win_rates.get(source, 0.5)
        wr_factor = 0.8 + 0.4 * wr  # Maps [0,1] → [0.8, 1.2]

        # Stagnation-aware scaling: when stuck, penalize PPO, boost LLM
        stag_factor = self._stagnation_factor(source, stagnation_steps)

        score = base_weight * advisory.confidence * quality * mat_factor * wr_factor * stag_factor

        # Expected value bonus (if available, e.g., from Q-values)
        if advisory.expected_value > 0:
            score += 0.1 * min(advisory.expected_value, 5.0)

        return score

    def _stagnation_factor(self, source: str, stagnation_steps: int) -> float:
        """Compute stagnation-dependent scaling factor for a source.

        When the agent is stagnating (no discoveries for many steps),
        PPO is clearly stuck on a bad action. We progressively:
        - Penalize PPO / RL sources (their policy has collapsed)
        - Boost LLM sources (they can reason about novel strategies)

        Thresholds:
            ≤10 steps: No effect (normal exploration noise)
            11-25:     Mild PPO penalty (0.7), slight LLM boost (1.2)
            26-50:     Strong PPO penalty (0.3), strong LLM boost (1.8)
            51+:       Emergency — PPO nearly silenced (0.1), LLM dominant (2.5)
        """
        if stagnation_steps <= 10:
            return 1.0

        _is_rl = source in ("ppo", "sac", "cognition")
        _is_llm = source in (
            "mentor", "micro_chain", "phase_guided", "codex_meta",
            "strategic_advisor",
        )
        _is_rigid = source in ("followup", "playbook", "skill")

        if stagnation_steps <= 25:
            if _is_rl:
                return 0.7
            if _is_llm:
                return 1.2
            if _is_rigid:
                return 0.6  # Penalize rigid queues when stagnating
            return 1.0
        if stagnation_steps <= 50:
            if _is_rl:
                return 0.3
            if _is_llm:
                return 1.8
            if _is_rigid:
                return 0.2  # Strong penalty — rigid queues causing stagnation
            return 1.0
        # Emergency: 51+ steps stagnating
        if _is_rl:
            return 0.1
        if _is_llm:
            return 2.5
        if _is_rigid:
            return 0.05  # Nearly silenced — let LLMs take over
        return 1.0

    def _maturity_factor(self, source: str, maturity: float) -> float:
        """Compute maturity-dependent scaling factor for a source.

        As maturity increases (agent becomes skilled):
        - PPO weight INCREASES (trust the learned policy)
        - Mentor/LLM weight DECREASES (less handholding needed)
        - Registry/Skill stays STABLE (always useful)

        Uses cosine interpolation for smooth transitions.
        """
        # PPO: 1.0 → 2.0 as maturity grows
        if source == "ppo":
            return 1.0 + maturity

        # Mentor/LLM sources: 1.0 → 0.2
        if source in ("mentor", "micro_chain", "phase_guided", "codex_meta",
                       "strategic_advisor"):
            return 1.0 - 0.8 * maturity

        # CognitionNode: 0.8 → 1.2 (grows slightly with data)
        if source == "cognition":
            return 0.8 + 0.4 * maturity

        # Playbook: 1.0 → 0.3 (anneals as agent learns)
        if source == "playbook":
            return 1.0 - 0.7 * maturity

        # Reflex: always high
        if source == "reflex":
            return 1.0

        # Default: stable
        return 1.0

    # ------------------------------------------------------------------
    # Source weight updates
    # ------------------------------------------------------------------

    def update_source_weight(self, source: str, weight: float) -> None:
        """Update a source's base weight (e.g., from win-rate tracker)."""
        self._source_weights[source] = max(0.1, min(2.0, weight))

    def get_source_weights(self) -> Dict[str, float]:
        """Get current source weight configuration."""
        return dict(self._source_weights)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _record_hit(self, source: str) -> None:
        """Track which sources win decisions."""
        self._source_hit_count[source] = self._source_hit_count.get(source, 0) + 1

    def get_hit_distribution(self) -> Dict[str, float]:
        """Get normalized hit distribution across sources."""
        total = sum(self._source_hit_count.values()) or 1
        return {
            src: count / total
            for src, count in sorted(
                self._source_hit_count.items(), key=lambda x: -x[1]
            )
        }

    def get_decision_source_entropy(self) -> float:
        """Compute entropy of decision source distribution.

        High entropy = decisions spread across many sources.
        Low entropy = one source dominates (may indicate collapse).
        """
        total = sum(self._source_hit_count.values()) or 1
        entropy = 0.0
        for count in self._source_hit_count.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p + 1e-10)
        return entropy

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def last_result(self) -> Optional[ArbitrationResult]:
        return self._last_result

    def reset_episode(self) -> None:
        """Reset per-episode counters (keep cross-episode stats)."""
        # Hit counts are cross-episode — don't reset
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for logging/checkpointing."""
        return {
            "decision_count": self._decision_count,
            "source_weights": dict(self._source_weights),
            "hit_distribution": self.get_hit_distribution(),
            "source_entropy": round(self.get_decision_source_entropy(), 4),
        }
