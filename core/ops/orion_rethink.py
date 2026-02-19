"""
core/ops/orion_rethink.py — Phase 39.1: Orion Deep-Rethink Escalation

Stall detector + structured escalation path.  When engagement stalls,
compacts context and produces a revised plan as structured JSON.

Feature flag: FF_ORION_RETHINK (default ON)

Stall signals (6):
  1. repeated_tool_failures   — same tool failed N+ times in window
  2. evidence_plateau         — no new evidence for K steps
  3. phase_oscillation        — bounced between 2 phases 3+ times
  4. hypothesis_invalidation  — hypotheses invalidated without replacement
  5. repeat_commands          — same command prefix reused M+ times
  6. reward_stagnation        — cumulative reward flat for L steps

When stall_score >= threshold → produces OrionRethinkPlan.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.ops.orion_rethink")

# ── Constants ──────────────────────────────────────────────────────────

STALL_THRESHOLD: float = 0.55
"""Stall score at or above which Orion rethink is triggered."""

EVIDENCE_PLATEAU_STEPS: int = 8
"""Steps without new evidence before stall signal fires."""

TOOL_FAILURE_WINDOW: int = 10
"""Window of steps to check for repeated tool failures."""

TOOL_FAILURE_COUNT: int = 3
"""Number of repeated failures for same tool in window."""

PHASE_OSCILLATION_COUNT: int = 3
"""Bounce between same 2 phases this many times → oscillation."""

COMMAND_REPEAT_THRESHOLD: int = 4
"""Same command prefix repeated this many times → stall signal."""

REWARD_FLAT_STEPS: int = 6
"""Steps of flat cumulative reward → reward stagnation."""

RETHINK_COOLDOWN_STEPS: int = 10
"""Minimum steps between rethink invocations."""

MAX_CONTEXT_DECISIONS: int = 15
"""Maximum recent decisions to include in compacted context."""


@dataclass
class StallSignals:
    """Individual stall signal values and combined score."""

    repeated_tool_failures: float = 0.0
    evidence_plateau: float = 0.0
    phase_oscillation: float = 0.0
    hypothesis_invalidation: float = 0.0
    repeat_commands: float = 0.0
    reward_stagnation: float = 0.0

    @property
    def score(self) -> float:
        """Weighted stall score in [0, 1]."""
        signals = [
            (self.repeated_tool_failures, 0.20),
            (self.evidence_plateau, 0.25),
            (self.phase_oscillation, 0.15),
            (self.hypothesis_invalidation, 0.15),
            (self.repeat_commands, 0.15),
            (self.reward_stagnation, 0.10),
        ]
        raw = sum(s * w for s, w in signals)
        return min(max(raw, 0.0), 1.0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["score"] = self.score
        return d


@dataclass
class OrionRethinkPlan:
    """Structured output from Orion rethink escalation."""

    new_hypotheses: List[str] = field(default_factory=list)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    why_now: str = ""
    expected_signal: str = ""
    stall_signals: Optional[StallSignals] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "new_hypotheses": self.new_hypotheses,
            "plan": self.plan,
            "why_now": self.why_now,
            "expected_signal": self.expected_signal,
            "timestamp": self.timestamp,
        }
        if self.stall_signals is not None:
            d["stall_signals"] = self.stall_signals.to_dict()
        return d


class OrionRethinkEngine:
    """
    Stall detector and Orion escalation engine.

    Usage:
        engine = OrionRethinkEngine()
        engine.record_step(step=5, command="nmap_scan", success=True,
                           phase="RECON", discoveries=2, reward=3.0)
        signals = engine.compute_stall_signals()
        if signals.score >= STALL_THRESHOLD:
            plan = engine.generate_rethink_plan(
                evidence_summary={...}, hypotheses=[...],
                gpt_manager=gpt,
            )
    """

    def __init__(
        self,
        stall_threshold: float = STALL_THRESHOLD,
        max_history: int = 50,
    ) -> None:
        self._threshold = stall_threshold
        self._max_history = max_history

        # Per-step history
        self._step_history: List[Dict[str, Any]] = []
        self._tool_results: deque = deque(maxlen=TOOL_FAILURE_WINDOW * 2)
        self._phase_history: List[str] = []
        self._evidence_counts: List[int] = []
        self._reward_history: List[float] = []
        self._command_history: List[str] = []
        self._hypotheses_invalidated: int = 0
        self._hypotheses_current: int = 0

        # Rethink state
        self._last_rethink_step: int = -RETHINK_COOLDOWN_STEPS
        self._rethink_count: int = 0
        self._rethink_history: List[OrionRethinkPlan] = []

    def record_step(
        self,
        step: int,
        command: str = "",
        template_name: str = "",
        success: bool = False,
        phase: str = "",
        discoveries: int = 0,
        reward: float = 0.0,
        hypotheses_count: int = 0,
        hypotheses_invalidated: int = 0,
    ) -> None:
        """Record a step's outcome for stall detection."""
        entry = {
            "step": step,
            "command": command[:80],
            "template_name": template_name,
            "success": success,
            "phase": phase.upper() if phase else "",
            "discoveries": discoveries,
            "reward": reward,
            "ts": time.time(),
        }
        self._step_history.append(entry)
        if len(self._step_history) > self._max_history:
            self._step_history = self._step_history[-self._max_history:]

        # Tool results tracking
        self._tool_results.append({
            "template": template_name or command[:30],
            "success": success,
        })

        # Phase tracking
        if phase:
            self._phase_history.append(phase.upper())
            if len(self._phase_history) > self._max_history:
                self._phase_history = self._phase_history[-self._max_history:]

        # Evidence tracking
        self._evidence_counts.append(discoveries)
        if len(self._evidence_counts) > self._max_history:
            self._evidence_counts = self._evidence_counts[-self._max_history:]

        # Reward tracking
        self._reward_history.append(reward)
        if len(self._reward_history) > self._max_history:
            self._reward_history = self._reward_history[-self._max_history:]

        # Command tracking
        if command:
            self._command_history.append(command[:50])
            if len(self._command_history) > self._max_history:
                self._command_history = self._command_history[-self._max_history:]

        # Hypothesis tracking
        self._hypotheses_current = hypotheses_count
        self._hypotheses_invalidated += hypotheses_invalidated

    def compute_stall_signals(self) -> StallSignals:
        """Compute all 6 stall signals from recent history."""
        signals = StallSignals()

        if len(self._step_history) < 3:
            return signals

        # 1. Repeated tool failures
        signals.repeated_tool_failures = self._check_repeated_failures()

        # 2. Evidence plateau
        signals.evidence_plateau = self._check_evidence_plateau()

        # 3. Phase oscillation
        signals.phase_oscillation = self._check_phase_oscillation()

        # 4. Hypothesis invalidation
        signals.hypothesis_invalidation = self._check_hypothesis_invalidation()

        # 5. Repeat commands
        signals.repeat_commands = self._check_repeat_commands()

        # 6. Reward stagnation
        signals.reward_stagnation = self._check_reward_stagnation()

        return signals

    def should_rethink(self, current_step: int) -> tuple:
        """
        Check if rethink should be triggered.

        Returns:
            (should_trigger, signals)
        """
        signals = self.compute_stall_signals()

        # Cooldown check
        if current_step - self._last_rethink_step < RETHINK_COOLDOWN_STEPS:
            return False, signals

        if signals.score >= self._threshold:
            return True, signals

        return False, signals

    def generate_rethink_plan(
        self,
        evidence_summary: Optional[Dict[str, Any]] = None,
        hypotheses: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        current_phase: str = "",
        current_step: int = 0,
        gpt_manager: Optional[Any] = None,
    ) -> OrionRethinkPlan:
        """
        Generate a structured rethink plan.

        If gpt_manager is available, uses LLM for plan generation.
        Otherwise falls back to heuristic plan.
        """
        signals = self.compute_stall_signals()
        self._last_rethink_step = current_step
        self._rethink_count += 1

        # Build compacted context
        context = self._build_compact_context(
            evidence_summary=evidence_summary or {},
            hypotheses=hypotheses or [],
            constraints=constraints or [],
            current_phase=current_phase,
            signals=signals,
        )

        plan: OrionRethinkPlan

        if gpt_manager is not None and hasattr(gpt_manager, "gpt_request"):
            plan = self._llm_rethink(gpt_manager, context, signals)
        else:
            plan = self._heuristic_rethink(context, signals, current_phase)

        plan.stall_signals = signals
        self._rethink_history.append(plan)

        logger.info(
            "Orion rethink #%d at step %d (stall=%.2f): %s",
            self._rethink_count, current_step, signals.score, plan.why_now,
        )

        return plan

    def reset(self) -> None:
        """Reset all state for a new engagement."""
        self._step_history.clear()
        self._tool_results.clear()
        self._phase_history.clear()
        self._evidence_counts.clear()
        self._reward_history.clear()
        self._command_history.clear()
        self._hypotheses_invalidated = 0
        self._hypotheses_current = 0
        self._last_rethink_step = -RETHINK_COOLDOWN_STEPS
        self._rethink_count = 0
        self._rethink_history.clear()

    @property
    def rethink_count(self) -> int:
        return self._rethink_count

    @property
    def rethink_history(self) -> List[OrionRethinkPlan]:
        return list(self._rethink_history)

    @property
    def last_rethink_step(self) -> int:
        return self._last_rethink_step

    # ── Stall signal checkers ────────────────────────────────────────────

    def _check_repeated_failures(self) -> float:
        """Signal 1: same tool failed N+ times in recent window."""
        recent = list(self._tool_results)[-TOOL_FAILURE_WINDOW:]
        if not recent:
            return 0.0

        failures = [r["template"] for r in recent if not r["success"]]
        if not failures:
            return 0.0

        counter = Counter(failures)
        max_repeats = counter.most_common(1)[0][1] if counter else 0
        if max_repeats >= TOOL_FAILURE_COUNT:
            return min(max_repeats / (TOOL_FAILURE_COUNT + 2), 1.0)
        return 0.0

    def _check_evidence_plateau(self) -> float:
        """Signal 2: no new evidence for K+ steps."""
        if len(self._evidence_counts) < EVIDENCE_PLATEAU_STEPS:
            return 0.0

        recent = self._evidence_counts[-EVIDENCE_PLATEAU_STEPS:]
        total_discoveries = sum(recent)
        if total_discoveries == 0:
            # Full plateau — scale by how far over threshold
            extra = len(self._evidence_counts) - EVIDENCE_PLATEAU_STEPS
            trailing_zeros = 0
            for v in reversed(self._evidence_counts):
                if v == 0:
                    trailing_zeros += 1
                else:
                    break
            return min(trailing_zeros / (EVIDENCE_PLATEAU_STEPS * 2), 1.0)
        return 0.0

    def _check_phase_oscillation(self) -> float:
        """Signal 3: bouncing between 2 phases repeatedly."""
        if len(self._phase_history) < 6:
            return 0.0

        recent = self._phase_history[-12:]
        transitions = []
        for i in range(1, len(recent)):
            if recent[i] != recent[i - 1]:
                transitions.append((recent[i - 1], recent[i]))

        if len(transitions) < PHASE_OSCILLATION_COUNT:
            return 0.0

        pair_counter = Counter(transitions)
        most_common_pair, count = pair_counter.most_common(1)[0]
        # Also check the reverse pair
        reverse = (most_common_pair[1], most_common_pair[0])
        reverse_count = pair_counter.get(reverse, 0)
        oscillation_total = count + reverse_count

        if oscillation_total >= PHASE_OSCILLATION_COUNT:
            return min(oscillation_total / (PHASE_OSCILLATION_COUNT + 2), 1.0)
        return 0.0

    def _check_hypothesis_invalidation(self) -> float:
        """Signal 4: hypotheses invalidated without replacement."""
        if self._hypotheses_invalidated <= 0:
            return 0.0

        if self._hypotheses_current == 0 and self._hypotheses_invalidated >= 2:
            return min(self._hypotheses_invalidated / 4.0, 1.0)
        elif self._hypotheses_invalidated >= 3:
            return min(self._hypotheses_invalidated / 5.0, 1.0)
        return 0.0

    def _check_repeat_commands(self) -> float:
        """Signal 5: same command prefix used M+ times."""
        if len(self._command_history) < COMMAND_REPEAT_THRESHOLD:
            return 0.0

        recent = self._command_history[-15:]
        # Use first 30 chars as prefix for matching
        prefixes = [c[:30] for c in recent]
        counter = Counter(prefixes)
        max_repeats = counter.most_common(1)[0][1] if counter else 0

        if max_repeats >= COMMAND_REPEAT_THRESHOLD:
            return min(max_repeats / (COMMAND_REPEAT_THRESHOLD + 3), 1.0)
        return 0.0

    def _check_reward_stagnation(self) -> float:
        """Signal 6: cumulative reward flat for L+ steps."""
        if len(self._reward_history) < REWARD_FLAT_STEPS:
            return 0.0

        recent = self._reward_history[-REWARD_FLAT_STEPS:]
        total = sum(recent)
        if abs(total) < 0.5:
            # Essentially zero reward for several steps
            return min(REWARD_FLAT_STEPS / (REWARD_FLAT_STEPS + 2), 1.0)

        # Check if reward is actually changing
        variance = sum((r - total / len(recent)) ** 2 for r in recent) / len(recent)
        if variance < 0.1:
            return 0.5

        return 0.0

    # ── Plan generators ──────────────────────────────────────────────────

    def _build_compact_context(
        self,
        evidence_summary: Dict[str, Any],
        hypotheses: List[str],
        constraints: List[str],
        current_phase: str,
        signals: StallSignals,
    ) -> Dict[str, Any]:
        """Build compacted context for rethink plan generation."""
        recent_decisions = []
        for entry in self._step_history[-MAX_CONTEXT_DECISIONS:]:
            recent_decisions.append({
                "step": entry["step"],
                "cmd": entry["command"],
                "ok": entry["success"],
                "phase": entry["phase"],
                "disc": entry["discoveries"],
            })

        return {
            "current_phase": current_phase,
            "stall_score": signals.score,
            "stall_signals": signals.to_dict(),
            "recent_decisions": recent_decisions,
            "evidence_summary": evidence_summary,
            "hypotheses": hypotheses,
            "constraints": constraints,
            "rethink_number": self._rethink_count,
        }

    def _llm_rethink(
        self,
        gpt_manager: Any,
        context: Dict[str, Any],
        signals: StallSignals,
    ) -> OrionRethinkPlan:
        """Generate rethink plan via GPT."""
        import json

        prompt = (
            "You are Orion, the strategic coordinator for an automated penetration "
            "testing system. The engagement has STALLED.\n\n"
            f"STALL SCORE: {signals.score:.2f}\n"
            f"STALL SIGNALS: {json.dumps(signals.to_dict(), indent=2)}\n\n"
            f"CURRENT PHASE: {context['current_phase']}\n"
            f"RECENT DECISIONS: {json.dumps(context['recent_decisions'], indent=2)}\n\n"
            f"EVIDENCE SO FAR: {json.dumps(context['evidence_summary'], default=str)}\n"
            f"CURRENT HYPOTHESES: {json.dumps(context['hypotheses'])}\n"
            f"CONSTRAINTS: {json.dumps(context['constraints'])}\n\n"
            "Output a revised plan as JSON with these fields:\n"
            '  "new_hypotheses": [...],  # 2-4 alternative attack hypotheses\n'
            '  "plan": [{"phase":"...", "actions":[{"tool":"...", "args":"..."}]}],\n'
            '  "why_now": "...",  # Why rethinking now\n'
            '  "expected_signal": "..."  # What will confirm/disprove\n'
            "Respond with ONLY valid JSON."
        )

        try:
            response = gpt_manager.gpt_request(
                prompt,
                task_type="strategic",
                agent_id="OrionAgent",
                max_tokens=600,
            )
            if response:
                parsed = json.loads(response)
                return OrionRethinkPlan(
                    new_hypotheses=parsed.get("new_hypotheses", []),
                    plan=parsed.get("plan", []),
                    why_now=parsed.get("why_now", "LLM-generated rethink"),
                    expected_signal=parsed.get("expected_signal", ""),
                )
        except Exception as e:
            logger.warning("LLM rethink failed, falling back to heuristic: %s", e)

        return self._heuristic_rethink(context, signals, context["current_phase"])

    def _heuristic_rethink(
        self,
        context: Dict[str, Any],
        signals: StallSignals,
        current_phase: str,
    ) -> OrionRethinkPlan:
        """Generate rethink plan from heuristics (no LLM needed)."""
        hypotheses = []
        plan_actions = []
        why_now_parts = []
        expected = ""

        phase_upper = current_phase.upper() if current_phase else "RECON"

        if signals.evidence_plateau > 0.3:
            why_now_parts.append("evidence plateau detected")
            if phase_upper in ("RECON", "ENUMERATION"):
                hypotheses.append("Try alternative recon vectors (UDP scan, deeper enum)")
                plan_actions.append({
                    "phase": "ENUMERATION",
                    "actions": [
                        {"tool": "nmap_udp", "args": "-sU --top-ports 50"},
                        {"tool": "enum4linux", "args": "-a"},
                    ],
                })
                expected = "New ports or services discovered"
            elif phase_upper == "EXPLOITATION":
                hypotheses.append("Current exploit vector exhausted, try alternative services")
                plan_actions.append({
                    "phase": "ENUMERATION",
                    "actions": [
                        {"tool": "nikto", "args": "web vuln scan"},
                        {"tool": "searchsploit", "args": "version-targeted search"},
                    ],
                })
                expected = "New vulnerability or service version found"

        if signals.repeated_tool_failures > 0.3:
            why_now_parts.append("repeated tool failures")
            hypotheses.append("Switch to different toolset for same objective")
            expected = expected or "Successful tool execution"

        if signals.phase_oscillation > 0.3:
            why_now_parts.append("phase oscillation detected")
            hypotheses.append("Commit to current phase before advancing")
            expected = expected or "Stable phase with forward progress"

        if signals.repeat_commands > 0.3:
            why_now_parts.append("command repetition detected")
            hypotheses.append("Expand command vocabulary for current phase")

        if signals.reward_stagnation > 0.3:
            why_now_parts.append("reward stagnation")
            hypotheses.append("Pursue higher-reward discovery targets")

        if not hypotheses:
            hypotheses.append("Re-examine evidence and try lateral approach")
            why_now_parts.append("general stall")

        return OrionRethinkPlan(
            new_hypotheses=hypotheses,
            plan=plan_actions,
            why_now="; ".join(why_now_parts) if why_now_parts else "engagement stalled",
            expected_signal=expected or "Any new evidence or successful command",
        )
