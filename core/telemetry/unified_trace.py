#!/usr/bin/env python3
"""
core/telemetry/unified_trace.py — Phase 11.0: Unified Step Trace

Canonical per-step data structure that captures everything about one agent's
step: decision, execution, parsing, discoveries, reward, and annotations.

This is the single-source-of-truth passed between SmartOrchestrator, ParserBroker,
LiveDashboard, LearningSignalExporter, and StepEvent telemetry.

Author: Filip Volf — Phase 11.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParseExplanation:
    """Explains what the parser found and how it found it.

    One per discovery type in a step. Powers the dashboard inline annotations
    and the learning signal exporter.
    """
    # Which stage produced this discovery
    stage: str = ""  # "regex", "sop_llm", "venice", "gpt_finaliser"
    stage_number: int = 0  # 1-4

    # What was found
    discovery_type: str = ""  # "open_port", "service", "credential", "shell", etc.
    discovery_value: str = ""  # "22", "ssh", "admin:password", etc.

    # Confidence and reasoning
    confidence: float = 1.0  # Parser confidence in this discovery (0.0-1.0)
    reasoning: str = ""  # Human-readable: "Regex matched 'PORT 22/tcp open' at line 3"

    # Raw evidence
    raw_match: str = ""  # The exact text that matched
    line_number: int = 0  # Line in output where match occurred (0 = unknown)

    # Timing
    latency_ms: float = 0.0  # How long this stage took

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_number": self.stage_number,
            "discovery_type": self.discovery_type,
            "discovery_value": self.discovery_value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "raw_match": self.raw_match[:200] if self.raw_match else "",
            "line_number": self.line_number,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class BudgetSnapshot:
    """Captures adaptive budget state at the moment of this step."""
    mentor_budget_remaining: int = 0
    mentor_budget_total: int = 0
    mentor_spend_rate: float = 0.0  # calls/step rolling average
    venice_budget_remaining: int = 0
    venice_budget_total: int = 0
    parse_llm_remaining: int = 0
    token_budget_remaining: int = 0
    token_budget_total: int = 0
    budget_pressure: float = 0.0  # 0.0=relaxed, 1.0=exhausted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mentor_remaining": self.mentor_budget_remaining,
            "mentor_total": self.mentor_budget_total,
            "mentor_spend_rate": round(self.mentor_spend_rate, 3),
            "venice_remaining": self.venice_budget_remaining,
            "venice_total": self.venice_budget_total,
            "parse_llm_remaining": self.parse_llm_remaining,
            "token_remaining": self.token_budget_remaining,
            "token_total": self.token_budget_total,
            "budget_pressure": round(self.budget_pressure, 3),
        }


@dataclass
class PhaseState:
    """Phase ladder state at this step."""
    current_phase: str = "RECON"
    steps_in_phase: int = 0
    min_steps_required: int = 0
    phase_index: int = 0  # 0-based index in kill chain
    can_advance: bool = False
    advance_blocked_reason: str = ""  # "min_steps_not_met", "no_discoveries", etc.
    teaching_point: str = ""  # "Cannot skip ENUMERATION — need service fingerprints first"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "steps_in_phase": self.steps_in_phase,
            "min_steps_required": self.min_steps_required,
            "phase_index": self.phase_index,
            "can_advance": self.can_advance,
            "advance_blocked_reason": self.advance_blocked_reason,
            "teaching_point": self.teaching_point,
        }


@dataclass
class ToolPrivilege:
    """Tool privilege validation result for this step's command."""
    tool_name: str = ""
    requires_root: bool = False
    requires_sudo: bool = False
    privilege_state: str = "user"  # "user", "sudo", "root", "unknown"
    validation_passed: bool = True
    blocked_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "requires_root": self.requires_root,
            "requires_sudo": self.requires_sudo,
            "privilege_state": self.privilege_state,
            "validation_passed": self.validation_passed,
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class UnifiedStepTrace:
    """Single-source-of-truth for one agent's complete step.

    Built incrementally by SmartOrchestrator after each sub-operation:
      1. Decision phase → agent_name, command, source, confidence, reasoning
      2. Execution phase → raw_output, execution_time_ms
      3. Parse phase → discoveries, parse_explanations, parse_mode, parse_latency_ms
      4. Reward phase → reward_total, reward_breakdown
      5. Annotation phase → phase_state, budget_snapshot, tool_privilege, teaching_points

    Consumed by:
      - LiveDashboard.print_step() for unified display
      - LearningSignalExporter for RL-ready JSONL
      - StepEvent for telemetry
    """
    # ── Identity ──────────────────────────────────────────────────
    ts: str = ""
    episode_id: int = 0
    step: int = 0
    agent_name: str = ""

    # ── Decision ──────────────────────────────────────────────────
    command: str = ""
    template_name: str = ""
    source: str = ""  # "ppo", "playbook", "registry", "mentor", "anti_repeat", etc.
    confidence: float = 0.0
    reasoning: str = ""  # Full decision chain
    mentor_call: bool = False
    mentor_reasoning: str = ""

    # ── Execution ─────────────────────────────────────────────────
    raw_output: str = ""
    execution_time_ms: float = 0.0
    execution_mode: str = "simulated"  # "simulated" | "live"

    # ── Parse ─────────────────────────────────────────────────────
    parse_mode: str = "fast"  # "fast" | "intelligent_fullparse"
    parse_latency_ms: float = 0.0
    parse_stage_reached: int = 0  # Highest stage number that fired (1-4)
    discoveries: Dict[str, Any] = field(default_factory=dict)  # flat format
    discovery_count: int = 0
    parse_explanations: List[ParseExplanation] = field(default_factory=list)

    # ── Reward ────────────────────────────────────────────────────
    reward_total: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)

    # ── Phase ─────────────────────────────────────────────────────
    phase_state: PhaseState = field(default_factory=PhaseState)

    # ── Budget ────────────────────────────────────────────────────
    budget_snapshot: BudgetSnapshot = field(default_factory=BudgetSnapshot)

    # ── Tool Privilege ────────────────────────────────────────────
    tool_privilege: ToolPrivilege = field(default_factory=ToolPrivilege)

    # ── Teaching Points ───────────────────────────────────────────
    teaching_points: List[str] = field(default_factory=list)

    # ── Evidence Gate (Phase 27) ──────────────────────────────────
    evidence_gate_result: str = ""  # "", "pass", "log_reject", "enforce_reject"
    evidence_gate_reasons: List[str] = field(default_factory=list)

    # ── Phase 29: Observability extensions ────────────────────────
    micro_chain_escalated: bool = False     # MicroChain escalated to codex
    micro_chain_cost_usd: float = 0.0       # MicroChain GPT cost
    self_debug_fix: str = ""                # corrected command from SelfDebugger
    retry_count: int = 0                    # transient-failure retries
    watchdog_trigger: str = ""              # watchdog trigger if fired

    # ── Tokens ────────────────────────────────────────────────────
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict for JSONL export."""
        return {
            "ts": self.ts or _now_iso(),
            "episode_id": self.episode_id,
            "step": self.step,
            "agent_name": self.agent_name,
            # Decision
            "command": self.command,
            "template_name": self.template_name,
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "mentor_call": self.mentor_call,
            "mentor_reasoning": self.mentor_reasoning,
            # Execution
            "raw_output": self.raw_output[:2000] if self.raw_output else "",
            "execution_time_ms": round(self.execution_time_ms, 2),
            "execution_mode": self.execution_mode,
            # Parse
            "parse_mode": self.parse_mode,
            "parse_latency_ms": round(self.parse_latency_ms, 2),
            "parse_stage_reached": self.parse_stage_reached,
            "discoveries": self.discoveries,
            "discovery_count": self.discovery_count,
            "parse_explanations": [e.to_dict() for e in self.parse_explanations],
            # Reward
            "reward_total": round(self.reward_total, 3),
            "reward_breakdown": {k: round(v, 3) for k, v in self.reward_breakdown.items()},
            # Phase
            "phase_state": self.phase_state.to_dict(),
            # Budget
            "budget_snapshot": self.budget_snapshot.to_dict(),
            # Tool privilege
            "tool_privilege": self.tool_privilege.to_dict(),
            # Teaching
            "teaching_points": self.teaching_points,
            # Evidence gate
            "evidence_gate_result": self.evidence_gate_result,
            "evidence_gate_reasons": self.evidence_gate_reasons,
            # Phase 29 observability
            "micro_chain_escalated": self.micro_chain_escalated,
            "micro_chain_cost_usd": round(self.micro_chain_cost_usd, 6),
            "self_debug_fix": self.self_debug_fix,
            "retry_count": self.retry_count,
            "watchdog_trigger": self.watchdog_trigger,
            # Tokens
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_decision_result(
        cls,
        decision: Any,
        episode_id: int,
        step: int,
    ) -> "UnifiedStepTrace":
        """Build a trace from a SmartDecisionResult (partial — execution/parse filled later)."""
        trace = cls(
            episode_id=episode_id,
            step=step,
            agent_name=getattr(decision, 'agent_name', '') or '',
            command=getattr(decision, 'command', '') or '',
            template_name=getattr(decision, 'template_name', '') or '',
            source=getattr(decision, 'source', 'unknown') or 'unknown',
            confidence=getattr(decision, 'confidence', 0.0),
            reasoning=getattr(decision, 'reasoning', '') or '',
            mentor_call=getattr(decision, 'mentor_call', False),
            mentor_reasoning=getattr(decision, 'mentor_reasoning', '') or '',
            tokens_used=getattr(decision, 'tokens_used', 0),
        )
        return trace


def _now_iso() -> str:
    """ISO 8601 timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
