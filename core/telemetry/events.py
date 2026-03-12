#!/usr/bin/env python3
"""
core/telemetry/events.py — Phase 9.7: Step event schema for JSONL telemetry

Defines the canonical StepEvent dataclass that captures every decision,
discovery, reward, and timing datum for one agent's step.

Schema is JSON-serializable via .to_dict(). All fields are optional-safe
with sensible defaults so partial events can still be logged.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMCallRecord:
    """One LLM API call within a step."""
    role: str = ""            # "mentor", "tactical", "strategic", "parser", etc.
    model: str = ""           # "local-llm", "local-llm", etc.
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role, "model": self.model,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms, "cached": self.cached,
        }


@dataclass
class AntiRepeatRecord:
    """Anti-repeat guard status for this step."""
    triggered: bool = False
    count: int = 0             # How many exact/prefix repeats detected
    action: str = ""           # "penalty", "replace", "none"

    def to_dict(self) -> Dict[str, Any]:
        return {"triggered": self.triggered, "count": self.count, "action": self.action}


@dataclass
class TimingRecord:
    """Per-phase timing in milliseconds."""
    decide_ms: int = 0
    execute_ms: int = 0
    parse_ms: int = 0
    reward_ms: int = 0
    total_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decide": self.decide_ms, "execute": self.execute_ms,
            "parse": self.parse_ms, "reward": self.reward_ms,
            "total": self.total_ms,
        }


@dataclass
class StepEvent:
    """Canonical per-step telemetry event. One per (agent, step) pair.

    Matches the JSONL schema specified in Phase 9.7 spec:
    {
      "ts", "run_id", "episode_id", "agent", "step", "phase",
      "selected_template", "selected_params", "source",
      "candidate_sources", "exec_guidance", "tactical_assessment",
      "discoveries", "reward_breakdown", "anti_repeat",
      "llm_calls", "time_ms"
    }
    """
    # Identity
    ts: str = ""
    run_id: str = ""
    episode_id: int = 0
    agent: str = ""
    step: int = 0
    phase: str = ""

    # Decision
    selected_template: str = ""
    selected_command: str = ""
    selected_params: Dict[str, str] = field(default_factory=dict)
    source: str = ""  # PPO|DDQN|COGNITION|SKILL|PLAYBOOK|CODEX|MENTOR|EC|TC_REDIRECT|FALLBACK
    confidence: float = 0.0

    # Pipeline context
    candidate_sources: Dict[str, Any] = field(default_factory=dict)
    exec_guidance: Dict[str, Any] = field(default_factory=dict)
    tactical_assessment: Dict[str, Any] = field(default_factory=dict)

    # Discoveries
    discoveries: List[Dict[str, Any]] = field(default_factory=list)
    discovery_count: int = 0

    # Reward
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    reward_total: float = 0.0

    # Anti-repeat
    anti_repeat: AntiRepeatRecord = field(default_factory=AntiRepeatRecord)

    # LLM usage
    llm_calls: List[LLMCallRecord] = field(default_factory=list)
    llm_call_count: int = 0
    llm_tokens_total: int = 0

    # Timing
    time_ms: TimingRecord = field(default_factory=TimingRecord)

    # DDQN metrics (Phase 9.5)
    ddqn_select_calls: int = 0
    ddqn_cached_calls: int = 0
    ddqn_epsilon: float = 0.0

    # Parse metrics (Phase 9.5)
    parse_calls: int = 0
    parse_cache_hits: int = 0

    # Phase 10.1: Privilege gating telemetry
    privilege_filtered: int = 0          # candidates filtered by privilege
    sudo_attempted: bool = False
    sudo_allowed: bool = False
    sudo_denied: bool = False
    tool_install_triggered: bool = False
    tool_install_success: bool = False
    wordlist_generated: bool = False
    wordlist_candidates: int = 0
    knock_attempted: bool = False
    knock_success: bool = False
    proxy_events_ingested: int = 0
    payload_transform_used: str = ""

    # Phase 11.0: Full Visibility telemetry
    parse_mode: str = ""                     # "fast" | "intelligent_fullparse"
    parse_latency_ms: float = 0.0
    parse_stage_reached: int = 0             # 1-4
    teaching_points: List[str] = field(default_factory=list)
    budget_pressure: float = 0.0             # 0.0-1.0
    mentor_budget_remaining: int = 0
    venice_budget_remaining: int = 0
    phase_steps_in_current: int = 0
    phase_ladder_blocked: bool = False
    tool_privilege_state: str = ""           # "user" | "sudo" | "root"
    tool_validation_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict matching JSONL schema."""
        return {
            "ts": self.ts or _now_iso(),
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "agent": self.agent,
            "step": self.step,
            "phase": self.phase,
            "selected_template": self.selected_template,
            "selected_command": self.selected_command,
            "selected_params": self.selected_params,
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "candidate_sources": self.candidate_sources,
            "exec_guidance": self.exec_guidance,
            "tactical_assessment": self.tactical_assessment,
            "discoveries": self.discoveries,
            "discovery_count": self.discovery_count,
            "reward_breakdown": self.reward_breakdown,
            "reward_total": round(self.reward_total, 3),
            "anti_repeat": self.anti_repeat.to_dict(),
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "llm_call_count": self.llm_call_count,
            "llm_tokens_total": self.llm_tokens_total,
            "time_ms": self.time_ms.to_dict(),
            "ddqn_select_calls": self.ddqn_select_calls,
            "ddqn_cached_calls": self.ddqn_cached_calls,
            "ddqn_epsilon": round(self.ddqn_epsilon, 5),
            "parse_calls": self.parse_calls,
            "parse_cache_hits": self.parse_cache_hits,
            # Phase 10.1
            "privilege_filtered": self.privilege_filtered,
            "sudo_attempted": self.sudo_attempted,
            "sudo_allowed": self.sudo_allowed,
            "sudo_denied": self.sudo_denied,
            "tool_install_triggered": self.tool_install_triggered,
            "tool_install_success": self.tool_install_success,
            "wordlist_generated": self.wordlist_generated,
            "wordlist_candidates": self.wordlist_candidates,
            "knock_attempted": self.knock_attempted,
            "knock_success": self.knock_success,
            "proxy_events_ingested": self.proxy_events_ingested,
            "payload_transform_used": self.payload_transform_used,
            # Phase 11.0
            "parse_mode": self.parse_mode,
            "parse_latency_ms": round(self.parse_latency_ms, 2),
            "parse_stage_reached": self.parse_stage_reached,
            "teaching_points": self.teaching_points,
            "budget_pressure": round(self.budget_pressure, 3),
            "mentor_budget_remaining": self.mentor_budget_remaining,
            "venice_budget_remaining": self.venice_budget_remaining,
            "phase_steps_in_current": self.phase_steps_in_current,
            "phase_ladder_blocked": self.phase_ladder_blocked,
            "tool_privilege_state": self.tool_privilege_state,
            "tool_validation_passed": self.tool_validation_passed,
        }


@dataclass
class EpisodeEvent:
    """Per-episode summary event."""
    ts: str = ""
    run_id: str = ""
    episode_id: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    final_phase: str = ""
    closeout: bool = False
    termination: str = ""
    unique_commands: int = 0
    diversity_ratio: float = 0.0
    total_discoveries: int = 0
    total_llm_calls: int = 0
    total_llm_tokens: int = 0
    total_parse_calls: int = 0
    total_parse_cache_hits: int = 0
    total_ddqn_calls: int = 0
    total_ddqn_cached: int = 0
    latency_avg_ms: float = 0.0
    anti_repeat_pct: float = 0.0
    source_distribution: Dict[str, int] = field(default_factory=dict)

    # Phase 10.1 episode-level counters
    total_privilege_filtered: int = 0
    total_sudo_attempts: int = 0
    total_sudo_allowed: int = 0
    total_sudo_denied: int = 0
    total_tool_installs: int = 0
    total_wordlists_generated: int = 0
    total_knock_attempts: int = 0
    total_proxy_events: int = 0
    total_payload_transforms: int = 0

    # Phase 11.0: Episode-level aggregates
    avg_budget_pressure: float = 0.0
    total_teaching_points: int = 0
    total_phase_ladder_blocks: int = 0
    total_tool_validation_failures: int = 0
    parse_mode_used: str = ""
    avg_parse_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts or _now_iso(),
            "type": "episode_summary",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "total_steps": self.total_steps,
            "total_reward": round(self.total_reward, 2),
            "final_phase": self.final_phase,
            "closeout": self.closeout,
            "termination": self.termination,
            "unique_commands": self.unique_commands,
            "diversity_ratio": round(self.diversity_ratio, 3),
            "total_discoveries": self.total_discoveries,
            "total_llm_calls": self.total_llm_calls,
            "total_llm_tokens": self.total_llm_tokens,
            "total_parse_calls": self.total_parse_calls,
            "total_parse_cache_hits": self.total_parse_cache_hits,
            "total_ddqn_calls": self.total_ddqn_calls,
            "total_ddqn_cached": self.total_ddqn_cached,
            "latency_avg_ms": round(self.latency_avg_ms, 1),
            "anti_repeat_pct": round(self.anti_repeat_pct, 1),
            "source_distribution": self.source_distribution,
            # Phase 10.1
            "total_privilege_filtered": self.total_privilege_filtered,
            "total_sudo_attempts": self.total_sudo_attempts,
            "total_sudo_allowed": self.total_sudo_allowed,
            "total_sudo_denied": self.total_sudo_denied,
            "total_tool_installs": self.total_tool_installs,
            "total_wordlists_generated": self.total_wordlists_generated,
            "total_knock_attempts": self.total_knock_attempts,
            "total_proxy_events": self.total_proxy_events,
            "total_payload_transforms": self.total_payload_transforms,
            # Phase 11.0
            "avg_budget_pressure": round(self.avg_budget_pressure, 3),
            "total_teaching_points": self.total_teaching_points,
            "total_phase_ladder_blocks": self.total_phase_ladder_blocks,
            "total_tool_validation_failures": self.total_tool_validation_failures,
            "parse_mode_used": self.parse_mode_used,
            "avg_parse_latency_ms": round(self.avg_parse_latency_ms, 2),
        }


def _now_iso() -> str:
    """ISO 8601 timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
