#!/usr/bin/env python3
"""
core/state/canonical_state.py — Phase 35: Single Source of Truth

Canonical state is the ONE authoritative snapshot of the engagement.
Every agent prompt, PhaseGuide prompt, and UI render MUST derive from it.
No local copies allowed unless version-checked.

Author: Phase 35
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

__all__ = ["CanonicalState", "CanonicalStateBuilder"]


@dataclass
class CanonicalState:
    """Immutable snapshot of the canonical engagement state."""

    # Identity
    episode_id: str = ""
    step_id: int = 0

    # Phase
    current_phase: str = "RECON"
    phase_confidence: float = 0.5
    steps_in_phase: int = 0
    stagnation_steps: int = 0

    # Discovery board (frozen copies)
    ports: List[int] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    web_paths_count: int = 0
    top_web_paths: List[str] = field(default_factory=list)
    users: List[str] = field(default_factory=list)
    credentials: List[str] = field(default_factory=list)
    vulns: List[str] = field(default_factory=list)
    shells: List[str] = field(default_factory=list)
    flags_set: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Phase state
    recent_commands: List[str] = field(default_factory=list)
    recent_discovery_deltas: Dict[str, int] = field(default_factory=dict)

    # Budget state
    mentor_budget_used: int = 0
    mentor_budget_cap: int = 0
    pressure_pct: float = 0.0
    model_usage: Dict[str, int] = field(default_factory=dict)

    # Trace
    canonical_hash: str = ""
    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "current_phase": self.current_phase,
            "phase_confidence": round(self.phase_confidence, 3),
            "discovery_board": {
                "ports": self.ports,
                "services": self.services,
                "web_paths_count": self.web_paths_count,
                "top_web_paths": self.top_web_paths[:20],
                "users": self.users,
                "credentials": self.credentials,
                "vulns": self.vulns,
                "shells": self.shells,
                "flags_set": self.flags_set,
                "notes": self.notes,
            },
            "phase_state": {
                "steps_in_phase": self.steps_in_phase,
                "stagnation_steps": self.stagnation_steps,
                "recent_commands": self.recent_commands[-10:],
                "recent_discovery_deltas": self.recent_discovery_deltas,
            },
            "budget_state": {
                "mentor_budget_used": self.mentor_budget_used,
                "mentor_budget_cap": self.mentor_budget_cap,
                "pressure_pct": round(self.pressure_pct, 2),
                "model_usage": self.model_usage,
            },
            "trace_state": {
                "canonical_hash": self.canonical_hash,
                "version": self.version,
            },
        }

    def compact_summary(self) -> str:
        """One-line summary for prompts — keeps token usage low."""
        return (
            f"Phase:{self.current_phase} "
            f"Ports:{len(self.ports)} Svcs:{len(self.services)} "
            f"Creds:{len(self.credentials)} Shells:{len(self.shells)} "
            f"WebPaths:{self.web_paths_count} Vulns:{len(self.vulns)} "
            f"Stag:{self.stagnation_steps} Conf:{self.phase_confidence:.2f}"
        )

    def evidence_counts(self) -> Dict[str, int]:
        return {
            "ports": len(self.ports),
            "services": len(self.services),
            "paths": self.web_paths_count,
            "creds": len(self.credentials),
            "shells": len(self.shells),
            "flags": len(self.flags_set),
            "vulns": len(self.vulns),
            "users": len(self.users),
        }


class CanonicalStateBuilder:
    """Builds canonical state from orchestrator data. Called once per step."""

    _version: int = 0

    @classmethod
    def build(
        cls,
        episode_id: str,
        step_id: int,
        discovery_board: Dict[str, Any],
        current_phase: str = "RECON",
        phase_confidence: float = 0.5,
        steps_in_phase: int = 0,
        stagnation_steps: int = 0,
        recent_commands: Optional[List[str]] = None,
        recent_discovery_deltas: Optional[Dict[str, int]] = None,
        mentor_budget_used: int = 0,
        mentor_budget_cap: int = 0,
        pressure_pct: float = 0.0,
        model_usage: Optional[Dict[str, int]] = None,
    ) -> CanonicalState:
        """Build a canonical state snapshot from live orchestrator data."""
        cls._version += 1

        # Extract discovery board items as sorted lists
        def _to_sorted_list(val: Any) -> list:
            if isinstance(val, (set, frozenset)):
                return sorted(str(x) for x in val)
            if isinstance(val, list):
                return sorted(str(x) for x in val)
            return []

        ports_raw = discovery_board.get("ports", set())
        ports = sorted(int(p) for p in ports_raw if str(p).isdigit()) if ports_raw else []
        services = _to_sorted_list(discovery_board.get("services", set()))
        web_paths_raw = discovery_board.get("web_paths", set())
        web_paths_count = len(web_paths_raw) if web_paths_raw else 0
        top_web_paths = sorted(str(p) for p in list(web_paths_raw)[:20]) if web_paths_raw else []
        users = _to_sorted_list(discovery_board.get("users", set()))
        credentials = _to_sorted_list(discovery_board.get("credentials", set()))
        vulns = _to_sorted_list(discovery_board.get("vulns", set()))
        shells = _to_sorted_list(discovery_board.get("shells", set()))
        flags_set = _to_sorted_list(discovery_board.get("flags_set", set()))
        notes = _to_sorted_list(discovery_board.get("notes", set()))

        state = CanonicalState(
            episode_id=str(episode_id),
            step_id=step_id,
            current_phase=str(current_phase).upper(),
            phase_confidence=phase_confidence,
            steps_in_phase=steps_in_phase,
            stagnation_steps=stagnation_steps,
            ports=ports,
            services=services,
            web_paths_count=web_paths_count,
            top_web_paths=top_web_paths,
            users=users,
            credentials=credentials,
            vulns=vulns,
            shells=shells,
            flags_set=flags_set,
            notes=notes,
            recent_commands=(recent_commands or [])[-10:],
            recent_discovery_deltas=recent_discovery_deltas or {},
            mentor_budget_used=mentor_budget_used,
            mentor_budget_cap=mentor_budget_cap,
            pressure_pct=pressure_pct,
            model_usage=model_usage or {},
            version=cls._version,
        )
        # Compute hash over canonical content
        state.canonical_hash = cls._compute_hash(state)
        return state

    @staticmethod
    def _compute_hash(state: CanonicalState) -> str:
        """SHA-256 over canonical state content (excluding hash itself)."""
        d = state.to_dict()
        d["trace_state"]["canonical_hash"] = ""  # exclude self-ref
        raw = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def reset_version(cls) -> None:
        cls._version = 0
