#!/usr/bin/env python3
"""
core/reasoning/strategy_plan.py — Phase 14.0: Executable Strategy IR

Contract C1.5: StrategyPlan is a JSON-only executable IR.
  - Orion outputs ONLY valid JSON matching this schema.
  - If not parseable JSON → rejected (ValueError) + template fallback used.
  - Each directive includes required_evidence_ids, completion_criteria,
    verification_commands[], and fallbacks[] (max 2).

Author: Phase 14.0 Contract C1.5 (Rule 4)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.strategy_plan")


class SubgoalStatus(str, Enum):
    """Status of a subgoal."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class Subgoal:
    """A single actionable subgoal in a strategy plan."""
    goal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""                          # ≤128 chars
    target_agent: str = ""                         # Agent ID
    required_evidence_ids: List[str] = field(default_factory=list)
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    verification_commands: List[str] = field(default_factory=list)
    fallbacks: List[str] = field(default_factory=list)  # Max 2
    priority: int = 3                               # 1-5
    status: SubgoalStatus = SubgoalStatus.PENDING

    def __post_init__(self) -> None:
        if len(self.description) > 128:
            self.description = self.description[:128]
        if len(self.fallbacks) > 2:
            self.fallbacks = self.fallbacks[:2]
        self.priority = max(1, min(5, self.priority))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subgoal":
        d = dict(data)
        if "status" in d and isinstance(d["status"], str):
            d["status"] = SubgoalStatus(d["status"])
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class Milestone:
    """Evidence-based milestone for plan progress tracking."""
    name: str = ""
    evidence_required: List[str] = field(default_factory=list)
    phase: str = ""  # Kill chain phase

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Milestone":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class Guardrail:
    """Risk guardrail that triggers protective actions."""
    metric: str = ""     # detection_risk | failed_attempts | steps_elapsed | mentor_reliance
    threshold: float = 0.0
    action: str = ""     # reduce_aggression | switch_agent | replan | boost_mentor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Guardrail":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class StrategyPlan:
    """
    Executable strategy IR. Contract C1.5 (Rule 4).

    MUST be valid JSON. If Orion produces non-JSON, it is REJECTED
    and a template fallback plan is used.
    """
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    subgoals: List[Subgoal] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    guardrails: List[Guardrail] = field(default_factory=list)
    current_subgoal_idx: int = 0
    created_episode: int = 0
    created_step: int = 0

    def get_active_subgoal(self) -> Optional[Subgoal]:
        """Get the currently active subgoal, or None if plan complete."""
        if 0 <= self.current_subgoal_idx < len(self.subgoals):
            return self.subgoals[self.current_subgoal_idx]
        return None

    def advance(self, evidence_graph: Any = None) -> bool:
        """
        Check milestones and advance to next subgoal if current is complete.

        Returns True if advanced, False if not.
        """
        if self.current_subgoal_idx >= len(self.subgoals):
            return False

        current = self.subgoals[self.current_subgoal_idx]

        # Check completion criteria against evidence graph
        if evidence_graph is not None and current.completion_criteria:
            # Check if all required evidence nodes exist
            for req_id in current.required_evidence_ids:
                if hasattr(evidence_graph, "_nodes"):
                    if req_id not in evidence_graph._nodes:
                        return False

        # Mark current as completed and advance
        current.status = SubgoalStatus.COMPLETED
        self.current_subgoal_idx += 1

        # Activate next subgoal if exists
        if self.current_subgoal_idx < len(self.subgoals):
            self.subgoals[self.current_subgoal_idx].status = SubgoalStatus.ACTIVE
            return True
        return True

    def is_blocked(self) -> bool:
        """Check if current subgoal is blocked."""
        current = self.get_active_subgoal()
        if current is None:
            return False
        return current.status == SubgoalStatus.BLOCKED

    def check_guardrails(self, state: Dict[str, Any]) -> Optional[Guardrail]:
        """
        Check all guardrails against current state.

        Returns the first triggered Guardrail, or None.
        """
        for gr in self.guardrails:
            metric_val = state.get(gr.metric, 0.0)
            try:
                if float(metric_val) >= gr.threshold:
                    return gr
            except (ValueError, TypeError):
                continue
        return None

    def to_json(self) -> str:
        """Serialize to JSON string. Contract Rule 4: must produce valid JSON."""
        data = {
            "plan_id": self.plan_id,
            "subgoals": [sg.to_dict() for sg in self.subgoals],
            "milestones": [m.to_dict() for m in self.milestones],
            "guardrails": [g.to_dict() for g in self.guardrails],
            "current_subgoal_idx": self.current_subgoal_idx,
            "created_episode": self.created_episode,
            "created_step": self.created_step,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "StrategyPlan":
        """
        Deserialize from JSON string.

        Contract Rule 4: If not parseable JSON → raises ValueError.
        Orion's freeform prose is REJECTED.
        """
        if not isinstance(data, str):
            raise ValueError("StrategyPlan.from_json requires a string input")

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"StrategyPlan: invalid JSON — {e}") from e

        if not isinstance(parsed, dict):
            raise ValueError("StrategyPlan: JSON must be an object, not a primitive or array")

        # Validate required structure
        if "subgoals" not in parsed:
            raise ValueError("StrategyPlan: missing required field 'subgoals'")

        subgoals = [Subgoal.from_dict(sg) for sg in parsed.get("subgoals", [])]
        milestones = [Milestone.from_dict(m) for m in parsed.get("milestones", [])]
        guardrails = [Guardrail.from_dict(g) for g in parsed.get("guardrails", [])]

        return cls(
            plan_id=parsed.get("plan_id", uuid.uuid4().hex[:12]),
            subgoals=subgoals,
            milestones=milestones,
            guardrails=guardrails,
            current_subgoal_idx=parsed.get("current_subgoal_idx", 0),
            created_episode=parsed.get("created_episode", 0),
            created_step=parsed.get("created_step", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self.to_json())


def create_template_fallback_plan(episode: int = 0, step: int = 0) -> StrategyPlan:
    """
    Create a template fallback plan when Orion output is invalid JSON.

    This provides a generic pentesting methodology plan.
    """
    return StrategyPlan(
        subgoals=[
            Subgoal(
                description="Port scan and service enumeration",
                target_agent="ScoutAgent",
                required_evidence_ids=[],
                completion_criteria={"min_ports_found": 3},
                verification_commands=["nmap_full_scan"],
                fallbacks=["nmap_quick_scan"],
                priority=5,
                status=SubgoalStatus.ACTIVE,
            ),
            Subgoal(
                description="Service version detection",
                target_agent="ScoutAgent",
                required_evidence_ids=[],
                completion_criteria={"min_services_found": 2},
                verification_commands=["nmap_version_scan"],
                fallbacks=["nmap_service_scan"],
                priority=4,
            ),
            Subgoal(
                description="Vulnerability identification",
                target_agent="RedAgent",
                required_evidence_ids=[],
                completion_criteria={"min_vulns_found": 1},
                verification_commands=["vuln_scan"],
                fallbacks=["manual_enum"],
                priority=3,
            ),
            Subgoal(
                description="Exploitation attempt",
                target_agent="RedAgent",
                required_evidence_ids=[],
                completion_criteria={"shell_obtained": True},
                verification_commands=["exploit_run"],
                fallbacks=["alt_exploit"],
                priority=2,
            ),
        ],
        milestones=[
            Milestone(name="foothold", evidence_required=[], phase="EXPLOITATION"),
            Milestone(name="privilege_escalation", evidence_required=[], phase="PRIVILEGE_ESCALATION"),
        ],
        guardrails=[
            Guardrail(metric="failed_attempts", threshold=10.0, action="replan"),
            Guardrail(metric="detection_risk", threshold=0.8, action="reduce_aggression"),
        ],
        created_episode=episode,
        created_step=step,
    )
