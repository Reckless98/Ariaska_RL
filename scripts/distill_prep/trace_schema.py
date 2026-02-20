"""Distillation Prep — strict trace & trajectory schemas.

Compatible with Ariaska episode_trace.py, episode_replayer.py, and
teacher_trace.py formats.  Pure dataclass + manual validation; no
pydantic dependency.

Version: v1
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.distill_prep.trace_schema")

DISTILL_PREP_VERSION = "v1"

# ---------------------------------------------------------------------------
# Canonical constants (mirror core)
# ---------------------------------------------------------------------------

VALID_PHASES = frozenset(
    {
        "RECON",
        "ENUMERATION",
        "EXPLOITATION",
        "PRIVILEGE_ESCALATION",
        "LATERAL_MOVEMENT",
        "EXFILTRATION",
        "POST_EXPLOITATION",
        "CLOSEOUT",
    }
)

PHASE_ORDER: List[str] = [
    "RECON",
    "ENUMERATION",
    "EXPLOITATION",
    "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT",
    "EXFILTRATION",
    "POST_EXPLOITATION",
    "CLOSEOUT",
]

VALID_DECISION_SOURCES = frozenset(
    {
        "playbook",
        "ppo",
        "registry",
        "mentor",
        "dual_mentor",
        "micro_chain",
        "phase_guided",
        "anti_repeat",
        "fallback",
    }
)

VALID_DISCOVERY_TYPES = frozenset(
    {
        "PORT",
        "SERVICE",
        "VERSION",
        "CREDENTIAL",
        "USER",
        "VULNERABILITY",
        "CVE",
        "SHELL",
        "ROOT_SHELL",
        "WEB_PATH",
        "SMB_SHARE",
        "OS_INFO",
        "HOSTNAME",
        "FLAG",
        "KEY",
        "HASH",
        "FILE_CONTENT",
        "PCAP_CRED",
        "CAPABILITY",
        "CONFIG_FILE",
        "TOKEN",
        "COOKIE",
        "DOMAIN_USER",
        "GPP_PASSWORD",
    }
)

REWARD_MIN = -15.0
REWARD_MAX = 100.0

# Key command families that the generator can assign.
COMMAND_FAMILIES: List[str] = [
    "nmap",
    "masscan",
    "whatweb",
    "curl",
    "gobuster",
    "ffuf",
    "nikto",
    "feroxbuster",
    "wfuzz",
    "smbclient",
    "smbmap",
    "enum4linux",
    "ldapsearch",
    "dig",
    "dnsrecon",
    "ssh_audit",
    "snmpwalk",
    "showmount",
    "rpcclient",
    "ftp",
    "hydra",
    "crackmapexec",
    "impacket",
    "sqlmap",
    "evil_winrm",
    "ssh",
    "exploit",
    "msfconsole",
    "linpeas",
    "winpeas",
    "sudo",
    "find_suid",
    "bloodhound",
    "chisel",
    "mimikatz",
    "nc",
    "python",
    "docker",
    "lxd",
    "certipy",
    "rubeus",
    "manual",
]


class DistillKind(str, Enum):
    """JSONL line discriminator — compatible with episode_replayer."""

    EPISODE_START = "episode_start"
    STEP = "step"
    EPISODE_END = "episode_end"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryRecord:
    """A single discovery made during a step."""

    discovery_type: str  # DiscoveryType name
    value: str
    confidence: float = 1.0
    source_stage: str = "regex"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.discovery_type not in VALID_DISCOVERY_TYPES:
            errors.append(f"unknown discovery_type: {self.discovery_type}")
        if not isinstance(self.confidence, (int, float)) or math.isnan(
            self.confidence
        ):
            errors.append(f"invalid confidence: {self.confidence}")
        return errors


@dataclass
class AgentDecisionRecord:
    """Per-agent decision within a step — mirrors ReplayAgentRecord."""

    agent_name: str = ""
    role: str = ""
    decision_source: str = ""
    phase: str = ""
    command: str = ""
    command_family: str = ""
    reward: float = 0.0
    mentor_call: bool = False
    discoveries: List[DiscoveryRecord] = field(default_factory=list)
    stdout_snippet: str = ""
    confidence: float = 0.0
    template_name: str = ""
    reasoning: str = ""
    is_wrong_move: bool = False
    tactical_lesson: str = ""

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.decision_source and self.decision_source not in VALID_DECISION_SOURCES:
            errors.append(f"unknown decision_source: {self.decision_source}")
        if self.phase and self.phase not in VALID_PHASES:
            errors.append(f"unknown phase: {self.phase}")
        if self.command_family and self.command_family not in COMMAND_FAMILIES:
            errors.append(f"unknown command_family: {self.command_family}")
        if not isinstance(self.reward, (int, float)) or math.isnan(self.reward):
            errors.append(f"invalid reward: {self.reward}")
        if not (REWARD_MIN <= self.reward <= REWARD_MAX):
            errors.append(
                f"reward {self.reward} out of range [{REWARD_MIN}, {REWARD_MAX}]"
            )
        for d in self.discoveries:
            errors.extend(d.validate())
        return errors


@dataclass
class TraceStep:
    """One step in a distillation trace — compatible with replayer 'step' kind."""

    step_num: int = 0
    phase_before: str = ""
    phase_after: str = ""
    step_reward_total: float = 0.0
    episode_reward_so_far: float = 0.0
    agent_records: List[AgentDecisionRecord] = field(default_factory=list)
    target_ip: str = ""
    timestamp: float = 0.0

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.phase_before not in VALID_PHASES:
            errors.append(f"unknown phase_before: {self.phase_before}")
        if self.phase_after not in VALID_PHASES:
            errors.append(f"unknown phase_after: {self.phase_after}")
        for r in self.agent_records:
            errors.extend(r.validate())
        if not isinstance(self.step_reward_total, (int, float)) or math.isnan(
            self.step_reward_total
        ):
            errors.append(f"invalid step_reward_total: {self.step_reward_total}")
        return errors

    def to_jsonl_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": DistillKind.STEP.value,
            "distill_prep_version": DISTILL_PREP_VERSION,
        }
        d.update(asdict(self))
        return d


@dataclass
class EpisodeStartRecord:
    """Episode start marker — compatible with replayer."""

    episode_id: str = ""
    episode_num: int = 0
    target_ip: str = ""
    difficulty: str = "medium"
    service_mix: str = ""
    seed: Optional[int] = None

    def to_jsonl_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": DistillKind.EPISODE_START.value,
            "distill_prep_version": DISTILL_PREP_VERSION,
        }
        d.update(asdict(self))
        return d


@dataclass
class EpisodeEndRecord:
    """Episode end marker — compatible with replayer."""

    episode_id: str = ""
    episode_num: int = 0
    total_reward: float = 0.0
    highest_phase: str = ""
    total_steps: int = 0
    target_ip: str = ""

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.highest_phase and self.highest_phase not in VALID_PHASES:
            errors.append(f"unknown highest_phase: {self.highest_phase}")
        if not isinstance(self.total_reward, (int, float)) or math.isnan(
            self.total_reward
        ):
            errors.append(f"invalid total_reward: {self.total_reward}")
        return errors

    def to_jsonl_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": DistillKind.EPISODE_END.value,
            "distill_prep_version": DISTILL_PREP_VERSION,
        }
        d.update(asdict(self))
        return d


# ---------------------------------------------------------------------------
# Teacher trajectory schema
# ---------------------------------------------------------------------------


@dataclass
class TeacherStep:
    """One step in a teacher trajectory — for distillation training."""

    phase: str = ""
    state_before: Dict[str, Any] = field(default_factory=dict)
    command_family: str = ""
    full_command: str = ""
    template_name: str = ""
    reasoning: str = ""
    expected_outcome: str = ""
    stdout_snippet: str = ""
    discoveries: List[DiscoveryRecord] = field(default_factory=list)
    reward: float = 0.0
    is_wrong_move: bool = False
    tactical_lesson: str = ""
    decision_source: str = "teacher"
    confidence: float = 0.8
    step_num: int = 0
    knowledge_candidate_id: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.phase not in VALID_PHASES:
            errors.append(f"unknown phase: {self.phase}")
        if self.command_family and self.command_family not in COMMAND_FAMILIES:
            errors.append(f"unknown command_family: {self.command_family}")
        if not isinstance(self.reward, (int, float)) or math.isnan(self.reward):
            errors.append(f"invalid reward: {self.reward}")
        if not (REWARD_MIN <= self.reward <= REWARD_MAX):
            errors.append(
                f"reward {self.reward} out of range [{REWARD_MIN}, {REWARD_MAX}]"
            )
        for d in self.discoveries:
            errors.extend(d.validate())
        return errors


@dataclass
class TeacherTrajectory:
    """Full teacher trajectory for one scenario."""

    trajectory_id: str = ""
    scenario_id: str = ""
    scenario_name: str = ""
    difficulty: str = "medium"
    service_mix: str = ""
    target_ip: str = ""
    steps: List[TeacherStep] = field(default_factory=list)
    total_reward: float = 0.0
    highest_phase: str = ""
    success: bool = False
    wrong_move_count: int = 0
    seed: Optional[int] = None
    distill_prep_version: str = DISTILL_PREP_VERSION

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.trajectory_id:
            errors.append("missing trajectory_id")
        if self.highest_phase and self.highest_phase not in VALID_PHASES:
            errors.append(f"unknown highest_phase: {self.highest_phase}")
        for i, s in enumerate(self.steps):
            step_errors = s.validate()
            for e in step_errors:
                errors.append(f"step[{i}]: {e}")
        return errors

    def to_jsonl_lines(self) -> List[str]:
        """Serialize to JSONL — one header + one line per step."""
        header = {
            "kind": "trajectory_start",
            "distill_prep_version": self.distill_prep_version,
            "trajectory_id": self.trajectory_id,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "difficulty": self.difficulty,
            "service_mix": self.service_mix,
            "target_ip": self.target_ip,
            "seed": self.seed,
        }
        lines = [json.dumps(header, separators=(",", ":"))]
        for step in self.steps:
            d = asdict(step)
            d["kind"] = "teacher_step"
            d["trajectory_id"] = self.trajectory_id
            d["distill_prep_version"] = self.distill_prep_version
            lines.append(json.dumps(d, separators=(",", ":")))
        footer = {
            "kind": "trajectory_end",
            "distill_prep_version": self.distill_prep_version,
            "trajectory_id": self.trajectory_id,
            "total_reward": round(self.total_reward, 3),
            "highest_phase": self.highest_phase,
            "success": self.success,
            "wrong_move_count": self.wrong_move_count,
            "total_steps": len(self.steps),
        }
        lines.append(json.dumps(footer, separators=(",", ":")))
        return lines


# ---------------------------------------------------------------------------
# Weakness report schema
# ---------------------------------------------------------------------------


@dataclass
class WeaknessReport:
    """Curriculum weakness report — identifies training gaps."""

    generated_at: str = ""
    distill_prep_version: str = DISTILL_PREP_VERSION
    total_traces: int = 0
    total_steps: int = 0
    phase_histogram: Dict[str, int] = field(default_factory=dict)
    repeated_command_patterns: List[Dict[str, Any]] = field(default_factory=list)
    tool_family_coverage: Dict[str, int] = field(default_factory=dict)
    avg_reward_by_phase: Dict[str, float] = field(default_factory=dict)
    avg_reward_by_tool: Dict[str, float] = field(default_factory=dict)
    decision_source_pct: Dict[str, float] = field(default_factory=dict)
    weakness_areas: List[str] = field(default_factory=list)
    coverage_gaps: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    """One file entry in the manifest."""

    path: str = ""
    sha256: str = ""
    size_bytes: int = 0
    line_count: int = 0


@dataclass
class Manifest:
    """Distill prep manifest with checksums and metadata."""

    distill_prep_version: str = DISTILL_PREP_VERSION
    git_commit: str = ""
    seed: Optional[int] = None
    generated_at: str = ""
    counts: Dict[str, int] = field(default_factory=dict)
    files: List[ManifestEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario profile
# ---------------------------------------------------------------------------


@dataclass
class ScenarioProfile:
    """Target scenario profile for synthetic generation."""

    scenario_id: str = ""
    name: str = ""
    difficulty: str = "medium"  # easy | medium | hard
    target_ip: str = "10.10.10.1"
    os_type: str = "linux"
    services: List[str] = field(default_factory=list)
    open_ports: List[int] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)
    intended_path: List[str] = field(default_factory=list)
    has_credentials: bool = False
    has_flags: bool = False
    max_steps: int = 80


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_jsonl_line(line_str: str) -> List[str]:
    """Validate a single JSONL line against the distill prep schema.

    Returns a list of error strings (empty = valid).
    """
    errors: List[str] = []
    try:
        obj = json.loads(line_str)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]

    kind = obj.get("kind")
    if kind is None:
        errors.append("missing 'kind' field")
        return errors

    if kind == DistillKind.EPISODE_START.value:
        for k in ("episode_id", "episode_num", "target_ip"):
            if k not in obj:
                errors.append(f"episode_start missing '{k}'")

    elif kind == DistillKind.STEP.value:
        for k in ("step_num", "phase_before", "phase_after"):
            if k not in obj:
                errors.append(f"step missing '{k}'")
        if obj.get("phase_before") not in VALID_PHASES:
            errors.append(f"unknown phase_before: {obj.get('phase_before')}")
        if obj.get("phase_after") not in VALID_PHASES:
            errors.append(f"unknown phase_after: {obj.get('phase_after')}")
        for i, rec in enumerate(obj.get("agent_records", [])):
            if rec.get("command_family") and rec["command_family"] not in COMMAND_FAMILIES:
                errors.append(f"agent_records[{i}]: unknown command_family: {rec['command_family']}")
            if rec.get("decision_source") and rec["decision_source"] not in VALID_DECISION_SOURCES:
                errors.append(f"agent_records[{i}]: unknown decision_source: {rec['decision_source']}")
            reward = rec.get("reward", 0.0)
            if isinstance(reward, float) and math.isnan(reward):
                errors.append(f"agent_records[{i}]: NaN reward")
            if not (REWARD_MIN <= reward <= REWARD_MAX):
                errors.append(f"agent_records[{i}]: reward {reward} out of range")
            for j, disc in enumerate(rec.get("discoveries", [])):
                dt = disc.get("discovery_type", "")
                if isinstance(dt, str) and dt not in VALID_DISCOVERY_TYPES:
                    errors.append(
                        f"agent_records[{i}].discoveries[{j}]: unknown type {dt}"
                    )

    elif kind == DistillKind.EPISODE_END.value:
        for k in ("episode_id", "total_reward", "highest_phase"):
            if k not in obj:
                errors.append(f"episode_end missing '{k}'")
        hp = obj.get("highest_phase", "")
        if hp and hp not in VALID_PHASES:
            errors.append(f"unknown highest_phase: {hp}")
        tr = obj.get("total_reward", 0.0)
        if isinstance(tr, float) and math.isnan(tr):
            errors.append("NaN total_reward")

    elif kind == "trajectory_start":
        for k in ("trajectory_id", "scenario_id"):
            if k not in obj:
                errors.append(f"trajectory_start missing '{k}'")

    elif kind == "teacher_step":
        phase = obj.get("phase", "")
        if phase not in VALID_PHASES:
            errors.append(f"unknown phase: {phase}")
        cf = obj.get("command_family", "")
        if cf and cf not in COMMAND_FAMILIES:
            errors.append(f"unknown command_family: {cf}")
        reward = obj.get("reward", 0.0)
        if isinstance(reward, float) and math.isnan(reward):
            errors.append("NaN reward")
        if not (REWARD_MIN <= reward <= REWARD_MAX):
            errors.append(f"reward {reward} out of range")

    elif kind == "trajectory_end":
        for k in ("trajectory_id", "total_reward", "highest_phase"):
            if k not in obj:
                errors.append(f"trajectory_end missing '{k}'")
    else:
        errors.append(f"unknown kind: {kind}")

    # Universal float check
    _check_floats_recursive(obj, "", errors)
    return errors


def _check_floats_recursive(
    obj: Any, path: str, errors: List[str]
) -> None:
    """Recursively check for NaN/Inf floats."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            errors.append(f"invalid float at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_floats_recursive(v, f"{path}.{k}", errors)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_floats_recursive(v, f"{path}[{i}]", errors)
