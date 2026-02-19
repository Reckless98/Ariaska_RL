"""State Builder — constructs and maintains state dictionaries.

Phase 41: Extracted from SmartOrchestrator for modularity.
Provides stateless utility functions for state construction.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Set, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ariaska.orchestration.state_builder")


@dataclass
class StateSnapshot:
    """Immutable snapshot of attack state at a point in time."""
    phase: str = "RECON"
    target_ip: str = ""
    open_ports: List[int] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    detection_risk: float = 0.0
    shell_obtained: bool = False
    credentials_found: int = 0
    vulns_found: int = 0
    step: int = 0
    episode: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "target_ip": self.target_ip,
            "open_ports": list(self.open_ports),
            "services": list(self.services),
            "detection_risk": self.detection_risk,
            "shell_obtained": self.shell_obtained,
            "credentials_found": self.credentials_found,
            "vulns_found": self.vulns_found,
            "step": self.step,
            "episode": self.episode,
        }


def build_default_state(target_ip: str = "") -> Dict[str, Any]:
    """Build a minimal default state dictionary."""
    return {
        "phase": "recon",
        "target_ip": target_ip,
        "open_ports": [],
        "detection_risk": 0.0,
        "services": [],
    }


def build_state_from_board(
    discovery_board: Dict[str, Any],
    phase: str = "RECON",
    target_ip: str = "",
    step: int = 0,
    episode: int = 0,
    detection_risk: float = 0.0,
) -> Dict[str, Any]:
    """Build a full state dictionary from a discovery board.

    Args:
        discovery_board: The shared discovery state
        phase: Current attack phase
        target_ip: Target IP
        step: Current step number
        episode: Current episode number
        detection_risk: Current detection risk level

    Returns:
        State dictionary suitable for agent consumption.
    """
    ports = sorted(discovery_board.get("ports", set()))
    services = sorted(discovery_board.get("services", set()))
    shells = discovery_board.get("shells", set())
    creds = discovery_board.get("credentials", set())
    vulns = discovery_board.get("vulns", set())
    users = discovery_board.get("users", set())
    web_paths = discovery_board.get("web_paths", set())
    flags = discovery_board.get("flags_set", set())

    return {
        "phase": phase.lower(),
        "target_ip": target_ip,
        "open_ports": ports,
        "services": services,
        "detection_risk": detection_risk,
        "shell_obtained": len(shells) > 0,
        "root_shell": any("root" in str(s) for s in shells),
        "credentials_found": len(creds),
        "vulns_found": len(vulns),
        "users_found": len(users),
        "web_paths_found": len(web_paths),
        "flags_captured": len(flags),
        "ports_discovered": len(ports) > 0,
        "step": step,
        "episode": episode,
        "discovery_count": (
            len(ports) + len(services) + len(creds)
            + len(vulns) + len(users) + len(web_paths) + len(flags)
        ),
    }


def compute_discoveries_delta(
    current: Dict[str, Any],
    previous: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the delta between current and previous discoveries.

    Args:
        current: Current discovery state
        previous: Previous discovery snapshot

    Returns:
        Dict containing only new/changed discoveries.
    """
    delta: Dict[str, Any] = {}
    for key in current:
        cur_val = current[key]
        prev_val = previous.get(key)
        if prev_val is None:
            delta[key] = cur_val
            continue
        if isinstance(cur_val, (set, frozenset)):
            new_items = cur_val - (prev_val if isinstance(prev_val, (set, frozenset)) else set())
            if new_items:
                delta[key] = new_items
        elif isinstance(cur_val, list):
            prev_set = set(prev_val) if isinstance(prev_val, list) else set()
            new_items = [x for x in cur_val if x not in prev_set]
            if new_items:
                delta[key] = new_items
        elif cur_val != prev_val:
            delta[key] = cur_val
    return delta
