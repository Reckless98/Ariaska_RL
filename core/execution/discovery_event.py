#!/usr/bin/env python3
"""
core/execution/discovery_event.py — Phase 10.0: Structured Discovery Events

Canonical schema for discoveries extracted from command output.
Every parser stage emits DiscoveryEvent objects — the parser broker
normalises and deduplicates them before passing to the reward system.

Author: Filip Volf / Ariaska System
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class DiscoveryType(Enum):
    """Types of discoveries that can be extracted from tool output."""
    PORT = "open_port"
    SERVICE = "service"
    VERSION = "version_info"
    CREDENTIAL = "credential"
    USER = "user"
    VULNERABILITY = "vulnerability"
    CVE = "cve"
    SHELL = "shell"
    ROOT_SHELL = "root_shell"
    WEB_PATH = "web_path"
    SMB_SHARE = "smb_share"
    OS_INFO = "os_info"
    HOSTNAME = "hostname"
    FLAG = "flag"
    KEY = "key"


@dataclass
class DiscoveryEvent:
    """
    A single discovery extracted from command output.

    Fields:
        discovery_type: What was found (port, service, credential, etc.)
        value: The actual discovery value (port number, service name, etc.)
        confidence: Extraction confidence 0.0–1.0
        source_stage: Which parser stage produced this (regex, llm, venice, broker)
        command: The command that produced the output
        agent: Which agent ran the command
        timestamp: When the discovery was made
        raw_evidence: The raw text evidence supporting this discovery
        kg_candidate_id: If mapped to a v2 knowledge candidate
        deduplicated: Whether this was already seen this episode
    """
    discovery_type: DiscoveryType
    value: Any
    confidence: float = 1.0
    source_stage: str = "regex"          # regex | llm | venice | broker
    command: str = ""
    agent: str = ""
    timestamp: float = field(default_factory=time.time)
    raw_evidence: str = ""
    kg_candidate_id: Optional[str] = None
    deduplicated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serialisable dict."""
        d = asdict(self)
        d["discovery_type"] = self.discovery_type.value
        return d

    @classmethod
    def from_flat_discoveries(
        cls,
        discoveries: Dict[str, Any],
        source_stage: str = "regex",
        command: str = "",
        agent: str = "",
    ) -> List["DiscoveryEvent"]:
        """
        Convert a flat discovery dict (SmartOutputParser format) to DiscoveryEvents.

        Args:
            discoveries: {"open_port": [80, 443], "service": ["http"], "credential": True, ...}
            source_stage: Parser stage that produced these
            command: The command that was run
            agent: The agent that ran it
        """
        events: List[DiscoveryEvent] = []

        type_map = {
            "open_port": DiscoveryType.PORT,
            "service": DiscoveryType.SERVICE,
            "version_info": DiscoveryType.VERSION,
            "credential": DiscoveryType.CREDENTIAL,
            "user": DiscoveryType.USER,
            "vulnerability": DiscoveryType.VULNERABILITY,
            "cve": DiscoveryType.CVE,
            "shell": DiscoveryType.SHELL,
            "root_shell": DiscoveryType.ROOT_SHELL,
            "web_path": DiscoveryType.WEB_PATH,
            "smb_share": DiscoveryType.SMB_SHARE,
            "os_info": DiscoveryType.OS_INFO,
            "hostname": DiscoveryType.HOSTNAME,
            "flag": DiscoveryType.FLAG,
            "key": DiscoveryType.KEY,
        }

        for key, value in discoveries.items():
            dtype = type_map.get(key)
            if dtype is None:
                continue

            if isinstance(value, list):
                for v in value:
                    events.append(cls(
                        discovery_type=dtype,
                        value=v,
                        source_stage=source_stage,
                        command=command,
                        agent=agent,
                    ))
            elif isinstance(value, bool) and value:
                events.append(cls(
                    discovery_type=dtype,
                    value=True,
                    source_stage=source_stage,
                    command=command,
                    agent=agent,
                ))
            elif value:
                events.append(cls(
                    discovery_type=dtype,
                    value=value,
                    source_stage=source_stage,
                    command=command,
                    agent=agent,
                ))

        return events

    @staticmethod
    def to_flat_discoveries(events: List["DiscoveryEvent"]) -> Dict[str, Any]:
        """Convert DiscoveryEvents back to flat discovery dict for backward compat."""
        flat: Dict[str, Any] = {}
        for e in events:
            key = e.discovery_type.value
            if isinstance(e.value, bool):
                flat[key] = True
            elif isinstance(e.value, (int, str)):
                flat.setdefault(key, [])
                if isinstance(flat[key], list):
                    flat[key].append(e.value)
                else:
                    flat[key] = [e.value]
            else:
                flat[key] = e.value
        return flat
