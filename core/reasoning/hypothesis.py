#!/usr/bin/env python3
"""
core/reasoning/hypothesis.py — Phase 14.0: Evidence-Based Hypothesis Testing

Generates hypotheses from EvidenceGraph + knowledge indices.
NO LLM by default — uses pattern-matching on service→CVE→exploit knowledge.

Contract C1.4:
  - Hypothesis: structured prediction about what exploit to try
  - HypothesisGenerator: generates from evidence, ranks by expected_value
  - Status transitions: UNTESTED → TESTING → CONFIRMED/REFUTED/EXPIRED

Author: Phase 14.0 Contract C1.4
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.hypothesis")


class HypothesisStatus(str, Enum):
    """Status of a hypothesis through its lifecycle."""
    UNTESTED = "untested"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    EXPIRED = "expired"


@dataclass
class Hypothesis:
    """
    Evidence-based prediction about what exploit/action to try.

    Contract C1.4: Generated from EvidenceGraph + knowledge indices.
    Ranked by expected_value() = confidence * reward_if_confirmed.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    if_observed: str = ""                     # Evidence condition (≤128 chars)
    then_try: str = ""                        # CommandTemplate name
    expected_result: str = ""                 # Expected outcome (≤128 chars)
    test_command: str = ""                    # Rendered command to test
    confidence: float = 0.5                   # 0-1 evidence strength
    risk_level: float = 0.3                   # 0-1 detection risk
    source: str = "knowledge_base"            # knowledge_base|skill_library|campaign|mentor
    evidence_ids: List[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    result_summary: Optional[str] = None
    reward_if_confirmed: float = 5.0
    created_step: int = 0
    tested_step: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.if_observed) > 128:
            self.if_observed = self.if_observed[:128]
        if len(self.expected_result) > 128:
            self.expected_result = self.expected_result[:128]

    def expected_value(self) -> float:
        """Expected value = confidence * reward_if_confirmed."""
        return self.confidence * self.reward_if_confirmed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "if_observed": self.if_observed,
            "then_try": self.then_try,
            "expected_result": self.expected_result,
            "test_command": self.test_command,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "source": self.source,
            "evidence_ids": self.evidence_ids,
            "status": self.status.value,
            "result_summary": self.result_summary,
            "reward_if_confirmed": self.reward_if_confirmed,
            "created_step": self.created_step,
            "tested_step": self.tested_step,
        }


# ── Service → Exploit Knowledge Map (NO LLM) ────────────────────────────
# Pattern-matching rules for hypothesis generation from discovered services.
# These are generic pentesting patterns, not MS2/MS3-specific.
SERVICE_EXPLOIT_PATTERNS: List[Dict[str, Any]] = [
    {
        "service_pattern": "vsftpd",
        "version_pattern": "2.3.4",
        "then_try": "vsftpd_backdoor",
        "expected_result": "Root shell via FTP backdoor",
        "confidence": 0.9,
        "reward": 50.0,
    },
    {
        "service_pattern": "ssh",
        "version_pattern": None,
        "then_try": "ssh_brute_force",
        "expected_result": "Credential discovery via brute force",
        "confidence": 0.4,
        "reward": 20.0,
    },
    {
        "service_pattern": "http",
        "version_pattern": None,
        "then_try": "web_enum_dirb",
        "expected_result": "Web path discovery",
        "confidence": 0.6,
        "reward": 5.0,
    },
    {
        "service_pattern": "smb",
        "version_pattern": None,
        "then_try": "smb_enum_shares",
        "expected_result": "SMB share listing with potential data",
        "confidence": 0.5,
        "reward": 8.0,
    },
    {
        "service_pattern": "mysql",
        "version_pattern": None,
        "then_try": "mysql_default_creds",
        "expected_result": "Database access via default credentials",
        "confidence": 0.5,
        "reward": 15.0,
    },
    {
        "service_pattern": "postgresql",
        "version_pattern": None,
        "then_try": "postgres_default_creds",
        "expected_result": "Database access via default credentials",
        "confidence": 0.4,
        "reward": 15.0,
    },
    {
        "service_pattern": "tomcat",
        "version_pattern": None,
        "then_try": "tomcat_manager_brute",
        "expected_result": "Tomcat manager access for WAR deploy",
        "confidence": 0.5,
        "reward": 25.0,
    },
    {
        "service_pattern": "irc",
        "version_pattern": "unreal",
        "then_try": "unreal_ircd_backdoor",
        "expected_result": "Root shell via IRC backdoor",
        "confidence": 0.85,
        "reward": 50.0,
    },
    {
        "service_pattern": "vnc",
        "version_pattern": None,
        "then_try": "vnc_password_test",
        "expected_result": "VNC desktop access",
        "confidence": 0.4,
        "reward": 10.0,
    },
    {
        "service_pattern": "nfs",
        "version_pattern": None,
        "then_try": "nfs_mount_enumerate",
        "expected_result": "NFS share mount with file access",
        "confidence": 0.5,
        "reward": 12.0,
    },
]


class HypothesisGenerator:
    """
    Generates hypotheses from EvidenceGraph + knowledge indices.

    Contract C1.4: NO LLM by default. Uses pattern-matching on
    service→exploit knowledge (SERVICE_EXPLOIT_PATTERNS).
    """

    def __init__(self) -> None:
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._tested_templates: set = set()

    def generate(
        self,
        evidence_graph: Any,  # EvidenceGraph
        knowledge_retriever: Any = None,
        campaign_memory: Any = None,
        max_hypotheses: int = 10,
        current_step: int = 0,
    ) -> List[Hypothesis]:
        """
        Generate hypotheses from evidence. NO LLM call.

        Process:
        1. Get discovered services from EvidenceGraph
        2. Match against SERVICE_EXPLOIT_PATTERNS
        3. Check campaign memory for prior successes
        4. Return ranked by expected_value
        """
        hypotheses: List[Hypothesis] = []

        # Get services from evidence graph
        services = []
        if evidence_graph is not None and hasattr(evidence_graph, "_nodes"):
            for node in evidence_graph._nodes.values():
                if hasattr(node, "node_type") and str(node.node_type) in (
                    "service", "EvidenceNodeType.SERVICE", "SERVICE"
                ):
                    services.append(node.properties)

        # Match against known patterns
        for svc_props in services:
            svc_name = str(svc_props.get("service", "")).lower()
            svc_version = str(svc_props.get("version", "")).lower()

            for pattern in SERVICE_EXPLOIT_PATTERNS:
                pat_svc = pattern["service_pattern"].lower()
                pat_ver = pattern.get("version_pattern")

                if pat_svc in svc_name:
                    # Version match if specified
                    if pat_ver and pat_ver.lower() not in svc_version:
                        continue

                    template = pattern["then_try"]
                    # Skip already-tested templates
                    if template in self._tested_templates:
                        continue

                    h = Hypothesis(
                        if_observed=f"Service {svc_name} detected",
                        then_try=template,
                        expected_result=pattern["expected_result"],
                        test_command=template,
                        confidence=pattern["confidence"],
                        reward_if_confirmed=pattern["reward"],
                        source="knowledge_base",
                        evidence_ids=[svc_props.get("node_id", "")],
                        created_step=current_step,
                    )
                    hypotheses.append(h)
                    self._hypotheses[h.id] = h

        # Sort by expected value descending
        hypotheses.sort(key=lambda h: h.expected_value(), reverse=True)
        return hypotheses[:max_hypotheses]

    def get_top_untested(self, evidence_graph: Any = None, n: int = 3) -> List[Hypothesis]:
        """Return top N untested hypotheses by expected_value."""
        untested = [
            h for h in self._hypotheses.values()
            if h.status == HypothesisStatus.UNTESTED
        ]
        untested.sort(key=lambda h: h.expected_value(), reverse=True)
        return untested[:n]

    def update_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        result_summary: str = "",
        tested_step: Optional[int] = None,
    ) -> None:
        """Update hypothesis status. Enforces valid transitions."""
        if hypothesis_id not in self._hypotheses:
            return

        h = self._hypotheses[hypothesis_id]
        # Enforce valid transitions: UNTESTED → TESTING → CONFIRMED/REFUTED/EXPIRED
        valid_transitions = {
            HypothesisStatus.UNTESTED: {HypothesisStatus.TESTING, HypothesisStatus.EXPIRED},
            HypothesisStatus.TESTING: {
                HypothesisStatus.CONFIRMED,
                HypothesisStatus.REFUTED,
                HypothesisStatus.EXPIRED,
            },
        }
        allowed = valid_transitions.get(h.status, set())
        if status not in allowed:
            logger.warning(
                f"Invalid hypothesis transition: {h.status} → {status} for {hypothesis_id}"
            )
            return

        h.status = status
        h.result_summary = result_summary
        if tested_step is not None:
            h.tested_step = tested_step
        if status in (HypothesisStatus.CONFIRMED, HypothesisStatus.REFUTED):
            self._tested_templates.add(h.then_try)

    def reset(self) -> None:
        """Reset for new episode."""
        self._hypotheses.clear()
        self._tested_templates.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return hypothesis statistics."""
        by_status = {}
        for h in self._hypotheses.values():
            s = h.status.value
            by_status[s] = by_status.get(s, 0) + 1
        return {
            "total": len(self._hypotheses),
            "by_status": by_status,
            "tested_templates": len(self._tested_templates),
        }
