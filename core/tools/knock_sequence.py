#!/usr/bin/env python3
"""
core/tools/knock_sequence.py — Phase 10.1D: Port Knocking Sequencing + Timing Inference

Treats port knocking as a reasoning object: sequence definition, timing
inference from knowledge candidates, verification via re-scan, and
attempt limiting to avoid infinite loops.

Architecture:
    KnockSequence — data object: ports, protocol, delays, verification
    KnockInferenceEngine — infers knock sequences from evidence/knowledge
    Registry templates — knock_sequence, verify_port_open

Usage:
    from core.tools.knock_sequence import KnockSequence, KnockInferenceEngine
    engine = KnockInferenceEngine()
    sequences = engine.infer(state, knowledge_candidates)
    for seq in sequences:
        print(seq.to_command("192.168.1.1"))
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.knock_sequence")


@dataclass
class KnockSequence:
    """A port knocking sequence with timing and verification.

    Attributes:
        sequence: Ordered list of ports to knock
        protocol: "tcp", "udp", or "mixed"
        delays_ms: Delay between knocks in milliseconds
        target_port: The port expected to open after knocking
        max_attempts: Maximum retry attempts
        jitter_ms: Random jitter tolerance per knock
        verify_after: Whether to verify target port after knocking
        source: Where this sequence was inferred from
        confidence: Confidence in this sequence (0.0 - 1.0)
    """
    sequence: List[int] = field(default_factory=list)
    protocol: str = "tcp"
    delays_ms: List[int] = field(default_factory=lambda: [500])
    target_port: int = 0
    max_attempts: int = 3
    jitter_ms: int = 100
    verify_after: bool = True
    source: str = ""  # "knowledge", "symptom", "ctf_pattern", "manual"
    confidence: float = 0.5

    def to_command(self, target_ip: str) -> str:
        """Render as a knock client command string.

        Uses the knock utility: knock <host> <port1> <port2> ...
        With optional -d <delay> for inter-knock timing.
        """
        ports_str = " ".join(str(p) for p in self.sequence)
        delay_sec = max(self.delays_ms) / 1000.0 if self.delays_ms else 0.5

        if self.protocol == "udp":
            # UDP knocking with nmap
            return (
                f"for port in {ports_str}; do "
                f"nmap -Pn -sU -p $port --max-retries 0 {target_ip} && "
                f"sleep {delay_sec}; done"
            )

        # Default TCP knock command
        return f"knock -d {int(delay_sec * 1000)} {target_ip} {ports_str}"

    def to_verify_command(self, target_ip: str) -> str:
        """Command to verify the target port opened."""
        if self.target_port:
            return f"nmap -Pn -sT -p {self.target_port} --max-retries 2 {target_ip}"
        return f"nmap -Pn -sT -F {target_ip}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "protocol": self.protocol,
            "delays_ms": self.delays_ms,
            "target_port": self.target_port,
            "max_attempts": self.max_attempts,
            "verify_after": self.verify_after,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class KnockTelemetry:
    """Telemetry for port knocking attempts."""
    knock_attempts: int = 0
    knock_success: int = 0
    verify_results: List[Dict[str, Any]] = field(default_factory=list)
    sequences_inferred: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": self.knock_attempts,
            "success": self.knock_success,
            "verify_results": self.verify_results,
            "sequences_inferred": self.sequences_inferred,
        }


# ============================================================================
# KNOWN KNOCK PATTERNS (from CTF/HTB common setups)
# ============================================================================

# Common CTF knock sequences (target_port → sequences)
KNOWN_PATTERNS: Dict[int, List[KnockSequence]] = {
    22: [  # SSH
        KnockSequence(
            sequence=[7000, 8000, 9000], protocol="tcp",
            delays_ms=[500], target_port=22,
            source="ctf_pattern", confidence=0.3,
        ),
        KnockSequence(
            sequence=[1111, 2222, 3333], protocol="tcp",
            delays_ms=[500], target_port=22,
            source="ctf_pattern", confidence=0.2,
        ),
    ],
    80: [  # HTTP
        KnockSequence(
            sequence=[571, 290, 911], protocol="tcp",
            delays_ms=[500], target_port=80,
            source="ctf_pattern", confidence=0.2,
        ),
    ],
}

# Tags/keywords that suggest port knocking is in play
KNOCK_INDICATORS = {
    "knock", "knockd", "port knocking", "hidden service",
    "filtered", "stealth", "sequence required",
    "knock.conf", "opencloseclose", "closeopenclose",
}


# ============================================================================
# KNOCK INFERENCE ENGINE
# ============================================================================

class KnockInferenceEngine:
    """Infers port knock sequences from state and knowledge evidence.

    Inference sources:
    1. Knowledge candidates tagged with 'knock' or 'knockd'
    2. Symptom analysis: filtered ports, SSH filtered, banner hints
    3. Known CTF patterns
    """

    def __init__(self) -> None:
        self._telemetry = KnockTelemetry()

    @property
    def telemetry(self) -> KnockTelemetry:
        return self._telemetry

    def infer(
        self,
        state: Dict[str, Any],
        knowledge_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[KnockSequence]:
        """Infer possible knock sequences from evidence.

        Args:
            state: Current environment state
            knowledge_candidates: Optional KCs that match current target

        Returns:
            List of KnockSequence objects, sorted by confidence descending
        """
        sequences: List[KnockSequence] = []

        # 1. From knowledge candidates
        if knowledge_candidates:
            sequences.extend(self._from_knowledge(knowledge_candidates))

        # 2. From symptoms (filtered ports, etc.)
        sequences.extend(self._from_symptoms(state))

        # 3. Known CTF patterns
        sequences.extend(self._from_ctf_patterns(state))

        # Deduplicate by sequence
        seen_seqs: Set[str] = set()
        unique: List[KnockSequence] = []
        for seq in sequences:
            key = f"{seq.sequence}-{seq.target_port}"
            if key not in seen_seqs:
                seen_seqs.add(key)
                unique.append(seq)

        # Sort by confidence
        unique.sort(key=lambda s: s.confidence, reverse=True)

        self._telemetry.sequences_inferred = len(unique)
        return unique

    def _from_knowledge(
        self, candidates: List[Dict[str, Any]]
    ) -> List[KnockSequence]:
        """Extract knock sequences from knowledge candidates."""
        results: List[KnockSequence] = []

        for kc in candidates:
            tags = set()
            taxonomy = kc.get("taxonomy", {})
            if isinstance(taxonomy, dict):
                tags = set(taxonomy.get("tags", []))

            # Check if this KC is about port knocking
            title = kc.get("title", "").lower()
            desc = kc.get("description", "").lower()
            text = f"{title} {desc}"

            is_knock = bool(tags & KNOCK_INDICATORS) or any(
                ind in text for ind in KNOCK_INDICATORS
            )
            if not is_knock:
                continue

            # Try to extract sequence from command templates
            execution = kc.get("execution", {})
            if isinstance(execution, dict):
                cmd = execution.get("original_command", "")
                seq = self._parse_sequence_from_command(cmd)
                if seq:
                    seq.source = "knowledge"
                    seq.confidence = 0.6
                    results.append(seq)

        return results

    def _from_symptoms(self, state: Dict[str, Any]) -> List[KnockSequence]:
        """Infer from environment symptoms."""
        results: List[KnockSequence] = []
        state_flags = state.get("state_flags", {})

        # Check for filtered SSH (common knock target)
        services = state.get("services", {})
        open_ports = set(state.get("open_ports", []))
        banners = state.get("service_banners", {})

        # If SSH is known but port is filtered/closed
        ssh_expected = False
        if isinstance(state_flags, set):
            ssh_expected = "ssh_service_found" not in state_flags
        else:
            ssh_expected = not state_flags.get("ssh_service_found", False)

        # Check banners for knock hints
        for port, banner in (banners.items() if isinstance(banners, dict) else []):
            banner_lower = str(banner).lower()
            if any(ind in banner_lower for ind in KNOCK_INDICATORS):
                # Try to extract sequence from banner
                seq = self._parse_sequence_from_text(banner)
                if seq:
                    seq.source = "symptom_banner"
                    seq.confidence = 0.7
                    results.append(seq)

        return results

    def _from_ctf_patterns(self, state: Dict[str, Any]) -> List[KnockSequence]:
        """Add known CTF patterns if conditions suggest knocking."""
        results: List[KnockSequence] = []
        open_ports = set(state.get("open_ports", []))

        # Only suggest CTF patterns if target port is not already open
        for target_port, patterns in KNOWN_PATTERNS.items():
            if target_port not in open_ports:
                results.extend(patterns)

        return results

    def _parse_sequence_from_command(self, cmd: str) -> Optional[KnockSequence]:
        """Parse a port sequence from a knock command string."""
        if not cmd:
            return None

        # Pattern: knock <host> <p1> <p2> <p3> ...
        match = re.search(r'knock\s+\S+\s+([\d\s]+)', cmd)
        if match:
            ports = [int(p) for p in match.group(1).split() if p.isdigit()]
            if len(ports) >= 2:
                return KnockSequence(sequence=ports)

        # Pattern: for port in <p1> <p2> <p3>
        match = re.search(r'for\s+port\s+in\s+([\d\s]+)', cmd)
        if match:
            ports = [int(p) for p in match.group(1).split() if p.isdigit()]
            if len(ports) >= 2:
                return KnockSequence(sequence=ports, protocol="udp")

        return None

    def _parse_sequence_from_text(self, text: str) -> Optional[KnockSequence]:
        """Parse a port sequence from free text (banners, descriptions)."""
        # Look for sequences of 3+ numbers that look like ports
        matches = re.findall(r'\b(\d{2,5})\b', text)
        ports = [int(m) for m in matches if 1 <= int(m) <= 65535]
        if len(ports) >= 3:
            return KnockSequence(sequence=ports[:5])
        return None

    def should_propose_knock(
        self, state: Dict[str, Any], step: int, max_step: int = 40
    ) -> bool:
        """Determine if knocking should be proposed.

        Only proposes when:
        - Feature flag enabled
        - Evidence suggests knock (filtered ports, knowledge matches)
        - Not too late in episode (avoid wasting final steps)
        - Attempt limits not exceeded
        """
        from core.feature_flags import get_feature_flags
        if not get_feature_flags().port_knocking:
            return False

        # Don't propose late in episode
        if step > max_step * 0.7:
            return False

        # Check attempt limits
        if self._telemetry.knock_attempts >= 5:
            return False

        return True
