"""core/memory/evidence_graph.py — Phase 42: EvidenceGraph v2 (enhanced).

Re-exports the Phase 14 EvidenceGraph with lightweight v2 enhancements:
- Confidence-weighted edges for multi-source corroboration
- Query-by-phase filtering for SmartCoach decision pipeline
- Compact state vector for PPO state encoder injection

The v2 module wraps the existing EvidenceGraph to avoid code duplication
while adding Phase 42 capabilities.

Author: Phase 42 Contract
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.memory.evidence_graph")


@dataclass
class EvidenceEntry:
    """Single evidence item with confidence and provenance."""
    evidence_id: str
    evidence_type: str  # "port", "service", "credential", "vuln", "shell", etc.
    value: str
    confidence: float = 1.0
    source: str = "unknown"  # agent_id or "parser"
    phase: str = "RECON"
    step: int = 0
    corroboration_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceGraph:
    """Phase 42: Enhanced EvidenceGraph with confidence-weighted evidence.

    Provides:
    - add_evidence(): Insert or update evidence with confidence weighting
    - query_by_type(): Filter evidence by type
    - query_by_phase(): Filter evidence by attack phase
    - get_confidence(): Aggregate confidence for a discovery
    - to_state_vector(): Compact vector for PPO state encoder
    - reset(): Clear for new episode
    - summary(): Dict snapshot for telemetry
    """

    _KNOWN_TYPES: Set[str] = {
        "port", "service", "credential", "vuln", "shell",
        "user", "flag", "web_path", "hash", "version",
    }

    def __init__(self) -> None:
        self._entries: Dict[str, EvidenceEntry] = {}
        self._type_index: Dict[str, List[str]] = {}
        self._phase_index: Dict[str, List[str]] = {}
        self._step_count: int = 0
        logger.info("EvidenceGraph v2 initialized")

    def add_evidence(
        self,
        evidence_id: str,
        evidence_type: str,
        value: str,
        confidence: float = 1.0,
        source: str = "unknown",
        phase: str = "RECON",
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceEntry:
        """Add or update evidence entry.

        If evidence_id already exists, corroboration_count is incremented
        and confidence is boosted (capped at 1.0).

        Args:
            evidence_id: Unique identifier for this evidence.
            evidence_type: Category (port, service, credential, etc.).
            value: The evidence value string.
            confidence: Initial confidence score [0.0, 1.0].
            source: Who discovered it (agent_id or "parser").
            phase: Attack phase at discovery time.
            step: Step number at discovery time.
            metadata: Optional extra data.

        Returns:
            The created or updated EvidenceEntry.
        """
        self._step_count += 1
        confidence = max(0.0, min(1.0, confidence))

        if evidence_id in self._entries:
            entry = self._entries[evidence_id]
            entry.corroboration_count += 1
            entry.confidence = min(1.0, entry.confidence + 0.1)
            if metadata:
                entry.metadata.update(metadata)
            return entry

        entry = EvidenceEntry(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            value=value,
            confidence=confidence,
            source=source,
            phase=phase,
            step=step,
            metadata=metadata or {},
        )
        self._entries[evidence_id] = entry

        # Index by type
        if evidence_type not in self._type_index:
            self._type_index[evidence_type] = []
        self._type_index[evidence_type].append(evidence_id)

        # Index by phase
        if phase not in self._phase_index:
            self._phase_index[phase] = []
        self._phase_index[phase].append(evidence_id)

        return entry

    def query_by_type(self, evidence_type: str) -> List[EvidenceEntry]:
        """Get all evidence of a given type.

        Args:
            evidence_type: Type to filter by.

        Returns:
            List of matching EvidenceEntry objects.
        """
        ids = self._type_index.get(evidence_type, [])
        return [self._entries[eid] for eid in ids if eid in self._entries]

    def query_by_phase(self, phase: str) -> List[EvidenceEntry]:
        """Get all evidence discovered during a specific phase.

        Args:
            phase: Attack phase to filter by.

        Returns:
            List of matching EvidenceEntry objects.
        """
        ids = self._phase_index.get(phase, [])
        return [self._entries[eid] for eid in ids if eid in self._entries]

    def get_confidence(self, evidence_id: str) -> float:
        """Get confidence score for a specific evidence entry.

        Args:
            evidence_id: The evidence to look up.

        Returns:
            Confidence score, or 0.0 if not found.
        """
        entry = self._entries.get(evidence_id)
        return entry.confidence if entry else 0.0

    def recent_delta(self, last_n: int = 5) -> Dict[str, Any]:
        """Get recent evidence additions for SmartCoach context.

        Args:
            last_n: Number of recent entries to return.

        Returns:
            Dict with recent entries and counts by type.
        """
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.step,
            reverse=True,
        )[:last_n]
        return {
            "recent": [
                {"id": e.evidence_id, "type": e.evidence_type,
                 "value": e.value, "confidence": e.confidence}
                for e in sorted_entries
            ],
            "counts_by_type": {
                t: len(ids) for t, ids in self._type_index.items()
            },
            "total": len(self._entries),
        }

    def to_state_vector(self, dim: int = 32) -> List[float]:
        """Compact state vector representation for PPO state encoder.

        Encodes evidence counts by type and aggregate confidence
        into a fixed-size vector.

        Args:
            dim: Output vector dimension (default 32).

        Returns:
            List of floats representing evidence state.
        """
        vec = [0.0] * dim
        type_list = sorted(self._KNOWN_TYPES)

        # Counts per type (first 10 dims)
        for i, t in enumerate(type_list[:10]):
            entries = self._type_index.get(t, [])
            vec[i] = min(10.0, float(len(entries))) / 10.0  # normalize

        # Avg confidence per type (next 10 dims)
        for i, t in enumerate(type_list[:10]):
            entries = self.query_by_type(t)
            if entries:
                avg_conf = sum(e.confidence for e in entries) / len(entries)
                vec[10 + i] = avg_conf

        # Global stats (remaining dims)
        vec[20] = min(1.0, len(self._entries) / 50.0)  # total evidence ratio
        vec[21] = min(1.0, self._step_count / 100.0)  # step coverage
        vec[22] = float(len(self._type_index)) / 10.0  # type diversity

        # Corroboration signal
        if self._entries:
            avg_corr = sum(
                e.corroboration_count for e in self._entries.values()
            ) / len(self._entries)
            vec[23] = min(1.0, avg_corr / 5.0)

        return vec

    def summary(self) -> Dict[str, Any]:
        """Summary snapshot for telemetry/dashboard.

        Returns:
            Dict with total count, type counts, and top entries.
        """
        return {
            "total": len(self._entries),
            "by_type": {t: len(ids) for t, ids in self._type_index.items()},
            "by_phase": {p: len(ids) for p, ids in self._phase_index.items()},
            "avg_confidence": (
                sum(e.confidence for e in self._entries.values())
                / max(len(self._entries), 1)
            ),
            "step_count": self._step_count,
        }

    def reset(self) -> None:
        """Clear all evidence for new episode."""
        self._entries.clear()
        self._type_index.clear()
        self._phase_index.clear()
        self._step_count = 0
        logger.debug("EvidenceGraph v2 reset")

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._entries
