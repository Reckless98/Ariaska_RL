#!/usr/bin/env python3
"""
core/knowledge/evidence_graph.py — Phase 14.0: Per-Episode Evidence Graph

Contract C1.3: Ephemeral per-episode knowledge graph tracking all
discovered evidence (hosts, ports, services, credentials, vulns, sessions).

Lightweight NetworkX-based — NOT LMDB. Seeds from CampaignMemory at
episode start, updates from parser DiscoveryEvents each step.

Author: Phase 14.0 Contract C1.3
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.evidence_graph")

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore


class EvidenceNodeType(str, Enum):
    """Types of evidence nodes."""
    HOST = "host"
    PORT = "port"
    SERVICE = "service"
    CREDENTIAL = "credential"
    VULNERABILITY = "vulnerability"
    SESSION = "session"
    PRIVILEGE = "privilege"
    WEB_PATH = "web_path"


class EvidenceEdgeType(str, Enum):
    """Types of evidence edges."""
    RUNS_ON = "runs_on"
    EXPLOITS = "exploits"
    AUTHENTICATES = "authenticates"
    ESCALATES_TO = "escalates_to"
    LEADS_TO = "leads_to"
    DISCOVERED_BY = "discovered_by"


@dataclass
class EvidenceNode:
    """A node in the per-episode evidence graph."""
    node_id: str
    node_type: EvidenceNodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_step: int = 0
    source_command: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d


@dataclass
class EvidenceEdge:
    """A directed edge in the evidence graph."""
    source_id: str
    target_id: str
    edge_type: EvidenceEdgeType
    confidence: float = 1.0
    evidence_command: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        return d


# Map DiscoveryType values to EvidenceNodeType
_DISCOVERY_TO_NODE_TYPE = {
    "open_port": EvidenceNodeType.PORT,
    "service": EvidenceNodeType.SERVICE,
    "version_info": EvidenceNodeType.SERVICE,
    "credential": EvidenceNodeType.CREDENTIAL,
    "user": EvidenceNodeType.CREDENTIAL,
    "vulnerability": EvidenceNodeType.VULNERABILITY,
    "cve": EvidenceNodeType.VULNERABILITY,
    "shell": EvidenceNodeType.SESSION,
    "root_shell": EvidenceNodeType.SESSION,
    "web_path": EvidenceNodeType.WEB_PATH,
    "flag": EvidenceNodeType.PRIVILEGE,
    "hash": EvidenceNodeType.CREDENTIAL,
    "token": EvidenceNodeType.CREDENTIAL,
}


class EvidenceGraph:
    """
    Per-episode ephemeral evidence graph.

    Contract C1.3: Lightweight NetworkX-based (no LMDB).
    Seeds from CampaignMemory, updates from DiscoveryEvents.
    """

    def __init__(self, target_ip: str = "") -> None:
        self._nodes: Dict[str, EvidenceNode] = {}
        self._edges: List[EvidenceEdge] = []
        self._target_ip = target_ip

        # Initialize NetworkX graph if available
        if nx is not None:
            self._graph = nx.DiGraph()
        else:
            self._graph = None

        # Add host node if target specified
        if target_ip:
            host_id = f"host_{target_ip}"
            self._add_node(EvidenceNode(
                node_id=host_id,
                node_type=EvidenceNodeType.HOST,
                properties={"ip": target_ip},
            ))

    def _add_node(self, node: EvidenceNode) -> bool:
        """Add a node. Returns True if new, False if duplicate."""
        if node.node_id in self._nodes:
            return False
        self._nodes[node.node_id] = node
        if self._graph is not None:
            self._graph.add_node(node.node_id, **node.to_dict())
        return True

    def _add_edge(self, edge: EvidenceEdge) -> bool:
        """Add an edge. Returns True if new."""
        # Check for duplicate
        for existing in self._edges:
            if (existing.source_id == edge.source_id and
                    existing.target_id == edge.target_id and
                    existing.edge_type == edge.edge_type):
                return False
        self._edges.append(edge)
        if self._graph is not None:
            self._graph.add_edge(
                edge.source_id, edge.target_id,
                edge_type=edge.edge_type.value,
                confidence=edge.confidence,
            )
        return True

    def seed_from_campaign(self, campaign: Any) -> int:
        """
        Seed graph from CampaignMemory prior knowledge.

        Returns number of nodes added.
        """
        added = 0
        if campaign is None:
            return added

        # Seed known ports
        if hasattr(campaign, "ports"):
            for port_num, port_obj in campaign.ports.items():
                port_id = f"port_{port_num}"
                props = {"port": int(port_num)}
                if hasattr(port_obj, "service") and port_obj.service:
                    props["service"] = port_obj.service
                if hasattr(port_obj, "version") and port_obj.version:
                    props["version"] = port_obj.version

                if self._add_node(EvidenceNode(
                    node_id=port_id,
                    node_type=EvidenceNodeType.PORT,
                    properties=props,
                    confidence=0.9,
                )):
                    added += 1

                # Link port to host
                if self._target_ip:
                    self._add_edge(EvidenceEdge(
                        source_id=port_id,
                        target_id=f"host_{self._target_ip}",
                        edge_type=EvidenceEdgeType.RUNS_ON,
                    ))

                # Add service node if service info available
                if props.get("service"):
                    svc_id = f"svc_{port_num}_{props['service']}"
                    if self._add_node(EvidenceNode(
                        node_id=svc_id,
                        node_type=EvidenceNodeType.SERVICE,
                        properties=props,
                        confidence=0.9,
                    )):
                        added += 1
                    self._add_edge(EvidenceEdge(
                        source_id=svc_id,
                        target_id=port_id,
                        edge_type=EvidenceEdgeType.RUNS_ON,
                    ))

        # Seed known credentials
        if hasattr(campaign, "credentials"):
            for i, cred in enumerate(campaign.credentials):
                cred_id = f"cred_{i}"
                props = {}
                if hasattr(cred, "username"):
                    props["username"] = cred.username
                if hasattr(cred, "password"):
                    props["password"] = cred.password
                if hasattr(cred, "service"):
                    props["service"] = cred.service

                if self._add_node(EvidenceNode(
                    node_id=cred_id,
                    node_type=EvidenceNodeType.CREDENTIAL,
                    properties=props,
                    confidence=0.95,
                )):
                    added += 1

        # Seed known vulns
        if hasattr(campaign, "vulns"):
            for vuln_id_str, vuln_obj in campaign.vulns.items():
                vid = f"vuln_{vuln_id_str}"
                props = {"vuln_id": vuln_id_str}
                if hasattr(vuln_obj, "vuln_type"):
                    props["vuln_type"] = vuln_obj.vuln_type

                if self._add_node(EvidenceNode(
                    node_id=vid,
                    node_type=EvidenceNodeType.VULNERABILITY,
                    properties=props,
                    confidence=0.9,
                )):
                    added += 1

        logger.debug(f"[EVIDENCE] Seeded {added} nodes from campaign memory")
        return added

    def update_from_discovery(self, event: Any, step: int = 0, command: str = "") -> bool:
        """
        Update graph from a DiscoveryEvent.

        Returns True if new evidence was added.
        """
        if event is None:
            return False

        # Get discovery type value
        dt_val = ""
        if hasattr(event, "discovery_type"):
            dt = event.discovery_type
            dt_val = dt.value if hasattr(dt, "value") else str(dt)

        value = getattr(event, "value", None)
        confidence = getattr(event, "confidence", 1.0)

        node_type = _DISCOVERY_TO_NODE_TYPE.get(dt_val)
        if node_type is None:
            return False

        # Generate node ID
        node_id = f"{dt_val}_{value}"
        props = {"value": value, "discovery_type": dt_val}

        node = EvidenceNode(
            node_id=node_id,
            node_type=node_type,
            properties=props,
            confidence=confidence,
            source_step=step,
            source_command=command,
        )

        added = self._add_node(node)

        # Create edges based on type
        if self._target_ip and node_type == EvidenceNodeType.PORT:
            self._add_edge(EvidenceEdge(
                source_id=node_id,
                target_id=f"host_{self._target_ip}",
                edge_type=EvidenceEdgeType.RUNS_ON,
                confidence=confidence,
                evidence_command=command,
            ))

        if node_type == EvidenceNodeType.SERVICE:
            # Link service to any matching port
            for nid, n in self._nodes.items():
                if n.node_type == EvidenceNodeType.PORT:
                    port_val = n.properties.get("value") or n.properties.get("port")
                    if port_val is not None:
                        self._add_edge(EvidenceEdge(
                            source_id=node_id,
                            target_id=nid,
                            edge_type=EvidenceEdgeType.RUNS_ON,
                        ))
                        break

        if node_type == EvidenceNodeType.CREDENTIAL:
            # Link credential to services
            for nid, n in self._nodes.items():
                if n.node_type == EvidenceNodeType.SERVICE:
                    self._add_edge(EvidenceEdge(
                        source_id=node_id,
                        target_id=nid,
                        edge_type=EvidenceEdgeType.AUTHENTICATES,
                    ))

        return added

    def get_attack_surface(self) -> List[Dict[str, Any]]:
        """Return services not yet linked to a SESSION (unexploited)."""
        exploited_services = set()
        for edge in self._edges:
            if edge.edge_type == EvidenceEdgeType.EXPLOITS:
                exploited_services.add(edge.target_id)

        surface = []
        for nid, node in self._nodes.items():
            if node.node_type == EvidenceNodeType.SERVICE and nid not in exploited_services:
                surface.append(node.properties)
        return surface

    def get_exploit_candidates(self) -> List[Dict[str, Any]]:
        """Return services with known vulnerabilities."""
        vuln_services = set()
        for edge in self._edges:
            if edge.edge_type == EvidenceEdgeType.EXPLOITS:
                vuln_services.add(edge.source_id)

        candidates = []
        for nid, node in self._nodes.items():
            if node.node_type == EvidenceNodeType.VULNERABILITY:
                candidates.append(node.properties)
        return candidates

    def get_credential_targets(self) -> List[Dict[str, Any]]:
        """Return services accepting auth (linked to credentials)."""
        auth_services = set()
        for edge in self._edges:
            if edge.edge_type == EvidenceEdgeType.AUTHENTICATES:
                auth_services.add(edge.target_id)

        targets = []
        for nid in auth_services:
            if nid in self._nodes:
                targets.append(self._nodes[nid].properties)
        return targets

    def to_state_vector(self, max_dims: int = 128) -> Any:
        """
        Encode evidence graph as a fixed-size tensor.

        Returns tensor of shape (max_dims,), always exactly this size.
        """
        if torch is None:
            return [0.0] * max_dims

        vec = torch.zeros(max_dims, dtype=torch.float32)

        # Count by node type (dims 0-7)
        type_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        type_order = [e.value for e in EvidenceNodeType]
        for i, t in enumerate(type_order):
            if i < 8:
                vec[i] = min(10.0, type_counts.get(t, 0)) / 10.0

        # Edge counts (dims 8-13)
        edge_counts: Dict[str, int] = {}
        for edge in self._edges:
            t = edge.edge_type.value
            edge_counts[t] = edge_counts.get(t, 0) + 1

        edge_order = [e.value for e in EvidenceEdgeType]
        for i, t in enumerate(edge_order):
            if i < 6:
                vec[8 + i] = min(10.0, edge_counts.get(t, 0)) / 10.0

        # Graph-level features (dims 14-19)
        vec[14] = min(1.0, self.node_count() / 50.0)
        vec[15] = min(1.0, self.edge_count() / 100.0)
        vec[16] = min(1.0, len(self.get_attack_surface()) / 10.0)
        vec[17] = min(1.0, len(self.get_exploit_candidates()) / 5.0)
        vec[18] = min(1.0, len(self.get_credential_targets()) / 5.0)

        # Confidence statistics (dims 19-21)
        if self._nodes:
            confs = [n.confidence for n in self._nodes.values()]
            vec[19] = sum(confs) / len(confs)  # mean confidence
            vec[20] = min(confs)               # min confidence

        # Port hash features (dims 22-53) — which ports are open
        for nid, node in self._nodes.items():
            if node.node_type == EvidenceNodeType.PORT:
                port_val = node.properties.get("value") or node.properties.get("port", 0)
                try:
                    port_num = int(port_val)
                    idx = 22 + (port_num % 32)
                    vec[idx] = 1.0
                except (ValueError, TypeError):
                    pass

        # Service hash features (dims 54-85)
        for nid, node in self._nodes.items():
            if node.node_type == EvidenceNodeType.SERVICE:
                svc_name = str(node.properties.get("service", node.properties.get("value", "")))
                if svc_name:
                    h = hash(svc_name) & 0xFFFF
                    idx = 54 + (h % 32)
                    vec[idx] = 1.0

        # Remaining dims 86-127 reserved (zero-padded)

        return vec

    def to_prompt_fragment(self, max_chars: int = 2000) -> str:
        """Create a compact text summary for LLM context injection."""
        parts = []

        # Ports
        ports = [n for n in self._nodes.values() if n.node_type == EvidenceNodeType.PORT]
        if ports:
            port_strs = [str(n.properties.get("value", n.properties.get("port", "?"))) for n in ports]
            parts.append(f"Open ports: {', '.join(port_strs[:20])}")

        # Services
        services = [n for n in self._nodes.values() if n.node_type == EvidenceNodeType.SERVICE]
        if services:
            svc_strs = []
            for s in services[:10]:
                name = s.properties.get("service", s.properties.get("value", "?"))
                ver = s.properties.get("version", "")
                svc_strs.append(f"{name}" + (f" {ver}" if ver else ""))
            parts.append(f"Services: {', '.join(svc_strs)}")

        # Credentials
        creds = [n for n in self._nodes.values() if n.node_type == EvidenceNodeType.CREDENTIAL]
        if creds:
            parts.append(f"Credentials found: {len(creds)}")

        # Vulns
        vulns = [n for n in self._nodes.values() if n.node_type == EvidenceNodeType.VULNERABILITY]
        if vulns:
            vuln_strs = [str(n.properties.get("vuln_id", n.properties.get("value", "?")))
                         for n in vulns[:5]]
            parts.append(f"Vulns: {', '.join(vuln_strs)}")

        # Sessions
        sessions = [n for n in self._nodes.values() if n.node_type == EvidenceNodeType.SESSION]
        if sessions:
            parts.append(f"Active sessions: {len(sessions)}")

        result = " | ".join(parts)
        return result[:max_chars]

    def diff_since(self, step: int) -> Dict[str, List[Dict]]:
        """Return nodes and edges added since the given step."""
        new_nodes = [
            n.to_dict() for n in self._nodes.values()
            if n.source_step >= step
        ]
        # Edges don't track step, so return all for now
        return {"nodes": new_nodes, "edges": []}

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def recent_delta(self, window: int = 3) -> int:
        """Return count of nodes added in the last `window` steps.

        Used by neuromodulator system to gauge evidence accumulation rate.
        If no step tracking is available, returns total node count as proxy.
        """
        if not self._nodes:
            return 0
        max_step = max((n.source_step for n in self._nodes.values()), default=0)
        cutoff = max(0, max_step - window)
        return sum(1 for n in self._nodes.values() if n.source_step > cutoff)

    def reset(self) -> None:
        """Reset for new episode."""
        self._nodes.clear()
        self._edges.clear()
        if self._graph is not None:
            self._graph.clear()
        if self._target_ip:
            host_id = f"host_{self._target_ip}"
            self._add_node(EvidenceNode(
                node_id=host_id,
                node_type=EvidenceNodeType.HOST,
                properties={"ip": self._target_ip},
            ))
