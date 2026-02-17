#!/usr/bin/env python3
"""
tests/test_phase14_evidence_graph.py — Phase 14.0: EvidenceGraph Tests

Contract C3.2: 10 required tests for evidence graph, nodes, edges, vectors.
"""

import os
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestEvidenceGraph:
    """C3.2: EvidenceGraph schema and operation tests."""

    def test_graph_creation_with_target(self):
        """Graph initializes with host node when target_ip provided."""
        from core.knowledge.evidence_graph import EvidenceGraph, EvidenceNodeType
        g = EvidenceGraph(target_ip="10.0.0.1")
        assert g.node_count() == 1  # host node
        assert g.edge_count() == 0

    def test_graph_creation_empty(self):
        """Graph initializes empty when no target_ip."""
        from core.knowledge.evidence_graph import EvidenceGraph
        g = EvidenceGraph()
        assert g.node_count() == 0
        assert g.edge_count() == 0

    def test_node_type_enum_values(self):
        """EvidenceNodeType has all 8 required types."""
        from core.knowledge.evidence_graph import EvidenceNodeType
        types = set(e.value for e in EvidenceNodeType)
        required = {"host", "port", "service", "credential",
                     "vulnerability", "session", "privilege", "web_path"}
        assert required == types

    def test_edge_type_enum_values(self):
        """EvidenceEdgeType has all 6 required types."""
        from core.knowledge.evidence_graph import EvidenceEdgeType
        types = set(e.value for e in EvidenceEdgeType)
        required = {"runs_on", "exploits", "authenticates",
                     "escalates_to", "leads_to", "discovered_by"}
        assert required == types

    def test_update_from_discovery_event(self):
        """update_from_discovery adds nodes from DiscoveryEvent-like objects."""
        from core.knowledge.evidence_graph import EvidenceGraph
        from types import SimpleNamespace

        g = EvidenceGraph(target_ip="10.0.0.1")
        event = SimpleNamespace(
            discovery_type=SimpleNamespace(value="open_port"),
            value="22",
            confidence=0.95,
        )
        added = g.update_from_discovery(event, step=1, command="nmap 10.0.0.1")
        assert added is True
        assert g.node_count() == 2  # host + port

    def test_update_from_discovery_dedup(self):
        """Duplicate discoveries are not re-added."""
        from core.knowledge.evidence_graph import EvidenceGraph
        from types import SimpleNamespace

        g = EvidenceGraph(target_ip="10.0.0.1")
        event = SimpleNamespace(
            discovery_type=SimpleNamespace(value="open_port"),
            value="80",
            confidence=0.9,
        )
        assert g.update_from_discovery(event) is True
        assert g.update_from_discovery(event) is False
        assert g.node_count() == 2  # host + one port (no double-add)

    def test_to_state_vector_shape(self):
        """to_state_vector returns exactly (128,) tensor."""
        import torch
        from core.knowledge.evidence_graph import EvidenceGraph
        from types import SimpleNamespace

        g = EvidenceGraph(target_ip="10.0.0.1")
        # Add some evidence
        for port in ["22", "80", "443"]:
            g.update_from_discovery(SimpleNamespace(
                discovery_type=SimpleNamespace(value="open_port"),
                value=port,
                confidence=0.9,
            ))
        vec = g.to_state_vector(max_dims=128)
        assert isinstance(vec, torch.Tensor)
        assert vec.shape == (128,)

    def test_to_prompt_fragment_bounded(self):
        """to_prompt_fragment respects max_chars limit."""
        from core.knowledge.evidence_graph import EvidenceGraph
        from types import SimpleNamespace

        g = EvidenceGraph(target_ip="10.0.0.1")
        for i in range(50):
            g.update_from_discovery(SimpleNamespace(
                discovery_type=SimpleNamespace(value="open_port"),
                value=str(1000 + i),
                confidence=0.5,
            ))
        fragment = g.to_prompt_fragment(max_chars=200)
        assert len(fragment) <= 200

    def test_get_attack_surface(self):
        """get_attack_surface returns unexploited services."""
        from core.knowledge.evidence_graph import EvidenceGraph
        from types import SimpleNamespace

        g = EvidenceGraph(target_ip="10.0.0.1")
        g.update_from_discovery(SimpleNamespace(
            discovery_type=SimpleNamespace(value="service"),
            value="ssh",
            confidence=0.9,
        ))
        surface = g.get_attack_surface()
        assert len(surface) >= 1

    def test_reset_clears_graph(self):
        """reset() removes all nodes/edges, re-adds host if target_ip."""
        from core.knowledge.evidence_graph import EvidenceGraph
        from types import SimpleNamespace

        g = EvidenceGraph(target_ip="10.0.0.1")
        g.update_from_discovery(SimpleNamespace(
            discovery_type=SimpleNamespace(value="open_port"),
            value="22",
            confidence=0.9,
        ))
        assert g.node_count() >= 2

        g.reset()
        assert g.node_count() == 1  # only host remains
        assert g.edge_count() == 0

    def test_diff_since_step(self):
        """diff_since returns nodes added after given step."""
        from core.knowledge.evidence_graph import EvidenceGraph
        from types import SimpleNamespace

        g = EvidenceGraph(target_ip="10.0.0.1")
        g.update_from_discovery(SimpleNamespace(
            discovery_type=SimpleNamespace(value="open_port"),
            value="22",
            confidence=0.9,
        ), step=1)
        g.update_from_discovery(SimpleNamespace(
            discovery_type=SimpleNamespace(value="open_port"),
            value="80",
            confidence=0.9,
        ), step=5)

        diff = g.diff_since(step=3)
        assert len(diff["nodes"]) >= 1  # port 80 was added at step 5
