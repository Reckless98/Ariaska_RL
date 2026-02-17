#!/usr/bin/env python3
"""
tests/test_phase14_hypothesis.py — Phase 14.0 T1: Hypothesis Engine Tests

Contract C3.7: 5 tests verifying hypothesis generation and selection wiring.
"""

import os
import sys
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHypothesisEngineWiring:
    """Verify HypothesisGenerator and _p14_hypothesis_select integration."""

    def _add_service_to_graph(self, eg, service_name, version=""):
        """Helper: add a service node to EvidenceGraph using internal API."""
        from core.knowledge.evidence_graph import EvidenceNode, EvidenceNodeType
        node = EvidenceNode(
            node_id=f"svc_{service_name}",
            node_type=EvidenceNodeType.SERVICE,
            properties={"service": service_name, "version": version},
            confidence=0.9,
        )
        eg._add_node(node)

    def test_hypothesis_generation_from_evidence(self):
        """HypothesisGenerator generates hypotheses from EvidenceGraph services."""
        from core.reasoning.hypothesis import HypothesisGenerator
        from core.knowledge.evidence_graph import EvidenceGraph

        eg = EvidenceGraph(target_ip="10.0.0.1")
        self._add_service_to_graph(eg, "vsftpd", "2.3.4")

        gen = HypothesisGenerator()
        hypotheses = gen.generate(eg)
        # Should generate at least one hypothesis for vsftpd service
        assert len(hypotheses) > 0
        vsftpd_hyps = [h for h in hypotheses if "vsftpd" in h.if_observed.lower()]
        assert len(vsftpd_hyps) > 0

    def test_hypothesis_selection_returns_untested(self):
        """get_top_untested returns only UNTESTED hypotheses."""
        from core.reasoning.hypothesis import HypothesisGenerator, HypothesisStatus
        from core.knowledge.evidence_graph import EvidenceGraph

        eg = EvidenceGraph()
        self._add_service_to_graph(eg, "ssh")
        self._add_service_to_graph(eg, "http")

        gen = HypothesisGenerator()
        gen.generate(eg)
        untested = gen.get_top_untested(n=5)
        for h in untested:
            assert h.status == HypothesisStatus.UNTESTED

    def test_hypothesis_status_transitions(self):
        """Hypothesis status transitions follow valid paths."""
        from core.reasoning.hypothesis import HypothesisGenerator, HypothesisStatus
        from core.knowledge.evidence_graph import EvidenceGraph

        eg = EvidenceGraph()
        self._add_service_to_graph(eg, "ssh")

        gen = HypothesisGenerator()
        hyps = gen.generate(eg)
        if not hyps:
            pytest.skip("No hypotheses generated for ssh")

        h = hyps[0]
        assert h.status == HypothesisStatus.UNTESTED

        # UNTESTED → TESTING
        gen.update_status(h.id, HypothesisStatus.TESTING)
        assert h.status == HypothesisStatus.TESTING

        # TESTING → CONFIRMED
        gen.update_status(h.id, HypothesisStatus.CONFIRMED)
        assert h.status == HypothesisStatus.CONFIRMED

    def test_get_top_untested_respects_limit(self):
        """get_top_untested(n=1) returns at most 1."""
        from core.reasoning.hypothesis import HypothesisGenerator
        from core.knowledge.evidence_graph import EvidenceGraph

        eg = EvidenceGraph()
        self._add_service_to_graph(eg, "ssh")
        self._add_service_to_graph(eg, "http")
        self._add_service_to_graph(eg, "mysql")

        gen = HypothesisGenerator()
        gen.generate(eg)
        top1 = gen.get_top_untested(n=1)
        assert len(top1) <= 1

    def test_hypothesis_select_produces_command(self):
        """When hypothesis engine is active, it produces a test command."""
        from core.reasoning.hypothesis import HypothesisGenerator, SERVICE_EXPLOIT_PATTERNS
        from core.knowledge.evidence_graph import EvidenceGraph

        eg = EvidenceGraph()
        # Add a service that has known patterns
        first_pattern = SERVICE_EXPLOIT_PATTERNS[0]
        self._add_service_to_graph(
            eg, first_pattern["service_pattern"],
            first_pattern.get("version_pattern", ""),
        )

        gen = HypothesisGenerator()
        hyps = gen.generate(eg)
        assert len(hyps) > 0
        h = hyps[0]
        assert h.test_command  # Must have a test command
        assert len(h.test_command) > 0
