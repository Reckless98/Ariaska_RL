"""Phase 42 Stage 2: EvidenceGraph v2 unit tests."""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestEvidenceGraphV2:
    """Tests for the enhanced EvidenceGraph module."""

    def _make_graph(self):
        from core.memory.evidence_graph import EvidenceGraph
        return EvidenceGraph()

    def test_init(self):
        """EvidenceGraph initializes empty."""
        g = self._make_graph()
        assert len(g) == 0
        assert g.summary()["total"] == 0

    def test_add_evidence(self):
        """add_evidence creates entry and indexes."""
        g = self._make_graph()
        e = g.add_evidence("port_80", "port", "80/tcp", confidence=0.9, phase="RECON")
        assert e.evidence_id == "port_80"
        assert e.confidence == 0.9
        assert len(g) == 1
        assert "port_80" in g

    def test_corroboration(self):
        """Repeated add_evidence increments corroboration."""
        g = self._make_graph()
        g.add_evidence("port_80", "port", "80/tcp", confidence=0.7)
        e2 = g.add_evidence("port_80", "port", "80/tcp", confidence=0.7)
        assert e2.corroboration_count == 2
        assert e2.confidence == pytest.approx(0.8, abs=0.01)
        assert len(g) == 1  # no duplicate

    def test_query_by_type(self):
        """query_by_type filters correctly."""
        g = self._make_graph()
        g.add_evidence("port_80", "port", "80/tcp")
        g.add_evidence("port_22", "port", "22/tcp")
        g.add_evidence("svc_http", "service", "http")
        ports = g.query_by_type("port")
        assert len(ports) == 2
        svcs = g.query_by_type("service")
        assert len(svcs) == 1

    def test_query_by_phase(self):
        """query_by_phase filters correctly."""
        g = self._make_graph()
        g.add_evidence("e1", "port", "80", phase="RECON")
        g.add_evidence("e2", "service", "http", phase="ENUMERATION")
        recon = g.query_by_phase("RECON")
        assert len(recon) == 1
        enum = g.query_by_phase("ENUMERATION")
        assert len(enum) == 1

    def test_get_confidence(self):
        """get_confidence returns 0.0 for missing entries."""
        g = self._make_graph()
        assert g.get_confidence("nonexistent") == 0.0
        g.add_evidence("e1", "port", "80", confidence=0.85)
        assert g.get_confidence("e1") == pytest.approx(0.85)

    def test_recent_delta(self):
        """recent_delta returns recent entries."""
        g = self._make_graph()
        for i in range(10):
            g.add_evidence(f"e_{i}", "port", str(i), step=i)
        delta = g.recent_delta(last_n=3)
        assert len(delta["recent"]) == 3
        assert delta["total"] == 10

    def test_to_state_vector(self):
        """to_state_vector returns fixed-dim vector."""
        g = self._make_graph()
        g.add_evidence("p1", "port", "80")
        g.add_evidence("s1", "service", "http")
        vec = g.to_state_vector(dim=32)
        assert len(vec) == 32
        assert any(v > 0 for v in vec)

    def test_summary(self):
        """summary returns correct structure."""
        g = self._make_graph()
        g.add_evidence("e1", "port", "80", phase="RECON")
        s = g.summary()
        assert s["total"] == 1
        assert "port" in s["by_type"]
        assert "RECON" in s["by_phase"]
        assert s["avg_confidence"] > 0

    def test_reset(self):
        """reset clears all state."""
        g = self._make_graph()
        g.add_evidence("e1", "port", "80")
        g.add_evidence("e2", "service", "http")
        assert len(g) == 2
        g.reset()
        assert len(g) == 0
        assert g.summary()["total"] == 0

    def test_confidence_clamped(self):
        """Confidence is clamped to [0.0, 1.0]."""
        g = self._make_graph()
        e = g.add_evidence("e1", "port", "80", confidence=1.5)
        assert e.confidence <= 1.0
        e2 = g.add_evidence("e2", "port", "22", confidence=-0.5)
        assert e2.confidence >= 0.0
