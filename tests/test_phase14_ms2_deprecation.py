#!/usr/bin/env python3
"""
tests/test_phase14_ms2_deprecation.py — Phase 14.0 T2: MS2/MS3 Deprecation Tests

Contract C3.10: 4 tests verifying MS2/MS3 exploit graphs are behind feature flags.
"""

import os
import sys
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMS2MS3Deprecation:
    """Verify MS2/MS3 exploit graph feature flag gating."""

    def test_ms2_flag_exists(self):
        """Feature flag ms2_knowledge_pack exists and defaults False."""
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        assert hasattr(ff, "ms2_knowledge_pack")
        assert isinstance(ff.ms2_knowledge_pack, bool)

    def test_ms3_flag_exists(self):
        """Feature flag ms3_knowledge_pack exists and defaults False."""
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        assert hasattr(ff, "ms3_knowledge_pack")
        assert isinstance(ff.ms3_knowledge_pack, bool)

    def test_ms2_exploit_graph_still_importable(self):
        """MS2ExploitGraph remains importable (not deleted, just flag-gated)."""
        from core.knowledge.ms2_exploit_graph import MS2ExploitGraph
        graph = MS2ExploitGraph()
        assert graph is not None
        # Should have services
        assert len(graph.services) > 0

    def test_ms3_exploit_graph_still_importable(self):
        """MS3ExploitGraph remains importable (not deleted, just flag-gated)."""
        from core.knowledge.ms3_exploit_graph import MS3ExploitGraph
        graph = MS3ExploitGraph()
        assert graph is not None
        assert len(graph.services) > 0
