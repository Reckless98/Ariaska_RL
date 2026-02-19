"""Tests for C4: Recursive web enumeration."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestRecursiveProber:
    def test_import(self):
        from core.ops.recursive_prober import RecursiveProber
        rp = RecursiveProber()
        assert rp is not None

    def test_feed_interesting(self):
        from core.ops.recursive_prober import RecursiveProber
        rp = RecursiveProber()
        rp.feed_discoveries(["/api", "/admin"], "http://10.10.10.1:80")
        assert rp.queued_count == 2

    def test_feed_uninteresting(self):
        from core.ops.recursive_prober import RecursiveProber
        rp = RecursiveProber()
        rp.feed_discoveries(["/favicon.ico"], "http://10.10.10.1:80")
        assert rp.queued_count == 0

    def test_get_probes(self):
        from core.ops.recursive_prober import RecursiveProber
        rp = RecursiveProber()
        rp.feed_discoveries(["/api"], "http://host:80")
        probes = rp.get_next_probes(max_probes=1)
        assert len(probes) == 1
        assert "gobuster" in probes[0]
        assert "/api" in probes[0]

    def test_no_repeat_probe(self):
        from core.ops.recursive_prober import RecursiveProber
        rp = RecursiveProber()
        rp.feed_discoveries(["/api"], "http://host:80")
        rp.get_next_probes()
        probes2 = rp.get_next_probes()
        assert len(probes2) == 0

    def test_max_queue(self):
        from core.ops.recursive_prober import RecursiveProber, RecursiveProberConfig
        rp = RecursiveProber(config=RecursiveProberConfig(max_queued=2))
        rp.feed_discoveries(["/api", "/admin", "/login", "/upload"], "http://h:80")
        assert rp.queued_count <= 2

    def test_depth_limiting(self):
        from core.ops.recursive_prober import RecursiveProber, RecursiveProberConfig
        rp = RecursiveProber(config=RecursiveProberConfig(max_depth=1))
        rp.feed_discoveries(["/api"], "http://h:80")
        rp.get_next_probes()
        rp.record_result("/api", ["/v1", "/v2"], base_url="http://h:80")
        # Depth of /api/v1 = 2, exceeds max_depth=1
        probes = rp.get_next_probes()
        assert len(probes) == 0

    def test_disabled(self):
        from core.ops.recursive_prober import RecursiveProber, RecursiveProberConfig
        rp = RecursiveProber(config=RecursiveProberConfig(enabled=False))
        rp.feed_discoveries(["/api"], "http://h:80")
        assert rp.queued_count == 0
