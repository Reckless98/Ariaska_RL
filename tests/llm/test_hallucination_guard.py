"""Phase 42 Stage 6: HallucinationGuard unit tests."""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestHallucinationGuard:
    """Tests for the HallucinationGuard module."""

    def _make_guard(self):
        from core.llm.hallucination_guard import HallucinationGuard
        return HallucinationGuard()

    def test_init(self):
        """Guard initializes with known tools."""
        g = self._make_guard()
        stats = g.get_stats()
        assert stats["known_tools"] > 50
        assert stats["total_checks"] == 0

    def test_valid_command(self):
        """Known tool command passes."""
        g = self._make_guard()
        r = g.check("nmap -sV -p 80 10.0.0.1")
        assert r.is_hallucinated is False
        assert r.confidence < 0.5

    def test_unknown_tool(self):
        """Unknown tool raises confidence."""
        g = self._make_guard()
        r = g.check("fabricatedtool123 --scan target")
        assert r.confidence >= 0.3
        assert any("Unknown tool" in reason for reason in r.reasons)

    def test_empty_command(self):
        """Empty command is flagged."""
        g = self._make_guard()
        r = g.check("")
        assert r.is_hallucinated is True
        assert r.confidence == 1.0

    def test_suspicious_long_flag(self):
        """Extremely long flags are flagged."""
        g = self._make_guard()
        r = g.check("nmap --verylongflagthatdoesnotexistanywhere target")
        assert any("long flag" in reason.lower() for reason in r.reasons)

    def test_unbalanced_quotes(self):
        """Unbalanced quotes are flagged."""
        g = self._make_guard()
        r = g.check("echo 'unbalanced quote")
        assert any("quote" in reason.lower() for reason in r.reasons)

    def test_register_custom_tool(self):
        """register_known_tool adds to known set."""
        g = self._make_guard()
        r1 = g.check("mycustomtool --scan")
        assert r1.confidence >= 0.3
        g.register_known_tool("mycustomtool")
        r2 = g.check("mycustomtool --scan")
        assert r2.confidence < r1.confidence

    def test_stats_tracking(self):
        """Stats track checks and flags."""
        g = self._make_guard()
        g.check("nmap -sV target")
        g.check("")  # flagged
        stats = g.get_stats()
        assert stats["total_checks"] == 2
        assert stats["total_flagged"] == 1

    def test_reset(self):
        """reset clears counters."""
        g = self._make_guard()
        g.check("nmap -sV target")
        g.check("")
        g.reset()
        stats = g.get_stats()
        assert stats["total_checks"] == 0
        assert stats["total_flagged"] == 0

    def test_fabricated_cve(self):
        """Implausibly long CVE number is flagged."""
        g = self._make_guard()
        r = g.check("exploit CVE-2024-123456789 --target x")
        assert any("CVE" in reason for reason in r.reasons)

    def test_path_tool_extraction(self):
        """Tools with paths are correctly extracted."""
        g = self._make_guard()
        r = g.check("/usr/bin/nmap -sV target")
        assert r.is_hallucinated is False
