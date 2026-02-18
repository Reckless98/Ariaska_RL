#!/usr/bin/env python3
"""Phase 33.4: Parser output hash cache tests."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestParserCacheHit:
    """Verify sha1 output hash cache in ParserBroker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.execution.parser_broker import ParserBroker
        self.broker = ParserBroker(
            gpt_manager=None, venice=None,
            enable_gpt=False, enable_venice=False,
            default_mode="fast",
        )

    def test_identical_output_cached(self):
        """Second parse of identical command+output returns cached result."""
        cmd = "nmap -sV 10.10.10.1"
        output = "22/tcp open ssh OpenSSH 7.2\n80/tcp open http Apache 2.4"
        r1 = self.broker.parse(cmd, output, agent_name="test")
        r2 = self.broker.parse(cmd, output, agent_name="test")
        assert r1 == r2
        stats = self.broker.get_stats()
        assert stats.get("cache_hits", 0) >= 1

    def test_different_output_not_cached(self):
        """Different output should NOT be a cache hit."""
        cmd = "nmap -sV 10.10.10.1"
        out1 = "22/tcp open ssh OpenSSH 7.2"
        out2 = "80/tcp open http Apache 2.4"
        self.broker.parse(cmd, out1, agent_name="test")
        self.broker.parse(cmd, out2, agent_name="test")
        stats = self.broker.get_stats()
        # total_calls=2, cache_hits should be 0
        assert stats.get("cache_hits", 0) == 0

    def test_cache_cleared_on_reset(self):
        """reset_episode clears the output cache."""
        cmd = "nmap -sV 10.10.10.1"
        output = "22/tcp open ssh OpenSSH 7.2"
        self.broker.parse(cmd, output, agent_name="test")
        self.broker.reset_episode()
        # After reset, same parse should NOT be a cache hit
        self.broker.parse(cmd, output, agent_name="test")
        stats = self.broker.get_stats()
        assert stats.get("cache_hits", 0) == 0

    def test_empty_output_not_cached(self):
        """Empty/short outputs are rejected before caching."""
        cmd = "echo hi"
        output = ""
        r1 = self.broker.parse(cmd, output, agent_name="test")
        assert r1 == []
        stats = self.broker.get_stats()
        assert stats.get("cache_hits", 0) == 0

    def test_ansi_stripped_before_cache(self):
        """ANSI codes are stripped, so coloured vs plain output match."""
        cmd = "nmap -sV 10.10.10.1"
        plain = "22/tcp open ssh OpenSSH 7.2"
        ansi = "\x1b[32m22/tcp open ssh OpenSSH 7.2\x1b[0m"
        r1 = self.broker.parse(cmd, plain, agent_name="test")
        r2 = self.broker.parse(cmd, ansi, agent_name="test")
        assert r1 == r2
        stats = self.broker.get_stats()
        assert stats.get("cache_hits", 0) >= 1


class TestParserStrictGating:
    """Verify strict LLM gating in fullparse mode."""

    def test_high_value_tools_constant(self):
        from core.execution.parser_broker import _HIGH_VALUE_TOOLS
        assert "nmap" in _HIGH_VALUE_TOOLS
        assert "linpeas" in _HIGH_VALUE_TOOLS
        assert "sudo" in _HIGH_VALUE_TOOLS

    def test_exploit_phases_constant(self):
        from core.execution.parser_broker import _EXPLOIT_PHASES
        assert "EXPLOITATION" in _EXPLOIT_PHASES
        assert "RECON" not in _EXPLOIT_PHASES

    def test_fullparse_regex_first(self):
        """In fullparse mode, regex runs first (no LLM without GPT manager)."""
        from core.execution.parser_broker import ParserBroker
        broker = ParserBroker(
            gpt_manager=None, venice=None,
            enable_gpt=False, enable_venice=False,
            default_mode="intelligent_fullparse",
        )
        output = "22/tcp open ssh OpenSSH 7.2\n80/tcp open http Apache 2.4"
        events = broker.parse("nmap -sV 10.10.10.1", output, "test")
        stats = broker.get_stats()
        # Should have found via regex, no LLM calls
        assert stats["stage1_hits"] >= 1
        assert stats["stage3_hits"] == 0
