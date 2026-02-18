#!/usr/bin/env python3
"""Tests for Phase 27 feature flags: micro_chain, strict_exploit_gate, parallel_agents."""

import os
import pytest


class TestPhase27FeatureFlags:
    """Verify new feature flag defaults and env overrides."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove any Phase 27 env overrides before each test."""
        for key in ("FF_USE_MICRO_CHAIN", "FF_STRICT_EXPLOIT_GATE",
                     "FF_PARALLEL_AGENTS", "FF_EPISODE_SUMMARY_EMBEDDING"):
            monkeypatch.delenv(key, raising=False)

    def _fresh_flags(self):
        from core.feature_flags import FeatureFlags
        return FeatureFlags()

    def test_defaults_are_on_and_log(self):
        ff = self._fresh_flags()
        assert ff.use_micro_chain is True
        assert ff.strict_exploit_gate == "enforce"  # P36: default changed to enforce
        assert ff.parallel_agents is True
        assert ff.episode_summary_embedding is True

    def test_env_override_disables_micro_chain(self, monkeypatch):
        monkeypatch.setenv("FF_USE_MICRO_CHAIN", "0")
        ff = self._fresh_flags()
        assert ff.use_micro_chain is False

    def test_env_override_disables_parallel_agents(self, monkeypatch):
        monkeypatch.setenv("FF_PARALLEL_AGENTS", "false")
        ff = self._fresh_flags()
        assert ff.parallel_agents is False

    def test_env_override_exploit_gate_enforce(self, monkeypatch):
        monkeypatch.setenv("FF_STRICT_EXPLOIT_GATE", "enforce")
        ff = self._fresh_flags()
        assert ff.strict_exploit_gate == "enforce"

    def test_env_override_exploit_gate_off(self, monkeypatch):
        monkeypatch.setenv("FF_STRICT_EXPLOIT_GATE", "off")
        ff = self._fresh_flags()
        assert ff.strict_exploit_gate == "off"

    def test_invalid_exploit_gate_falls_back_to_log(self, monkeypatch):
        monkeypatch.setenv("FF_STRICT_EXPLOIT_GATE", "INVALID_VALUE")
        ff = self._fresh_flags()
        assert ff.strict_exploit_gate == "log"

    def test_validate_exploit_gate_helper(self):
        from core.feature_flags import _validate_exploit_gate
        assert _validate_exploit_gate("log") == "log"
        assert _validate_exploit_gate("enforce") == "enforce"
        assert _validate_exploit_gate("off") == "off"
        assert _validate_exploit_gate("  LOG  ") == "log"
        assert _validate_exploit_gate("bad") == "log"
