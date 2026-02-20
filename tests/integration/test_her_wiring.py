"""Tests for Phase 42 Stage 1A: HindsightReplay wiring into SmartOrchestrator."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestHERWiring:
    """Verify HindsightReplay is lazily wired into SmartOrchestrator."""

    def test_wire_her_initializes(self, minimal_orchestrator):
        """_wire_her() lazy-inits HindsightReplay when flag is on."""
        orch = minimal_orchestrator
        orch._ppo_trajectory = [
            {"phase": 1, "state": [0.0] * 10, "action": 0, "reward": 1.0},
        ]
        orch._wire_her()
        assert orch._her is not None

    def test_wire_her_processes_transitions(self, minimal_orchestrator):
        """HER processes transitions without error."""
        orch = minimal_orchestrator
        orch._ppo_trajectory = [
            {"phase": 1, "state": [0.0] * 10, "action": 0, "reward": 0.5},
            {"phase": 3, "state": [0.1] * 10, "action": 1, "reward": 1.0},
            {"phase": 2, "state": [0.2] * 10, "action": 2, "reward": 0.8},
        ]
        orch._wire_her()
        assert orch._her is not None

    def test_wire_her_empty_trajectory(self, minimal_orchestrator):
        """Empty trajectory doesn't crash."""
        orch = minimal_orchestrator
        orch._ppo_trajectory = []
        orch._wire_her()
        # HER may or may not init (flag-dependent), but no crash

    def test_wire_her_no_trajectory_attr(self, minimal_orchestrator):
        """Missing _ppo_trajectory attribute doesn't crash."""
        orch = minimal_orchestrator
        if hasattr(orch, '_ppo_trajectory'):
            delattr(orch, '_ppo_trajectory')
        orch._wire_her()

    def test_wire_her_flag_off(self, minimal_orchestrator, monkeypatch):
        """When flag is off, HER remains None."""
        monkeypatch.setenv("FF_HER_WIRING", "0")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

        orch = minimal_orchestrator
        orch._her = None
        orch._ppo_trajectory = [
            {"phase": 1, "state": [0.0] * 10, "action": 0, "reward": 1.0},
        ]
        orch._wire_her()
        assert orch._her is None
