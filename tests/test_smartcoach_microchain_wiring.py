#!/usr/bin/env python3
"""Test SmartCoach micro-chain wiring — Phase 27.3."""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestSmartCoachMicroChainWiring:
    """Verify micro-chain integrates into SmartCoach decide() cascade."""

    def test_micro_chain_attribute_present_when_flag_on(self, monkeypatch):
        """SmartCoach inits _micro_chain when FF_USE_MICRO_CHAIN=1."""
        monkeypatch.setenv("FF_USE_MICRO_CHAIN", "1")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

        from core.testing.fake_gpt_manager import FakeGPTManager
        gpt = FakeGPTManager(seed=42)
        from core.training.smart_coach import SmartCoach
        coach = SmartCoach(
            agent_name="ScoutAgent",
            gpt_manager=gpt,  # type: ignore[arg-type]
        )
        assert coach._micro_chain is not None

    def test_micro_chain_attribute_none_when_flag_off(self, monkeypatch):
        """SmartCoach should NOT init _micro_chain when flag is off."""
        monkeypatch.setenv("FF_USE_MICRO_CHAIN", "0")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

        from core.testing.fake_gpt_manager import FakeGPTManager
        gpt = FakeGPTManager(seed=42)
        from core.training.smart_coach import SmartCoach
        coach = SmartCoach(
            agent_name="ScoutAgent",
            gpt_manager=gpt,  # type: ignore[arg-type]
        )
        assert coach._micro_chain is None

    def test_evidence_gate_counters_initialized(self):
        """SmartCoach should have evidence gate counters."""
        from core.testing.fake_gpt_manager import FakeGPTManager
        gpt = FakeGPTManager(seed=42)
        from core.training.smart_coach import SmartCoach
        coach = SmartCoach(
            agent_name="RedAgent",
            gpt_manager=gpt,  # type: ignore[arg-type]
        )
        assert hasattr(coach, '_evidence_gate_total')
        assert coach._evidence_gate_total == 0
        assert hasattr(coach, '_evidence_gate_rejects')
        assert hasattr(coach, '_evidence_gate_reject_but_discovered')
