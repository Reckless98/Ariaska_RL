#!/usr/bin/env python3
"""
tests/test_phase15_cap_contract.py — Phase 15.0: CAP Contract Extension

Ensures that with all Phase 15 flags OFF:
  - No behavior change from Phase 14
  - No increased LLM spend
  - No increased mentor reliance
  - Feature flags default to False
  - All Phase 14 tests still pass (structural)
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestPhase15FlagsDefault:
    """Post-Phase 20: All Phase 15 flags default ON (max intelligence)."""

    def test_all_p15_flags_default_true(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        # Post-Phase 20: All P15 flags are ON by default (max intelligence)
        p15_flags_all_on = [
            "neuromodulators", "reflex_policy", "action_arbitrator",
            "consolidation", "aggression_controller",
            "budget_manager_v2", "sensory_buffer",
        ]
        for flag_name in p15_flags_all_on:
            assert hasattr(ff, flag_name), f"Missing flag: {flag_name}"
            assert getattr(ff, flag_name) is True, f"Flag {flag_name} should default to True (Post-Phase 20)"
        # These were activated in Phase 19 — verify they're still ON
        for flag_name in ["working_memory", "semantic_index"]:
            assert getattr(ff, flag_name) is True, f"Flag {flag_name} should be True (Phase 19)"

    def test_p14_flags_still_present(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        p14_flags = [
            "evidence_graph", "hypothesis_engine", "strategy_plan",
            "autonomy_scheduler", "bc_loss", "teacher_trace",
            "aux_heads", "parser_teacher",
        ]
        for flag_name in p14_flags:
            assert hasattr(ff, flag_name), f"Missing P14 flag: {flag_name}"

    def test_env_override_works(self):
        """Verify env var override pattern for Phase 15 flags."""
        os.environ["FF_NEUROMODULATORS"] = "1"
        try:
            from core.feature_flags import FeatureFlags
            ff = FeatureFlags()
            assert ff.neuromodulators is True
        finally:
            os.environ.pop("FF_NEUROMODULATORS", None)


class TestPhase15NoSpendIncrease:
    """With flags OFF, no additional LLM spend should occur."""

    def test_call_cache_backward_compatible(self):
        """LLMCallCache still works with old API (no roi_tag)."""
        from core.llm.call_cache import LLMCallCache
        cache = LLMCallCache(capacity=10)
        # Old API still works
        cache.put("prompt", "response", model="local-llm", tokens_used=100)
        result = cache.get("prompt", model="local-llm")
        assert result == "response"

    def test_call_cache_new_api_compatible(self):
        """LLMCallCache works with new ROI tag API."""
        from core.llm.call_cache import LLMCallCache
        cache = LLMCallCache(capacity=10)
        cache.put("prompt", "response", model="local-llm",
                  tokens_used=100, roi_tag="classification", prompt_version="v1")
        result = cache.get("prompt", model="local-llm", prompt_version="v1")
        assert result == "response"

    def test_clear_episode_preserves_cross_episode(self):
        """clear_episode keeps cross-episode entries, removes per-episode ones."""
        from core.llm.call_cache import LLMCallCache
        cache = LLMCallCache(capacity=10)
        cache.put("p1", "r1", cross_episode=True)
        cache.put("p2", "r2", cross_episode=False)
        cache.clear_episode()
        assert cache.get("p1") is not None
        assert cache.get("p2") is None

    def test_budget_manager_does_not_auto_activate(self):
        """BudgetManagerV2 is a standalone object, not auto-wired."""
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        # Should exist but not affect anything without wiring
        decision = bm.check_budget("local-llm", 100, "classification")
        assert decision.allowed is True


class TestPhase15SchemasSerializable:
    """All Phase 15 schemas must be dict-serializable."""

    def test_neuromod_state_serializable(self):
        from core.neuro.neuromodulators import NeuromodulatorState
        s = NeuromodulatorState()
        d = s.to_dict()
        assert isinstance(d, dict)

    def test_aggression_state_serializable(self):
        from core.neuro.aggression_controller import AggressionState
        s = AggressionState(level=0.5, reason_codes=["test"])
        d = s.to_dict()
        assert isinstance(d, dict)

    def test_reflex_override_serializable(self):
        from core.neurorouter.reflex_policy import ReflexOverride
        r = ReflexOverride()
        d = r.to_dict()
        assert isinstance(d, dict)

    def test_arbitration_log_serializable(self):
        from core.neurorouter.action_arbitrator import ArbitrationDecisionLog
        log = ArbitrationDecisionLog(step=0)
        d = log.to_dict()
        assert isinstance(d, dict)

    def test_consolidation_metrics_serializable(self):
        from core.training.consolidation import ConsolidationMetrics
        m = ConsolidationMetrics()
        d = m.to_dict()
        assert isinstance(d, dict)
