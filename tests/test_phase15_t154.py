#!/usr/bin/env python3
"""
tests/test_phase15_t154.py — T15.4 Final Tier Tests

Validates:
1. P15 telemetry collection from SmartCoach instances
2. BudgetManagerV2 ROI summary + budget pressure
3. LLMCallCache cross-episode clearing
4. Full P15 integration: flags ON orchestration safety
5. Final CAP contract: full suite stable

Phase 15.0 — Neurovortex Tier 15.4
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


def _reset_flags():
    from core.feature_flags import reset_feature_flags
    reset_feature_flags()


def _make_fake_gpt():
    from core.testing.fake_gpt_manager import FakeGPTManager
    return FakeGPTManager(seed=42)


def _make_coach(agent_name="TestAgent"):
    gpt = _make_fake_gpt()
    from core.training.smart_coach import SmartCoach
    return SmartCoach(
        agent_name=agent_name,
        gpt_manager=gpt,  # type: ignore[arg-type]
        model="test-model",
    )


# ── P15 Telemetry Tests ────────────────────────────────────────────────────

class TestP15Telemetry:
    """Test p15_telemetry.py standalone collection."""

    def test_collect_agent_snapshot_defaults(self):
        """Snapshot from coach with flags OFF returns safe defaults."""
        _reset_flags()
        from core.telemetry.p15_telemetry import collect_agent_snapshot
        coach = _make_coach()
        snap = collect_agent_snapshot(coach, agent_name="TestAgent")
        assert snap.agent_name == "TestAgent"
        assert snap.neuromod_da == 0.5  # default
        assert snap.aggression_level == 0.3  # default
        assert snap.working_memory_slots == 0
        assert snap.semantic_index_entries == 0
        assert snap.consolidation_samples == 0

    def test_collect_agent_snapshot_with_flags(self, monkeypatch):
        """Snapshot with flags ON captures component state."""
        monkeypatch.setenv("FF_WORKING_MEMORY", "1")
        monkeypatch.setenv("FF_SEMANTIC_INDEX", "1")
        _reset_flags()
        from core.telemetry.p15_telemetry import collect_agent_snapshot
        coach = _make_coach()
        # Push something into working memory
        assert coach._p15_working_memory is not None
        assert coach._p15_semantic_index is not None
        coach._p15_working_memory.push("test", "content")
        coach._p15_semantic_index.add("nmap -sV")
        snap = collect_agent_snapshot(coach)
        assert snap.working_memory_slots == 1
        assert snap.semantic_index_entries == 1
        _reset_flags()

    def test_collect_episode_metrics(self):
        """collect_episode_metrics aggregates across coaches."""
        _reset_flags()
        from core.telemetry.p15_telemetry import collect_episode_metrics
        coaches = {
            "c1": _make_coach("C1"),
            "c2": _make_coach("C2"),
        }
        metrics = collect_episode_metrics(coaches, episode_id="ep_42")
        assert metrics.episode_id == "ep_42"
        assert len(metrics.agent_snapshots) == 2
        d = metrics.to_dict()
        assert "agents" in d
        assert len(d["agents"]) == 2

    def test_collect_episode_with_budget(self, monkeypatch):
        """Budget stats included when BudgetManagerV2 is active."""
        monkeypatch.setenv("FF_BUDGET_MANAGER_V2", "1")
        _reset_flags()
        from core.gpt_manager import GPTManager
        from core.telemetry.p15_telemetry import collect_episode_metrics
        gpt = GPTManager(offline=True)
        assert gpt._budget_manager_v2 is not None
        metrics = collect_episode_metrics({}, gpt_manager=gpt, episode_id="ep_1")
        assert metrics.budget_stats is not None
        assert "total_budget" in metrics.budget_stats
        _reset_flags()


# ── BudgetManagerV2 ROI Tests ──────────────────────────────────────────────

class TestBudgetROI:
    """Test BudgetManagerV2 ROI and pressure methods."""

    def test_roi_summary_empty(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        summary = bm.get_roi_summary()
        assert summary == {}

    def test_roi_summary_after_spend(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.record_spend("local-llm", 100, "tactical")
        bm.record_spend("local-llm", 200, "tactical")
        bm.record_spend("local-llm", 0, "tactical", cache_hit=True)
        summary = bm.get_roi_summary()
        assert "tactical" in summary
        assert summary["tactical"]["calls"] == 3
        assert summary["tactical"]["tokens"] == 300
        assert summary["tactical"]["cache_hits"] == 1

    def test_budget_pressure(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2(total_budget=1000, tier_budgets={"mini": 1000})
        assert bm.get_budget_pressure() == 0.0
        bm.record_spend("local-llm", 500, "tactical")
        assert abs(bm.get_budget_pressure() - 0.5) < 0.01

    def test_budget_pressure_reset(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2(total_budget=1000, tier_budgets={"mini": 1000})
        bm.record_spend("local-llm", 500, "tactical")
        bm.reset_episode("ep_2")
        assert bm.get_budget_pressure() == 0.0


# ── CallCache Cross-Episode Tests ──────────────────────────────────────────

class TestCallCacheCrossEpisode:
    """Test LLMCallCache cross-episode clearing."""

    def test_clear_episode_keeps_cross(self):
        from core.llm.call_cache import LLMCallCache
        cache = LLMCallCache()
        cache.put("p1", "r1", cross_episode=True, roi_tag="classification")
        cache.put("p2", "r2", cross_episode=False, roi_tag="tactical")
        assert cache.get("p1") == "r1"
        assert cache.get("p2") == "r2"
        cache.clear_episode()
        assert cache.get("p1") == "r1"  # kept
        assert cache.get("p2") is None  # removed

    def test_roi_stats(self):
        from core.llm.call_cache import LLMCallCache
        cache = LLMCallCache()
        cache.put("p1", "r1", roi_tag="tactical")
        cache.put("p2", "r2", roi_tag="classification")
        cache.put("p3", "r3", roi_tag="tactical")
        stats = cache.get_roi_stats()
        assert stats.get("tactical", 0) == 2
        assert stats.get("classification", 0) == 1


# ── Final CAP Contract ─────────────────────────────────────────────────────

class TestFinalCAPContract:
    """Final CAP: flags OFF = zero behavior change."""

    def test_all_p15_none_flags_off(self, monkeypatch):
        """P15 components are None when explicitly turned OFF."""
        for flag in [
            "FF_NEUROMODULATORS", "FF_REFLEX_POLICY", "FF_ACTION_ARBITRATOR",
            "FF_WORKING_MEMORY", "FF_CONSOLIDATION", "FF_AGGRESSION_CONTROLLER",
            "FF_SEMANTIC_INDEX", "FF_SENSORY_BUFFER",
        ]:
            monkeypatch.setenv(flag, "0")
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_neuromod_engine is None
        assert coach._p15_neuromod_state is None
        assert coach._p15_neuromod_history is None
        assert coach._p15_sensory_buffer is None
        assert coach._p15_aggression_controller is None
        assert coach._p15_aggression_history is None
        assert coach._p15_reflex_policy is None
        assert coach._p15_action_arbitrator is None
        assert coach._p15_working_memory is None
        assert coach._p15_consolidation_engine is None
        assert coach._p15_semantic_index is None
        _reset_flags()

    def test_gpt_manager_no_bm2_flags_off(self, monkeypatch):
        """GPTManager has no budget manager when flag explicitly OFF."""
        monkeypatch.setenv("FF_BUDGET_MANAGER_V2", "0")
        _reset_flags()
        from core.gpt_manager import GPTManager
        gpt = GPTManager(offline=True)
        assert gpt._budget_manager_v2 is None
        _reset_flags()

    def test_all_flags_on_coach_init(self):
        """Post-Phase 20: All P15 flags default ON, coach initializes all components."""
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_neuromod_engine is not None
        assert coach._p15_reflex_policy is not None
        assert coach._p15_action_arbitrator is not None
        assert coach._p15_working_memory is not None
        assert coach._p15_consolidation_engine is not None
        assert coach._p15_semantic_index is not None
        assert coach._p15_sensory_buffer is not None
        assert coach._p15_aggression_controller is not None
        assert coach._p15_initialized is True
        _reset_flags()

    def test_reset_episode_all_flags_no_crash(self):
        """Post-Phase 20: reset_episode with all flags ON (default) doesn't crash."""
        _reset_flags()
        coach = _make_coach()
        # Push some state into components
        assert coach._p15_working_memory is not None
        assert coach._p15_semantic_index is not None
        coach._p15_working_memory.push("k", "v")
        coach._p15_semantic_index.add("test")
        coach._p15_consolidation_samples = [1, 2, 3]
        # Reset
        coach.reset_episode(episode=5)
        assert len(coach._p15_working_memory) == 0
        assert len(coach._p15_semantic_index) == 0
        assert coach._p15_consolidation_samples == []
        assert coach._p15_aggression_level == 0.3
        _reset_flags()
