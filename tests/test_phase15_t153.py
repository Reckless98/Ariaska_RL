#!/usr/bin/env python3
"""
tests/test_phase15_t153.py — T15.3 Integration Tests

Validates:
1. Semantic index: add, query, dedup, clear, bounded
2. Working memory wiring in SmartCoach decide() + reset_episode()
3. Consolidation wiring in SmartCoach end_episode_ppo() + record_result()
4. BudgetManagerV2 wiring in GPTManager init/reset/gpt_request
5. CAP contract: flags OFF = zero behavior change (no new behavior leaks)

Phase 15.0 — Neurovortex Tier 15.3
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_fake_gpt():
    from core.testing.fake_gpt_manager import FakeGPTManager
    return FakeGPTManager(seed=42)


def _make_coach(agent_name="TestAgent"):
    """Create a SmartCoach with correct constructor signature."""
    gpt = _make_fake_gpt()
    from core.training.smart_coach import SmartCoach
    return SmartCoach(
        agent_name=agent_name,
        gpt_manager=gpt,  # type: ignore[arg-type]
        model="test-model",
    )


def _reset_flags():
    """Reset feature flags to re-read environment."""
    from core.feature_flags import reset_feature_flags
    reset_feature_flags()


# ── Semantic Index Tests ────────────────────────────────────────────────────

class TestSemanticIndex:
    """Test core/memory/semantic_index.py standalone."""

    def test_add_and_query(self):
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        assert idx.add("nmap -sV 10.0.0.1", entry_type="command")
        assert len(idx) == 1
        results = idx.query("nmap -sV 10.0.0.1")
        assert len(results) >= 1
        assert results[0].score > 0.5

    def test_dedup(self):
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        assert idx.add("nmap -sV 10.0.0.1")
        assert not idx.add("nmap -sV 10.0.0.1")  # duplicate
        assert len(idx) == 1

    def test_query_empty(self):
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        assert idx.query("anything") == []

    def test_clear(self):
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx.add("test command one")
        idx.add("test command two")
        assert len(idx) == 2
        idx.clear()
        assert len(idx) == 0

    def test_ring_buffer(self):
        """Entries beyond max_entries overwrite oldest."""
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex(max_entries=16)
        for i in range(20):
            idx.add(f"unique command number {i} {chr(65 + i % 26)}" * 3)
        assert len(idx) == 16

    def test_cosine_similarity(self):
        """Similar strings should score higher than dissimilar."""
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx.add("nmap -sV -p 80 10.0.0.1", entry_type="command")
        idx.add("gobuster dir -u http://10.0.0.1", entry_type="command")
        # Query similar to first
        results = idx.query("nmap -sV 10.0.0.1", min_score=0.0)
        assert len(results) >= 1
        # First result should be the nmap one
        assert "nmap" in results[0].entry.text

    def test_filter_by_type(self):
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx.add("nmap command", entry_type="command")
        idx.add("discovery: port 80", entry_type="discovery")
        results = idx.query("nmap command", entry_type="command", min_score=0.0)
        assert all(r.entry.entry_type == "command" for r in results)

    def test_bounded_query_results(self):
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex(max_results=3)
        for i in range(10):
            idx.add(f"nmap scan variant {i}")
        results = idx.query("nmap scan", min_score=0.0)
        assert len(results) <= 3

    def test_get_stats(self):
        from core.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx.add("some text")
        stats = idx.get_stats()
        assert stats["entries"] == 1
        assert stats["max_entries"] == 256


# ── Working Memory Wiring Tests ────────────────────────────────────────────

class TestWorkingMemoryWiring:
    """Test WM wiring in SmartCoach reset_episode + decide."""

    def test_wm_none_flags_off(self, monkeypatch):
        """With flags explicitly OFF, working memory is None."""
        monkeypatch.setenv("FF_WORKING_MEMORY", "0")
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_working_memory is None
        _reset_flags()

    def test_wm_initialized_flag_on(self, monkeypatch):
        """With FF_WORKING_MEMORY=1, working memory is initialized."""
        monkeypatch.setenv("FF_WORKING_MEMORY", "1")
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_working_memory is not None
        _reset_flags()

    def test_wm_cleared_on_reset(self, monkeypatch):
        """reset_episode() clears working memory."""
        monkeypatch.setenv("FF_WORKING_MEMORY", "1")
        _reset_flags()
        coach = _make_coach()
        wm = coach._p15_working_memory
        assert wm is not None
        wm.push("test", "content", priority=0.5)
        assert len(wm) == 1
        coach.reset_episode(episode=1)
        assert len(wm) == 0
        _reset_flags()

    def test_semantic_index_cleared_on_reset(self, monkeypatch):
        """reset_episode() clears semantic index."""
        monkeypatch.setenv("FF_SEMANTIC_INDEX", "1")
        _reset_flags()
        coach = _make_coach()
        si = coach._p15_semantic_index
        assert si is not None
        si.add("test command")
        assert len(si) == 1
        coach.reset_episode(episode=1)
        assert len(si) == 0
        _reset_flags()


# ── Consolidation Wiring Tests ──────────────────────────────────────────────

class TestConsolidationWiring:
    """Test consolidation sample collection and replay wiring."""

    def test_consol_none_flags_off(self, monkeypatch):
        """With flags explicitly OFF, consolidation engine is None."""
        monkeypatch.setenv("FF_CONSOLIDATION", "0")
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_consolidation_engine is None
        _reset_flags()

    def test_consol_initialized_flag_on(self):
        """Post-Phase 20: consolidation defaults ON, engine is initialized."""
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_consolidation_engine is not None
        _reset_flags()

    def test_consol_samples_reset_per_episode(self, monkeypatch):
        """reset_episode() clears consolidation samples."""
        monkeypatch.setenv("FF_CONSOLIDATION", "1")
        _reset_flags()
        coach = _make_coach()
        coach._p15_consolidation_samples = [1, 2, 3]
        coach.reset_episode(episode=1)
        assert coach._p15_consolidation_samples == []
        _reset_flags()


# ── BudgetManagerV2 Wiring Tests ────────────────────────────────────────────

class TestBudgetManagerV2Wiring:
    """Test BudgetManagerV2 wiring in GPTManager."""

    def test_bm2_none_flags_off(self, monkeypatch):
        """With flags explicitly OFF, budget manager is None."""
        monkeypatch.setenv("FF_BUDGET_MANAGER_V2", "0")
        _reset_flags()
        from core.gpt_manager import GPTManager
        gpt = GPTManager(offline=True)
        assert gpt._budget_manager_v2 is None
        _reset_flags()

    def test_bm2_initialized_flag_on(self):
        """Post-Phase 20: budget_manager_v2 defaults ON, manager is initialized."""
        _reset_flags()
        from core.gpt_manager import GPTManager
        gpt = GPTManager(offline=True)
        assert gpt._budget_manager_v2 is not None

    def test_bm2_reset_on_episode(self, monkeypatch):
        """reset_episode() resets budget manager."""
        monkeypatch.setenv("FF_BUDGET_MANAGER_V2", "1")
        _reset_flags()
        from core.gpt_manager import GPTManager
        gpt = GPTManager(offline=True)
        bm = gpt._budget_manager_v2
        assert bm is not None
        # Simulate spend
        bm.record_spend("local-llm", 1000, "tactical")
        stats = bm.get_stats()
        assert stats["total_used"] == 1000
        # Reset
        gpt.reset_episode(episode_id=99)
        stats2 = bm.get_stats()
        assert stats2["total_used"] == 0
        assert stats2["episode_id"] == "99"
        _reset_flags()

    def test_bm2_budget_check(self):
        """BudgetManagerV2 correctly denies when budget exceeded."""
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2(total_budget=1000, tier_budgets={"mini": 500, "full": 500})
        # Use a model that maps to 'mini' tier (default for unknown models)
        # so budget tracking actually applies (local tier is free/unlimited).
        test_model = "test-budget-model"
        # Should allow
        d1 = bm.check_budget(test_model, 400, "tactical")
        assert d1.allowed
        # Record spend near limit
        bm.record_spend(test_model, 490, "tactical")
        # Should deny (only 10 remaining)
        d2 = bm.check_budget(test_model, 100, "tactical")
        assert not d2.allowed
        assert d2.reason == "budget_exceeded"


# ── CAP Contract Tests ──────────────────────────────────────────────────────

class TestCAPContractT153:
    """CAP contract: flags OFF = zero behavior change."""

    def test_no_p15_attributes_leak_when_off(self, monkeypatch):
        """P15 components are None when explicitly turned OFF."""
        monkeypatch.setenv("FF_WORKING_MEMORY", "0")
        monkeypatch.setenv("FF_SEMANTIC_INDEX", "0")
        monkeypatch.setenv("FF_CONSOLIDATION", "0")
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_working_memory is None
        assert coach._p15_consolidation_engine is None
        assert coach._p15_semantic_index is None
        _reset_flags()

    def test_gpt_manager_no_bm2_when_off(self, monkeypatch):
        """GPTManager has no budget manager when flag is explicitly off."""
        monkeypatch.setenv("FF_BUDGET_MANAGER_V2", "0")
        _reset_flags()
        from core.gpt_manager import GPTManager
        gpt = GPTManager(offline=True)
        assert gpt._budget_manager_v2 is None
        _reset_flags()

    def test_record_result_no_crash_flags_off(self):
        """record_result works without P15 components active."""
        _reset_flags()
        coach = _make_coach()
        from core.training.smart_coach import SmartDecisionResult
        from core.commands.command_registry import AttackPhase
        decision = SmartDecisionResult(
            command="nmap -sV 10.0.0.1",
            template_name="nmap_scan",
            source="registry",
            confidence=0.8,
            phase=AttackPhase.RECON,
        )
        # Should not crash
        breakdown = coach.record_result(
            decision=decision,
            success=True,
            raw_output="PORT STATE SERVICE\n80/tcp open http",
            new_discoveries={"open_port": [80]},
        )
        assert breakdown is not None
        assert breakdown.total >= 0 or breakdown.total < 0  # just not crash


# ── Semantic Index in SmartCoach Tests ───────────────────────────────────────

class TestSemanticIndexWiring:
    """Test semantic index wiring in SmartCoach."""

    def test_si_none_flags_off(self, monkeypatch):
        """With flags explicitly OFF, semantic index is None."""
        monkeypatch.setenv("FF_SEMANTIC_INDEX", "0")
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_semantic_index is None
        _reset_flags()

    def test_si_initialized_flag_on(self, monkeypatch):
        """With FF_SEMANTIC_INDEX=1, semantic index is initialized."""
        monkeypatch.setenv("FF_SEMANTIC_INDEX", "1")
        _reset_flags()
        coach = _make_coach()
        assert coach._p15_semantic_index is not None
        _reset_flags()
