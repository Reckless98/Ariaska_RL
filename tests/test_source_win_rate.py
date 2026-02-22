#!/usr/bin/env python3
"""
tests/test_source_win_rate.py — C06: Decision source win-rate EMA tests

Verifies:
  1. SourceWinRateTracker EMA computation correctness
  2. SourceStats raw vs EMA win rates
  3. get_best_source with min-sample gate
  4. get_summary serialization
  5. SmartCoach integration — record_result populates tracker
  6. DecisionPacket source_attribution.source_win_rates populated
  7. Reset behavior
  8. Known sources constant
"""

import os
import sys
import pytest
import torch

os.environ["ARIASKA_DRY_RUN"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════
# 1. EMA computation
# ═══════════════════════════════════════════════════════════════════


class TestEMAComputation:
    """Verify EMA math is correct."""

    def test_initial_ema_is_prior(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.1)
        assert t.get_win_rate("ppo") == 0.5  # unseen → 0.5 prior

    def test_single_success_raises_ema(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.1)
        t.record("ppo", success=True, reward=5.0)
        # EMA = 0.1 * 1.0 + 0.9 * 0.5 = 0.55
        assert abs(t.get_win_rate("ppo") - 0.55) < 1e-6

    def test_single_failure_lowers_ema(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.1)
        t.record("ppo", success=False, reward=-1.0)
        # EMA = 0.1 * 0.0 + 0.9 * 0.5 = 0.45
        assert abs(t.get_win_rate("ppo") - 0.45) < 1e-6

    def test_many_successes_converge_to_1(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.3)
        for _ in range(50):
            t.record("ppo", success=True)
        assert t.get_win_rate("ppo") > 0.99

    def test_many_failures_converge_to_0(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.3)
        for _ in range(50):
            t.record("ppo", success=False)
        assert t.get_win_rate("ppo") < 0.01

    def test_ema_reward_tracking(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.1, reward_alpha=0.5)
        t.record("mentor", success=True, reward=10.0)
        # EMA reward = 0.5 * 10.0 + 0.5 * 0.0 = 5.0
        assert abs(t.get_reward_ema("mentor") - 5.0) < 1e-6

    def test_multiple_sources_independent(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.5)
        for _ in range(10):
            t.record("ppo", success=True)
            t.record("mentor", success=False)
        assert t.get_win_rate("ppo") > 0.9
        assert t.get_win_rate("mentor") < 0.1


# ═══════════════════════════════════════════════════════════════════
# 2. SourceStats
# ═══════════════════════════════════════════════════════════════════


class TestSourceStats:
    """SourceStats raw vs EMA win rates."""

    def test_raw_win_rate_matches_exact(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.1)
        t.record("ppo", success=True)
        t.record("ppo", success=True)
        t.record("ppo", success=False)
        stats = t.sources["ppo"]
        assert stats.wins == 2
        assert stats.total == 3
        assert abs(stats.raw_win_rate - 2 / 3) < 1e-6

    def test_raw_win_rate_zero_when_zero_total(self):
        from core.training.source_win_rate import SourceStats

        s = SourceStats()
        assert s.raw_win_rate == 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. get_best_source
# ═══════════════════════════════════════════════════════════════════


class TestBestSource:
    """get_best_source respects min-sample gate."""

    def test_no_sources_returns_none(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker()
        assert t.get_best_source() is None

    def test_under_5_samples_returns_none(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.5)
        for _ in range(4):
            t.record("ppo", success=True)
        assert t.get_best_source() is None  # 4 < 5

    def test_over_5_samples_returns_best(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker(alpha=0.3)
        for _ in range(6):
            t.record("ppo", success=True)
        for _ in range(6):
            t.record("mentor", success=False)
        assert t.get_best_source() == "ppo"


# ═══════════════════════════════════════════════════════════════════
# 4. Serialization
# ═══════════════════════════════════════════════════════════════════


class TestSerialization:
    """get_summary / to_dict output."""

    def test_summary_keys(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker()
        t.record("ppo", success=True, reward=5.0)
        summary = t.get_summary()
        assert "ppo" in summary
        assert set(summary["ppo"].keys()) == {
            "ema_win_rate",
            "raw_win_rate",
            "ema_reward",
            "total",
        }

    def test_to_dict_alias(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker()
        t.record("ppo", success=True)
        assert t.to_dict() == t.get_summary()


# ═══════════════════════════════════════════════════════════════════
# 5. SmartCoach integration
# ═══════════════════════════════════════════════════════════════════


class TestSmartCoachIntegration:
    """SmartCoach.record_result updates source_win_rate."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager

        self.gpt = FakeGPTManager(seed=42)

    def test_source_win_rate_initialized(self):
        from core.training.smart_coach import SmartCoach

        coach = SmartCoach(agent_name="ScoutAgent", gpt_manager=self.gpt)
        assert hasattr(coach, "source_win_rate")
        assert coach.source_win_rate.get_win_rate("ppo") == 0.5

    def test_record_result_updates_tracker(self):
        from core.training.smart_coach import SmartCoach, SmartDecisionResult
        from core.llm.smart_mentor import AttackContext

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="10.10.10.1")

        decision = SmartDecisionResult(
            command="nmap -sV 10.10.10.1",
            template_name="nmap_service_scan",
            source="ppo",
            confidence=0.8,
        )
        coach.record_result(
            decision=decision,
            success=True,
            raw_output="PORT STATE SERVICE\n22/tcp open ssh",
            new_discoveries={"open_port": ["22"]},
        )
        assert coach.source_win_rate.get_total("ppo") == 1

    def test_multiple_sources_tracked(self):
        from core.training.smart_coach import SmartCoach, SmartDecisionResult
        from core.llm.smart_mentor import AttackContext

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="10.10.10.1")

        for source in ["ppo", "mentor", "registry"]:
            decision = SmartDecisionResult(
                command=f"cmd_{source}",
                template_name=f"template_{source}",
                source=source,
            )
            coach.record_result(
                decision=decision,
                success=(source == "ppo"),
                raw_output="output",
            )

        assert coach.source_win_rate.get_total("ppo") == 1
        assert coach.source_win_rate.get_total("mentor") == 1
        assert coach.source_win_rate.get_total("registry") == 1


# ═══════════════════════════════════════════════════════════════════
# 6. DecisionPacket source_attribution population
# ═══════════════════════════════════════════════════════════════════


class TestDecisionPacketPopulation:
    """DecisionPacket.source_attribution.source_win_rates populated."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager

        self.gpt = FakeGPTManager(seed=42)

    def test_source_win_rates_populated_on_dp(self):
        from core.training.smart_coach import SmartCoach, SmartDecisionResult
        from core.training.decision_packet import DecisionPacket
        from core.llm.smart_mentor import AttackContext

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="10.10.10.1")

        class _MockCtx:
            episode = 1
            step = 1
            agent_name = "RedAgent"
            attack_context = None
            state = {}

        dp = DecisionPacket.from_step_context(_MockCtx())
        coach._current_decision_packet = dp

        decision = SmartDecisionResult(
            command="nmap 10.10.10.1",
            template_name="nmap_basic",
            source="registry",
        )
        coach.record_result(
            decision=decision,
            success=True,
            raw_output="Nmap scan report\n22/tcp open ssh",
            new_discoveries={"open_port": ["22"]},
        )

        # source_win_rates should have at least "registry"
        assert "registry" in dp.attribution.source_win_rates
        assert dp.attribution.source == "registry"


# ═══════════════════════════════════════════════════════════════════
# 7. Reset behavior
# ═══════════════════════════════════════════════════════════════════


class TestReset:
    """SourceWinRateTracker.reset()."""

    def test_reset_clears_all(self):
        from core.training.source_win_rate import SourceWinRateTracker

        t = SourceWinRateTracker()
        t.record("ppo", success=True)
        t.record("mentor", success=True)
        t.reset()
        assert t.get_summary() == {}
        assert t.get_best_source() is None


# ═══════════════════════════════════════════════════════════════════
# 8. Known sources
# ═══════════════════════════════════════════════════════════════════


class TestKnownSources:
    """KNOWN_SOURCES constant."""

    def test_known_sources_is_frozenset(self):
        from core.training.source_win_rate import KNOWN_SOURCES

        assert isinstance(KNOWN_SOURCES, frozenset)

    def test_expected_sources_present(self):
        from core.training.source_win_rate import KNOWN_SOURCES

        for s in ["ppo", "mentor", "registry", "playbook", "fallback", "micro_chain"]:
            assert s in KNOWN_SOURCES
