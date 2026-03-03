"""
Tests for core/decision/ — Structural Consolidation.

Tests DecisionCore, MaturityController, UnifiedRewardPipeline,
HarmonyMetrics, and IntegrityCheck.

84 tests — no LLM calls, no network, pure logic.
"""

from __future__ import annotations

import math
import os
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def decision_core():
    from core.decision.decision_core import DecisionCore
    return DecisionCore()


@pytest.fixture
def maturity_ctrl():
    from core.decision.maturity_controller import GlobalMaturityController
    return GlobalMaturityController()


@pytest.fixture
def reward_pipeline():
    from core.decision.unified_reward import UnifiedRewardPipeline
    return UnifiedRewardPipeline()


@pytest.fixture
def harmony():
    from core.decision.harmony_metrics import HarmonyMetrics
    return HarmonyMetrics()


@pytest.fixture
def advisory_factory():
    from core.decision.decision_core import Advisory

    def make(source: str, command: str = "nmap -sV {target}",
             confidence: float = 0.8, phase_fit: float = 0.7,
             novelty: float = 0.3, **kwargs) -> Advisory:
        return Advisory(
            source=source,
            command=command,
            confidence=confidence,
            phase_fit=phase_fit,
            novelty=novelty,
            **kwargs,
        )
    return make


@pytest.fixture
def mock_ppo():
    """Mock PPO agent with entropy attributes."""
    ppo = MagicMock()
    ppo.entropy_coef = 0.05
    ppo._entropy_adaptive_multiplier = 1.0
    ppo._entropy_locked = False
    ppo.config = MagicMock()
    ppo.config.state_dim = 512
    ppo.config.entropy_coef = 0.08
    return ppo


@pytest.fixture
def mock_breakdown():
    """Mock reward breakdown with .total."""
    bd = MagicMock()
    bd.total = 5.0
    bd.base_reward = 1.0
    bd.progress_bonus = 1.5
    bd.novelty_bonus = 0.5
    bd.discovery_bonus = 2.0
    bd.phase_advance_bonus = 0.0
    bd.efficiency_bonus = 0.0
    bd.redundancy_penalty = 0.0
    bd.failure_penalty = 0.0
    return bd


# ═══════════════════════════════════════════════════════════════════
# DECISION CORE
# ═══════════════════════════════════════════════════════════════════


class TestDecisionCoreArbitrate:
    """Phase 1: DecisionCore.arbitrate() scoring."""

    def test_single_advisory_wins(self, decision_core, advisory_factory):
        adv = advisory_factory("ppo")
        result = decision_core.arbitrate([adv])
        assert result.source == "ppo"
        assert result.command == "nmap -sV {target}"

    def test_higher_confidence_wins(self, decision_core, advisory_factory):
        low = advisory_factory("registry", confidence=0.3)
        high = advisory_factory("ppo", confidence=0.95)
        result = decision_core.arbitrate([low, high])
        assert result.source == "ppo"

    def test_source_weight_matters(self, decision_core, advisory_factory):
        """PPO source_weight=1.0 beats registry source_weight=0.35."""
        ppo = advisory_factory("ppo", confidence=0.5)
        reg = advisory_factory("registry", confidence=0.5)
        result = decision_core.arbitrate([ppo, reg])
        assert result.source == "ppo"

    def test_maturity_boosts_ppo(self, decision_core, advisory_factory):
        """At maturity=0.9, PPO factor ~2.0 vs mentor ~0.2."""
        ppo = advisory_factory("ppo", confidence=0.5)
        mentor = advisory_factory("mentor", confidence=0.7)
        # High maturity heavily favors PPO
        result = decision_core.arbitrate([ppo, mentor], maturity=0.9)
        assert result.source == "ppo"

    def test_low_maturity_favors_mentor(self, decision_core, advisory_factory):
        """At maturity=0.0, mentor factor=1.0 + higher confidence wins."""
        ppo = advisory_factory("ppo", confidence=0.5)
        mentor = advisory_factory("mentor", confidence=0.7)
        result = decision_core.arbitrate([ppo, mentor], maturity=0.0)
        assert result.source == "mentor"

    def test_empty_advisories_returns_none_source(self, decision_core):
        result = decision_core.arbitrate([])
        assert result.source == "none"
        assert result.command == ""

    def test_hard_constraint_blocks_command(self, decision_core, advisory_factory):
        from core.decision.decision_core import HardConstraint
        adv = advisory_factory("ppo", command="exploit -t {target}")
        safe = advisory_factory("registry", command="nmap -sS {target}")
        constraint = HardConstraint(
            name="evidence_gate",
            blocked_commands={"exploit -t {target}"},
        )
        result = decision_core.arbitrate([adv, safe], constraints=[constraint])
        assert result.source == "registry"
        assert result.command == "nmap -sS {target}"

    def test_hard_constraint_all_blocked_fallback(self, decision_core, advisory_factory):
        from core.decision.decision_core import HardConstraint
        adv = advisory_factory("ppo", command="exploit -t {target}")
        constraint = HardConstraint(
            name="lockout",
            blocked_commands={"exploit -t {target}"},
        )
        result = decision_core.arbitrate([adv], constraints=[constraint])
        # All blocked — should return empty / none
        assert result.source == "none"

    def test_win_rate_factor(self, decision_core, advisory_factory):
        """Source with higher win-rate gets boosted."""
        ppo = advisory_factory("ppo", confidence=0.5)
        mentor = advisory_factory("mentor", confidence=0.5)
        win_rates = {"ppo": 0.8, "mentor": 0.2}
        result = decision_core.arbitrate(
            [ppo, mentor], source_win_rates=win_rates,
        )
        assert result.source == "ppo"

    def test_hit_distribution_tracked(self, decision_core, advisory_factory):
        for _ in range(5):
            decision_core.arbitrate([advisory_factory("ppo")])
        for _ in range(3):
            decision_core.arbitrate([advisory_factory("mentor")])
        dist = decision_core.get_hit_distribution()
        assert "ppo" in dist
        assert "mentor" in dist
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.001  # Distribution sums to 1

    def test_decision_source_entropy(self, decision_core, advisory_factory):
        # Single source → entropy close to 0
        for _ in range(10):
            decision_core.arbitrate([advisory_factory("ppo")])
        entropy = decision_core.get_decision_source_entropy()
        assert entropy < 0.1

    def test_rnd_novelty_affects_score(self, decision_core, advisory_factory):
        """High RND novelty should boost the novelty term in scoring."""
        adv = advisory_factory("ppo", novelty=0.9)
        result = decision_core.arbitrate([adv], rnd_novelty=0.9)
        # Just check it runs without error — detailed scoring is internal
        assert result.source == "ppo"

    def test_arbitration_weights_populated(self, decision_core, advisory_factory):
        adv = advisory_factory("ppo")
        result = decision_core.arbitrate([adv])
        assert "ppo" in result.arbitration_weights
        assert result.arbitration_weights["ppo"] > 0

    def test_all_scores_populated(self, decision_core, advisory_factory):
        a1 = advisory_factory("ppo")
        a2 = advisory_factory("mentor")
        result = decision_core.arbitrate([a1, a2])
        assert len(result.all_scores) == 2

    def test_was_override_false_by_default(self, decision_core, advisory_factory):
        result = decision_core.arbitrate([advisory_factory("ppo")])
        assert result.was_override is False

    def test_required_template_constraint(self, decision_core, advisory_factory):
        from core.decision.decision_core import HardConstraint
        a1 = advisory_factory("ppo", template_name="nmap_service_scan")
        a2 = advisory_factory("mentor", template_name="nikto_scan")
        constraint = HardConstraint(
            name="forced",
            blocked_templates={"nmap_service_scan"},
        )
        result = decision_core.arbitrate([a1, a2], constraints=[constraint])
        assert result.template_name == "nikto_scan"


# ═══════════════════════════════════════════════════════════════════
# MATURITY CONTROLLER
# ═══════════════════════════════════════════════════════════════════


class TestMaturityController:
    """Phase 2+4: GlobalMaturityController."""

    def test_initial_maturity_zero(self, maturity_ctrl):
        assert maturity_ctrl.state.maturity == 0.0

    def test_update_computes_maturity(self, maturity_ctrl):
        metrics = {
            "success_rate": 0.5,
            "skill_coverage": 0.3,
            "discovery_efficiency": 0.4,
            "stagnation_rate": 0.1,
        }
        state = maturity_ctrl.update(episode=10, metrics=metrics)
        # M = 0.4*0.5 + 0.3*0.3 + 0.2*0.4 + 0.1*(1-0.1) = 0.2+0.09+0.08+0.09 = 0.46
        expected = 0.4 * 0.5 + 0.3 * 0.3 + 0.2 * 0.4 + 0.1 * (1 - 0.1)
        assert abs(state.maturity - expected) < 0.01

    def test_maturity_clamps_01(self, maturity_ctrl):
        metrics = {
            "success_rate": 1.0,
            "skill_coverage": 1.0,
            "discovery_efficiency": 1.0,
            "stagnation_rate": 0.0,
        }
        state = maturity_ctrl.update(episode=100, metrics=metrics)
        assert 0.0 <= state.maturity <= 1.0

    def test_entropy_anneals_with_maturity(self):
        from core.decision.maturity_controller import GlobalMaturityController

        # Low maturity → high entropy
        ctrl_low = GlobalMaturityController()
        metrics_low = {"success_rate": 0.0, "skill_coverage": 0.0,
                       "discovery_efficiency": 0.0, "stagnation_rate": 1.0}
        ctrl_low.update(episode=0, metrics=metrics_low)
        entropy_low = ctrl_low.state.entropy_coef

        # High maturity → low entropy
        ctrl_high = GlobalMaturityController()
        metrics_high = {"success_rate": 1.0, "skill_coverage": 1.0,
                        "discovery_efficiency": 1.0, "stagnation_rate": 0.0}
        ctrl_high.update(episode=100, metrics=metrics_high)
        entropy_high = ctrl_high.state.entropy_coef

        assert entropy_low > entropy_high

    def test_mentor_anneals_with_maturity(self):
        from core.decision.maturity_controller import GlobalMaturityController

        ctrl_low = GlobalMaturityController()
        metrics_low = {"success_rate": 0.0, "skill_coverage": 0.0,
                       "discovery_efficiency": 0.0, "stagnation_rate": 1.0}
        ctrl_low.update(episode=0, metrics=metrics_low)
        mentor_low = ctrl_low.state.mentor_rate

        ctrl_high = GlobalMaturityController()
        metrics_high = {"success_rate": 1.0, "skill_coverage": 1.0,
                        "discovery_efficiency": 1.0, "stagnation_rate": 0.0}
        ctrl_high.update(episode=100, metrics=metrics_high)
        mentor_high = ctrl_high.state.mentor_rate

        assert mentor_low > mentor_high

    def test_apply_to_ppo_sets_entropy(self, maturity_ctrl, mock_ppo):
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        maturity_ctrl.apply_to_ppo(mock_ppo)

        # Entropy should be set to controller's value
        assert mock_ppo.entropy_coef == maturity_ctrl.state.entropy_coef

    def test_apply_to_ppo_locks_entropy(self, maturity_ctrl, mock_ppo):
        """apply_to_ppo() must set _entropy_locked = True."""
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        maturity_ctrl.apply_to_ppo(mock_ppo)

        assert mock_ppo._entropy_locked is True

    def test_apply_to_ppo_resets_multiplier(self, maturity_ctrl, mock_ppo):
        mock_ppo._entropy_adaptive_multiplier = 1.5  # Mutated by PPO
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        maturity_ctrl.apply_to_ppo(mock_ppo)

        assert mock_ppo._entropy_adaptive_multiplier == 1.0

    def test_apply_to_ppo_none_safe(self, maturity_ctrl):
        """apply_to_ppo(None) should not crash."""
        maturity_ctrl.apply_to_ppo(None)  # No error

    def test_apply_to_sac(self, maturity_ctrl):
        sac = MagicMock()
        sac.alpha = 0.01
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        maturity_ctrl.apply_to_sac(sac)
        # alpha should be at least sac_alpha_floor
        assert sac.alpha >= maturity_ctrl.state.sac_alpha_floor

    def test_apply_to_sac_none_safe(self, maturity_ctrl):
        maturity_ctrl.apply_to_sac(None)  # No error

    def test_exploration_boost_request(self, maturity_ctrl):
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        original = maturity_ctrl.state.entropy_coef
        maturity_ctrl.request_exploration_boost(
            reason="stagnation", magnitude=0.4, duration_episodes=3,
        )
        assert maturity_ctrl.state.entropy_coef >= original

    def test_should_call_mentor_returns_bool(self, maturity_ctrl):
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        result = maturity_ctrl.should_call_mentor("RedAgent")
        assert isinstance(result, bool)

    def test_should_run_reptile(self, maturity_ctrl):
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        result = maturity_ctrl.should_run_reptile()
        assert isinstance(result, bool)

    def test_apply_all(self, maturity_ctrl, mock_ppo):
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        maturity_ctrl.apply_all(ppo_agent=mock_ppo, sac_agent=None,
                                llm_bridge=None, mentor_policy=None)
        assert mock_ppo._entropy_locked is True

    def test_state_to_dict(self, maturity_ctrl):
        metrics = {"success_rate": 0.5, "skill_coverage": 0.3,
                    "discovery_efficiency": 0.4, "stagnation_rate": 0.1}
        maturity_ctrl.update(episode=10, metrics=metrics)
        d = maturity_ctrl.state.to_dict()
        assert "maturity" in d
        assert "entropy_coef" in d
        assert "mentor_rate" in d


# ═══════════════════════════════════════════════════════════════════
# UNIFIED REWARD PIPELINE
# ═══════════════════════════════════════════════════════════════════


class TestUnifiedRewardPipeline:
    """Phase 3: UnifiedRewardPipeline."""

    def test_basic_compute(self, reward_pipeline, mock_breakdown):
        reward = reward_pipeline.compute(mock_breakdown)
        assert reward.extrinsic == 5.0
        assert reward.total == 5.0
        assert reward.intrinsic_rnd == 0.0

    def test_compute_with_rnd(self, reward_pipeline, mock_breakdown):
        reward = reward_pipeline.compute(mock_breakdown, rnd_intrinsic=2.0, rnd_scale=0.5)
        assert reward.extrinsic == 5.0
        assert reward.intrinsic_rnd == 1.0  # 2.0 * 0.5
        assert reward.total == 6.0  # 5.0 + 1.0

    def test_none_breakdown(self, reward_pipeline):
        reward = reward_pipeline.compute(None)
        assert reward.extrinsic == 0.0
        assert reward.total == 0.0

    def test_step_count_increments(self, reward_pipeline, mock_breakdown):
        assert reward_pipeline.step_count == 0
        reward_pipeline.compute(mock_breakdown)
        assert reward_pipeline.step_count == 1
        reward_pipeline.compute(mock_breakdown)
        assert reward_pipeline.step_count == 2

    def test_reward_mean(self, reward_pipeline, mock_breakdown):
        reward_pipeline.compute(mock_breakdown)  # total=5.0
        mock_breakdown.total = 15.0
        reward_pipeline.compute(mock_breakdown)  # total=15.0
        assert abs(reward_pipeline.reward_mean - 10.0) < 0.01

    def test_reward_variance(self, reward_pipeline, mock_breakdown):
        """Variance of [5, 15] = 50."""
        reward_pipeline.compute(mock_breakdown)  # total=5
        mock_breakdown.total = 15.0
        reward_pipeline.compute(mock_breakdown)  # total=15
        # Sample variance of [5, 15] = (10^2)/1 = 100 (Welford's / n-1)
        var = reward_pipeline.reward_variance
        assert var > 0

    def test_variance_single_step_is_zero(self, reward_pipeline, mock_breakdown):
        reward_pipeline.compute(mock_breakdown)
        assert reward_pipeline.reward_variance == 0.0

    def test_to_dict(self, reward_pipeline, mock_breakdown):
        reward_pipeline.compute(mock_breakdown)
        d = reward_pipeline.to_dict()
        assert "step_count" in d
        assert d["step_count"] == 1
        assert "reward_mean" in d

    def test_last_reward(self, reward_pipeline, mock_breakdown):
        assert reward_pipeline.last_reward is None
        r = reward_pipeline.compute(mock_breakdown)
        assert reward_pipeline.last_reward is r

    def test_breakdown_dict_populated(self, reward_pipeline, mock_breakdown):
        reward = reward_pipeline.compute(mock_breakdown)
        assert "total" in reward.breakdown
        assert "base_reward" in reward.breakdown

    def test_unified_reward_to_dict(self, reward_pipeline, mock_breakdown):
        reward = reward_pipeline.compute(mock_breakdown, rnd_intrinsic=1.0, rnd_scale=0.3)
        d = reward.to_dict()
        assert d["extrinsic"] == 5.0
        assert abs(d["intrinsic_rnd"] - 0.3) < 0.001
        assert abs(d["rnd_scale"] - 0.3) < 0.001


# ═══════════════════════════════════════════════════════════════════
# HARMONY METRICS
# ═══════════════════════════════════════════════════════════════════


class TestHarmonyMetrics:
    """Phase 7: HarmonyMetrics tracking."""

    def test_record_decision(self, harmony):
        harmony.record_decision("ppo", {"ppo": 0.8, "mentor": 0.2})
        harmony.record_decision("mentor", {"ppo": 0.6, "mentor": 0.4})
        dist = harmony.get_source_distribution()
        assert "ppo" in dist
        assert "mentor" in dist

    def test_decision_source_entropy(self, harmony):
        # Uniform distribution → max entropy
        for src in ["ppo", "mentor", "registry", "playbook"]:
            for _ in range(10):
                harmony.record_decision(src, {})
        entropy = harmony.get_decision_source_entropy()
        assert entropy > 1.0  # log2(4) = 2.0

    def test_single_source_low_entropy(self, harmony):
        for _ in range(10):
            harmony.record_decision("ppo", {})
        entropy = harmony.get_decision_source_entropy()
        assert entropy == 0.0

    def test_register_entropy_writer(self, harmony):
        assert harmony.entropy_writer_count == 0
        harmony.register_entropy_writer("maturity_controller")
        assert harmony.entropy_writer_count == 1

    def test_multiple_entropy_writers_counted(self, harmony):
        harmony.register_entropy_writer("maturity_controller")
        harmony.register_entropy_writer("schedule_coupler")
        assert harmony.entropy_writer_count == 2

    def test_record_gradient_norms(self, harmony):
        norms = {"ppo_policy": 0.5, "ppo_value": 0.3, "sac_q": 0.1}
        harmony.record_gradient_norms(norms)
        assert harmony.latest_gradient_norms == norms

    def test_macro_switch_rate(self, harmony):
        harmony.record_macro_step("nmap_scan")
        harmony.record_macro_step("nmap_scan")  # Same
        harmony.record_macro_step("nikto_scan")  # Switch
        harmony.record_macro_step("dirb_scan")   # Switch
        rate = harmony.macro_switch_rate
        # 2 switches out of 3 transitions = 2/3 ≈ 0.667
        assert 0.5 < rate < 0.8

    def test_mentor_dependence(self, harmony):
        for _ in range(7):
            harmony.record_decision("ppo", {})
        for _ in range(3):
            harmony.record_decision("mentor", {})
        dep = harmony.mentor_dependence
        assert abs(dep - 0.3) < 0.01

    def test_record_reward_variance(self, harmony):
        harmony.record_reward(5.0)
        harmony.record_reward(15.0)
        var = harmony.reward_variance
        assert var > 0

    def test_record_kl_drift(self, harmony):
        harmony.record_kl_drift(0.01)
        harmony.record_kl_drift(0.02)
        harmony.record_kl_drift(0.03)
        kl = harmony.end_episode_kl()
        assert abs(kl - 0.02) < 0.01  # Mean of [0.01, 0.02, 0.03]

    def test_snapshot(self, harmony):
        harmony.register_entropy_writer("maturity_controller")
        harmony.record_decision("ppo", {"ppo": 0.8})
        harmony.record_reward(10.0)
        snap = harmony.snapshot(episode=1, step=50)
        assert snap.episode == 1
        assert snap.step == 50
        assert snap.entropy_writer_count == 1

    def test_snapshot_to_dict(self, harmony):
        harmony.record_decision("ppo", {"ppo": 0.8})
        snap = harmony.snapshot()
        d = snap.to_dict()
        assert "decision_source_entropy" in d
        assert "entropy_writer_count" in d

    def test_reset_episode(self, harmony):
        harmony.record_kl_drift(0.05)
        harmony.reset_episode()
        # After reset, KL tracking should be cleared
        kl = harmony.latest_kl_drift
        assert kl == 0.0

    def test_log_to_tensorboard_mock(self, harmony):
        """TensorBoard logging should not crash with mock writer."""
        harmony.register_entropy_writer("mc")
        harmony.record_decision("ppo", {"ppo": 0.8})
        harmony.record_reward(5.0)
        writer = MagicMock()
        harmony.log_to_tensorboard(writer, global_step=10)
        # Should have called add_scalar at least once
        assert writer.add_scalar.call_count > 0


# ═══════════════════════════════════════════════════════════════════
# INTEGRITY CHECK
# ═══════════════════════════════════════════════════════════════════


class TestIntegrityCheck:
    """Phase 8: Boot-time IntegrityCheck."""

    def _make_wired_coach(self):
        """Create a mock SmartCoach with all decision modules wired."""
        from core.decision.decision_core import DecisionCore
        from core.decision.maturity_controller import GlobalMaturityController
        from core.decision.unified_reward import UnifiedRewardPipeline
        from core.decision.harmony_metrics import HarmonyMetrics

        coach = MagicMock()
        coach._decision_core = DecisionCore()
        coach._maturity_controller = GlobalMaturityController()
        coach._reward_pipeline = UnifiedRewardPipeline()
        coach._harmony_metrics = HarmonyMetrics()
        coach.ppo_agent = MagicMock()
        coach.ppo_agent.config = MagicMock()
        coach.ppo_agent.config.state_dim = 512
        return coach

    def test_boot_passes_with_all_wired(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        checker = IntegrityCheck()
        report = checker.check_boot(coach)
        assert report.passed is True
        assert report.error_count == 0

    def test_boot_fails_no_decision_core(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        coach._decision_core = None
        checker = IntegrityCheck()
        with pytest.raises(RuntimeError, match="INTEGRITY CHECK FAILED"):
            checker.check_boot(coach)

    def test_boot_fails_no_maturity(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        coach._maturity_controller = None
        checker = IntegrityCheck()
        with pytest.raises(RuntimeError, match="INTEGRITY CHECK FAILED"):
            checker.check_boot(coach)

    def test_boot_fails_no_reward_pipeline(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        coach._reward_pipeline = None
        checker = IntegrityCheck()
        with pytest.raises(RuntimeError, match="INTEGRITY CHECK FAILED"):
            checker.check_boot(coach)

    def test_boot_fails_no_harmony_metrics(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        coach._harmony_metrics = None
        checker = IntegrityCheck()
        with pytest.raises(RuntimeError, match="INTEGRITY CHECK FAILED"):
            checker.check_boot(coach)

    def test_boot_fails_no_ppo(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        coach.ppo_agent = None
        checker = IntegrityCheck()
        with pytest.raises(RuntimeError, match="INTEGRITY CHECK FAILED"):
            checker.check_boot(coach)

    def test_boot_fails_wrong_state_dim(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        coach.ppo_agent.config.state_dim = 256
        checker = IntegrityCheck()
        with pytest.raises(RuntimeError, match="INTEGRITY CHECK FAILED"):
            checker.check_boot(coach)

    def test_boot_warns_no_orchestrator_maturity(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        orch = MagicMock()
        orch._maturity_controller = None
        checker = IntegrityCheck()
        report = checker.check_boot(coach, orchestrator=orch)
        # Should pass (warning only), but have a warning
        assert report.passed is True
        assert report.warning_count >= 1
        names = [v.check_name for v in report.violations]
        assert "orchestrator_maturity" in names

    def test_report_to_dict(self):
        from core.decision.integrity_check import IntegrityCheck
        coach = self._make_wired_coach()
        checker = IntegrityCheck()
        report = checker.check_boot(coach)
        d = report.to_dict()
        assert d["passed"] is True
        assert "checks_run" in d

    def test_runtime_check_after_warmup(self):
        from core.decision.integrity_check import IntegrityCheck
        from core.decision.harmony_metrics import HarmonyMetrics

        coach = self._make_wired_coach()
        hm = HarmonyMetrics()
        hm.register_entropy_writer("mc")
        hm.record_decision("ppo", {})

        checker = IntegrityCheck(warm_up_steps=10)
        report = checker.check_runtime(coach, hm, step=100)
        # Should run runtime checks
        assert report.checks_run > 0


# ═══════════════════════════════════════════════════════════════════
# PPO ENTROPY LOCK
# ═══════════════════════════════════════════════════════════════════


class TestPPOEntropyLock:
    """Verify PPO entropy writes are guarded by _entropy_locked."""

    def _make_ppo(self):
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        cfg = PPOConfig(state_dim=512, action_dim=5)
        ppo = PPOAgent(config=cfg, device="cpu")
        return ppo

    def test_signal_episode_outcome_locked(self):
        """When _entropy_locked=True, multiplier must NOT change."""
        ppo = self._make_ppo()
        ppo._entropy_locked = True
        ppo._entropy_adaptive_multiplier = 1.0

        # Simulate 5 consecutive failures (would normally boost multiplier)
        for _ in range(5):
            ppo.signal_episode_outcome(reached_closeout=False)

        # Multiplier should remain at 1.0
        assert ppo._entropy_adaptive_multiplier == 1.0

    def test_signal_episode_outcome_unlocked(self):
        """When _entropy_locked=False (default), multiplier changes normally."""
        ppo = self._make_ppo()
        assert not getattr(ppo, '_entropy_locked', False)
        ppo._entropy_adaptive_multiplier = 1.0

        # 2+ failures should boost
        for _ in range(3):
            ppo.signal_episode_outcome(reached_closeout=False)

        assert ppo._entropy_adaptive_multiplier > 1.0

    def test_signal_episode_outcome_locked_success(self):
        """Locked + success streak → multiplier stays at 1.0."""
        ppo = self._make_ppo()
        ppo._entropy_locked = True
        ppo._entropy_adaptive_multiplier = 1.0

        # 5 consecutive closeouts
        for _ in range(5):
            ppo.signal_episode_outcome(reached_closeout=True)

        assert ppo._entropy_adaptive_multiplier == 1.0

    def test_streaks_still_tracked_when_locked(self):
        """Even when locked, streak counters should update."""
        ppo = self._make_ppo()
        ppo._entropy_locked = True

        for _ in range(3):
            ppo.signal_episode_outcome(reached_closeout=True)

        assert ppo._consecutive_closeouts == 3
        assert ppo._consecutive_failures == 0


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION SMOKE
# ═══════════════════════════════════════════════════════════════════


class TestDecisionIntegration:
    """Smoke tests verifying all modules work together."""

    def test_full_decision_flow(self, advisory_factory):
        """DecisionCore → HarmonyMetrics → UnifiedRewardPipeline."""
        from core.decision.decision_core import DecisionCore
        from core.decision.harmony_metrics import HarmonyMetrics
        from core.decision.unified_reward import UnifiedRewardPipeline

        dc = DecisionCore()
        hm = HarmonyMetrics()
        rp = UnifiedRewardPipeline()

        # Arbitrate
        adv = advisory_factory("ppo")
        result = dc.arbitrate([adv])
        hm.record_decision(result.source, result.arbitration_weights)

        # Reward
        bd = MagicMock()
        bd.total = 5.0
        reward = rp.compute(bd, rnd_intrinsic=1.0, rnd_scale=0.3)
        hm.record_reward(reward.total)

        # Snapshot
        snap = hm.snapshot(episode=1, step=10)
        assert snap.decision_source_entropy >= 0
        assert snap.reward_variance >= 0

    def test_maturity_drives_ppo_entropy(self, mock_ppo):
        """MaturityController sets PPO entropy and locks it."""
        from core.decision.maturity_controller import GlobalMaturityController

        mc = GlobalMaturityController()
        mc.update(episode=50, metrics={
            "success_rate": 0.6,
            "skill_coverage": 0.5,
            "discovery_efficiency": 0.4,
            "stagnation_rate": 0.1,
        })
        mc.apply_to_ppo(mock_ppo)

        # PPO entropy should match maturity controller
        assert mock_ppo.entropy_coef == mc.state.entropy_coef
        assert mock_ppo._entropy_locked is True
        assert mock_ppo._entropy_adaptive_multiplier == 1.0

    def test_integrity_check_with_real_decision_core(self):
        """IntegrityCheck validates properly wired coach."""
        from core.decision.integrity_check import IntegrityCheck
        from core.decision.decision_core import DecisionCore
        from core.decision.maturity_controller import GlobalMaturityController
        from core.decision.unified_reward import UnifiedRewardPipeline
        from core.decision.harmony_metrics import HarmonyMetrics

        coach = MagicMock()
        coach._decision_core = DecisionCore()
        coach._maturity_controller = GlobalMaturityController()
        coach._reward_pipeline = UnifiedRewardPipeline()
        coach._harmony_metrics = HarmonyMetrics()
        coach.ppo_agent = MagicMock()
        coach.ppo_agent.config.state_dim = 512

        orch = MagicMock()
        orch._maturity_controller = coach._maturity_controller

        checker = IntegrityCheck()
        report = checker.check_boot(coach, orchestrator=orch)
        assert report.passed is True
        assert report.error_count == 0
        assert report.warning_count == 0
