"""
tests/test_ops_token_flex_metrics.py — Phase E: Token Flex + Engagement Metrics

Tests for:
  - EngagementMetrics: step recording, flag capture, progress, stagnation,
    phase velocity, token efficiency, snapshots, reset
  - TokenFlexEngine: phase-based scaling, stagnation boost, time pressure,
    flag reduction, exploit boost, discovery adj, tier hints, clamping
  - Feature flags for Phase 38.4
  - Integration between EngagementMetrics → TokenFlexEngine
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ═══════════════════════════════════════════════════════════════════════════
# EngagementMetrics
# ═══════════════════════════════════════════════════════════════════════════


class TestEngagementMetrics:
    """Test the engagement-level metrics aggregator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.engagement_metrics import EngagementMetrics
        self.metrics = EngagementMetrics()

    def test_initial_state(self):
        progress = self.metrics.get_progress()
        assert progress["total_steps"] == 0
        assert progress["total_discoveries"] == 0
        assert progress["current_phase"] == "RECON"
        assert progress["highest_phase"] == "RECON"
        assert progress["flags_count"] == 0
        assert progress["shells_obtained"] == 0

    def test_record_step_increments(self):
        self.metrics.record_step(step=1, phase="RECON", discoveries=2,
                                 command="nmap -sV", tokens=100)
        self.metrics.record_step(step=2, phase="RECON", discoveries=1,
                                 command="nmap -sC", tokens=200)
        progress = self.metrics.get_progress()
        assert progress["total_steps"] == 2
        assert progress["total_discoveries"] == 3
        assert progress["unique_commands"] == 2
        assert progress["tokens_used"] == 300

    def test_phase_transition_tracking(self):
        self.metrics.record_step(step=1, phase="RECON")
        self.metrics.record_step(step=2, phase="ENUMERATION")
        self.metrics.record_step(step=3, phase="EXPLOITATION")
        progress = self.metrics.get_progress()
        assert progress["current_phase"] == "EXPLOITATION"
        assert progress["highest_phase"] == "EXPLOITATION"
        assert progress["phase_transitions"] == 2

    def test_phase_progress_fraction(self):
        # EXPLOITATION = index 2 out of 7 (CLOSEOUT at index 7)
        self.metrics.record_step(step=1, phase="EXPLOITATION")
        progress = self.metrics.get_progress()
        assert progress["phase_progress"] == pytest.approx(2 / 7, abs=0.01)

    def test_highest_phase_does_not_regress(self):
        self.metrics.record_step(step=1, phase="EXPLOITATION")
        self.metrics.record_step(step=2, phase="RECON")  # fallback
        progress = self.metrics.get_progress()
        assert progress["highest_phase"] == "EXPLOITATION"
        assert progress["current_phase"] == "RECON"

    def test_flag_capture(self):
        self.metrics.record_flag("user_flag", step=10)
        self.metrics.record_flag("root_flag", step=50)
        progress = self.metrics.get_progress()
        assert progress["flags_count"] == 2
        assert progress["flags_captured"]["user_flag"] == 10
        assert progress["flags_captured"]["root_flag"] == 50

    def test_duplicate_flag_ignored(self):
        self.metrics.record_flag("user_flag", step=10)
        self.metrics.record_flag("user_flag", step=20)  # duplicate
        assert self.metrics.get_progress()["flags_count"] == 1
        assert self.metrics.get_progress()["flags_captured"]["user_flag"] == 10

    def test_shell_tracking(self):
        self.metrics.record_step(step=1, shell_obtained=True)
        self.metrics.record_step(step=2, shell_obtained=True)
        assert self.metrics.get_progress()["shells_obtained"] == 2

    def test_stagnation_counter(self):
        # steps with no discoveries increase stagnation
        for i in range(10):
            self.metrics.record_step(step=i, discoveries=0)
        assert self.metrics.get_stagnation_level() == pytest.approx(0.5)

    def test_stagnation_resets_on_discovery(self):
        for i in range(10):
            self.metrics.record_step(step=i, discoveries=0)
        self.metrics.record_step(step=11, discoveries=1)
        assert self.metrics.get_stagnation_level() == 0.0

    def test_stagnation_caps_at_one(self):
        for i in range(30):
            self.metrics.record_step(step=i, discoveries=0)
        assert self.metrics.get_stagnation_level() == 1.0

    def test_exploit_attempt_tracking(self):
        self.metrics.record_exploit_attempt(success=True)
        self.metrics.record_exploit_attempt(success=False)
        self.metrics.record_exploit_attempt(success=True)
        progress = self.metrics.get_progress()
        assert progress["exploit_success_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_phase_velocity(self):
        self.metrics.record_step(step=1, phase="RECON")
        self.metrics.record_step(step=2, phase="ENUMERATION")
        # 1 transition in 2 steps = 50 transitions per 100 steps
        assert self.metrics.get_phase_velocity() == pytest.approx(50.0)

    def test_token_efficiency(self):
        self.metrics.record_step(step=1, discoveries=5, tokens=1000)
        # 5 discoveries / 1.0K tokens = 5.0
        assert self.metrics.get_token_efficiency() == pytest.approx(5.0)

    def test_token_efficiency_zero_tokens(self):
        assert self.metrics.get_token_efficiency() == 0.0

    def test_episode_end_snapshot(self):
        self.metrics.record_step(step=1, phase="RECON", discoveries=2)
        self.metrics.record_episode_end(episode=0)
        snaps = self.metrics.get_snapshots()
        assert len(snaps) == 1
        assert snaps[0].episode == 0
        assert snaps[0].total_discoveries == 2

    def test_multiple_episode_snapshots(self):
        self.metrics.record_step(step=1, discoveries=1)
        self.metrics.record_episode_end(episode=0)
        self.metrics.record_step(step=2, discoveries=3)
        self.metrics.record_episode_end(episode=1)
        snaps = self.metrics.get_snapshots()
        assert len(snaps) == 2
        assert snaps[1].total_discoveries == 4  # cumulative

    def test_command_diversity(self):
        self.metrics.record_step(step=1, command="nmap -sV")
        self.metrics.record_step(step=2, command="nmap -sV")  # repeat
        self.metrics.record_step(step=3, command="gobuster dir")
        progress = self.metrics.get_progress()
        # 2 unique / 3 total
        assert progress["command_diversity"] == pytest.approx(2 / 3, abs=0.01)

    def test_discovery_rate(self):
        self.metrics.record_step(step=1, discoveries=3)
        self.metrics.record_step(step=2, discoveries=0)
        progress = self.metrics.get_progress()
        assert progress["discovery_rate"] == pytest.approx(1.5)

    def test_reset(self):
        self.metrics.record_step(step=1, discoveries=5, tokens=500)
        self.metrics.record_flag("user_flag", step=1)
        self.metrics.reset()
        progress = self.metrics.get_progress()
        assert progress["total_steps"] == 0
        assert progress["total_discoveries"] == 0
        assert progress["flags_count"] == 0
        assert progress["tokens_used"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# TokenFlexEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestTokenFlexEngine:
    """Test the engagement-progress-aware token budget flex engine."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.token_flex import TokenFlexEngine
        self.engine = TokenFlexEngine(max_steps=500)

    def test_default_recon_scale(self):
        result = self.engine.compute(phase="RECON")
        assert result.scale == pytest.approx(1.0)
        assert "RECON" in result.reason

    def test_exploitation_phase_boost(self):
        result = self.engine.compute(phase="EXPLOITATION")
        assert result.scale > 1.0
        assert result.scale == pytest.approx(1.15)

    def test_privesc_phase_boost(self):
        result = self.engine.compute(phase="PRIVILEGE_ESCALATION")
        assert result.scale >= 1.15

    def test_closeout_reduction(self):
        result = self.engine.compute(phase="CLOSEOUT")
        assert result.scale < 1.0
        assert result.scale == pytest.approx(0.60)

    def test_stagnation_boost(self):
        result = self.engine.compute(
            phase="RECON", stagnation_level=0.8)
        assert result.scale > 1.0  # base 1.0 + stagnation boost
        assert "stagnation" in result.reason

    def test_stagnation_below_threshold_no_boost(self):
        result = self.engine.compute(
            phase="RECON", stagnation_level=0.3)
        assert result.scale == pytest.approx(1.0)

    def test_time_pressure_boost(self):
        # step 450/500 = 90%, no flags -> time pressure
        result = self.engine.compute(
            phase="EXPLOITATION", step=450, flags_captured=0)
        assert result.scale > 1.15  # base exploitation + time pressure
        assert "time_pressure" in result.reason

    def test_time_pressure_disabled_with_flags(self):
        # step 450/500 BUT flag already captured -> no time pressure
        result = self.engine.compute(
            phase="EXPLOITATION", step=450, flags_captured=1)
        assert "time_pressure" not in result.reason

    def test_flag_reduction_two_flags(self):
        result = self.engine.compute(
            phase="RECON", flags_captured=2)
        assert result.scale < 1.0
        assert "flag_adj" in result.reason

    def test_flag_reduction_one_flag(self):
        result = self.engine.compute(
            phase="RECON", flags_captured=1)
        assert result.scale < 1.0  # base 1.0 - 0.05 = 0.95

    def test_exploit_struggle_boost(self):
        result = self.engine.compute(
            phase="EXPLOITATION", step=50,
            exploit_success_rate=0.05)
        assert "exploit_struggle" in result.reason
        assert result.scale > 1.15  # base + exploit boost

    def test_discovery_surplus_reduction(self):
        result = self.engine.compute(
            phase="RECON", discovery_rate=0.8)
        assert result.scale < 1.0  # 1.0 - 0.05

    def test_discovery_drought_boost(self):
        result = self.engine.compute(
            phase="RECON", step=50, discovery_rate=0.02)
        assert result.scale > 1.0

    def test_scale_clamped_min(self):
        # All negative signals at once
        result = self.engine.compute(
            phase="CLOSEOUT", flags_captured=2, discovery_rate=0.9)
        assert result.scale >= 0.50

    def test_scale_clamped_max(self):
        # All positive signals at once
        result = self.engine.compute(
            phase="PRIVILEGE_ESCALATION", step=480,
            stagnation_level=1.0, exploit_success_rate=0.05,
            discovery_rate=0.01)
        assert result.scale <= 1.50

    def test_tier_hints_exploitation(self):
        result = self.engine.compute(phase="EXPLOITATION")
        assert result.tier_hints["codex"] > 1.0
        assert result.tier_hints["nano"] < 1.0

    def test_tier_hints_recon(self):
        result = self.engine.compute(phase="RECON")
        assert result.tier_hints["nano"] > 1.0
        assert result.tier_hints["codex"] < 1.0

    def test_tier_hints_stagnation_escalation(self):
        result = self.engine.compute(
            phase="RECON", stagnation_level=0.7)
        assert result.tier_hints["codex"] >= 1.0

    def test_to_dict(self):
        result = self.engine.compute(phase="RECON")
        d = result.to_dict()
        assert "scale" in d
        assert "reason" in d
        assert "tier_hints" in d
        assert "signals" in d

    def test_update_max_steps(self):
        self.engine.update_max_steps(1000)
        # step 450/1000 = 45%, no time pressure
        result = self.engine.compute(
            phase="RECON", step=450, flags_captured=0)
        assert "time_pressure" not in result.reason

    def test_unknown_phase_defaults(self):
        result = self.engine.compute(phase="NONEXISTENT")
        assert result.scale == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Flags
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseEFlags:
    """Phase 38.4 feature flag verification."""

    def test_engagement_metrics_flag_exists(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "engagement_metrics")
        assert ff.engagement_metrics is True

    def test_token_flex_flag_exists(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "token_flex")
        assert ff.token_flex is True


# ═══════════════════════════════════════════════════════════════════════════
# Integration: EngagementMetrics → TokenFlexEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseEIntegration:
    """Integration tests: metrics feed into token flex."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.engagement_metrics import EngagementMetrics
        from core.ops.token_flex import TokenFlexEngine
        self.metrics = EngagementMetrics()
        self.engine = TokenFlexEngine(max_steps=200)

    def test_metrics_to_flex_basic(self):
        """Record engagement progress and feed into flex engine."""
        self.metrics.record_step(step=1, phase="RECON", discoveries=3)
        self.metrics.record_step(step=2, phase="ENUMERATION", discoveries=1)
        progress = self.metrics.get_progress()

        result = self.engine.compute(
            phase=progress["current_phase"],
            step=progress["total_steps"],
            stagnation_level=self.metrics.get_stagnation_level(),
            flags_captured=progress["flags_count"],
            shells_obtained=progress["shells_obtained"],
            exploit_success_rate=progress["exploit_success_rate"],
            discovery_rate=progress["discovery_rate"],
        )
        assert 0.50 <= result.scale <= 1.50

    def test_stagnation_drives_flex_boost(self):
        """Stagnation in metrics should increase flex scale."""
        for i in range(15):
            self.metrics.record_step(step=i, discoveries=0)
        stag = self.metrics.get_stagnation_level()
        assert stag >= 0.40

        result = self.engine.compute(
            phase="EXPLOITATION",
            step=15,
            stagnation_level=stag,
        )
        assert result.scale > 1.15  # base exploitation + stagnation

    def test_flags_drive_flex_reduction(self):
        """Flags captured should reduce flex scale."""
        self.metrics.record_flag("user_flag", step=10)
        self.metrics.record_flag("root_flag", step=20)
        progress = self.metrics.get_progress()

        result = self.engine.compute(
            phase="RECON",
            flags_captured=progress["flags_count"],
        )
        assert result.scale < 1.0

    def test_full_pipeline(self):
        """Full engagement → progress → flex pipeline."""
        # Simulate a 100-step engagement
        for i in range(1, 101):
            phase = "RECON" if i <= 30 else "ENUMERATION" if i <= 60 else "EXPLOITATION"
            disc = 1 if i % 5 == 0 else 0
            self.metrics.record_step(
                step=i, phase=phase, discoveries=disc,
                command=f"cmd_{i % 20}", tokens=50)

        self.metrics.record_exploit_attempt(success=False)
        self.metrics.record_exploit_attempt(success=True)
        self.metrics.record_episode_end(episode=0)

        progress = self.metrics.get_progress()
        assert progress["total_steps"] == 100
        assert progress["total_episodes"] == 1
        assert progress["current_phase"] == "EXPLOITATION"

        result = self.engine.compute(
            phase=progress["current_phase"],
            step=progress["total_steps"],
            stagnation_level=self.metrics.get_stagnation_level(),
            flags_captured=progress["flags_count"],
            exploit_success_rate=progress["exploit_success_rate"],
            discovery_rate=progress["discovery_rate"],
        )
        assert 0.50 <= result.scale <= 1.50
        assert result.tier_hints  # should have tier hints

    def test_ops_init_exports(self):
        """Verify Phase E classes are accessible from core.ops."""
        from core.ops import __all__
        assert "EngagementMetrics" in __all__
        assert "TokenFlexEngine" in __all__
