"""
tests/test_ops_state_dashboard.py — Phase F: StateEncoder + Dashboard

Tests for:
  - OPS State Encoder: inject_ops_features, collect_ops_signals
  - OPS Dashboard Panels: ops_status, engagement, token_flex, domain_intel
  - Feature flags for Phase 38.5
  - Integration: live modules → collect_ops_signals → inject_ops_features
"""

import os
import numpy as np
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ═══════════════════════════════════════════════════════════════════════════
# OPS State Encoder — inject_ops_features
# ═══════════════════════════════════════════════════════════════════════════


class TestInjectOpsFeatures:
    """Test direct injection of OPS features into state vector."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.ops_state_encoder import (
            inject_ops_features,
            OPS_SECTION_START,
            OPS_SECTION_END,
        )
        self.inject = inject_ops_features
        self.start = OPS_SECTION_START
        self.end = OPS_SECTION_END

    def _make_vec(self) -> np.ndarray:
        return np.zeros(512, dtype=np.float32)

    def test_default_injection_zeros(self):
        vec = self._make_vec()
        result = self.inject(vec)
        # Default injection should produce mostly zeros (except reserve dims)
        assert result is vec  # in-place modification
        # Confidence defaults are 0.5 (not zero)
        assert vec[240] == pytest.approx(0.5)  # confidence_mean
        assert vec[241] == pytest.approx(0.5)  # confidence_min
        assert vec[242] == pytest.approx(0.5)  # confidence_max

    def test_lockout_features(self):
        vec = self._make_vec()
        self.inject(vec, lockout_pressure=0.6, lockout_count=5)
        assert vec[237] == pytest.approx(0.6)
        assert vec[238] == pytest.approx(5 / 20)  # default max=20
        assert vec[239] == 1.0  # any lockout active

    def test_lockout_zero(self):
        vec = self._make_vec()
        self.inject(vec, lockout_pressure=0.0, lockout_count=0)
        assert vec[239] == 0.0  # no lockout

    def test_confidence_features(self):
        vec = self._make_vec()
        self.inject(
            vec,
            confidence_mean=0.7,
            confidence_min=0.3,
            confidence_max=0.95,
            low_confidence_ratio=0.2,
        )
        assert vec[240] == pytest.approx(0.7)
        assert vec[241] == pytest.approx(0.3)
        assert vec[242] == pytest.approx(0.95)
        assert vec[243] == pytest.approx(0.2)

    def test_cooldown_features(self):
        vec = self._make_vec()
        self.inject(vec, cooldown_pressure=0.4, cooldown_active_count=3)
        assert vec[244] == pytest.approx(0.4)
        assert vec[245] == pytest.approx(3 / 20)
        assert vec[246] == 1.0

    def test_engagement_features(self):
        vec = self._make_vec()
        self.inject(
            vec,
            stagnation_level=0.5,
            phase_velocity=3.0,
            token_efficiency=2.0,
            engagement_progress=0.4,
            discovery_rate=0.3,
            exploit_success_rate=0.6,
        )
        assert vec[247] == pytest.approx(0.5)   # stagnation
        assert vec[248] == pytest.approx(0.3)    # velocity / 10
        assert vec[249] == pytest.approx(0.4)    # efficiency / 5
        assert vec[250] == pytest.approx(0.4)    # progress
        assert vec[251] == pytest.approx(0.3)    # discovery_rate
        assert vec[252] == pytest.approx(0.6)    # exploit_sr

    def test_token_flex_features(self):
        vec = self._make_vec()
        self.inject(
            vec,
            token_flex_scale=1.3,
            tokens_used=500_000,
            codex_tier_pressure=0.8,
            mini_tier_pressure=0.5,
        )
        # scale normalised: (1.3 - 0.5) / 1.0 = 0.8
        assert vec[253] == pytest.approx(0.8)
        assert vec[254] == pytest.approx(500_000 / 1_148_850, abs=0.001)
        assert vec[255] == pytest.approx(0.8)
        assert vec[256] == pytest.approx(0.5)

    def test_shell_flag_features(self):
        vec = self._make_vec()
        self.inject(vec, shells_obtained=2, flags_captured=1)
        assert vec[257] == pytest.approx(2 / 3)
        assert vec[258] == 1.0   # shell binary
        assert vec[259] == pytest.approx(0.5)  # 1/2 flags
        assert vec[260] == 0.0   # not full capture

    def test_full_flag_capture(self):
        vec = self._make_vec()
        self.inject(vec, flags_captured=2)
        assert vec[260] == 1.0  # full capture

    def test_domain_features(self):
        vec = self._make_vec()
        self.inject(vec, domain_count=5, vhost_count=3)
        assert vec[261] == pytest.approx(0.5)
        assert vec[262] == pytest.approx(0.3)
        assert vec[263] == 1.0   # domain discovered

    def test_phase_dynamics(self):
        vec = self._make_vec()
        self.inject(
            vec,
            phase_transitions_total=4,
            engagement_progress=0.6,
            stagnation_level=0.5,
            cooldown_pressure=0.3,
            lockout_pressure=0.2,
        )
        assert vec[264] == pytest.approx(0.4)    # 4/10
        assert vec[265] == pytest.approx(0.36)   # 0.6^2
        # compound: 0.5*0.5 + 0.3*0.3 + 0.2*0.2 = 0.25+0.09+0.04 = 0.38
        assert vec[266] == pytest.approx(0.38)

    def test_clamping(self):
        vec = self._make_vec()
        self.inject(vec, lockout_pressure=5.0, stagnation_level=3.0)
        assert vec[237] == 1.0  # clamped
        assert vec[247] == 1.0  # clamped

    def test_short_vector_skipped(self):
        short_vec = np.zeros(100, dtype=np.float32)
        result = self.inject(short_vec)
        # Should not crash, returns unchanged vector
        assert result is short_vec

    def test_non_destructive_to_existing_dims(self):
        vec = self._make_vec()
        vec[0] = 0.99  # Phase one-hot
        vec[100] = 0.77  # Temporal
        self.inject(vec, lockout_pressure=0.5)
        assert vec[0] == pytest.approx(0.99)    # untouched
        assert vec[100] == pytest.approx(0.77)  # untouched


# ═══════════════════════════════════════════════════════════════════════════
# OPS State Encoder — collect_ops_signals
# ═══════════════════════════════════════════════════════════════════════════


class TestCollectOpsSignals:
    """Test signal collection from live module instances."""

    def test_all_none_defaults(self):
        from core.ops.ops_state_encoder import collect_ops_signals
        signals = collect_ops_signals()
        assert signals["lockout_pressure"] == 0.0
        assert signals["confidence_mean"] == 0.5
        assert signals["cooldown_pressure"] == 0.0
        assert signals["stagnation_level"] == 0.0
        assert signals["token_flex_scale"] == 1.0
        assert signals["domain_count"] == 0

    def test_with_lockout(self):
        from core.ops.ops_state_encoder import collect_ops_signals
        from core.ops.command_lockout import CommandLockout
        lockout = CommandLockout()
        # Record 3 failures (hits threshold)
        for _ in range(3):
            lockout.record_result("nmap_scan", success=False, step=1)
        signals = collect_ops_signals(lockout=lockout)
        assert signals["lockout_count"] == 1
        assert signals["lockout_pressure"] > 0

    def test_with_confidence(self):
        from core.ops.ops_state_encoder import collect_ops_signals
        from core.ops.exploit_confidence import ExploitConfidenceTracker
        conf = ExploitConfidenceTracker()
        conf.register_exploit("ms08_067", base_confidence=0.8)
        conf.register_exploit("eternalblue", base_confidence=0.4)
        signals = collect_ops_signals(confidence=conf)
        assert signals["confidence_mean"] == pytest.approx(0.6)
        assert signals["confidence_min"] == pytest.approx(0.4)
        assert signals["confidence_max"] == pytest.approx(0.8)

    def test_with_metrics(self):
        from core.ops.ops_state_encoder import collect_ops_signals
        from core.ops.engagement_metrics import EngagementMetrics
        metrics = EngagementMetrics()
        metrics.record_step(step=1, phase="ENUMERATION", discoveries=3)
        signals = collect_ops_signals(metrics=metrics)
        assert signals["discovery_rate"] == pytest.approx(3.0)
        assert signals["stagnation_level"] == 0.0

    def test_with_flex_result(self):
        from core.ops.ops_state_encoder import collect_ops_signals
        from core.ops.token_flex import TokenFlexEngine
        engine = TokenFlexEngine(max_steps=100)
        result = engine.compute(phase="EXPLOITATION")
        signals = collect_ops_signals(flex_result=result)
        assert signals["token_flex_scale"] == pytest.approx(1.15)

    def test_with_domain_manager(self):
        from core.ops.ops_state_encoder import collect_ops_signals
        from core.ops.domain_manager import DomainManager
        dm = DomainManager()
        dm.set_primary("permx.htb", ip="10.10.11.23")
        signals = collect_ops_signals(domain_manager=dm)
        assert signals["domain_count"] >= 1
        assert signals["vhost_count"] >= 0  # set_primary may auto-expand subdomains

    def test_signals_usable_by_inject(self):
        """Collected signals can be passed directly to inject_ops_features."""
        from core.ops.ops_state_encoder import collect_ops_signals, inject_ops_features
        signals = collect_ops_signals()
        vec = np.zeros(512, dtype=np.float32)
        # Should not crash — all keys valid
        inject_ops_features(vec, **signals)
        assert vec.shape == (512,)


# ═══════════════════════════════════════════════════════════════════════════
# OPS Dashboard Panels
# ═══════════════════════════════════════════════════════════════════════════


class TestOpsDashboardPanels:
    """Test OPS dashboard panel data generators."""

    def test_ops_status_empty(self):
        from core.ops.ops_dashboard_panels import ops_status_panel
        data = ops_status_panel()
        assert data["title"] == "OPS Status"
        assert data["lockout"]["locked_count"] == 0
        assert data["confidence"]["tracked"] == 0
        assert data["cooldown"]["active"] == 0

    def test_ops_status_with_lockout(self):
        from core.ops.ops_dashboard_panels import ops_status_panel
        from core.ops.command_lockout import CommandLockout
        lockout = CommandLockout()
        for _ in range(3):
            lockout.record_result("nmap_scan", success=False, step=1)
        data = ops_status_panel(lockout=lockout)
        assert data["lockout"]["locked_count"] == 1

    def test_engagement_panel_empty(self):
        from core.ops.ops_dashboard_panels import engagement_panel
        data = engagement_panel()
        assert data["title"] == "Engagement Metrics"
        assert data["total_steps"] == 0

    def test_engagement_panel_with_data(self):
        from core.ops.ops_dashboard_panels import engagement_panel
        from core.ops.engagement_metrics import EngagementMetrics
        m = EngagementMetrics()
        m.record_step(step=1, phase="EXPLOITATION", discoveries=5)
        data = engagement_panel(metrics=m)
        assert data["total_steps"] == 1
        assert data["current_phase"] == "EXPLOITATION"
        assert data["total_discoveries"] == 5

    def test_token_flex_panel_empty(self):
        from core.ops.ops_dashboard_panels import token_flex_panel
        data = token_flex_panel()
        assert data["scale"] == 1.0
        assert data["reason"] == "default"

    def test_token_flex_panel_with_result(self):
        from core.ops.ops_dashboard_panels import token_flex_panel
        from core.ops.token_flex import TokenFlexEngine
        engine = TokenFlexEngine(max_steps=100)
        result = engine.compute(phase="EXPLOITATION", stagnation_level=0.5)
        data = token_flex_panel(flex_result=result)
        assert data["scale"] > 1.0
        assert "EXPLOITATION" in data["reason"]

    def test_domain_intel_panel_empty(self):
        from core.ops.ops_dashboard_panels import domain_intel_panel
        data = domain_intel_panel()
        assert data["primary"] is None
        assert data["confirmed_count"] == 0

    def test_domain_intel_panel_with_data(self):
        from core.ops.ops_dashboard_panels import domain_intel_panel
        from core.ops.domain_manager import DomainManager
        dm = DomainManager()
        dm.set_primary("permx.htb", ip="10.10.11.23")
        data = domain_intel_panel(domain_manager=dm)
        assert data["primary"] == "permx.htb"
        assert data["confirmed_count"] >= 1
        assert len(data["hosts_entries"]) >= 1

    def test_all_ops_panels(self):
        from core.ops.ops_dashboard_panels import all_ops_panels
        panels = all_ops_panels()
        assert "ops_status" in panels
        assert "engagement" in panels
        assert "token_flex" in panels
        assert "domain_intel" in panels
        assert panels["ops_status"]["title"] == "OPS Status"
        assert panels["engagement"]["title"] == "Engagement Metrics"

    def test_all_ops_panels_with_modules(self):
        from core.ops.ops_dashboard_panels import all_ops_panels
        from core.ops.engagement_metrics import EngagementMetrics
        from core.ops.token_flex import TokenFlexEngine

        m = EngagementMetrics()
        m.record_step(step=1, discoveries=3)
        engine = TokenFlexEngine(max_steps=100)
        flex = engine.compute(phase="RECON")

        panels = all_ops_panels(metrics=m, flex_result=flex)
        assert panels["engagement"]["total_discoveries"] == 3
        assert panels["token_flex"]["scale"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Flags
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseFFlags:
    """Phase 38.5 feature flag verification."""

    def test_ops_state_encoder_flag(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "ops_state_encoder")
        assert ff.ops_state_encoder is True

    def test_ops_dashboard_flag(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "ops_dashboard")
        assert ff.ops_dashboard is True


# ═══════════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseFIntegration:
    """Integration: live modules → collect → inject → verify."""

    def test_full_pipeline(self):
        from core.ops.command_lockout import CommandLockout
        from core.ops.engagement_metrics import EngagementMetrics
        from core.ops.exploit_confidence import ExploitConfidenceTracker
        from core.ops.exploit_cooldown import ExploitCooldownManager
        from core.ops.domain_manager import DomainManager
        from core.ops.token_flex import TokenFlexEngine
        from core.ops.ops_state_encoder import collect_ops_signals, inject_ops_features

        # Create all modules
        lockout = CommandLockout()
        conf = ExploitConfidenceTracker()
        cooldown = ExploitCooldownManager()
        metrics = EngagementMetrics()
        dm = DomainManager()
        engine = TokenFlexEngine(max_steps=200)

        # Simulate usage
        lockout.record_result("dirb_scan", success=False, step=1)
        lockout.record_result("dirb_scan", success=False, step=2)
        lockout.record_result("dirb_scan", success=False, step=3)

        conf.register_exploit("ms08_067", base_confidence=0.7)
        conf.record_attempt("ms08_067", success=False)

        cooldown.record_attempt("ms08_067", step=5, success=False)

        metrics.record_step(step=1, phase="RECON", discoveries=3, tokens=200)
        metrics.record_step(step=2, phase="ENUMERATION", discoveries=1, tokens=150)

        dm.set_primary("permx.htb", ip="10.10.11.23")

        flex = engine.compute(
            phase="ENUMERATION",
            stagnation_level=metrics.get_stagnation_level(),
        )

        # Collect and inject
        signals = collect_ops_signals(
            lockout=lockout,
            confidence=conf,
            cooldown=cooldown,
            metrics=metrics,
            flex_result=flex,
            domain_manager=dm,
            current_step=5,
        )
        vec = np.zeros(512, dtype=np.float32)
        inject_ops_features(vec, **signals)

        # Verify OPS section has non-zero values
        ops_section = vec[237:270]
        assert np.any(ops_section != 0), "OPS section should have non-zero values"

        # Verify lockout injected
        assert vec[237] > 0  # lockout pressure > 0

    def test_ops_init_exports(self):
        """Verify Phase F classes are accessible from core.ops."""
        from core.ops import __all__
        assert "inject_ops_features" in __all__
        assert "collect_ops_signals" in __all__
        assert "all_ops_panels" in __all__
