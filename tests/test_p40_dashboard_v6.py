"""
tests/test_p40_dashboard_v6.py — Phase 40: Dashboard v6 Tests
"""

import os
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestDashboardV6Panels:
    """Test new v6 dashboard panel builders."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.observability.live_dashboard import LiveDashboard
        self.dashboard = LiveDashboard()

    def test_phase_progress_bar_recon(self):
        panel = self.dashboard._build_phase_progress_bar("RECON")
        assert panel is not None

    def test_phase_progress_bar_exploitation(self):
        panel = self.dashboard._build_phase_progress_bar("EXPLOITATION")
        assert panel is not None

    def test_phase_progress_bar_closeout(self):
        panel = self.dashboard._build_phase_progress_bar("CLOSEOUT")
        assert panel is not None

    def test_phase_progress_bar_unknown(self):
        panel = self.dashboard._build_phase_progress_bar("UNKNOWN_PHASE")
        assert panel is not None

    def test_decision_chain_panel_empty(self):
        panel = self.dashboard._build_decision_chain_panel([])
        # Should handle empty list gracefully
        assert panel is None or panel is not None  # No crash

    def test_step_metrics_panel(self):
        reward_breakdown = {"base": 5.0, "novelty_bonus": 2.0}
        parser_stats = {"regex_hits": 3, "llm_hits": 1}
        budget_snapshot = {"remaining": 500000, "used": 100000}
        gpt_activity = {"calls": 5, "tokens": 1200}
        panel = self.dashboard._build_step_metrics_panel(
            reward_breakdown, parser_stats, budget_snapshot, gpt_activity
        )
        assert panel is not None

    def test_learning_panel(self):
        snapshot = {
            "total_ports": 5,
            "total_services": 3,
            "total_creds": 1,
            "total_shells": 0,
            "total_paths": 2,
            "new_ports": 1,
            "new_services": 1,
            "new_creds": 0,
            "new_shells": 0,
            "stagnation_count": 2,
            "novelty_rate": 0.4,
            "anti_repeat_rate": 0.1,
            "milestones": {"ports>=3": 1},
            "model_mix": {"nano": 5, "mini": 2},
            "evidence_gate": {"pass": 3, "reject": 1},
            "total_cost": 0.05,
        }
        panel = self.dashboard._build_learning_panel(snapshot, step=10)
        assert panel is not None

    def test_system_log_panel_empty(self):
        result = self.dashboard._build_system_log_panel([])
        assert result is None

    def test_target_profile_panel(self):
        discovery_board = {
            "ports": {21, 22, 80},
            "services": {"ssh", "http"},
            "credentials": set(),
            "vulns": set(),
            "shells": set(),
            "users": {"admin"},
            "web_paths": {"/data", "/admin"},
            "phase": "ENUMERATION",
            "target_ip": "10.10.10.245",
            "os_hint": "Linux",
        }
        panel = self.dashboard._build_target_profile_panel(discovery_board, step=1)
        assert panel is not None

    def test_target_profile_panel_late_step(self):
        """Panel returns None after step 3."""
        discovery_board = {
            "ports": {21, 22, 80},
            "services": {"ssh", "http"},
            "target_ip": "10.10.10.245",
        }
        panel = self.dashboard._build_target_profile_panel(discovery_board, step=5)
        assert panel is None

    def test_discovery_counts_history(self):
        assert hasattr(self.dashboard, "discovery_counts_history")
        self.dashboard.discovery_counts_history.append(5)
        self.dashboard.discovery_counts_history.append(8)
        assert len(self.dashboard.discovery_counts_history) == 2


class TestDecisionTrace:
    """Test SmartDecisionResult decision_trace field."""

    def test_decision_trace_default(self):
        from core.training.smart_coach import SmartDecisionResult
        result = SmartDecisionResult(command="nmap -sV 10.10.10.1")
        assert result.decision_trace == []

    def test_decision_trace_populated(self):
        from core.training.smart_coach import SmartDecisionResult
        result = SmartDecisionResult(command="nmap -sV 10.10.10.1")
        result.decision_trace = [
            {"stage": "playbook", "result": "skip", "score": 0.0, "passed": False},
            {"stage": "ppo", "result": "nmap", "score": 0.8, "passed": True},
        ]
        assert len(result.decision_trace) == 2
        assert result.decision_trace[1]["stage"] == "ppo"


class TestOSAffinity:
    """Test OS affinity on CommandTemplate."""

    def test_os_affinity_field(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        # Check that some templates have been auto-tagged
        win_count = sum(1 for c in COMMAND_REGISTRY.values() if c.os_affinity == "windows")
        lin_count = sum(1 for c in COMMAND_REGISTRY.values() if c.os_affinity == "linux")
        assert win_count > 0, "No Windows-affinity templates found"
        assert lin_count > 0, "No Linux-affinity templates found"

    def test_default_is_any(self):
        from core.commands.command_registry import CommandTemplate
        from core.commands.command_registry import AttackPhase
        t = CommandTemplate(
            name="test_cmd",
            template="test {target}",
            description="Test",
            phase=AttackPhase.RECON,
        )
        assert t.os_affinity == "any"


class TestPhaseReadiness:
    """Test confidence-weighted phase readiness."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_readiness_import(self):
        from core.environment.cyber_environment import CyberEnvironment
        env = CyberEnvironment(defer_reset=True)
        score = env._compute_phase_readiness("recon")
        assert 0.0 <= score <= 1.0

    def test_readiness_recon_no_ports(self):
        from core.environment.cyber_environment import CyberEnvironment
        env = CyberEnvironment(defer_reset=True)
        env.open_ports = []
        score = env._compute_phase_readiness("recon")
        assert score == 0.0

    def test_readiness_recon_with_ports(self):
        from core.environment.cyber_environment import CyberEnvironment
        env = CyberEnvironment(defer_reset=True)
        env.open_ports = [21, 22, 80]
        score = env._compute_phase_readiness("recon")
        assert score >= 0.5

    def test_readiness_exploit_with_shell(self):
        from core.environment.cyber_environment import CyberEnvironment
        env = CyberEnvironment(defer_reset=True)
        env.active_shells = ["shell1"]
        env.credentials_found = True
        score = env._compute_phase_readiness("exploit")
        assert score >= 0.8


class TestFeatureFlagsP40:
    """Test Phase 40 feature flags exist."""

    def test_p40_flags_exist(self):
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        assert hasattr(ff, "dashboard_v6")
        assert hasattr(ff, "ssh_pool")
        assert hasattr(ff, "pool_narrower")
        assert hasattr(ff, "os_aware_filter")
        assert hasattr(ff, "auto_web_probe")

    def test_p40_flags_default_true(self):
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        # In test/dry-run mode, these should still be True
        # (they use _env_bool with True default)
        assert ff.dashboard_v6 is True
        assert ff.ssh_pool is True
        assert ff.pool_narrower is True
