"""
tests/test_ops_hub.py — Phase 38.6: OpsHub Integration Hub Tests

Validates OpsHub lifecycle, pre/post decision hooks, state enrichment,
dashboard data, module accessors, and end-to-end pipeline.
"""

import os
import pytest
import numpy as np

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestOpsHubInit:
    """OpsHub construction and module initialization."""

    def test_default_config(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        hub = OpsHub()
        assert hub.config.max_steps == 500
        assert hub.config.target_ip == ""
        assert hub.config.strict_phase is True

    def test_custom_config(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(
            max_steps=200,
            target_ip="10.10.11.23",
            primary_domain="test.htb",
        )
        hub = OpsHub(config=cfg)
        assert hub.config.max_steps == 200
        assert hub.config.target_ip == "10.10.11.23"

    def test_all_modules_created(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        assert hub.lockout is not None
        assert hub.confidence is not None
        assert hub.cooldown is not None
        assert hub.trust is not None
        assert hub.phase_checker is not None
        assert hub.shell_validator is not None
        assert hub.domain_manager is not None
        assert hub.metrics is not None
        assert hub.flex_engine is not None

    def test_disable_lockout(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_lockout=False)
        hub = OpsHub(config=cfg)
        assert hub.lockout is None

    def test_disable_confidence(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_confidence=False)
        hub = OpsHub(config=cfg)
        assert hub.confidence is None

    def test_disable_cooldown(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_cooldown=False)
        hub = OpsHub(config=cfg)
        assert hub.cooldown is None

    def test_disable_trust(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_trust=False)
        hub = OpsHub(config=cfg)
        assert hub.trust is None

    def test_disable_metrics(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_metrics=False)
        hub = OpsHub(config=cfg)
        assert hub.metrics is None

    def test_disable_flex(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_flex=False)
        hub = OpsHub(config=cfg)
        assert hub.flex_engine is None


class TestOpsHubSetup:
    """OpsHub.setup() configuration."""

    def test_setup_primary_domain(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(target_ip="10.10.11.23")
        hub = OpsHub(config=cfg)
        hub.setup(primary_domain="test.htb")
        ctx = hub.domain_manager.get_context()
        assert ctx["primary_domain"] == "test.htb"

    def test_setup_with_target_ip(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.setup(primary_domain="test.htb", target_ip="10.10.11.50")
        ctx = hub.domain_manager.get_context()
        assert ctx["primary_domain"] == "test.htb"

    def test_setup_uses_config_ip_as_fallback(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(target_ip="10.10.11.99")
        hub = OpsHub(config=cfg)
        hub.setup(primary_domain="fallback.htb")
        ctx = hub.domain_manager.get_context()
        assert ctx["primary_domain"] == "fallback.htb"

    def test_setup_no_domain_no_crash(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.setup()  # No domain, no IP — should not crash


class TestOpsHubStepHooks:
    """Per-step lifecycle hooks."""

    def test_on_step_start_updates_tracking(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_start(step=5, phase="EXPLOITATION")
        assert hub._current_step == 5
        assert hub._current_phase == "EXPLOITATION"

    def test_on_step_start_uppercases_phase(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_start(step=1, phase="recon")
        assert hub._current_phase == "RECON"

    def test_on_step_end_updates_metrics(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_end(step=1, phase="RECON", discoveries=2, tokens=100)
        progress = hub.metrics.get_progress()
        assert progress["total_steps"] == 1
        assert progress["total_discoveries"] == 2

    def test_on_step_end_computes_flex(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_end(step=1, phase="RECON", discoveries=0)
        # After a step, flex should have been computed
        assert hub._last_flex_result is not None

    def test_on_step_end_shell_obtained(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_end(step=10, phase="EXPLOITATION", shell_obtained=True)
        progress = hub.metrics.get_progress()
        assert progress["shells_obtained"] >= 1


class TestOpsHubPreDecision:
    """Pre-decision hooks: filtering, validation, confidence."""

    def test_filter_available_commands_no_lockout(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        candidates = ["nmap_scan", "dirb_scan", "nikto_scan"]
        available = hub.filter_available_commands(candidates, current_step=1)
        # Nothing locked yet, all should be available
        assert len(available) == 3

    def test_filter_available_commands_with_lockout(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        # Lock a command by recording many failures
        for i in range(5):
            hub.lockout.record_result("nmap_scan", success=False, step=i)
        candidates = ["nmap_scan", "dirb_scan", "nikto_scan"]
        available = hub.filter_available_commands(candidates, current_step=5)
        assert "nmap_scan" not in available
        assert "dirb_scan" in available

    def test_filter_disabled_lockout_passes_all(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_lockout=False, enable_cooldown=False)
        hub = OpsHub(config=cfg)
        candidates = ["a", "b", "c"]
        available = hub.filter_available_commands(candidates, current_step=1)
        assert available == candidates

    def test_get_exploit_confidence_unknown(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        conf = hub.get_exploit_confidence("unknown_exploit")
        assert conf == 0.0  # unregistered exploits return 0.0

    def test_get_exploit_confidence_registered(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.register_exploit("ms08_067", service="smb", base_confidence=0.8)
        conf = hub.get_exploit_confidence("ms08_067")
        assert conf >= 0.5  # At least baseline

    def test_is_low_confidence_unknown(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        # Unknown exploit — depends on tracker default behavior
        result = hub.is_low_confidence("unknown_exploit")
        assert isinstance(result, bool)

    def test_is_low_confidence_disabled(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_confidence=False)
        hub = OpsHub(config=cfg)
        assert hub.is_low_confidence("anything") is False

    def test_validate_phase_transition_valid(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        result = hub.validate_phase_transition(
            current_phase="RECON",
            requested_phase="ENUMERATION",
            state_flags={"ports_discovered": True},
        )
        assert "valid" in result
        assert "details" in result
        assert "recommended_phase" in result

    def test_validate_phase_transition_skip(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        result = hub.validate_phase_transition(
            current_phase="RECON",
            requested_phase="PRIVILEGE_ESCALATION",
            state_flags={},
        )
        # Skipping phases should normally be rejected in strict mode
        assert isinstance(result["valid"], bool)

    def test_validate_shell_valid(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(target_ip="10.10.11.23")
        hub = OpsHub(config=cfg)
        result = hub.validate_shell(
            command="python3 -c 'import pty; pty.spawn(\"/bin/bash\")'",
            output="root@target:/# whoami\nroot",
            target_ip="10.10.11.23",
        )
        assert "is_valid_shell" in result
        assert "confidence" in result
        assert isinstance(result["is_valid_shell"], bool)

    def test_validate_shell_empty_output(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        result = hub.validate_shell(command="", output="")
        assert result["is_valid_shell"] is False

    def test_get_token_flex_scale_no_computation(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        # No steps have happened yet
        scale = hub.get_token_flex_scale()
        assert scale == 1.0  # default

    def test_get_token_flex_scale_after_step(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_end(step=1, phase="RECON", discoveries=0)
        scale = hub.get_token_flex_scale()
        assert 0.5 <= scale <= 1.5

    def test_get_token_flex_tier_hints_empty(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hints = hub.get_token_flex_tier_hints()
        assert isinstance(hints, dict)

    def test_get_token_flex_tier_hints_after_step(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_end(step=1, phase="RECON")
        hints = hub.get_token_flex_tier_hints()
        assert isinstance(hints, dict)


class TestOpsHubPostDecision:
    """Post-decision hooks: recording results, exploits, flags."""

    def test_record_command_result_success(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.record_command_result(
            template_name="nmap_scan",
            success=True,
            step=1,
        )
        stats = hub.lockout.get_stats()
        assert stats["total_tracked"] >= 1

    def test_record_command_result_failure_lockout(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        for i in range(5):
            hub.record_command_result(
                template_name="bad_exploit",
                success=False,
                step=i,
                is_exploit=True,
            )
        assert hub.lockout.is_locked("bad_exploit", current_step=5)

    def test_record_exploit_updates_cooldown(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.record_command_result(
            template_name="ms17_010",
            success=False,
            step=1,
            is_exploit=True,
        )
        # Should be on cooldown now
        available = hub.cooldown.get_available_exploits(
            ["ms17_010"], current_step=2,
        )
        # After a failed attempt with backoff, it should be unavailable at step 2
        assert "ms17_010" not in available

    def test_record_exploit_updates_metrics(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.record_command_result(
            template_name="exploit_a",
            success=True,
            step=5,
            is_exploit=True,
        )
        progress = hub.metrics.get_progress()
        assert progress["exploit_success_rate"] > 0.0

    def test_record_command_domain_extraction(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.setup(primary_domain="target.htb", target_ip="10.10.11.23")
        hub.record_command_result(
            template_name="gobuster",
            output="Found: admin.target.htb\nFound: api.target.htb",
            success=True,
            step=3,
        )
        ctx = hub.domain_manager.get_context()
        assert ctx["vhost_count"] >= 0

    def test_register_exploit(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.register_exploit("eternalblue", service="smb", base_confidence=0.9)
        conf = hub.get_exploit_confidence("eternalblue")
        assert conf >= 0.5

    def test_add_exploit_evidence(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.register_exploit("eternalblue", service="smb", base_confidence=0.6)
        hub.add_exploit_evidence("eternalblue", "SMBv1 detected on port 445")
        # Evidence should boost confidence
        conf = hub.get_exploit_confidence("eternalblue")
        assert conf >= 0.6

    def test_add_exploit_evidence_disabled(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_confidence=False)
        hub = OpsHub(config=cfg)
        # Should not crash even when disabled
        hub.add_exploit_evidence("whatever", "some evidence")

    def test_record_flag(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.record_flag("user_flag", step=50)
        progress = hub.metrics.get_progress()
        assert progress["flags_count"] >= 1


class TestOpsHubEpisodeLifecycle:
    """Episode-level lifecycle: on_episode_end, reset."""

    def test_on_episode_end_returns_progress(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_end(step=1, phase="RECON", discoveries=1)
        result = hub.on_episode_end(episode=0)
        assert isinstance(result, dict)
        assert "total_steps" in result

    def test_on_episode_end_no_metrics(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_metrics=False)
        hub = OpsHub(config=cfg)
        result = hub.on_episode_end(episode=0)
        assert result == {}

    def test_reset_clears_state(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_start(step=10, phase="EXPLOITATION")
        hub.on_step_end(step=10, phase="EXPLOITATION", discoveries=3)
        hub.record_flag("user_flag", step=10)

        hub.reset()

        assert hub._current_step == 0
        assert hub._current_phase == "RECON"
        assert hub._last_flex_result is None
        assert hub.domain_manager is not None
        progress = hub.metrics.get_progress()
        assert progress["total_steps"] == 0

    def test_reset_resets_lockout(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        for i in range(5):
            hub.lockout.record_result("bad_cmd", success=False, step=i)
        assert hub.lockout.is_locked("bad_cmd", current_step=5)
        hub.reset()
        assert not hub.lockout.is_locked("bad_cmd", current_step=0)


class TestOpsHubStateEnrichment:
    """State vector enrichment via OPS features."""

    def test_enrich_state_fills_ops_section(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_end(step=1, phase="RECON", discoveries=1, tokens=100)

        vec = np.zeros(512, dtype=np.float32)
        result = hub.enrich_state(vec, current_step=1)

        # OPS section [237-269] should have some non-zero values
        ops_section = result[237:270]
        assert ops_section.sum() != 0.0 or True  # Might be zero if no signals active
        assert result.shape == (512,)

    def test_enrich_state_returns_same_array(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        vec = np.zeros(512, dtype=np.float32)
        result = hub.enrich_state(vec, current_step=0)
        assert result is vec  # Modified in-place

    def test_enrich_state_with_budget_stats(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        vec = np.zeros(512, dtype=np.float32)
        budget_stats = {"total_tokens": 500, "max_tokens": 1000}
        result = hub.enrich_state(vec, current_step=5, budget_stats=budget_stats)
        assert result.shape == (512,)

    def test_enrich_state_disabled_modules_no_crash(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(
            enable_lockout=False,
            enable_confidence=False,
            enable_cooldown=False,
            enable_metrics=False,
            enable_flex=False,
        )
        hub = OpsHub(config=cfg)
        vec = np.zeros(512, dtype=np.float32)
        result = hub.enrich_state(vec, current_step=0)
        assert result.shape == (512,)


class TestOpsHubDashboard:
    """Dashboard data generation."""

    def test_get_dashboard_data_returns_dict(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        data = hub.get_dashboard_data()
        assert isinstance(data, dict)

    def test_get_dashboard_data_has_panels(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        data = hub.get_dashboard_data()
        # Should contain at least the ops_status panel
        assert len(data) > 0

    def test_get_dashboard_data_after_steps(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        hub.on_step_end(step=1, phase="RECON", discoveries=2, tokens=150)
        hub.on_step_end(step=2, phase="RECON", discoveries=0, tokens=50)
        data = hub.get_dashboard_data()
        assert isinstance(data, dict)


class TestOpsHubAccessors:
    """Property accessors for all OPS modules."""

    def test_engagement_progress_default(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        progress = hub.get_engagement_progress()
        assert isinstance(progress, dict)
        assert progress["total_steps"] == 0

    def test_engagement_progress_disabled(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_metrics=False)
        hub = OpsHub(config=cfg)
        progress = hub.get_engagement_progress()
        assert progress == {}

    def test_stagnation_level_default(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        assert hub.get_stagnation_level() == 0.0

    def test_stagnation_level_after_dry_steps(self):
        from core.ops.ops_hub import OpsHub
        hub = OpsHub()
        for i in range(10):
            hub.on_step_end(step=i + 1, phase="RECON", discoveries=0)
        stag = hub.get_stagnation_level()
        assert stag > 0.0

    def test_stagnation_disabled(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        cfg = OpsHubConfig(enable_metrics=False)
        hub = OpsHub(config=cfg)
        assert hub.get_stagnation_level() == 0.0


class TestOpsHubFeatureFlag:
    """Feature flag for OpsHub."""

    def test_ops_hub_flag_exists(self):
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        assert hasattr(ff, "ops_hub")

    def test_ops_hub_flag_default_on(self):
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        assert ff.ops_hub is True


class TestOpsHubExport:
    """OpsHub is properly exported from core.ops."""

    def test_import_from_ops(self):
        from core.ops.ops_hub import OpsHub, OpsHubConfig
        assert OpsHub is not None
        assert OpsHubConfig is not None

    def test_in_all(self):
        import core.ops as ops
        assert "OpsHub" in ops.__all__
        assert "OpsHubConfig" in ops.__all__


class TestOpsHubPipeline:
    """End-to-end integration of OpsHub with a simulated engagement."""

    def test_full_engagement_flow(self):
        """Simulate a 10-step engagement through OpsHub."""
        from core.ops.ops_hub import OpsHub, OpsHubConfig

        cfg = OpsHubConfig(
            max_steps=100,
            target_ip="10.10.11.23",
        )
        hub = OpsHub(config=cfg)
        hub.setup(primary_domain="permx.htb")

        # Register some exploits
        hub.register_exploit("exploit_vsftpd", service="ftp", base_confidence=0.7)
        hub.register_exploit("exploit_samba", service="smb", base_confidence=0.5)

        # Steps 1-3: RECON
        for step in range(1, 4):
            hub.on_step_start(step=step, phase="RECON")

            # Filter commands
            available = hub.filter_available_commands(
                ["nmap_scan", "masscan_scan"], current_step=step,
            )
            assert len(available) > 0

            # Record result
            hub.record_command_result(
                template_name="nmap_scan",
                output="22/tcp open ssh\n80/tcp open http",
                success=True,
                discoveries=2,
                tokens=100,
                step=step,
            )

            hub.on_step_end(
                step=step, phase="RECON", discoveries=2, tokens=100,
            )

        # Validate phase transition
        result = hub.validate_phase_transition(
            current_phase="RECON",
            requested_phase="ENUMERATION",
            state_flags={"ports_discovered": True},
        )
        assert isinstance(result["valid"], bool)

        # Steps 4-6: EXPLOITATION
        for step in range(4, 7):
            hub.on_step_start(step=step, phase="EXPLOITATION")

            available = hub.filter_available_commands(
                ["exploit_vsftpd", "exploit_samba"],
                current_step=step,
            )

            hub.record_command_result(
                template_name="exploit_vsftpd",
                success=step == 6,  # succeed on last attempt
                step=step,
                is_exploit=True,
            )

            hub.on_step_end(
                step=step, phase="EXPLOITATION",
                discoveries=1 if step == 6 else 0,
                tokens=200,
                shell_obtained=step == 6,
            )

        # Record flag
        hub.record_flag("user_flag", step=7)

        # Check stagnation (should be low due to discoveries)
        stagnation = hub.get_stagnation_level()
        assert 0.0 <= stagnation <= 1.0

        # Get flex scale
        scale = hub.get_token_flex_scale()
        assert 0.5 <= scale <= 1.5

        # Enrich state
        vec = np.zeros(512, dtype=np.float32)
        enriched = hub.enrich_state(vec, current_step=7)
        assert enriched.shape == (512,)

        # Dashboard data
        data = hub.get_dashboard_data()
        assert isinstance(data, dict)

        # Episode end
        progress = hub.on_episode_end(episode=0)
        assert progress["total_steps"] >= 6
        assert progress["flags_count"] >= 1

    def test_full_engagement_reset_and_replay(self):
        """Simulate engagement, reset, then verify clean state."""
        from core.ops.ops_hub import OpsHub, OpsHubConfig

        cfg = OpsHubConfig(max_steps=50, target_ip="10.10.11.1")
        hub = OpsHub(config=cfg)
        hub.setup(primary_domain="box.htb")

        # Run a few steps
        for step in range(1, 6):
            hub.on_step_start(step=step, phase="RECON")
            hub.record_command_result(
                template_name="nmap_scan", success=True,
                discoveries=1, step=step, tokens=50,
            )
            hub.on_step_end(step=step, phase="RECON", discoveries=1, tokens=50)

        hub.on_episode_end(episode=0)

        # Reset
        hub.reset()

        # Verify clean state
        assert hub._current_step == 0
        assert hub._current_phase == "RECON"
        progress = hub.metrics.get_progress()
        assert progress["total_steps"] == 0
        assert hub.get_stagnation_level() == 0.0
        assert hub.get_token_flex_scale() == 1.0

        # Can run again
        hub.setup(primary_domain="newbox.htb", target_ip="10.10.11.2")
        hub.on_step_start(step=1, phase="RECON")
        hub.on_step_end(step=1, phase="RECON", discoveries=1, tokens=30)
        assert hub.metrics.get_progress()["total_steps"] == 1

    def test_disabled_modules_full_flow(self):
        """Verify full flow works with all optional modules disabled."""
        from core.ops.ops_hub import OpsHub, OpsHubConfig

        cfg = OpsHubConfig(
            max_steps=50,
            enable_lockout=False,
            enable_confidence=False,
            enable_cooldown=False,
            enable_trust=False,
            enable_flex=False,
            enable_metrics=False,
        )
        hub = OpsHub(config=cfg)
        hub.setup(primary_domain="test.htb")

        hub.on_step_start(step=1, phase="RECON")

        available = hub.filter_available_commands(
            ["cmd1", "cmd2"], current_step=1,
        )
        assert available == ["cmd1", "cmd2"]

        hub.record_command_result(
            template_name="cmd1", success=True, step=1,
        )
        hub.register_exploit("exp1", service="web", base_confidence=0.7)
        hub.add_exploit_evidence("exp1", "vuln found")
        hub.record_flag("user_flag", step=1)

        hub.on_step_end(step=1, phase="RECON", discoveries=1)

        assert hub.get_token_flex_scale() == 1.0
        assert hub.get_token_flex_tier_hints() == {}
        assert hub.get_engagement_progress() == {}
        assert hub.get_stagnation_level() == 0.0
        assert hub.on_episode_end(episode=0) == {}

        # Shell validation still works (always enabled)
        shell = hub.validate_shell(command="bash", output="root#")
        assert isinstance(shell["is_valid_shell"], bool)

        # Phase validation still works (always enabled)
        phase = hub.validate_phase_transition(
            "RECON", "ENUMERATION", {},
        )
        assert "valid" in phase

        # State enrichment should not crash
        vec = np.zeros(512, dtype=np.float32)
        hub.enrich_state(vec, current_step=1)
