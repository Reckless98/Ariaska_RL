#!/usr/bin/env python3
"""
tests/test_phase101_privilege.py — Phase 10.1A: Privilege Gating Tests

Tests for privilege-aware command template filtering,
sudo gating, and privilege telemetry counters.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestPrivilegeLevel:
    """Test PrivilegeLevel enum and basic filtering."""

    def test_privilege_level_values(self):
        from core.commands.privilege import PrivilegeLevel
        assert PrivilegeLevel.NONE.value == "none"
        assert PrivilegeLevel.SUDO.value == "sudo"
        assert PrivilegeLevel.ROOT.value == "root"

    def test_filter_none_privilege_passes(self):
        """Commands with NONE privilege should always pass."""
        from core.commands.privilege import filter_by_privilege, PrivilegeTelemetry
        from core.commands.command_registry import CommandTemplate, AttackPhase

        cmd = CommandTemplate(
            name="nmap_scan", template="nmap {target}",
            description="Scan", phase=AttackPhase.RECON,
            requires_privilege="none",
        )
        state = {"phase": "RECON", "state_flags": {}}
        tel = PrivilegeTelemetry()
        result = filter_by_privilege([cmd], state, telemetry=tel)

        assert len(result.allowed) == 1
        assert len(result.filtered) == 0
        assert tel.candidates_filtered_no_privilege == 0

    def test_filter_sudo_without_rights(self):
        """Sudo commands should be filtered when no sudo rights."""
        from core.commands.privilege import filter_by_privilege, PrivilegeTelemetry
        from core.commands.command_registry import CommandTemplate, AttackPhase

        cmd = CommandTemplate(
            name="read_shadow", template="sudo cat /etc/shadow",
            description="Read shadow file", phase=AttackPhase.PRIVILEGE_ESCALATION,
            requires_privilege="sudo", privilege_reason="Needs sudo to read shadow",
        )
        state = {
            "phase": "PRIVILEGE_ESCALATION",
            "state_flags": {"shell_obtained": True},
            "privilege_level": "user",
        }
        tel = PrivilegeTelemetry()
        result = filter_by_privilege([cmd], state, telemetry=tel)

        assert len(result.allowed) == 0
        assert len(result.filtered) == 1
        assert tel.candidates_filtered_no_sudo == 1
        assert "read_shadow" in result.filter_reasons

    def test_filter_sudo_with_rights(self):
        """Sudo commands should pass when sudo rights are discovered."""
        from core.commands.privilege import filter_by_privilege, PrivilegeTelemetry
        from core.commands.command_registry import CommandTemplate, AttackPhase

        cmd = CommandTemplate(
            name="read_shadow", template="sudo cat /etc/shadow",
            description="Read shadow", phase=AttackPhase.PRIVILEGE_ESCALATION,
            requires_privilege="sudo",
        )
        state = {
            "phase": "PRIVILEGE_ESCALATION",
            "state_flags": {"shell_obtained": True, "sudo_rights_discovered": True},
            "privilege_level": "user",
        }
        tel = PrivilegeTelemetry()
        result = filter_by_privilege([cmd], state, telemetry=tel)

        assert len(result.allowed) == 1
        assert len(result.filtered) == 0

    def test_filter_root_without_root(self):
        """Root commands filtered when only user shell."""
        from core.commands.privilege import filter_by_privilege, PrivilegeTelemetry
        from core.commands.command_registry import CommandTemplate, AttackPhase

        cmd = CommandTemplate(
            name="install_backdoor", template="cp backdoor /root/.ssh/",
            description="Install backdoor", phase=AttackPhase.POST_EXPLOITATION,
            requires_privilege="root",
        )
        state = {
            "phase": "POST_EXPLOITATION",
            "state_flags": {"shell_obtained": True},
            "privilege_level": "user",
        }
        tel = PrivilegeTelemetry()
        result = filter_by_privilege([cmd], state, telemetry=tel)

        assert len(result.filtered) == 1
        assert tel.candidates_filtered_no_privilege == 1

    def test_filter_root_with_root(self):
        """Root commands pass when root shell."""
        from core.commands.privilege import filter_by_privilege, PrivilegeTelemetry
        from core.commands.command_registry import CommandTemplate, AttackPhase

        cmd = CommandTemplate(
            name="install_backdoor", template="cp backdoor /root/.ssh/",
            description="Install backdoor", phase=AttackPhase.POST_EXPLOITATION,
            requires_privilege="root",
        )
        state = {
            "phase": "POST_EXPLOITATION",
            "state_flags": {"root_shell_obtained": True},
            "privilege_level": "root",
        }
        result = filter_by_privilege([cmd], state)
        assert len(result.allowed) == 1

    def test_phase_gate_blocks_early_phase(self):
        """Privilege commands blocked in RECON phase without shell."""
        from core.commands.privilege import filter_by_privilege, PrivilegeTelemetry
        from core.commands.command_registry import CommandTemplate, AttackPhase

        cmd = CommandTemplate(
            name="sudo_ls", template="sudo ls /root",
            description="List root dir", phase=AttackPhase.RECON,
            requires_privilege="sudo",
        )
        state = {"phase": "RECON", "state_flags": {}, "privilege_level": "none"}
        tel = PrivilegeTelemetry()
        result = filter_by_privilege([cmd], state, telemetry=tel)

        assert len(result.filtered) == 1
        assert tel.candidates_filtered_phase_gate == 1

    def test_mixed_privilege_filtering(self):
        """Mix of NONE and SUDO commands — only SUDO filtered."""
        from core.commands.privilege import filter_by_privilege
        from core.commands.command_registry import CommandTemplate, AttackPhase

        cmds = [
            CommandTemplate(name="nmap", template="nmap {target}",
                          description="Scan", phase=AttackPhase.RECON,
                          requires_privilege="none"),
            CommandTemplate(name="sudo_cat", template="sudo cat /etc/shadow",
                          description="Shadow", phase=AttackPhase.PRIVILEGE_ESCALATION,
                          requires_privilege="sudo"),
            CommandTemplate(name="dirb", template="dirb {url}",
                          description="Dir bust", phase=AttackPhase.ENUMERATION,
                          requires_privilege="none"),
        ]
        state = {"phase": "RECON", "state_flags": {}, "privilege_level": "none"}
        result = filter_by_privilege(cmds, state)

        assert len(result.allowed) == 2
        assert all(c.name in ("nmap", "dirb") for c in result.allowed)
        assert len(result.filtered) == 1


class TestSudoGating:
    """Test check_sudo_allowed logic."""

    def test_sudo_denied_when_flag_off(self):
        from core.commands.privilege import check_sudo_allowed
        state = {"state_flags": {"sudo_rights_discovered": True}}
        assert check_sudo_allowed(state, ff_allow_sudo=False) is False

    def test_sudo_denied_when_no_rights(self):
        from core.commands.privilege import check_sudo_allowed
        state = {"state_flags": {}}
        assert check_sudo_allowed(state, ff_allow_sudo=True) is False

    def test_sudo_allowed_with_flag_and_rights(self):
        from core.commands.privilege import check_sudo_allowed
        state = {"state_flags": {"sudo_rights_discovered": True}}
        assert check_sudo_allowed(state, ff_allow_sudo=True) is True

    def test_sudo_allowed_with_root_shell(self):
        from core.commands.privilege import check_sudo_allowed
        state = {"state_flags": {"root_shell_obtained": True}}
        assert check_sudo_allowed(state, ff_allow_sudo=True) is True


class TestPrivilegeTelemetry:
    """Test telemetry counter integrity."""

    def test_telemetry_to_dict(self):
        from core.commands.privilege import PrivilegeTelemetry
        tel = PrivilegeTelemetry(
            candidates_filtered_no_privilege=3,
            sudo_attempted=2,
            sudo_allowed=1,
            sudo_denied=1,
        )
        d = tel.to_dict()
        assert d["filtered_no_privilege"] == 3
        assert d["sudo_attempted"] == 2
        assert d["sudo_allowed"] == 1
        assert d["sudo_denied"] == 1

    def test_telemetry_merge(self):
        from core.commands.privilege import PrivilegeTelemetry
        a = PrivilegeTelemetry(candidates_filtered_no_privilege=2, sudo_attempted=1)
        b = PrivilegeTelemetry(candidates_filtered_no_privilege=3, sudo_denied=2)
        a.merge(b)
        assert a.candidates_filtered_no_privilege == 5
        assert a.sudo_attempted == 1
        assert a.sudo_denied == 2


class TestCommandTemplatePrivilegeFields:
    """Test that new fields on CommandTemplate work correctly."""

    def test_default_privilege_none(self):
        from core.commands.command_registry import CommandTemplate, AttackPhase
        cmd = CommandTemplate(name="t", template="t", description="t", phase=AttackPhase.RECON)
        assert cmd.requires_privilege == "none"
        assert cmd.privilege_reason == ""
        assert cmd.safety_tags == set()
        assert cmd.verify_template == ""
        assert cmd.required_tool == ""

    def test_privilege_fields_set(self):
        from core.commands.command_registry import CommandTemplate, AttackPhase
        cmd = CommandTemplate(
            name="tcpdump_capture",
            template="sudo tcpdump -i eth0 -w capture.pcap",
            description="Capture traffic",
            phase=AttackPhase.POST_EXPLOITATION,
            requires_privilege="root",
            privilege_reason="tcpdump requires root for raw socket access",
            safety_tags={"requires_root", "noisy", "network_disruptive"},
            verify_template="verify_tcpdump",
            required_tool="tcpdump",
        )
        assert cmd.requires_privilege == "root"
        assert cmd.privilege_reason == "tcpdump requires root for raw socket access"
        assert "requires_root" in cmd.safety_tags
        assert "noisy" in cmd.safety_tags
        assert cmd.required_tool == "tcpdump"

    def test_usage_context_includes_privilege(self):
        from core.commands.command_registry import CommandTemplate, AttackPhase
        cmd = CommandTemplate(
            name="t", template="t", description="desc",
            phase=AttackPhase.RECON,
            requires_privilege="sudo",
            safety_tags={"noisy"},
        )
        ctx = cmd.get_usage_context()
        assert "PRIVILEGE: sudo" in ctx
        assert "SAFETY: noisy" in ctx


class TestStepEventTelemetry:
    """Test Phase 10.1 telemetry fields on StepEvent."""

    def test_step_event_has_privilege_fields(self):
        from core.telemetry.events import StepEvent
        ev = StepEvent(privilege_filtered=5, sudo_attempted=True)
        d = ev.to_dict()
        assert d["privilege_filtered"] == 5
        assert d["sudo_attempted"] is True
        assert d["tool_install_triggered"] is False
        assert d["payload_transform_used"] == ""

    def test_episode_event_has_privilege_fields(self):
        from core.telemetry.events import EpisodeEvent
        ev = EpisodeEvent(
            total_privilege_filtered=10,
            total_sudo_attempts=3,
            total_sudo_allowed=2,
            total_sudo_denied=1,
        )
        d = ev.to_dict()
        assert d["total_privilege_filtered"] == 10
        assert d["total_sudo_attempts"] == 3


class TestFeatureFlagPrivilege:
    """Test Phase 10.1 feature flags."""

    def test_privilege_gating_default_on(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.privilege_gating is True

    def test_allow_sudo_default_off(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.allow_sudo is False

    def test_allow_live_install_default_off(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.allow_live_install is False

    def test_sudo_mode_default(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.sudo_mode == "prompt"
