#!/usr/bin/env python3
"""
tests/test_phase101_tools.py — Phase 10.1B: ToolRegistry + Live Install Tests

Tests for tool registry, availability checking, install gating, and
bootstrap reporting.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestToolRegistry:
    """Test ToolRegistry core operations."""

    def setup_method(self):
        from core.tools.tool_registry import reset_tool_registry
        reset_tool_registry()

    def test_registry_populated(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        # Should have many tools
        assert reg.get_tool("nmap") is not None
        assert reg.get_tool("hydra") is not None
        assert reg.get_tool("gobuster") is not None
        assert reg.get_tool("nonexistent_tool_xyz") is None

    def test_is_registered(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        assert reg.is_registered("nmap") is True
        assert reg.is_registered("notreal") is False

    def test_tool_entry_fields(self):
        from core.tools.tool_registry import get_tool_registry, ToolGroup, InstallMethod
        reg = get_tool_registry()
        entry = reg.get_tool("nmap")
        assert entry is not None
        assert entry.group == ToolGroup.RECON
        assert entry.install_method == InstallMethod.APT
        assert entry.install_target == "nmap"
        assert entry.requires_sudo_install is True
        assert len(entry.check_commands) > 0

    def test_tool_profiles_exist(self):
        from core.tools.tool_registry import TOOL_PROFILES
        assert "htb" in TOOL_PROFILES
        assert "ms3" in TOOL_PROFILES
        assert "ms2" in TOOL_PROFILES
        assert "dev" in TOOL_PROFILES
        assert len(TOOL_PROFILES["htb"]) > 10
        assert len(TOOL_PROFILES["dev"]) >= 2

    def test_get_profile_tools(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        tools = reg.get_profile_tools("dev")
        assert "nmap" in tools
        assert "curl" in tools

    def test_install_unregistered_tool_fails(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        result = reg.install_tool("totally_fake_tool_xyz")
        assert result.success is False
        assert "not in registry" in result.message

    def test_install_sudo_blocked_when_no_flag(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        # Mock tool as not available
        reg._availability_cache["nmap"] = False
        result = reg.install_tool("nmap", allow_sudo=False)
        assert result.success is False
        assert "sudo" in result.message.lower()

    def test_dry_run_install(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        reg._availability_cache["nmap"] = False
        result = reg.install_tool("nmap", dry_run=True, allow_sudo=True)
        assert result.success is False
        assert "DRY_RUN" in result.message

    def test_already_installed_skips(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        reg._availability_cache["nmap"] = True
        result = reg.install_tool("nmap")
        assert result.success is True
        assert "Already installed" in result.message


class TestToolRegistryLiveCaps:
    """Test live install caps and safety."""

    def setup_method(self):
        from core.tools.tool_registry import reset_tool_registry
        reset_tool_registry()

    def test_can_live_install_respects_flag(self):
        from core.tools.tool_registry import get_tool_registry
        from core.feature_flags import set_feature_flag
        set_feature_flag("allow_live_install", False)
        reg = get_tool_registry()
        assert reg.can_live_install() is False

    def test_can_live_install_when_enabled(self):
        from core.tools.tool_registry import get_tool_registry
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("allow_live_install", True)
        reg = get_tool_registry()
        assert reg.can_live_install() is True

    def test_install_cap_per_episode(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        reg._install_count_episode = reg.MAX_INSTALLS_PER_EPISODE
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("allow_live_install", True)
        assert reg.can_live_install() is False

    def test_install_cap_per_run(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        reg._install_count_run = reg.MAX_INSTALLS_PER_RUN
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("allow_live_install", True)
        assert reg.can_live_install() is False

    def test_episode_reset(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        reg._install_count_episode = 3
        reg.reset_episode()
        assert reg._install_count_episode == 0

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestBootstrapReport:
    """Test bootstrap reporting."""

    def setup_method(self):
        from core.tools.tool_registry import reset_tool_registry
        reset_tool_registry()

    def test_bootstrap_dry_run(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        report = reg.bootstrap(profile="dev", dry_run=True)
        # Should have skipped (present) + missing (not present) but no installed
        assert len(report.installed) == 0
        total = len(report.skipped) + len(report.missing)
        assert total > 0

    def test_bootstrap_report_to_dict(self):
        from core.tools.tool_registry import BootstrapReport, InstallResult
        report = BootstrapReport(
            profile="test",
            skipped=["nmap"],
            missing=["gobuster"],
            failures=[InstallResult(tool_name="ffuf", success=False, message="no sudo")],
        )
        d = report.to_dict()
        assert d["profile"] == "test"
        assert d["summary"]["skipped"] == 1
        assert d["summary"]["missing"] == 1
        assert d["summary"]["failures"] == 1

    def test_install_result_to_dict(self):
        from core.tools.tool_registry import InstallResult
        r = InstallResult(
            tool_name="nmap", success=True,
            method="apt", duration_ms=1500,
        )
        d = r.to_dict()
        assert d["tool"] == "nmap"
        assert d["success"] is True
        assert d["duration_ms"] == 1500


class TestInstallTemplates:
    """Test that install templates are registered in CommandRegistry."""

    def test_install_apt_template_exists(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        assert "install_apt_package" in COMMAND_REGISTRY
        cmd = COMMAND_REGISTRY["install_apt_package"]
        assert cmd.requires_privilege == "sudo"
        assert "package" in cmd.required_params

    def test_install_pipx_template_exists(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        assert "install_pipx_package" in COMMAND_REGISTRY

    def test_install_pip_template_exists(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        assert "install_pip_package" in COMMAND_REGISTRY

    def test_install_go_template_exists(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        assert "install_go_tool" in COMMAND_REGISTRY

    def test_clone_repo_template_exists(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        assert "clone_repo_tool" in COMMAND_REGISTRY

    def test_install_templates_zero_reward(self):
        """Install templates should not contribute to reward."""
        from core.commands.command_registry import COMMAND_REGISTRY
        for name in ["install_apt_package", "install_pipx_package",
                     "install_pip_package", "install_go_tool", "clone_repo_tool"]:
            assert COMMAND_REGISTRY[name].typical_reward == 0.0


class TestToolGroups:
    """Test tool group and enumeration."""

    def test_all_groups_have_tools(self):
        from core.tools.tool_registry import get_tool_registry, ToolGroup, _TOOL_ENTRIES
        reg = get_tool_registry()
        groups_with_tools = set()
        for entry in _TOOL_ENTRIES.values():
            groups_with_tools.add(entry.group)
        # At minimum these groups should be populated
        assert ToolGroup.RECON in groups_with_tools
        assert ToolGroup.EXPLOIT in groups_with_tools
        assert ToolGroup.BRUTE in groups_with_tools
        assert ToolGroup.WEB in groups_with_tools

    def test_cache_invalidation(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        reg._availability_cache["test_tool"] = True
        assert reg._availability_cache.get("test_tool") is True
        reg.invalidate_cache("test_tool")
        assert "test_tool" not in reg._availability_cache

    def test_full_cache_invalidation(self):
        from core.tools.tool_registry import get_tool_registry
        reg = get_tool_registry()
        reg._availability_cache["a"] = True
        reg._availability_cache["b"] = False
        reg.invalidate_cache()
        assert len(reg._availability_cache) == 0
