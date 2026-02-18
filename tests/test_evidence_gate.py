#!/usr/bin/env python3
"""Tests for Phase 27.4 Evidence Gate in SmartCoach."""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


def _make_result(command: str):
    """Create a minimal SmartDecisionResult for testing."""
    from core.training.smart_coach import SmartDecisionResult
    return SmartDecisionResult(command=command, source="registry", confidence=0.7)


def _make_coach():
    """Create a SmartCoach with FakeGPTManager."""
    from core.testing.fake_gpt_manager import FakeGPTManager
    from core.training.smart_coach import SmartCoach
    gpt = FakeGPTManager(seed=42)
    return SmartCoach(agent_name="RedAgent", gpt_manager=gpt)  # type: ignore[arg-type]


class TestEvidenceGate:

    def test_gate_empty_discovery_board_rejects_exploit_port(self):
        """Port-targeting command should be rejected when port not discovered."""
        coach = _make_coach()
        from core.commands.command_registry import AttackPhase
        board = {"ports": set(), "services": set(), "vulns": set()}
        result = _make_result("hydra -l admin -P /tmp/pass.txt ssh://10.10.10.1 -p 22")
        valid, reasons = coach._validate_exploit_evidence(result, board, AttackPhase.EXPLOITATION)
        assert not valid
        assert any("port_22" in r for r in reasons)

    def test_gate_port_present_passes(self):
        """Command targeting a discovered port should pass."""
        coach = _make_coach()
        from core.commands.command_registry import AttackPhase
        board = {"ports": {22, 80}, "services": {"ssh", "http"}, "vulns": set()}
        result = _make_result("hydra -l admin -P /tmp/pass.txt ssh://10.10.10.1 -p 22")
        valid, reasons = coach._validate_exploit_evidence(result, board, AttackPhase.EXPLOITATION)
        assert valid

    def test_gate_privesc_command_bypass_passes(self):
        """Post-foothold local commands should pass even in PRIVESC phase."""
        coach = _make_coach()
        from core.commands.command_registry import AttackPhase
        board = {"ports": set(), "services": set(), "vulns": set()}
        result = _make_result("sudo -l")
        valid, reasons = coach._validate_exploit_evidence(result, board, AttackPhase.PRIVILEGE_ESCALATION)
        assert valid

    def test_gate_recon_phase_always_passes(self):
        """Gate should not fire for RECON/ENUMERATION phases."""
        coach = _make_coach()
        from core.commands.command_registry import AttackPhase
        board = {"ports": set(), "services": set(), "vulns": set()}
        result = _make_result("nmap -sV 10.10.10.1 -p 9999")
        valid, reasons = coach._validate_exploit_evidence(result, board, AttackPhase.RECON)
        assert valid

    def test_gate_log_mode_does_not_block(self, monkeypatch):
        """In 'log' mode, evidence gate should reject but NOT block execution."""
        monkeypatch.setenv("FF_STRICT_EXPLOIT_GATE", "log")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        coach = _make_coach()
        from core.commands.command_registry import AttackPhase
        board = {"ports": set(), "services": set(), "vulns": set()}
        result = _make_result("sqlmap -u http://10.10.10.1/vuln -p 4444")
        valid, reasons = coach._validate_exploit_evidence(result, board, AttackPhase.EXPLOITATION)
        # Gate says invalid but log mode means we still execute
        assert not valid
        assert len(reasons) > 0

    def test_gate_enforce_mode_blocks_and_falls_back(self, monkeypatch):
        """Testing that enforce mode triggers, gate rejects."""
        monkeypatch.setenv("FF_STRICT_EXPLOIT_GATE", "enforce")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        coach = _make_coach()
        from core.commands.command_registry import AttackPhase
        board = {"ports": set(), "services": set(), "vulns": set()}
        result = _make_result("curl http://10.10.10.1:8080/exploit")
        valid, reasons = coach._validate_exploit_evidence(result, board, AttackPhase.EXPLOITATION)
        assert not valid
        assert any("port_8080" in r or "service_evidence" in r for r in reasons)

    def test_cve_evidence_check(self):
        """CVE reference should be checked against vulns in discovery board."""
        coach = _make_coach()
        from core.commands.command_registry import AttackPhase
        board = {"ports": {80}, "services": {"http"}, "vulns": set()}
        result = _make_result("exploit CVE-2021-12345 --target 10.10.10.1")
        valid, reasons = coach._validate_exploit_evidence(result, board, AttackPhase.EXPLOITATION)
        assert not valid
        assert any("CVE-2021-12345" in r for r in reasons)

    def test_ssh_wrapped_allowed(self):
        """SSH-wrapped commands should pass (assume foothold)."""
        coach = _make_coach()
        from core.commands.command_registry import AttackPhase
        board = {"ports": set(), "services": set(), "vulns": set()}
        result = _make_result("sshpass -p 'pass' ssh user@10.10.10.1 'cat /etc/passwd'")
        valid, reasons = coach._validate_exploit_evidence(result, board, AttackPhase.EXPLOITATION)
        assert valid
