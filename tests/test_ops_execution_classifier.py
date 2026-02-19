"""
tests/test_ops_execution_classifier.py — ExecutionClassifier invariant tests

Covers:
  - LOCAL_OPS vs REMOTE classification.
  - Phase validation (exploit in RECON blocked).
  - Discovery extraction gating.
  - Target reference detection.
  - Edge cases (empty command, ambiguous).
"""

import os
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestExecutionClassifier:
    """Test command classification logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.execution_classifier import ExecutionClassifier, ExecutionClass
        self.classifier = ExecutionClassifier
        self.EC = ExecutionClass

    # ── LOCAL_OPS ────────────────────────────────────────────────────

    def test_apt_install_is_local(self):
        result = self.classifier.classify("apt install -y nmap")
        assert result == self.EC.LOCAL_OPS

    def test_pip_install_is_local(self):
        result = self.classifier.classify("pip install requests")
        assert result == self.EC.LOCAL_OPS

    def test_local_filesystem_is_local(self):
        result = self.classifier.classify("ls /usr/share/wordlists/")
        assert result == self.EC.LOCAL_OPS

    def test_searchsploit_no_target_is_local(self):
        result = self.classifier.classify("searchsploit vsftpd 2.3.4")
        assert result == self.EC.LOCAL_OPS

    def test_command_v_is_local(self):
        result = self.classifier.classify("command -v nmap")
        assert result == self.EC.LOCAL_OPS

    # ── REMOTE ───────────────────────────────────────────────────────

    def test_nmap_with_target_is_remote_recon(self):
        result = self.classifier.classify("nmap -sV 10.10.10.50", target_ip="10.10.10.50")
        assert result == self.EC.REMOTE_RECON

    def test_gobuster_with_target_is_remote_recon(self):
        result = self.classifier.classify(
            "gobuster dir -u http://10.10.10.50", target_ip="10.10.10.50",
        )
        assert result == self.EC.REMOTE_RECON

    def test_hydra_with_target_is_remote_exploit(self):
        result = self.classifier.classify(
            "hydra -l admin -P pass.txt ssh://10.10.10.50", target_ip="10.10.10.50",
        )
        assert result == self.EC.REMOTE_EXPLOIT

    def test_msfconsole_exploit_with_target(self):
        result = self.classifier.classify(
            "msfconsole -q -x 'use exploit/unix/ftp/vsftpd_234_backdoor'",
            target_ip="10.10.10.50",
        )
        # No target IP in string → ambiguous
        assert result == self.EC.AMBIGUOUS

    def test_ssh_with_domain_is_remote(self):
        result = self.classifier.classify(
            "ssh user@soulmate.htb", domain="soulmate.htb",
        )
        assert result == self.EC.REMOTE_RECON

    def test_curl_with_domain(self):
        result = self.classifier.classify(
            "curl -s http://soulmate.htb/robots.txt", domain="soulmate.htb",
        )
        assert result == self.EC.REMOTE_RECON

    # ── Phase validation ─────────────────────────────────────────────

    def test_exploit_in_recon_rejected(self):
        valid, reason = self.classifier.validate_execution_context(
            "hydra -l admin -P pass.txt ssh://10.10.10.50",
            self.EC.REMOTE_EXPLOIT,
            "recon",
        )
        assert not valid
        assert "not appropriate" in reason.lower()

    def test_recon_in_recon_allowed(self):
        valid, reason = self.classifier.validate_execution_context(
            "nmap -sV 10.10.10.50",
            self.EC.REMOTE_RECON,
            "recon",
        )
        assert valid

    def test_local_ops_in_exploit_allowed_no_discovery(self):
        valid, reason = self.classifier.validate_execution_context(
            "searchsploit vsftpd",
            self.EC.LOCAL_OPS,
            "exploit",
        )
        assert valid
        assert "no discoveries" in reason.lower()

    # ── Discovery extraction gating ──────────────────────────────────

    def test_local_ops_no_discovery_extraction(self):
        assert not self.classifier.should_extract_discoveries(self.EC.LOCAL_OPS)

    def test_remote_recon_extracts_discoveries(self):
        assert self.classifier.should_extract_discoveries(self.EC.REMOTE_RECON)

    def test_remote_exploit_extracts_discoveries(self):
        assert self.classifier.should_extract_discoveries(self.EC.REMOTE_EXPLOIT)

    def test_ambiguous_no_discovery_extraction(self):
        assert not self.classifier.should_extract_discoveries(self.EC.AMBIGUOUS)

    # ── Edge cases ───────────────────────────────────────────────────

    def test_empty_command(self):
        result = self.classifier.classify("")
        assert result == self.EC.AMBIGUOUS

    def test_none_command(self):
        result = self.classifier.classify(None)
        assert result == self.EC.AMBIGUOUS

    def test_target_placeholder(self):
        result = self.classifier.classify("nmap -sV {target}")
        assert result in (self.EC.REMOTE_RECON, self.EC.AMBIGUOUS)


class TestShadowAgentOPS:
    """Test ShadowAgent OPS extensions."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("ARIASKA_DRY_RUN", "1")
        monkeypatch.setenv("SUDO_PASSWORD", "test_pw")
        from core.agents.shadow_agent import ShadowAgent
        self.shadow = ShadowAgent(verbosity="quiet")

    def test_classify_execution_local(self):
        result = self.shadow.classify_execution("apt install nmap")
        assert result == "local_ops"

    def test_classify_execution_remote(self):
        result = self.shadow.classify_execution(
            "nmap -sV 10.10.10.50", target_ip="10.10.10.50",
        )
        assert result == "remote_recon"

    def test_validate_command_safety_ok(self):
        safe, reason = self.shadow.validate_command_safety(
            "nmap -sV 10.10.10.50", "recon", target_ip="10.10.10.50",
        )
        assert safe

    def test_validate_command_safety_exploit_in_recon(self):
        safe, reason = self.shadow.validate_command_safety(
            "hydra -l admin -P pass.txt ssh://10.10.10.50",
            "recon", target_ip="10.10.10.50",
        )
        assert not safe

    def test_audit_state_flags_shell_inconsistency(self):
        warnings = self.shadow.audit_state_flags({
            "shell_obtained": True,
            "shells": set(),  # Empty
        })
        assert len(warnings) >= 1
        assert any("shell" in w.lower() for w in warnings)

    def test_audit_state_flags_clean(self):
        warnings = self.shadow.audit_state_flags({
            "shell_obtained": False,
            "phase": "recon",
        })
        assert len(warnings) == 0


class TestScoutAgentOPS:
    """Test ScoutAgent OPS extensions."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("ARIASKA_DRY_RUN", "1")
        monkeypatch.setenv("SUDO_PASSWORD", "test_pw")
        from core.agents.scout_agent import ScoutAgent
        self.scout = ScoutAgent(verbosity="quiet")

    def test_ops_pre_flight_returns_dict(self):
        result = self.scout.ops_pre_flight("10.10.10.50")
        assert isinstance(result, dict)
        assert result["preflight_complete"] is True

    def test_ops_pre_flight_with_domain(self):
        result = self.scout.ops_pre_flight("10.10.10.50", domain="soulmate.htb")
        assert result["domain"] == "soulmate.htb"

    def test_ops_pre_flight_with_tools(self):
        result = self.scout.ops_pre_flight(
            "10.10.10.50", tools=["nmap", "gobuster"],
        )
        assert "tools_verified" in result

    def test_discover_domain_from_redirect(self):
        output = "HTTP/1.1 301 Moved\nLocation: http://soulmate.htb/dashboard\n"
        domain = self.scout.discover_domain(output)
        assert domain == "soulmate.htb"

    def test_discover_domain_from_ssl(self):
        output = "Subject: commonName = admin.htb"
        domain = self.scout.discover_domain(output)
        assert domain == "admin.htb"

    def test_discover_domain_from_dns(self):
        output = "DNS:portal.htb"
        domain = self.scout.discover_domain(output)
        assert domain == "portal.htb"

    def test_discover_domain_no_match(self):
        output = "No domain information found"
        domain = self.scout.discover_domain(output)
        assert domain is None

    def test_reset_clears_ops_state(self):
        self.scout._discovered_domain = "test.htb"
        self.scout._ops_preflight_done = True
        self.scout.reset()
        assert self.scout._discovered_domain is None
        assert self.scout._ops_preflight_done is False
