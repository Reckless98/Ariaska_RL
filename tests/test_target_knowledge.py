#!/usr/bin/env python3
"""
tests/test_target_knowledge.py — Phase 58: Tests for TargetKnowledge

Tests:
  - Persistence (save/load round-trip)
  - Service recording
  - Exploit result recording
  - Credential recording
  - Privesc recording
  - Attack chain recording
  - Failed exploit tracking
  - Hypothesis boost calculation
  - State merge
  - Target ID sanitization
  - Stats
"""

import json
import os
import tempfile

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestTargetKnowledgePersistence:
    """Test save/load round-trip."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tmpdir = tempfile.mkdtemp()
        self.tk = TargetKnowledge(base_dir=self.tmpdir)

    def test_save_load_roundtrip(self):
        """Data should survive save/load cycle."""
        self.tk.load("10.0.0.1")
        self.tk.record_service(22, "ssh", "OpenSSH 7.2")
        self.tk.record_service(80, "http", "Apache 2.4.7")
        self.tk.record_exploit_result("vsftpd_backdoor", True, 50.0, episode=1)
        self.tk.record_credential("admin", "password123", "mysql")
        self.tk.save()

        # Load into new instance
        from core.memory.target_knowledge import TargetKnowledge
        tk2 = TargetKnowledge(base_dir=self.tmpdir)
        loaded = tk2.load("10.0.0.1")
        assert loaded is True
        assert 22 in tk2.services
        assert tk2.services[22].service == "ssh"
        assert 80 in tk2.services
        assert "vsftpd_backdoor" in tk2.exploit_attempts
        assert tk2.exploit_attempts["vsftpd_backdoor"].success is True
        assert len(tk2.credentials) == 1

    def test_load_nonexistent(self):
        """Loading nonexistent target should return False."""
        loaded = self.tk.load("255.255.255.255")
        assert loaded is False

    def test_save_creates_directory(self):
        """Save should create the base directory if needed."""
        from core.memory.target_knowledge import TargetKnowledge
        nested = os.path.join(self.tmpdir, "sub", "dir")
        tk = TargetKnowledge(base_dir=nested)
        tk.load("10.0.0.1")
        tk.record_service(22, "ssh")
        tk.save()
        assert os.path.exists(os.path.join(nested, "10.0.0.1.json"))


class TestTargetKnowledgeServices:
    """Test service recording."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_record_service(self):
        """Recording a service should add it to the store."""
        self.tk.record_service(22, "ssh", "OpenSSH 7.2")
        assert 22 in self.tk.services
        assert self.tk.services[22].service == "ssh"
        assert self.tk.services[22].version == "OpenSSH 7.2"

    def test_record_same_service_increments(self):
        """Recording same port twice should increment confirmed_count."""
        self.tk.record_service(22, "ssh", "OpenSSH 7.2")
        self.tk.record_service(22, "ssh", "OpenSSH 7.2")
        assert self.tk.services[22].confirmed_count == 2

    def test_version_update(self):
        """Recording with version when none existed should update it."""
        self.tk.record_service(80, "http")
        assert self.tk.services[80].version == ""
        self.tk.record_service(80, "http", "Apache 2.4.7")
        assert self.tk.services[80].version == "Apache 2.4.7"

    def test_get_service_list(self):
        """get_service_list should return dicts for ExploitReasoner."""
        self.tk.record_service(22, "ssh", "OpenSSH 7.2")
        self.tk.record_service(80, "http", "Apache 2.4.7")
        svc_list = self.tk.get_service_list()
        assert len(svc_list) == 2
        assert all("service" in s and "version" in s and "port" in s for s in svc_list)


class TestTargetKnowledgeExploits:
    """Test exploit result recording."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_record_success(self):
        """Successful exploit should be recorded with success=True."""
        self.tk.record_exploit_result("vsftpd_backdoor", True, 50.0, episode=1)
        assert "vsftpd_backdoor" in self.tk.exploit_attempts
        assert self.tk.exploit_attempts["vsftpd_backdoor"].success is True
        assert self.tk.exploit_attempts["vsftpd_backdoor"].reward == 50.0

    def test_record_failure(self):
        """Failed exploit should be recorded with success=False."""
        self.tk.record_exploit_result("ssh_brute", False, episode=1)
        assert "ssh_brute" in self.tk.exploit_attempts
        assert self.tk.exploit_attempts["ssh_brute"].success is False

    def test_repeated_attempt_increments(self):
        """Multiple attempts should increment attempt_count."""
        self.tk.record_exploit_result("ssh_brute", False, episode=1)
        self.tk.record_exploit_result("ssh_brute", False, episode=2)
        assert self.tk.exploit_attempts["ssh_brute"].attempt_count == 2

    def test_failed_then_success(self):
        """Success after failure should update to success."""
        self.tk.record_exploit_result("ssh_brute", False, episode=1)
        self.tk.record_exploit_result("ssh_brute", True, 20.0, episode=2)
        assert self.tk.exploit_attempts["ssh_brute"].success is True

    def test_get_failed_exploits(self):
        """get_failed_exploits should return consistently failing ones."""
        self.tk.record_exploit_result("bad_exploit", False, episode=1)
        self.tk.record_exploit_result("bad_exploit", False, episode=2)
        failed = self.tk.get_failed_exploits()
        assert "bad_exploit" in failed

    def test_get_successful_exploits(self):
        """get_successful_exploits should return ones that worked."""
        self.tk.record_exploit_result("good_exploit", True, 50.0, episode=1)
        success = self.tk.get_successful_exploits()
        assert "good_exploit" in success


class TestTargetKnowledgeCredentials:
    """Test credential recording."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_record_credential(self):
        """Recording a credential should add it."""
        self.tk.record_credential("admin", "password123", "mysql")
        assert len(self.tk.credentials) == 1
        assert self.tk.credentials[0]["username"] == "admin"

    def test_dedup_credential(self):
        """Same credential twice should not duplicate."""
        self.tk.record_credential("admin", "password123", "mysql")
        self.tk.record_credential("admin", "password123", "mysql")
        assert len(self.tk.credentials) == 1

    def test_different_credentials(self):
        """Different credentials should both be stored."""
        self.tk.record_credential("admin", "pass1")
        self.tk.record_credential("root", "pass2")
        assert len(self.tk.credentials) == 2

    def test_get_credential_list(self):
        """get_credential_list should return all credentials."""
        self.tk.record_credential("test", "test123")
        cl = self.tk.get_credential_list()
        assert len(cl) == 1


class TestTargetKnowledgePrivesc:
    """Test privesc recording."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_record_privesc_success(self):
        """Successful privesc should be recorded."""
        self.tk.record_privesc("sudo_all", True, "sudo su", episode=1)
        assert "sudo_all" in self.tk.privesc_vectors
        assert self.tk.privesc_vectors["sudo_all"].success is True

    def test_privesc_attempt_count(self):
        """Multiple attempts should increment count."""
        self.tk.record_privesc("suid_nmap", False, episode=1)
        self.tk.record_privesc("suid_nmap", False, episode=2)
        assert self.tk.privesc_vectors["suid_nmap"].attempt_count == 2


class TestTargetKnowledgeAttackChains:
    """Test attack chain recording."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_record_chain(self):
        """Recording a chain should add it."""
        self.tk.record_attack_chain(
            ["nmap_scan", "vsftpd_exploit", "privesc_sudo"],
            "EXFILTRATION", 100.0, episode=1,
        )
        assert len(self.tk.attack_chains) == 1
        assert self.tk.attack_chains[0].highest_phase == "EXFILTRATION"

    def test_best_chain(self):
        """get_best_chain should return highest-phase chain."""
        self.tk.record_attack_chain(["nmap"], "RECON", 10.0, episode=1)
        self.tk.record_attack_chain(
            ["nmap", "exploit", "sudo"], "EXFILTRATION", 100.0, episode=2
        )
        best = self.tk.get_best_chain()
        assert best is not None
        assert best.highest_phase == "EXFILTRATION"

    def test_chain_cap(self):
        """Should keep max 10 chains."""
        for i in range(15):
            self.tk.record_attack_chain([f"step_{i}"], "RECON", float(i), episode=i)
        assert len(self.tk.attack_chains) <= 10

    def test_best_phase_updates(self):
        """best_phase should track highest reached."""
        self.tk.record_attack_chain(["scan"], "RECON", 5.0, episode=1)
        assert self.tk.best_phase == "RECON"
        self.tk.record_attack_chain(["exploit"], "EXPLOITATION", 50.0, episode=2)
        assert self.tk.best_phase == "EXPLOITATION"


class TestTargetKnowledgeHypothesisBoost:
    """Test hypothesis confidence boost calculation."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_unknown_template_zero_boost(self):
        """Unknown templates should have zero boost."""
        assert self.tk.get_hypothesis_boost("never_tried") == 0.0

    def test_successful_positive_boost(self):
        """Successful exploits should get positive boost."""
        self.tk.record_exploit_result("good", True, 50.0, episode=1)
        boost = self.tk.get_hypothesis_boost("good")
        assert boost > 0.0

    def test_failed_negative_boost(self):
        """Failed exploits should get negative boost."""
        self.tk.record_exploit_result("bad", False, episode=1)
        self.tk.record_exploit_result("bad", False, episode=2)
        boost = self.tk.get_hypothesis_boost("bad")
        assert boost < 0.0

    def test_boost_capped(self):
        """Boost should be capped at ±0.30."""
        for i in range(20):
            self.tk.record_exploit_result("many_fails", False, episode=i)
        boost = self.tk.get_hypothesis_boost("many_fails")
        assert boost >= -0.30


class TestTargetKnowledgeStateMerge:
    """Test merging target knowledge into state dict."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_merge_services(self):
        """Merging with known services should set ports_discovered."""
        self.tk.record_service(22, "ssh")
        state = {}
        result = self.tk.merge_into_state(state)
        assert result.get("ports_discovered") is True
        assert 22 in result.get("known_ports", set())

    def test_merge_credentials(self):
        """Merging with credentials should set has_prior_creds."""
        self.tk.record_credential("admin", "pass")
        state = {}
        result = self.tk.merge_into_state(state)
        assert result.get("has_prior_creds") is True

    def test_merge_os_info(self):
        """OS info should be included in state."""
        self.tk.record_os_info("Ubuntu 14.04", "3.13.0-24")
        state = {}
        result = self.tk.merge_into_state(state)
        assert result.get("os_info") == "Ubuntu 14.04"
        assert result.get("kernel_version") == "3.13.0-24"

    def test_merge_empty_no_change(self):
        """Empty knowledge should not add keys."""
        state = {"existing": True}
        result = self.tk.merge_into_state(state)
        assert result == {"existing": True}


class TestTargetKnowledgeSanitization:
    """Test target ID sanitization."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())

    def test_ip_sanitized(self):
        """IP addresses should pass through with dots."""
        assert self.tk._sanitize_target_id("10.0.0.1") == "10.0.0.1"

    def test_hostname_sanitized(self):
        """Hostnames should pass through."""
        assert self.tk._sanitize_target_id("target.htb") == "target.htb"

    def test_special_chars_replaced(self):
        """Special characters should be replaced with underscore."""
        assert self.tk._sanitize_target_id("test/../../etc") == "test_.._.._etc"

    def test_empty_returns_unknown(self):
        """Empty string should return 'unknown'."""
        assert self.tk._sanitize_target_id("") == "unknown"


class TestTargetKnowledgeStats:
    """Test stats reporting."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_stats_structure(self):
        """Stats should have all expected keys."""
        stats = self.tk.get_stats()
        expected_keys = {
            "target", "services", "exploit_attempts", "successful_exploits",
            "failed_exploits", "credentials", "privesc_vectors",
            "attack_chains", "best_phase", "total_engagements",
        }
        assert expected_keys.issubset(stats.keys())

    def test_stats_count_accuracy(self):
        """Stats counts should match actual data."""
        self.tk.record_service(22, "ssh")
        self.tk.record_service(80, "http")
        self.tk.record_exploit_result("good", True, 50.0, episode=1)
        self.tk.record_exploit_result("bad", False, episode=1)
        self.tk.record_exploit_result("bad", False, episode=2)
        stats = self.tk.get_stats()
        assert stats["services"] == 2
        assert stats["exploit_attempts"] == 2
        assert stats["successful_exploits"] == 1
        assert stats["failed_exploits"] == 1  # bad has 2 attempts but >= 2 threshold


class TestTargetKnowledgeEpisodeLifecycle:
    """Test episode lifecycle management."""

    def setup_method(self):
        from core.memory.target_knowledge import TargetKnowledge
        self.tk = TargetKnowledge(base_dir=tempfile.mkdtemp())
        self.tk.load("10.0.0.1")

    def test_reset_episode_increments(self):
        """reset_episode should increment engagement count."""
        assert self.tk.total_engagements == 0
        self.tk.reset_episode()
        assert self.tk.total_engagements == 1
        self.tk.reset_episode()
        assert self.tk.total_engagements == 2
