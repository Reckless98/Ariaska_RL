"""
tests/test_ops_discovery_trust.py — Phase 38.1: Discovery Trust Engine tests

Covers:
  - VerificationLevel assignment.
  - Trust scoring for different source stages.
  - Cross-step corroboration.
  - Spike guard enforcement.
  - High-value discovery downgrading.
  - Integration with RewardCalculator.
  - OutputParser normalization (service, credential, web path).
  - DiscoveryEvent new fields.
"""

import os
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestVerificationLevels:
    """Test verification level assignment logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.discovery_trust import (
            DiscoveryTrustEngine,
            VerificationLevel,
        )
        self.engine = DiscoveryTrustEngine()
        self.VL = VerificationLevel

    def test_regex_source_single_source(self):
        level = self.engine.assign_verification_level("open_port", 80, "regex")
        assert level == self.VL.SINGLE_SOURCE

    def test_llm_source_corroborated(self):
        level = self.engine.assign_verification_level("credential", "admin:pw", "llm")
        assert level == self.VL.CORROBORATED

    def test_unknown_source_unverified(self):
        level = self.engine.assign_verification_level("vulnerability", "CVE-2024-1234", "unknown_stage")
        assert level == self.VL.UNVERIFIED

    def test_cross_command_corroboration(self):
        # First observation
        self.engine.assign_verification_level("open_port", 22, "regex", "nmap -sV 10.10.10.1")
        # Same discovery from different command → confirmed
        level = self.engine.assign_verification_level("open_port", 22, "regex", "masscan 10.10.10.1")
        assert level == self.VL.CONFIRMED

    def test_same_command_no_confirm(self):
        self.engine.assign_verification_level("service", "ssh", "regex", "nmap -sV 10.10.10.1")
        level = self.engine.assign_verification_level("service", "ssh", "regex", "nmap -sV 10.10.10.1")
        # Same command → stays at single_source
        assert level != self.VL.CONFIRMED

    def test_trust_score_monotonic(self):
        assert self.VL.UNVERIFIED.trust_score < self.VL.SINGLE_SOURCE.trust_score
        assert self.VL.SINGLE_SOURCE.trust_score < self.VL.CORROBORATED.trust_score
        assert self.VL.CORROBORATED.trust_score < self.VL.CONFIRMED.trust_score

    def test_reset_clears_state(self):
        self.engine.assign_verification_level("open_port", 80, "regex", "nmap")
        self.engine.reset()
        stats = self.engine.get_stats()
        assert stats["total_tracked"] == 0


class TestTrustEvaluation:
    """Test trust scoring and spike guard."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.discovery_trust import DiscoveryTrustEngine
        self.engine = DiscoveryTrustEngine(max_step_bonus=45.0)
        self.bonus_table = {
            "open_port": 2.5,
            "service": 5.0,
            "credential": 20.0,
            "shell": 40.0,
        }

    def test_basic_trust_scaling(self):
        discoveries = {"open_port": [22, 80]}
        result = self.engine.evaluate(
            discoveries, "regex", "nmap -sV", self.bonus_table,
        )
        # 2 ports * 2.5 * 0.6 (single_source trust) = 3.0
        assert result.trusted_bonus > 0
        assert result.trusted_bonus <= result.original_bonus

    def test_spike_guard_caps_bonus(self):
        # Use a lower cap to trigger spike guard even after trust scaling
        from core.ops.discovery_trust import DiscoveryTrustEngine
        engine = DiscoveryTrustEngine(max_step_bonus=10.0)
        discoveries = {
            "shell": [True],
            "credential": ["admin:pw", "root:toor"],
            "open_port": [22, 80, 443, 8080, 8443],
        }
        result = engine.evaluate(
            discoveries, "regex", "big_scan", self.bonus_table,
        )
        assert result.trusted_bonus <= 10.0
        assert result.spike_capped

    def test_high_value_downgrade(self):
        discoveries = {"credential": ["admin:password123"]}
        result = self.engine.evaluate(
            discoveries, "regex", "hydra -l admin", self.bonus_table,
        )
        # Credential from regex only → downgraded (single_source × 0.5 extra)
        assert result.discoveries_downgraded >= 1
        assert result.trusted_bonus < result.original_bonus

    def test_llm_confirmed_no_downgrade(self):
        discoveries = {"credential": ["admin:password123"]}
        result = self.engine.evaluate(
            discoveries, "llm", "hydra -l admin", self.bonus_table,
        )
        # LLM source → corroborated, no extra penalty
        assert result.discoveries_downgraded == 0

    def test_empty_discoveries(self):
        result = self.engine.evaluate({}, "regex", "nmap", self.bonus_table)
        assert result.original_bonus == 0
        assert result.trusted_bonus == 0
        assert not result.spike_capped


class TestOutputParserNormalization:
    """Test Phase 38.1 normalization in OutputParser."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.execution.output_parser import OutputParser, ParsedOutput
        self.parser = OutputParser()
        self.PO = ParsedOutput

    def test_service_normalization_httpd(self):
        result = self.PO(command="nmap -sV", services={80: "httpd"})
        self.parser._normalize_services(result)
        assert result.services[80] == "http"

    def test_service_normalization_vsftpd(self):
        result = self.PO(command="nmap -sV", services={21: "vsftpd"})
        self.parser._normalize_services(result)
        assert result.services[21] == "ftp"

    def test_service_normalization_sshd(self):
        result = self.PO(command="nmap -sV", services={22: "sshd"})
        self.parser._normalize_services(result)
        assert result.services[22] == "ssh"

    def test_service_normalization_unknown_kept(self):
        result = self.PO(command="nmap -sV", services={9999: "custom_svc"})
        self.parser._normalize_services(result)
        assert result.services[9999] == "custom_svc"

    def test_credential_guard_placeholder(self):
        result = self.PO(command="test", credentials=[
            {"username": "admin", "password": "admin"},  # exact match → filtered
            {"username": "real_user", "password": "S3cureP@ss!"},
            {"username": "admin", "password": "admin123"},  # admin:admin123 ≠ admin:admin
        ])
        self.parser._guard_credentials(result)
        assert len(result.credentials) == 2
        assert result.credentials[0]["username"] == "real_user"
        assert result.credentials[1]["username"] == "admin"

    def test_credential_guard_empty(self):
        result = self.PO(command="test", credentials=[
            {"username": "", "password": ""},
        ])
        self.parser._guard_credentials(result)
        assert len(result.credentials) == 0

    def test_credential_guard_short_pw(self):
        result = self.PO(command="test", credentials=[
            {"username": "user", "password": "x"},
        ])
        self.parser._guard_credentials(result)
        assert len(result.credentials) == 0

    def test_web_path_sanitize_junk(self):
        result = self.PO(command="test", web_paths=[
            "/admin", "/", "/index.html", "/secret/panel",
        ])
        self.parser._sanitize_web_paths(result)
        assert "/" not in result.web_paths
        assert "/index.html" not in result.web_paths
        assert "/admin" in result.web_paths
        assert "/secret/panel" in result.web_paths

    def test_web_path_sanitize_local(self):
        result = self.PO(command="test", web_paths=[
            "/usr/share/wordlists", "/api/users",
        ])
        self.parser._sanitize_web_paths(result)
        assert "/usr/share/wordlists" not in result.web_paths
        assert "/api/users" in result.web_paths

    def test_web_path_dedup(self):
        result = self.PO(command="test", web_paths=[
            "/admin", "/admin/", "/admin",
        ])
        self.parser._sanitize_web_paths(result)
        assert len(result.web_paths) == 1


class TestDiscoveryEventTrustFields:
    """Test DiscoveryEvent Phase 38.1 fields."""

    def test_default_verification_level(self):
        from core.execution.discovery_event import DiscoveryEvent, DiscoveryType
        event = DiscoveryEvent(
            discovery_type=DiscoveryType.PORT,
            value=80,
        )
        assert event.verification_level == "single_source"
        assert event.trust_score == 0.6

    def test_custom_trust_score(self):
        from core.execution.discovery_event import DiscoveryEvent, DiscoveryType
        event = DiscoveryEvent(
            discovery_type=DiscoveryType.CREDENTIAL,
            value="admin:pw",
            verification_level="corroborated",
            trust_score=0.8,
        )
        assert event.trust_score == 0.8

    def test_to_dict_includes_trust(self):
        from core.execution.discovery_event import DiscoveryEvent, DiscoveryType
        event = DiscoveryEvent(
            discovery_type=DiscoveryType.SERVICE,
            value="ssh",
            verification_level="confirmed",
            trust_score=1.0,
        )
        d = event.to_dict()
        assert d["verification_level"] == "confirmed"
        assert d["trust_score"] == 1.0


class TestRewardCalculatorTrustIntegration:
    """Test trust engine integration in SmartRewardCalculator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.llm.reward_calculator import SmartRewardCalculator
        from core.commands.command_registry import AttackPhase
        self.calc = SmartRewardCalculator()
        self.AP = AttackPhase

    def test_trust_engine_lazy_loaded(self):
        engine = self.calc._get_trust_engine()
        assert engine is not None

    def test_trust_reduces_discovery_bonus_for_high_value(self):
        # A credential discovered via regex should be trust-reduced
        result = self.calc.calculate_reward(
            template_name="hydra_ssh",
            command="hydra -l admin -P pass.txt ssh://10.10.10.1",
            success=True,
            raw_output="[22][ssh] host: 10.10.10.1 login: admin password: secret",
            current_phase=self.AP.EXPLOITATION,
            state_flags={"ports_discovered": True},
            new_discoveries={"credential": ["admin:secret"]},
        )
        # With trust, bonus should be less than raw 20.0
        assert result.discovery_bonus < 20.0
        assert result.discovery_bonus > 0

    def test_reset_resets_trust(self):
        engine = self.calc._get_trust_engine()
        engine.assign_verification_level("open_port", 80, "regex", "nmap")
        self.calc.reset()
        stats = engine.get_stats()
        assert stats["total_tracked"] == 0
