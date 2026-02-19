"""
tests/test_ops_hypothesis_cooldown.py — Phase D: Hypothesis + Cooldown Tests

Covers:
  - CommandLockout: failure tracking, lockout, decay
  - ExploitConfidenceTracker: evidence, attempts, ranking
  - ExploitCooldownManager: progressive backoff, availability
  - Feature flags for Phase 38.3
  - Integration: lockout + confidence + cooldown workflow
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ─────────────────────────────────────────────────────────────────────────────
# Command Lockout Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCommandLockout:
    """Tests for CommandLockout."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.command_lockout import CommandLockout
        self.lockout = CommandLockout(threshold=3, decay_steps=15)

    def test_not_locked_initially(self):
        """Unknown template is not locked."""
        assert not self.lockout.is_locked("nmap_scan", current_step=0)

    def test_single_failure_not_locked(self):
        """Single failure does not trigger lockout."""
        self.lockout.record_result("nmap_scan", success=False, step=1)
        assert not self.lockout.is_locked("nmap_scan", current_step=1)

    def test_three_failures_locks(self):
        """Three consecutive failures trigger lockout."""
        for i in range(3):
            self.lockout.record_result("vsftpd_backdoor", success=False, step=i)
        assert self.lockout.is_locked("vsftpd_backdoor", current_step=3)

    def test_success_resets_failures(self):
        """Success resets consecutive failure count."""
        self.lockout.record_result("ssh_brute", success=False, step=1)
        self.lockout.record_result("ssh_brute", success=False, step=2)
        self.lockout.record_result("ssh_brute", success=True, step=3)
        self.lockout.record_result("ssh_brute", success=False, step=4)
        assert not self.lockout.is_locked("ssh_brute", current_step=4)

    def test_lockout_decays(self):
        """Lockout expires after decay_steps."""
        for i in range(3):
            self.lockout.record_result("exploit_a", success=False, step=i)
        assert self.lockout.is_locked("exploit_a", current_step=5)
        assert not self.lockout.is_locked("exploit_a", current_step=20)

    def test_success_unlocks(self):
        """Success after lockout removes the lock."""
        for i in range(3):
            self.lockout.record_result("exploit_b", success=False, step=i)
        assert self.lockout.is_locked("exploit_b", current_step=3)
        self.lockout.record_result("exploit_b", success=True, step=4)
        assert not self.lockout.is_locked("exploit_b", current_step=4)

    def test_get_locked_templates(self):
        """get_locked_templates returns currently locked ones."""
        for i in range(3):
            self.lockout.record_result("exploit_c", success=False, step=i)
        locked = self.lockout.get_locked_templates(current_step=3)
        assert "exploit_c" in locked

    def test_get_entry(self):
        """get_entry returns entry with correct stats."""
        self.lockout.record_result("test_cmd", success=False, step=1)
        self.lockout.record_result("test_cmd", success=True, step=2)
        entry = self.lockout.get_entry("test_cmd")
        assert entry is not None
        assert entry.total_attempts == 2
        assert entry.total_failures == 1
        assert entry.success_rate == 0.5

    def test_get_stats(self):
        """Stats returns summary."""
        self.lockout.record_result("cmd_1", success=False, step=1)
        stats = self.lockout.get_stats()
        assert "total_tracked" in stats
        assert stats["total_tracked"] == 1

    def test_empty_template_ignored(self):
        """Empty template name is ignored."""
        result = self.lockout.record_result("", success=False, step=1)
        assert result is False
        assert self.lockout.get_stats()["total_tracked"] == 0

    def test_reset(self):
        """Reset clears all entries."""
        self.lockout.record_result("cmd", success=False, step=1)
        self.lockout.reset()
        assert self.lockout.get_stats()["total_tracked"] == 0

    def test_record_returns_lockout_state(self):
        """record_result returns True when lockout triggers."""
        self.lockout.record_result("x", success=False, step=1)
        self.lockout.record_result("x", success=False, step=2)
        result = self.lockout.record_result("x", success=False, step=3)
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# Exploit Confidence Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExploitConfidence:
    """Tests for ExploitConfidenceTracker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.exploit_confidence import ExploitConfidenceTracker
        self.tracker = ExploitConfidenceTracker()

    def test_register_exploit(self):
        """Register sets base confidence."""
        entry = self.tracker.register_exploit("vsftpd_backdoor", base_confidence=0.8)
        assert entry.current_confidence == 0.8
        assert entry.template_name == "vsftpd_backdoor"

    def test_register_duplicate_returns_existing(self):
        """Re-registering returns existing entry."""
        e1 = self.tracker.register_exploit("ssh_brute", base_confidence=0.5)
        e2 = self.tracker.register_exploit("ssh_brute", base_confidence=0.9)
        assert e1 is e2
        assert e1.current_confidence == 0.5  # Not overwritten

    def test_evidence_boosts_confidence(self):
        """Adding evidence increases confidence."""
        self.tracker.register_exploit("sqli", base_confidence=0.4)
        conf = self.tracker.add_evidence("sqli", "port_80_open")
        assert conf > 0.4

    def test_evidence_diminishing_returns(self):
        """Multiple evidence items have diminishing returns."""
        self.tracker.register_exploit("sqli", base_confidence=0.3)
        boost1 = self.tracker.add_evidence("sqli", "ev1") - 0.3
        conf_after_1 = self.tracker.get_confidence("sqli")
        boost2 = self.tracker.add_evidence("sqli", "ev2") - conf_after_1
        assert boost2 < boost1  # Diminishing

    def test_duplicate_evidence_ignored(self):
        """Same evidence doesn't boost twice."""
        self.tracker.register_exploit("sqli", base_confidence=0.4)
        c1 = self.tracker.add_evidence("sqli", "port_80")
        c2 = self.tracker.add_evidence("sqli", "port_80")
        assert c1 == c2

    def test_failure_reduces_confidence(self):
        """Failed attempt reduces confidence."""
        self.tracker.register_exploit("rce", base_confidence=0.6)
        conf = self.tracker.record_attempt("rce", success=False, step=1)
        assert conf < 0.6

    def test_success_increases_confidence(self):
        """Successful attempt boosts confidence."""
        self.tracker.register_exploit("rce", base_confidence=0.5)
        conf = self.tracker.record_attempt("rce", success=True, step=1)
        assert conf > 0.5

    def test_max_confidence_capped(self):
        """Confidence cannot exceed MAX_CONFIDENCE."""
        from core.ops.exploit_confidence import MAX_CONFIDENCE
        self.tracker.register_exploit("rce", base_confidence=0.9)
        for i in range(10):
            self.tracker.record_attempt("rce", success=True, step=i)
        assert self.tracker.get_confidence("rce") <= MAX_CONFIDENCE

    def test_is_low_confidence(self):
        """is_low_confidence flags below threshold."""
        self.tracker.register_exploit("weak", base_confidence=0.1)
        assert self.tracker.is_low_confidence("weak")

    def test_unknown_template_zero_confidence(self):
        """Unknown template returns 0.0 confidence."""
        assert self.tracker.get_confidence("nonexistent") == 0.0

    def test_get_ranked_exploits(self):
        """Ranked list sorted by confidence descending."""
        self.tracker.register_exploit("a", base_confidence=0.3)
        self.tracker.register_exploit("b", base_confidence=0.8)
        self.tracker.register_exploit("c", base_confidence=0.5)
        ranked = self.tracker.get_ranked_exploits()
        assert ranked[0].template_name == "b"
        assert ranked[-1].template_name == "a"

    def test_get_ranked_with_min_filter(self):
        """min_confidence filters low entries."""
        self.tracker.register_exploit("a", base_confidence=0.1)
        self.tracker.register_exploit("b", base_confidence=0.8)
        ranked = self.tracker.get_ranked_exploits(min_confidence=0.5)
        assert len(ranked) == 1
        assert ranked[0].template_name == "b"

    def test_get_stats(self):
        """Stats summary is accurate."""
        self.tracker.register_exploit("x", base_confidence=0.2)
        self.tracker.register_exploit("y", base_confidence=0.8)
        stats = self.tracker.get_stats()
        assert stats["total_tracked"] == 2
        assert stats["avg_confidence"] == pytest.approx(0.5, abs=0.01)

    def test_reset(self):
        """Reset clears all entries."""
        self.tracker.register_exploit("cmd", base_confidence=0.5)
        self.tracker.reset()
        assert self.tracker.get_stats()["total_tracked"] == 0

    def test_empty_template_raises(self):
        """Empty template name raises ValueError."""
        with pytest.raises(ValueError):
            self.tracker.register_exploit("")

    def test_auto_register_on_evidence(self):
        """Adding evidence auto-registers if not found."""
        conf = self.tracker.add_evidence("new_exploit", "ev1")
        assert conf > 0.0
        assert self.tracker.get_entry("new_exploit") is not None


# ─────────────────────────────────────────────────────────────────────────────
# Exploit Cooldown Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExploitCooldown:
    """Tests for ExploitCooldownManager."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.ops.exploit_cooldown import ExploitCooldownManager
        self.cdm = ExploitCooldownManager(
            base_cooldown=3, max_cooldown=20, backoff=1.5,
        )

    def test_initially_available(self):
        """Unknown template is always available."""
        assert self.cdm.is_available("exploit_a", current_step=0)

    def test_cooldown_after_attempt(self):
        """Template not available during cooldown period."""
        self.cdm.record_attempt("exploit_a", step=5)
        assert not self.cdm.is_available("exploit_a", current_step=6)
        assert not self.cdm.is_available("exploit_a", current_step=7)

    def test_available_after_cooldown(self):
        """Template available after cooldown expires."""
        self.cdm.record_attempt("exploit_a", step=5)
        # After first attempt: cooldown = int(3*1.5)=4, next_available=5+4=9
        assert not self.cdm.is_available("exploit_a", current_step=8)
        assert self.cdm.is_available("exploit_a", current_step=9)

    def test_progressive_backoff(self):
        """Cooldown increases with each attempt."""
        self.cdm.record_attempt("exploit_b", step=0)  # cooldown=3
        cd1 = self.cdm.get_entry("exploit_b").current_cooldown
        self.cdm.record_attempt("exploit_b", step=5)  # cooldown=4 (3*1.5=4.5→4)
        cd2 = self.cdm.get_entry("exploit_b").current_cooldown
        assert cd2 > cd1

    def test_max_cooldown_capped(self):
        """Cooldown cannot exceed max."""
        for i in range(20):
            self.cdm.record_attempt("exploit_c", step=i * 50)
        entry = self.cdm.get_entry("exploit_c")
        assert entry.current_cooldown <= 20

    def test_success_resets_cooldown(self):
        """Success resets cooldown to base."""
        self.cdm.record_attempt("exploit_d", step=0)
        self.cdm.record_attempt("exploit_d", step=10)
        self.cdm.record_attempt("exploit_d", step=20, success=True)
        entry = self.cdm.get_entry("exploit_d")
        assert entry.current_cooldown == 3  # Reset to base
        assert entry.succeeded is True

    def test_steps_until_available(self):
        """steps_until_available returns remaining count."""
        self.cdm.record_attempt("exploit_e", step=10)
        # After attempt: cooldown = int(3*1.5)=4, next_available=14
        assert self.cdm.steps_until_available("exploit_e", 10) == 4
        assert self.cdm.steps_until_available("exploit_e", 12) == 2
        assert self.cdm.steps_until_available("exploit_e", 14) == 0

    def test_steps_until_unknown_is_zero(self):
        """Unknown template has 0 steps remaining."""
        assert self.cdm.steps_until_available("unknown", 10) == 0

    def test_get_available_exploits(self):
        """Filter templates by availability."""
        self.cdm.record_attempt("a", step=10)
        self.cdm.record_attempt("b", step=5)
        available = self.cdm.get_available_exploits(["a", "b", "c"], current_step=10)
        assert "c" in available  # Never attempted
        assert "b" in available  # 5+3=8 < 10
        assert "a" not in available  # 10+3=13 > 10

    def test_get_stats(self):
        """Stats returns summary."""
        self.cdm.record_attempt("x", step=0)
        self.cdm.record_attempt("y", step=0, success=True)
        stats = self.cdm.get_stats()
        assert stats["total_tracked"] == 2
        assert stats["succeeded"] == 1

    def test_reset(self):
        """Reset clears all cooldown state."""
        self.cdm.record_attempt("z", step=0)
        self.cdm.reset()
        assert self.cdm.get_stats()["total_tracked"] == 0

    def test_empty_template_ignored(self):
        """Empty template returns 0."""
        result = self.cdm.record_attempt("", step=0)
        assert result == 0


# ─────────────────────────────────────────────────────────────────────────────
# Feature Flag Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseD_Flags:
    """Phase 38.3 feature flags exist."""

    def test_command_lockout_flag(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "command_lockout")
        assert ff.command_lockout is True

    def test_exploit_confidence_flag(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "exploit_confidence")
        assert ff.exploit_confidence is True

    def test_exploit_cooldown_flag(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "exploit_cooldown")
        assert ff.exploit_cooldown is True


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseDIntegration:
    """Cross-module integration tests."""

    def test_lockout_and_cooldown_combined(self):
        """Lockout AND cooldown can independently block same template."""
        from core.ops.command_lockout import CommandLockout
        from core.ops.exploit_cooldown import ExploitCooldownManager

        lockout = CommandLockout(threshold=2, decay_steps=10)
        cooldown = ExploitCooldownManager(base_cooldown=5)

        # First attempt: cooldown starts, no lockout yet
        cooldown.record_attempt("exploit_x", step=0)
        lockout.record_result("exploit_x", success=False, step=0)
        assert not cooldown.is_available("exploit_x", 1)
        assert not lockout.is_locked("exploit_x", 1)

        # Second failure: lockout triggers
        lockout.record_result("exploit_x", success=False, step=1)
        assert lockout.is_locked("exploit_x", 2)

        # Both blocking
        is_blocked = lockout.is_locked("exploit_x", 2) or not cooldown.is_available("exploit_x", 2)
        assert is_blocked

    def test_confidence_gates_cooldown(self):
        """Low confidence should discourage attempts (integration pattern)."""
        from core.ops.exploit_confidence import ExploitConfidenceTracker, MIN_CONFIDENCE
        from core.ops.exploit_cooldown import ExploitCooldownManager

        conf = ExploitConfidenceTracker()
        cooldown = ExploitCooldownManager()

        conf.register_exploit("weak_exploit", base_confidence=0.1)

        # Pattern: check confidence before deciding to attempt
        should_attempt = (
            not conf.is_low_confidence("weak_exploit")
            and cooldown.is_available("weak_exploit", current_step=0)
        )
        assert not should_attempt  # Low confidence blocks it

    def test_hypothesis_system_exists(self):
        """Existing hypothesis system is importable."""
        from core.reasoning.hypothesis import HypothesisGenerator, Hypothesis
        gen = HypothesisGenerator()
        assert gen is not None

    def test_ops_package_exports(self):
        """All Phase D classes are importable from core.ops."""
        from core.ops import __all__
        assert "CommandLockout" in __all__
        assert "ExploitConfidenceTracker" in __all__
        assert "ExploitCooldownManager" in __all__
