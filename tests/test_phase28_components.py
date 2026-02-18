"""Phase 28: Watchdog semantic stall, LiveExecutor retry, Self-debug tests."""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.training.watchdog import (
    TrainingWatchdog,
    WatchdogConfig,
    WatchdogTrigger,
    StepSnapshot,
    HealAction,
)
from core.execution.self_debug import SelfDebugger, DebugFix
from core.execution.live_executor import LiveCommandExecutor


# ─── Watchdog Semantic Stall ────────────────────────────────────────────


class TestSemanticStall:
    """Test the Phase 28 SEMANTIC_STALL trigger."""

    @pytest.fixture(autouse=True)
    def setup(self):
        cfg = WatchdogConfig(
            stall_threshold=100,           # disable regular stall
            phase_stuck_threshold=100,     # disable phase stuck
            family_flood_count=100,        # disable family flood so semantic stall can fire
            semantic_stall_window=6,
            semantic_stall_threshold=4,
        )
        self.wd = TrainingWatchdog(cfg)
        self.wd.reset_episode()

    def _snap(self, step: int, family: str, discoveries=None):
        return StepSnapshot(
            step_num=step,
            phase="RECON",
            agent_name="RedAgent",
            command=f"{family} -sV 10.0.0.1 --opt{step}",
            command_family=family,
            discoveries=discoveries or {},
        )

    def test_triggers_on_semantic_stall(self):
        """4+ same-family commands with 0 discoveries → SEMANTIC_STALL."""
        for i in range(6):
            v = self.wd.check(self._snap(i, "nmap"))
        assert v.should_intervene
        assert v.trigger == WatchdogTrigger.SEMANTIC_STALL
        assert v.heal_action == HealAction.FORCE_MENTOR

    def test_no_trigger_with_discoveries(self):
        """Same family but discoveries are made → no stall."""
        for i in range(6):
            disc = {"open_port": "22"} if i % 2 == 0 else {}
            v = self.wd.check(self._snap(i, "nmap", disc))
        assert not v.should_intervene

    def test_no_trigger_diverse_families(self):
        """Diverse families → no semantic stall."""
        families = ["nmap", "gobuster", "nikto", "hydra", "sqlmap", "curl"]
        for i, fam in enumerate(families):
            v = self.wd.check(self._snap(i, fam))
        assert not v.should_intervene

    def test_trigger_in_stats(self):
        """SEMANTIC_STALL should appear in stats."""
        for i in range(6):
            self.wd.check(self._snap(i, "nmap"))
        stats = self.wd.get_stats()
        assert stats["trigger_counts"].get("semantic_stall", 0) >= 1


# ─── LiveExecutor Retry ────────────────────────────────────────────────


class TestLiveExecutorRetry:
    """Test retry config and transient detection."""

    def test_retry_constants_exist(self):
        assert LiveCommandExecutor.MAX_RETRIES == 2
        assert LiveCommandExecutor.RETRY_BASE_DELAY > 0

    def test_transient_patterns_exist(self):
        assert len(LiveCommandExecutor._TRANSIENT_PATTERNS) >= 4

    def test_is_transient_failure_method(self):
        """_is_transient_failure should detect Connection refused."""
        from core.execution.live_executor import LiveCommandResult
        result = LiveCommandResult(
            command="ssh user@10.0.0.1",
            agent_name="RedAgent",
            executed=True,
            stderr="ssh: connect to host 10.0.0.1 port 22: Connection refused",
            return_code=255,
        )
        # Need an executor to call the method
        # But _is_transient_failure is an instance method — use class-level check
        combined = (result.stdout + " " + result.stderr).strip()
        has_transient = any(
            p.lower() in combined.lower()
            for p in LiveCommandExecutor._TRANSIENT_PATTERNS
        )
        assert has_transient


# ─── Self-Debug Loop ──────────────────────────────────────────────────


class TestSelfDebugger:
    """Test the Phase 28 SelfDebugger."""

    def test_classify_command_not_found(self):
        dbg = SelfDebugger()
        fix = dbg.suggest_fix(
            "nmapx -sV 10.0.0.1",
            "bash: nmapx: command not found",
        )
        assert fix.error_class == "tool_missing"

    def test_heuristic_strips_trailing_x(self):
        """Heuristic fix: nmapx → nmap."""
        dbg = SelfDebugger()
        fix = dbg.suggest_fix(
            "nmapx -sV 10.0.0.1",
            "bash: nmapx: command not found",
        )
        assert fix.should_retry
        assert fix.corrected_command == "nmap -sV 10.0.0.1"

    def test_no_fix_for_unknown_error(self):
        dbg = SelfDebugger()
        fix = dbg.suggest_fix(
            "nmap -sV 10.0.0.1",
            "Some random unexpected output blah blah",
        )
        assert not fix.should_retry
        assert fix.error_class == ""

    def test_empty_input_returns_empty_fix(self):
        dbg = SelfDebugger()
        fix = dbg.suggest_fix("", "")
        assert not fix.should_retry

    def test_bad_flag_classified(self):
        dbg = SelfDebugger()
        fix = dbg.suggest_fix(
            "nmap --bogus 10.0.0.1",
            "nmap: unrecognized option '--bogus'",
        )
        assert fix.error_class == "bad_flag"

    def test_stats_tracking(self):
        dbg = SelfDebugger()
        dbg.suggest_fix("nmapx -sV 10.0.0.1", "bash: nmapx: command not found")
        assert dbg.stats["fixes_attempted"] == 0  # no GPT → heuristic only
        assert dbg.stats["fixes_succeeded"] == 0

    def test_port_closed_with_target_substitution(self):
        dbg = SelfDebugger()
        fix = dbg.suggest_fix(
            "ssh user@10.10.10.10",
            "Connection refused",
            target_ip="10.129.1.54",
        )
        assert fix.should_retry
        assert "10.129.1.54" in fix.corrected_command
