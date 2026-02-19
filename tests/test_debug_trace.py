"""
tests/test_debug_trace.py — Phase 39.4: DebugTracer tests

Tests structured JSONL debug telemetry: event logging, file creation,
rotation, entry format, close behavior.
"""

import json
import os
import tempfile
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.ops.debug_trace import DebugTraceEntry, DebugTracer


class TestDebugTraceEntry:
    """Tests for DebugTraceEntry dataclass."""

    def test_defaults(self):
        e = DebugTraceEntry()
        assert e.run_id == ""
        assert e.event_type == ""
        assert e.step == 0
        assert e.data == {}

    def test_to_jsonl(self):
        e = DebugTraceEntry(
            run_id="r80",
            event_type="stall",
            step=10,
            episode=1,
            phase="RECON",
            data={"stall_score": 0.65},
        )
        line = e.to_jsonl()
        parsed = json.loads(line)
        assert parsed["run_id"] == "r80"
        assert parsed["event"] == "stall"
        assert parsed["step"] == 10
        assert parsed["stall_score"] == 0.65

    def test_to_jsonl_compact(self):
        e = DebugTraceEntry(run_id="a", event_type="b")
        line = e.to_jsonl()
        # Compact JSON should have no spaces after separators
        assert ": " not in line
        assert ", " not in line

    def test_to_jsonl_nested_data(self):
        e = DebugTraceEntry(
            run_id="r1",
            event_type="test",
            data={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        parsed = json.loads(e.to_jsonl())
        assert parsed["nested"]["key"] == "value"
        assert parsed["list"] == [1, 2, 3]


class TestDebugTracerInit:
    """Tests for DebugTracer initialization."""

    def test_disabled(self):
        tracer = DebugTracer(enabled=False)
        assert tracer.entries_written == 0
        tracer.log_stall(step=1, stall_score=0.5)
        assert tracer.entries_written == 0
        tracer.close()

    def test_creates_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "nested", "debug")
            tracer = DebugTracer(log_dir=log_dir, run_id="init_test")
            assert os.path.isdir(log_dir)
            tracer.close()

    def test_creates_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = DebugTracer(log_dir=tmpdir, run_id="ftest")
            tracer.log_stall(step=0, stall_score=0.1)
            tracer.close()
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].startswith("debug_ftest_")
            assert files[0].endswith(".jsonl")

    def test_run_id_property(self):
        tracer = DebugTracer(enabled=False, run_id="test123")
        assert tracer.run_id == "test123"
        tracer.close()


class TestLogMethods:
    """Tests for all log_* methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self._tmpdir = tempfile.mkdtemp()
        self.tracer = DebugTracer(
            log_dir=self._tmpdir, run_id="logtest",
        )
        yield
        self.tracer.close()

    def _read_entries(self):
        """Read all JSONL entries from the log file."""
        entries = []
        for f in os.listdir(self._tmpdir):
            path = os.path.join(self._tmpdir, f)
            with open(path) as fh:
                for line in fh:
                    entries.append(json.loads(line))
        return entries

    def test_log_stall(self):
        self.tracer.log_stall(
            step=10, stall_score=0.65,
            signals={"evidence_plateau": 0.8},
            episode=1, phase="RECON",
        )
        entries = self._read_entries()
        assert len(entries) == 1
        assert entries[0]["event"] == "stall"
        assert entries[0]["stall_score"] == 0.65
        assert entries[0]["signals"]["evidence_plateau"] == 0.8

    def test_log_phase_transition(self):
        self.tracer.log_phase_transition(
            step=12, from_phase="RECON", to_phase="ENUMERATION",
            episode=1, reason="discoveries_complete",
        )
        entries = self._read_entries()
        assert entries[0]["event"] == "phase_transition"
        assert entries[0]["from"] == "RECON"
        assert entries[0]["to"] == "ENUMERATION"
        assert entries[0]["reason"] == "discoveries_complete"

    def test_log_evidence_update(self):
        self.tracer.log_evidence_update(
            step=5, evidence_count=10, delta=3,
            episode=0, phase="ENUM",
        )
        entries = self._read_entries()
        assert entries[0]["event"] == "evidence_update"
        assert entries[0]["evidence_count"] == 10

    def test_log_trust_update(self):
        self.tracer.log_trust_update(
            step=15, source="gpt", trust=0.72, delta=-0.08, event="failed",
        )
        entries = self._read_entries()
        assert entries[0]["event"] == "trust_update"
        assert entries[0]["source"] == "gpt"
        assert entries[0]["trust"] == 0.72
        assert entries[0]["trust_event"] == "failed"

    def test_log_prior_injection(self):
        self.tracer.log_prior_injection(
            step=20, source="mentor", magnitude=0.45, trust=0.80,
            changed_action=True, original_action=2, influenced_action=0,
        )
        entries = self._read_entries()
        assert entries[0]["event"] == "prior_injection"
        assert entries[0]["changed_action"] is True
        assert entries[0]["original"] == 2
        assert entries[0]["influenced"] == 0

    def test_log_alternatives(self):
        alts = [
            {"cmd": "nmap -sC", "reason": "already_run"},
            {"cmd": "nikto", "reason": "low_score"},
        ]
        self.tracer.log_alternatives(
            step=5, chosen="gobuster", alternatives=alts,
            episode=1, phase="ENUM",
        )
        entries = self._read_entries()
        assert entries[0]["event"] == "alternatives"
        assert entries[0]["chosen"] == "gobuster"
        assert len(entries[0]["alternatives"]) == 2

    def test_log_rethink(self):
        self.tracer.log_rethink(
            step=30, stall_score=0.7,
            plan_summary={"why_now": "plateau", "hypotheses": ["h1"]},
            episode=2, phase="EXPLOITATION",
        )
        entries = self._read_entries()
        assert entries[0]["event"] == "orion_rethink"
        assert entries[0]["stall_score"] == 0.7

    def test_log_decision(self):
        self.tracer.log_decision(
            step=8, agent="RedAgent",
            command="hydra -l admin -P wordlist.txt ssh://target",
            source="ppo", confidence=0.85,
            episode=1, phase="EXPLOITATION",
            extra={"reward": 5.0},
        )
        entries = self._read_entries()
        assert entries[0]["event"] == "decision"
        assert entries[0]["agent"] == "RedAgent"
        assert entries[0]["source"] == "ppo"
        assert entries[0]["reward"] == 5.0

    def test_log_cap_eval(self):
        self.tracer.log_cap_eval(
            scenario="cap",
            success_rate=0.95,
            runs=20,
            avg_steps=142.5,
            failures=[{"run": "3", "reason": "timeout"}],
            git_sha="abc123",
        )
        entries = self._read_entries()
        assert entries[0]["event"] == "cap_eval"
        assert entries[0]["success_rate"] == 0.95
        assert entries[0]["runs"] == 20


class TestMultipleEntries:
    """Tests for writing multiple entries."""

    def test_multiple_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = DebugTracer(log_dir=tmpdir, run_id="multi")
            for i in range(50):
                tracer.log_stall(step=i, stall_score=i / 100)
            assert tracer.entries_written == 50
            tracer.close()

    def test_entries_are_line_separated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = DebugTracer(log_dir=tmpdir, run_id="lines")
            tracer.log_stall(step=1, stall_score=0.1)
            tracer.log_stall(step=2, stall_score=0.2)
            tracer.close()

            files = os.listdir(tmpdir)
            with open(os.path.join(tmpdir, files[0])) as f:
                lines = f.readlines()
            assert len(lines) == 2
            for line in lines:
                json.loads(line)  # Must be valid JSON


class TestRunIdInEntries:
    """Tests that run_id is in all entries."""

    def test_run_id_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = DebugTracer(log_dir=tmpdir, run_id="rid42")
            tracer.log_stall(step=0, stall_score=0.0)
            tracer.log_phase_transition(step=1, from_phase="A", to_phase="B")
            tracer.log_evidence_update(step=2, evidence_count=5)
            tracer.log_trust_update(step=3, source="x", trust=0.5)
            tracer.log_decision(step=4, agent="a", command="c", source="s", confidence=0.5)
            tracer.close()

            for f in os.listdir(tmpdir):
                with open(os.path.join(tmpdir, f)) as fh:
                    for line in fh:
                        data = json.loads(line)
                        assert data["run_id"] == "rid42"


class TestCloseAndIdempotency:
    """Tests for close() behavior."""

    def test_close_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = DebugTracer(log_dir=tmpdir, run_id="close")
            tracer.log_stall(step=0, stall_score=0.1)
            tracer.close()
            tracer.close()  # Should not raise

    def test_writes_after_close_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = DebugTracer(log_dir=tmpdir, run_id="post")
            tracer.log_stall(step=0, stall_score=0.1)
            tracer.close()
            tracer.log_stall(step=1, stall_score=0.2)  # Should not crash
            assert tracer.entries_written == 1


class TestDisabledTracer:
    """Tests for disabled tracer — no side effects."""

    def test_no_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.join(tmpdir, "disabled_sub")
            tracer = DebugTracer(log_dir=sub, run_id="off", enabled=False)
            tracer.log_stall(step=0, stall_score=0.5)
            tracer.log_decision(step=1, agent="a", command="b", source="c", confidence=0.5)
            tracer.close()
            assert not os.path.exists(sub)

    def test_entries_written_stays_zero(self):
        tracer = DebugTracer(enabled=False)
        for i in range(10):
            tracer.log_stall(step=i, stall_score=0.1)
        assert tracer.entries_written == 0
        tracer.close()


class TestCommandTruncation:
    """Tests for long command truncation in log_decision."""

    def test_long_command_truncated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = DebugTracer(log_dir=tmpdir, run_id="trunc")
            long_cmd = "x" * 200
            tracer.log_decision(
                step=0, agent="a", command=long_cmd,
                source="s", confidence=0.5,
            )
            tracer.close()

            for f in os.listdir(tmpdir):
                with open(os.path.join(tmpdir, f)) as fh:
                    data = json.loads(fh.readline())
                    assert len(data["command"]) <= 100
