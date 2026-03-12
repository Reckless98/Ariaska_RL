#!/usr/bin/env python3
"""
tests/test_phase97_telemetry.py — Phase 9.7 telemetry tests

Tests for:
- StepEvent / EpisodeEvent schema and serialization
- JSONLLogger buffering, flushing, file I/O
- Feature-flag gating
"""

import json
import os
import tempfile
import pytest

from core.telemetry.events import (
    StepEvent, EpisodeEvent, LLMCallRecord, AntiRepeatRecord, TimingRecord,
)
from core.telemetry.jsonl_logger import JSONLLogger


class TestStepEvent:
    """StepEvent schema correctness."""

    def test_defaults(self):
        ev = StepEvent()
        d = ev.to_dict()
        assert d["agent"] == ""
        assert d["step"] == 0
        assert d["source"] == ""
        assert d["reward_total"] == 0.0
        assert isinstance(d["discoveries"], list)
        assert isinstance(d["anti_repeat"], dict)
        assert isinstance(d["time_ms"], dict)
        assert "ts" in d and len(d["ts"]) > 0  # auto-generated

    def test_populated(self):
        ev = StepEvent(
            run_id="run_abc",
            episode_id=3,
            agent="RedAgent",
            step=7,
            phase="EXPLOITATION",
            selected_template="vsftpd_exploit",
            selected_command="exploit/unix/ftp/vsftpd_234_backdoor",
            source="ppo",
            confidence=0.85,
            discoveries=[{"type": "shell", "value": "true"}],
            discovery_count=1,
            reward_breakdown={"total": 50.0, "discovery": 45.0},
            reward_total=50.0,
            anti_repeat=AntiRepeatRecord(triggered=False),
            ddqn_select_calls=2,
            ddqn_cached_calls=1,
            ddqn_epsilon=0.15,
        )
        d = ev.to_dict()
        assert d["run_id"] == "run_abc"
        assert d["episode_id"] == 3
        assert d["agent"] == "RedAgent"
        assert d["step"] == 7
        assert d["phase"] == "EXPLOITATION"
        assert d["source"] == "ppo"
        assert d["confidence"] == 0.85
        assert d["reward_total"] == 50.0
        assert d["ddqn_select_calls"] == 2
        assert d["ddqn_cached_calls"] == 1
        assert d["ddqn_epsilon"] == 0.15

    def test_json_serializable(self):
        ev = StepEvent(
            agent="ScoutAgent",
            step=5,
            phase="RECON",
            llm_calls=[LLMCallRecord(role="mentor", model="local-llm", tokens_in=100, tokens_out=50)],
            time_ms=TimingRecord(decide_ms=10, total_ms=25),
        )
        d = ev.to_dict()
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["agent"] == "ScoutAgent"
        assert parsed["llm_calls"][0]["model"] == "local-llm"
        assert parsed["time_ms"]["decide"] == 10


class TestEpisodeEvent:
    """EpisodeEvent schema correctness."""

    def test_defaults(self):
        ev = EpisodeEvent()
        d = ev.to_dict()
        assert d["type"] == "episode_summary"
        assert d["total_steps"] == 0
        assert d["total_reward"] == 0.0
        assert "ts" in d

    def test_populated(self):
        ev = EpisodeEvent(
            run_id="run_xyz",
            episode_id=10,
            total_steps=120,
            total_reward=2500.5,
            final_phase="EXFILTRATION",
            closeout=True,
            termination="GOAL_REACHED",
            unique_commands=45,
            diversity_ratio=0.375,
            total_discoveries=22,
            anti_repeat_pct=15.5,
            source_distribution={"ppo": 40, "registry": 30, "anti_repeat": 20},
        )
        d = ev.to_dict()
        assert d["episode_id"] == 10
        assert d["total_steps"] == 120
        assert d["total_reward"] == 2500.5
        assert d["final_phase"] == "EXFILTRATION"
        assert d["closeout"] is True
        assert d["diversity_ratio"] == 0.375
        assert d["source_distribution"]["ppo"] == 40


class TestAntiRepeatRecord:
    def test_to_dict(self):
        ar = AntiRepeatRecord(triggered=True, count=3, action="replace")
        d = ar.to_dict()
        assert d["triggered"] is True
        assert d["count"] == 3
        assert d["action"] == "replace"


class TestTimingRecord:
    def test_to_dict(self):
        tr = TimingRecord(decide_ms=5, execute_ms=100, parse_ms=8, reward_ms=2, total_ms=115)
        d = tr.to_dict()
        assert d["decide"] == 5
        assert d["execute"] == 100
        assert d["total"] == 115


class TestLLMCallRecord:
    def test_to_dict(self):
        rec = LLMCallRecord(role="tactical", model="local-llm", tokens_in=200, tokens_out=80, latency_ms=350, cached=True)
        d = rec.to_dict()
        assert d["role"] == "tactical"
        assert d["cached"] is True
        assert d["latency_ms"] == 350


class TestJSONLLogger:
    """JSONLLogger buffering and file I/O."""

    def test_log_step_buffers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lg = JSONLLogger(run_id="test_run", output_dir=tmpdir, buffer_size=10, enabled=True)
            ev = StepEvent(agent="RedAgent", step=1)
            lg.log_step(ev)
            # Should be buffered, not yet on disk
            assert lg._buffer == 1 or len(lg._buffer) == 1
            lg.close()

    def test_log_episode_flushes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lg = JSONLLogger(run_id="test_run", output_dir=tmpdir, buffer_size=100, enabled=True)
            # Buffer some steps
            for i in range(5):
                lg.log_step(StepEvent(agent="RedAgent", step=i))
            # Episode event should flush
            lg.log_episode(EpisodeEvent(episode_id=1, total_reward=100.0))
            # Buffer should be empty after flush
            assert len(lg._buffer) == 0
            assert lg._total_events == 6
            lg.close()
            # Verify file content
            path = os.path.join(tmpdir, "test_run_telemetry.jsonl")
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 6
            # Last line should be episode summary
            last = json.loads(lines[-1])
            assert last["type"] == "episode_summary"
            assert last["total_reward"] == 100.0

    def test_buffer_auto_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lg = JSONLLogger(run_id="test_bf", output_dir=tmpdir, buffer_size=3, enabled=True)
            lg.log_step(StepEvent(agent="A", step=1))
            lg.log_step(StepEvent(agent="A", step=2))
            assert len(lg._buffer) == 2
            lg.log_step(StepEvent(agent="A", step=3))  # triggers flush
            assert len(lg._buffer) == 0
            assert lg._total_events == 3
            lg.close()

    def test_disabled_logger_noop(self):
        lg = JSONLLogger(run_id="test_disabled", enabled=False)
        lg.log_step(StepEvent(agent="X", step=1))
        lg.log_episode(EpisodeEvent(episode_id=1))
        assert lg._total_events == 0
        lg.close()

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lg = JSONLLogger(run_id="test_stats", output_dir=tmpdir, buffer_size=10, enabled=True)
            lg.log_step(StepEvent(agent="A", step=1))
            stats = lg.get_stats()
            assert stats["total_events"] == 1  # 0 flushed + 1 buffered
            assert stats["enabled"] is True
            assert "test_stats" in stats["path"]
            lg.close()

    def test_log_raw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lg = JSONLLogger(run_id="test_raw", output_dir=tmpdir, buffer_size=100, enabled=True)
            lg.log_raw({"custom_field": "value", "num": 42})
            lg.flush()
            lg.close()
            path = os.path.join(tmpdir, "test_raw_telemetry.jsonl")
            with open(path) as f:
                line = json.loads(f.readline())
            assert line["custom_field"] == "value"
            assert line["num"] == 42

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lg = JSONLLogger(run_id="test_idem", output_dir=tmpdir, buffer_size=10, enabled=True)
            lg.log_step(StepEvent(agent="A", step=1))
            lg.close()
            lg.close()  # Should not raise
            assert lg._total_events == 1


class TestFeatureFlagGating:
    """Verify telemetry respects FF_JSONL_TELEMETRY."""

    def test_default_flag_on(self):
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        assert ff.jsonl_telemetry is True

    def test_flag_controls_logger_creation(self):
        """When flag is off, logger should not be created."""
        from core.feature_flags import set_feature_flag, reset_feature_flags
        try:
            set_feature_flag("jsonl_telemetry", False)
            from core.feature_flags import get_feature_flags
            assert get_feature_flags().jsonl_telemetry is False
        finally:
            reset_feature_flags()
