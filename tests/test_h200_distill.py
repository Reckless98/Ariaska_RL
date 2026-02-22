"""Tests for H200 online distillation script.

Uses FakeGPTManager / StubToolRunner / ARIASKA_DRY_RUN=1 per project rules.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure ARIASKA_DRY_RUN is set
os.environ["ARIASKA_DRY_RUN"] = "1"
os.environ.setdefault("FF_LOCAL_LLM", "0")


class TestAnnealController:
    """Test the anneal schedule controller."""

    def test_initial_progress_is_zero(self) -> None:
        from scripts.h200_run_distill_3h import AnnealController
        ctrl = AnnealController(total_duration_sec=3600)
        assert ctrl.progress < 0.01

    def test_phase_names(self) -> None:
        from scripts.h200_run_distill_3h import AnnealController
        ctrl = AnnealController(total_duration_sec=100)
        assert ctrl.phase_name == "heavy"

    def test_should_query_mentor_heavy_phase(self) -> None:
        from scripts.h200_run_distill_3h import AnnealController
        ctrl = AnnealController(total_duration_sec=10000)
        # In heavy phase, every step should query
        assert ctrl.should_query_mentor(0) is True
        assert ctrl.should_query_mentor(1) is True
        assert ctrl.should_query_mentor(5) is True

    def test_bc_coef_starts_high(self) -> None:
        from scripts.h200_run_distill_3h import AnnealController, BC_COEF_MAX
        ctrl = AnnealController(total_duration_sec=10000)
        assert ctrl.bc_coef() == pytest.approx(BC_COEF_MAX, abs=0.01)

    def test_kl_coef_starts_high(self) -> None:
        from scripts.h200_run_distill_3h import AnnealController, KL_COEF_MAX
        ctrl = AnnealController(total_duration_sec=10000)
        assert ctrl.kl_coef() == pytest.approx(KL_COEF_MAX, abs=0.01)

    def test_prior_alpha_starts_high(self) -> None:
        from scripts.h200_run_distill_3h import AnnealController
        ctrl = AnnealController(total_duration_sec=10000)
        assert ctrl.prior_alpha() >= 0.35  # Phase 51: max is 0.40 (was 0.50)

    def test_snapshot_has_required_keys(self) -> None:
        from scripts.h200_run_distill_3h import AnnealController
        ctrl = AnnealController(total_duration_sec=3600)
        snap = ctrl.snapshot()
        assert "progress" in snap
        assert "phase" in snap
        assert "bc_coef" in snap
        assert "kl_coef" in snap
        assert "prior_alpha" in snap


class TestMentorConfig:
    """Test MentorConfig dataclass."""

    def test_default_values(self) -> None:
        from scripts.h200_run_distill_3h import MentorConfig
        cfg = MentorConfig()
        assert cfg.base_url == "http://127.0.0.1:8192/v1"
        assert cfg.enabled is True
        assert cfg.timeout == 45.0

    def test_from_env(self) -> None:
        from scripts.h200_run_distill_3h import MentorConfig
        with patch.dict(os.environ, {"FF_LOCAL_LLM": "0"}):
            cfg = MentorConfig.from_env()
            assert cfg.enabled is False


class TestLocalMentorClient:
    """Test the HTTP mentor client."""

    def test_health_check_disabled(self) -> None:
        from scripts.h200_run_distill_3h import LocalMentorClient, MentorConfig
        cfg = MentorConfig(enabled=False)
        client = LocalMentorClient(cfg)
        assert client.health_check() is False

    def test_validate_response_valid(self) -> None:
        from scripts.h200_run_distill_3h import LocalMentorClient
        resp = {
            "command": "nmap -sV target",
            "confidence": 0.85,
            "reasoning": "Port scan",
            "teacher_action": 0,
            "action_probs": [0.4, 0.3, 0.2, 0.05, 0.05],
        }
        result = LocalMentorClient._validate_response(resp)
        assert result is not None
        assert result["command"] == "nmap -sV target"
        assert abs(sum(result["action_probs"]) - 1.0) < 0.01

    def test_validate_response_missing_command(self) -> None:
        from scripts.h200_run_distill_3h import LocalMentorClient
        resp = {"confidence": 0.5}
        result = LocalMentorClient._validate_response(resp)
        assert result is None

    def test_validate_response_normalizes_probs(self) -> None:
        from scripts.h200_run_distill_3h import LocalMentorClient
        resp = {
            "command": "test",
            "confidence": 0.5,
            "teacher_action": 0,
            "action_probs": [2, 2, 2, 2, 2],
        }
        result = LocalMentorClient._validate_response(resp)
        assert result is not None
        assert abs(sum(result["action_probs"]) - 1.0) < 0.01

    def test_validate_response_bad_probs_defaults(self) -> None:
        from scripts.h200_run_distill_3h import LocalMentorClient
        resp = {
            "command": "test",
            "confidence": 0.5,
            "teacher_action": 0,
            "action_probs": "bad",
        }
        result = LocalMentorClient._validate_response(resp)
        assert result is not None
        assert len(result["action_probs"]) == 5


class TestRunMetrics:
    """Test RunMetrics dataclass."""

    def test_avg_reward_empty(self) -> None:
        from scripts.h200_run_distill_3h import RunMetrics
        m = RunMetrics()
        assert m.avg_reward() == 0.0

    def test_avg_reward_window(self) -> None:
        from scripts.h200_run_distill_3h import RunMetrics
        m = RunMetrics()
        m.total_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert m.avg_reward(window=2) == pytest.approx(4.5)
        assert m.avg_reward() == pytest.approx(3.0)

    def test_wall_hours(self) -> None:
        from scripts.h200_run_distill_3h import RunMetrics
        m = RunMetrics()
        m.wall_start = 1000.0
        m.wall_end = 4600.0
        assert m.wall_hours == pytest.approx(1.0)


class TestParseTimeInterval:
    """Test CLI time interval parsing."""

    def test_minutes(self) -> None:
        from scripts.h200_run_distill_3h import _parse_time_interval
        assert _parse_time_interval("10m") == 600.0

    def test_seconds(self) -> None:
        from scripts.h200_run_distill_3h import _parse_time_interval
        assert _parse_time_interval("30s") == 30.0

    def test_hours(self) -> None:
        from scripts.h200_run_distill_3h import _parse_time_interval
        assert _parse_time_interval("1h") == 3600.0

    def test_raw_number(self) -> None:
        from scripts.h200_run_distill_3h import _parse_time_interval
        assert _parse_time_interval("120") == 120.0


class TestH200DistillationRunnerInit:
    """Test runner initialization (no actual training)."""

    def test_runner_creates_with_defaults(self) -> None:
        from scripts.h200_run_distill_3h import H200DistillationRunner
        runner = H200DistillationRunner(
            seed=42, max_hours=0.001, max_episodes=1,
            no_mentor=True, device="cpu",
        )
        assert runner.seed == 42
        assert runner.no_mentor is True
        assert runner.eval_only is False

    def test_runner_eval_mode(self) -> None:
        from scripts.h200_run_distill_3h import H200DistillationRunner
        runner = H200DistillationRunner(
            seed=42, max_hours=0.001, max_episodes=1,
            eval_only=True, no_mentor=True, device="cpu",
        )
        assert runner.eval_only is True

    def test_extract_family(self) -> None:
        from scripts.h200_run_distill_3h import H200DistillationRunner
        assert H200DistillationRunner._extract_family("nmap -sV target") == "nmap"
        assert H200DistillationRunner._extract_family("/usr/bin/gobuster dir") == "gobuster"
        assert H200DistillationRunner._extract_family("") == "unknown"


class TestActionToCommand:
    """Test action_idx → command mapping."""

    def test_fallback_recon(self) -> None:
        from scripts.h200_run_distill_3h import H200DistillationRunner
        runner = H200DistillationRunner(
            seed=42, max_hours=0.001, max_episodes=1,
            no_mentor=True, device="cpu",
        )
        cmd = runner._action_to_command(0, {"phase": "RECON"})
        assert "nmap" in cmd or "action_0" in cmd


class TestEpisodeResult:
    """Test EpisodeResult dataclass."""

    def test_creation(self) -> None:
        from scripts.h200_run_distill_3h import EpisodeResult
        r = EpisodeResult(
            episode_id=0, steps=10, total_reward=15.5,
            phase_reached="EXPLOITATION", discoveries=3,
            mentor_calls=5, ppo_updates=1,
        )
        assert r.episode_id == 0
        assert r.total_reward == 15.5
        assert r.phase_reached == "EXPLOITATION"
