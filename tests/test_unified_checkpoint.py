"""Tests for core.checkpoints.unified_checkpoint.

Validates the unified checkpoint format: save/load, legacy migration,
apply methods, convenience builders, find_best, and helper functions.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for checkpoint files."""
    return tmp_path


@pytest.fixture
def sample_ppo_state() -> Dict[str, Any]:
    """Minimal PPO-like state dict for testing."""
    import torch

    return {
        "network_state_dict": {"layer.weight": torch.randn(5, 512)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {},
        "total_steps": 1000,
        "updates_done": 50,
        "entropy_coef": 0.01,
        "config": {"state_dim": 512, "action_dim": 5},
        "training_metrics": {"avg_reward": 42.0},
        "reward_norm": {
            "mean": 1.5,
            "var": 2.0,
            "count": 100,
            "return_mean": 3.0,
            "return_var": 4.0,
            "return_count": 50,
        },
        "adaptive_entropy": {
            "multiplier": 1.2,
            "consecutive_closeouts": 3,
            "consecutive_failures": 1,
        },
        "has_phase_gates": False,
        "sil_baseline": 10.0,
        "sil_count": 20,
        "use_symlog": True,
        "use_cosine_entropy": True,
        "ema_network_state": None,
        "clip_epsilon_current": 0.2,
        "clip_fraction_history": [0.1, 0.15],
        "entropy_below_count": 0,
    }


@pytest.fixture
def sample_ddqn_state() -> Dict[str, Any]:
    """Minimal DDQN-like state dict."""
    import torch

    return {
        "online_net": {"fc.weight": torch.randn(3, 3)},
        "target_net": {"fc.weight": torch.randn(3, 3)},
        "total_steps": 200,
    }


@pytest.fixture
def sample_sac_state() -> Dict[str, Any]:
    """Minimal SAC-like state dict."""
    import torch

    return {
        "actor": {"fc.weight": torch.randn(3, 3)},
        "critic": {"fc.weight": torch.randn(3, 3)},
        "step_count": 300,
    }


@pytest.fixture
def sample_agent_brain_state() -> Dict[str, Any]:
    """Minimal agent-brain state dict."""
    import torch

    return {
        "policy_network_state": {"layer.weight": torch.randn(4, 4)},
        "value_network_state": {"layer.weight": torch.randn(4, 4)},
    }


# ── Test: Save / Load round-trip ─────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_dir: Path, sample_ppo_state: Dict) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint(
            run_id="test_run",
            episode=5,
            source="test",
            ppo_state=sample_ppo_state,
            metadata={"scenario": "htb_easy"},
        )

        path = tmp_dir / "test_ckpt.pt"
        saved = ckpt.save(path)
        assert Path(saved).exists()

        loaded = UnifiedCheckpoint.load(path)
        assert loaded.format_version == 2
        assert loaded.run_id == "test_run"
        assert loaded.episode == 5
        assert loaded.source == "test"
        assert loaded.ppo_state is not None
        assert loaded.ppo_state["total_steps"] == 1000
        assert loaded.metadata["scenario"] == "htb_easy"

    def test_save_creates_parent_dirs(self, tmp_dir: Path) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint(source="test")
        deep_path = tmp_dir / "a" / "b" / "c" / "test.pt"
        ckpt.save(deep_path)
        assert deep_path.exists()

    def test_save_sets_timestamp_if_empty(self, tmp_dir: Path) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint(source="test")
        assert ckpt.timestamp == ""
        path = tmp_dir / "ts_test.pt"
        ckpt.save(path)
        assert ckpt.timestamp != ""
        assert "T" in ckpt.timestamp

    def test_load_nonexistent_raises(self, tmp_dir: Path) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        with pytest.raises(FileNotFoundError):
            UnifiedCheckpoint.load(tmp_dir / "nope.pt")

    def test_load_unrecognized_raises(self, tmp_dir: Path) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        path = tmp_dir / "bad.pt"
        torch.save({"random_key": 123}, str(path))
        with pytest.raises(ValueError, match="Unrecognized"):
            UnifiedCheckpoint.load(path)

    def test_empty_checkpoint_roundtrip(self, tmp_dir: Path) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint(source="empty_test")
        path = tmp_dir / "empty.pt"
        ckpt.save(path)

        loaded = UnifiedCheckpoint.load(path)
        assert loaded.ppo_state is None
        assert loaded.ddqn_states == {}
        assert loaded.sac_state is None
        assert loaded.agent_states == {}

    def test_all_algorithm_states_roundtrip(
        self,
        tmp_dir: Path,
        sample_ppo_state: Dict,
        sample_ddqn_state: Dict,
        sample_sac_state: Dict,
    ) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint(
            run_id="multi",
            episode=10,
            source="test",
            ppo_state=sample_ppo_state,
            ddqn_states={"RedAgent": sample_ddqn_state},
            sac_state=sample_sac_state,
            agent_states={"RedAgent": {"policy_network_state": {}}},
        )
        path = tmp_dir / "multi.pt"
        ckpt.save(path)

        loaded = UnifiedCheckpoint.load(path)
        assert loaded.ppo_state is not None
        assert "RedAgent" in loaded.ddqn_states
        assert loaded.sac_state is not None
        assert "RedAgent" in loaded.agent_states


# ── Test: Legacy migration ────────────────────────────────────────────────


class TestLegacyMigration:
    def test_legacy_ppo_format(self, tmp_dir: Path, sample_ppo_state: Dict) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        path = tmp_dir / "h200_20260101T120000Z_ep0005.pt"
        torch.save(sample_ppo_state, str(path))

        loaded = UnifiedCheckpoint.load(path)
        assert loaded.format_version == 1
        assert loaded.source == "legacy_ppo"
        assert loaded.ppo_state is not None
        assert loaded.episode == 5

    def test_legacy_ddqn_format(self, tmp_dir: Path, sample_ddqn_state: Dict) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        path = tmp_dir / "ddqn_RedAgent.pt"
        torch.save(sample_ddqn_state, str(path))

        loaded = UnifiedCheckpoint.load(path)
        assert loaded.source == "legacy_ddqn"
        assert "RedAgent" in loaded.ddqn_states

    def test_legacy_sac_format(self, tmp_dir: Path, sample_sac_state: Dict) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        path = tmp_dir / "sac_model.pt"
        torch.save(sample_sac_state, str(path))

        loaded = UnifiedCheckpoint.load(path)
        assert loaded.source == "legacy_sac"
        assert loaded.sac_state is not None

    def test_legacy_agent_brain_format(
        self, tmp_dir: Path, sample_agent_brain_state: Dict
    ) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        path = tmp_dir / "red_agent"
        torch.save(sample_agent_brain_state, str(path))

        loaded = UnifiedCheckpoint.load(path)
        assert loaded.source == "legacy_agent_brain"
        assert "RedAgent" in loaded.agent_states


# ── Test: Apply methods ──────────────────────────────────────────────────


class TestApplyMethods:
    def test_apply_ppo_calls_load_from_state_dict(self, sample_ppo_state: Dict) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        mock_ppo = MagicMock()
        ckpt = UnifiedCheckpoint(ppo_state=sample_ppo_state)
        result = ckpt.apply_ppo(mock_ppo)
        assert result is True
        mock_ppo.load_from_state_dict.assert_called_once_with(sample_ppo_state)

    def test_apply_ppo_none_state(self) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint()
        assert ckpt.apply_ppo(MagicMock()) is False

    def test_apply_ppo_handles_exception(self, sample_ppo_state: Dict) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        mock_ppo = MagicMock()
        mock_ppo.load_from_state_dict.side_effect = RuntimeError("bad weights")
        ckpt = UnifiedCheckpoint(ppo_state=sample_ppo_state)
        assert ckpt.apply_ppo(mock_ppo) is False

    def test_apply_ddqn(self) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        mock_ddqn = MagicMock()
        ckpt = UnifiedCheckpoint(ddqn_states={"RedAgent": {"data": 1}})
        assert ckpt.apply_ddqn(mock_ddqn, "RedAgent") is True
        mock_ddqn.load_state_dict.assert_called_once()

    def test_apply_ddqn_missing_agent(self) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint(ddqn_states={"RedAgent": {"data": 1}})
        assert ckpt.apply_ddqn(MagicMock(), "BlueAgent") is False

    def test_apply_agent_brains(self) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        brain_state = {
            "policy_network_state": {"w": torch.randn(2, 2)},
            "value_network_state": {"w": torch.randn(2, 2)},
        }

        mock_brain = MagicMock()
        mock_agent = MagicMock()
        mock_agent.brain = mock_brain
        mock_coach = MagicMock()
        mock_coach.agent = mock_agent

        ckpt = UnifiedCheckpoint(agent_states={"red": brain_state})
        restored = ckpt.apply_agent_brains({"red": mock_coach})
        assert restored >= 1

    def test_apply_agent_brains_empty(self) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint()
        assert ckpt.apply_agent_brains({}) == 0


# ── Test: Convenience builders ───────────────────────────────────────────


class TestBuilders:
    def test_from_ppo_agent(self) -> None:
        """Test _capture_ppo_state is used correctly."""
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        mock_ppo = MagicMock()
        # Set all attributes that _capture_ppo_state reads
        import torch

        mock_ppo.network.state_dict.return_value = {"w": torch.randn(2, 2)}
        mock_ppo.optimizer.state_dict.return_value = {"state": {}}
        mock_ppo.lr_scheduler.state_dict.return_value = {}
        mock_ppo.total_steps = 500
        mock_ppo.updates_done = 25
        mock_ppo.entropy_coef = 0.01
        mock_ppo.config = MagicMock()
        mock_ppo.config.use_symlog = True
        mock_ppo.config.use_cosine_entropy = True
        mock_ppo.config.clip_epsilon = 0.2
        mock_ppo.training_metrics = {}
        mock_ppo._reward_mean = 0.0
        mock_ppo._reward_var = 1.0
        mock_ppo._reward_count = 0
        mock_ppo._return_mean = 0.0
        mock_ppo._return_var = 1.0
        mock_ppo._return_count = 0
        mock_ppo._entropy_adaptive_multiplier = 1.0
        mock_ppo._consecutive_closeouts = 0
        mock_ppo._consecutive_failures = 0
        mock_ppo.network.has_phase_gates = False
        mock_ppo.sil_buffer._return_baseline = 0.0
        mock_ppo.sil_buffer._return_count = 0
        mock_ppo.ema_network = None
        mock_ppo._clip_fraction_history = []
        mock_ppo._entropy_below_count = 0

        ckpt = UnifiedCheckpoint.from_ppo_agent(
            mock_ppo, run_id="test", episode=3, source="test"
        )

        assert ckpt.ppo_state is not None
        assert ckpt.ppo_state["total_steps"] == 500
        assert ckpt.run_id == "test"
        assert ckpt.episode == 3

    def test_from_coaches(self) -> None:
        """Test from_coaches captures PPO and agent states."""
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        import torch

        mock_ppo = MagicMock()
        mock_ppo.network.state_dict.return_value = {"w": torch.randn(2, 2)}
        mock_ppo.optimizer.state_dict.return_value = {}
        mock_ppo.lr_scheduler.state_dict.return_value = {}
        mock_ppo.total_steps = 100
        mock_ppo.updates_done = 5
        mock_ppo.entropy_coef = 0.01
        mock_ppo.config = MagicMock()
        mock_ppo.config.use_symlog = True
        mock_ppo.config.use_cosine_entropy = True
        mock_ppo.config.clip_epsilon = 0.2
        mock_ppo.training_metrics = {}
        mock_ppo._reward_mean = 0.0
        mock_ppo._reward_var = 1.0
        mock_ppo._reward_count = 0
        mock_ppo._return_mean = 0.0
        mock_ppo._return_var = 1.0
        mock_ppo._return_count = 0
        mock_ppo._entropy_adaptive_multiplier = 1.0
        mock_ppo._consecutive_closeouts = 0
        mock_ppo._consecutive_failures = 0
        mock_ppo.network.has_phase_gates = False
        mock_ppo.sil_buffer._return_baseline = 0.0
        mock_ppo.sil_buffer._return_count = 0
        mock_ppo.ema_network = None
        mock_ppo._clip_fraction_history = []
        mock_ppo._entropy_below_count = 0

        mock_coach = MagicMock()
        mock_coach.ppo_agent = mock_ppo
        mock_coach.ddqn_macro = None
        mock_coach.sac_agent = None
        mock_coach.agent = MagicMock()
        mock_coach.agent.brain = None  # No brain networks

        ckpt = UnifiedCheckpoint.from_coaches(
            coaches={"red": mock_coach},
            run_id="coaches_test",
            episode=1,
        )

        assert ckpt.ppo_state is not None
        assert ckpt.ppo_state["total_steps"] == 100
        assert ckpt.run_id == "coaches_test"


# ── Test: find_best ──────────────────────────────────────────────────────


class TestFindBest:
    def test_find_best_unified_wins(self, tmp_dir: Path) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        # Create unified dir
        unified = tmp_dir / "unified"
        unified.mkdir()
        uni_path = unified / "ariaska_run1_ep0003.pt"
        UnifiedCheckpoint(run_id="run1", episode=3, source="test").save(uni_path)

        # Create distilled dir
        distilled = tmp_dir / "distilled"
        distilled.mkdir()
        dist_path = distilled / "h200_run1_ep0010.pt"
        torch.save(
            {"network_state_dict": {}, "total_steps": 0},
            str(dist_path),
        )

        best = UnifiedCheckpoint.find_best(directories=[unified, distilled])
        assert best is not None
        assert "ariaska_" in best.name  # unified wins even with lower episode

    def test_find_best_empty_dirs(self, tmp_dir: Path) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        empty = tmp_dir / "empty"
        empty.mkdir()
        assert UnifiedCheckpoint.find_best(directories=[empty]) is None

    def test_find_best_distilled_fallback(self, tmp_dir: Path) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        distilled = tmp_dir / "distilled"
        distilled.mkdir()
        path = distilled / "h200_run1_ep0005.pt"
        torch.save(
            {"network_state_dict": {}, "total_steps": 0},
            str(path),
        )

        best = UnifiedCheckpoint.find_best(directories=[distilled])
        assert best is not None
        assert "h200_" in best.name


# ── Test: Summary ─────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_ppo(self, sample_ppo_state: Dict) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint(
            run_id="run1", episode=5, source="gpu_distill", ppo_state=sample_ppo_state
        )
        s = ckpt.summary()
        assert "run=run1" in s
        assert "ep=5" in s
        assert "PPO" in s
        assert "1,000" in s  # total_steps formatted

    def test_summary_empty(self) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

        ckpt = UnifiedCheckpoint()
        s = ckpt.summary()
        assert "empty" in s


# ── Test: Migration ──────────────────────────────────────────────────────


class TestMigration:
    def test_migrate_directory(self, tmp_dir: Path, sample_ppo_state: Dict) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import migrate_directory

        # Create legacy files
        src = tmp_dir / "legacy"
        src.mkdir()
        torch.save(sample_ppo_state, str(src / "h200_run1_ep0001.pt"))
        torch.save(sample_ppo_state, str(src / "h200_run1_ep0002.pt"))

        dst = tmp_dir / "unified_output"
        count = migrate_directory(src, dst)
        assert count == 2
        assert len(list(dst.glob("ariaska_*.pt"))) == 2

    def test_migrate_skips_unified(self, tmp_dir: Path) -> None:
        from core.checkpoints.unified_checkpoint import UnifiedCheckpoint, migrate_directory

        src = tmp_dir / "src"
        src.mkdir()
        # File starting with ariaska_ should be skipped
        UnifiedCheckpoint(source="test").save(src / "ariaska_run1_ep0001.pt")

        dst = tmp_dir / "dst"
        count = migrate_directory(src, dst)
        assert count == 0


# ── Test: Helper functions ────────────────────────────────────────────────


class TestHelpers:
    def test_extract_run_ep_distilled(self) -> None:
        from core.checkpoints.unified_checkpoint import _extract_run_ep

        run_id, ep = _extract_run_ep(Path("h200_20260101T120000Z_ep0005.pt"))
        assert run_id == "20260101T120000Z"
        assert ep == 5

    def test_guess_agent_name_red(self) -> None:
        from core.checkpoints.unified_checkpoint import _guess_agent_name

        assert _guess_agent_name(Path("red_agent")) == "RedAgent"
        assert _guess_agent_name(Path("ddqn_RedAgent.pt")) == "RedAgent"
        assert _guess_agent_name(Path("some_BlueAgent_ckpt.pt")) == "BlueAgent"

    def test_capture_ppo_state(self) -> None:
        """Test that _capture_ppo_state builds the right dict shape."""
        import torch

        from core.checkpoints.unified_checkpoint import _capture_ppo_state

        mock_ppo = MagicMock()
        mock_ppo.network.state_dict.return_value = {"w": torch.randn(2, 2)}
        mock_ppo.optimizer.state_dict.return_value = {}
        mock_ppo.lr_scheduler.state_dict.return_value = {}
        mock_ppo.total_steps = 999
        mock_ppo.updates_done = 10
        mock_ppo.entropy_coef = 0.02
        mock_ppo.config = MagicMock()
        mock_ppo.config.use_symlog = True
        mock_ppo.config.use_cosine_entropy = True
        mock_ppo.config.clip_epsilon = 0.2
        mock_ppo.training_metrics = {}
        mock_ppo._reward_mean = 1.0
        mock_ppo._reward_var = 2.0
        mock_ppo._reward_count = 50
        mock_ppo._return_mean = 1.5
        mock_ppo._return_var = 2.5
        mock_ppo._return_count = 30
        mock_ppo._entropy_adaptive_multiplier = 0.9
        mock_ppo._consecutive_closeouts = 2
        mock_ppo._consecutive_failures = 0
        mock_ppo.network.has_phase_gates = True
        mock_ppo.sil_buffer._return_baseline = 5.0
        mock_ppo.sil_buffer._return_count = 10
        mock_ppo.ema_network = None
        mock_ppo._clip_fraction_history = [0.1]
        mock_ppo._entropy_below_count = 3

        state = _capture_ppo_state(mock_ppo)

        assert state["total_steps"] == 999
        assert state["updates_done"] == 10
        assert state["reward_norm"]["mean"] == 1.0
        assert state["adaptive_entropy"]["multiplier"] == 0.9
        assert state["has_phase_gates"] is True
        assert state["sil_baseline"] == 5.0
        assert state["clip_fraction_history"] == [0.1]
        assert state["entropy_below_count"] == 3

    def test_capture_agent_brain_with_brain(self) -> None:
        import torch

        from core.checkpoints.unified_checkpoint import _capture_agent_brain

        mock_agent = MagicMock()
        mock_agent.brain.policy_network.state_dict.return_value = {"w": torch.randn(2, 2)}
        mock_agent.brain.value_network.state_dict.return_value = {"v": torch.randn(2, 2)}

        result = _capture_agent_brain(mock_agent)
        assert result is not None
        assert "policy_network_state" in result
        assert "value_network_state" in result

    def test_capture_agent_brain_no_brain(self) -> None:
        from core.checkpoints.unified_checkpoint import _capture_agent_brain

        mock_agent = MagicMock(spec=[])  # No attributes at all
        result = _capture_agent_brain(mock_agent)
        assert result is None
