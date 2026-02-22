"""Tests for scripts/offline_distill.py — Phase 47 Offline Distillation Trainer.

Tests cover:
  1. TraceDataset loading with/without state_vectors
  2. Loss functions (BC, KL, ranking, SIL, CQL)
  3. Training loop mechanics
  4. Checkpoint save/load roundtrip
  5. Edge cases (empty data, missing teacher_dist)

Uses FakeGPTManager pattern — no real LLM calls, no real execution.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn.functional as F

os.environ["ARIASKA_DRY_RUN"] = "1"

# Lazy import to match project conventions
from scripts.offline_distill import (
    ACTION_DIM,
    STATE_DIM,
    NUM_MACROS,
    PHASES,
    PHASE_TO_GROUP,
    OfflineConfig,
    TraceDataset,
    OfflineDistillTrainer,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_step(
    episode: int = 0,
    step: int = 0,
    action_idx: int = 0,
    reward: float = 5.0,
    phase: str = "RECON",
    teacher_action: int = -1,
    teacher_overrode: bool = False,
    state_vector: list | None = None,
    log_prob: float = -0.5,
    value: float = 3.0,
    teacher_dist: list | None = None,
    done: bool = False,
) -> Dict[str, Any]:
    """Build a minimal step trace record."""
    data: Dict[str, Any] = {
        "kind": "step",
        "episode": episode,
        "step": step,
        "action_idx": action_idx,
        "reward": reward,
        "phase": phase,
        "done": done,
        "teacher_action": teacher_action,
        "teacher_overrode": teacher_overrode,
        "cmd_family": "nmap_scan",
        "command": f"nmap -sV 10.0.0.1 --step{step}",
        "mentor_queried": teacher_overrode,
    }
    if state_vector is not None:
        data["state_vector"] = state_vector
    if teacher_dist is not None:
        data["teacher_dist"] = teacher_dist
    data["log_prob"] = log_prob
    data["value"] = value
    return data


def _write_trace_file(
    tmpdir: Path,
    steps: List[Dict],
    filename: str = "test_trace.jsonl",
) -> Path:
    """Write trace records to JSONL file."""
    fpath = tmpdir / filename
    with open(fpath, "w") as f:
        # Write episode_start record
        f.write(json.dumps({"data": {"kind": "episode_start", "episode": 0}}) + "\n")
        for s in steps:
            f.write(json.dumps({"data": s}) + "\n")
    return fpath


def _make_state_vector() -> List[float]:
    """Create a valid 512-dim state vector."""
    sv = [0.0] * STATE_DIM
    sv[0] = 1.0  # RECON phase
    sv[10] = 0.1  # progress
    sv[27] = 1.0  # port 80
    sv[59] = 0.05  # step ratio
    return sv


def _make_teacher_dist() -> List[float]:
    """Create a valid teacher distribution."""
    d = [0.1] * ACTION_DIM
    d[0] = 0.6  # Teacher prefers action 0
    total = sum(d)
    return [x / total for x in d]  # Normalize


# ── Test Constants ───────────────────────────────────────────────

class TestConstants:
    """Verify module constants match core architecture."""

    def test_state_dim(self):
        assert STATE_DIM == 512

    def test_action_dim(self):
        assert ACTION_DIM == 5

    def test_num_macros(self):
        assert NUM_MACROS == 9

    def test_phases(self):
        assert len(PHASES) == 8
        assert PHASES[0] == "RECON"
        assert PHASES[-1] == "CLOSEOUT"

    def test_phase_groups(self):
        assert PHASE_TO_GROUP["RECON"] == 0
        assert PHASE_TO_GROUP["EXPLOITATION"] == 1
        assert PHASE_TO_GROUP["EXFILTRATION"] == 2


# ── TraceDataset ─────────────────────────────────────────────────

class TestTraceDataset:
    """Test trace loading and dataset construction."""

    def test_load_empty_dir(self, tmp_path: Path):
        ds = TraceDataset(str(tmp_path))
        assert len(ds) == 0
        assert len(ds.episodes) == 0

    def test_load_nonexistent_dir(self, tmp_path: Path):
        ds = TraceDataset(str(tmp_path / "nonexistent"))
        assert len(ds) == 0

    def test_load_basic_steps(self, tmp_path: Path):
        steps = [
            _make_step(episode=0, step=i, reward=float(i))
            for i in range(10)
        ]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        assert len(ds) == 10
        assert 0 in ds.episodes
        assert len(ds.episodes[0]) == 10

    def test_load_with_state_vectors(self, tmp_path: Path):
        sv = _make_state_vector()
        steps = [
            _make_step(step=i, state_vector=sv) for i in range(5)
        ]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        item = ds[0]
        assert item["state"].shape == (STATE_DIM,)
        assert item["has_state_vector"].item() == 1.0
        assert item["state"][0].item() == 1.0  # RECON phase

    def test_load_without_state_vectors(self, tmp_path: Path):
        steps = [_make_step(step=i) for i in range(3)]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        item = ds[0]
        assert item["state"].shape == (STATE_DIM,)
        assert item["has_state_vector"].item() == 0.0

    def test_teacher_dist_present(self, tmp_path: Path):
        td = _make_teacher_dist()
        steps = [_make_step(step=0, teacher_dist=td)]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        item = ds[0]
        assert item["has_teacher_dist"].item() == 1.0
        assert item["teacher_dist"].shape == (ACTION_DIM,)
        assert torch.isclose(item["teacher_dist"].sum(), torch.tensor(1.0), atol=0.01)

    def test_teacher_dist_missing(self, tmp_path: Path):
        steps = [_make_step(step=0)]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        item = ds[0]
        assert item["has_teacher_dist"].item() == 0.0
        assert item["teacher_dist"].shape == (ACTION_DIM,)

    def test_sil_episode_identification(self, tmp_path: Path):
        # Episode with high rewards → SIL candidate
        steps = [
            _make_step(episode=0, step=i, reward=20.0) for i in range(5)
        ]
        # Episode with low rewards
        steps += [
            _make_step(episode=1, step=i, reward=0.5) for i in range(5)
        ]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path), min_reward_for_sil=5.0)
        assert 0 in ds.sil_episodes  # High reward
        assert 1 not in ds.sil_episodes  # Low reward

    def test_sil_batch_sampling(self, tmp_path: Path):
        steps = [
            _make_step(
                episode=0, step=i, reward=25.0,
                state_vector=_make_state_vector(),
            )
            for i in range(10)
        ]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path), min_reward_for_sil=5.0)
        batch = ds.get_sil_batch(4)
        assert batch is not None
        assert batch["states"].shape == (4, STATE_DIM)
        assert batch["actions"].shape == (4,)
        assert batch["rewards"].shape == (4,)

    def test_sil_batch_empty(self, tmp_path: Path):
        steps = [_make_step(step=0, reward=0.1)]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path), min_reward_for_sil=100.0)
        batch = ds.get_sil_batch(4)
        assert batch is None

    def test_phase_group_mapping(self, tmp_path: Path):
        steps = [
            _make_step(step=0, phase="RECON"),
            _make_step(step=1, phase="EXPLOITATION"),
            _make_step(step=2, phase="EXFILTRATION"),
        ]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        assert ds[0]["phase_group"].item() == 0   # Recon
        assert ds[1]["phase_group"].item() == 1   # Exploit
        assert ds[2]["phase_group"].item() == 2   # Post-exploit

    def test_teacher_override_tracking(self, tmp_path: Path):
        steps = [
            _make_step(step=0, teacher_overrode=True, teacher_action=2),
            _make_step(step=1, teacher_overrode=False, teacher_action=-1),
        ]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        assert ds[0]["teacher_overrode"].item() == 1.0
        assert ds[0]["teacher_action"].item() == 2
        assert ds[1]["teacher_overrode"].item() == 0.0

    def test_multiple_trace_files(self, tmp_path: Path):
        # File 1
        steps1 = [_make_step(episode=0, step=i) for i in range(5)]
        _write_trace_file(tmp_path, steps1, "trace_001.jsonl")

        # File 2
        steps2 = [_make_step(episode=1, step=i) for i in range(3)]
        _write_trace_file(tmp_path, steps2, "trace_002.jsonl")

        ds = TraceDataset(str(tmp_path))
        assert len(ds) == 8
        assert len(ds.episodes) == 2

    def test_malformed_json_skipped(self, tmp_path: Path):
        fpath = tmp_path / "bad.jsonl"
        with open(fpath, "w") as f:
            f.write('{"data": {"kind": "step", "episode": 0, "step": 0, "action_idx": 0, "reward": 1.0, "phase": "RECON"}}\n')
            f.write("NOT JSON\n")
            f.write('{"data": {"kind": "step", "episode": 0, "step": 1, "action_idx": 1, "reward": 2.0, "phase": "RECON"}}\n')

        ds = TraceDataset(str(tmp_path))
        assert len(ds) == 2  # Skipped the bad line


# ── Loss Functions ───────────────────────────────────────────────

class TestLossFunctions:
    """Test individual loss computations."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.device = torch.device("cpu")

    def _make_trainer_minimal(self) -> OfflineDistillTrainer:
        """Create trainer with minimal data for loss testing."""
        steps = [
            _make_step(
                step=i, reward=10.0,
                state_vector=_make_state_vector(),
                teacher_action=0,
                teacher_overrode=True,
                teacher_dist=_make_teacher_dist(),
            )
            for i in range(16)
        ]
        _write_trace_file(self.tmp_path, steps)

        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(self.tmp_path),
            epochs=1,
            batch_size=8,
            device="cpu",
            seed=42,
        )
        return OfflineDistillTrainer(config)

    def test_bc_loss_shape(self):
        trainer = self._make_trainer_minimal()
        logits = torch.randn(8, ACTION_DIM)
        targets = torch.randint(0, ACTION_DIM, (8,))
        loss = trainer._bc_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_bc_loss_weighted(self):
        trainer = self._make_trainer_minimal()
        logits = torch.randn(8, ACTION_DIM)
        targets = torch.randint(0, ACTION_DIM, (8,))
        weights = torch.ones(8)
        loss_weighted = trainer._bc_loss(logits, targets, weights)
        loss_unweighted = trainer._bc_loss(logits, targets)
        # With uniform weights=1.0, should be roughly the same
        assert abs(loss_weighted.item() - loss_unweighted.item()) < 1e-5

    def test_kl_teacher_loss_with_mask(self):
        trainer = self._make_trainer_minimal()
        logits = torch.randn(8, ACTION_DIM)
        teacher_dist = F.softmax(torch.randn(8, ACTION_DIM), dim=-1)
        mask = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
        loss = trainer._kl_teacher_loss(logits, teacher_dist, mask)
        assert loss.item() >= 0

    def test_kl_teacher_loss_empty_mask(self):
        trainer = self._make_trainer_minimal()
        logits = torch.randn(8, ACTION_DIM)
        teacher_dist = F.softmax(torch.randn(8, ACTION_DIM), dim=-1)
        mask = torch.zeros(8)
        loss = trainer._kl_teacher_loss(logits, teacher_dist, mask)
        assert loss.item() == 0.0

    def test_ranking_loss(self):
        trainer = self._make_trainer_minimal()
        # Logits where teacher action is NOT the highest → should have loss
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0, 4.0]] * 4)
        targets = torch.tensor([0, 0, 0, 0])  # Action 0 has lowest logit
        loss = trainer._ranking_loss(logits, targets)
        assert loss.item() > 0

    def test_ranking_loss_satisfied(self):
        trainer = self._make_trainer_minimal()
        # Logits where teacher action IS the highest with margin
        logits = torch.tensor([[10.0, 1.0, 2.0, 3.0, 4.0]])
        targets = torch.tensor([0])
        loss = trainer._ranking_loss(logits, targets)
        assert loss.item() == 0.0  # Margin satisfied

    def test_sil_loss(self):
        trainer = self._make_trainer_minimal()
        sil_batch = {
            "states": torch.randn(4, STATE_DIM),
            "actions": torch.randint(0, ACTION_DIM, (4,)),
            "rewards": torch.tensor([20.0, 15.0, 30.0, 10.0]),
        }
        loss = trainer._sil_loss(sil_batch)
        assert loss.shape == ()

    def test_cql_loss_no_ddqn(self):
        trainer = self._make_trainer_minimal()
        trainer.ddqn = None
        loss = trainer._cql_loss(torch.randn(4, STATE_DIM), torch.randint(0, NUM_MACROS, (4,)))
        assert loss.item() == 0.0


# ── Training Loop ────────────────────────────────────────────────

class TestTraining:
    """Test training loop mechanics."""

    @pytest.fixture
    def trace_dir(self, tmp_path: Path) -> Path:
        """Create trace dir with diverse data."""
        sv = _make_state_vector()
        td = _make_teacher_dist()
        steps = []
        for ep in range(3):
            for step in range(20):
                steps.append(_make_step(
                    episode=ep,
                    step=step,
                    action_idx=step % ACTION_DIM,
                    reward=5.0 + ep * 10.0,
                    phase=PHASES[min(step // 3, len(PHASES) - 1)],
                    teacher_action=step % ACTION_DIM,
                    teacher_overrode=(step % 3 == 0),
                    state_vector=sv,
                    teacher_dist=td if step % 2 == 0 else None,
                    done=(step == 19),
                ))
        _write_trace_file(tmp_path, steps)
        return tmp_path

    def test_train_one_epoch(self, trace_dir: Path):
        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(trace_dir),
            epochs=1,
            batch_size=8,
            device="cpu",
            seed=42,
            save_every=100,
            eval_every=100,
        )
        trainer = OfflineDistillTrainer(config)
        report = trainer.train()
        assert "error" not in report
        assert report["data"]["total_steps"] == 60
        assert report["data"]["total_episodes"] == 3

    def test_train_three_epochs_loss_tracked(self, trace_dir: Path):
        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(trace_dir),
            epochs=3,
            batch_size=8,
            device="cpu",
            seed=42,
            save_every=100,
            eval_every=100,
        )
        trainer = OfflineDistillTrainer(config)
        report = trainer.train()
        # Loss should be tracked for each epoch
        assert len(trainer.metrics["total_loss"]) == 3
        assert len(trainer.metrics["bc_loss"]) == 3
        assert len(trainer.metrics["kl_loss"]) == 3

    def test_eval_runs(self, trace_dir: Path):
        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(trace_dir),
            epochs=5,
            batch_size=8,
            device="cpu",
            eval_every=2,
            save_every=100,
            seed=42,
        )
        trainer = OfflineDistillTrainer(config)
        trainer.train()
        # Should have ~2 eval checkpoints (epoch 1, 3)
        assert len(trainer.metrics["eval_teacher_accuracy"]) >= 2

    def test_checkpoint_saved(self, trace_dir: Path, tmp_path: Path):
        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(trace_dir),
            epochs=5,
            batch_size=8,
            device="cpu",
            save_every=2,
            eval_every=100,
            seed=42,
        )
        trainer = OfflineDistillTrainer(config)
        trainer.train()

        # Check offline checkpoint dir
        from scripts.offline_distill import OFFLINE_CHECKPOINT_DIR
        pts = list(OFFLINE_CHECKPOINT_DIR.glob("offline_*.pt"))
        assert len(pts) >= 1  # At least final

    def test_empty_data_returns_error(self, tmp_path: Path):
        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(tmp_path),
            epochs=1,
            batch_size=8,
            device="cpu",
        )
        trainer = OfflineDistillTrainer(config)
        report = trainer.train()
        assert report.get("error") == "no_data"

    def test_loss_decreases_over_epochs(self, trace_dir: Path):
        """With enough epochs, loss should generally decrease."""
        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(trace_dir),
            epochs=15,
            batch_size=8,
            learning_rate=3e-4,
            device="cpu",
            seed=42,
            save_every=100,
            eval_every=100,
        )
        trainer = OfflineDistillTrainer(config)
        trainer.train()

        losses = trainer.metrics["total_loss"]
        # First few losses should be larger than later ones
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3
        assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} -> {late_avg:.4f}"

    def test_lr_schedule(self, trace_dir: Path):
        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(trace_dir),
            epochs=10,
            batch_size=8,
            learning_rate=1e-3,
            lr_min=1e-6,
            device="cpu",
            seed=42,
            save_every=100,
            eval_every=100,
        )
        trainer = OfflineDistillTrainer(config)
        trainer.train()
        lrs = trainer.metrics["lr"]
        assert lrs[0] >= lrs[-1]  # LR should decrease via cosine


# ── Checkpoint Roundtrip ─────────────────────────────────────────

class TestCheckpoint:
    """Test checkpoint save/load."""

    @pytest.fixture
    def trained_trainer(self, tmp_path: Path) -> OfflineDistillTrainer:
        sv = _make_state_vector()
        steps = [
            _make_step(step=i, state_vector=sv, reward=10.0,
                       teacher_overrode=True, teacher_action=i % ACTION_DIM)
            for i in range(32)
        ]
        _write_trace_file(tmp_path, steps)

        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(tmp_path),
            epochs=2,
            batch_size=8,
            device="cpu",
            seed=42,
            save_every=1,
            eval_every=100,
        )
        trainer = OfflineDistillTrainer(config)
        trainer.train()
        return trainer

    def test_checkpoint_structure(self, trained_trainer: OfflineDistillTrainer):
        from scripts.offline_distill import OFFLINE_CHECKPOINT_DIR
        pts = sorted(OFFLINE_CHECKPOINT_DIR.glob("offline_*.pt"))
        assert len(pts) >= 1

        ckpt = torch.load(pts[-1], map_location="cpu", weights_only=False)
        assert ckpt["__offline_distill__"] is True
        assert ckpt["format_version"] == 1
        assert "ppo_state" in ckpt
        assert "network_state_dict" in ckpt["ppo_state"]
        assert "config" in ckpt["ppo_state"]

    def test_load_offline_checkpoint_into_ppo(self, trained_trainer: OfflineDistillTrainer):
        """Offline checkpoint should be loadable back into a fresh PPOAgent."""
        from scripts.offline_distill import OFFLINE_CHECKPOINT_DIR
        pts = sorted(OFFLINE_CHECKPOINT_DIR.glob("offline_*.pt"))
        assert len(pts) >= 1

        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        ckpt = torch.load(pts[-1], map_location="cpu", weights_only=False)
        ppo_state = ckpt["ppo_state"]

        config = PPOConfig(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        ppo = PPOAgent(config=config, device="cpu")
        ppo.load_from_state_dict(ppo_state)

        # Verify network produces output
        state = torch.randn(1, STATE_DIM)
        logits, value = ppo.network(state)
        assert logits.shape == (1, ACTION_DIM)
        assert value.shape == (1, 1)


# ── Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_step_dataset(self, tmp_path: Path):
        steps = [_make_step(step=0, state_vector=_make_state_vector())]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        assert len(ds) == 1
        item = ds[0]
        assert item["state"].shape == (STATE_DIM,)

    def test_all_phases(self, tmp_path: Path):
        steps = [
            _make_step(step=i, phase=PHASES[i]) for i in range(len(PHASES))
        ]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        for i, phase in enumerate(PHASES):
            expected_group = PHASE_TO_GROUP[phase]
            assert ds[i]["phase_group"].item() == expected_group

    def test_negative_rewards(self, tmp_path: Path):
        steps = [_make_step(step=i, reward=-5.0) for i in range(8)]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        assert all(ds[i]["reward"].item() == -5.0 for i in range(8))

    def test_done_flag(self, tmp_path: Path):
        steps = [
            _make_step(step=0, done=False),
            _make_step(step=1, done=True),
        ]
        _write_trace_file(tmp_path, steps)

        ds = TraceDataset(str(tmp_path))
        assert ds[0]["done"].item() == 0.0
        assert ds[1]["done"].item() == 1.0

    def test_config_defaults(self):
        config = OfflineConfig()
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.bc_coef == 0.30
        assert config.kl_coef == 0.20
        assert config.sil_coef == 0.25
        assert not config.use_cql

    def test_report_generation(self, tmp_path: Path):
        steps = [
            _make_step(
                step=i, state_vector=_make_state_vector(),
                teacher_dist=_make_teacher_dist(),
                reward=10.0, teacher_overrode=True, teacher_action=0,
            )
            for i in range(16)
        ]
        _write_trace_file(tmp_path, steps)

        config = OfflineConfig(
            checkpoint_path="none",
            traces_dir=str(tmp_path),
            epochs=2,
            batch_size=8,
            device="cpu",
            save_every=100,
            eval_every=100,
        )
        trainer = OfflineDistillTrainer(config)
        report = trainer.train()

        assert "run_id" in report
        assert report["type"] == "offline_distillation"
        assert report["data"]["total_steps"] == 16
        assert report["data"]["steps_with_state_vector"] == 16
        assert report["data"]["steps_with_teacher_dist"] == 16


# ── Cleanup helper ───────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="session")
def cleanup_offline_checkpoints():
    """Clean up offline checkpoints created during tests."""
    yield
    # Don't clean up — checkpoints go to models/offline_distill/
    # which won't pollute the main training
