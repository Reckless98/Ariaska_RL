"""tests/test_phase48_modules.py — Phase 48: Tests for all 6 new modules.

Tests for:
  1. Enhanced HER (hindsight_replay.py)
  2. Reptile Meta-Learning (reptile_meta.py)
  3. Self-Play Training (self_play.py)
  4. Attention-Enhanced PPO (attention_ppo.py)
  5. Hyperparameter Sweep (scripts/hyperparam_sweep.py)
  6. Transfer Learning (transfer_learning.py)
"""
from __future__ import annotations

import copy
import math
import os
import pytest
import random
from unittest.mock import MagicMock, patch

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════════
# 1. ENHANCED HER TESTS
# ════════════════════════════════════════════════════════════════

class TestHERConfig:
    """Test HERConfig dataclass."""

    def test_default_config(self):
        from core.algorithms.hindsight_replay import HERConfig
        cfg = HERConfig()
        assert cfg.enabled is True
        assert cfg.k_future == 4
        assert cfg.strategy == "future"
        assert cfg.relabel_reward_success == 10.0
        assert cfg.relabel_reward_fail == -1.0
        assert cfg.discovery_relabel is True
        assert cfg.graduated_rewards is True
        assert cfg.priority_weight == 2.0
        assert cfg.memory_capacity == 5000

    def test_custom_config(self):
        from core.algorithms.hindsight_replay import HERConfig
        cfg = HERConfig(
            strategy="discovery",
            k_future=8,
            priority_weight=3.0,
            memory_capacity=1000,
        )
        assert cfg.strategy == "discovery"
        assert cfg.k_future == 8
        assert cfg.priority_weight == 3.0
        assert cfg.memory_capacity == 1000


class TestHERMemoryBank:
    """Test HERMemoryBank thread-safe buffer."""

    def test_add_and_size(self):
        from core.algorithms.hindsight_replay import HERMemoryBank
        bank = HERMemoryBank(capacity=100)
        assert bank.size == 0
        bank.add([{"reward": 1.0, "priority": 1.0}, {"reward": 2.0, "priority": 2.0}])
        assert bank.size == 2
        assert bank.total_stored == 2

    def test_capacity_limit(self):
        from core.algorithms.hindsight_replay import HERMemoryBank
        bank = HERMemoryBank(capacity=5)
        transitions = [{"reward": float(i), "priority": 1.0} for i in range(10)]
        bank.add(transitions)
        assert bank.size == 5
        assert bank.total_stored == 10

    def test_sample(self):
        from core.algorithms.hindsight_replay import HERMemoryBank
        bank = HERMemoryBank(capacity=100)
        transitions = [{"reward": float(i), "priority": 1.0} for i in range(20)]
        bank.add(transitions)
        samples = bank.sample(5)
        assert len(samples) == 5

    def test_sample_empty(self):
        from core.algorithms.hindsight_replay import HERMemoryBank
        bank = HERMemoryBank(capacity=100)
        assert bank.sample(5) == []

    def test_get_stats(self):
        from core.algorithms.hindsight_replay import HERMemoryBank
        bank = HERMemoryBank(capacity=100)
        bank.add([{"reward": 5.0, "priority": 2.0}])
        stats = bank.get_stats()
        assert stats["size"] == 1
        assert stats["avg_reward"] == 5.0
        assert stats["avg_priority"] == 2.0


class TestHindsightReplay:
    """Test enhanced HindsightReplay class."""

    @pytest.fixture
    def her(self):
        from core.algorithms.hindsight_replay import HindsightReplay, HERConfig
        return HindsightReplay(HERConfig(memory_capacity=500))

    @pytest.fixture
    def transitions(self):
        return [
            {"phase": "RECON", "command": "nmap", "reward": 0.5},
            {"phase": "ENUMERATION", "command": "gobuster", "reward": 0.3},
            {"phase": "EXPLOITATION", "command": "sqlmap", "reward": -1.0},
        ]

    def test_relabel_episode_basic(self, her, transitions):
        relabeled = her.relabel_episode(
            transitions, achieved_phase="ENUMERATION", target_phase="EXFILTRATION"
        )
        assert len(relabeled) == 3
        assert all(t["is_her"] for t in relabeled)
        assert all(t["relabeled_goal"] == "ENUMERATION" for t in relabeled)
        assert all(t["original_goal"] == "EXFILTRATION" for t in relabeled)

    def test_relabel_graduated_rewards(self, her, transitions):
        relabeled = her.relabel_episode(
            transitions, achieved_phase="EXPLOITATION", target_phase="EXFILTRATION"
        )
        # RECON (rank 0) should get partial credit, EXPLOITATION (rank 2) should get full success
        recon_t = relabeled[0]
        exploit_t = relabeled[2]
        assert exploit_t["reward"] == her.config.relabel_reward_success
        assert recon_t["reward"] < exploit_t["reward"]

    def test_relabel_binary_rewards(self):
        from core.algorithms.hindsight_replay import HindsightReplay, HERConfig
        cfg = HERConfig(graduated_rewards=False)
        her = HindsightReplay(cfg)
        transitions = [
            {"phase": "RECON", "command": "nmap", "reward": 0.5},
            {"phase": "EXPLOITATION", "command": "sqlmap", "reward": -1.0},
        ]
        relabeled = her.relabel_episode(
            transitions, achieved_phase="EXPLOITATION", target_phase="EXFILTRATION"
        )
        assert relabeled[0]["reward"] == cfg.relabel_reward_fail
        assert relabeled[1]["reward"] == cfg.relabel_reward_success

    def test_relabel_empty(self, her):
        assert her.relabel_episode([], "RECON", "EXFILTRATION") == []

    def test_relabel_invalid_phase(self, her, transitions):
        assert her.relabel_episode(transitions, "INVALID", "RECON") == []

    def test_relabel_priority_weighting(self, her, transitions):
        relabeled = her.relabel_episode(
            transitions, achieved_phase="EXPLOITATION", target_phase="EXFILTRATION"
        )
        # Closer to achieved phase should have higher priority
        for t in relabeled:
            assert "priority" in t
            assert t["priority"] >= 0

    def test_relabel_by_discovery(self, her, transitions):
        discoveries = ["port:22", "service:ssh", "credential:admin"]
        relabeled = her.relabel_by_discovery(transitions, discoveries)
        assert len(relabeled) == 3
        assert all(t["her_strategy"] == "discovery" for t in relabeled)
        # Later transitions should have higher reward (progress weighting)
        assert relabeled[-1]["reward"] >= relabeled[0]["reward"]

    def test_relabel_by_discovery_empty(self, her):
        assert her.relabel_by_discovery([], ["port:22"]) == []
        assert her.relabel_by_discovery([{"phase": "RECON"}], []) == []

    def test_process_episode_future_strategy(self, her, transitions):
        count = her.process_episode(
            transitions,
            target_phase="EXFILTRATION",
            achieved_phase="EXPLOITATION",
        )
        assert count > 0
        assert her.total_relabeled == count
        assert her.memory_bank.size > 0

    def test_process_episode_final_strategy(self, transitions):
        from core.algorithms.hindsight_replay import HindsightReplay, HERConfig
        cfg = HERConfig(strategy="final")
        her = HindsightReplay(cfg)
        count = her.process_episode(
            transitions,
            target_phase="EXFILTRATION",
            achieved_phase="EXPLOITATION",
        )
        assert count == 3  # All transitions relabeled once

    def test_process_episode_with_discoveries(self, her, transitions):
        discoveries = ["port:80", "service:http", "shell:bash"]
        count = her.process_episode(
            transitions,
            target_phase="EXFILTRATION",
            achieved_phase="EXPLOITATION",
            episode_discoveries=discoveries,
        )
        assert count > 3  # Phase + discovery relabeling

    def test_process_episode_disabled(self, transitions):
        from core.algorithms.hindsight_replay import HindsightReplay, HERConfig
        cfg = HERConfig(enabled=False)
        her = HindsightReplay(cfg)
        assert her.process_episode(transitions, "EXFILTRATION", "RECON") == 0

    def test_process_episode_already_succeeded(self, her, transitions):
        count = her.process_episode(
            transitions,
            target_phase="RECON",
            achieved_phase="EXPLOITATION",
        )
        # Should still generate discovery-based relabeling if discoveries given
        assert count == 0  # No phase relabeling when achieved >= target

    def test_sample_for_training(self, her, transitions):
        her.process_episode(transitions, "EXFILTRATION", "EXPLOITATION")
        samples = her.sample_for_training(batch_size=5)
        assert len(samples) > 0

    def test_get_stats(self, her, transitions):
        her.process_episode(transitions, "EXFILTRATION", "EXPLOITATION")
        stats = her.get_stats()
        assert stats["total_relabeled"] > 0
        assert stats["episodes_processed"] == 1
        assert "memory_bank" in stats


# ════════════════════════════════════════════════════════════════
# 2. REPTILE META-LEARNING TESTS
# ════════════════════════════════════════════════════════════════

class TestReptileConfig:
    """Test ReptileConfig defaults."""

    def test_defaults(self):
        from core.algorithms.reptile_meta import ReptileConfig
        cfg = ReptileConfig()
        assert cfg.enabled is True
        assert cfg.outer_lr == 0.1
        assert cfg.inner_steps == 5
        assert cfg.scenarios_per_step == 3
        assert cfg.warmup_steps == 100
        assert len(cfg.scenario_pool) > 10


class TestScenarioProfiles:
    """Test scenario profile definitions."""

    def test_profiles_exist(self):
        from core.algorithms.reptile_meta import SCENARIO_PROFILES
        assert len(SCENARIO_PROFILES) >= 18
        assert "generic_linux" in SCENARIO_PROFILES
        assert "htb_web_easy" in SCENARIO_PROFILES
        assert "htb_ad_windows" in SCENARIO_PROFILES

    def test_profile_structure(self):
        from core.algorithms.reptile_meta import SCENARIO_PROFILES
        for name, profile in SCENARIO_PROFILES.items():
            assert "phase_weights" in profile, f"{name} missing phase_weights"
            assert "typical_ports" in profile, f"{name} missing typical_ports"
            assert "difficulty" in profile, f"{name} missing difficulty"
            assert 0 <= profile["difficulty"] <= 1.0


class TestScenarioSampler:
    """Test curriculum-aware scenario sampling."""

    def test_sample_basic(self):
        from core.algorithms.reptile_meta import ScenarioSampler
        pool = ["a", "b", "c", "d", "e"]
        sampler = ScenarioSampler(pool, curriculum=False)
        samples = sampler.sample(3)
        assert len(samples) == 3
        assert all(s in pool for s in samples)

    def test_sample_curriculum_low_maturity(self):
        from core.algorithms.reptile_meta import ScenarioSampler, SCENARIO_PROFILES
        pool = list(SCENARIO_PROFILES.keys())
        sampler = ScenarioSampler(pool, curriculum=True)
        samples = sampler.sample(5, maturity=0.0)
        assert len(samples) == 5
        assert all(s in pool for s in samples)

    def test_record_result(self):
        from core.algorithms.reptile_meta import ScenarioSampler
        sampler = ScenarioSampler(["a", "b"], curriculum=False)
        sampler.record_result("a", True)
        sampler.record_result("a", True)
        sampler.record_result("b", False)
        stats = sampler.get_stats()
        assert stats["attempt_counts"]["a"] == 2
        assert stats["success_rates"]["a"] > 0


class TestReptileMeta:
    """Test Reptile meta-learning algorithm."""

    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    def test_should_run_disabled(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        cfg = ReptileConfig(enabled=False)
        reptile = ReptileMeta(cfg)
        assert reptile.should_run(1000) is False

    def test_should_run_warmup(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        cfg = ReptileConfig(warmup_steps=100)
        reptile = ReptileMeta(cfg)
        assert reptile.should_run(50) is False
        assert reptile.should_run(100) is True

    def test_outer_lr_annealing(self):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        cfg = ReptileConfig(outer_lr=0.1, anneal_outer_lr=0.01, cosine_anneal_steps=100)
        reptile = ReptileMeta(cfg)
        lr_start = reptile._current_outer_lr(0)
        lr_mid = reptile._current_outer_lr(50)
        lr_end = reptile._current_outer_lr(100)
        assert abs(lr_start - 0.1) < 0.01
        assert lr_mid < lr_start
        assert abs(lr_end - 0.01) < 0.01

    def test_meta_step(self, simple_model):
        from core.algorithms.reptile_meta import ReptileMeta, ReptileConfig
        cfg = ReptileConfig(
            inner_steps=2,
            scenarios_per_step=2,
            scenario_pool=["ms2_easy", "generic_linux"],
        )
        reptile = ReptileMeta(cfg)

        original_weights = copy.deepcopy(simple_model.state_dict())

        def inner_train(model, scenario, steps):
            # Simulate inner training: perturb weights
            for p in model.parameters():
                p.data += torch.randn_like(p.data) * 0.01
            return {"loss": 0.5, "reward": 1.0}

        stats = reptile.meta_step(
            model=simple_model,
            inner_train_fn=inner_train,
            global_step=200,
            maturity=0.3,
        )

        assert stats["meta_step"] == 1
        assert len(stats["scenarios"]) == 2
        assert stats["n_params_updated"] > 0

        # Weights should have changed (interpolated)
        for key in original_weights:
            new_w = simple_model.state_dict()[key]
            old_w = original_weights[key]
            # They should differ (Reptile interpolated)
            # But not by too much (outer_lr controls it)
            assert not torch.equal(new_w, old_w)

    def test_get_stats(self):
        from core.algorithms.reptile_meta import ReptileMeta
        reptile = ReptileMeta()
        stats = reptile.get_stats()
        assert stats["meta_steps_done"] == 0
        assert "config" in stats
        assert "sampler" in stats

    def test_get_scenario_profile(self):
        from core.algorithms.reptile_meta import ReptileMeta
        reptile = ReptileMeta()
        profile = reptile.get_scenario_profile("htb_web_easy")
        assert "phase_weights" in profile
        assert profile["difficulty"] == 0.3


# ════════════════════════════════════════════════════════════════
# 3. SELF-PLAY TESTS
# ════════════════════════════════════════════════════════════════

class TestELO:
    """Test ELO rating system."""

    def test_expected_score_equal(self):
        from core.algorithms.self_play import _expected_score
        score = _expected_score(1200, 1200)
        assert abs(score - 0.5) < 0.01

    def test_expected_score_higher_rated(self):
        from core.algorithms.self_play import _expected_score
        score = _expected_score(1600, 1200)
        assert score > 0.5

    def test_update_elo_win(self):
        from core.algorithms.self_play import _update_elo
        new_a, new_b = _update_elo(1200, 1200, 1.0)
        assert new_a > 1200
        assert new_b < 1200

    def test_update_elo_loss(self):
        from core.algorithms.self_play import _update_elo
        new_a, new_b = _update_elo(1200, 1200, 0.0)
        assert new_a < 1200
        assert new_b > 1200

    def test_update_elo_draw(self):
        from core.algorithms.self_play import _update_elo
        new_a, new_b = _update_elo(1200, 1200, 0.5)
        assert abs(new_a - 1200) < 0.01
        assert abs(new_b - 1200) < 0.01


class TestOpponentPool:
    """Test OpponentPool."""

    def test_add_and_size(self):
        from core.algorithms.self_play import OpponentPool
        pool = OpponentPool(max_size=5)
        pool.add_snapshot({"w": 1}, step=0)
        pool.add_snapshot({"w": 2}, step=1)
        assert pool.size == 2

    def test_eviction(self):
        from core.algorithms.self_play import OpponentPool
        pool = OpponentPool(max_size=3)
        for i in range(5):
            pool.add_snapshot({"w": i}, step=i)
        assert pool.size == 3

    def test_select_opponent(self):
        from core.algorithms.self_play import OpponentPool
        pool = OpponentPool(max_size=10)
        for i in range(5):
            pool.add_snapshot({"w": i}, step=i)
        opp = pool.select_opponent(current_elo=1200)
        assert opp is not None

    def test_select_best_response(self):
        from core.algorithms.self_play import OpponentPool
        pool = OpponentPool(max_size=10)
        pool.add_snapshot({"w": 1}, step=0)
        pool.add_snapshot({"w": 2}, step=1)
        # Update ELO for differentiation
        pool.update_elo("opponent_0001", 1200, won=True)
        best = pool.select_opponent(current_elo=1200, best_response=True)
        assert best is not None

    def test_get_rankings(self):
        from core.algorithms.self_play import OpponentPool
        pool = OpponentPool(max_size=10)
        pool.add_snapshot({"w": 1}, step=0)
        pool.add_snapshot({"w": 2}, step=1)
        rankings = pool.get_rankings()
        assert len(rankings) == 2

    def test_empty_pool(self):
        from core.algorithms.self_play import OpponentPool
        pool = OpponentPool()
        assert pool.select_opponent(1200) is None
        assert pool.get_best() is None


class TestSelfPlayTrainer:
    """Test SelfPlayTrainer."""

    @pytest.fixture
    def trainer(self):
        from core.algorithms.self_play import SelfPlayTrainer, SelfPlayConfig
        cfg = SelfPlayConfig(pool_size=10, snapshot_interval=2, min_pool_size=2)
        return SelfPlayTrainer(cfg)

    def test_save_snapshot(self, trainer):
        sid = trainer.save_snapshot({"w": 1}, step=0)
        assert sid.startswith("opponent_")

    def test_should_play_insufficient_pool(self, trainer):
        trainer.save_snapshot({"w": 1}, step=0)
        assert trainer.should_play(2) is False  # Only 1 opponent, need 2

    def test_should_play_ready(self, trainer):
        trainer.save_snapshot({"w": 1}, step=0)
        trainer.save_snapshot({"w": 2}, step=1)
        assert trainer.should_play(2) is True  # 2 opponents, interval 2

    def test_play_match(self, trainer):
        trainer.save_snapshot({"w": 1}, step=0)
        trainer.save_snapshot({"w": 2}, step=1)

        def mock_match(model, opp_weights, scenario):
            return (10.0, 5.0, 3)  # agent_score, opp_score, discoveries

        result = trainer.play_match(
            agent_model=None,
            match_fn=mock_match,
            scenario="generic_linux",
        )
        assert result is not None
        assert result.agent_won is True
        assert trainer.agent_elo > 1200  # Won, so ELO should increase

    def test_get_stats(self, trainer):
        stats = trainer.get_stats()
        assert stats["match_count"] == 0
        assert stats["agent_elo"] == 1200.0

    def test_get_rankings(self, trainer):
        trainer.save_snapshot({"w": 1}, step=0)
        rankings = trainer.get_rankings()
        assert rankings[0]["snapshot_id"] == "CURRENT_AGENT"


# ════════════════════════════════════════════════════════════════
# 4. ATTENTION PPO TESTS
# ════════════════════════════════════════════════════════════════

class TestAttentionPPOConfig:
    """Test AttentionPPOConfig."""

    def test_defaults(self):
        from core.algorithms.attention_ppo import AttentionPPOConfig
        cfg = AttentionPPOConfig()
        assert cfg.state_dim == 512
        assert cfg.attention_dim == 128
        assert cfg.num_heads == 4
        assert cfg.temporal_window == 16
        assert cfg.use_temporal is True
        assert cfg.use_cross_feature is True


class TestRelativePositionEncoding:
    """Test relative position encoding."""

    def test_forward_shape(self):
        from core.algorithms.attention_ppo import RelativePositionEncoding
        rpe = RelativePositionEncoding(max_len=16, num_heads=4)
        bias = rpe(seq_len=8)
        assert bias.shape == (4, 8, 8)


class TestTemporalAttention:
    """Test temporal attention module."""

    def test_forward(self):
        from core.algorithms.attention_ppo import TemporalAttention, AttentionPPOConfig
        cfg = AttentionPPOConfig(temporal_window=8, attention_dim=64, num_heads=4)
        attn = TemporalAttention(cfg)
        history = torch.randn(2, 8, 512)  # (B, T, state_dim)
        out = attn(history)
        assert out.shape == (2, 64)


class TestCrossFeatureAttention:
    """Test cross-feature attention module."""

    def test_forward(self):
        from core.algorithms.attention_ppo import CrossFeatureAttention, AttentionPPOConfig
        cfg = AttentionPPOConfig(attention_dim=64, cross_feature_heads=4)
        cross = CrossFeatureAttention(cfg)
        state = torch.randn(2, 512)
        out = cross(state)
        assert out.shape == (2, 64)


class TestAttentionEnhancedPPO:
    """Test AttentionEnhancedPPO wrapper."""

    @pytest.fixture
    def attn_ppo(self):
        from core.algorithms.attention_ppo import AttentionEnhancedPPO, AttentionPPOConfig
        cfg = AttentionPPOConfig(
            attention_dim=64, num_heads=4,
            temporal_window=8, state_dim=512,
        )
        return AttentionEnhancedPPO(cfg)

    def test_forward(self, attn_ppo):
        state = torch.randn(2, 512)
        features, gate = attn_ppo(state)
        assert features.shape == (2, 64)
        assert gate.shape == (2, 64)
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_push_state_and_history(self, attn_ppo):
        for i in range(10):
            attn_ppo.push_state(torch.randn(512))
        state = torch.randn(1, 512)
        history = attn_ppo.get_state_history(state)
        assert history.shape == (1, 8, 512)

    def test_reset_history(self, attn_ppo):
        attn_ppo.push_state(torch.randn(512))
        attn_ppo.reset_history()
        assert len(attn_ppo._state_history) == 0

    def test_gate_initial_bias(self, attn_ppo):
        # Gate should start mostly closed (preferring backbone)
        state = torch.randn(2, 512)
        _, gate = attn_ppo(state)
        mean_gate = gate.mean().item()
        # With bias=-2.0, sigmoid(-2) ≈ 0.12
        assert mean_gate < 0.5  # Should be biased toward backbone

    def test_get_stats(self, attn_ppo):
        stats = attn_ppo.get_stats()
        assert stats["attention_dim"] == 64
        assert stats["total_params"] > 0

    def test_no_attention_modules(self):
        from core.algorithms.attention_ppo import AttentionEnhancedPPO, AttentionPPOConfig
        cfg = AttentionPPOConfig(use_temporal=False, use_cross_feature=False)
        ppo = AttentionEnhancedPPO(cfg)
        state = torch.randn(2, 512)
        features, gate = ppo(state)
        assert features.shape[0] == 2


# ════════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER SWEEP TESTS
# ════════════════════════════════════════════════════════════════

class TestTrialConfig:
    """Test TrialConfig dataclass."""

    def test_defaults(self):
        from scripts.hyperparam_sweep import TrialConfig
        cfg = TrialConfig()
        assert cfg.clip_epsilon == 0.2
        assert cfg.learning_rate == 3e-4
        assert cfg.hidden_dims == [512, 512, 256]


class TestComputeObjective:
    """Test objective function."""

    def test_basic(self):
        from scripts.hyperparam_sweep import compute_objective
        metrics = {
            "unique_commands": 20,
            "diversity_ratio": 0.5,
            "total_discoveries": 5,
            "max_phase_reached": 3,
        }
        score = compute_objective(metrics)
        assert score > 0

    def test_zero_commands(self):
        from scripts.hyperparam_sweep import compute_objective
        metrics = {
            "unique_commands": 0,
            "diversity_ratio": 0.0,
            "total_discoveries": 0,
            "max_phase_reached": 0,
        }
        score = compute_objective(metrics)
        assert score == 0.0

    def test_phase_bonus(self):
        from scripts.hyperparam_sweep import compute_objective
        low = compute_objective({"unique_commands": 10, "diversity_ratio": 0.5,
                                  "total_discoveries": 0, "max_phase_reached": 0})
        high = compute_objective({"unique_commands": 10, "diversity_ratio": 0.5,
                                   "total_discoveries": 0, "max_phase_reached": 5})
        assert high > low


class TestSweepConfig:
    """Test SweepConfig."""

    def test_defaults(self):
        from scripts.hyperparam_sweep import SweepConfig
        cfg = SweepConfig()
        assert cfg.n_trials == 50
        assert cfg.direction == "maximize"


class TestHyperparamSweep:
    """Test HyperparamSweep class."""

    def test_init(self):
        from scripts.hyperparam_sweep import HyperparamSweep
        sweep = HyperparamSweep()
        assert sweep._best_config is None
        assert sweep._results == []

    def test_get_results_empty(self):
        from scripts.hyperparam_sweep import HyperparamSweep
        sweep = HyperparamSweep()
        assert sweep.get_results() == []
        assert sweep.get_best_config() is None


# ════════════════════════════════════════════════════════════════
# 6. TRANSFER LEARNING TESTS
# ════════════════════════════════════════════════════════════════

class TestTransferConfig:
    """Test TransferConfig."""

    def test_defaults(self):
        from core.algorithms.transfer_learning import TransferConfig
        cfg = TransferConfig()
        assert cfg.enabled is True
        assert cfg.mode == "progressive"
        assert cfg.backbone_lr_scale == 0.1
        assert cfg.head_lr_scale == 1.0

    def test_difficulty_tiers(self):
        from core.algorithms.transfer_learning import DIFFICULTY_TIERS
        assert "easy" in DIFFICULTY_TIERS
        assert "medium" in DIFFICULTY_TIERS
        assert "hard" in DIFFICULTY_TIERS
        assert len(DIFFICULTY_TIERS["easy"]) >= 3


class TestLayerFreezer:
    """Test LayerFreezer."""

    @pytest.fixture
    def model(self):
        """Create a mock PPOActorCritic-like model."""
        model = nn.Module()
        model.input_proj = nn.Linear(512, 256)
        model.input_norm = nn.LayerNorm(256)
        model.shared_backbone = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        model.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
        model.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        return model

    def test_freeze_group(self, model):
        from core.algorithms.transfer_learning import LayerFreezer
        freezer = LayerFreezer(model)
        count = freezer.freeze_group("backbone")
        assert count > 0
        # Check params are actually frozen
        for p in model.shared_backbone.parameters():
            assert p.requires_grad is False

    def test_unfreeze_group(self, model):
        from core.algorithms.transfer_learning import LayerFreezer
        freezer = LayerFreezer(model)
        freezer.freeze_group("backbone")
        freezer.unfreeze_group("backbone")
        for p in model.shared_backbone.parameters():
            assert p.requires_grad is True

    def test_freeze_all_except(self, model):
        from core.algorithms.transfer_learning import LayerFreezer
        freezer = LayerFreezer(model)
        freezer.freeze_all_except(["actor"])
        # Actor should be trainable
        for p in model.actor.parameters():
            assert p.requires_grad is True
        # Backbone should be frozen
        for p in model.shared_backbone.parameters():
            assert p.requires_grad is False

    def test_frozen_ratio(self, model):
        from core.algorithms.transfer_learning import LayerFreezer
        freezer = LayerFreezer(model)
        assert freezer.frozen_ratio == 0.0
        freezer.freeze_group("backbone")
        assert freezer.frozen_ratio > 0.0

    def test_unfreeze_all(self, model):
        from core.algorithms.transfer_learning import LayerFreezer
        freezer = LayerFreezer(model)
        freezer.freeze_all_except([])
        freezer.unfreeze_all()
        assert freezer.frozen_ratio == 0.0

    def test_param_groups(self, model):
        from core.algorithms.transfer_learning import LayerFreezer
        freezer = LayerFreezer(model)
        groups = freezer.get_param_groups_for_optimizer(
            base_lr=1e-3, backbone_scale=0.1, head_scale=1.0
        )
        assert len(groups) > 0
        # Check that backbone has lower LR
        for g in groups:
            if g.get("name") == "backbone":
                assert g["lr"] == pytest.approx(1e-4)

    def test_get_stats(self, model):
        from core.algorithms.transfer_learning import LayerFreezer
        freezer = LayerFreezer(model)
        stats = freezer.get_stats()
        assert stats["total_params"] > 0
        assert stats["frozen_ratio"] == 0.0


class TestTransferLearning:
    """Test TransferLearning manager."""

    @pytest.fixture
    def model(self):
        model = nn.Module()
        model.input_proj = nn.Linear(512, 256)
        model.input_norm = nn.LayerNorm(256)
        model.shared_backbone = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        model.actor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5)
        )
        model.critic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        return model

    def test_setup_progressive(self, model):
        from core.algorithms.transfer_learning import TransferLearning, TransferConfig
        cfg = TransferConfig(mode="progressive")
        tl = TransferLearning(cfg)
        tl.setup(model)
        assert tl._phase == "progressive"
        # Actor and critic should be trainable
        for p in model.actor.parameters():
            assert p.requires_grad is True

    def test_setup_freeze(self, model):
        from core.algorithms.transfer_learning import TransferLearning, TransferConfig
        cfg = TransferConfig(mode="freeze", freeze_groups=["input", "backbone"])
        tl = TransferLearning(cfg)
        tl.setup(model)
        # Input and backbone should be frozen
        for p in model.input_proj.parameters():
            assert p.requires_grad is False
        for p in model.shared_backbone.parameters():
            assert p.requires_grad is False

    def test_setup_pretrain(self, model):
        from core.algorithms.transfer_learning import TransferLearning, TransferConfig
        cfg = TransferConfig(mode="pretrain")
        tl = TransferLearning(cfg)
        tl.setup(model)
        # Everything should be trainable
        for p in model.parameters():
            assert p.requires_grad is True

    def test_progressive_unfreezing(self, model):
        from core.algorithms.transfer_learning import TransferLearning, TransferConfig
        cfg = TransferConfig(
            mode="progressive",
            unfreeze_schedule={
                "critic": 0,
                "actor": 0,
                "residual": 10,
                "backbone": 20,
                "input": 30,
            }
        )
        tl = TransferLearning(cfg)
        tl.setup(model)

        # At step 0, should not need to unfreeze (critic/actor already free)
        assert not tl.should_unfreeze(0)

        # At step 20, backbone should unfreeze
        assert tl.should_unfreeze(20)
        unfrozen = tl.step_unfreeze(20)
        assert "backbone" in unfrozen

    def test_get_param_groups(self, model):
        from core.algorithms.transfer_learning import TransferLearning, TransferConfig
        cfg = TransferConfig(mode="fine_tune")
        tl = TransferLearning(cfg)
        tl.setup(model)
        groups = tl.get_param_groups(base_lr=1e-3)
        assert len(groups) > 0

    def test_save_checkpoint(self, model, tmp_path):
        from core.algorithms.transfer_learning import TransferLearning, TransferConfig
        cfg = TransferConfig(mode="pretrain", checkpoint_dir=str(tmp_path))
        tl = TransferLearning(cfg)
        tl.setup(model)
        path = tl.save_checkpoint(model, step=100, scenario="test")
        assert os.path.exists(path)

    def test_load_pretrained(self, model, tmp_path):
        from core.algorithms.transfer_learning import TransferLearning, TransferConfig
        cfg = TransferConfig(checkpoint_dir=str(tmp_path))
        tl = TransferLearning(cfg)

        # Save and load
        ckpt_path = str(tmp_path / "test.pt")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        model2 = copy.deepcopy(model)
        # Perturb weights
        for p in model2.parameters():
            p.data += torch.randn_like(p.data)

        info = tl.load_pretrained(model2, ckpt_path)
        assert info["keys_transferred"] > 0
        assert info["keys_missing"] == 0

    def test_get_stats(self, model):
        from core.algorithms.transfer_learning import TransferLearning, TransferConfig
        cfg = TransferConfig(mode="progressive")
        tl = TransferLearning(cfg)
        tl.setup(model)
        stats = tl.get_stats()
        assert stats["mode"] == "progressive"
        assert stats["phase"] == "progressive"
        assert "freezer" in stats
