"""tests/test_phase49_modules.py — Phase 49: Tests for all 10 new modules.

Tests for:
  1. Ensemble Voting (ensemble_voting.py)
  2. Auto Curriculum Learning (auto_curriculum.py)
  3. LLM Reward Shaping (llm_reward_shaper.py)
  4. Options Framework / HRL (options_hrl.py)
  5. Offline RL CQL/IQL (offline_rl.py)
  6. Goal-Conditioned RL (goal_conditioned.py)
  7. Multi-Agent Communication (agent_comm.py)
  8. World Model DreamerV3 (world_model.py)
  9. Population-Based Training (pbt.py)
  10. Enhanced Contrastive State (contrastive_state.py)
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════════
# 1. ENSEMBLE VOTING TESTS
# ════════════════════════════════════════════════════════════════


class TestEnsembleConfig:
    def test_default_config(self):
        from core.algorithms.ensemble_voting import EnsembleConfig
        cfg = EnsembleConfig()
        assert cfg.n_heads == 5
        assert cfg.state_dim == 512
        assert cfg.action_dim == 5
        assert cfg.voting == "weighted"
        assert cfg.temperature == 1.0
        assert cfg.enabled is True

    def test_custom_config(self):
        from core.algorithms.ensemble_voting import EnsembleConfig
        cfg = EnsembleConfig(n_heads=7, voting="boltzmann")
        assert cfg.n_heads == 7
        assert cfg.voting == "boltzmann"


class TestActorHead:
    def test_forward_shape(self):
        from core.algorithms.ensemble_voting import ActorHead
        head = ActorHead(input_dim=256, action_dim=5)
        x = torch.randn(4, 256)
        logits = head(x)
        assert logits.shape == (4, 5)

    def test_output_varies(self):
        from core.algorithms.ensemble_voting import ActorHead
        head = ActorHead(input_dim=256, action_dim=5)
        x = torch.randn(1, 256)
        logits = head(x)
        assert logits.abs().sum() > 0


class TestEnsembleVoter:
    def test_creation(self):
        from core.algorithms.ensemble_voting import EnsembleVoter, EnsembleConfig
        cfg = EnsembleConfig(n_heads=4)
        voter = EnsembleVoter(cfg)
        assert len(voter.heads) == 4

    def test_vote_weighted(self):
        from core.algorithms.ensemble_voting import EnsembleVoter, EnsembleConfig
        cfg = EnsembleConfig(n_heads=3, voting="weighted", hidden_dim=512)
        voter = EnsembleVoter(cfg)
        features = torch.randn(1, 512)
        actions, info = voter.vote(features)
        assert actions.shape[0] == 1
        assert "agreement" in info
        assert "disagreement_bonus" in info
        assert "strategy" in info

    def test_vote_majority(self):
        from core.algorithms.ensemble_voting import EnsembleVoter, EnsembleConfig
        cfg = EnsembleConfig(n_heads=5, voting="majority", hidden_dim=512)
        voter = EnsembleVoter(cfg)
        features = torch.randn(1, 512)
        actions, info = voter.vote(features)
        assert actions.shape[0] == 1
        assert info["strategy"] == "majority"

    def test_vote_ranked(self):
        from core.algorithms.ensemble_voting import EnsembleVoter, EnsembleConfig
        cfg = EnsembleConfig(n_heads=3, voting="ranked", hidden_dim=512)
        voter = EnsembleVoter(cfg)
        features = torch.randn(1, 512)
        actions, info = voter.vote(features)
        assert actions.shape[0] == 1

    def test_vote_boltzmann(self):
        from core.algorithms.ensemble_voting import EnsembleVoter, EnsembleConfig
        cfg = EnsembleConfig(n_heads=3, voting="boltzmann", hidden_dim=512)
        voter = EnsembleVoter(cfg)
        features = torch.randn(1, 512)
        actions, info = voter.vote(features)
        assert actions.shape[0] == 1

    def test_diversity_loss(self):
        from core.algorithms.ensemble_voting import EnsembleVoter, EnsembleConfig
        cfg = EnsembleConfig(n_heads=3, hidden_dim=512)
        voter = EnsembleVoter(cfg)
        features = torch.randn(8, 512)
        loss = voter.compute_diversity_loss(features)
        assert loss.shape == ()
        assert loss.item() >= -1e-5  # Allow tiny floating-point undershoot

    def test_vote_batch(self):
        from core.algorithms.ensemble_voting import EnsembleVoter, EnsembleConfig
        cfg = EnsembleConfig(n_heads=3, hidden_dim=512)
        voter = EnsembleVoter(cfg)
        features = torch.randn(4, 512)
        actions, info = voter.vote(features)
        assert actions.shape[0] == 4

    def test_get_stats(self):
        from core.algorithms.ensemble_voting import EnsembleVoter
        voter = EnsembleVoter()
        stats = voter.get_stats()
        assert isinstance(stats, dict)


# ════════════════════════════════════════════════════════════════
# 2. AUTO CURRICULUM LEARNING TESTS
# ════════════════════════════════════════════════════════════════


class TestCurriculumConfig:
    def test_defaults(self):
        from core.algorithms.auto_curriculum import CurriculumConfig
        cfg = CurriculumConfig()
        assert cfg.mastery_threshold == 0.85
        assert cfg.zpd_low < cfg.zpd_high

    def test_custom(self):
        from core.algorithms.auto_curriculum import CurriculumConfig
        cfg = CurriculumConfig(mastery_threshold=0.9, zpd_low=0.25)
        assert cfg.mastery_threshold == 0.9
        assert cfg.zpd_low == 0.25


class TestScenarioStats:
    def test_competence(self):
        from core.algorithms.auto_curriculum import ScenarioStats
        stats = ScenarioStats(name="test", difficulty=0.5)
        c = stats.competence
        assert 0 < c < 1

    def test_update(self):
        from core.algorithms.auto_curriculum import ScenarioStats
        stats = ScenarioStats(name="test", difficulty=0.5)
        initial = stats.competence
        stats.alpha += 1  # Simulate success update
        assert stats.competence > initial

    def test_win_rate(self):
        from core.algorithms.auto_curriculum import ScenarioStats
        stats = ScenarioStats(name="test", difficulty=0.5, attempts=10, successes=7)
        assert abs(stats.win_rate - 0.7) < 0.01


class TestCurriculumScheduler:
    def test_register_scenario(self):
        from core.algorithms.auto_curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        sched.register_scenario("port_scan", difficulty=0.2)
        sched.register_scenario("exploit", difficulty=0.7)
        assert len(sched._scenarios) == 2

    def test_next_scenario(self):
        from core.algorithms.auto_curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        sched.register_scenario("port_scan", difficulty=0.3)
        sched.register_scenario("exploit", difficulty=0.7)
        sched.register_scenario("privesc", difficulty=0.9)
        result = sched.next_scenario()
        assert result.name in ["port_scan", "exploit", "privesc"]

    def test_update_outcome(self):
        from core.algorithms.auto_curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        sched.register_scenario("port_scan", difficulty=0.2)
        result = sched.update("port_scan", reward=5.0, success=True, steps=10)
        assert "competence" in result

    def test_mastery_report(self):
        from core.algorithms.auto_curriculum import CurriculumScheduler, CurriculumConfig
        cfg = CurriculumConfig(mastery_threshold=0.6)
        sched = CurriculumScheduler(cfg)
        sched.register_scenario("easy", difficulty=0.1)
        # Simulate many successes via alpha
        sched._scenarios["easy"].alpha += 20
        report = sched.get_mastery_report()
        assert "mastered_names" in report
        assert "easy" in report["mastered_names"]

    def test_empty_raises(self):
        from core.algorithms.auto_curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        with pytest.raises(RuntimeError):
            sched.next_scenario()

    def test_register_scenarios_batch(self):
        from core.algorithms.auto_curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        sched.register_scenarios([("a", 0.1), ("b", 0.5), ("c", 0.9)])
        assert len(sched._scenarios) == 3

    def test_reset(self):
        from core.algorithms.auto_curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        sched.register_scenario("test", difficulty=0.5)
        sched.update("test", reward=10.0, success=True, steps=5)
        sched.reset()
        # reset zeroes stats but keeps scenario registrations
        assert len(sched._scenarios) == 1
        assert sched._scenarios["test"].attempts == 0
        assert sched._scenarios["test"].successes == 0


# ════════════════════════════════════════════════════════════════
# 3. LLM REWARD SHAPING TESTS
# ════════════════════════════════════════════════════════════════


class TestRewardShaperConfig:
    def test_defaults(self):
        from core.algorithms.llm_reward_shaper import RewardShaperConfig
        cfg = RewardShaperConfig()
        assert cfg.enabled is True
        assert 0 < cfg.alpha_init <= 1.0
        assert cfg.alpha_min >= 0
        assert cfg.anneal_episodes > 0


class TestLLMRewardShaper:
    def test_creation(self):
        from core.algorithms.llm_reward_shaper import LLMRewardShaper
        shaper = LLMRewardShaper()
        assert shaper is not None

    def test_alpha_property(self):
        from core.algorithms.llm_reward_shaper import LLMRewardShaper, RewardShaperConfig
        cfg = RewardShaperConfig(alpha_init=0.5, alpha_min=0.05, anneal_episodes=10)
        shaper = LLMRewardShaper(config=cfg)
        alpha_start = shaper.alpha
        assert abs(alpha_start - 0.5) < 0.01

    def test_alpha_anneals(self):
        from core.algorithms.llm_reward_shaper import LLMRewardShaper, RewardShaperConfig
        cfg = RewardShaperConfig(alpha_init=0.5, alpha_min=0.05, anneal_episodes=10)
        shaper = LLMRewardShaper(config=cfg)
        alpha_start = shaper.alpha
        shaper._episode_count = 10  # Simulate full anneal
        alpha_end = shaper.alpha
        assert alpha_end <= alpha_start

    def test_get_stats(self):
        from core.algorithms.llm_reward_shaper import LLMRewardShaper
        shaper = LLMRewardShaper()
        stats = shaper.get_stats()
        assert isinstance(stats, dict)

    def test_reset(self):
        from core.algorithms.llm_reward_shaper import LLMRewardShaper
        shaper = LLMRewardShaper()
        shaper._episode_count = 5
        shaper.reset()
        assert shaper._episode_count == 0


# ════════════════════════════════════════════════════════════════
# 4. OPTIONS FRAMEWORK / HRL TESTS
# ════════════════════════════════════════════════════════════════


class TestOptionConfig:
    def test_defaults(self):
        from core.algorithms.options_hrl import OptionConfig
        cfg = OptionConfig(name="test_option")
        assert cfg.max_steps == 10
        assert cfg.completion_bonus == 5.0

    def test_custom(self):
        from core.algorithms.options_hrl import OptionConfig
        cfg = OptionConfig(name="scan", max_steps=20, valid_phases=["RECON"])
        assert cfg.name == "scan"
        assert cfg.max_steps == 20
        assert "RECON" in cfg.valid_phases


class TestHRLConfig:
    def test_defaults(self):
        from core.algorithms.options_hrl import HRLConfig
        cfg = HRLConfig()
        assert cfg.state_dim == 512
        assert cfg.n_primitive_actions == 5
        assert cfg.enabled is True


class TestOption:
    def test_creation(self):
        from core.algorithms.options_hrl import Option, OptionConfig
        cfg = OptionConfig(
            name="test_skill",
            valid_phases=["RECON"],
            command_templates=["nmap -sV {target}"],
            max_steps=10,
        )
        opt = Option(cfg)
        assert not opt.is_active

    def test_can_initiate(self):
        from core.algorithms.options_hrl import Option, OptionConfig
        cfg = OptionConfig(
            name="recon_skill",
            valid_phases=["RECON", "ENUMERATION"],
        )
        opt = Option(cfg)
        assert opt.can_initiate({"phase": "RECON"})
        assert not opt.can_initiate({"phase": "EXPLOITATION"})

    def test_start_and_terminate(self):
        from core.algorithms.options_hrl import Option, OptionConfig
        cfg = OptionConfig(
            name="test",
            valid_phases=["RECON"],
            command_templates=["cmd1", "cmd2"],
            max_steps=5,
        )
        opt = Option(cfg)
        opt.start()
        assert opt.is_active
        result = opt.terminate()
        assert "option" in result
        assert not opt.is_active

    def test_next_command(self):
        from core.algorithms.options_hrl import Option, OptionConfig
        cfg = OptionConfig(
            name="test",
            valid_phases=["RECON"],
            command_templates=["nmap -sV {target}", "gobuster dir -u {url}"],
        )
        opt = Option(cfg)
        opt.start()
        cmd = opt.next_command()
        assert cmd is not None


class TestOptionsManager:
    def test_creation_with_defaults(self):
        from core.algorithms.options_hrl import OptionsManager
        mgr = OptionsManager()
        mgr.register_default_pentesting_options()
        assert len(mgr._options) > 0

    def test_get_stats(self):
        from core.algorithms.options_hrl import OptionsManager
        mgr = OptionsManager()
        stats = mgr.get_stats()
        assert isinstance(stats, dict)

    def test_reset(self):
        from core.algorithms.options_hrl import OptionsManager
        mgr = OptionsManager()
        mgr.reset()


class TestHighLevelPolicy:
    def test_forward(self):
        from core.algorithms.options_hrl import HighLevelPolicy
        policy = HighLevelPolicy(state_dim=512, n_options=6)
        state = torch.randn(4, 512)
        logits = policy(state)
        assert logits.shape == (4, 6)

    def test_select_option(self):
        from core.algorithms.options_hrl import HighLevelPolicy
        policy = HighLevelPolicy(state_dim=512, n_options=6)
        state = torch.randn(512)
        idx, log_prob, entropy = policy.select_option(state)
        assert 0 <= idx < 6
        assert isinstance(entropy, float)


# ════════════════════════════════════════════════════════════════
# 5. OFFLINE RL CQL/IQL TESTS
# ════════════════════════════════════════════════════════════════


class TestOfflineRLConfig:
    def test_defaults(self):
        from core.algorithms.offline_rl import OfflineRLConfig
        cfg = OfflineRLConfig()
        assert cfg.state_dim == 512
        assert cfg.action_dim == 5
        assert cfg.algorithm == "cql"


class TestQNetwork:
    def test_forward(self):
        from core.algorithms.offline_rl import QNetwork
        net = QNetwork(state_dim=512, action_dim=5)
        s = torch.randn(8, 512)
        q = net(s)
        assert q.shape == (8, 5)


class TestValueNetwork:
    def test_forward(self):
        from core.algorithms.offline_rl import ValueNetwork
        net = ValueNetwork(state_dim=512)
        s = torch.randn(8, 512)
        v = net(s)
        assert v.shape == (8, 1)


class TestOfflineDataset:
    def test_from_transitions(self):
        from core.algorithms.offline_rl import OfflineDataset
        transitions = [
            {
                "state": [0.0] * 512,
                "action": 2,
                "reward": 1.5,
                "next_state": [0.1] * 512,
                "done": False,
            }
            for _ in range(10)
        ]
        ds = OfflineDataset.from_transitions(transitions)
        assert len(ds) == 10
        s, a, r, ns, d = ds[0]
        assert s.shape == (512,)
        assert isinstance(a.item(), int)

    def test_empty_dataset(self):
        from core.algorithms.offline_rl import OfflineDataset
        ds = OfflineDataset.from_transitions([])
        assert len(ds) == 0


class TestCQLTrainer:
    def test_train_step(self):
        from core.algorithms.offline_rl import CQLTrainer, OfflineRLConfig
        cfg = OfflineRLConfig(state_dim=32, action_dim=3, hidden_dim=16, lr=1e-3)
        trainer = CQLTrainer(cfg)
        batch = {
            "states": torch.randn(4, 32),
            "actions": torch.randint(0, 3, (4,)),
            "rewards": torch.randn(4),
            "next_states": torch.randn(4, 32),
            "dones": torch.zeros(4),
        }
        losses = trainer.train_step(batch)
        assert "bellman_loss" in losses
        assert "cql_loss" in losses
        assert "total_loss" in losses
        assert all(isinstance(v, float) for v in losses.values())


class TestIQLTrainer:
    def test_train_step(self):
        from core.algorithms.offline_rl import IQLTrainer, OfflineRLConfig
        cfg = OfflineRLConfig(
            state_dim=32, action_dim=3, hidden_dim=16,
            lr=1e-3, algorithm="iql",
        )
        trainer = IQLTrainer(cfg)
        batch = {
            "states": torch.randn(4, 32),
            "actions": torch.randint(0, 3, (4,)),
            "rewards": torch.randn(4),
            "next_states": torch.randn(4, 32),
            "dones": torch.zeros(4),
        }
        losses = trainer.train_step(batch)
        assert "value_loss" in losses
        assert "q_loss" in losses


class TestOfflineRLTrainer:
    def test_cql_train(self):
        from core.algorithms.offline_rl import OfflineRLTrainer, OfflineDataset, OfflineRLConfig
        cfg = OfflineRLConfig(state_dim=32, action_dim=3, hidden_dim=16, batch_size=4)
        trainer = OfflineRLTrainer(cfg)
        transitions = [
            {
                "state": [float(i)] * 32,
                "action": i % 3,
                "reward": float(i),
                "next_state": [float(i + 1)] * 32,
                "done": False,
            }
            for i in range(16)
        ]
        ds = OfflineDataset.from_transitions(transitions, state_dim=32)
        losses = trainer.train(ds, epochs=2)
        assert len(losses) == 2
        assert "total_loss" in losses[0]

    def test_empty_dataset(self):
        from core.algorithms.offline_rl import OfflineRLTrainer, OfflineDataset
        trainer = OfflineRLTrainer()
        ds = OfflineDataset.from_transitions([])
        losses = trainer.train(ds, epochs=5)
        assert losses == []

    def test_get_stats(self):
        from core.algorithms.offline_rl import OfflineRLTrainer
        trainer = OfflineRLTrainer()
        stats = trainer.get_stats()
        assert stats["algorithm"] == "cql"


# ════════════════════════════════════════════════════════════════
# 6. GOAL-CONDITIONED RL TESTS
# ════════════════════════════════════════════════════════════════


class TestGoalConfig:
    def test_defaults(self):
        from core.algorithms.goal_conditioned import GoalConfig
        cfg = GoalConfig()
        assert cfg.state_dim == 512
        assert cfg.n_goals == 8
        assert cfg.goal_dim == 64


class TestGoalSelector:
    def test_select_goal(self):
        from core.algorithms.goal_conditioned import GoalSelector
        selector = GoalSelector()
        goal_id, goal_name, goal_emb = selector.select_goal(
            state={"phase": "RECON", "ports": set()},
        )
        assert isinstance(goal_id, int)
        assert isinstance(goal_name, str)
        assert isinstance(goal_emb, torch.Tensor)

    def test_check_achievement(self):
        from core.algorithms.goal_conditioned import GoalSelector
        selector = GoalSelector()
        achieved, bonus = selector.check_achievement(
            goal_id=0,
            state={"ports": {22, 80}, "phase": "RECON"},
        )
        assert isinstance(achieved, bool)
        assert isinstance(bonus, float)

    def test_get_stats(self):
        from core.algorithms.goal_conditioned import GoalSelector
        selector = GoalSelector()
        stats = selector.get_stats()
        assert isinstance(stats, dict)


class TestGoalConditionedPolicy:
    def test_forward(self):
        from core.algorithms.goal_conditioned import GoalConditionedPolicy, GoalConfig
        cfg = GoalConfig(state_dim=32, n_goals=4, goal_dim=16)
        policy = GoalConditionedPolicy(cfg)
        state = torch.randn(4, 32)
        goal = torch.randn(4, 16)  # goal embeddings
        logits, value = policy(state, goal)
        assert logits.shape == (4, 5)
        assert value.shape == (4, 1)


class TestGoalEmbedding:
    def test_forward(self):
        from core.algorithms.goal_conditioned import GoalEmbedding
        emb = GoalEmbedding(n_goals=8, goal_dim=64)
        ids = torch.tensor([0, 3, 7])
        out = emb(ids)
        assert out.shape == (3, 64)

    def test_all_embeddings(self):
        from core.algorithms.goal_conditioned import GoalEmbedding
        emb = GoalEmbedding(n_goals=8, goal_dim=64)
        all_emb = emb.all_embeddings()
        assert all_emb.shape == (8, 64)


class TestPentestingGoals:
    def test_goals_exist(self):
        from core.algorithms.goal_conditioned import PENTESTING_GOALS
        assert len(PENTESTING_GOALS) == 8
        for goal in PENTESTING_GOALS:
            assert "id" in goal
            assert "name" in goal


class TestGoalRelabelling:
    def test_relabel_trajectory(self):
        from core.algorithms.goal_conditioned import GoalSelector
        selector = GoalSelector()
        trajectory = [
            {
                "state": {"ports": set(), "phase": "RECON"},
                "action": 0,
                "reward": 0.0,
            },
            {
                "state": {"ports": {22}, "phase": "RECON"},
                "action": 1,
                "reward": 1.0,
            },
        ]
        relabelled = selector.relabel_trajectory(trajectory, original_goal_id=0)
        assert isinstance(relabelled, list)


# ════════════════════════════════════════════════════════════════
# 7. MULTI-AGENT COMMUNICATION TESTS
# ════════════════════════════════════════════════════════════════


class TestMessageType:
    def test_all_types_exist(self):
        from core.multiagent.agent_comm import MessageType
        assert MessageType.OBSERVATION is not None
        assert MessageType.REQUEST is not None
        assert MessageType.DIRECTIVE is not None
        assert MessageType.ACK is not None
        assert MessageType.ALERT is not None
        assert MessageType.DISCOVERY is not None


class TestAgentMessage:
    def test_creation(self):
        from core.multiagent.agent_comm import AgentMessage, MessageType
        msg = AgentMessage(
            sender="RedAgent",
            receiver="ScoutAgent",
            msg_type=MessageType.REQUEST,
            content={"scan": "full"},
        )
        assert msg.sender == "RedAgent"
        assert not msg.is_expired
        assert msg.age >= 0

    def test_expiry(self):
        from core.multiagent.agent_comm import AgentMessage, MessageType
        msg = AgentMessage(
            sender="A", receiver="B",
            msg_type=MessageType.OBSERVATION,
            content={}, ttl=0.01,
        )
        time.sleep(0.02)
        assert msg.is_expired


class TestCommChannel:
    def test_send_and_receive(self):
        from core.multiagent.agent_comm import CommChannel, MessageType, AgentMessage
        ch = CommChannel("RedAgent")
        msg = AgentMessage(
            sender="ScoutAgent", receiver="RedAgent",
            msg_type=MessageType.DISCOVERY,
            content={"port": 22},
        )
        ch.receive(msg)
        msgs = ch.get_messages()
        assert len(msgs) == 1
        assert msgs[0].content["port"] == 22

    def test_priority_ordering(self):
        from core.multiagent.agent_comm import CommChannel, MessageType, AgentMessage
        ch = CommChannel("RedAgent")
        ch.receive(AgentMessage(
            sender="A", receiver="RedAgent",
            msg_type=MessageType.OBSERVATION, content={"x": 1}, priority=0,
        ))
        ch.receive(AgentMessage(
            sender="B", receiver="RedAgent",
            msg_type=MessageType.ALERT, content={"x": 2}, priority=3,
        ))
        msgs = ch.get_messages()
        assert msgs[0].priority >= msgs[1].priority

    def test_queue_limit(self):
        from core.multiagent.agent_comm import CommChannel, CommConfig, MessageType, AgentMessage
        cfg = CommConfig(max_queue_size=3)
        ch = CommChannel("TestAgent", config=cfg)
        for i in range(10):
            ch.receive(AgentMessage(
                sender="X", receiver="TestAgent",
                msg_type=MessageType.OBSERVATION, content={"i": i}, priority=1,
            ))
        assert len(ch.get_messages()) <= 3

    def test_flush_outbox(self):
        from core.multiagent.agent_comm import CommChannel, MessageType
        ch = CommChannel("RedAgent")
        ch.send("ScoutAgent", MessageType.REQUEST, {"scan": "ports"})
        msgs = ch.flush_outbox()
        assert len(msgs) == 1
        assert ch.flush_outbox() == []

    def test_stats(self):
        from core.multiagent.agent_comm import CommChannel, MessageType
        ch = CommChannel("RedAgent")
        ch.send("ScoutAgent", MessageType.REQUEST, {"test": True})
        stats = ch.get_stats()
        assert stats["sent_total"] == 1
        assert stats["agent_id"] == "RedAgent"

    def test_fuse_messages_no_messages(self):
        from core.multiagent.agent_comm import CommChannel
        ch = CommChannel("RedAgent")
        state = torch.randn(512)
        fused = ch.fuse_messages_into_state(state)
        assert fused.shape == state.shape
        assert torch.allclose(fused, state)


class TestMessageBus:
    def test_register_agents(self):
        from core.multiagent.agent_comm import MessageBus
        bus = MessageBus()
        bus.register_agent("RedAgent")
        bus.register_agent("ScoutAgent")
        assert len(bus.channels) == 2

    def test_route_messages(self):
        from core.multiagent.agent_comm import MessageBus, MessageType
        bus = MessageBus()
        bus.register_agent("RedAgent")
        bus.register_agent("ScoutAgent")
        bus.channels["RedAgent"].send("ScoutAgent", MessageType.REQUEST, {"scan": True})
        routed = bus.route_messages()
        assert routed == 1
        msgs = bus.channels["ScoutAgent"].get_messages()
        assert len(msgs) == 1

    def test_broadcast(self):
        from core.multiagent.agent_comm import MessageBus, MessageType
        bus = MessageBus()
        bus.register_agent("OrionAgent")
        bus.register_agent("RedAgent")
        bus.register_agent("BlueAgent")
        bus.broadcast("OrionAgent", MessageType.DIRECTIVE, {"phase": "EXPLOIT"})
        bus.route_messages()
        assert len(bus.channels["RedAgent"].get_messages()) == 1
        assert len(bus.channels["BlueAgent"].get_messages()) == 1
        assert len(bus.channels["OrionAgent"].get_messages()) == 0

    def test_stats(self):
        from core.multiagent.agent_comm import MessageBus
        bus = MessageBus()
        bus.register_agent("A")
        stats = bus.get_stats()
        assert stats["n_agents"] == 1

    def test_reset(self):
        from core.multiagent.agent_comm import MessageBus, MessageType
        bus = MessageBus()
        bus.register_agent("A")
        bus.register_agent("B")
        bus.channels["A"].send("B", MessageType.OBSERVATION, {})
        bus.route_messages()
        bus.reset()
        assert len(bus.channels["B"].get_messages()) == 0


# ════════════════════════════════════════════════════════════════
# 8. WORLD MODEL TESTS
# ════════════════════════════════════════════════════════════════


class TestWorldModelConfig:
    def test_defaults(self):
        from core.algorithms.world_model import WorldModelConfig
        cfg = WorldModelConfig()
        assert cfg.state_dim == 512
        assert cfg.action_dim == 5
        assert cfg.imagination_horizon == 5


class TestSymlogFunctions:
    def test_symlog_inverse(self):
        from core.algorithms.world_model import symlog, symexp
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 5.0])
        recovered = symexp(symlog(x))
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_symlog_zero(self):
        from core.algorithms.world_model import symlog
        assert symlog(torch.tensor(0.0)).item() == 0.0


class TestWorldModel:
    def test_creation(self):
        from core.algorithms.world_model import WorldModel, WorldModelConfig
        cfg = WorldModelConfig(state_dim=32, hidden_dim=16, stoch_dim=8)
        model = WorldModel(cfg)
        stats = model.get_stats()
        assert stats["hidden_dim"] == 16
        assert stats["n_params"] > 0

    def test_observe(self):
        from core.algorithms.world_model import WorldModel, WorldModelConfig
        cfg = WorldModelConfig(state_dim=32, hidden_dim=16, stoch_dim=8, action_dim=3)
        model = WorldModel(cfg)
        obs = torch.randn(2, 5, 32)
        acts = torch.randint(0, 3, (2, 5))
        result = model.observe(obs, acts)
        assert result["hiddens"].shape == (2, 5, 16)
        assert result["latents"].shape == (2, 5, 24)
        assert len(result["priors"]) == 5

    def test_imagine(self):
        from core.algorithms.world_model import WorldModel, WorldModelConfig
        cfg = WorldModelConfig(state_dim=32, hidden_dim=16, stoch_dim=8, action_dim=3)
        model = WorldModel(cfg)
        state = torch.randn(32)
        actions = torch.tensor([0, 1, 2])
        result = model.imagine(state, actions)
        assert result["rewards"].shape == (1, 3)
        assert result["continues"].shape == (1, 3)
        assert result["states"].shape == (1, 3, 32)

    def test_train_step(self):
        from core.algorithms.world_model import WorldModel, WorldModelConfig
        cfg = WorldModelConfig(state_dim=32, hidden_dim=16, stoch_dim=8, action_dim=3)
        model = WorldModel(cfg)
        obs = torch.randn(4, 5, 32)
        acts = torch.randint(0, 3, (4, 5))
        rews = torch.randn(4, 5)
        dones = torch.zeros(4, 5)
        losses = model.train_step(obs, acts, rews, dones)
        assert "total" in losses
        assert "reward" in losses
        assert "kl" in losses
        assert all(isinstance(v, float) for v in losses.values())

    def test_evaluate_action_sequences(self):
        from core.algorithms.world_model import WorldModel, WorldModelConfig
        cfg = WorldModelConfig(state_dim=32, hidden_dim=16, stoch_dim=8, action_dim=3)
        model = WorldModel(cfg)
        state = torch.randn(32)
        candidates = [[0, 1, 2], [2, 2, 0], [1, 0, 1]]
        scores = model.evaluate_action_sequence(state, candidates)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)


# ════════════════════════════════════════════════════════════════
# 9. POPULATION-BASED TRAINING TESTS
# ════════════════════════════════════════════════════════════════


class TestPBTConfig:
    def test_defaults(self):
        from core.algorithms.pbt import PBTConfig
        cfg = PBTConfig()
        assert cfg.population_size == 8
        assert cfg.exploit_top_frac > 0
        assert cfg.explore_perturb_factor > 0


class TestHyperConfig:
    def test_to_dict(self):
        from core.algorithms.pbt import HyperConfig
        hp = HyperConfig()
        d = hp.to_dict()
        assert "learning_rate" in d
        assert "clip_epsilon" in d
        assert len(d) == 10

    def test_custom_values(self):
        from core.algorithms.pbt import HyperConfig
        hp = HyperConfig(learning_rate=1e-4, clip_epsilon=0.15)
        assert hp.learning_rate == 1e-4
        assert hp.clip_epsilon == 0.15


class TestPBTManager:
    def test_initialize_population(self):
        from core.algorithms.pbt import PBTManager, PBTConfig
        cfg = PBTConfig(population_size=4, seed=42)
        mgr = PBTManager(cfg)
        pop = mgr.initialize_population()
        assert len(pop) == 4
        assert pop[0].hyperparams.learning_rate == 3e-4

    def test_evolve(self):
        from core.algorithms.pbt import PBTManager, PBTConfig
        cfg = PBTConfig(population_size=6, exploit_top_frac=0.33, seed=42)
        mgr = PBTManager(cfg)
        pop = mgr.initialize_population()
        for i, member in enumerate(pop):
            member.record_fitness(float(i))
        evolved = mgr.evolve(pop)
        assert len(evolved) == 6
        best = mgr.get_best(evolved)
        assert best.fitness == 5.0

    def test_best_ever(self):
        from core.algorithms.pbt import PBTManager, PBTConfig
        cfg = PBTConfig(population_size=4, seed=42)
        mgr = PBTManager(cfg)
        pop = mgr.initialize_population()
        for i, m in enumerate(pop):
            m.record_fitness(float(i))
        mgr.evolve(pop)
        best_fitness, best_config = mgr.get_best_ever()
        assert best_fitness == 3.0
        assert best_config is not None

    def test_diversity(self):
        from core.algorithms.pbt import PBTManager, PBTConfig
        cfg = PBTConfig(population_size=8, seed=42)
        mgr = PBTManager(cfg)
        pop = mgr.initialize_population()
        diversity = mgr.get_diversity(pop)
        assert "learning_rate" in diversity
        assert any(v > 0 for v in diversity.values())

    def test_stats(self):
        from core.algorithms.pbt import PBTManager
        mgr = PBTManager()
        stats = mgr.get_stats()
        assert stats["generation"] == 0
        assert stats["population_size"] == 8

    def test_member_history(self):
        from core.algorithms.pbt import PopulationMember, HyperConfig
        member = PopulationMember(member_id=0, hyperparams=HyperConfig())
        member.record_fitness(1.0)
        member.record_fitness(2.0)
        member.record_fitness(3.0)
        assert member.history == [1.0, 2.0, 3.0]
        assert member.fitness == 3.0


# ════════════════════════════════════════════════════════════════
# 10. ENHANCED CONTRASTIVE STATE TESTS
# ════════════════════════════════════════════════════════════════


class TestContrastiveConfig:
    def test_defaults(self):
        from core.algorithms.contrastive_state import ContrastiveConfig
        cfg = ContrastiveConfig()
        assert cfg.temperature == 0.1
        assert cfg.projection_dim == 64
        assert cfg.augment_noise > 0
        assert cfg.momentum > 0.9

    def test_custom(self):
        from core.algorithms.contrastive_state import ContrastiveConfig
        cfg = ContrastiveConfig(temperature=0.05, hard_negative_frac=0.3)
        assert cfg.temperature == 0.05
        assert cfg.hard_negative_frac == 0.3


class TestStateAugmentor:
    def test_augment_changes_tensor(self):
        from core.algorithms.contrastive_state import StateAugmentor, ContrastiveConfig
        cfg = ContrastiveConfig(augment_noise=0.1)
        aug = StateAugmentor(cfg)
        x = torch.randn(4, 256)
        x_aug = aug.augment(x)
        assert x_aug.shape == x.shape
        assert not torch.allclose(x_aug, x)

    def test_augment_1d(self):
        from core.algorithms.contrastive_state import StateAugmentor, ContrastiveConfig
        aug = StateAugmentor(ContrastiveConfig())
        x = torch.randn(256)
        x_aug = aug.augment(x)
        assert x_aug.shape == (256,)


class TestContrastiveLoss:
    def test_basic_loss(self):
        from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
        cfg = ContrastiveConfig(feature_dim=32, projection_dim=16)
        loss_fn = ContrastiveLoss(cfg)
        features = torch.randn(8, 32)
        phases = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn.compute_loss(features, phases)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_batch_too_small(self):
        from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
        cfg = ContrastiveConfig(feature_dim=32)
        loss_fn = ContrastiveLoss(cfg)
        features = torch.randn(1, 32)
        phases = torch.tensor([0])
        loss = loss_fn.compute_loss(features, phases)
        assert loss.item() == 0.0

    def test_single_phase(self):
        from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
        cfg = ContrastiveConfig(feature_dim=32)
        loss_fn = ContrastiveLoss(cfg)
        features = torch.randn(4, 32)
        phases = torch.tensor([0, 0, 0, 0])
        loss = loss_fn.compute_loss(features, phases)
        assert loss.item() == 0.0

    def test_discovery_weighting(self):
        from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
        cfg = ContrastiveConfig(feature_dim=32, projection_dim=16)
        loss_fn = ContrastiveLoss(cfg)
        features = torch.randn(8, 32)
        phases = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        discoveries = torch.tensor([0, 1, 0, 2, 0, 3, 0, 1], dtype=torch.float)
        loss = loss_fn.compute_loss(features, phases, discovery_counts=discoveries)
        assert loss.item() >= 0

    def test_trajectory_loss(self):
        from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
        cfg = ContrastiveConfig(feature_dim=32, projection_dim=16)
        loss_fn = ContrastiveLoss(cfg)
        success = torch.randn(5, 32)
        failure = torch.randn(5, 32)
        loss = loss_fn.compute_trajectory_loss(success, failure)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_temporal_loss(self):
        from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
        cfg = ContrastiveConfig(feature_dim=32, projection_dim=16, temporal_window=2)
        loss_fn = ContrastiveLoss(cfg)
        trajectory = torch.randn(10, 32)
        loss = loss_fn.compute_temporal_loss(trajectory)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_get_representations(self):
        from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
        cfg = ContrastiveConfig(feature_dim=32, projection_dim=16)
        loss_fn = ContrastiveLoss(cfg)
        features = torch.randn(4, 32)
        reps = loss_fn.get_representations(features)
        assert reps.shape == (4, 16)
        norms = reps.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_momentum_update(self):
        from core.algorithms.contrastive_state import ContrastiveLoss, ContrastiveConfig
        cfg = ContrastiveConfig(feature_dim=32, projection_dim=16, momentum=0.99)
        loss_fn = ContrastiveLoss(cfg)
        before = {
            n: p.clone()
            for n, p in loss_fn.momentum_projector.net.named_parameters()
        }
        for p in loss_fn.projector.net.parameters():
            p.data += 0.1
        loss_fn._update_momentum()
        for n, p in loss_fn.momentum_projector.net.named_parameters():
            assert not torch.allclose(p, before[n])

    def test_stats(self):
        from core.algorithms.contrastive_state import ContrastiveLoss
        loss_fn = ContrastiveLoss()
        stats = loss_fn.get_stats()
        assert "steps" in stats
        assert "temperature" in stats


# ════════════════════════════════════════════════════════════════
# INTEGRATION SMOKE TEST
# ════════════════════════════════════════════════════════════════


class TestPhase49Integration:
    """Smoke test that all 10 modules import and initialise cleanly."""

    def test_all_modules_import(self):
        from core.algorithms.ensemble_voting import EnsembleVoter
        from core.algorithms.auto_curriculum import CurriculumScheduler
        from core.algorithms.llm_reward_shaper import LLMRewardShaper
        from core.algorithms.options_hrl import OptionsManager
        from core.algorithms.offline_rl import OfflineRLTrainer
        from core.algorithms.goal_conditioned import GoalSelector
        from core.multiagent.agent_comm import MessageBus
        from core.algorithms.world_model import WorldModel
        from core.algorithms.pbt import PBTManager
        from core.algorithms.contrastive_state import ContrastiveLoss

        assert EnsembleVoter() is not None
        assert CurriculumScheduler() is not None
        assert LLMRewardShaper() is not None
        assert OptionsManager() is not None
        assert GoalSelector() is not None
        assert MessageBus() is not None
        assert WorldModel() is not None
        assert PBTManager() is not None
        assert ContrastiveLoss() is not None
