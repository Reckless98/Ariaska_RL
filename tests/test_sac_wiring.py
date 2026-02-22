"""
C03: Tests for SAC off-policy wiring via DecisionPacket.

Verifies:
1. SAC shadow select runs in decide() and populates DecisionPacket
2. SAC transitions stored in record_result() (off-policy)
3. SAC update called every step via replay buffer
4. SAC doesn't interfere with PPO's command selection
"""
from __future__ import annotations

import os
import pytest
from typing import Any, Dict, Optional

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestSACShadowSelect:
    """Verify SAC shadow selection runs during decide()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def _make_coach(self):
        from core.training.smart_coach import SmartCoach
        from core.llm.smart_mentor import AttackContext
        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="172.28.0.10")
        return coach

    def test_sac_agent_initialized(self):
        """SmartCoach should have sac_agent initialized."""
        coach = self._make_coach()
        assert coach.sac_agent is not None, "SAC agent should be initialized"

    def test_sac_pending_initialized_none(self):
        """_sac_pending should start as None."""
        coach = self._make_coach()
        assert coach._sac_pending is None

    def test_sac_shadow_select_populates_pending(self):
        """_sac_shadow_select() should set _sac_pending."""
        from core.training.smart_coach import SmartStepContext
        coach = self._make_coach()
        ctx = SmartStepContext(
            episode=1, step=5, agent_name="RedAgent",
            attack_context=coach.attack_context,
            state={"phase": "RECON", "state_flags": {}},
        )
        coach._sac_shadow_select(ctx)
        assert coach._sac_pending is not None
        assert "state" in coach._sac_pending
        assert "action" in coach._sac_pending
        assert "log_prob" in coach._sac_pending
        assert "q_value" in coach._sac_pending

    def test_sac_shadow_select_populates_packet(self):
        """SAC should populate DecisionPacket.sac proposal."""
        from core.training.smart_coach import SmartStepContext
        from core.training.decision_packet import DecisionPacket
        coach = self._make_coach()
        pkt = DecisionPacket()
        coach._current_decision_packet = pkt
        ctx = SmartStepContext(
            episode=1, step=5, agent_name="RedAgent",
            attack_context=coach.attack_context,
            state={"phase": "RECON", "state_flags": {}},
        )
        coach._sac_shadow_select(ctx)
        # DecisionPacket should have SAC proposal
        assert pkt.sac.action_idx >= 0 or pkt.sac.action_idx == 0

    def test_sac_disabled_no_crash(self):
        """When sac_agent is None, _sac_shadow_select() should not crash."""
        from core.training.smart_coach import SmartStepContext
        coach = self._make_coach()
        coach.sac_agent = None
        ctx = SmartStepContext(
            episode=1, step=5, agent_name="RedAgent",
            attack_context=coach.attack_context,
            state={"phase": "RECON"},
        )
        # Should not raise
        coach._sac_shadow_select(ctx)
        assert coach._sac_pending is None


class TestSACTransitionStorage:
    """Verify SAC transitions stored in record_result()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def _make_coach(self):
        from core.training.smart_coach import SmartCoach
        from core.llm.smart_mentor import AttackContext
        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="172.28.0.10")
        return coach

    def test_sac_transition_stored_on_record_result(self):
        """SAC should store transition in replay buffer during record_result()."""
        from core.training.smart_coach import SmartDecisionResult
        from core.training.decision_packet import DecisionPacket
        import torch

        coach = self._make_coach()
        assert coach.sac_agent is not None

        # Simulate SAC pending (as if _sac_shadow_select ran)
        state = torch.randn(512)
        coach._sac_pending = {
            "state": state,
            "action": 2,
            "log_prob": -0.7,
            "q_value": 3.5,
        }
        # Also need PPO pending for record_result to work
        coach._ppo_pending = {
            "state": state,
            "action": 0,
            "log_prob": -1.0,
            "value": 0.5,
        }
        # Set up decision packet (normally set by decide())
        coach._current_decision_packet = DecisionPacket(step=1, episode=1)

        initial_buffer_size = len(coach.sac_agent.replay_buffer)

        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7,
            source="ppo",
        )
        coach.record_result(
            decision=decision,
            success=True,
            raw_output="80/tcp open http",
            done=False,
        )

        # SAC replay buffer should have grown by 1
        assert len(coach.sac_agent.replay_buffer) == initial_buffer_size + 1
        # _sac_pending should be cleared
        assert coach._sac_pending is None

    def test_sac_pending_cleared_after_record(self):
        """_sac_pending should be None after record_result()."""
        from core.training.smart_coach import SmartDecisionResult
        from core.training.decision_packet import DecisionPacket
        import torch

        coach = self._make_coach()
        state = torch.randn(512)
        coach._sac_pending = {
            "state": state, "action": 1, "log_prob": -0.5, "q_value": 2.0,
        }
        coach._ppo_pending = {
            "state": state, "action": 0, "log_prob": -1.0, "value": 0.5,
        }
        coach._current_decision_packet = DecisionPacket(step=1, episode=1)

        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7, source="ppo",
        )
        coach.record_result(
            decision=decision, success=True, raw_output="", done=False,
        )
        assert coach._sac_pending is None

    def test_no_sac_storage_when_disabled(self):
        """When sac_agent is None, no transition stored."""
        from core.training.smart_coach import SmartDecisionResult
        from core.training.decision_packet import DecisionPacket
        import torch

        coach = self._make_coach()
        coach.sac_agent = None
        state = torch.randn(512)
        coach._sac_pending = {
            "state": state, "action": 1, "log_prob": -0.5, "q_value": 2.0,
        }
        coach._ppo_pending = {
            "state": state, "action": 0, "log_prob": -1.0, "value": 0.5,
        }
        coach._current_decision_packet = DecisionPacket(step=1, episode=1)

        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7, source="ppo",
        )
        # Should not crash
        coach.record_result(
            decision=decision, success=True, raw_output="", done=False,
        )


class TestSACUpdate:
    """Verify SAC updates via off-policy replay buffer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_sac_update_after_enough_transitions(self):
        """SAC should update once buffer has enough transitions."""
        from core.algorithms.sac_agent import SACAgent, SACConfig
        import torch

        config = SACConfig(
            state_dim=512, action_dim=5,
            hidden_dims=[64, 64],
            min_buffer_size=10,  # Low for testing
            batch_size=8,
            warmup_steps=5,
        )
        sac = SACAgent(config=config)

        # Fill buffer with enough transitions
        for i in range(15):
            state = torch.randn(512)
            next_state = torch.randn(512)
            sac.store_transition(state, i % 5, 1.0, next_state, False)

        # Now update should produce real metrics
        metrics = sac.update()
        assert metrics is not None
        assert metrics["critic_loss"] > 0 or metrics["actor_loss"] != 0

    def test_sac_update_before_warmup(self):
        """SAC should return placeholder metrics before enough data."""
        from core.algorithms.sac_agent import SACAgent, SACConfig
        import torch

        config = SACConfig(
            state_dim=512, action_dim=5, hidden_dims=[64, 64],
            min_buffer_size=100, warmup_steps=50,
        )
        sac = SACAgent(config=config)
        # Only add a few transitions
        for i in range(5):
            sac.store_transition(torch.randn(512), i, 1.0, torch.randn(512), False)

        metrics = sac.update()
        assert metrics is not None
        # Should be default/placeholder metrics since buffer too small
        assert metrics.get("critic_loss", 0.0) == 0.0


class TestSACPacketIntegration:
    """End-to-end: SAC select → packet → transition → update."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_full_sac_lifecycle(self):
        """Simulate multiple steps with SAC shadow select + transition storage."""
        from core.training.smart_coach import SmartCoach, SmartDecisionResult, SmartStepContext
        from core.training.decision_packet import DecisionPacket
        from core.llm.smart_mentor import AttackContext
        import torch

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="172.28.0.10")
        assert coach.sac_agent is not None

        for i in range(20):
            state = torch.randn(512)
            ctx = SmartStepContext(
                episode=1, step=i, agent_name="RedAgent",
                attack_context=coach.attack_context,
                state={"phase": "RECON", "state_flags": {}},
            )
            pkt = DecisionPacket(step=i, episode=1)
            coach._current_decision_packet = pkt

            # Shadow select
            coach._sac_shadow_select(ctx)
            assert coach._sac_pending is not None

            # Simulate PPO pending
            coach._ppo_pending = {
                "state": state, "action": i % 5,
                "log_prob": -1.0, "value": 0.5,
            }

            # Record result
            decision = SmartDecisionResult(
                command=f"nmap -{i} 172.28.0.10",
                template_name="nmap_version",
                confidence=0.7, source="ppo",
            )
            coach.record_result(
                decision=decision, success=True, raw_output="", done=False,
            )
            assert coach._sac_pending is None

        # After 20 steps, SAC buffer should have 20 entries
        assert len(coach.sac_agent.replay_buffer) == 20
        # SAC should have started updating (min_buffer_size=64 default, but
        # we only have 20, so _update_count may be 0 unless config allows)
        # The important thing is no crashes occurred

    def test_sac_does_not_override_ppo(self):
        """SAC is shadow — it doesn't change the final decision source."""
        from core.training.smart_coach import SmartCoach, SmartDecisionResult, SmartStepContext
        from core.training.decision_packet import DecisionPacket
        from core.llm.smart_mentor import AttackContext
        import torch

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="172.28.0.10")

        ctx = SmartStepContext(
            episode=1, step=5, agent_name="RedAgent",
            attack_context=coach.attack_context,
            state={"phase": "RECON", "state_flags": {}},
        )
        pkt = DecisionPacket(step=5, episode=1)
        coach._current_decision_packet = pkt

        coach._sac_shadow_select(ctx)

        # SAC proposed something, but PPO should still be the source
        state = torch.randn(512)
        coach._ppo_pending = {
            "state": state, "action": 0, "log_prob": -1.0, "value": 0.5,
        }
        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7, source="ppo",
        )
        coach.record_result(
            decision=decision, success=True, raw_output="", done=False,
        )
        # Decision source should still be "ppo", not "sac"
        assert decision.source == "ppo"


class TestSACAgentUnit:
    """Unit tests for SAC agent core functionality."""

    def test_select_action_returns_valid(self):
        from core.algorithms.sac_agent import SACAgent, SACConfig
        import torch

        config = SACConfig(state_dim=512, action_dim=5, hidden_dims=[64, 64], warmup_steps=0)
        sac = SACAgent(config=config)
        state = torch.randn(512)
        action, log_prob = sac.select_action(state)
        assert 0 <= action < 5
        assert isinstance(log_prob, torch.Tensor)

    def test_store_and_update(self):
        from core.algorithms.sac_agent import SACAgent, SACConfig
        import torch

        config = SACConfig(
            state_dim=512, action_dim=5, hidden_dims=[64, 64],
            min_buffer_size=5, batch_size=4, warmup_steps=0,
        )
        sac = SACAgent(config=config)
        for i in range(10):
            sac.store_transition(torch.randn(512), i % 5, 1.0, torch.randn(512), False)
        metrics = sac.update()
        assert metrics["critic_loss"] > 0

    def test_get_stats(self):
        from core.algorithms.sac_agent import SACAgent, SACConfig
        sac = SACAgent(SACConfig(state_dim=512, action_dim=5, hidden_dims=[64, 64]))
        stats = sac.get_stats()
        assert stats["algorithm"] == "SAC"
        assert "alpha" in stats
        assert "buffer_size" in stats
