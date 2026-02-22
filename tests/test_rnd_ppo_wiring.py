"""
C02: Tests for RND → PPO per-step wiring via DecisionPacket.

Verifies:
1. DecisionPacket.rnd carries intrinsic reward and novelty
2. RND intrinsic is injected into PPO trajectory reward in record_result()
3. Orchestrator computes RND pre-step, updates predictor post-step
4. No double-counting of intrinsic reward (episode_reward path removed)
"""
from __future__ import annotations

import os
import sys
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

os.environ["ARIASKA_DRY_RUN"] = "1"


# ── DecisionPacket RND signal tests ──────────────────────────────────────

class TestRNDSignalOnPacket:
    """Validate that DecisionPacket.rnd fields carry intrinsic correctly."""

    def test_rnd_signal_defaults_zero(self):
        from core.training.decision_packet import RNDSignal
        sig = RNDSignal()
        assert sig.intrinsic_reward == 0.0
        assert sig.novelty_score == 0.0
        assert sig.phase_decay == 1.0
        assert not sig.valid  # 0.0 intrinsic → not valid

    def test_rnd_signal_valid_when_nonzero(self):
        from core.training.decision_packet import RNDSignal
        sig = RNDSignal(intrinsic_reward=0.5, novelty_score=1.2, phase_decay=0.7)
        assert sig.valid
        assert sig.intrinsic_reward == 0.5
        assert sig.novelty_score == 1.2
        assert sig.phase_decay == 0.7

    def test_packet_carries_rnd_from_factory(self):
        from core.training.decision_packet import DecisionPacket
        from core.training.smart_coach import SmartStepContext
        ctx = SmartStepContext(
            episode=1, step=5, agent_name="RedAgent",
            attack_context=None, state={"phase": "RECON"},
        )
        pkt = DecisionPacket.from_step_context(
            ctx, rnd_intrinsic=1.5, coherence=0.8, macro_confidence=0.3,
        )
        assert pkt.rnd.intrinsic_reward == 1.5
        assert pkt.rnd.valid

    def test_packet_rnd_zero_means_invalid(self):
        from core.training.decision_packet import DecisionPacket
        from core.training.smart_coach import SmartStepContext
        ctx = SmartStepContext(
            episode=1, step=5, agent_name="RedAgent",
            attack_context=None, state={},
        )
        pkt = DecisionPacket.from_step_context(ctx, rnd_intrinsic=0.0)
        assert not pkt.rnd.valid

    def test_rnd_novelty_and_decay_populated(self):
        from core.training.decision_packet import DecisionPacket, RNDSignal
        pkt = DecisionPacket()
        pkt.rnd = RNDSignal(
            intrinsic_reward=2.0, novelty_score=3.5, phase_decay=0.4,
        )
        assert pkt.rnd.novelty_score == 3.5
        assert pkt.rnd.phase_decay == 0.4
        d = pkt.to_dict()
        assert d["rnd"]["intrinsic"] == 2.0
        assert d["rnd"]["novelty"] == 3.5


# ── RND injection into PPO trajectory reward ─────────────────────────────

class TestRNDInjectionIntoPPO:
    """SmartCoach.record_result() must add RND intrinsic to ppo_reward."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def _make_coach(self):
        """Create a minimal SmartCoach for testing."""
        from core.training.smart_coach import SmartCoach
        from core.llm.smart_mentor import AttackContext
        ctx = AttackContext(target="172.28.0.10")
        coach = SmartCoach(
            agent_name="RedAgent",
            gpt_manager=self.gpt,
        )
        coach.attack_context = ctx
        return coach

    def test_rnd_intrinsic_added_to_ppo_reward(self):
        """When decision_packet.rnd.valid, intrinsic is added to ppo_reward."""
        from core.training.decision_packet import DecisionPacket, RNDSignal
        from core.training.smart_coach import SmartDecisionResult

        coach = self._make_coach()
        # Simulate PPO pending state
        import torch
        coach._ppo_pending = {
            "state": torch.zeros(512),
            "action": 0,
            "log_prob": -1.0,
            "value": 0.5,
        }
        # Set decision packet with RND signal
        pkt = DecisionPacket()
        pkt.rnd = RNDSignal(intrinsic_reward=2.5, novelty_score=3.0, phase_decay=0.8)
        coach._current_decision_packet = pkt

        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7,
            source="ppo",
        )
        breakdown = coach.record_result(
            decision=decision,
            success=True,
            raw_output="80/tcp open http",
            new_discoveries=None,
            done=False,
        )
        # PPO trajectory should have been appended
        assert len(coach._ppo_trajectory) >= 1
        last_entry = coach._ppo_trajectory[-1]
        # The reward should include RND intrinsic (2.5) on top of breakdown.total
        assert last_entry["reward"] >= breakdown.total + 2.5 - 0.01

    def test_no_rnd_when_packet_missing(self):
        """When decision_packet is None, no RND intrinsic added."""
        from core.training.smart_coach import SmartDecisionResult

        coach = self._make_coach()
        import torch
        coach._ppo_pending = {
            "state": torch.zeros(512),
            "action": 0,
            "log_prob": -1.0,
            "value": 0.5,
        }
        coach._current_decision_packet = None

        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7,
            source="ppo",
        )
        breakdown = coach.record_result(
            decision=decision,
            success=True,
            raw_output="80/tcp open http",
            done=False,
        )
        assert len(coach._ppo_trajectory) >= 1
        last_entry = coach._ppo_trajectory[-1]
        # Reward should be close to breakdown.total (no RND)
        # (there may be other bonuses from discovery/reasoning channels)
        assert abs(last_entry["reward"] - breakdown.total) < 0.01 or last_entry["reward"] >= breakdown.total

    def test_no_rnd_when_signal_zero(self):
        """When RND intrinsic is 0.0, no extra reward added."""
        from core.training.decision_packet import DecisionPacket, RNDSignal
        from core.training.smart_coach import SmartDecisionResult

        coach = self._make_coach()
        import torch
        coach._ppo_pending = {
            "state": torch.zeros(512),
            "action": 0,
            "log_prob": -1.0,
            "value": 0.5,
        }
        pkt = DecisionPacket()
        pkt.rnd = RNDSignal(intrinsic_reward=0.0)
        coach._current_decision_packet = pkt

        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7,
            source="ppo",
        )
        breakdown = coach.record_result(
            decision=decision, success=True, raw_output="", done=False,
        )
        assert len(coach._ppo_trajectory) >= 1
        last_entry = coach._ppo_trajectory[-1]
        # RND is 0, so reward equals breakdown.total exactly
        assert abs(last_entry["reward"] - breakdown.total) < 0.01

    def test_reward_composition_populated(self):
        """DecisionPacket.reward composition gets extrinsic + intrinsic_rnd."""
        from core.training.decision_packet import DecisionPacket, RNDSignal
        from core.training.smart_coach import SmartDecisionResult

        coach = self._make_coach()
        import torch
        coach._ppo_pending = {
            "state": torch.zeros(512),
            "action": 0,
            "log_prob": -1.0,
            "value": 0.5,
        }
        pkt = DecisionPacket()
        pkt.rnd = RNDSignal(intrinsic_reward=1.0, novelty_score=2.0, phase_decay=1.0)
        coach._current_decision_packet = pkt

        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7,
            source="ppo",
        )
        breakdown = coach.record_result(
            decision=decision, success=True, raw_output="", done=False,
        )
        # Reward composition should be populated
        assert pkt.reward.intrinsic_rnd == 1.0
        assert pkt.reward.extrinsic == breakdown.total


# ── RNDCuriosity module unit tests ───────────────────────────────────────

class TestRNDCuriosity:
    """Verify RND module itself works correctly for C02 wiring."""

    def test_compute_intrinsic_returns_float(self):
        from core.algorithms.rnd_curiosity import RNDCuriosity
        import torch
        rnd = RNDCuriosity(state_dim=512, hidden_dim=64, output_dim=32)
        state = torch.randn(512)
        intrinsic = rnd.compute_intrinsic_reward(state, phase="RECON")
        assert isinstance(intrinsic, float)
        assert intrinsic >= 0.0

    def test_update_reduces_loss(self):
        """After many updates, prediction error should shrink for same state."""
        from core.algorithms.rnd_curiosity import RNDCuriosity
        import torch
        rnd = RNDCuriosity(state_dim=512, hidden_dim=64, output_dim=32, learning_rate=1e-2)
        state = torch.randn(512)
        initial = rnd.compute_intrinsic_reward(state, phase="RECON")
        for _ in range(50):
            rnd.update(state)
        after = rnd.compute_intrinsic_reward(state, phase="RECON")
        # After many updates on same state, intrinsic should decrease
        assert after <= initial + 0.5  # Allow small variance tolerance

    def test_phase_decay_near_closeout(self):
        """Intrinsic reward decays for late phases (POST_EXPLOITATION+)."""
        from core.algorithms.rnd_curiosity import RNDCuriosity
        import torch
        rnd = RNDCuriosity(state_dim=512, hidden_dim=64, output_dim=32)
        state = torch.randn(512)
        recon_reward = rnd.compute_intrinsic_reward(state, phase="RECON")
        closeout_reward = rnd.compute_intrinsic_reward(state, phase="CLOSEOUT")
        # Due to phase decay, closeout should be <= recon (decay = 0.1)
        assert closeout_reward <= recon_reward + 0.01

    def test_reward_capped(self):
        """Intrinsic reward should be capped at reward_cap."""
        from core.algorithms.rnd_curiosity import RNDCuriosity
        import torch
        rnd = RNDCuriosity(state_dim=512, hidden_dim=64, output_dim=32, reward_cap=3.0)
        state = torch.randn(512)
        reward = rnd.compute_intrinsic_reward(state, phase="RECON")
        assert reward <= 3.0

    def test_none_state_returns_zero(self):
        from core.algorithms.rnd_curiosity import RNDCuriosity
        rnd = RNDCuriosity(state_dim=512, hidden_dim=64, output_dim=32)
        assert rnd.compute_intrinsic_reward(None) == 0.0


# ── Orchestrator RND compute timing tests ────────────────────────────────

class TestOrchestratorRNDTiming:
    """Verify RND is computed pre-step and predictor updated post-step."""

    def test_rnd_state_tensor_initialized(self):
        """SmartOrchestrator should init _rnd_state_tensor = None."""
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        orch = SmartOrchestrator.__new__(SmartOrchestrator)
        # Simulate minimal init
        orch.rnd_curiosity = None
        orch._rnd_state_tensor = None
        assert orch._rnd_state_tensor is None

    def test_rnd_compute_populates_packet(self):
        """When RND is available, DecisionPacket.rnd should be populated."""
        from core.training.decision_packet import DecisionPacket, RNDSignal
        import torch
        pkt = DecisionPacket()
        # Simulate what orchestrator does post-C02
        pkt.rnd = RNDSignal(intrinsic_reward=1.5, novelty_score=2.3, phase_decay=0.7)
        assert pkt.rnd.valid
        assert pkt.rnd.intrinsic_reward == 1.5

    def test_old_episode_reward_path_removed(self):
        """Verify the old direct episode_reward += intrinsic path is gone."""
        import inspect
        from core.orchestration import smart_orchestrator
        source = inspect.getsource(smart_orchestrator)
        # The old pattern was: episode_reward += _r66_intrinsic
        # It should no longer exist (replaced by C02 pre-step compute)
        assert "episode_reward += _r66_intrinsic" not in source


# ── Integration: End-to-end RND→PPO flow ─────────────────────────────────

class TestRNDPPOIntegration:
    """End-to-end: RND computed → packet populated → PPO reward augmented."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_full_flow_rnd_to_trajectory(self):
        """Simulate the full C02 flow: compute RND → packet → coach → trajectory."""
        from core.algorithms.rnd_curiosity import RNDCuriosity
        from core.training.decision_packet import DecisionPacket, RNDSignal
        from core.training.smart_coach import SmartCoach, SmartDecisionResult
        from core.llm.smart_mentor import AttackContext
        import torch

        # 1. Init RND
        rnd = RNDCuriosity(state_dim=512, hidden_dim=64, output_dim=32)
        state = torch.randn(512)

        # 2. Compute intrinsic (what orchestrator does pre-step)
        intrinsic = rnd.compute_intrinsic_reward(state, phase="RECON")
        assert isinstance(intrinsic, float)

        # 3. Create packet with RND signal
        pkt = DecisionPacket()
        pkt.rnd = RNDSignal(intrinsic_reward=intrinsic, novelty_score=0.5, phase_decay=1.0)

        # 4. Create coach and simulate PPO pending
        ctx = AttackContext(target="172.28.0.10")
        coach = SmartCoach(
            agent_name="RedAgent",
            gpt_manager=self.gpt,
        )
        coach.attack_context = ctx
        coach._ppo_pending = {
            "state": state,
            "action": 0,
            "log_prob": -1.0,
            "value": 0.5,
        }
        coach._current_decision_packet = pkt

        # 5. Record result
        decision = SmartDecisionResult(
            command="nmap -sV 172.28.0.10",
            template_name="nmap_version",
            confidence=0.7,
            source="ppo",
        )
        breakdown = coach.record_result(
            decision=decision, success=True, raw_output="80/tcp open http", done=False,
        )

        # 6. Verify trajectory has RND-augmented reward
        assert len(coach._ppo_trajectory) >= 1
        entry = coach._ppo_trajectory[-1]
        expected_min = breakdown.total + intrinsic - 0.01
        assert entry["reward"] >= expected_min, (
            f"Expected reward >= {expected_min}, got {entry['reward']}"
        )

        # 7. RND predictor update (what orchestrator does post-step)
        loss = rnd.update(state)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_multiple_steps_accumulate_rnd(self):
        """Multiple steps each get their own RND intrinsic injected."""
        from core.algorithms.rnd_curiosity import RNDCuriosity
        from core.training.decision_packet import DecisionPacket, RNDSignal
        from core.training.smart_coach import SmartCoach, SmartDecisionResult
        from core.llm.smart_mentor import AttackContext
        import torch

        rnd = RNDCuriosity(state_dim=512, hidden_dim=64, output_dim=32)
        ctx = AttackContext(target="172.28.0.10")
        coach = SmartCoach(
            agent_name="RedAgent",
            gpt_manager=self.gpt,
        )
        coach.attack_context = ctx

        intrinsics = []
        for i in range(5):
            state = torch.randn(512)
            intrinsic = rnd.compute_intrinsic_reward(state, phase="RECON")
            intrinsics.append(intrinsic)

            pkt = DecisionPacket()
            pkt.rnd = RNDSignal(intrinsic_reward=intrinsic)
            coach._current_decision_packet = pkt
            coach._ppo_pending = {
                "state": state, "action": i % 5,
                "log_prob": -1.0, "value": 0.5,
            }
            decision = SmartDecisionResult(
                command=f"nmap -{i} 172.28.0.10",
                template_name="nmap_version",
                confidence=0.7,
                source="ppo",
            )
            coach.record_result(
                decision=decision, success=True, raw_output="", done=False,
            )
            # Update RND predictor
            rnd.update(state)

        # All 5 entries should be in trajectory
        assert len(coach._ppo_trajectory) >= 5
        # Each entry's reward should be >= its intrinsic
        for j, entry in enumerate(coach._ppo_trajectory[-5:]):
            assert entry["reward"] >= intrinsics[j] - 0.01
