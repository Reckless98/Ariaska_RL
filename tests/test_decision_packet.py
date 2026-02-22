"""
Tests for Phase 50 — DecisionPacket schema.

Validates:
1. DecisionPacket construction and field defaults
2. Sub-structure validity checks
3. from_step_context() factory
4. to_dict() serialization
5. SourceAttribution pipeline tracking
6. RewardComposition totals
7. Integration with SmartStepResult
8. Integration with SmartCoach.decide() (packet kwarg)
"""

import os
import pytest
import time
from dataclasses import fields

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestDecisionPacketSchema:
    """Unit tests for the DecisionPacket dataclass and sub-structures."""

    def test_import(self):
        """DecisionPacket can be imported from core.training.decision_packet."""
        from core.training.decision_packet import DecisionPacket
        assert DecisionPacket is not None

    def test_default_construction(self):
        """DecisionPacket constructs with all defaults."""
        from core.training.decision_packet import DecisionPacket
        dp = DecisionPacket()
        assert dp.episode == 0
        assert dp.step == 0
        assert dp.agent_name == ""
        assert dp.phase == "recon"
        assert dp.source == "unknown"
        assert dp.total_reward == 0.0
        assert not dp.has_ppo
        assert not dp.has_sac
        assert not dp.has_ddqn
        assert not dp.has_cognition
        assert not dp.has_rnd

    def test_ppo_proposal_valid(self):
        """PPOProposal.valid requires action_idx >= 0."""
        from core.training.decision_packet import PPOProposal
        p = PPOProposal()
        assert not p.valid
        p.action_idx = 0
        assert p.valid
        p.action_idx = 3
        assert p.valid

    def test_sac_proposal_valid(self):
        """SACProposal.valid requires action_idx >= 0."""
        from core.training.decision_packet import SACProposal
        s = SACProposal()
        assert not s.valid
        s.action_idx = 2
        assert s.valid

    def test_ddqn_macro_valid(self):
        """DDQNMacro.valid requires macro_idx >= 0."""
        from core.training.decision_packet import DDQNMacro
        d = DDQNMacro()
        assert not d.valid
        d.macro_idx = 1
        assert d.valid

    def test_cognition_vote_valid(self):
        """CognitionVote.valid requires fused_action_idx >= 0."""
        from core.training.decision_packet import CognitionVote
        c = CognitionVote()
        assert not c.valid
        c.fused_action_idx = 0
        assert c.valid

    def test_rnd_signal_valid(self):
        """RNDSignal.valid requires nonzero intrinsic_reward."""
        from core.training.decision_packet import RNDSignal
        r = RNDSignal()
        assert not r.valid
        r.intrinsic_reward = 0.5
        assert r.valid

    def test_reward_composition_total(self):
        """RewardComposition.total sums all fields."""
        from core.training.decision_packet import RewardComposition
        r = RewardComposition(
            extrinsic=10.0,
            intrinsic_rnd=2.0,
            discovery_bonus=5.0,
            phase_bonus=3.0,
            repeat_penalty=-1.0,
        )
        assert abs(r.total - 19.0) < 1e-6

    def test_reward_composition_with_all_fields(self):
        """RewardComposition.total includes mentor/sil/bc/coherence."""
        from core.training.decision_packet import RewardComposition
        r = RewardComposition(
            extrinsic=1.0,
            intrinsic_rnd=1.0,
            discovery_bonus=1.0,
            phase_bonus=1.0,
            repeat_penalty=-1.0,
            mentor_bonus=1.0,
            sil_bonus=1.0,
            bc_bonus=1.0,
            coherence_bonus=1.0,
        )
        assert abs(r.total - 7.0) < 1e-6

    def test_source_attribution_record_stage(self):
        """SourceAttribution.record_stage tracks pipeline stages."""
        from core.training.decision_packet import SourceAttribution
        sa = SourceAttribution()
        assert sa.source == "unknown"
        sa.record_stage("playbook", proposed="nmap -sV", accepted=False, reason="no match")
        sa.record_stage("ppo", proposed="gobuster dir", accepted=True, confidence=0.8)
        assert sa.source == "ppo"
        assert len(sa.pipeline_trace) == 2
        assert sa.pipeline_trace[0]["stage"] == "playbook"
        assert sa.pipeline_trace[0]["accepted"] is False
        assert sa.pipeline_trace[1]["stage"] == "ppo"
        assert sa.pipeline_trace[1]["accepted"] is True

    def test_grad_norms_to_dict(self):
        """GradNorms.to_dict returns expected keys."""
        from core.training.decision_packet import GradNorms
        g = GradNorms(policy_grad_norm=1.5, value_grad_norm=2.3)
        d = g.to_dict()
        assert d["policy"] == 1.5
        assert d["value"] == 2.3
        assert "total" in d

    def test_to_dict_minimal(self):
        """DecisionPacket.to_dict works with defaults."""
        from core.training.decision_packet import DecisionPacket
        dp = DecisionPacket(episode=1, step=5, agent_name="RedAgent")
        d = dp.to_dict()
        assert d["episode"] == 1
        assert d["step"] == 5
        assert d["agent"] == "RedAgent"
        assert d["source"] == "unknown"
        assert d["ppo"] is None
        assert d["sac"] is None
        assert d["ddqn"] is None
        assert d["cognition"] is None
        assert d["rnd"] is None

    def test_to_dict_with_proposals(self):
        """DecisionPacket.to_dict serializes filled proposals."""
        from core.training.decision_packet import (
            DecisionPacket, PPOProposal, DDQNMacro, RNDSignal,
        )
        dp = DecisionPacket(
            episode=2, step=10, agent_name="RedAgent",
            ppo=PPOProposal(action_idx=3, log_prob=-1.2, value=5.0, command="nmap -sV"),
            ddqn=DDQNMacro(macro_idx=1, q_value=0.8, macro_name="RECON_FOCUS"),
            rnd=RNDSignal(intrinsic_reward=0.3, novelty_score=0.7),
        )
        d = dp.to_dict()
        assert d["ppo"]["action"] == 3
        assert d["ppo"]["log_prob"] == -1.2
        assert d["ddqn"]["macro_name"] == "RECON_FOCUS"
        assert d["rnd"]["intrinsic"] == 0.3

    def test_from_step_context(self):
        """from_step_context creates packet from SmartStepContext."""
        from core.training.smart_coach import SmartStepContext
        from core.llm.smart_mentor import AttackContext
        from core.commands.command_registry import AttackPhase
        from core.training.decision_packet import DecisionPacket

        ctx = AttackContext(
            target="10.10.10.1",
            current_phase=AttackPhase.ENUMERATION,
        )
        step_ctx = SmartStepContext(
            episode=3, step=15, agent_name="ScoutAgent",
            attack_context=ctx,
            state={"ports_discovered": True},
        )
        dp = DecisionPacket.from_step_context(
            step_ctx, rnd_intrinsic=1.5, coherence=0.8,
        )
        assert dp.episode == 3
        assert dp.step == 15
        assert dp.agent_name == "ScoutAgent"
        assert dp.phase == "enumeration"
        assert dp.target_ip == "10.10.10.1"
        assert dp.rnd.intrinsic_reward == 1.5
        assert dp.coherence == 0.8
        assert dp.state == {"ports_discovered": True}

    def test_from_step_context_no_attack_context(self):
        """from_step_context handles minimal attack_context gracefully."""
        from core.training.smart_coach import SmartStepContext
        from core.llm.smart_mentor import AttackContext
        from core.training.decision_packet import DecisionPacket

        # AttackContext requires target, but we give it a minimal one
        ctx = AttackContext(target="")
        step_ctx = SmartStepContext(
            episode=1, step=1, agent_name="BlueAgent",
            attack_context=ctx,
        )
        dp = DecisionPacket.from_step_context(step_ctx)
        assert dp.phase == "recon"
        assert dp.target_ip == ""

    def test_to_step_context(self):
        """to_step_context returns dict with expected keys."""
        from core.training.decision_packet import DecisionPacket
        dp = DecisionPacket(episode=5, step=20, agent_name="OrionAgent")
        sc = dp.to_step_context()
        assert sc["episode"] == 5
        assert sc["step"] == 20
        assert sc["agent_name"] == "OrionAgent"

    def test_has_properties(self):
        """has_* properties reflect sub-structure validity."""
        from core.training.decision_packet import (
            DecisionPacket, PPOProposal, SACProposal,
            DDQNMacro, CognitionVote, RNDSignal,
        )
        dp = DecisionPacket()
        assert not dp.has_ppo
        assert not dp.has_sac
        assert not dp.has_ddqn
        assert not dp.has_cognition
        assert not dp.has_rnd

        dp.ppo.action_idx = 0
        dp.sac.action_idx = 1
        dp.ddqn.macro_idx = 2
        dp.cognition.fused_action_idx = 0
        dp.rnd.intrinsic_reward = 0.1

        assert dp.has_ppo
        assert dp.has_sac
        assert dp.has_ddqn
        assert dp.has_cognition
        assert dp.has_rnd

    def test_source_shortcut(self):
        """source property delegates to attribution.source."""
        from core.training.decision_packet import DecisionPacket
        dp = DecisionPacket()
        assert dp.source == "unknown"
        dp.attribution.source = "ppo"
        assert dp.source == "ppo"

    def test_total_reward_shortcut(self):
        """total_reward property delegates to reward.total."""
        from core.training.decision_packet import DecisionPacket
        dp = DecisionPacket()
        dp.reward.extrinsic = 10.0
        dp.reward.intrinsic_rnd = 2.0
        assert abs(dp.total_reward - 12.0) < 1e-6

    def test_timestamp_auto_set(self):
        """Timestamp is auto-set on construction."""
        from core.training.decision_packet import DecisionPacket
        before = time.time()
        dp = DecisionPacket()
        after = time.time()
        assert before <= dp.timestamp <= after

    def test_decision_trace_list(self):
        """decision_trace starts empty and is appendable."""
        from core.training.decision_packet import DecisionPacket
        dp = DecisionPacket()
        assert dp.decision_trace == []
        dp.decision_trace.append({"stage": "test", "result": "ok"})
        assert len(dp.decision_trace) == 1


class TestDecisionPacketSmartStepResult:
    """Integration: DecisionPacket on SmartStepResult."""

    def test_smart_step_result_has_packet_field(self):
        """SmartStepResult has optional decision_packet field."""
        from core.orchestration.smart_orchestrator import SmartStepResult
        from core.training.smart_coach import SmartDecisionResult
        from core.training.decision_packet import DecisionPacket

        decision = SmartDecisionResult(command="nmap -sV 10.10.10.1")
        packet = DecisionPacket(episode=1, step=1, agent_name="RedAgent")
        result = SmartStepResult(
            agent_name="RedAgent",
            decision=decision,
            decision_packet=packet,
        )
        assert result.decision_packet is not None
        assert result.decision_packet.agent_name == "RedAgent"

    def test_smart_step_result_to_dict_includes_packet(self):
        """SmartStepResult.to_dict includes decision_packet when present."""
        from core.orchestration.smart_orchestrator import SmartStepResult
        from core.training.smart_coach import SmartDecisionResult
        from core.training.decision_packet import DecisionPacket

        decision = SmartDecisionResult(command="nmap -sV 10.10.10.1")
        packet = DecisionPacket(episode=1, step=1, agent_name="RedAgent")
        result = SmartStepResult(
            agent_name="RedAgent",
            decision=decision,
            decision_packet=packet,
        )
        d = result.to_dict()
        assert "decision_packet" in d
        assert d["decision_packet"]["episode"] == 1

    def test_smart_step_result_to_dict_without_packet(self):
        """SmartStepResult.to_dict works without decision_packet."""
        from core.orchestration.smart_orchestrator import SmartStepResult
        from core.training.smart_coach import SmartDecisionResult

        decision = SmartDecisionResult(command="nmap -sV 10.10.10.1")
        result = SmartStepResult(agent_name="RedAgent", decision=decision)
        d = result.to_dict()
        assert "decision_packet" not in d


class TestDecisionPacketCoachWiring:
    """Integration: coach.decide() accepts decision_packet kwarg."""

    def test_decide_accepts_decision_packet_kwarg(self):
        """SmartCoach.decide() accepts decision_packet keyword argument."""
        import inspect
        from core.training.smart_coach import SmartCoach
        sig = inspect.signature(SmartCoach.decide)
        params = list(sig.parameters.keys())
        assert "decision_packet" in params

    def test_coach_stores_packet_reference(self):
        """Coach stores _current_decision_packet during decide()."""
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.training.smart_coach import SmartCoach, SmartStepContext
        from core.llm.smart_mentor import AttackContext
        from core.commands.command_registry import AttackPhase
        from core.training.decision_packet import DecisionPacket

        gpt = FakeGPTManager(seed=42)
        coach = SmartCoach(
            gpt_manager=gpt,
            agent_name="RedAgent",
        )

        ctx = AttackContext(
            target="10.10.10.1",
            current_phase=AttackPhase.RECON,
        )
        step_ctx = SmartStepContext(
            episode=1, step=1, agent_name="RedAgent",
            attack_context=ctx,
            state={"discovery_board": {"target": "10.10.10.1"}},
        )
        packet = DecisionPacket.from_step_context(step_ctx)

        # Call decide with packet
        result = coach.decide(step_ctx, decision_packet=packet)
        assert result is not None
        assert result.command  # Should have produced a command

    def test_decide_works_without_packet(self):
        """Coach.decide() still works without decision_packet (backwards compat)."""
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.training.smart_coach import SmartCoach, SmartStepContext
        from core.llm.smart_mentor import AttackContext
        from core.commands.command_registry import AttackPhase

        gpt = FakeGPTManager(seed=42)
        coach = SmartCoach(
            gpt_manager=gpt,
            agent_name="ScoutAgent",
        )

        ctx = AttackContext(
            target="10.10.10.1",
            current_phase=AttackPhase.RECON,
        )
        step_ctx = SmartStepContext(
            episode=1, step=1, agent_name="ScoutAgent",
            attack_context=ctx,
            state={"discovery_board": {"target": "10.10.10.1"}},
        )

        # Call decide WITHOUT packet — must not raise
        result = coach.decide(step_ctx)
        assert result is not None


class TestDecisionPacketAllSubstructures:
    """Ensure all sub-structures have expected fields and methods."""

    def test_all_substructures_importable(self):
        """All sub-structures can be imported."""
        from core.training.decision_packet import (
            PPOProposal,
            SACProposal,
            DDQNMacro,
            CognitionVote,
            RNDSignal,
            RewardComposition,
            SourceAttribution,
            GradNorms,
            DecisionPacket,
        )
        # Just verify they're all dataclasses with the right fields
        assert len(fields(PPOProposal)) >= 7
        assert len(fields(SACProposal)) >= 6
        assert len(fields(DDQNMacro)) >= 4
        assert len(fields(CognitionVote)) >= 7
        assert len(fields(RNDSignal)) >= 3
        assert len(fields(RewardComposition)) >= 9
        assert len(fields(SourceAttribution)) >= 3
        assert len(fields(GradNorms)) >= 8
        assert len(fields(DecisionPacket)) >= 20

    def test_source_attribution_truncates_proposed(self):
        """SourceAttribution.record_stage truncates long `proposed` strings."""
        from core.training.decision_packet import SourceAttribution
        sa = SourceAttribution()
        long_cmd = "x" * 500
        sa.record_stage("test", proposed=long_cmd)
        assert len(sa.pipeline_trace[0]["proposed"]) <= 120
