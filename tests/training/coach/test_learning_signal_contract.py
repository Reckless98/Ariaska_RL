"""
tests/training/coach/test_learning_signal_contract.py — Learning-signal integrity.

Pins the Phase-1 refactor contract between SmartCoach and its learners:

  LS-01  the executed PPO command is paired with ITS OWN (state, action,
         log_prob, value) sample even when later producers re-sample PPO
  LS-04  PPO's teacher_action is the mentor's REAL action index (never the
         student's own sample) when a mentor template maps to an action
  LS-05  every accepted mentor intervention becomes a TeacherTrace that is
         finalized with the executed step's outcome and stored in BCBuffer
  LS-07  SkillCard auto-promote / bridge outcome hooks run for mentor-won steps
  LS-09  episode-end side effects run even when the PPO trajectory is empty
  LS-10  the postmortem transcript is built before the trajectory is cleared
  F3     registry-family decisions declare source="registry"
  ledger MentorLedger counts proposals / rejections / wins / outcomes

Deterministic: FakeGPTManager, no network, CPU torch.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── Pure units ───────────────────────────────────────────────────────────────

class TestTeacherTraceOutcome:
    def test_finalize_mentor_won_success(self):
        from core.reasoning.teacher_trace import TeacherTrace
        tt = TeacherTrace(teacher_action_idx=2, student_action_idx=1, confidence=0.8)
        assert tt.finalized is False and tt.teacher_was_correct is None
        tt.finalize_outcome(actual_reward=6.5, produced_discovery=False, mentor_won=True, actual_success=True)
        assert tt.finalized and tt.mentor_won is True
        assert tt.actual_reward == 6.5 and tt.teacher_was_correct is True

    def test_finalize_mentor_won_failure_and_lost(self):
        from core.reasoning.teacher_trace import TeacherTrace
        tt = TeacherTrace(teacher_action_idx=2, confidence=0.8)
        tt.finalize_outcome(actual_reward=-3.0, produced_discovery=False, mentor_won=True, actual_success=False)
        assert tt.teacher_was_correct is False
        tt2 = TeacherTrace(teacher_action_idx=2, confidence=0.8)
        tt2.finalize_outcome(actual_reward=9.0, produced_discovery=True, mentor_won=False)
        assert tt2.mentor_won is False and tt2.teacher_was_correct is None  # no counterfactual

    def test_backward_compatible_from_dict_and_compact(self):
        from core.reasoning.teacher_trace import TeacherTrace
        legacy = {"state_id": "x", "teacher_action_idx": 1, "teacher_command": "nmap", "confidence": 0.5,
                  "episode": 3, "step": 7, "agent_id": "RedAgent", "phase": "RECON"}
        tt = TeacherTrace.from_dict(legacy)
        assert tt.finalized is False and tt.actual_reward is None
        tt.finalize_outcome(1.0, False, True, actual_success=True)
        d = tt.to_compact()
        assert d["finalized"] is True and d["mentor_won"] is True and "state_vector" not in d
        assert TeacherTrace.from_dict(d).teacher_was_correct is True

    def test_bc_sample_weight_unchanged_by_outcome_fields(self):
        """BCSample content is identical to legacy: weight == confidence * decay."""
        from core.reasoning.teacher_trace import TeacherTrace, BCBuffer
        tt = TeacherTrace(state_id="s1", state_vector=[0.0] * 512, teacher_action_idx=1, confidence=0.6)
        tt.finalize_outcome(-5.0, False, True)
        buf = BCBuffer(capacity=10)
        assert buf.store(tt, decay_factor=1.0) is True
        sample = buf.sample(1)[0]
        assert sample.weight == pytest.approx(0.6) and sample.teacher_action == 1


class TestMentorLedger:
    def test_counts_and_rates(self):
        from core.training.mentor_ledger import MentorLedger
        from core.reasoning.teacher_trace import TeacherTrace
        led = MentorLedger()
        led.record_rejection("role_violation")
        led.record_rejection("exact_repeat")
        led.record_accepted()
        led.record_accepted()
        assert led.proposed == 4 and led.accepted == 2 and led.rejected == 2
        assert led.rejected_by_reason == {"role_violation": 1, "exact_repeat": 1}
        assert led.acceptance_rate == pytest.approx(0.5)

        won = TeacherTrace(teacher_action_idx=1)
        won.finalize_outcome(5.0, True, mentor_won=True, actual_success=True)
        lost = TeacherTrace(teacher_action_idx=1)
        lost.finalize_outcome(2.0, False, mentor_won=False)
        led.record_outcome("mentor", 5.0, True, phase="RECON", trace=won, bc_stored=True)
        led.record_outcome("ppo", 2.0, False, phase="RECON", trace=lost, bc_stored=True)
        led.record_outcome("ppo", 1.0, False, phase="RECON")  # plain policy step, no trace
        s = led.summary()
        assert s["won"] == 1 and s["lost"] == 1 and s["win_rate"] == pytest.approx(0.5)
        assert s["correct_when_won"] == 1 and s["correct_rate"] == pytest.approx(1.0)
        assert s["finalized_traces"] == 2 and s["bc_samples_stored"] == 2
        assert s["mentor_steps"]["RECON"]["count"] == 1 and s["policy_steps"]["RECON"]["count"] == 2
        assert s["mentor_vs_policy_reward_delta"] == pytest.approx(5.0 - 1.5)
        led.reset_episode()
        assert led.summary()["episode"] == {"proposed": 0, "won": 0} and led.proposed == 4

    def test_outcome_without_trace_infers_from_source(self):
        from core.training.mentor_ledger import MentorLedger
        led = MentorLedger()
        led.record_outcome("mentor", -1.0, False)
        assert led.won == 1 and led.correct_when_won == 0
        assert led.reward_delta("ALL") is None  # no policy steps yet


# ── SmartCoach-level contract (FakeGPTManager, real coach) ───────────────────

def _make_coach(agent_name: str = "RedAgent"):
    from core.testing.fake_gpt_manager import FakeGPTManager
    from core.training.smart_coach import SmartCoach
    coach = SmartCoach(agent_name=agent_name, gpt_manager=FakeGPTManager(seed=42))
    coach.init_attack_context(target="192.168.56.20", difficulty="medium", platform="linux")
    return coach


def _step_ctx(coach, step: int = 1):
    from core.training.smart_coach import SmartStepContext
    return SmartStepContext(episode=0, step=step, agent_name=coach.agent_name,
                            attack_context=coach.attack_context, state={})


class TestRegistrySource:
    def test_registry_decision_declares_source(self):
        coach = _make_coach("RedAgent")
        coach.ppo_agent = None  # force the pure registry path
        res = coach._decide_from_registry(_step_ctx(coach), None, 0.5)
        assert res is not None and res.source == "registry"

    def test_blue_and_orion_registry_paths_declare_source(self):
        coach = _make_coach("BlueAgent")
        blue = coach._get_blue_agent_command(coach.attack_context)
        assert blue is not None and blue.source == "registry"
        coach_o = _make_coach("OrionAgent")
        orion = coach_o._get_orion_coordination(coach_o.attack_context, step=1)
        assert orion.source == "registry"


class TestPPOPairing:
    def test_restore_pairing_helper(self):
        coach = _make_coach("RedAgent")
        ppo_res = SimpleNamespace(source="ppo")
        snapshot = {"state": "s", "action": 3, "log_prob": -0.3, "value": 0.1}
        coach._ppo_pending = {"state": "s2", "action": 1, "log_prob": -1.1, "value": 0.2}  # overwritten by a re-sample
        assert coach._restore_ppo_pairing(ppo_res, ppo_res, snapshot) is True
        assert coach._ppo_pending["action"] == 3 and coach._ppo_pending["log_prob"] == -0.3
        # already paired → no-op; different winner → no-op
        assert coach._restore_ppo_pairing(ppo_res, ppo_res, snapshot) is False
        other = SimpleNamespace(source="mentor")
        coach._ppo_pending = {"action": 9, "log_prob": -9.0}
        assert coach._restore_ppo_pairing(other, ppo_res, snapshot) is False
        assert coach._ppo_pending["action"] == 9

    def test_executed_ppo_command_matches_stored_sample(self, monkeypatch):
        """End-to-end: force the PPO advisory to win while later producers re-sample PPO."""
        coach = _make_coach("RedAgent")
        if coach.ppo_agent is None or coach.action_mapper is None or coach._decision_core is None:
            pytest.skip("PPO/DecisionCore not available for RedAgent in this configuration")

        real_select = coach.ppo_agent.select_action
        calls: List[Dict[str, Any]] = []
        counter = {"n": 0}

        def _select(state_tensor, training=True, action_mask=None, **kw):
            # Rotate through DIFFERENT actions with distinct log_probs so any
            # pairing mix-up between the 1st and later samples is detectable.
            n = counter["n"]
            counter["n"] += 1
            dim = int(getattr(coach.ppo_agent.config, "action_dim", 5))
            if action_mask is not None:
                allowed = [i for i in range(dim) if bool(action_mask[i])] or list(range(dim))
            else:
                allowed = list(range(dim))
            action = allowed[n % len(allowed)]
            lp = -0.5 - 0.25 * n
            calls.append({"action": action, "log_prob": lp})
            return action, lp, 0.1 * n

        monkeypatch.setattr(coach.ppo_agent, "select_action", _select)

        from core.decision.decision_core import ArbitrationResult

        def _arbitrate(advisories, **kwargs):
            ppo_adv = [a for a in advisories if a.source == "ppo"]
            assert ppo_adv, "PPO advisory must be present"
            a = ppo_adv[0]
            return ArbitrationResult(command=a.command, template_name=a.template_name,
                                     source="ppo", confidence=a.confidence)

        monkeypatch.setattr(coach._decision_core, "arbitrate", _arbitrate)

        decision = coach.decide(_step_ctx(coach), proposed_action=None, confidence=0.8)
        assert decision is not None
        if decision.source != "ppo" or coach._ppo_pending is None:
            pytest.skip(f"post-arbitration mutator replaced the PPO decision (source={decision.source})")
        expected_action = coach.action_mapper.command_to_action(decision.template_name)
        assert expected_action >= 0
        assert coach._ppo_pending["action"] == expected_action, (
            f"executed template {decision.template_name!r} (action {expected_action}) is paired "
            f"with stored action {coach._ppo_pending['action']} — samples: {calls}"
        )
        # If PPO was sampled more than once, the stored log_prob must be the one
        # that produced the executed command (not the last sample's).
        matching = [c for c in calls if c["action"] == expected_action]
        assert any(c["log_prob"] == coach._ppo_pending["log_prob"] for c in matching)
        if len(calls) > 1 and calls[-1]["action"] != expected_action:
            assert coach._mentor_ledger.ppo_pairing_restored >= 1


class TestMentorOutcomeLedgerEndToEnd:
    def _force_mentor(self, coach, monkeypatch, template_name: str, command: str):
        from core.llm.smart_mentor import MentorResponse
        from core.commands.command_registry import AttackPhase

        def _get_command(ctx, filtered=None):
            return MentorResponse(command=command, template_name=template_name, params={},
                                  reasoning="test mentor", confidence=0.9, phase=AttackPhase.RECON,
                                  tokens_used=10)
        if coach.smart_mentor is None:
            # DETERMINISTIC profile builds no SmartMentor; a minimal stand-in exposes
            # exactly the surface SmartCoach touches.
            coach.smart_mentor = SimpleNamespace(get_command=_get_command, target_profile="generic",
                                                 record_result=lambda *a, **k: None)
        else:
            monkeypatch.setattr(coach.smart_mentor, "get_command", _get_command)
        if coach._maturity_controller is not None:
            monkeypatch.setattr(coach._maturity_controller, "should_call_mentor", lambda *_a, **_k: True)
        else:
            import random as _r
            monkeypatch.setattr(_r, "random", lambda: 0.0)

    def test_mentor_intervention_is_measured(self, monkeypatch):
        coach = _make_coach("RedAgent")
        if coach._p14_bc_buffer is None or coach.ppo_agent is None:
            pytest.skip("BCBuffer/PPO not configured for RedAgent in this profile")
        # pick a real offensive template the mapper knows about
        from core.commands.command_registry import COMMAND_REGISTRY, render_command
        chosen = None
        for name, tpl in COMMAND_REGISTRY.items():
            if coach.action_mapper is not None and coach.action_mapper.command_to_action(name) >= 0 \
                    and not tpl.preconditions:
                chosen = (name, tpl)
                break
        if chosen is None:
            pytest.skip("no mapper-known precondition-free template")
        name, tpl = chosen
        params = {p: coach._get_default_param(p, coach.attack_context) for p in tpl.required_params}
        params.update(tpl.optional_params)
        command = render_command(tpl, params)
        self._force_mentor(coach, monkeypatch, name, command)

        bc_before = len(coach._p14_bc_buffer)
        decision = coach.decide(_step_ctx(coach), proposed_action=None, confidence=0.1)
        assert decision is not None
        ledger = coach._mentor_ledger
        if ledger.proposed == 0:
            pytest.skip("mentor gate did not open in this configuration (budget/phase gate)")
        assert ledger.accepted >= 1
        trace = coach._pending_teacher_trace
        assert trace is not None, "accepted mentor answer must leave a TeacherTrace awaiting its outcome"
        assert len(coach._p14_bc_buffer) == bc_before, "BC storage is deferred until the outcome is known"
        expected_teacher = coach.action_mapper.command_to_action(name)
        if coach._ppo_pending is not None and expected_teacher >= 0:
            assert coach._ppo_pending.get("teacher_action") == expected_teacher  # LS-04

        coach.record_result(decision=decision, success=True, raw_output="80/tcp open http",
                            new_discoveries={"service": ["http"]}, done=False)
        assert trace.finalized is True and trace.produced_discovery is True
        assert trace.mentor_won == (decision.source == "mentor")
        assert len(coach._p14_bc_buffer) == bc_before + 1
        assert coach._pending_teacher_trace is None
        s = coach.get_stats()["mentor_ledger"]
        assert s["finalized_traces"] == 1 and s["bc_samples_stored"] == 1
        assert (s["won"] + s["lost"]) == 1

    def test_rejected_mentor_answers_are_counted(self, monkeypatch):
        coach = _make_coach("RedAgent")
        from core.gpt_manager import GPTManager
        placeholder = GPTManager.OFFLINE_SENTINEL + " reasoning unavailable."
        self._force_mentor(coach, monkeypatch, "nmap_version", placeholder)
        res = coach._decide_with_mentor(_step_ctx(coach), None, 0.5)
        assert res is not None and res.mentor_call is False
        assert coach._mentor_ledger.rejected_by_reason.get("offline_placeholder") == 1
        assert coach._pending_teacher_trace is None


class TestEpisodeEndSideEffects:
    def test_cleanup_runs_with_empty_trajectory(self):
        coach = _make_coach("RedAgent")
        calls = {"ddqn": 0, "cog": 0}
        coach.ddqn_macro = SimpleNamespace(reset_episode=lambda: calls.__setitem__("ddqn", calls["ddqn"] + 1))
        coach.cognition_node = SimpleNamespace(
            end_episode=lambda: (calls.__setitem__("cog", calls["cog"] + 1) or {}),
            reset_episode=lambda: None,
        )
        coach._ppo_trajectory = []
        coach._active_macro = "X"
        assert coach.end_episode_ppo(done=True, highest_phase="RECON") is None
        assert calls["ddqn"] == 1 and calls["cog"] == 1, "LS-09: side effects must run without a PPO update"
        assert coach._active_macro is None

    def test_postmortem_transcript_built_before_clear(self, monkeypatch):
        import torch
        coach = _make_coach("RedAgent")
        if coach.ppo_agent is None:
            pytest.skip("no PPO agent")
        seen: Dict[str, Any] = {}

        class _Extractor:
            def can_call(self): return True
            def extract_skills(self, transcript, total_reward, highest_phase):
                seen["transcript"] = transcript
                return []
        coach._postmortem_extractor = _Extractor()
        monkeypatch.setattr(coach.ppo_agent, "update", lambda *a, **k: {})
        coach._ppo_trajectory = [{
            "state": torch.zeros(512), "action": 0, "log_prob": -1.0, "value": 0.0,
            "reward": 2.0, "done": True, "_r69_template": "nmap_version", "_r69_discoveries": [],
            "_r69_preconditions": set(),
        }]
        coach.end_episode_ppo(done=True, highest_phase="RECON")
        assert "nmap_version" in seen.get("transcript", ""), "LS-10: transcript must reflect the episode"
        assert coach._ppo_trajectory == []
