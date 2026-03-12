#!/usr/bin/env python3
"""Tests for core/llm/phase_guided_llm.py — Phase 34 PhaseGuidedLLM."""

import json
import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── Stub GPT ────────────────────────────────────────────────────────────────

class _StubGPT:
    """Minimal GPT stub for PhaseGuidedLLM tests."""

    def __init__(self, response: str = "", budget_ok: bool = True):
        self._response = response
        self._budget_ok = budget_ok
        self._calls: list = []

    def can_make_request(self, **kw) -> bool:
        return self._budget_ok

    def gpt_request(self, prompt: str, task_type: str = "", agent_id: str = "",
                    max_tokens: int = 100, model: str | None = None,
                    system_prompt: str | None = None, **kw) -> str:
        self._calls.append({"model": model, "prompt": prompt[:100]})
        return self._response


def _good_response(**overrides) -> str:
    """Build a valid PhaseGuidedLLM JSON response."""
    data = {
        "phase_decision": {
            "chosen_phase": "RECON",
            "phase_confidence": 0.75,
            "phase_goal": "Discover open services",
            "stay_conditions": ["fewer than 5 ports"],
            "move_on_conditions": ["3+ services confirmed"],
            "contradictions": [],
            "phase_tag": "P34",
        },
        "anomalies": [
            {"finding": "unusual port 9999", "why_interesting": "non-standard",
             "probe_template": "nmap_version", "risk": "low"},
        ],
        "candidates": [
            {"template_name": "nmap_full", "family": "nmap",
             "why": "Need full port scan", "expected_outcome": "port list",
             "stop_condition": "scan complete", "confidence": 0.8,
             "risk": "low", "tags": ["EVIDENCE_DRIVEN"]},
            {"template_name": "nmap_scripts", "family": "nmap",
             "why": "Script scan for services", "expected_outcome": "service details",
             "stop_condition": "no new services", "confidence": 0.7,
             "risk": "low", "tags": ["EVIDENCE_DRIVEN"]},
            {"template_name": "gobuster_dir", "family": "web",
             "why": "Web path discovery", "expected_outcome": "web paths",
             "stop_condition": "no 200 responses", "confidence": 0.6,
             "risk": "low", "tags": ["EVIDENCE_DRIVEN"]},
        ],
        "selection": {
            "best_template_name": "nmap_full",
            "runner_up_template_name": "nmap_scripts",
            "selection_reason": "Full port scan needed first",
            "should_escalate_to_codex": False,
            "escalation_reason": "",
        },
        "distillation_packet": {
            "observation": "2 ports, 1 service discovered",
            "reasoning": "Sparse evidence, need comprehensive scan",
            "action_target": {"template_name": "nmap_full", "why": "discovery"},
            "expected_outcome": "Complete port inventory",
            "phase_target": "RECON",
            "confidence_target": 0.75,
            "gating_notes": {
                "expected_gate_result": "PASS",
                "reasons": ["Evidence supports RECON"],
            },
            "phase_tag": "P34",
        },
    }
    data.update(overrides)
    return json.dumps(data)


def _make_board(**overrides) -> dict:
    """Build a minimal discovery board."""
    board = {
        "ports": [22, 80],
        "services": ["ssh", "http"],
        "web_paths": [],
        "users": [],
        "creds": [],
        "versions": [],
        "vulns": [],
        "shells": [],
        "flags": [],
        "notes": [],
    }
    board.update(overrides)
    return board


def _make_phase_state(**overrides) -> dict:
    board = {
        "phase_progress_score": 0.2,
        "steps_in_phase": 5,
        "stagnation_steps": 0,
        "recent_discovery_deltas": [1, 0, 0, 1, 0],
        "recent_commands": ["nmap -sS 10.0.0.1"],
    }
    board.update(overrides)
    return board


# ── Tests ────────────────────────────────────────────────────────────────────

class TestPhaseGuidedLLM:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.board = _make_board()
        self.phase_state = _make_phase_state()
        self.templates = [
            {"name": "nmap_full", "family": "nmap", "phase_fit": ["RECON", "ENUM"], "risk": "low"},
            {"name": "nmap_scripts", "family": "nmap", "phase_fit": ["ENUM"], "risk": "low"},
            {"name": "gobuster_dir", "family": "web", "phase_fit": ["ENUM"], "risk": "low"},
        ]

    def test_guide_returns_valid_result(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(response=_good_response())
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=5, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        assert result.validate()

    def test_phase_tag_enforced(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM, _PHASE_TAG
        gpt = _StubGPT(response=_good_response())
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=1, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        d = result.to_dict()
        assert d["phase_decision"]["phase_tag"] == _PHASE_TAG
        assert d["distillation_packet"]["phase_tag"] == _PHASE_TAG

    def test_phase_tag_forced_when_missing(self):
        """If LLM omits phase_tag, we force it."""
        from core.llm.phase_guided_llm import PhaseGuidedLLM, _PHASE_TAG
        bad = json.loads(_good_response())
        del bad["phase_decision"]["phase_tag"]
        del bad["distillation_packet"]["phase_tag"]
        gpt = _StubGPT(response=json.dumps(bad))
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=1, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        assert result.validate()  # Should have been forced

    def test_budget_exhausted_returns_none(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(budget_ok=False)
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=1, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is None

    def test_malformed_json_triggers_heuristic_fallback(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(response="this is not json at all")
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=1, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        assert result.validate()
        assert result.selection.should_escalate_to_codex is True
        assert "fallback" in result.selection.selection_reason.lower()

    def test_escalation_on_stagnation(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(response=_good_response())
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        stagnant_state = _make_phase_state(stagnation_steps=10)
        result = pg.guide(
            episode_id="ep1", step_id=10, agent_role="Red",
            current_phase="ENUM", phase_state=stagnant_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        # With stagnation >= 8, should use codex
        assert any(c["model"] == "local-llm" for c in gpt._calls)

    def test_escalation_on_contradictions(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(response=_good_response())
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        # Claim PRIVESC but no shells → contradiction
        result = pg.guide(
            episode_id="ep1", step_id=10, agent_role="Red",
            current_phase="PRIVILEGE_ESCALATION", phase_state=self.phase_state,
            discovery_board=_make_board(shells=[]),
            available_templates=self.templates,
        )
        assert result is not None
        assert any(c["model"] == "local-llm" for c in gpt._calls)

    def test_escalation_on_low_confidence(self):
        """Phase 38 D1: Post-parse escalation removed (model always codex).
        Now verify low-confidence input still produces a valid result with single call."""
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        low_conf = json.loads(_good_response())
        low_conf["phase_decision"]["phase_confidence"] = 0.3
        gpt = _StubGPT(response=json.dumps(low_conf))
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=5, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        # Phase 38 D1: Only 1 call — model is always codex, no post-parse re-run
        assert len(gpt._calls) == 1
        assert gpt._calls[0]["model"] == "local-llm"

    def test_candidates_capped(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM, _MAX_CANDIDATES
        many_candidates = json.loads(_good_response())
        many_candidates["candidates"] = [
            {"template_name": f"tmpl_{i}", "family": "misc", "why": "test",
             "expected_outcome": "test", "stop_condition": "test",
             "confidence": 0.5, "risk": "low", "tags": []}
            for i in range(10)
        ]
        gpt = _StubGPT(response=json.dumps(many_candidates))
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=1, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        assert len(result.candidates) <= _MAX_CANDIDATES

    def test_to_dict_serializable(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(response=_good_response())
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=1, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        d = result.to_dict()
        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_to_mentor_trace_kwargs(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(response=_good_response())
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        result = pg.guide(
            episode_id="ep1", step_id=5, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert result is not None
        kwargs = pg.to_mentor_trace_kwargs(result, step=5, episode=1, agent_id="Scout")
        assert "template_name" in kwargs
        assert "reasoning" in kwargs
        assert "confidence" in kwargs
        assert kwargs["phase"] == result.phase_decision.chosen_phase

    def test_reset_episode_clears_state(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(response=_good_response())
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        pg.guide(
            episode_id="ep1", step_id=1, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        assert pg._call_count > 0
        pg.reset_episode()
        assert pg._call_count == 0
        assert pg._escalation_count == 0

    def test_get_stats(self):
        from core.llm.phase_guided_llm import PhaseGuidedLLM
        gpt = _StubGPT(response=_good_response())
        pg = PhaseGuidedLLM(gpt)  # type: ignore[arg-type]
        pg.guide(
            episode_id="ep1", step_id=1, agent_role="Scout",
            current_phase="RECON", phase_state=self.phase_state,
            discovery_board=self.board, available_templates=self.templates,
        )
        stats = pg.get_stats()
        assert stats["guide_calls"] >= 1
        assert "escalations" in stats


class TestPhaseInference:

    def test_infer_recon_sparse(self):
        from core.llm.phase_guided_llm import _infer_phase
        phase = _infer_phase({"ports": [22], "services": [], "creds": [],
                              "shells": [], "flags": [], "vulns": []}, "RECON")
        assert phase == "RECON"

    def test_infer_enum_with_services(self):
        from core.llm.phase_guided_llm import _infer_phase
        phase = _infer_phase({"ports": [22, 80, 443], "services": ["ssh", "http", "https"],
                              "creds": [], "shells": [], "flags": [], "vulns": []}, "RECON")
        assert phase == "ENUM"

    def test_infer_exploit_with_creds(self):
        from core.llm.phase_guided_llm import _infer_phase
        phase = _infer_phase({"ports": [22], "services": ["ssh"],
                              "creds": [{"user": "admin", "secret": "pass", "source": "brute"}],
                              "shells": [], "flags": [], "vulns": []}, "ENUM")
        assert phase == "EXPLOIT"

    def test_infer_privesc_with_shell(self):
        from core.llm.phase_guided_llm import _infer_phase
        phase = _infer_phase({"ports": [22], "services": ["ssh"], "creds": [],
                              "shells": [{"type": "user", "how": "ssh"}],
                              "flags": [], "vulns": []}, "EXPLOIT")
        assert phase == "PRIVESC"

    def test_infer_post_with_root_shell(self):
        from core.llm.phase_guided_llm import _infer_phase
        phase = _infer_phase({"ports": [22], "services": ["ssh"], "creds": [],
                              "shells": [{"type": "root", "how": "sudo"}],
                              "flags": [], "vulns": []}, "PRIVESC")
        assert phase == "POST"

    def test_infer_exfil_with_flags(self):
        from core.llm.phase_guided_llm import _infer_phase
        phase = _infer_phase({"ports": [22], "services": ["ssh"], "creds": [],
                              "shells": [], "flags": [{"path": "/root/flag.txt", "kind": "root"}],
                              "vulns": []}, "POST")
        assert phase == "EXFIL"


class TestExtractJson:

    def test_plain_json(self):
        from core.llm.phase_guided_llm import _extract_json
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        from core.llm.phase_guided_llm import _extract_json
        result = _extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_garbage_returns_none(self):
        from core.llm.phase_guided_llm import _extract_json
        assert _extract_json("not json at all") is None

    def test_empty_returns_none(self):
        from core.llm.phase_guided_llm import _extract_json
        assert _extract_json("") is None

    def test_none_returns_none(self):
        from core.llm.phase_guided_llm import _extract_json
        assert _extract_json(None) is None  # type: ignore[arg-type]


class TestDataclasses:

    def test_phase_decision_default_tag(self):
        from core.llm.phase_guided_llm import PhaseDecision, _PHASE_TAG
        pd = PhaseDecision()
        assert pd.phase_tag == _PHASE_TAG

    def test_distillation_packet_default_tag(self):
        from core.llm.phase_guided_llm import DistillationPacket, _PHASE_TAG
        dp = DistillationPacket()
        assert dp.phase_tag == _PHASE_TAG

    def test_guidance_result_validate(self):
        from core.llm.phase_guided_llm import PhaseGuidanceResult
        r = PhaseGuidanceResult()
        assert r.validate()  # defaults have P34

    def test_guidance_result_invalid_tag(self):
        from core.llm.phase_guided_llm import PhaseGuidanceResult, PhaseDecision
        r = PhaseGuidanceResult(phase_decision=PhaseDecision(phase_tag="WRONG"))
        assert not r.validate()

    def test_candidate_to_dict(self):
        from core.llm.phase_guided_llm import Candidate
        c = Candidate(template_name="nmap", family="nmap", confidence=0.8)
        d = c.to_dict()
        assert d["template_name"] == "nmap"
        assert d["confidence"] == 0.8

    def test_selection_to_dict(self):
        from core.llm.phase_guided_llm import Selection
        s = Selection(best_template_name="nmap", should_escalate_to_codex=True)
        d = s.to_dict()
        assert d["should_escalate_to_codex"] is True
