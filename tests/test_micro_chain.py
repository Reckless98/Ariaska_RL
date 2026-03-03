#!/usr/bin/env python3
"""Tests for core/llm/micro_chain.py — Phase 27.2 MicroChain."""

import os
import json
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class _StubGPT:
    """Minimal GPT stub for micro-chain tests."""

    def __init__(self, responses=None, budget_ok=True):
        self._responses = responses or {}
        self._budget_ok = budget_ok
        self._calls = []

    def can_make_request(self, **kw):
        return self._budget_ok

    def gpt_request(self, prompt, task_type="", agent_id="", max_tokens=100,
                    model=None, **kw):
        self._calls.append({"model": model, "task_type": task_type, "prompt": prompt[:80]})
        # Return model-keyed response if configured
        if model in self._responses:
            return self._responses[model]
        # Default responses per model
        if model == "gpt-5-nano":
            if "Score" in prompt or "score" in prompt:
                return json.dumps([
                    {"idx": 0, "phase_fit": 0.8, "evidence_support": 0.7, "novelty": 0.6},
                    {"idx": 1, "phase_fit": 0.5, "evidence_support": 0.4, "novelty": 0.3},
                ])
            return "recon_gap"
        if model == "gpt-5.2-mini":
            # Stage 3 scoring uses mini — detect scoring prompts
            if "Score" in prompt or "score" in prompt or "phase_fit" in prompt:
                return json.dumps([
                    {"idx": 0, "phase_fit": 0.8, "evidence_support": 0.7, "novelty": 0.6},
                    {"idx": 1, "phase_fit": 0.5, "evidence_support": 0.4, "novelty": 0.3},
                ])
            return json.dumps([
                {"command": "nmap -sV -p- 10.10.10.1", "template_name": "nmap_full", "reasoning": "full scan"},
                {"command": "gobuster dir -u http://10.10.10.1", "template_name": "gobuster_dir", "reasoning": "web enum"},
            ])
        if model == "gpt-5.2-codex":
            # Score/rating prompts → return JSON array of scores
            if "Score" in prompt or "score" in prompt or "phase_fit" in prompt:
                return json.dumps([
                    {"idx": 0, "phase_fit": 0.8, "evidence_support": 0.7, "novelty": 0.6},
                    {"idx": 1, "phase_fit": 0.5, "evidence_support": 0.4, "novelty": 0.3},
                ])
            # Candidate generation prompts → return JSON array of candidates
            if "candidate" in prompt.lower() or "Generate" in prompt:
                return json.dumps([
                    {"command": "nmap -sV -p- 10.10.10.1", "template_name": "nmap_full", "reasoning": "full scan"},
                    {"command": "gobuster dir -u http://10.10.10.1", "template_name": "gobuster_dir", "reasoning": "web enum"},
                ])
            # Fill/enrich prompts → return JSON array
            if "evidence_used" in prompt or "hypothesis" in prompt:
                return json.dumps([
                    {"idx": 0, "evidence_used": ["port_22"], "hypothesis": "ssh brute", "test": "nmap_full",
                     "expected_observable": "open port", "stop_condition": "done", "confidence": 0.7},
                ])
            # Generic codex fallback (single object)
            return json.dumps({
                "command": "nikto -h http://10.10.10.1",
                "template_name": "nikto_scan",
                "reasoning": "codex suggests nikto"
            })
        return "{}"


class TestMicroChain:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.board = {"ports": {22, 80}, "services": {"ssh", "http"}, "credentials": set()}

    def test_micro_chain_budget_exhausted_returns_none(self):
        from core.llm.micro_chain import MicroChain
        gpt = _StubGPT(budget_ok=False)
        mc = MicroChain(gpt)  # type: ignore[arg-type]
        result = mc.decide(
            phase="RECON", discovery_board=self.board,
            recent_commands=[], available_templates=["nmap_full"],
            agent_role="recon",
        )
        assert result is None

    def test_micro_chain_malformed_codex_json_returns_none(self):
        from core.llm.micro_chain import MicroChain
        # Stage 2 uses gpt-5.2-codex for complex JSON generation
        gpt = _StubGPT(responses={"gpt-5.2-codex": "this is not json at all!!!"})
        mc = MicroChain(gpt)  # type: ignore[arg-type]
        result = mc.decide(
            phase="RECON", discovery_board=self.board,
            recent_commands=[], available_templates=["nmap_full"],
            agent_role="recon",
        )
        assert result is None

    def test_micro_chain_scoring_selects_best(self):
        from core.llm.micro_chain import MicroChain
        gpt = _StubGPT()
        mc = MicroChain(gpt)  # type: ignore[arg-type]
        result = mc.decide(
            phase="RECON", discovery_board=self.board,
            recent_commands=[], available_templates=["nmap_full", "gobuster_dir"],
            agent_role="recon",
        )
        assert result is not None
        assert result.selected.command != ""
        # Best candidate should be the one with highest score (nmap, idx=0)
        assert result.selected.score > 0
        # Stage 1 can be nano or heuristic depending on MC_NANO_ABLATION env var
        assert "nano_classify" in result.stages_used or "heuristic_classify" in result.stages_used
        assert "codex_generate" in result.stages_used
        assert not result.escalated

    def test_micro_chain_escalates_when_low_score(self, monkeypatch):
        import core.llm.micro_chain as mc_mod
        monkeypatch.setattr(mc_mod, "NANO_ABLATION", False)
        from core.llm.micro_chain import MicroChain
        # Force low scores from codex scorer (Stage 3 now uses codex)
        low_scores = json.dumps([
            {"idx": 0, "phase_fit": 0.1, "evidence_support": 0.1, "novelty": 0.1},
            {"idx": 1, "phase_fit": 0.1, "evidence_support": 0.1, "novelty": 0.1},
        ])
        gpt = _StubGPT()
        original_gpt_request = gpt.gpt_request

        def patched_request(prompt, task_type="", agent_id="", max_tokens=100,
                            model=None, **kw):
            # Stage 1: classify (nano)
            if model == "gpt-5-nano" and "Classify" in prompt:
                return "recon_gap"
            # Stage 3: scoring (mini) — return low scores to trigger escalation
            if model == "gpt-5.2-mini" and ("Score" in prompt or "phase_fit" in prompt):
                return low_scores
            # Escalation: "scored poorly" prompt (codex) — return valid single object
            if model == "gpt-5.2-codex" and "scored poorly" in prompt:
                return json.dumps({
                    "command": "nmap -A -T4 10.10.10.1",
                    "template_name": "nmap_aggressive",
                    "reasoning": "codex escalation: aggressive scan"
                })
            return original_gpt_request(prompt, task_type=task_type,
                                        agent_id=agent_id, max_tokens=max_tokens,
                                        model=model, **kw)

        gpt.gpt_request = patched_request
        mc = MicroChain(gpt)  # type: ignore[arg-type]
        result = mc.decide(
            phase="RECON", discovery_board=self.board,
            recent_commands=[], available_templates=["nmap_full"],
            agent_role="recon",
            stagnation_steps=6,  # bucket=2 required for escalation after Phase 33.3
        )
        assert result is not None
        assert result.escalated is True
        assert "codex_escalate" in result.stages_used

    def test_micro_chain_stage3_fail_selects_heuristic(self, monkeypatch):
        import core.llm.micro_chain as mc_mod
        monkeypatch.setattr(mc_mod, "NANO_ABLATION", False)
        from core.llm.micro_chain import MicroChain

        call_count = [0]

        def custom_request(prompt, task_type="", agent_id="", max_tokens=100,
                           model=None, **kw):
            call_count[0] += 1
            if model == "gpt-5-nano":
                return "recon_gap"
            if model == "gpt-5.2-codex":
                # Generation prompts → return valid candidates
                return json.dumps([
                    {"command": "nmap -sV 10.10.10.1", "template_name": "nmap", "reasoning": "scan"},
                ])
            if model == "gpt-5.2-mini":
                # Scoring prompts → return invalid JSON to trigger heuristic
                if "Score" in prompt or "score" in prompt or "phase_fit" in prompt:
                    return "NOT VALID JSON AT ALL"
            return "{}"

        gpt = _StubGPT()
        gpt.gpt_request = custom_request
        mc = MicroChain(gpt)  # type: ignore[arg-type]
        result = mc.decide(
            phase="RECON", discovery_board=self.board,
            recent_commands=[], available_templates=["nmap"],
            agent_role="recon",
        )
        assert result is not None
        assert "heuristic_score" in result.stages_used
        assert result.selected.command != ""

    def test_cache_hit(self):
        from core.llm.micro_chain import MicroChain
        gpt = _StubGPT()
        mc = MicroChain(gpt)  # type: ignore[arg-type]
        r1 = mc.decide(
            phase="RECON", discovery_board=self.board,
            recent_commands=[], available_templates=["nmap_full", "gobuster_dir"],
            agent_role="recon",
        )
        calls_after_first = len(gpt._calls)
        r2 = mc.decide(
            phase="RECON", discovery_board=self.board,
            recent_commands=[], available_templates=["nmap_full", "gobuster_dir"],
            agent_role="recon",
        )
        # Should be cache hit — no new GPT calls
        assert len(gpt._calls) == calls_after_first
        assert r1 is r2

    def test_safe_json_load_fenced(self):
        from core.llm.micro_chain import _safe_json_load
        fenced = '```json\n{"key": "value"}\n```'
        result = _safe_json_load(fenced)
        assert result == {"key": "value"}

    def test_safe_json_load_returns_none_on_garbage(self):
        from core.llm.micro_chain import _safe_json_load
        assert _safe_json_load("not json") is None
        assert _safe_json_load("") is None
        assert _safe_json_load(None) is None  # type: ignore[arg-type]
