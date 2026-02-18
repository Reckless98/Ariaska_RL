"""
Phase 35: Coherence Chain + Canonical State tests.

T1: Ports exist → no "no ports" in teaching/guidance
T2: Shells exist → must permit PRIVESC phase
T3: Empty ports → must NOT permit EXPLOIT phase
T4: Web paths exist → web-family actions suggested
T5: Contradiction injection → chain must flag
T6: CanonicalState build + hash determinism
T7: CoherenceChain full 4-step with heuristic
T8: CoherenceChain with FakeGPTManager (nano path)
T9: Heuristic score repeat_risk sensitivity
T10: Phase stagnation desync detection

Run: .venv/bin/python -m pytest tests/test_phase35_coherence.py -v
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_state(**overrides):
    """Build a CanonicalState with sensible defaults, overrides applied."""
    from core.state.canonical_state import CanonicalState

    defaults = dict(
        episode_id="test-ep",
        step_id=5,
        current_phase="RECON",
        phase_confidence=0.5,
        steps_in_phase=3,
        stagnation_steps=0,
        ports=[],
        services=[],
        web_paths_count=0,
        top_web_paths=[],
        users=[],
        credentials=[],
        vulns=[],
        shells=[],
        flags_set=[],
        notes=[],
        recent_commands=[],
        recent_discovery_deltas={},
        mentor_budget_used=0,
        mentor_budget_cap=1000,
        pressure_pct=0.0,
        model_usage={},
        canonical_hash="",
        version=1,
    )
    defaults.update(overrides)
    return CanonicalState(**defaults)


def _make_board(**overrides):
    """Build a discovery_board dict for CanonicalStateBuilder."""
    board = {
        "ports": set(),
        "services": set(),
        "web_paths": set(),
        "users": set(),
        "credentials": set(),
        "vulns": set(),
        "shells": set(),
        "flags_set": set(),
    }
    board.update(overrides)
    return board


# ── T1: Ports exist → no "no ports" contradiction ─────────────────────────

class TestT1PortsExist:
    """When ports are discovered, chain must NOT report 'no ports'."""

    def test_classify_with_ports(self):
        from core.state.coherence_chain import heuristic_classify

        state = _make_state(ports=[22, 80, 443], services=["ssh", "http"])
        result = heuristic_classify(state)
        assert result.phase_guess != "RECON" or result.phase_confidence < 0.60
        assert any("port" in e.lower() for e in result.key_evidence)
        assert "no ports discovered" not in result.missing_evidence

    def test_no_contradiction_with_ports(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(ports=[22, 80], services=["ssh"])
        result = heuristic_contradiction_check(
            state, guidance_claims={"claims_no_ports": True}
        )
        assert result.contradiction_detected is True
        assert any("ports" in c.lower() for c in result.contradictions)


# ── T2: Shells exist → must permit PRIVESC ─────────────────────────────────

class TestT2ShellsPermitPrivesc:
    """When shells exist, PRIVESC should be valid."""

    def test_classify_with_shell(self):
        from core.state.coherence_chain import heuristic_classify

        state = _make_state(
            ports=[22, 80], services=["ssh", "http"],
            shells=["root@target"],
        )
        result = heuristic_classify(state)
        assert result.phase_guess in (
            "PRIVILEGE_ESCALATION", "EXFILTRATION", "POST_EXPLOITATION"
        )
        assert result.phase_confidence >= 0.70

    def test_no_contradiction_for_privesc_with_shell(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(shells=["user@target"])
        result = heuristic_contradiction_check(state, proposed_phase="PRIVILEGE_ESCALATION")
        assert result.contradiction_detected is False


# ── T3: Empty ports → must NOT permit EXPLOIT ──────────────────────────────

class TestT3EmptyPortsNoExploit:
    """With zero ports, exploitation should be impossible."""

    def test_classify_no_ports(self):
        from core.state.coherence_chain import heuristic_classify

        state = _make_state()
        result = heuristic_classify(state)
        assert result.phase_guess == "RECON"
        assert "no ports discovered" in result.missing_evidence

    def test_exploit_without_shell_flagged(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(current_phase="EXPLOITATION")
        result = heuristic_contradiction_check(
            state, proposed_phase="PRIVILEGE_ESCALATION"
        )
        assert result.contradiction_detected is True
        assert any("shell" in c.lower() for c in result.contradictions)


# ── T4: Web paths → web-family actions ─────────────────────────────────────

class TestT4WebPathsSuggestWebFamily:
    """Web paths should suggest ENUMERATION and web-related families."""

    def test_classify_with_web_paths(self):
        from core.state.coherence_chain import heuristic_classify

        state = _make_state(
            ports=[80, 443], services=["http", "https"],
            web_paths_count=5, top_web_paths=["/admin", "/login"],
        )
        result = heuristic_classify(state)
        assert "web" in result.next_best_families


# ── T5: Contradiction injection → chain flags it ──────────────────────────

class TestT5ContradictionInjection:
    """Injected contradictions must be detected and flagged."""

    def test_shell_but_claims_no_foothold(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(shells=["low-priv@target"])
        result = heuristic_contradiction_check(
            state, guidance_claims={"claims_no_foothold": True}
        )
        assert result.contradiction_detected is True
        assert result.severity in ("med", "high")

    def test_creds_but_claims_no_creds(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(credentials=["admin:password"])
        result = heuristic_contradiction_check(
            state, guidance_claims={"claims_no_creds": True}
        )
        assert result.contradiction_detected is True

    def test_clean_state_no_contradiction(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(ports=[22], services=["ssh"])
        result = heuristic_contradiction_check(state)
        assert result.contradiction_detected is False

    def test_stagnation_desync(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(
            ports=[22, 80, 443], services=["ssh", "http"],
            current_phase="RECON", stagnation_steps=10,
        )
        result = heuristic_contradiction_check(state)
        assert result.contradiction_detected is True
        assert any("stagnation" in c.lower() for c in result.contradictions)


# ── T6: CanonicalState build + hash determinism ───────────────────────────

class TestT6CanonicalStateBuild:
    """Builder produces valid, deterministic canonical states."""

    def test_build_from_board(self):
        from core.state.canonical_state import CanonicalStateBuilder

        CanonicalStateBuilder.reset_version()
        board = _make_board(
            ports={22, 80}, services={"ssh", "http"}, web_paths={"/admin"},
        )
        state = CanonicalStateBuilder.build(
            episode_id="ep-1", step_id=3, discovery_board=board,
            current_phase="ENUMERATION",
        )
        assert state.ports == [22, 80]
        assert set(state.services) == {"http", "ssh"}
        assert state.web_paths_count == 1
        assert state.canonical_hash != ""
        assert state.version == 1

    def test_hash_stability(self):
        from core.state.canonical_state import CanonicalStateBuilder

        CanonicalStateBuilder.reset_version()
        board = _make_board(ports={22, 80}, services={"ssh"})
        s1 = CanonicalStateBuilder.build("ep", 1, board)
        CanonicalStateBuilder.reset_version()
        s2 = CanonicalStateBuilder.build("ep", 1, board)
        assert s1.canonical_hash == s2.canonical_hash

    def test_compact_summary(self):
        state = _make_state(ports=[22, 80], services=["ssh"])
        summary = state.compact_summary()
        assert "Ports:2" in summary
        assert "Svcs:1" in summary
        assert "Phase:RECON" in summary

    def test_evidence_counts(self):
        state = _make_state(
            ports=[22, 80], services=["ssh"], credentials=["a:b"],
            shells=["shell1"], vulns=["CVE-2024-1234"],
        )
        counts = state.evidence_counts()
        assert counts["ports"] == 2
        assert counts["services"] == 1
        assert counts["creds"] == 1
        assert counts["shells"] == 1
        assert counts["vulns"] == 1


# ── T7: Full 4-step chain (heuristic only) ────────────────────────────────

class TestT7FullChainHeuristic:
    """End-to-end coherence chain without LLM calls."""

    def test_full_chain_clean(self):
        from core.state.coherence_chain import CoherenceChain

        state = _make_state(ports=[22, 80], services=["ssh", "http"])
        chain = CoherenceChain(gpt_manager=None)
        result = chain.run(state, use_llm=False)

        assert result.classify.phase_guess in (
            "ENUMERATION", "EXPLOITATION", "RECON"
        )
        assert result.contradiction.contradiction_detected is False
        assert result.summary.postcard != ""
        assert result.score.coherence_score > 0
        assert result.elapsed_ms >= 0

    def test_full_chain_with_contradiction(self):
        from core.state.coherence_chain import CoherenceChain

        state = _make_state(
            current_phase="RECON",
            ports=[22, 80, 443], services=["ssh", "http", "https"],
            stagnation_steps=10,
        )
        chain = CoherenceChain(gpt_manager=None)
        result = chain.run(state, use_llm=False)

        assert result.contradiction.contradiction_detected is True
        assert result.score.novelty_score < 0.5  # High stagnation

    def test_chain_call_count(self):
        from core.state.coherence_chain import CoherenceChain

        chain = CoherenceChain(gpt_manager=None)
        state = _make_state()
        chain.run(state, use_llm=False)
        chain.run(state, use_llm=False)
        assert chain.call_count == 2
        chain.reset()
        assert chain.call_count == 0


# ── T8: Chain with FakeGPTManager ──────────────────────────────────────────

class TestT8ChainWithGPT:
    """Chain with FakeGPTManager exercises nano verification path."""

    def test_chain_with_fake_gpt(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.state.coherence_chain import CoherenceChain

        gpt = FakeGPTManager(seed=42)
        state = _make_state(
            step_id=5,
            ports=[22, 80], services=["ssh", "http"],
        )
        chain = CoherenceChain(gpt_manager=gpt)
        result = chain.run(state, use_llm=True)

        # Should still produce valid results even with fake GPT
        assert result.classify.phase_guess != ""
        assert result.summary.postcard != ""
        assert result.score.coherence_score > 0

    def test_chain_no_llm_early_steps(self):
        """Nano verification skipped for steps < 3."""
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.state.coherence_chain import CoherenceChain

        gpt = FakeGPTManager(seed=42)
        state = _make_state(step_id=1, ports=[22])
        chain = CoherenceChain(gpt_manager=gpt)
        result = chain.run(state, use_llm=True)

        # Should work but skip nano verify
        assert result.classify.phase_guess != ""


# ── T9: Score repeat_risk sensitivity ──────────────────────────────────────

class TestT9RepeatRisk:
    """Repeat risk tracks command diversity."""

    def test_high_repeat_risk(self):
        from core.state.coherence_chain import heuristic_score

        state = _make_state(
            recent_commands=["nmap -sV target"] * 10,
        )
        score = heuristic_score(state)
        assert score.repeat_risk >= 0.8

    def test_low_repeat_risk(self):
        from core.state.coherence_chain import heuristic_score

        state = _make_state(
            recent_commands=[
                "nmap -sV target", "nikto -h target", "gobuster dir ...",
                "ssh user@target", "curl target/login", "sqlmap -u ...",
            ],
        )
        score = heuristic_score(state)
        assert score.repeat_risk < 0.3


# ── T10: Phase stagnation desync ───────────────────────────────────────────

class TestT10StagnationDesync:
    """High stagnation with evidence must flag desync."""

    def test_recon_stagnation_flagged(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(
            current_phase="RECON",
            ports=[22, 80], services=["ssh"],
            stagnation_steps=8,
        )
        result = heuristic_contradiction_check(state)
        assert result.contradiction_detected is True

    def test_no_stagnation_early(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(
            current_phase="RECON",
            ports=[22, 80], services=["ssh"],
            stagnation_steps=3,
        )
        result = heuristic_contradiction_check(state)
        # Low stagnation should not flag
        assert not any("stagnation" in c.lower() for c in result.contradictions)

    def test_severity_escalation(self):
        from core.state.coherence_chain import heuristic_contradiction_check

        state = _make_state(
            current_phase="RECON",
            ports=[22, 80, 443], services=["ssh", "http", "https"],
            stagnation_steps=12,
            shells=["user@target"],
        )
        # Also inject impossible guidance
        result = heuristic_contradiction_check(
            state,
            proposed_phase="EXFILTRATION",
            guidance_claims={"claims_no_foothold": True},
        )
        assert result.severity in ("med", "high")
        assert len(result.contradictions) >= 2
