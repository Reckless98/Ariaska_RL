#!/usr/bin/env python3
"""
tests/test_phase15_arbitrator.py — Phase 15.0: Action Arbitrator tests

Tests ArbitrationCandidate scoring, ActionArbitrator selection,
ArbitrationDecisionLog, and bounded candidate count.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestArbitrationCandidate:
    """ArbitrationCandidate schema and scoring tests."""

    def test_default_score(self):
        from core.neurorouter.action_arbitrator import ArbitrationCandidate
        c = ArbitrationCandidate()
        assert c.score() == 0.0  # all zeros

    def test_score_computation(self):
        from core.neurorouter.action_arbitrator import ArbitrationCandidate
        c = ArbitrationCandidate(
            source="ppo",
            expected_value=1.0,
            confidence=0.8,
            phase_fit=0.9,
        )
        s = c.score(aggression=0.3)
        assert s > 0.0
        assert s == pytest.approx(1.0 * 0.8 * 0.9, abs=0.01)

    def test_aggression_boost(self):
        from core.neurorouter.action_arbitrator import ArbitrationCandidate
        c = ArbitrationCandidate(
            source="ppo",
            expected_value=1.0,
            confidence=0.8,
            phase_fit=0.9,
        )
        low = c.score(aggression=0.3)
        high = c.score(aggression=0.8)
        assert high > low  # PPO gets boost at high aggression

    def test_to_dict(self):
        from core.neurorouter.action_arbitrator import ArbitrationCandidate
        c = ArbitrationCandidate(source="mentor", command="nmap -sV {target}")
        d = c.to_dict()
        assert d["source"] == "mentor"
        assert d["command"] == "nmap -sV {target}"


class TestActionArbitrator:
    """ActionArbitrator selection tests."""

    def test_empty_candidates(self):
        from core.neurorouter.action_arbitrator import ActionArbitrator
        arb = ActionArbitrator()
        log = arb.arbitrate([], step=0)
        assert log.winner_source == ""
        assert "no_candidates" in log.reason_codes

    def test_single_candidate_wins(self):
        from core.neurorouter.action_arbitrator import ActionArbitrator, ArbitrationCandidate
        arb = ActionArbitrator()
        candidates = [
            ArbitrationCandidate(source="ppo", command="test", expected_value=1.0, confidence=0.8, phase_fit=0.9),
        ]
        log = arb.arbitrate(candidates, step=5)
        assert log.winner_source == "ppo"
        assert log.winner_command == "test"
        assert log.candidates_count == 1

    def test_highest_score_wins(self):
        from core.neurorouter.action_arbitrator import ActionArbitrator, ArbitrationCandidate
        arb = ActionArbitrator()
        candidates = [
            ArbitrationCandidate(source="ppo", expected_value=0.5, confidence=0.5, phase_fit=0.5),
            ArbitrationCandidate(source="hypothesis", expected_value=2.0, confidence=0.9, phase_fit=0.9),
            ArbitrationCandidate(source="registry", expected_value=0.3, confidence=0.3, phase_fit=0.3),
        ]
        log = arb.arbitrate(candidates, step=1)
        assert log.winner_source == "hypothesis"

    def test_bounded_candidates(self):
        from core.neurorouter.action_arbitrator import ActionArbitrator, ArbitrationCandidate
        arb = ActionArbitrator()
        candidates = [
            ArbitrationCandidate(source=f"src_{i}", expected_value=float(i))
            for i in range(10)
        ]
        log = arb.arbitrate(candidates, step=0)
        assert log.candidates_count <= 6  # MAX_CANDIDATES

    def test_decision_log_compact(self):
        from core.neurorouter.action_arbitrator import ArbitrationDecisionLog
        log = ArbitrationDecisionLog(
            step=5,
            winner_source="ppo",
            winner_command="nmap -sV --long-command-here-with-many-args target",
            winner_score=1.5,
            candidates_count=3,
        )
        c = log.compact()
        assert len(c["cmd"]) <= 60
        assert c["src"] == "ppo"
