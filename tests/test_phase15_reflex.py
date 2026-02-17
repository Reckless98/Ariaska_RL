#!/usr/bin/env python3
"""
tests/test_phase15_reflex.py — Phase 15.0: Reflex Policy tests

Tests ReflexPolicy deterministic rules, ReflexOverride schema,
and that reflex never produces exploitation commands.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestReflexOverride:
    """ReflexOverride schema tests."""

    def test_default_not_triggered(self):
        from core.neurorouter.reflex_policy import ReflexOverride
        r = ReflexOverride()
        assert r.triggered is False
        assert r.action is None

    def test_to_dict(self):
        from core.neurorouter.reflex_policy import ReflexOverride, ReflexAction
        r = ReflexOverride(
            triggered=True,
            action=ReflexAction.STOP_NOISY,
            reason="test",
            confidence=0.9,
        )
        d = r.to_dict()
        assert d["triggered"] is True
        assert d["action"] == "stop_noisy"


class TestReflexPolicy:
    """ReflexPolicy evaluation tests."""

    def test_no_trigger_on_safe_context(self):
        from core.neurorouter.reflex_policy import ReflexPolicy, ReflexContext
        policy = ReflexPolicy()
        ctx = ReflexContext()  # all defaults = safe
        result = policy.evaluate(ctx)
        assert result.triggered is False

    def test_stop_noisy_on_high_detection(self):
        from core.neurorouter.reflex_policy import ReflexPolicy, ReflexContext, ReflexAction
        policy = ReflexPolicy()
        ctx = ReflexContext(detection_risk=0.9)
        result = policy.evaluate(ctx)
        assert result.triggered is True
        assert result.action == ReflexAction.STOP_NOISY

    def test_reduce_aggression_on_failures(self):
        from core.neurorouter.reflex_policy import ReflexPolicy, ReflexContext, ReflexAction
        policy = ReflexPolicy()
        ctx = ReflexContext(
            aggression_level=0.9,
            repeated_failures=3,
        )
        result = policy.evaluate(ctx)
        assert result.triggered is True
        assert result.action == ReflexAction.REDUCE_AGGRESSION

    def test_verify_on_unverified_findings(self):
        from core.neurorouter.reflex_policy import ReflexPolicy, ReflexContext, ReflexAction
        policy = ReflexPolicy()
        ctx = ReflexContext(unverified_findings=4)
        result = policy.evaluate(ctx)
        assert result.triggered is True
        assert result.action == ReflexAction.VERIFY

    def test_collect_evidence_on_gaps(self):
        from core.neurorouter.reflex_policy import ReflexPolicy, ReflexContext, ReflexAction
        policy = ReflexPolicy()
        ctx = ReflexContext(evidence_gaps=3, steps_since_discovery=5)
        result = policy.evaluate(ctx)
        assert result.triggered is True
        assert result.action == ReflexAction.COLLECT_EVIDENCE

    def test_reflex_command_is_safe(self):
        """Reflex commands must never include exploitation keywords."""
        from core.neurorouter.reflex_policy import ReflexPolicy, ReflexOverride, ReflexAction
        policy = ReflexPolicy()
        exploit_keywords = ["exploit", "payload", "meterpreter", "reverse_tcp", "shell"]
        for action in ReflexAction:
            override = ReflexOverride(triggered=True, action=action)
            cmd = policy.get_reflex_command(override)
            if cmd:
                for kw in exploit_keywords:
                    assert kw not in cmd.lower(), f"Reflex command contains '{kw}': {cmd}"

    def test_priority_ordering(self):
        """STOP_NOISY should have higher priority than VERIFY."""
        from core.neurorouter.reflex_policy import ReflexPolicy, ReflexContext, ReflexAction
        policy = ReflexPolicy()
        ctx = ReflexContext(
            detection_risk=0.9,  # triggers STOP_NOISY
            unverified_findings=5,  # triggers VERIFY
        )
        result = policy.evaluate(ctx)
        assert result.action == ReflexAction.STOP_NOISY  # higher priority
