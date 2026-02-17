#!/usr/bin/env python3
"""
tests/test_phase14_strategy_plan.py — Phase 14.0: StrategyPlan Tests

Contract C3.4: 6 required tests for strategy plan, subgoals, JSON validation.
"""

import os
import json
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestStrategyPlan:
    """C3.4: StrategyPlan JSON-only IR validation."""

    def test_plan_to_json_valid(self):
        """to_json() produces valid JSON."""
        from core.reasoning.strategy_plan import (
            StrategyPlan, Subgoal, Milestone, Guardrail, SubgoalStatus,
        )
        plan = StrategyPlan(
            plan_id="plan_001",
            subgoals=[
                Subgoal(
                    goal_id="sg_1",
                    description="Scan ports",
                    target_agent="ScoutAgent",
                    verification_commands=["nmap_full"],
                    required_evidence_ids=["open_port"],
                ),
            ],
            milestones=[
                Milestone(name="ports_found", evidence_required=["open_port"], phase="RECON"),
            ],
            guardrails=[
                Guardrail(metric="detection_risk", threshold=0.8, action="switch_to_stealth"),
            ],
        )
        json_str = plan.to_json()
        parsed = json.loads(json_str)
        assert "subgoals" in parsed
        assert len(parsed["subgoals"]) == 1

    def test_plan_from_json_valid(self):
        """from_json() reconstructs plan from valid JSON."""
        from core.reasoning.strategy_plan import StrategyPlan, Subgoal

        plan = StrategyPlan(
            plan_id="plan_002",
            subgoals=[
                Subgoal(goal_id="sg_1", description="Scan",
                        target_agent="ScoutAgent"),
            ],
        )
        json_str = plan.to_json()
        rebuilt = StrategyPlan.from_json(json_str)
        assert rebuilt.plan_id == "plan_002"
        assert len(rebuilt.subgoals) == 1

    def test_plan_from_json_rejects_invalid(self):
        """from_json() raises ValueError on invalid JSON."""
        from core.reasoning.strategy_plan import StrategyPlan
        with pytest.raises(ValueError):
            StrategyPlan.from_json("not valid json {{{")

    def test_plan_from_json_rejects_prose(self):
        """from_json() rejects freeform prose (non-dict)."""
        from core.reasoning.strategy_plan import StrategyPlan
        with pytest.raises(ValueError):
            StrategyPlan.from_json('"just a string"')

    def test_subgoal_description_bounded(self):
        """Subgoal description truncated to ≤128 chars."""
        from core.reasoning.strategy_plan import Subgoal
        sg = Subgoal(
            goal_id="sg_long",
            description="x" * 300,
            target_agent="ScoutAgent",
        )
        assert len(sg.description) <= 128

    def test_template_fallback_plan(self):
        """create_template_fallback_plan() returns valid plan with 4 subgoals."""
        from core.reasoning.strategy_plan import create_template_fallback_plan
        plan = create_template_fallback_plan()
        assert len(plan.subgoals) == 4
        # All subgoals start PENDING or ACTIVE (first one is ACTIVE)
        from core.reasoning.strategy_plan import SubgoalStatus
        for sg in plan.subgoals[1:]:
            assert sg.status == SubgoalStatus.PENDING
