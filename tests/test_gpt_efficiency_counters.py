#!/usr/bin/env python3
"""Phase 34: GPT Efficiency Counters tests."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestGPTEfficiencyCountersExist:
    """Verify all required counter fields exist."""

    def test_counter_fields(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        d = c.to_dict()
        required = {
            "gpt_calls_by_tier", "gpt_calls_total", "gpt_calls_wasted",
            "wasted_call_rate", "mentor_calls_total", "mentor_calls_changed_action",
            "mentor_efficiency_pct", "ppo_proposals", "ppo_accepted",
            "ppo_agreement_rate", "total_steps", "steps_with_mentor",
            "mentor_dependency", "skill_cards_produced", "distillation_yield",
            "ladder_violations",
        }
        assert required.issubset(set(d.keys()))

    def test_tier_keys(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        assert set(c.gpt_calls_by_tier.keys()) == {"nano", "mini", "full", "codex"}


class TestGPTEfficiencyMetrics:
    """Verify derived metric calculations."""

    def test_mentor_efficiency(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        c.record_mentor_call(changed_action=True)
        c.record_mentor_call(changed_action=False)
        c.record_mentor_call(changed_action=True)
        assert c.mentor_efficiency_pct == pytest.approx(2 / 3, abs=0.01)

    def test_ppo_agreement(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        c.record_ppo_proposal(accepted=True)
        c.record_ppo_proposal(accepted=True)
        c.record_ppo_proposal(accepted=False)
        assert c.ppo_agreement_rate == pytest.approx(2 / 3, abs=0.01)

    def test_mentor_dependency(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        c.record_step(used_mentor=True)
        c.record_step(used_mentor=False)
        c.record_step(used_mentor=False)
        c.record_step(used_mentor=True)
        assert c.mentor_dependency == pytest.approx(0.5, abs=0.01)

    def test_distillation_yield(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        c.record_mentor_call(changed_action=True)
        c.record_mentor_call(changed_action=True)
        c.record_skill_card()
        assert c.distillation_yield == pytest.approx(0.5, abs=0.01)

    def test_wasted_calls(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        c.record_gpt_call(tier="nano", changed_action=False, produced_artifact=False)
        c.record_gpt_call(tier="mini", changed_action=True, produced_artifact=False)
        c.record_gpt_call(tier="codex", changed_action=False, produced_artifact=True)
        assert c.gpt_calls_wasted == 1
        assert c.wasted_call_rate == pytest.approx(1 / 3, abs=0.01)

    def test_ladder_violation_tracking(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        c.record_ladder_violation()
        c.record_ladder_violation()
        assert c.ladder_violations == 2

    def test_zero_division_safe(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        assert c.mentor_efficiency_pct == 0.0
        assert c.ppo_agreement_rate == 0.0
        assert c.mentor_dependency == 0.0
        assert c.distillation_yield == 0.0
        assert c.wasted_call_rate == 0.0

    def test_reset_clears(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        c.record_gpt_call(tier="nano", changed_action=True)
        c.record_mentor_call(changed_action=True)
        c.record_step(used_mentor=True)
        c.record_skill_card()
        c.reset()
        d = c.to_dict()
        assert d["gpt_calls_total"] == 0
        assert d["mentor_calls_total"] == 0
        assert d["total_steps"] == 0
        assert d["skill_cards_produced"] == 0
        assert d["ladder_violations"] == 0

    def test_gpt_calls_by_tier_increments(self):
        from core.telemetry.gpt_efficiency import GPTEfficiencyCounters
        c = GPTEfficiencyCounters()
        c.record_gpt_call(tier="nano", changed_action=True)
        c.record_gpt_call(tier="nano", changed_action=False)
        c.record_gpt_call(tier="codex", changed_action=True, produced_artifact=True)
        assert c.gpt_calls_by_tier["nano"] == 2
        assert c.gpt_calls_by_tier["codex"] == 1
        assert c.gpt_calls_total == 3
