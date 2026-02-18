#!/usr/bin/env python3
"""
core/telemetry/gpt_efficiency.py — Phase 34: GPT Usage Efficiency Counters

Tracks reasoning-efficiency metrics across an episode:
  - gpt_calls_by_tier     — call count per model tier
  - mentor_efficiency_pct  — mentor calls that changed action / total mentor calls
  - ppo_agreement_rate     — fraction of steps where PPO and final action agreed
  - mentor_dependency      — fraction of total steps that used mentor
  - distillation_yield     — SkillCards produced / mentor calls
  - wasted_calls           — GPT calls that did NOT change action or produce artifact

Reasoning ladder enforcement check:
  nano classify → micro-chain → mentor → codex
  Each rung must be exhausted before escalating.

Author: Phase 34
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger("ariaska.telemetry.gpt_efficiency")


@dataclass
class GPTEfficiencyCounters:
    """Per-episode GPT efficiency tracking."""

    # Call counts per tier
    gpt_calls_by_tier: Dict[str, int] = field(default_factory=lambda: {
        "nano": 0, "mini": 0, "full": 0, "codex": 0,
    })

    # Mentor efficiency
    mentor_calls_total: int = 0
    mentor_calls_changed_action: int = 0

    # PPO agreement
    ppo_proposals: int = 0
    ppo_accepted: int = 0

    # Mentor dependency
    total_steps: int = 0
    steps_with_mentor: int = 0

    # Distillation
    skill_cards_produced: int = 0

    # Waste tracking
    gpt_calls_total: int = 0
    gpt_calls_wasted: int = 0  # did not change action AND no artifact

    # Ladder violations (escalated without exhausting lower tier)
    ladder_violations: int = 0

    def record_gpt_call(
        self,
        tier: str,
        changed_action: bool = False,
        produced_artifact: bool = False,
    ) -> None:
        """Record a single GPT call with its impact."""
        self.gpt_calls_total += 1
        if tier in self.gpt_calls_by_tier:
            self.gpt_calls_by_tier[tier] += 1
        else:
            self.gpt_calls_by_tier[tier] = 1

        if not changed_action and not produced_artifact:
            self.gpt_calls_wasted += 1

    def record_mentor_call(self, changed_action: bool = False) -> None:
        """Record a mentor call and whether it changed the action."""
        self.mentor_calls_total += 1
        if changed_action:
            self.mentor_calls_changed_action += 1

    def record_ppo_proposal(self, accepted: bool = False) -> None:
        """Record a PPO action proposal and whether it was accepted."""
        self.ppo_proposals += 1
        if accepted:
            self.ppo_accepted += 1

    def record_step(self, used_mentor: bool = False) -> None:
        """Record a training step."""
        self.total_steps += 1
        if used_mentor:
            self.steps_with_mentor += 1

    def record_skill_card(self) -> None:
        """Record a SkillCard produced via distillation."""
        self.skill_cards_produced += 1

    def record_ladder_violation(self) -> None:
        """Record a reasoning ladder violation."""
        self.ladder_violations += 1

    @property
    def mentor_efficiency_pct(self) -> float:
        """Fraction of mentor calls that changed the selected action."""
        if self.mentor_calls_total == 0:
            return 0.0
        return round(self.mentor_calls_changed_action / self.mentor_calls_total, 4)

    @property
    def ppo_agreement_rate(self) -> float:
        """Fraction of PPO proposals that were accepted as final action."""
        if self.ppo_proposals == 0:
            return 0.0
        return round(self.ppo_accepted / self.ppo_proposals, 4)

    @property
    def mentor_dependency(self) -> float:
        """Fraction of total steps that used mentor."""
        if self.total_steps == 0:
            return 0.0
        return round(self.steps_with_mentor / self.total_steps, 4)

    @property
    def distillation_yield(self) -> float:
        """SkillCards produced per mentor call."""
        if self.mentor_calls_total == 0:
            return 0.0
        return round(self.skill_cards_produced / self.mentor_calls_total, 4)

    @property
    def wasted_call_rate(self) -> float:
        """Fraction of GPT calls that were wasted."""
        if self.gpt_calls_total == 0:
            return 0.0
        return round(self.gpt_calls_wasted / self.gpt_calls_total, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Export all counters and derived metrics."""
        return {
            "gpt_calls_by_tier": dict(self.gpt_calls_by_tier),
            "gpt_calls_total": self.gpt_calls_total,
            "gpt_calls_wasted": self.gpt_calls_wasted,
            "wasted_call_rate": self.wasted_call_rate,
            "mentor_calls_total": self.mentor_calls_total,
            "mentor_calls_changed_action": self.mentor_calls_changed_action,
            "mentor_efficiency_pct": self.mentor_efficiency_pct,
            "ppo_proposals": self.ppo_proposals,
            "ppo_accepted": self.ppo_accepted,
            "ppo_agreement_rate": self.ppo_agreement_rate,
            "total_steps": self.total_steps,
            "steps_with_mentor": self.steps_with_mentor,
            "mentor_dependency": self.mentor_dependency,
            "skill_cards_produced": self.skill_cards_produced,
            "distillation_yield": self.distillation_yield,
            "ladder_violations": self.ladder_violations,
        }

    def reset(self) -> None:
        """Reset all counters for a new episode."""
        self.gpt_calls_by_tier = {
            "nano": 0, "mini": 0, "full": 0, "codex": 0,
        }
        self.mentor_calls_total = 0
        self.mentor_calls_changed_action = 0
        self.ppo_proposals = 0
        self.ppo_accepted = 0
        self.total_steps = 0
        self.steps_with_mentor = 0
        self.skill_cards_produced = 0
        self.gpt_calls_total = 0
        self.gpt_calls_wasted = 0
        self.ladder_violations = 0


# ── GPT Call Inventory ──────────────────────────────────────────────────────
# Phase 34: Authoritative inventory of ALL GPT call sites in Ariaska.
#
# Format: file | function | model_tier | gating_condition | cache_key | artifact?
#
# core/gpt_manager.py
#   gpt_request()            | routed    | budget_check + can_make_request | state_fp | NO
#
# core/llm/micro_chain.py
#   _classify()              | nano      | budget_check                    | mc_cache | NO
#   _generate_candidates()   | mini      | budget_check                    | mc_cache | NO
#   _score_candidates()      | nano      | budget_check                    | mc_cache | NO
#   _escalate_to_codex()     | codex     | score < threshold + stag/phase  | none     | NO
#
# core/llm/smart_mentor.py
#   mentor_request()         | codex/full| mentor_policy + budget          | state_fp | YES (lesson)
#
# core/execution/parser_broker.py
#   _parse_fullparse()       | codex     | high_value + low_conf + phase   | sha1hash | YES (lesson)
#   _gpt_finalise()          | nano      | fallback only                   | none     | NO
#
# core/training/smart_coach.py
#   _call_mentor()           | codex     | mentor_policy + annealing       | state_fp | YES (skill)
#   _micro_chain_decide()    | nano/mini | always (primary pipeline)       | mc_cache | NO
#
# core/postmortem/orion_postmortem.py
#   generate_postmortem()    | codex     | episode_end                     | ep_id    | YES (report)
#
# core/llm/reflective_cortex.py
#   reflect()                | mini      | phase_transition                | phase_id | NO
#
# core/cortex/executive_cortex.py
#   decide()                 | codex     | strategic_review                | none     | YES (directive)
#
# Reasoning ladder:  nano classify → micro-chain → mentor → codex
# Each tier NOT called unless lower tier exhausted or insufficient.
