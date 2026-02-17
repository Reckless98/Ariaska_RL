#!/usr/bin/env python3
"""
core/reasoning/ — Phase 14.0: Autonomous Reasoning Package

Provides intermediate representations for evidence-based reasoning:
  - TeacherTrace + BCBuffer: First-class mentor learning artifacts
  - Hypothesis + HypothesisGenerator: Evidence-driven hypothesis testing
  - StrategyPlan: Executable IR for multi-agent coordination
  - Lesson + LessonExtractor: Compact teaching points for distillation

All modules are feature-flag-gated (FF_TEACHER_TRACE, FF_HYPOTHESIS_ENGINE,
FF_STRATEGY_PLAN) and default OFF for safe rollout.
"""
