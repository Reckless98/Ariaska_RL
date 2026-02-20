"""
tests/integration/conftest.py — Shared fixtures for Phase 42 integration tests.

Provides lightweight orchestrator and coach fixtures for wiring verification.
"""

from __future__ import annotations

import os
import pytest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


@pytest.fixture
def fake_gpt_manager():
    """Create a FakeGPTManager for deterministic tests."""
    from core.testing.fake_gpt_manager import FakeGPTManager
    return FakeGPTManager(seed=42)


@pytest.fixture
def stub_tool_runner():
    """Create a StubToolRunner for tests."""
    from core.testing.tool_runner import get_tool_runner
    return get_tool_runner(testing=True)


@pytest.fixture
def minimal_coach(fake_gpt_manager):
    """Create a minimally configured SmartCoach for wiring tests.

    Uses FakeGPTManager and minimal initialization to avoid
    heavy dependency chains.
    """
    from core.training.smart_coach import SmartCoach

    coach = SmartCoach.__new__(SmartCoach)
    # Inject minimal attributes expected by wiring code
    coach.gpt_manager = fake_gpt_manager
    coach.agent_name = "test_agent"
    coach.role = "offensive"
    coach._ppo_trajectory = []
    coach._ppo_pending = None
    coach._stagnation_steps = 0
    coach._step_count = 0
    coach._episode_count = 0
    coach.episode_used_commands = set()
    coach.command_repeat_count = {}
    coach.decisions = []
    coach.attack_context = None
    coach.ppo_agent = None
    coach.mentor_controller = None
    coach.smart_mentor = None
    coach.dual_mentor = None
    coach.learned_store = MagicMock()
    coach.reward_calculator = MagicMock()
    coach.mentor_policy = MagicMock()
    coach._last_step_had_discovery = False
    coach._reasoning_failures = []
    coach._ssh_failures_this_episode = 0
    coach._reasoning_step_rewards = []
    coach._reasoning_highest_reward = 0.0
    coach._reasoning_total_commands = 0
    coach._reasoning_total_decisions = 0
    coach._reasoning_failed_commands = 0
    coach._reasoning_ppo_decisions = 0
    coach._reasoning_anti_repeat_decisions = 0
    coach._reasoning_last_discovery_step = 0
    # Phase 42 wiring attributes — will be set by _ensure_* methods
    coach._dagger_buffer = None
    coach._phase_timeout = None
    coach._ctf_tracker = None
    coach._cred_sprayer = None
    coach._action_grammar = None
    coach._hallucination_guard = None
    return coach


@pytest.fixture
def minimal_orchestrator(fake_gpt_manager):
    """Create a minimally configured SmartOrchestrator for wiring tests.

    Uses FakeGPTManager and avoids full __init__ to prevent
    heavy dependency chains.
    """
    from core.orchestration.smart_orchestrator import SmartOrchestrator

    orch = SmartOrchestrator.__new__(SmartOrchestrator)
    orch.gpt_manager = fake_gpt_manager
    orch.coaches = {}
    orch.agents = {}
    orch._ppo_trajectory = []
    orch._episode_count = 0
    orch._episode_reward = 0.0
    orch._max_phase_reached = 0
    orch._step_count = 0
    orch._discoveries_this_episode = []
    orch.discovery_board = {
        "ports": set(), "services": set(), "credentials": set(),
        "vulns": set(), "shells": set(), "users": set(),
        "web_paths": set(), "phase": "RECON", "flags_set": set(),
    }
    orch.ppo_agent = None
    orch.dashboard = None
    orch.postmortem = None
    orch.skill_library = None
    orch._is_live_mode = False
    # Phase 42 wiring attributes
    orch._her = None
    orch._meta_learner = None
    orch._reflection_context = ""
    orch._evidence_graph = None
    orch._ttf_tracker = None
    return orch
