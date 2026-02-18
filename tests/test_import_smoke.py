#!/usr/bin/env python3
"""Phase 35: Import smoke test — verify all core modules import without error."""

import os
import sys
import importlib
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")

# Modules that must import cleanly (no side effects expected)
_CORE_MODULES = [
    "core.llm.budget_manager",
    "core.llm.micro_chain",
    "core.execution.parser_broker",
    "core.execution.discovery_event",
    "core.telemetry.gpt_efficiency",
    "core.telemetry.p15_telemetry",
    "core.telemetry.unified_trace",
    "core.telemetry.events",
    "core.telemetry.jsonl_logger",
    "core.feature_flags",
    "core.runtime_flags",
    "core.gpt_manager",
    "core.models.state_encoder",
    "core.algorithms.ppo_agent",
    "core.algorithms.replay_buffer",
    "core.commands.command_registry",
    "core.knowledge.knowledge_candidate_v2",
    "core.knowledge.knowledge_query",
    "core.interfaces.agent_interface",
    "core.interfaces.memory_sync_interface",
]


class TestImportSmoke:
    """Every module in _CORE_MODULES must import without error."""

    @pytest.mark.parametrize("module_path", _CORE_MODULES)
    def test_import(self, module_path: str):
        mod = importlib.import_module(module_path)
        assert mod is not None
