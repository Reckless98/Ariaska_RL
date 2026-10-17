"""
Smoke test for multi-agent training.

Tests:
- Offline run completes without errors
- All 5 agents appear in steps.jsonl
- mentor.jsonl is created
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def clean_traces(project_root):
    """Clean up traces before/after test."""
    traces_dir = project_root / "traces"
    # Get list of existing trace files before test
    existing = set(traces_dir.glob("*")) if traces_dir.exists() else set()
    yield traces_dir
    # Optionally clean up new trace files after test
    # (disabled to allow inspection)


def test_offline_multiagent_smoke(tmp_path):
    """Real entry point and all five traces, with guarded simulation only.

    Offline execution may produce no mentor entries; fabricating one is not
    required. This CLI uses the legacy Orchestrator, not the verified PPO path.
    """
    from core.testing.guarded_cli import run_trainer_cli
    result = run_trainer_cli(['--offline', '--episodes', '2', '--max-steps', '3'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.guard_report['network_or_tool_executions'] == 0
    runs = list((tmp_path / 'traces').glob('run_*'))
    assert len(runs) == 1
    steps = [json.loads(line) for line in (runs[0] / 'steps.jsonl').read_text().splitlines() if line]
    expected = {'ScoutAgent', 'RedAgent', 'BlueAgent', 'OrionAgent', 'ShadowAgent'}
    assert expected.issubset({row.get('agent') for row in steps})
    mentor = runs[0] / 'mentor.jsonl'
    assert mentor.exists()
    entries = [json.loads(line) for line in mentor.read_text().splitlines() if line]
    assert all(row.get('model_used') in (None, '', 'offline') for row in entries)


def test_all_agent_types_present_in_orchestrator(project_root):
    """Verify Orchestrator defines all 5 agent types."""
    orchestrator_file = project_root / "core" / "orchestration" / "orchestrator.py"
    if not orchestrator_file.exists():
        pytest.skip("Orchestrator not yet created")
    
    content = orchestrator_file.read_text()
    
    expected_agents = ["ScoutAgent", "RedAgent", "BlueAgent", "OrionAgent", "ShadowAgent"]
    for agent in expected_agents:
        assert agent in content, f"{agent} not found in orchestrator.py"


def test_mentor_policy_modes(project_root):
    """Verify MentorPolicy supports required modes."""
    mentor_policy_file = project_root / "core" / "training" / "mentor_policy.py"
    if not mentor_policy_file.exists():
        pytest.skip("MentorPolicy not yet created")
    
    content = mentor_policy_file.read_text()
    
    expected_modes = ["anneal", "threshold", "always", "never"]
    for mode in expected_modes:
        assert mode in content, f"Mode '{mode}' not found in mentor_policy.py"
