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


def test_offline_multiagent_smoke(project_root, clean_traces, tmp_path):
    """
    Run offline training and verify all 5 agents appear in traces.
    
    Acceptance Criteria A:
    - exit 0
    - no GPT API call logs
    - steps.jsonl includes all 5 agents
    - mentor.jsonl created
    """
    python = project_root / ".venv" / "bin" / "python"
    if not python.exists():
        pytest.skip("Virtual environment not found. Run 'make venv' first.")
    
    # Clear OPENAI_API_KEY to ensure offline mode
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    
    # Run training
    result = subprocess.run(
        [
            str(python),
            "-m", "core.training.ariaska_trainer",
            "--offline",
            "--episodes", "2",
            "--max-steps", "3",
            "--seed", "1337",
            "--verbosity", "quiet",
        ],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    
    # Check exit code
    assert result.returncode == 0, f"Training failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    
    # Find the latest trace directory
    traces_dir = project_root / "traces"
    assert traces_dir.exists(), "traces/ directory not found"
    
    trace_dirs = sorted(traces_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert len(trace_dirs) > 0, "No trace directories found"
    
    latest_trace = trace_dirs[0]
    
    # Check steps.jsonl exists and contains all 5 agents
    steps_file = latest_trace / "steps.jsonl"
    assert steps_file.exists(), f"steps.jsonl not found in {latest_trace}"
    
    with open(steps_file) as f:
        steps = [json.loads(line) for line in f if line.strip()]
    
    agents_found = set()
    for step in steps:
        if "agent" in step:
            agents_found.add(step["agent"])
    
    expected_agents = {"ScoutAgent", "RedAgent", "BlueAgent", "OrionAgent", "ShadowAgent"}
    assert expected_agents.issubset(agents_found), (
        f"Missing agents. Expected: {expected_agents}, Found: {agents_found}"
    )
    
    # Check mentor.jsonl exists
    mentor_file = latest_trace / "mentor.jsonl"
    assert mentor_file.exists(), f"mentor.jsonl not found in {latest_trace}"
    
    # Verify mentor.jsonl has content
    with open(mentor_file) as f:
        mentor_entries = [json.loads(line) for line in f if line.strip()]
    
    assert len(mentor_entries) > 0, "mentor.jsonl is empty"
    
    # Verify offline mode (model_used should be "offline" or null)
    for entry in mentor_entries:
        model = entry.get("model_used")
        assert model in (None, "offline", ""), (
            f"Expected offline mode but got model_used={model}"
        )


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
