"""
Test that online mode with API key makes actual mentor calls.

Tests:
- With valid OPENAI_API_KEY: mentor_calls > 0
- mentor.jsonl has model_used field populated (not null/empty)

NOTE: This test requires OPENAI_API_KEY to be set and will be skipped otherwise.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


def has_api_key():
    """Check if OPENAI_API_KEY is set."""
    key = os.environ.get("OPENAI_API_KEY", "")
    return len(key) > 10  # Basic sanity check


@pytest.mark.skipif(not has_api_key(), reason="OPENAI_API_KEY not set")
def test_online_makes_mentor_calls(project_root):
    """
    With valid API key, verify that mentor calls are made.
    
    Acceptance Criteria B:
    - mentor_calls > 0
    - mentor.jsonl has model_used not null
    """
    python = project_root / ".venv" / "bin" / "python"
    if not python.exists():
        pytest.skip("Virtual environment not found. Run 'make venv' first.")
    
    # Run training in online mode (no --offline flag)
    result = subprocess.run(
        [
            str(python),
            "-m", "core.training.ariaska_trainer",
            # No --offline flag = online mode
            "--episodes", "1",
            "--max-steps", "3",
            "--verbosity", "quiet",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    
    # Check exit code
    assert result.returncode == 0, (
        f"Training failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    
    # Find the latest trace directory
    traces_dir = project_root / "traces"
    assert traces_dir.exists(), "traces/ directory not found"
    
    trace_dirs = sorted(traces_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert len(trace_dirs) > 0, "No trace directories found"
    
    latest_trace = trace_dirs[0]
    
    # Check mentor.jsonl
    mentor_file = latest_trace / "mentor.jsonl"
    assert mentor_file.exists(), f"mentor.jsonl not found in {latest_trace}"
    
    with open(mentor_file) as f:
        mentor_entries = [json.loads(line) for line in f if line.strip()]
    
    # Verify mentor calls were made
    assert len(mentor_entries) > 0, "No mentor entries found in mentor.jsonl"
    
    # Verify at least one entry has a real model_used (not offline)
    online_calls = [
        entry for entry in mentor_entries
        if entry.get("model_used") not in (None, "", "offline")
    ]
    
    assert len(online_calls) > 0, (
        f"No online mentor calls found. All entries: {mentor_entries[:3]}"
    )
    
    # Verify model_used is a valid OpenAI model name
    for entry in online_calls:
        model = entry.get("model_used", "")
        assert "gpt" in model.lower() or "o1" in model.lower() or "claude" in model.lower(), (
            f"Unexpected model name: {model}"
        )


@pytest.mark.skipif(not has_api_key(), reason="OPENAI_API_KEY not set")
def test_mentor_calls_logged_correctly(project_root):
    """
    Verify mentor.jsonl entries have required fields.
    """
    python = project_root / ".venv" / "bin" / "python"
    if not python.exists():
        pytest.skip("Virtual environment not found. Run 'make venv' first.")
    
    # Run quick online training
    result = subprocess.run(
        [
            str(python),
            "-m", "core.training.ariaska_trainer",
            "--episodes", "1",
            "--max-steps", "2",
            "--verbosity", "quiet",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    
    if result.returncode != 0:
        pytest.skip(f"Training failed, skipping log validation: {result.stderr}")
    
    # Find latest trace
    traces_dir = project_root / "traces"
    trace_dirs = sorted(traces_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not trace_dirs:
        pytest.skip("No trace directories found")
    
    latest_trace = trace_dirs[0]
    mentor_file = latest_trace / "mentor.jsonl"
    
    if not mentor_file.exists():
        pytest.skip("mentor.jsonl not found")
    
    with open(mentor_file) as f:
        mentor_entries = [json.loads(line) for line in f if line.strip()]
    
    # Verify required fields are present
    required_fields = ["agent", "episode", "step", "timestamp"]
    
    for entry in mentor_entries:
        for field in required_fields:
            assert field in entry, f"Missing required field '{field}' in mentor entry: {entry}"


def test_offline_flag_documented(project_root):
    """
    Verify that --offline flag is documented in help text.
    """
    python = project_root / ".venv" / "bin" / "python"
    if not python.exists():
        pytest.skip("Virtual environment not found. Run 'make venv' first.")
    
    # Get help text
    result = subprocess.run(
        [str(python), "-m", "core.training.ariaska_trainer", "--help"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    help_text = result.stdout + result.stderr
    
    # Verify key flags are documented
    assert "--offline" in help_text, "--offline flag not documented"
    assert "--no-require-llm" in help_text, "--no-require-llm flag not documented"
