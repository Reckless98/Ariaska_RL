#!/usr/bin/env python3
"""
tests/test_cli_behavior.py — Tests for CLI behavior and LLM mode handling

Validates:
- Default behavior fails fast without API key
- --offline mode works without API key  
- --no-require-llm gracefully degrades
- Online mode with FakeGPTManager injection
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIBehavior:
    """Tests for CLI argument handling and LLM mode behavior."""
    
    def test_default_fails_fast_without_key(self):
        """Default CLI run MUST fail fast when OPENAI_API_KEY is missing."""
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)  # Ensure no key
        
        result = subprocess.run(
            [sys.executable, "-m", "core.training.ariaska_trainer", "--episodes", "1"],
            capture_output=True,
            text=True,
            env=env,
            cwd=Path(__file__).parent.parent
        )
        
        # Should fail with exit code 1
        assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"
        
        # Should have clear error message
        assert "OPENAI_API_KEY" in result.stdout or "OPENAI_API_KEY" in result.stderr, \
            f"Expected API key error message, got:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    
    def test_offline_mode_succeeds_without_key(self):
        """--offline mode should succeed without API key."""
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)  # Ensure no key
        
        result = subprocess.run(
            [
                sys.executable, "-m", "core.training.ariaska_trainer",
                "--offline", "--episodes", "1", "--max-steps", "2", "--verbosity", "quiet"
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=Path(__file__).parent.parent,
            timeout=60
        )
        
        # Should succeed
        assert result.returncode == 0, \
            f"Expected exit code 0, got {result.returncode}\nstderr: {result.stderr}"
        
        # Should complete training
        assert "TRAINING SUMMARY" in result.stdout or "total_episodes" in result.stdout, \
            f"Expected training summary, got:\n{result.stdout}"
    
    def test_no_require_llm_graceful_degrade(self):
        """--no-require-llm should gracefully degrade without API key."""
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)  # Ensure no key
        
        result = subprocess.run(
            [
                sys.executable, "-m", "core.training.ariaska_trainer",
                "--no-require-llm", "--episodes", "1", "--max-steps", "2", 
                "--verbosity", "quiet"
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=Path(__file__).parent.parent,
            timeout=60
        )
        
        # Should succeed (graceful degradation)
        assert result.returncode == 0, \
            f"Expected exit code 0 for graceful degrade, got {result.returncode}\nstderr: {result.stderr}"


class TestGPTManagerIntegration:
    """Tests for GPTManager integration with trainer."""
    
    def test_mentor_call_uses_gpt_manager(self):
        """Verify that mentor calls go through GPTManager.request()."""
        from core.testing import FakeGPTManager
        from core.training.ariaska_trainer import AriaskaTrainer, TrainingConfig
        from core.training.apprentice_trainer import ApprenticeTrainer
        
        import tempfile
        import shutil
        import json
        
        temp_dir = tempfile.mkdtemp(prefix="ariaska_test_")
        skill_lib_path = os.path.join(temp_dir, "skill_library.json")
        with open(skill_lib_path, "w") as f:
            json.dump({"skills": {}, "audit_log": []}, f)
        
        try:
            # Create config with low threshold to force mentor calls
            config = TrainingConfig(
                episodes=1,
                max_steps_per_episode=5,
                seed=42,
                initial_confidence_threshold=0.99,  # Very high = force mentor calls
                max_mentor_calls_per_episode=5,
                enable_postmortem=False,
                trace_output_dir=os.path.join(temp_dir, "traces"),
                checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
                skill_library_path=skill_lib_path,
                verbosity="quiet",
                # Use online mode with mocked GPT
                offline=False,
                enable_llm=True,
                require_llm=False  # Don't fail, we'll inject fake
            )
            
            # Create trainer
            trainer = AriaskaTrainer(config=config)
            
            # Inject FakeGPTManager
            fake_gpt = FakeGPTManager(seed=42)
            trainer.gpt_manager = fake_gpt
            trainer.apprentice_trainer.gpt_manager = fake_gpt
            
            # Run training
            result = trainer.train()
            
            # Verify GPT was called
            requests = fake_gpt.get_requests()
            assert len(requests) > 0, "Expected FakeGPTManager to receive requests"
            
            # Verify at least one tactical request
            tactical_requests = [r for r in requests if r["task_type"] == "tactical"]
            assert len(tactical_requests) > 0, "Expected at least one tactical GPT request"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
