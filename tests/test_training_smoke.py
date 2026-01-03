#!/usr/bin/env python3
"""
tests/test_training_smoke.py — Smoke tests for end-to-end training

Validates that the training pipeline works end-to-end with:
- Deterministic event_ids
- Canonical StepTrace schema
- Offline postmortem mode (no OPENAI_API_KEY)
"""

import os
import json
import shutil
import tempfile
from pathlib import Path

import pytest


class TestTrainingSmoke:
    """Smoke tests for full training pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup test directories and cleanup after."""
        # Create temp directories
        self.temp_dir = tempfile.mkdtemp(prefix="ariaska_test_")
        self.trace_dir = os.path.join(self.temp_dir, "traces")
        self.postmortem_dir = os.path.join(self.temp_dir, "postmortems")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.skill_library_path = os.path.join(self.temp_dir, "skill_library.json")
        
        # Create empty skill library with correct format
        with open(self.skill_library_path, "w") as f:
            json.dump({"skills": {}, "audit_log": []}, f)
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_training_completes_without_exception(self):
        """Test that training completes without crashing (offline mode)."""
        # Ensure no API key (offline mode)
        os.environ.pop("OPENAI_API_KEY", None)
        
        from core.training.ariaska_trainer import AriaskaTrainer, TrainingConfig
        
        config = TrainingConfig(
            episodes=2,
            max_steps_per_episode=3,
            seed=1337,
            enable_postmortem=True,
            apply_skill_updates=False,  # Dry run
            trace_output_dir=self.trace_dir,
            postmortem_output_dir=self.postmortem_dir,
            checkpoint_dir=self.checkpoint_dir,
            skill_library_path=self.skill_library_path,
            verbosity="quiet"
        )
        
        trainer = AriaskaTrainer(config=config)
        result = trainer.train()
        
        # Verify training completed
        assert result is not None
        assert result["total_episodes"] == 2
        assert result["total_steps"] > 0
    
    def test_trace_files_created(self):
        """Test that trace files are created with correct structure."""
        os.environ.pop("OPENAI_API_KEY", None)
        
        from core.training.ariaska_trainer import AriaskaTrainer, TrainingConfig
        
        config = TrainingConfig(
            episodes=2,
            max_steps_per_episode=3,
            seed=1337,
            enable_postmortem=False,
            trace_output_dir=self.trace_dir,
            checkpoint_dir=self.checkpoint_dir,
            skill_library_path=self.skill_library_path,
            verbosity="quiet"
        )
        
        trainer = AriaskaTrainer(config=config)
        trainer.train()
        
        # Find the run directory
        trace_path = Path(self.trace_dir)
        run_dirs = list(trace_path.glob("run_*"))
        assert len(run_dirs) >= 1, "No run directories created"
        
        run_dir = run_dirs[0]
        
        # Check required files exist
        assert (run_dir / "run.json").exists(), "run.json not created"
        assert (run_dir / "steps.jsonl").exists(), "steps.jsonl not created"
        
        # Load and verify steps.jsonl contains canonical fields
        with open(run_dir / "steps.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        assert len(lines) > 0, "No steps logged"
        
        first_step = lines[0]
        
        # Verify canonical field names
        assert "agent" in first_step, "Missing 'agent' field (got old 'agent_id'?)"
        assert "chosen_action" in first_step, "Missing 'chosen_action' field"
        assert "proposed_action" in first_step, "Missing 'proposed_action' field"
        assert "event_id" in first_step, "Missing 'event_id' field"
        
        # Verify OLD field names are NOT present
        assert "agent_id" not in first_step, "Still using old 'agent_id' field"
        assert "action_final" not in first_step, "Still using old 'action_final' field"
        assert "action_proposed" not in first_step, "Still using old 'action_proposed' field"
        assert "mentor_model" not in first_step, "Still using old 'mentor_model' field"
    
    def test_deterministic_event_ids(self):
        """Test that event IDs follow deterministic format."""
        os.environ.pop("OPENAI_API_KEY", None)
        
        from core.training.ariaska_trainer import AriaskaTrainer, TrainingConfig
        
        config = TrainingConfig(
            episodes=2,
            max_steps_per_episode=3,
            seed=1337,
            enable_postmortem=False,
            trace_output_dir=self.trace_dir,
            checkpoint_dir=self.checkpoint_dir,
            skill_library_path=self.skill_library_path,
            verbosity="quiet"
        )
        
        trainer = AriaskaTrainer(config=config)
        trainer.train()
        
        # Load steps
        trace_path = Path(self.trace_dir)
        run_dir = list(trace_path.glob("run_*"))[0]
        
        with open(run_dir / "steps.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        for step in lines:
            event_id = step.get("event_id", "")
            # Format: {episode_id}:{step:04d}:{agent}
            parts = event_id.split(":")
            assert len(parts) == 3, f"Invalid event_id format: {event_id}"
            
            episode_id, step_num, agent = parts
            assert step_num.isdigit(), f"Step number should be numeric: {step_num}"
            assert len(step_num) == 4, f"Step should be 4 digits: {step_num}"
    
    def test_postmortem_offline_mode(self):
        """Test that postmortem runs in offline mode and creates output."""
        os.environ.pop("OPENAI_API_KEY", None)
        
        from core.training.ariaska_trainer import AriaskaTrainer, TrainingConfig
        
        config = TrainingConfig(
            episodes=2,
            max_steps_per_episode=3,
            seed=1337,
            enable_postmortem=True,
            apply_skill_updates=True,  # Not dry_run - saves postmortem file
            trace_output_dir=self.trace_dir,
            postmortem_output_dir=self.postmortem_dir,
            checkpoint_dir=self.checkpoint_dir,
            skill_library_path=self.skill_library_path,
            verbosity="quiet"
        )
        
        trainer = AriaskaTrainer(config=config)
        result = trainer.train()
        
        # Verify postmortem ran
        assert result.get("postmortem_ran") is True
        assert result.get("postmortem_passed") is True
        
        # Check postmortem output exists
        postmortem_path = Path(self.postmortem_dir)
        postmortem_files = list(postmortem_path.glob("postmortem_*.json"))
        assert len(postmortem_files) >= 1, "No postmortem file created"
        
        # Verify postmortem structure
        with open(postmortem_files[0]) as f:
            pm_data = json.load(f)
        
        assert pm_data.get("model_used") == "offline"
        assert pm_data.get("validation_passed") is True
    
    def test_seed_reproducibility(self):
        """Test that same seed produces identical event_id sequences."""
        os.environ.pop("OPENAI_API_KEY", None)
        
        from core.training.ariaska_trainer import AriaskaTrainer, TrainingConfig
        
        event_id_lists = []
        
        for run_idx in range(2):
            # Use separate trace dirs for each run
            run_trace_dir = os.path.join(self.temp_dir, f"traces_{run_idx}")
            
            config = TrainingConfig(
                episodes=2,
                max_steps_per_episode=3,
                seed=42,  # Same seed
                enable_postmortem=False,
                trace_output_dir=run_trace_dir,
                checkpoint_dir=os.path.join(self.temp_dir, f"ckpt_{run_idx}"),
                skill_library_path=self.skill_library_path,
                verbosity="quiet"
            )
            
            trainer = AriaskaTrainer(config=config)
            trainer.train()
            
            # Collect event IDs
            trace_path = Path(run_trace_dir)
            run_dir = list(trace_path.glob("run_*"))[0]
            
            with open(run_dir / "steps.jsonl") as f:
                event_ids = [json.loads(line).get("event_id") for line in f if line.strip()]
            
            event_id_lists.append(event_ids)
        
        # The event_ids should contain same step patterns (episode_id will differ by timestamp)
        # Check same number of steps and same step:agent structure
        assert len(event_id_lists[0]) == len(event_id_lists[1]), "Different number of steps"
        
        for id1, id2 in zip(event_id_lists[0], event_id_lists[1]):
            # Extract step:agent parts (last two colon-separated parts)
            suffix1 = ":".join(id1.split(":")[-2:])
            suffix2 = ":".join(id2.split(":")[-2:])
            assert suffix1 == suffix2, f"Step/agent mismatch: {suffix1} vs {suffix2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
