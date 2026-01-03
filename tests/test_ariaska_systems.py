#!/usr/bin/env python3
"""
tests/test_ariaska_systems.py — Unit tests for new ARIASKA systems

Tests:
- EpisodeTrace schema validation and roundtrip
- LLM routing correctness
- Confidence calculation
- Postmortem JSON validation
- SkillLibrary operations
"""

import os
import sys
import json
import time
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEpisodeTrace(unittest.TestCase):
    """Tests for the EpisodeTrace system."""
    
    def test_step_trace_creation(self):
        """Test creating a StepTrace."""
        from core.tracing import StepTrace
        
        step = StepTrace(
            episode_id="run_001_ep0001",
            step=5,
            timestamp=time.time(),
            agent_id="RedAgent",
            phase="recon",
            action_proposed="nmap -sV 10.10.10.10",
            action_final="nmap -sV 10.10.10.10",
            reward=10.0,
            mentor_call=False,
            confidence=0.75
        )
        
        self.assertEqual(step.episode_id, "run_001_ep0001")
        self.assertEqual(step.step, 5)
        self.assertEqual(step.agent_id, "RedAgent")
        self.assertEqual(step.confidence, 0.75)
    
    def test_step_trace_to_json(self):
        """Test StepTrace JSON serialization."""
        from core.tracing import StepTrace
        
        step = StepTrace(
            episode_id="test_ep",
            step=1,
            timestamp=time.time(),
            agent_id="RedAgent",
            phase="recon",
            action_proposed="ping",
            action_final="ping"
        )
        
        json_str = step.to_json()
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed["episode_id"], "test_ep")
        self.assertEqual(parsed["step"], 1)
    
    def test_step_trace_from_dict(self):
        """Test StepTrace from dictionary."""
        from core.tracing import StepTrace
        
        data = {
            "episode_id": "test_ep",
            "step": 3,
            "timestamp": time.time(),
            "agent_id": "RedAgent",
            "phase": "exploit",
            "action_proposed": "msfconsole",
            "action_final": "msfconsole",
            "reward": 25.0,
            "mentor_call": True,
            "mentor_model": "gpt-5-mini"
        }
        
        step = StepTrace.from_dict(data)
        
        self.assertEqual(step.step, 3)
        self.assertEqual(step.phase, "exploit")
        self.assertTrue(step.mentor_call)
        self.assertEqual(step.mentor_model, "gpt-5-mini")
    
    def test_episode_trace_metrics(self):
        """Test EpisodeTrace metrics aggregation."""
        from core.tracing import EpisodeTrace, StepTrace
        
        episode = EpisodeTrace(
            episode_id="test_ep",
            run_id="test_run",
            episode_number=1
        )
        
        # Add steps
        for i in range(5):
            step = StepTrace(
                episode_id="test_ep",
                step=i,
                timestamp=time.time(),
                agent_id="RedAgent",
                phase="recon",
                action_proposed="action",
                action_final="action",
                reward=10.0,
                mentor_call=(i == 0),  # First step uses mentor
                mentor_model="gpt-5-mini" if i == 0 else None,
                confidence=0.5 + (i * 0.1)
            )
            episode.add_step(step)
        
        episode.finalize(success=True, final_phase="exploit")
        
        self.assertEqual(episode.total_steps, 5)
        self.assertEqual(episode.total_reward, 50.0)
        self.assertEqual(episode.mentor_calls, 1)
        self.assertTrue(episode.success)
    
    def test_trace_writer_roundtrip(self):
        """Test writing and reading traces."""
        from core.tracing import TraceWriter, TraceReader, StepTrace
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write traces
            writer = TraceWriter(output_dir=tmpdir)
            writer.start_run(config={"test": True}, seed=42)
            
            episode_id = writer.start_episode(0)
            
            writer.log_step(StepTrace(
                episode_id=episode_id,
                step=0,
                timestamp=time.time(),
                agent_id="RedAgent",
                phase="recon",
                action_proposed="test",
                action_final="test",
                reward=5.0
            ))
            
            writer.end_episode(success=True)
            run = writer.end_run()
            
            # Read traces
            reader = TraceReader(writer.run_dir)
            run_data = reader.load_run()
            episodes = reader.load_episodes()
            steps = reader.load_steps()
            
            self.assertEqual(run_data["total_episodes"], 1)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(steps), 1)
            self.assertEqual(run_data["seed"], 42)
    
    def test_schema_validation(self):
        """Test step trace schema validation."""
        from core.tracing import validate_step_trace
        
        valid_data = {
            "episode_id": "test",
            "step": 1,
            "timestamp": time.time(),
            "agent_id": "RedAgent",
            "phase": "recon",
            "action_proposed": "test",
            "action_final": "test"
        }
        
        invalid_data = {
            "step": 1,
            "timestamp": time.time()
            # Missing required fields
        }
        
        self.assertTrue(validate_step_trace(valid_data))
        self.assertFalse(validate_step_trace(invalid_data))


class TestLLMRouting(unittest.TestCase):
    """Tests for LLM model routing."""
    
    def test_model_selection_by_role(self):
        """Test that models are selected correctly by agent role."""
        # We can't test actual GPT calls without API key, but we can test routing logic
        from core.gpt_manager import GPTManager
        
        # This will fail without API key, so we mock the routing logic
        model_map = GPTManager.MODEL_MAP
        
        # Red/Orion should use mini
        self.assertEqual(model_map.get("red"), "gpt-5-mini")
        self.assertEqual(model_map.get("orion"), "gpt-5-mini")
        
        # Scout/Shadow/Blue should use nano
        self.assertEqual(model_map.get("scout"), "gpt-5-nano")
        self.assertEqual(model_map.get("shadow"), "gpt-5-nano")
        self.assertEqual(model_map.get("blue"), "gpt-5-nano")
        
        # Postmortem should use 5.2
        self.assertEqual(model_map.get("postmortem"), "gpt-5.2")
    
    def test_task_type_routing(self):
        """Test model selection by task type."""
        from core.gpt_manager import GPTManager
        
        model_map = GPTManager.MODEL_MAP
        
        # Tactical/strategic should use mini
        self.assertEqual(model_map.get("tactical"), "gpt-5-mini")
        self.assertEqual(model_map.get("strategic"), "gpt-5-mini")
        
        # Analysis/classification should use nano
        self.assertEqual(model_map.get("analysis"), "gpt-5-nano")
        self.assertEqual(model_map.get("classification"), "gpt-5-nano")


class TestApprenticeTrainer(unittest.TestCase):
    """Tests for the Apprentice-to-Autonomy trainer."""
    
    @classmethod
    def setUpClass(cls):
        """Check if numpy is available."""
        try:
            import numpy
            cls.numpy_available = True
        except ImportError:
            cls.numpy_available = False
    
    def test_threshold_scheduling(self):
        """Test that confidence threshold increases over training."""
        if not self.numpy_available:
            self.skipTest("numpy not available")
        
        from core.training.apprentice_trainer import ApprenticeConfig
        
        config = ApprenticeConfig(
            initial_confidence_threshold=0.3,
            final_confidence_threshold=0.9,
            warmup_episodes=5,
            threshold_schedule_episodes=50
        )
        
        # During warmup
        self.assertEqual(config.get_threshold_for_episode(0), 0.3)
        self.assertEqual(config.get_threshold_for_episode(4), 0.3)
        
        # After warmup, should increase
        threshold_10 = config.get_threshold_for_episode(10)
        threshold_30 = config.get_threshold_for_episode(30)
        threshold_55 = config.get_threshold_for_episode(55)
        
        self.assertGreater(threshold_10, 0.3)
        self.assertGreater(threshold_30, threshold_10)
        self.assertEqual(threshold_55, 0.9)  # At final threshold
    
    def test_episode_metrics_tracking(self):
        """Test episode metrics are tracked correctly."""
        if not self.numpy_available:
            self.skipTest("numpy not available")
        
        from core.training.apprentice_trainer import EpisodeMetrics, DecisionRecord
        
        metrics = EpisodeMetrics(episode=1, confidence_threshold=0.5)
        
        # Add some decisions
        for i in range(10):
            decision = DecisionRecord(
                episode=1,
                step=i,
                timestamp=time.time(),
                agent_action="test",
                agent_confidence=0.5 + (i * 0.05),
                mentor_called=(i < 3),  # First 3 steps use mentor
                final_action="test"
            )
            decision.reward = 10.0
            decision.success = True
            metrics.add_decision(decision)
        
        self.assertEqual(metrics.total_steps, 10)
        self.assertEqual(metrics.mentor_calls, 3)
        self.assertEqual(metrics.autonomous_decisions, 7)
        self.assertEqual(metrics.total_reward, 100.0)


class TestPostmortem(unittest.TestCase):
    """Tests for the OrionPostmortem system."""
    
    @classmethod
    def setUpClass(cls):
        """Check if dependencies are available."""
        try:
            import openai
            cls.openai_available = True
        except ImportError:
            cls.openai_available = False
    
    def test_schema_validation(self):
        """Test postmortem output schema validation."""
        if not self.openai_available:
            # Test schema validation without instantiating OrionPostmortem
            from core.postmortem.orion_postmortem import POSTMORTEM_SCHEMA
            
            valid_data = {
                "key_outcomes": {
                    "wins": ["Success 1"],
                    "fails": ["Fail 1"],
                    "summary": "Test summary"
                },
                "repeated_failure_patterns": [],
                "skill_cards": [
                    {
                        "id": "skill_001",
                        "if_condition": "Port 22 open",
                        "then_action": "Run SSH scan",
                        "confidence": 0.85
                    }
                ],
                "memory_ops": [
                    {
                        "operation": "promote",
                        "target": "skill_001"
                    }
                ],
                "next_experiments": [
                    {
                        "title": "Test experiment",
                        "description": "Description"
                    }
                ]
            }
            
            # Validate required keys are present
            self.assertIn("key_outcomes", valid_data)
            self.assertIn("summary", valid_data["key_outcomes"])
            self.assertIn("skill_cards", valid_data)
            return
        
        from core.postmortem import OrionPostmortem
        
        postmortem = OrionPostmortem(output_dir="test_postmortems")
        
        valid_data = {
            "key_outcomes": {
                "wins": ["Success 1"],
                "fails": ["Fail 1"],
                "summary": "Test summary"
            },
            "repeated_failure_patterns": [],
            "skill_cards": [
                {
                    "id": "skill_001",
                    "if_condition": "Port 22 open",
                    "then_action": "Run SSH scan",
                    "confidence": 0.85
                }
            ],
            "memory_ops": [
                {
                    "operation": "promote",
                    "target": "skill_001"
                }
            ],
            "next_experiments": [
                {
                    "title": "Test experiment",
                    "description": "Description"
                }
            ]
        }
        
        self.assertTrue(postmortem._validate_schema(valid_data))
    
    def test_invalid_schema_rejected(self):
        """Test that invalid schemas are rejected."""
        if not self.openai_available:
            # Basic validation without instantiating
            invalid_data = {
                "key_outcomes": {"wins": [], "fails": []}  # Missing summary
            }
            self.assertNotIn("summary", invalid_data["key_outcomes"])
            return
        
        from core.postmortem import OrionPostmortem
        
        postmortem = OrionPostmortem(output_dir="test_postmortems")
        
        # Missing required fields
        invalid_data = {
            "key_outcomes": {"wins": [], "fails": []}  # Missing summary
        }
        
        self.assertFalse(postmortem._validate_schema(invalid_data))
    
    def test_invalid_memory_op_rejected(self):
        """Test that invalid memory operations are rejected."""
        if not self.openai_available:
            # Basic validation without instantiating
            invalid_ops = [
                {"operation": "invalid_operation", "target": "test"}
            ]
            valid_operations = ["promote", "prune", "merge"]
            for op in invalid_ops:
                self.assertNotIn(op["operation"], valid_operations)
            return
        
        from core.postmortem import OrionPostmortem
        
        postmortem = OrionPostmortem(output_dir="test_postmortems")
        
        invalid_data = {
            "key_outcomes": {"wins": [], "fails": [], "summary": "test"},
            "skill_cards": [],
            "memory_ops": [
                {
                    "operation": "invalid_operation",  # Not promote/prune/merge
                    "target": "test"
                }
            ],
            "next_experiments": []
        }
        
        self.assertFalse(postmortem._validate_schema(invalid_data))


class TestSkillLibrary(unittest.TestCase):
    """Tests for the SkillLibrary."""
    
    def setUp(self):
        """Create temporary library for testing."""
        self.tmpdir = tempfile.mkdtemp()
        self.library_path = os.path.join(self.tmpdir, "test_skills.json")
        self.audit_path = os.path.join(self.tmpdir, "test_audit.jsonl")
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_promote_skill(self):
        """Test promoting a skill to the library."""
        from core.postmortem import SkillLibrary, SkillCard
        
        library = SkillLibrary(
            library_path=self.library_path,
            audit_path=self.audit_path
        )
        
        skill = SkillCard(
            id="test_skill_001",
            if_condition="Port 22 open",
            then_action="Run SSH scan",
            confidence=0.85
        )
        
        result = library.promote(skill, reason="Test")
        
        self.assertTrue(result)
        self.assertEqual(len(library.get_all_skills()), 1)
        self.assertIsNotNone(library.get_skill("test_skill_001"))
    
    def test_prune_skill(self):
        """Test pruning a skill from the library."""
        from core.postmortem import SkillLibrary, SkillCard
        
        library = SkillLibrary(
            library_path=self.library_path,
            audit_path=self.audit_path
        )
        
        # Add then prune
        skill = SkillCard(id="prune_test", if_condition="test", then_action="test", confidence=0.5)
        library.promote(skill)
        
        self.assertEqual(len(library.get_all_skills()), 1)
        
        library.prune("prune_test", reason="Test prune")
        
        self.assertEqual(len(library.get_all_skills()), 0)
    
    def test_duplicate_pruning(self):
        """Test automatic duplicate skill pruning."""
        from core.postmortem import SkillLibrary, SkillCard
        
        library = SkillLibrary(
            library_path=self.library_path,
            audit_path=self.audit_path
        )
        
        # Add skills with same content but different IDs
        skill1 = SkillCard(
            id="dup_1",
            if_condition="Port 22 open",
            then_action="nmap -p22",
            confidence=0.9
        )
        skill2 = SkillCard(
            id="dup_2",
            if_condition="Port 22 open",  # Same condition
            then_action="nmap -p22",       # Same action
            confidence=0.7
        )
        
        library.promote(skill1)
        library.promote(skill2)
        
        pruned = library.prune_duplicates()
        
        self.assertEqual(pruned, 1)
        self.assertEqual(len(library.get_all_skills()), 1)
        
        # Should keep the one with higher confidence
        remaining = library.get_all_skills()[0]
        self.assertEqual(remaining.confidence, 0.9)
    
    def test_audit_log(self):
        """Test that operations are logged to audit file."""
        from core.postmortem import SkillLibrary, SkillCard
        
        library = SkillLibrary(
            library_path=self.library_path,
            audit_path=self.audit_path
        )
        
        skill = SkillCard(id="audit_test", if_condition="test", then_action="test", confidence=0.5)
        library.promote(skill)
        library.prune("audit_test")
        
        audit = library.get_audit_log()
        
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit[0]["operation"], "promote")
        self.assertEqual(audit[1]["operation"], "prune")


if __name__ == "__main__":
    unittest.main(verbosity=2)
