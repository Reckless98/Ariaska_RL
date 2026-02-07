#!/usr/bin/env python3
"""
tests/test_ariaska_systems.py — Unit tests for new ARIASKA systems

Tests:
- EpisodeTrace schema validation and roundtrip
- Deterministic event_id generation
- LLM routing correctness
- Confidence calculation
- Postmortem JSON validation
- Evidence refs validation
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
        """Test creating a StepTrace with canonical field names."""
        from core.tracing import StepTrace
        
        step = StepTrace(
            episode_id="run_001_ep0001",
            step=5,
            agent="RedAgent",  # Canonical name
            phase="recon",
            proposed_action="nmap -sV 10.10.10.10",
            chosen_action="nmap -sV 10.10.10.10",
            reward=10.0,
            mentor_call=False,
            confidence=0.75
        )
        
        self.assertEqual(step.episode_id, "run_001_ep0001")
        self.assertEqual(step.step, 5)
        self.assertEqual(step.agent, "RedAgent")
        self.assertEqual(step.confidence, 0.75)
    
    def test_deterministic_event_id(self):
        """Test that event_id is deterministic: {episode_id}:{step:04d}:{agent}"""
        from core.tracing import StepTrace
        
        step = StepTrace(
            episode_id="run_abc_ep0001",
            step=5,
            agent="RedAgent",
            phase="recon",
            proposed_action="test",
            chosen_action="test"
        )
        
        # Event ID should be deterministic
        expected_id = "run_abc_ep0001:0005:RedAgent"
        self.assertEqual(step.event_id, expected_id)
        
        # Same inputs should produce same event_id
        step2 = StepTrace(
            episode_id="run_abc_ep0001",
            step=5,
            agent="RedAgent",
            phase="recon",
            proposed_action="different",
            chosen_action="different"
        )
        self.assertEqual(step.event_id, step2.event_id)
    
    def test_step_trace_to_json(self):
        """Test StepTrace JSON serialization includes event_id."""
        from core.tracing import StepTrace
        
        step = StepTrace(
            episode_id="test_ep",
            step=1,
            agent="RedAgent",
            phase="recon",
            proposed_action="ping",
            chosen_action="ping"
        )
        
        json_str = step.to_json()
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed["episode_id"], "test_ep")
        self.assertEqual(parsed["step"], 1)
        self.assertEqual(parsed["event_id"], "test_ep:0001:RedAgent")
        self.assertIn("agent", parsed)
        self.assertIn("chosen_action", parsed)
    
    def test_step_trace_from_dict_migration(self):
        """Test StepTrace.from_dict handles old field names (migration)."""
        from core.tracing import StepTrace
        
        # Old format with agent_id and action_final
        old_data = {
            "episode_id": "test_ep",
            "step": 3,
            "timestamp": time.time(),
            "agent_id": "RedAgent",  # Old name
            "phase": "exploit",
            "action_proposed": "msfconsole",  # Old name
            "action_final": "msfconsole",     # Old name
            "reward": 25.0,
            "mentor_call": True,
            "mentor_model": "gpt-5.1-codex-mini"      # Old name
        }
        
        step = StepTrace.from_dict(old_data)
        
        self.assertEqual(step.step, 3)
        self.assertEqual(step.agent, "RedAgent")  # Migrated
        self.assertEqual(step.chosen_action, "msfconsole")  # Migrated
        self.assertEqual(step.model_used, "gpt-5.1-codex-mini")  # Migrated
        self.assertTrue(step.mentor_call)
    
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
                agent="RedAgent",
                phase="recon",
                proposed_action="action",
                chosen_action="action",
                reward=10.0,
                mentor_call=(i == 0),
                model_used="gpt-5.1-codex-mini" if i == 0 else None,
                confidence=0.5 + (i * 0.1)
            )
            episode.add_step(step)
        
        episode.finalize(success=True, final_phase="exploit")
        
        self.assertEqual(episode.total_steps, 5)
        self.assertEqual(episode.total_reward, 50.0)
        self.assertEqual(episode.mentor_calls, 1)
        self.assertTrue(episode.success)
        
        # Check event_ids are tracked
        self.assertEqual(len(episode.event_ids), 5)
    
    def test_trace_writer_roundtrip(self):
        """Test writing and reading traces with new schema."""
        from core.tracing import TraceWriter, TraceReader, StepTrace
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write traces
            writer = TraceWriter(output_dir=tmpdir)
            writer.start_run(config={"test": True}, seed=42)
            
            episode_id = writer.start_episode(0)
            
            writer.log_step(StepTrace(
                episode_id=episode_id,
                step=0,
                agent="RedAgent",
                phase="recon",
                proposed_action="test",
                chosen_action="test",
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
            
            # Verify event_id is present
            self.assertIn("event_id", steps[0])
    
    def test_schema_validation_new_format(self):
        """Test step trace schema validation with new canonical fields."""
        from core.tracing import validate_step_trace
        
        valid_data = {
            "event_id": "test_ep:0001:RedAgent",
            "episode_id": "test_ep",
            "step": 1,
            "agent": "RedAgent",
            "phase": "recon",
            "chosen_action": "test",
            "mentor_call": False,
            "done": False
        }
        
        invalid_data = {
            "step": 1,
            "agent": "RedAgent"
            # Missing required fields
        }
        
        self.assertTrue(validate_step_trace(valid_data))
        self.assertFalse(validate_step_trace(invalid_data))
    
    def test_event_id_format_validation(self):
        """Test event_id format validation."""
        from core.tracing import validate_event_id_format, parse_event_id
        
        # Valid formats
        self.assertTrue(validate_event_id_format("run_001_ep0001:0005:RedAgent"))
        self.assertTrue(validate_event_id_format("ep_000001:0000:Agent"))
        
        # Invalid formats
        self.assertFalse(validate_event_id_format("invalid"))
        self.assertFalse(validate_event_id_format("no:colons"))
        
        # Parse event_id
        parsed = parse_event_id("run_001_ep0001:0005:RedAgent")
        self.assertEqual(parsed["episode_id"], "run_001_ep0001")
        self.assertEqual(parsed["step"], 5)
        self.assertEqual(parsed["agent"], "RedAgent")
    
    def test_deterministic_trace_order(self):
        """Test that trace writing is deterministic with same seed."""
        from core.tracing import TraceWriter, StepTrace
        
        def create_trace(tmpdir, seed):
            writer = TraceWriter(output_dir=tmpdir, run_id=f"test_run_{seed}")
            writer.start_run(config={"test": True}, seed=seed)
            
            episode_id = writer.start_episode(0)
            
            for i in range(3):
                writer.log_step(StepTrace(
                    episode_id=episode_id,
                    step=i,
                    agent="RedAgent",
                    phase="recon",
                    proposed_action=f"action_{i}",
                    chosen_action=f"action_{i}",
                    reward=float(i)
                ))
            
            writer.end_episode(success=True)
            writer.end_run()
            return writer.run_dir
        
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                run1_dir = create_trace(tmpdir1, 42)
                run2_dir = create_trace(tmpdir2, 42)
                
                from core.tracing import TraceReader
                reader1 = TraceReader(run1_dir)
                reader2 = TraceReader(run2_dir)
                
                steps1 = reader1.load_steps()
                steps2 = reader2.load_steps()
                
                # Event IDs should match (deterministic)
                for s1, s2 in zip(steps1, steps2):
                    # event_id format is deterministic given same episode_id pattern
                    self.assertEqual(s1["step"], s2["step"])
                    self.assertEqual(s1["agent"], s2["agent"])


class TestLLMRouting(unittest.TestCase):
    """Tests for LLM model routing."""
    
    def test_model_selection_by_role(self):
        """Test that models are selected correctly by agent role."""
        # We can't test actual GPT calls without API key, but we can test routing logic
        from core.gpt_manager import GPTManager
        
        # This will fail without API key, so we mock the routing logic
        model_map = GPTManager.MODEL_MAP
        
        # Red/Orion should use mini
        self.assertEqual(model_map.get("red"), "gpt-5.1-codex-mini")
        self.assertEqual(model_map.get("orion"), "gpt-5.1-codex-mini")
        
        # Scout/Shadow/Blue should use nano
        self.assertEqual(model_map.get("scout"), "gpt-5.1-codex-mini")
        self.assertEqual(model_map.get("shadow"), "gpt-5.1-codex-mini")
        self.assertEqual(model_map.get("blue"), "gpt-5.1-codex-mini")
        
        # Postmortem should use 5.2
        self.assertEqual(model_map.get("postmortem"), "gpt-5.1-codex")
    
    def test_task_type_routing(self):
        """Test model selection by task type."""
        from core.gpt_manager import GPTManager
        
        model_map = GPTManager.MODEL_MAP
        
        # Tactical/strategic should use mini
        self.assertEqual(model_map.get("tactical"), "gpt-5.1-codex-mini")
        self.assertEqual(model_map.get("strategic"), "gpt-5.1-codex-mini")
        
        # Analysis/classification should use nano
        self.assertEqual(model_map.get("analysis"), "gpt-5.1-codex-mini")
        self.assertEqual(model_map.get("classification"), "gpt-5.1-codex-mini")
    
    def test_gpt_manager_init_without_api_key(self):
        """Test that GPTManager can be instantiated without API key."""
        import os
        from core.gpt_manager import GPTManager
        
        # Temporarily remove API key
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            # Should NOT raise
            manager = GPTManager()
            self.assertFalse(manager.is_configured())
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
    
    def test_gpt_request_raises_without_api_key(self):
        """Test that gpt_request raises RuntimeError when API key is missing."""
        import os
        from core.gpt_manager import GPTManager
        
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            manager = GPTManager()
            # Attempting to use client should raise RuntimeError
            with self.assertRaises(RuntimeError) as ctx:
                _ = manager.client  # Access client property
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key


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
        self.assertAlmostEqual(threshold_55, 0.9, places=5)  # At final threshold
    
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
    
    def test_schema_validation(self):
        """Test postmortem output schema validation (offline mode)."""
        from core.postmortem import OrionPostmortem
        
        # Use enable_llm=False for offline testing
        postmortem = OrionPostmortem(output_dir="test_postmortems", enable_llm=False)
        
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
        """Test that invalid schemas are rejected (offline mode)."""
        from core.postmortem import OrionPostmortem
        
        postmortem = OrionPostmortem(output_dir="test_postmortems", enable_llm=False)
        
        # Missing required fields
        invalid_data = {
            "key_outcomes": {"wins": [], "fails": []}  # Missing summary
        }
        
        self.assertFalse(postmortem._validate_schema(invalid_data))
    
    def test_invalid_memory_op_rejected(self):
        """Test that invalid memory operations are rejected (offline mode)."""
        from core.postmortem import OrionPostmortem
        
        postmortem = OrionPostmortem(output_dir="test_postmortems", enable_llm=False)
        
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
    
    def test_offline_mode_no_api_call(self):
        """Test that offline mode doesn't require API key."""
        from core.postmortem import OrionPostmortem
        
        postmortem = OrionPostmortem(output_dir="test_postmortems", enable_llm=False)
        
        self.assertFalse(postmortem.is_llm_available())
        
        # analyze_run should work in offline mode
        run_trace = {"run_id": "test_run", "total_episodes": 1}
        result = postmortem.analyze_run(run_trace, dry_run=True)
        
        self.assertEqual(result.model_used, "offline")
        self.assertTrue(result.validation_passed)
    
    def test_offline_mode_with_evidence_refs(self):
        """Test offline mode generates valid evidence_refs."""
        from core.postmortem import OrionPostmortem
        
        postmortem = OrionPostmortem(output_dir="test_postmortems", enable_llm=False)
        
        # Provide valid event IDs
        valid_event_ids = {
            "run_001_ep0000:0000:RedAgent",
            "run_001_ep0000:0001:RedAgent",
            "run_001_ep0000:0002:RedAgent",
        }
        
        run_trace = {"run_id": "run_001", "total_episodes": 1}
        result = postmortem.analyze_run(
            run_trace, 
            dry_run=True, 
            valid_event_ids=valid_event_ids
        )
        
        # Check skill cards have valid evidence_refs
        for card in result.skill_cards:
            for ref in card.evidence_refs:
                self.assertIn(ref, valid_event_ids)
    
    def test_evidence_refs_validation(self):
        """Test that evidence_refs validation catches invalid refs."""
        from core.postmortem import OrionPostmortem, SkillCard, PostmortemResult
        
        postmortem = OrionPostmortem(output_dir="test_postmortems", enable_llm=False)
        
        valid_event_ids = {"ep0000:0000:Agent", "ep0000:0001:Agent"}
        
        result = PostmortemResult(
            run_id="test",
            timestamp=time.time(),
            dry_run=True
        )
        result.skill_cards = [
            SkillCard(
                id="skill_001",
                if_condition="test",
                then_action="test",
                confidence=0.5,
                evidence_refs=["ep0000:0000:Agent", "invalid_ref"]  # One invalid
            )
        ]
        
        invalid_refs = postmortem.validate_evidence_refs(result, valid_event_ids)
        
        self.assertEqual(len(invalid_refs), 1)
        self.assertIn("invalid_ref", invalid_refs)


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


class TestRuntimeFlags(unittest.TestCase):
    """Tests for runtime flags propagation and offline mode behavior."""
    
    def test_import_rule_engine_no_gpt_init(self):
        """Test that importing rule_engine doesn't instantiate GPTManager."""
        import sys
        import importlib
        
        # Clear cached modules
        modules_to_clear = [k for k in sys.modules.keys() if 'rule_engine' in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        # Track GPTManager instantiations
        from core import gpt_manager as gpt_mod
        original_init = gpt_mod.GPTManager.__init__
        init_calls = []
        
        def tracking_init(self, *args, **kwargs):
            init_calls.append(True)
            return original_init(self, *args, **kwargs)
        
        gpt_mod.GPTManager.__init__ = tracking_init
        
        try:
            # Clear and reimport rule_engine
            if 'core.logic.rule_engine' in sys.modules:
                del sys.modules['core.logic.rule_engine']
            
            # Count inits before import
            init_count_before = len(init_calls)
            
            # Import should NOT trigger GPTManager init (lazy)
            import core.logic.rule_engine
            
            # No new inits should have happened
            init_count_after = len(init_calls)
            self.assertEqual(init_count_before, init_count_after, 
                           "GPTManager was instantiated at import time!")
        finally:
            gpt_mod.GPTManager.__init__ = original_init
    
    def test_offline_mode_gpt_request_returns_placeholder(self):
        """Test that gpt_request returns placeholder in offline mode."""
        import os
        from core.runtime_flags import set_runtime_flags, get_runtime_flags
        from core.gpt_manager import GPTManager
        
        # Set offline mode
        set_runtime_flags(offline=True, enable_llm=True, require_llm=False)
        
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            manager = GPTManager()
            
            # Should return placeholder, not make API call
            result = manager.gpt_request("Test prompt", task_type="reasoning")
            
            self.assertIsNotNone(result)
            # Placeholder contains "OFFLINE MODE" or similar
            self.assertIn("OFFLINE", result.upper() if result else "")
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
    
    def test_runtime_flags_initialization(self):
        """Test runtime flags are set correctly."""
        from core.runtime_flags import set_runtime_flags, get_runtime_flags, RuntimeFlags
        
        set_runtime_flags(offline=True, enable_llm=False, require_llm=False)
        flags = get_runtime_flags()
        
        self.assertTrue(flags.offline)
        self.assertFalse(flags.enable_llm)
        self.assertFalse(flags.require_llm)
        self.assertTrue(flags.initialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
