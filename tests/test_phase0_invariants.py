"""
Phase 0 Invariant Tests for Ariaska_RL

These tests lock in the Phase 0 stabilization fixes to prevent regression:
1. Environment sharing between Red and Blue agents
2. BlueAgent.react_to_action() method and schema
3. BlueAgent.simulate_step() does NOT call env.step()
4. Blue team alertness uses 0-100 scale
5. Deterministic mode reproducibility

Run with: .venv/bin/python -m pytest tests/test_phase0_invariants.py -v
"""

import os
import random
import pytest
import numpy as np


class TestEnvironmentSharing:
    """Test that Red and Blue agents share the same environment instance."""
    
    def test_env_is_shared_between_red_and_blue(self):
        """
        Phase 0 Fix #1: Red and Blue agents must share the same environment.
        
        Previously, each agent created its own environment in their constructors,
        bypassing the sharing logic. Now AgentManager forces env sharing.
        """
        from core.multiagent.agent_manager import AgentManager
        
        # Initialize with quiet mode to reduce output
        am = AgentManager(verbosity="quiet")
        
        # Verify agents exist
        assert am.red_agent is not None, "RedAgent should be initialized"
        assert am.blue_agent is not None, "BlueAgent should be initialized"
        
        # Verify environments exist
        assert am.red_agent.env is not None, "RedAgent should have an environment"
        assert am.blue_agent.env is not None, "BlueAgent should have an environment"
        
        # THE KEY ASSERTION: same environment instance
        assert am.red_agent.env is am.blue_agent.env, \
            "Red and Blue agents must share the SAME environment instance"


class TestBlueAgentReactToAction:
    """Test BlueAgent.react_to_action() method exists and returns correct schema."""
    
    def test_blueagent_has_react_to_action_method(self):
        """BlueAgent must have react_to_action method (Phase 0 Fix #3)."""
        from core.agents.blue_agent import BlueAgent
        
        assert hasattr(BlueAgent, 'react_to_action'), \
            "BlueAgent must have react_to_action method"
    
    def test_react_to_action_returns_correct_schema(self):
        """
        react_to_action must return dict with:
        - honeypots_deployed: list
        - credentials_reset: bool
        - alert_increase: number (float)
        """
        from core.multiagent.agent_manager import AgentManager
        
        am = AgentManager(verbosity="quiet")
        blue = am.blue_agent
        
        # Call react_to_action with a sample red action
        result = blue.react_to_action("nmap -sV 10.0.0.1")
        
        # Verify return type
        assert isinstance(result, dict), "react_to_action must return a dict"
        
        # Verify required keys exist
        assert "honeypots_deployed" in result, "Must have 'honeypots_deployed' key"
        assert "credentials_reset" in result, "Must have 'credentials_reset' key"
        assert "alert_increase" in result, "Must have 'alert_increase' key"
        
        # Verify value types
        assert isinstance(result["honeypots_deployed"], list), \
            "honeypots_deployed must be a list"
        assert isinstance(result["credentials_reset"], bool), \
            "credentials_reset must be a bool"
        assert isinstance(result["alert_increase"], (int, float)), \
            "alert_increase must be a number"


class TestBlueAgentSimulateStepNoEnvStep:
    """Test that BlueAgent.simulate_step() does NOT call env.step()."""
    
    def test_blueagent_simulate_step_does_not_call_env_step(self):
        """
        Phase 0 Fix #2: BlueAgent.simulate_step() must NOT call env.step().
        
        Previously both agents called env.step(), causing double-stepping.
        Now only AgentManager calls env.step() once per turn.
        """
        from core.multiagent.agent_manager import AgentManager
        
        am = AgentManager(verbosity="quiet")
        blue = am.blue_agent
        env = blue.env
        
        # Create a flag to detect if env.step is called
        step_was_called = {"called": False}
        original_step = env.step
        
        def spy_step(*args, **kwargs):
            step_was_called["called"] = True
            raise AssertionError("env.step() should NOT be called by BlueAgent.simulate_step()!")
        
        # Monkeypatch env.step
        env.step = spy_step
        
        try:
            # Get initial state for simulate_step
            state = env.reset()
            
            # Call simulate_step - this should NOT call env.step
            # simulate_step returns (action, reward, done, info)
            result = blue.simulate_step(state)
            
            # If we get here, env.step was not called (good!)
            assert not step_was_called["called"], \
                "BlueAgent.simulate_step() must not call env.step()"
            
        finally:
            # Restore original step method
            env.step = original_step


class TestAlertScale0To100:
    """Test that Blue Team alertness uses 0-100 scale (not 0-10)."""
    
    def test_alert_scale_is_0_to_100(self):
        """
        Phase 0 Fix #4: Alertness must be on 0-100 scale.
        
        Previously mixed 0-10 and 0-100 scales caused traceback_threshold (75)
        to be unreachable when alertness capped at 10.
        """
        from core.environment.cyber_environment import CyberEnvironment
        
        env = CyberEnvironment()
        
        # Initial alertness should be 0
        assert env.blue_team_alert >= 0, "Alertness should start >= 0"
        
        # Apply multiple high-impact actions
        for _ in range(20):
            env._update_blue_team_alertness(5.0)  # High impact
        
        # Alertness should be capped at 100, not 10
        assert env.blue_team_alert <= 100, \
            f"Alertness {env.blue_team_alert} should be <= 100 (not using old 0-10 scale)"
        
        # If we had many high-impact actions, alertness should be significant
        # (testing that the scale actually goes up to reasonable values)
        assert env.blue_team_alert > 10 or env.blue_team_alert == 100, \
            f"Alertness {env.blue_team_alert} should reach values > 10 on 0-100 scale"
    
    def test_stealth_report_uses_100_scale(self):
        """Stealth report should show percent-style values consistent with 0-100."""
        from core.environment.cyber_environment import CyberEnvironment
        
        env = CyberEnvironment()
        
        # Set alertness to a known value
        env.blue_team_alert = 50.0
        
        report = env.get_basic_stealth_report()
        
        assert isinstance(report, dict), "Stealth report should be a dict"
        assert "stealth_score" in report, "Report should have stealth_score"
        assert "blue_team_alertness" in report, "Report should have blue_team_alertness"
        
        # Stealth score should be 100 - alertness = 50 on 0-100 scale
        assert 0 <= report["stealth_score"] <= 100, \
            f"Stealth score {report['stealth_score']} should be in 0-100 range"
        
        # Blue team alertness in report should match env value
        assert report["blue_team_alertness"] == 50.0, \
            f"Report alertness {report['blue_team_alertness']} should match env value 50.0"


class TestDeterministicMode:
    """Test deterministic mode reproducibility."""
    
    def test_deterministic_mode_reproducibility(self):
        """
        Phase 0 Fix #6: Deterministic mode must produce reproducible random sequences.
        
        When ARIASKA_DETERMINISTIC=true and ARIASKA_SEED=42, calling
        _init_deterministic_mode() should reset RNG state to produce
        identical sequences each time.
        """
        # Set environment variables
        os.environ["ARIASKA_DETERMINISTIC"] = "true"
        os.environ["ARIASKA_SEED"] = "42"
        
        try:
            from ariaska_cli import _init_deterministic_mode
            
            # First run
            _init_deterministic_mode()
            rand_seq1 = [random.random() for _ in range(5)]
            np_seq1 = np.random.random(5).tolist()
            
            # Reset and second run
            _init_deterministic_mode()
            rand_seq2 = [random.random() for _ in range(5)]
            np_seq2 = np.random.random(5).tolist()
            
            # Sequences must be identical
            assert rand_seq1 == rand_seq2, \
                f"Random sequences should be identical: {rand_seq1} vs {rand_seq2}"
            assert np_seq1 == np_seq2, \
                f"NumPy sequences should be identical: {np_seq1} vs {np_seq2}"
            
        finally:
            # Clean up environment
            os.environ.pop("ARIASKA_DETERMINISTIC", None)
            os.environ.pop("ARIASKA_SEED", None)


class TestAgentManagerQuietMode:
    """Test that quiet mode still initializes all components (Phase 1 fix)."""
    
    def test_quiet_mode_initializes_agents(self):
        """AgentManager with verbosity='quiet' must still create all agents."""
        from core.multiagent.agent_manager import AgentManager
        
        am = AgentManager(verbosity="quiet")
        
        # All agents should be initialized
        assert am.red_agent is not None, "RedAgent must be initialized in quiet mode"
        assert am.blue_agent is not None, "BlueAgent must be initialized in quiet mode"
        assert am.scout_agent is not None, "ScoutAgent must be initialized in quiet mode"
        assert am.shadow_agent is not None, "ShadowAgent must be initialized in quiet mode"
        assert am.orion_agent is not None, "OrionAgent must be initialized in quiet mode"
        
        # GPT manager should be initialized
        assert am.gpt_manager is not None, "GPTManager must be initialized in quiet mode"
    
    def test_silent_mode_initializes_agents(self):
        """AgentManager with verbosity='silent' must still create all agents."""
        from core.multiagent.agent_manager import AgentManager
        
        am = AgentManager(verbosity="silent")
        
        # All agents should be initialized
        assert am.red_agent is not None, "RedAgent must be initialized in silent mode"
        assert am.blue_agent is not None, "BlueAgent must be initialized in silent mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
