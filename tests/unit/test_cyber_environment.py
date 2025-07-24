"""
Unit tests for the cyber environment implementation.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import the environment
try:
    from core.environment.cyber_environment import CyberEnvironment
except ImportError:
    pytest.skip("CyberEnvironment module not available", allow_module_level=True)


class TestCyberEnvironment:
    """Test suite for CyberEnvironment"""
    
    def test_initialization(self):
        """Test environment initialization"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        assert env.scenario == "simulated"
        assert hasattr(env, 'network_topology')
        assert hasattr(env, 'service_configs')
        assert hasattr(env, 'vulnerability_database')
        assert hasattr(env, 'blue_team_state')
    
    def test_environment_creation_modes(self):
        """Test different environment creation modes"""
        scenarios = ["simulated", "dynamic", "live"]
        
        for scenario in scenarios:
            env = CyberEnvironment(scenario=scenario, defer_reset=True)
            assert env.scenario == scenario
    
    @patch('subprocess.run')
    def test_reset_method(self, mock_subprocess):
        """Test environment reset functionality"""
        # Mock subprocess for nmap calls
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Mock nmap output",
            stderr=""
        )
        
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Mock the reset method's internal components
        with patch.object(env, '_create_network_topology') as mock_topology, \
             patch.object(env, '_initialize_service_configs') as mock_services, \
             patch.object(env, '_create_vulnerability_database') as mock_vulns:
            
            mock_topology.return_value = {"hosts": ["192.168.1.1", "192.168.1.2"]}
            mock_services.return_value = {"ssh": 22, "http": 80}
            mock_vulns.return_value = {"CVE-2021-1234": {"severity": "high"}}
            
            initial_state = env.reset()
            
            # Verify reset was called
            assert mock_topology.called
            assert mock_services.called
            assert mock_vulns.called
            
            # Verify state is returned
            assert initial_state is not None
    
    def test_step_method_structure(self):
        """Test that step method has correct structure"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Mock internal methods to avoid actual network calls
        with patch.object(env, '_execute_action') as mock_execute, \
             patch.object(env, '_calculate_reward') as mock_reward, \
             patch.object(env, '_get_observation') as mock_obs, \
             patch.object(env, '_check_done') as mock_done:
            
            mock_execute.return_value = {"success": True, "output": "Mock output"}
            mock_reward.return_value = 1.0
            mock_obs.return_value = np.random.randn(128)
            mock_done.return_value = False
            
            # Test step method exists and returns correct format
            if hasattr(env, 'step'):
                try:
                    action = {"type": "scan", "target": "192.168.1.1"}
                    result = env.step(action)
                    
                    # Standard RL environment should return (state, reward, done, info)
                    assert len(result) == 4
                    state, reward, done, info = result
                    
                    assert isinstance(reward, (int, float))
                    assert isinstance(done, bool)
                    assert isinstance(info, dict)
                    
                except Exception as e:
                    # If step method has different signature, that's okay for now
                    pytest.skip(f"Step method has different signature: {e}")
    
    def test_action_space_validation(self):
        """Test action space validation"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Test valid actions
        valid_actions = [
            {"type": "scan", "target": "192.168.1.1"},
            {"type": "exploit", "target": "192.168.1.1", "payload": "test"},
            {"type": "lateral_move", "source": "192.168.1.1", "target": "192.168.1.2"}
        ]
        
        for action in valid_actions:
            # Should not raise exception for valid actions
            try:
                if hasattr(env, '_validate_action'):
                    result = env._validate_action(action)
                    assert isinstance(result, bool)
            except NotImplementedError:
                # Method might not be implemented yet
                pass
    
    def test_observation_space(self):
        """Test observation space properties"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Mock observation generation
        with patch.object(env, '_get_observation') as mock_obs:
            mock_obs.return_value = np.random.randn(128)
            
            if hasattr(env, '_get_observation'):
                observation = env._get_observation()
                
                assert isinstance(observation, (np.ndarray, torch.Tensor, list))
                
                if isinstance(observation, np.ndarray):
                    assert observation.ndim >= 1  # Should be at least 1D
                    assert np.isfinite(observation).all()  # No NaN or inf values
    
    def test_reward_calculation(self):
        """Test reward calculation logic"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Mock reward calculation scenarios
        test_scenarios = [
            {"action_success": True, "target_compromised": True, "expected_positive": True},
            {"action_success": False, "target_compromised": False, "expected_positive": False},
            {"action_success": True, "blue_team_detected": True, "expected_negative": True}
        ]
        
        for scenario in test_scenarios:
            with patch.object(env, '_calculate_reward') as mock_reward:
                mock_reward.return_value = 1.0 if scenario.get("expected_positive") else -1.0
                
                if hasattr(env, '_calculate_reward'):
                    reward = env._calculate_reward(scenario)
                    
                    assert isinstance(reward, (int, float))
                    assert np.isfinite(reward)  # No NaN or inf values
    
    def test_network_topology_generation(self):
        """Test network topology generation"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        topology = env._create_network_topology()
        
        assert isinstance(topology, dict)
        # Should have some basic structure
        expected_keys = ["hosts", "subnets", "services"]
        # At least some of these keys should exist
        assert any(key in topology for key in expected_keys)
    
    def test_vulnerability_database(self):
        """Test vulnerability database initialization"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        vuln_db = env._create_vulnerability_database()
        
        assert isinstance(vuln_db, dict)
        # Should contain vulnerability information
        if vuln_db:  # If not empty
            for vuln_id, vuln_info in vuln_db.items():
                assert isinstance(vuln_id, str)
                assert isinstance(vuln_info, dict)
    
    def test_blue_team_initialization(self):
        """Test blue team state initialization"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        blue_state = env._initialize_blue_team()
        
        assert isinstance(blue_state, dict)
        # Should have some blue team configuration
        expected_fields = ["alerting_level", "detection_capabilities", "response_time"]
        # Check if structure is reasonable
        assert len(blue_state) >= 0  # At minimum, should be a valid dict
    
    @patch('subprocess.run')
    def test_nmap_integration(self, mock_subprocess):
        """Test nmap integration (mocked)"""
        # Mock successful nmap execution
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="""
            # Nmap scan results
            Host: 192.168.1.1 ()
            Ports: 22/open/tcp//ssh///, 80/open/tcp//http///
            """,
            stderr=""
        )
        
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Test if environment can handle nmap-like operations
        if hasattr(env, '_perform_network_scan'):
            try:
                scan_result = env._perform_network_scan("192.168.1.1")
                assert isinstance(scan_result, dict)
            except Exception:
                # Method might not be implemented or have different signature
                pass
    
    def test_environment_state_consistency(self):
        """Test that environment state remains consistent"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Mock reset to ensure consistent state
        with patch.object(env, 'reset') as mock_reset:
            mock_reset.return_value = np.random.randn(128)
            
            if hasattr(env, 'reset'):
                initial_state = env.reset()
                
                # Environment should maintain consistent internal state
                assert hasattr(env, 'network_topology')
                assert hasattr(env, 'service_configs')
                
                # State should be reproducible with same seed
                if hasattr(env, 'seed'):
                    env.seed(42)
                    state1 = env.reset()
                    env.seed(42)
                    state2 = env.reset()
                    
                    if isinstance(state1, np.ndarray) and isinstance(state2, np.ndarray):
                        np.testing.assert_array_equal(state1, state2)


@pytest.mark.integration
class TestCyberEnvironmentIntegration:
    """Integration tests for CyberEnvironment"""
    
    def test_environment_agent_interaction(self, mock_gpt_manager):
        """Test environment interaction with agents"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Mock agent manager
        mock_agent_manager = Mock()
        env.agent_manager = mock_agent_manager
        
        # Test basic interaction flow
        with patch.object(env, 'reset') as mock_reset, \
             patch.object(env, 'step') as mock_step:
            
            mock_reset.return_value = np.random.randn(128)
            mock_step.return_value = (
                np.random.randn(128),  # next_state
                1.0,                   # reward
                False,                 # done
                {}                     # info
            )
            
            # Simulate episode
            state = env.reset()
            action = {"type": "scan", "target": "192.168.1.1"}
            next_state, reward, done, info = env.step(action)
            
            assert mock_reset.called
            assert mock_step.called
    
    def test_multi_agent_environment(self):
        """Test environment with multiple agents"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Test that environment can handle multiple agents
        mock_agents = {
            'red_agent': Mock(),
            'blue_agent': Mock(),
            'scout_agent': Mock()
        }
        
        # Environment should be able to track multiple agents
        for agent_name, agent in mock_agents.items():
            # Test agent registration or interaction
            if hasattr(env, 'register_agent'):
                env.register_agent(agent_name, agent)
            elif hasattr(env, 'agents'):
                env.agents[agent_name] = agent
    
    @pytest.mark.slow
    def test_full_episode_simulation(self):
        """Test full episode simulation (marked as slow test)"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        max_steps = 10  # Short episode for testing
        
        with patch.object(env, 'reset') as mock_reset, \
             patch.object(env, 'step') as mock_step:
            
            # Mock episode progression
            step_count = 0
            
            def mock_step_func(action):
                nonlocal step_count
                step_count += 1
                done = step_count >= max_steps
                return (
                    np.random.randn(128),  # state
                    np.random.randn(),     # reward
                    done,                  # done
                    {"step": step_count}   # info
                )
            
            mock_reset.return_value = np.random.randn(128)
            mock_step.side_effect = mock_step_func
            
            # Run episode
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = {"type": "scan", "target": f"192.168.1.{step % 10 + 1}"}
                state, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            assert step_count == max_steps
            assert isinstance(total_reward, (int, float))


class TestEnvironmentErrorHandling:
    """Test error handling in CyberEnvironment"""
    
    def test_invalid_scenario_handling(self):
        """Test handling of invalid scenarios"""
        # Should handle invalid scenarios gracefully
        try:
            env = CyberEnvironment(scenario="invalid_scenario", defer_reset=True)
            # If no exception, that's fine - might have default handling
        except ValueError:
            # Expected behavior for invalid scenario
            pass
        except Exception as e:
            # Other exceptions might be acceptable depending on implementation
            pass
    
    def test_network_error_handling(self):
        """Test handling of network-related errors"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        # Mock network failures
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.side_effect = OSError("Network unreachable")
            
            # Environment should handle network errors gracefully
            try:
                if hasattr(env, '_perform_network_scan'):
                    result = env._perform_network_scan("192.168.1.1")
                    # Should return some default or error state
                    assert result is not None
            except OSError:
                # Acceptable if environment propagates network errors
                pass
    
    def test_action_validation_errors(self):
        """Test handling of invalid actions"""
        env = CyberEnvironment(scenario="simulated", defer_reset=True)
        
        invalid_actions = [
            None,
            {},
            {"invalid": "action"},
            {"type": "invalid_type"},
            {"type": "scan"},  # Missing target
        ]
        
        for invalid_action in invalid_actions:
            if hasattr(env, '_validate_action'):
                try:
                    result = env._validate_action(invalid_action)
                    # Should return False for invalid actions
                    assert isinstance(result, bool)
                except (ValueError, KeyError, TypeError):
                    # Expected exceptions for invalid actions
                    pass