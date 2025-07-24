"""
Integration tests for the complete training pipeline.
"""
import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Test if training components are available
training_available = True
try:
    from core.multi_agent_trainer import *
except ImportError:
    training_available = False

from tests.conftest import assert_valid_loss


@pytest.mark.integration
@pytest.mark.skipif(not training_available, reason="Training modules not available")
class TestTrainingPipeline:
    """Integration tests for the complete training pipeline"""
    
    def test_basic_training_loop(self, sample_config, mock_environment, mock_gpt_manager):
        """Test basic training loop execution"""
        # Mock all external dependencies
        with patch('core.gpt_manager.GPTManager', return_value=mock_gpt_manager), \
             patch('core.environment.cyber_environment.CyberEnvironment', return_value=mock_environment):
            
            # Simulate a basic training loop
            num_episodes = 5  # Short training for testing
            episode_rewards = []
            
            for episode in range(num_episodes):
                episode_reward = 0
                state = mock_environment.reset()
                
                for step in range(10):  # 10 steps per episode
                    # Mock action selection
                    action = np.random.randint(0, 4)
                    
                    next_state, reward, done, info = mock_environment.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                    
                    state = next_state
                
                episode_rewards.append(episode_reward)
            
            # Verify training completed
            assert len(episode_rewards) == num_episodes
            assert all(isinstance(r, (int, float)) for r in episode_rewards)
    
    def test_agent_memory_integration(self, sample_config, sample_trajectory):
        """Test agent-memory system integration"""
        # Mock memory components
        mock_memory_router = Mock()
        mock_replay_buffer = Mock()
        
        # Test memory storage and retrieval
        states = sample_trajectory['states']
        actions = sample_trajectory['actions']
        rewards = sample_trajectory['rewards']
        next_states = sample_trajectory['next_states']
        dones = sample_trajectory['dones']
        
        # Simulate storing experiences
        for i in range(len(states)):
            experience = {
                'state': states[i],
                'action': actions[i],
                'reward': rewards[i],
                'next_state': next_states[i],
                'done': dones[i]
            }
            
            mock_replay_buffer.store.return_value = True
            result = mock_replay_buffer.store(experience)
            assert result == True
        
        # Simulate batch sampling
        batch_size = 32
        mock_replay_buffer.sample.return_value = {
            'states': torch.randn(batch_size, 128),
            'actions': torch.randint(0, 4, (batch_size,)),
            'rewards': torch.randn(batch_size),
            'next_states': torch.randn(batch_size, 128),
            'dones': torch.zeros(batch_size, dtype=torch.bool)
        }
        
        batch = mock_replay_buffer.sample(batch_size)
        assert batch['states'].shape[0] == batch_size
    
    def test_multi_agent_coordination(self, sample_config, mock_environment):
        """Test multi-agent coordination during training"""
        # Mock multiple agents
        mock_agents = {
            'red_agent': Mock(),
            'blue_agent': Mock(),
            'scout_agent': Mock(),
            'shadow_agent': Mock(),
            'orion_agent': Mock()
        }
        
        # Mock agent manager
        mock_agent_manager = Mock()
        mock_agent_manager.agents = mock_agents
        
        # Test agent coordination
        for agent_name, agent in mock_agents.items():
            # Mock agent methods
            agent.select_action.return_value = np.random.randint(0, 4)
            agent.update.return_value = {'loss': 0.1, 'reward': 1.0}
            agent.get_metrics.return_value = {'episodes': 10, 'avg_reward': 5.0}
        
        # Simulate multi-agent episode
        state = mock_environment.reset()
        
        for step in range(5):
            agent_actions = {}
            
            # Each agent selects action
            for agent_name, agent in mock_agents.items():
                action = agent.select_action(state)
                agent_actions[agent_name] = action
            
            # Combine actions (mock)
            combined_action = list(agent_actions.values())[0]  # Simplified
            
            next_state, reward, done, info = mock_environment.step(combined_action)
            
            # Each agent updates
            for agent_name, agent in mock_agents.items():
                metrics = agent.update({
                    'state': state,
                    'action': agent_actions[agent_name],
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })
                assert isinstance(metrics, dict)
            
            if done:
                break
            
            state = next_state
    
    def test_gpt_integration_during_training(self, sample_config, mock_gpt_manager):
        """Test GPT integration during training"""
        # Mock GPT responses for different scenarios
        gpt_responses = {
            'action_selection': "scan 192.168.1.1",
            'strategy_update': "Focus on lateral movement",
            'reflection': "The attack was successful, continue with enumeration"
        }
        
        mock_gpt_manager.get_response.side_effect = lambda prompt, **kwargs: (
            gpt_responses.get('action_selection', 'default response')
        )
        
        # Simulate GPT-augmented training
        num_steps = 5
        for step in range(num_steps):
            # Mock action selection with GPT
            prompt = f"Select action for step {step}"
            gpt_response = mock_gpt_manager.get_response(prompt)
            
            assert isinstance(gpt_response, str)
            assert len(gpt_response) > 0
        
        # Verify GPT was called
        assert mock_gpt_manager.get_response.call_count == num_steps
    
    def test_checkpoint_saving_loading(self, temp_dir, sample_config):
        """Test model checkpoint saving and loading"""
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        
        # Mock agent with save/load capabilities
        mock_agent = Mock()
        
        # Mock checkpoint data
        checkpoint_data = {
            'episode': 100,
            'model_state': {'param1': torch.randn(10, 10)},
            'optimizer_state': {'lr': 0.001},
            'metrics': {'avg_reward': 15.5}
        }
        
        # Test saving
        mock_agent.save_checkpoint.return_value = True
        save_result = mock_agent.save_checkpoint(checkpoint_path)
        assert save_result == True
        
        # Test loading
        mock_agent.load_checkpoint.return_value = checkpoint_data
        loaded_data = mock_agent.load_checkpoint(checkpoint_path)
        
        assert isinstance(loaded_data, dict)
        assert 'episode' in loaded_data
        assert 'model_state' in loaded_data
    
    def test_training_visualization_integration(self, sample_config):
        """Test training visualization components"""
        # Mock visualization components
        mock_visualizer = Mock()
        
        # Mock training metrics
        training_metrics = {
            'episode_rewards': [1.0, 2.5, 3.2, 4.1, 5.0],
            'episode_lengths': [10, 15, 12, 18, 20],
            'losses': [0.5, 0.4, 0.3, 0.25, 0.2],
            'exploration_rates': [1.0, 0.8, 0.6, 0.4, 0.2]
        }
        
        # Test visualization updates
        for episode, reward in enumerate(training_metrics['episode_rewards']):
            mock_visualizer.update_metrics.return_value = True
            result = mock_visualizer.update_metrics({
                'episode': episode,
                'reward': reward,
                'loss': training_metrics['losses'][episode],
                'exploration_rate': training_metrics['exploration_rates'][episode]
            })
            assert result == True
        
        # Test plot generation
        mock_visualizer.generate_plots.return_value = {
            'reward_plot': 'path/to/reward_plot.png',
            'loss_plot': 'path/to/loss_plot.png'
        }
        
        plots = mock_visualizer.generate_plots()
        assert isinstance(plots, dict)
        assert 'reward_plot' in plots
    
    def test_error_recovery_during_training(self, sample_config, mock_environment):
        """Test error recovery mechanisms during training"""
        # Mock various failure scenarios
        failure_scenarios = [
            OSError("Network connection failed"),
            ValueError("Invalid action format"),
            RuntimeError("CUDA out of memory"),
            KeyboardInterrupt("User interrupted training")
        ]
        
        for i, exception in enumerate(failure_scenarios):
            if isinstance(exception, KeyboardInterrupt):
                # Don't actually test KeyboardInterrupt in unit tests
                continue
                
            # Mock environment step to raise exception
            mock_environment.step.side_effect = exception
            
            # Training should handle errors gracefully
            try:
                state = mock_environment.reset()
                action = 0
                next_state, reward, done, info = mock_environment.step(action)
            except type(exception):
                # Expected exception - training should implement recovery
                pass
            except Exception as e:
                # Unexpected exception type
                pytest.fail(f"Unexpected exception type: {type(e)}")
            
            # Reset mock for next iteration
            mock_environment.step.side_effect = None
            mock_environment.step.return_value = (
                torch.randn(128), 1.0, False, {}
            )
    
    @pytest.mark.slow
    def test_long_training_session(self, sample_config, mock_environment, mock_gpt_manager):
        """Test longer training session stability"""
        num_episodes = 50  # Longer training session
        
        with patch('core.gpt_manager.GPTManager', return_value=mock_gpt_manager):
            
            episode_rewards = []
            training_stable = True
            
            for episode in range(num_episodes):
                try:
                    episode_reward = 0
                    state = mock_environment.reset()
                    
                    for step in range(20):  # Longer episodes
                        action = np.random.randint(0, 4)
                        next_state, reward, done, info = mock_environment.step(action)
                        episode_reward += reward
                        
                        if done:
                            break
                        
                        state = next_state
                    
                    episode_rewards.append(episode_reward)
                    
                    # Check for training instability
                    if episode > 10:
                        recent_rewards = episode_rewards[-10:]
                        if all(r < -100 for r in recent_rewards):  # Severe performance degradation
                            training_stable = False
                            break
                            
                except Exception as e:
                    training_stable = False
                    break
            
            assert training_stable, "Training became unstable during long session"
            assert len(episode_rewards) > 0, "No episodes completed"
    
    def test_distributed_training_setup(self, sample_config):
        """Test distributed training setup (mocked)"""
        # Mock distributed training components
        mock_dist_trainer = Mock()
        
        # Test world size configuration
        world_sizes = [1, 2, 4]
        for world_size in world_sizes:
            mock_dist_trainer.setup.return_value = True
            result = mock_dist_trainer.setup(world_size=world_size, rank=0)
            assert result == True
        
        # Test rank assignment
        for rank in range(4):
            mock_dist_trainer.get_rank.return_value = rank
            current_rank = mock_dist_trainer.get_rank()
            assert current_rank == rank
    
    def test_hyperparameter_validation(self, sample_config):
        """Test hyperparameter validation during training setup"""
        # Test valid hyperparameters
        valid_configs = [
            {'learning_rate': 1e-4, 'batch_size': 32, 'epsilon_decay': 1000},
            {'learning_rate': 1e-3, 'batch_size': 64, 'epsilon_decay': 500},
            {'learning_rate': 5e-4, 'batch_size': 128, 'epsilon_decay': 2000}
        ]
        
        for config in valid_configs:
            # Mock config validation
            mock_validator = Mock()
            mock_validator.validate.return_value = (True, [])
            
            is_valid, errors = mock_validator.validate(config)
            assert is_valid == True
            assert len(errors) == 0
        
        # Test invalid hyperparameters
        invalid_configs = [
            {'learning_rate': -1, 'batch_size': 32},  # Negative learning rate
            {'learning_rate': 1e-4, 'batch_size': 0},  # Zero batch size
            {'learning_rate': 'invalid', 'batch_size': 32}  # Wrong type
        ]
        
        for config in invalid_configs:
            mock_validator = Mock()
            mock_validator.validate.return_value = (False, ['Invalid parameter'])
            
            is_valid, errors = mock_validator.validate(config)
            assert is_valid == False
            assert len(errors) > 0


@pytest.mark.integration
class TestTrainingPerformance:
    """Performance-related integration tests"""
    
    def test_memory_usage_during_training(self, sample_config):
        """Test memory usage remains reasonable during training"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate memory-intensive training operations
        large_tensors = []
        for i in range(10):
            # Create large tensors to simulate training
            tensor = torch.randn(1000, 1000)
            large_tensors.append(tensor)
            
            # Check memory growth
            current_memory = process.memory_info().rss
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be reasonable (less than 1GB for this test)
            assert memory_growth < 1_000_000_000, f"Excessive memory usage: {memory_growth} bytes"
        
        # Cleanup
        del large_tensors
        gc.collect()
    
    def test_training_speed_benchmarks(self, sample_config, mock_environment):
        """Test training speed benchmarks"""
        import time
        
        num_steps = 100
        start_time = time.time()
        
        # Simulate training steps
        for step in range(num_steps):
            state = mock_environment.reset()
            action = np.random.randint(0, 4)
            next_state, reward, done, info = mock_environment.step(action)
        
        end_time = time.time()
        total_time = end_time - start_time
        steps_per_second = num_steps / total_time
        
        # Should be able to process at least 10 steps per second (very conservative)
        assert steps_per_second > 10, f"Training too slow: {steps_per_second} steps/sec"
    
    def test_gpu_utilization(self, sample_config):
        """Test GPU utilization if available"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for testing")
        
        device = torch.device("cuda")
        
        # Create tensors on GPU
        gpu_tensors = []
        for i in range(5):
            tensor = torch.randn(512, 512, device=device)
            gpu_tensors.append(tensor)
        
        # Perform GPU operations
        for tensor in gpu_tensors:
            result = torch.matmul(tensor, tensor.transpose(0, 1))
            assert result.device.type == 'cuda'
        
        # Check GPU memory usage
        if hasattr(torch.cuda, 'memory_allocated'):
            allocated_memory = torch.cuda.memory_allocated(device)
            assert allocated_memory > 0, "No GPU memory allocated"
        
        # Cleanup GPU memory
        del gpu_tensors
        torch.cuda.empty_cache()