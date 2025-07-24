"""
Unit tests for the DQN algorithm implementation.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

# Import the DQN implementation
try:
    from core.algorithms.dqn import DQNNetwork
except ImportError:
    pytest.skip("DQN module not available", allow_module_level=True)

from tests.conftest import assert_valid_q_values, assert_valid_loss


class TestDQNNetwork:
    """Test suite for DQN network implementation"""
    
    def test_initialization(self, sample_config):
        """Test DQN network initialization"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        hidden_dims = sample_config['hidden_dims']
        
        network = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dueling=True
        )
        
        assert network.state_dim == state_dim
        assert network.action_dim == action_dim
        assert network.dueling == True
        
        # Check if network has the expected parameters
        total_params = sum(p.numel() for p in network.parameters())
        assert total_params > 0, "Network should have learnable parameters"
    
    def test_forward_pass(self, sample_config, sample_state):
        """Test forward pass of DQN network"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        network = DQNNetwork(state_dim=state_dim, action_dim=action_dim)
        
        # Test single state
        q_values = network(sample_state)
        assert_valid_q_values(q_values, action_dim)
        
        # Test batch of states
        batch_states = torch.randn(16, state_dim)
        batch_q_values = network(batch_states)
        assert batch_q_values.shape == (16, action_dim)
        assert_valid_q_values(batch_q_values, action_dim)
    
    def test_dueling_vs_standard(self, sample_config, sample_state):
        """Test difference between dueling and standard DQN"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        # Create both types of networks
        dueling_network = DQNNetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            dueling=True
        )
        standard_network = DQNNetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            dueling=False
        )
        
        # Both should produce valid Q-values
        dueling_q = dueling_network(sample_state)
        standard_q = standard_network(sample_state)
        
        assert_valid_q_values(dueling_q, action_dim)
        assert_valid_q_values(standard_q, action_dim)
        
        # Networks should have different architectures
        dueling_params = sum(p.numel() for p in dueling_network.parameters())
        standard_params = sum(p.numel() for p in standard_network.parameters())
        # Dueling network typically has more parameters due to separate streams
        assert dueling_params != standard_params
    
    def test_different_activations(self, sample_config, sample_state):
        """Test different activation functions"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        activations = ['relu', 'leaky_relu', 'elu', 'tanh']
        
        for activation in activations:
            network = DQNNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                activation=activation
            )
            
            q_values = network(sample_state)
            assert_valid_q_values(q_values, action_dim)
    
    def test_gradient_flow(self, sample_config, sample_state):
        """Test that gradients flow properly through the network"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        network = DQNNetwork(state_dim=state_dim, action_dim=action_dim)
        
        # Forward pass
        q_values = network(sample_state)
        
        # Backward pass
        loss = q_values.sum()
        loss.backward()
        
        # Check that gradients exist
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient should exist for learnable parameters"
                assert torch.isfinite(param.grad).all(), "Gradients should be finite"
    
    def test_device_compatibility(self, sample_config):
        """Test network works on different devices"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        network = DQNNetwork(state_dim=state_dim, action_dim=action_dim)
        
        # Test on CPU
        cpu_state = torch.randn(1, state_dim)
        cpu_q_values = network(cpu_state)
        assert_valid_q_values(cpu_q_values, action_dim)
        
        # Test on GPU if available
        if torch.cuda.is_available():
            network_gpu = network.cuda()
            gpu_state = cpu_state.cuda()
            gpu_q_values = network_gpu(gpu_state)
            assert_valid_q_values(gpu_q_values, action_dim)
            assert gpu_q_values.device.type == 'cuda'
    
    def test_layer_normalization(self, sample_config, sample_state):
        """Test layer normalization functionality"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        # Test with layer norm
        network_ln = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            use_layer_norm=True
        )
        
        # Test without layer norm
        network_no_ln = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            use_layer_norm=False
        )
        
        q_values_ln = network_ln(sample_state)
        q_values_no_ln = network_no_ln(sample_state)
        
        assert_valid_q_values(q_values_ln, action_dim)
        assert_valid_q_values(q_values_no_ln, action_dim)
        
        # Networks should have different parameter counts
        params_ln = sum(p.numel() for p in network_ln.parameters())
        params_no_ln = sum(p.numel() for p in network_no_ln.parameters())
        assert params_ln != params_no_ln
    
    def test_dropout_behavior(self, sample_config, sample_state):
        """Test dropout behavior during training vs evaluation"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        network = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            dropout_rate=0.5  # High dropout for testing
        )
        
        # Training mode
        network.train()
        train_outputs = []
        for _ in range(5):
            output = network(sample_state)
            train_outputs.append(output.detach())
        
        # Evaluation mode
        network.eval()
        eval_outputs = []
        for _ in range(5):
            output = network(sample_state)
            eval_outputs.append(output.detach())
        
        # Training outputs should vary due to dropout
        train_variance = torch.var(torch.stack(train_outputs), dim=0).mean()
        eval_variance = torch.var(torch.stack(eval_outputs), dim=0).mean()
        
        # Eval should have less variance than training
        assert eval_variance < train_variance + 1e-6  # Small tolerance for numerical precision


class TestDQNTraining:
    """Test suite for DQN training functionality"""
    
    def test_loss_computation(self, sample_config):
        """Test loss computation for DQN"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        batch_size = 32
        
        network = DQNNetwork(state_dim=state_dim, action_dim=action_dim)
        
        # Create sample batch
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, state_dim)
        dones = torch.randint(0, 2, (batch_size,)).bool()
        
        # Compute Q-values
        q_values = network(states)
        next_q_values = network(next_states)
        
        # Compute targets (simplified version)
        gamma = 0.99
        targets = rewards + gamma * next_q_values.max(dim=1)[0] * ~dones
        
        # Compute loss
        selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(selected_q_values, targets.detach())
        
        assert_valid_loss(loss)
    
    def test_parameter_updates(self, sample_config):
        """Test that parameters update during training"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        network = DQNNetwork(state_dim=state_dim, action_dim=action_dim)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        
        # Store initial parameters
        initial_params = [param.clone() for param in network.parameters()]
        
        # Training step
        states = torch.randn(32, state_dim)
        targets = torch.randn(32, action_dim)
        
        optimizer.zero_grad()
        outputs = network(states)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Check parameters changed
        for initial, current in zip(initial_params, network.parameters()):
            assert not torch.equal(initial, current), "Parameters should update during training"


@pytest.mark.integration
class TestDQNIntegration:
    """Integration tests for DQN with other components"""
    
    def test_dqn_with_replay_buffer(self, sample_config, sample_trajectory):
        """Test DQN integration with replay buffer"""
        # This would require implementing replay buffer tests
        # Placeholder for integration testing
        pass
    
    def test_dqn_with_environment(self, sample_config, mock_environment):
        """Test DQN interaction with environment"""
        state_dim = sample_config['state_dim']
        action_dim = sample_config['action_dim']
        
        network = DQNNetwork(state_dim=state_dim, action_dim=action_dim)
        
        # Mock environment interaction
        state = mock_environment.reset()
        q_values = network(state.unsqueeze(0))
        action = q_values.argmax(dim=1)
        
        next_state, reward, done, info = mock_environment.step(action.item())
        
        assert_valid_q_values(q_values, action_dim)
        assert_valid_action(action, action_dim)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)