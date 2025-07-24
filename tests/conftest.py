"""
Test configuration and fixtures for ARIASKA_RL test suite.
"""
import pytest
import torch
import numpy as np
import tempfile
import os
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

# Test configuration
@pytest.fixture
def device():
    """Provide test device (CPU for CI/CD compatibility)"""
    return torch.device("cpu")

@pytest.fixture
def sample_state():
    """Provide sample state tensor for testing"""
    return torch.randn(1, 128)  # Batch size 1, state dim 128

@pytest.fixture
def sample_action():
    """Provide sample action for testing"""
    return torch.tensor([0])  # Single action

@pytest.fixture
def sample_reward():
    """Provide sample reward for testing"""
    return torch.tensor([1.0])

@pytest.fixture
def sample_config():
    """Provide sample agent configuration"""
    return {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'memory_size': 1000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 100,
        'hidden_dims': [64, 64],
        'state_dim': 128,
        'action_dim': 4
    }

@pytest.fixture
def temp_dir():
    """Provide temporary directory for testing file operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_gpt_manager():
    """Mock GPT manager for testing without API calls"""
    mock = Mock()
    mock.get_response.return_value = "Mock GPT response"
    mock.get_embedding.return_value = np.random.randn(512)  # Mock embedding
    return mock

@pytest.fixture
def mock_environment():
    """Mock cyber environment for testing"""
    mock = Mock()
    mock.reset.return_value = torch.randn(128)  # Mock state
    mock.step.return_value = (
        torch.randn(128),  # next_state
        1.0,              # reward
        False,            # done
        {}                # info
    )
    mock.action_space = Mock()
    mock.action_space.n = 4
    mock.observation_space = Mock()
    mock.observation_space.shape = (128,)
    return mock

@pytest.fixture
def sample_trajectory():
    """Provide sample trajectory data for testing memory systems"""
    return {
        'states': torch.randn(10, 128),
        'actions': torch.randint(0, 4, (10,)),
        'rewards': torch.randn(10),
        'next_states': torch.randn(10, 128),
        'dones': torch.zeros(10, dtype=torch.bool)
    }

# Custom assertions for RL testing
def assert_valid_q_values(q_values: torch.Tensor, action_dim: int):
    """Assert Q-values are valid"""
    assert q_values.shape[-1] == action_dim, f"Expected {action_dim} Q-values, got {q_values.shape[-1]}"
    assert torch.isfinite(q_values).all(), "Q-values contain infinite or NaN values"

def assert_valid_action(action: torch.Tensor, action_dim: int):
    """Assert action is valid"""
    assert 0 <= action.item() < action_dim, f"Action {action.item()} is out of bounds [0, {action_dim})"

def assert_valid_loss(loss: torch.Tensor):
    """Assert loss is valid for training"""
    assert torch.isfinite(loss), "Loss is infinite or NaN"
    assert loss >= 0, "Loss should be non-negative"

# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")