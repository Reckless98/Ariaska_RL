# core/algorithms/dqn.py — ARIASKA_RL Deep Q-Network Implementation v2.1 APEX
# 🧠 Policy Network & Value Network Architecture | 🎯 Double DQN | ⚡ Dueling Networks

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from rich.console import Console

console = Console()

class DQNNetwork(nn.Module):
    """
    Advanced Deep Q-Network implementation with dueling network architecture.
    
    Features:
    - Dueling networks (separate value and advantage streams)
    - Flexible input/output dimensions
    - Dropout for regularization
    - Layer normalization for stable training
    - Configurable activation functions
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        dueling: bool = True,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True
    ):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        
        # Define activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
        
        # Build shared feature layers
        self.feature_layers = nn.ModuleList()
        last_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            self.feature_layers.append(nn.Linear(last_dim, hidden_dim))
            if use_layer_norm:
                self.feature_layers.append(nn.LayerNorm(hidden_dim))
            if dropout_rate > 0:
                self.feature_layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        
        # Dueling architecture with separate value and advantage streams
        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(last_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(last_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )
        else:
            # Traditional Q-network
            self.q_output = nn.Sequential(
                nn.Linear(last_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Process shared features
        x = state
        for layer in self.feature_layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.activation(x)
        
        # Dueling architecture
        if self.dueling:
            value = self.value_stream(x)
            advantages = self.advantage_stream(x)
            
            # Combine value and advantages using the dueling formula
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            # Traditional Q-network
            q_values = self.q_output(x)
            
        return q_values
        
    def save(self, path: str) -> None:
        """Save model weights to file."""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str, device: str = None) -> None:
        """Load model weights from file."""
        if device:
            self.load_state_dict(torch.load(path, map_location=device))
        else:
            self.load_state_dict(torch.load(path))


class DQN:
    """
    Double DQN Agent with prioritized experience replay.
    
    Features:
    - Double Q-learning for more stable target values
    - Prioritized Experience Replay
    - Epsilon-greedy exploration with annealing
    - Target network for stable training
    - Gradient clipping to prevent exploding gradients
    - Learning rate scheduler
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_buffer = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 10000,
        target_update_freq: int = 1000,
        dueling: bool = True,
        double_q: bool = True,
        gradient_clip: float = 1.0,
        tau: float = 0.005,  # Soft update parameter
        scheduler_step_size: int = 10000,
        scheduler_gamma: float = 0.9,
        **kwargs
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.target_update_freq = target_update_freq
        self.dueling = dueling
        self.double_q = double_q
        self.gradient_clip = gradient_clip
        self.tau = tau
        self.steps_done = 0
        
        # Initialize policy network (active) and target network
        self.policy_net = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dueling=dueling
        ).to(device)
        
        self.target_net = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dueling=dueling
        ).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )
        
        # Set up replay buffer
        self.replay_buffer = replay_buffer
        
        console.print(f"[green]✓ DQN Agent initialized on {device}[/green]")
        if self.dueling:
            console.print("[cyan]Using Dueling Network Architecture[/cyan]")
        if self.double_q:
            console.print("[cyan]Using Double Q-Learning[/cyan]")
    
    def select_action(self, state: Union[np.ndarray, torch.Tensor], eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        # Handle different input types
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            state = state.unsqueeze(0).to(self.device)
        else:
            state = state.to(self.device)
            
        # Use greedy policy for evaluation
        sample = random.random()
        if eval_mode:
            eps_threshold = 0.01  # Small epsilon for evaluation
        else:
            # Annealing epsilon
            eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            np.exp(-1. * self.steps_done / self.epsilon_decay)
            self.epsilon = eps_threshold
            
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                # Return action with highest Q-value
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            # Return random action
            return random.randrange(self.action_dim)
    
    def optimize_model(self, batch_size: int = 64) -> Dict[str, float]:
        """Perform one step of optimization."""
        if not self.replay_buffer or len(self.replay_buffer) < batch_size:
            return {"loss": 0.0}
            
        # Sample from replay buffer
        batch = self.replay_buffer.sample_and_split(batch_size)
        if not batch:
            return {"loss": 0.0}
            
        # Extract batch components
        states = batch['state']
        actions = batch['action']
        rewards = batch['reward']
        next_states = batch['next_state']
        dones = batch['done']
        weights = batch.get('weights', torch.ones(batch_size))
        indices = batch.get('indices', None)
        
        # Convert to tensors if needed
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions).to(self.device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.FloatTensor(rewards).to(self.device)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.FloatTensor(next_states).to(self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.FloatTensor(dones).to(self.device)
        if not isinstance(weights, torch.Tensor):
            weights = torch.FloatTensor(weights).to(self.device)
            
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next state values using Double DQN or regular DQN
        with torch.no_grad():
            if self.double_q:
                # Double DQN: get actions from policy net, values from target net
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Regular DQN: get max values directly from target net
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # Compute expected Q values
            expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values, reduction='none')
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights.unsqueeze(1)).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Clip gradients to prevent exploding gradients
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self._update_target_network()
            
        # Update priorities in replay buffer
        if self.replay_buffer and hasattr(self.replay_buffer, 'update_priorities') and indices is not None:
            with torch.no_grad():
                td_errors = torch.abs(q_values - expected_q_values).detach().cpu().numpy()
                self.replay_buffer.update_priorities(indices, td_errors.flatten())
        
        return {
            "loss": weighted_loss.item(),
            "q_mean": q_values.mean().item(),
            "q_max": q_values.max().item(),
            "q_min": q_values.min().item(),
            "epsilon": self.epsilon
        }
    
    def _update_target_network(self):
        """Update target network using soft or hard update."""
        if self.tau < 1.0:
            # Soft update
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * policy_param.data
                )
        else:
            # Hard update
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def save_models(self, policy_path: str, target_path: str = None):
        """Save model weights."""
        self.policy_net.save(policy_path)
        if target_path:
            self.target_net.save(target_path)
            
    def load_models(self, policy_path: str, target_path: str = None):
        """Load model weights."""
        self.policy_net.load(policy_path, self.device)
        if target_path:
            self.target_net.load(target_path, self.device)
        else:
            # Copy policy weights to target
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def to_device(self, device):
        """Move model to specified device."""
        self.device = device
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        
    def train(self):
        """Set model to training mode."""
        self.policy_net.train()
        
    def eval(self):
        """Set model to evaluation mode."""
        self.policy_net.eval()
        self.target_net.eval()

# Add imports required at the top of the file
import random

# ─────────────────────────────────────────────
# 🚀 CLI Test Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    console.print("[bold magenta]🚀 Testing Double DQN Implementation[/bold magenta]")
    
    # Create a simple environment for testing
    state_size = 8
    action_size = 4
    
    # Create DQN
    dqn = DQN(
        state_dim=state_size,
        action_dim=action_size,
        hidden_dims=[128, 128],
        device="cpu",
        dueling=True,
        double_q=True
    )
    
    # Test forward pass
    random_state = np.random.rand(state_size)
    action = dqn.select_action(random_state, eval_mode=False)
    console.print(f"Selected action: {action}")
    
    # Test update with dummy batch
    batch = []
    for _ in range(10):
        state = np.random.rand(state_size)
        action = np.random.randint(action_size)
        reward = np.random.rand()
        next_state = np.random.rand(state_size)
        done = np.random.rand() > 0.8
        batch.append((state, action, reward, next_state, done))
        
    loss = dqn.optimize_model(batch_size=10)
    console.print(f"Training loss: {loss['loss']:.4f}")
    
    # Test checkpoint save/load
    test_policy_path = "test_policy_checkpoint.pt"
    test_target_path = "test_target_checkpoint.pt"
    dqn.save_models(test_policy_path, test_target_path)
    dqn.load_models(test_policy_path, test_target_path)
    
    if os.path.exists(test_policy_path):
        os.remove(test_policy_path)
        console.print(f"[green]✓ Test policy checkpoint deleted[/green]")
    if os.path.exists(test_target_path):
        os.remove(test_target_path)
        console.print(f"[green]✓ Test target checkpoint deleted[/green]")
