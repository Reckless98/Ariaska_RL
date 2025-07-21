#!/usr/bin/env python3
"""
Advanced Neural Architectures for ARIASKA_RL
🧠 Double DQN | 🎯 Dueling Networks | 🔄 Prioritized Experience Replay | 🚀 Rainbow DQN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
import math

# Advanced replay buffer with prioritized sampling
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for enhanced learning efficiency."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength
        self.beta = beta    # Importance sampling correction
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small positive constant to prevent zero priorities
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done, priority=None):
        """Add experience with priority."""
        if priority is None:
            priority = self.max_priority
            
        experience = Experience(state, action, reward, next_state, done, priority)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with importance weights."""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
            
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network with separate value and advantage streams."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Optional[List[int]] = None):
        super(DuelingDQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through dueling architecture."""
        # Feature extraction
        x = state
        for layer in self.feature_layers:
            x = layer(x)
        
        # Separate value and advantage computation
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine using dueling architecture formula
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values

class NoisyLinear(nn.Module):
    """Noisy linear layer for parameter space noise exploration."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Generate new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int):
        """Generate factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    """Rainbow DQN combining multiple DQN improvements."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Optional[List[int]] = None,
                 atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        super(RainbowDQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Distributional RL support
        self.register_buffer('supports', torch.linspace(v_min, v_max, atoms))
        self.delta_z = (v_max - v_min) / (atoms - 1)
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # Dueling architecture with noisy layers
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            NoisyLinear(hidden_dims[1], atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            NoisyLinear(hidden_dims[1], action_dim * atoms)
        )
        
    def forward(self, state, log: bool = False):
        """Forward pass returning value distribution."""
        batch_size = state.size(0)
        
        # Feature extraction
        features = self.features(state)
        
        # Value and advantage streams
        value = self.value_stream(features)  # (batch_size, atoms)
        advantages = self.advantage_stream(features)  # (batch_size, action_dim * atoms)
        
        # Reshape advantages
        advantages = advantages.view(batch_size, self.action_dim, self.atoms)
        
        # Dueling aggregation
        value = value.view(batch_size, 1, self.atoms)
        advantages = advantages - advantages.mean(dim=1, keepdim=True)
        q_atoms = value + advantages  # (batch_size, action_dim, atoms)
        
        if log:
            q_atoms = F.log_softmax(q_atoms, dim=2)
        else:
            q_atoms = F.softmax(q_atoms, dim=2)
            
        return q_atoms
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class AdvancedDQNTrainer:
    """Advanced DQN trainer with Rainbow improvements."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 1e-4, gamma: float = 0.99,
                 tau: float = 0.005, device: str = 'cuda'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.q_network = RainbowDQN(state_dim, action_dim).to(device)
        self.target_network = RainbowDQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=50000)
        
        # Training metrics
        self.training_step = 0
        self.losses = []
        self.q_values_history = []
        
        # Multi-step learning
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy or noisy exploration."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Reset noise for exploration
        self.q_network.reset_noise()
        
        with torch.no_grad():
            # Get Q-value distribution
            q_dist = self.q_network(state_tensor)
            
            # Calculate expected Q-values
            q_values = (q_dist * self.q_network.supports).sum(dim=2)
            
            # Store Q-values for monitoring
            self.q_values_history.append(q_values.max().item())
            if len(self.q_values_history) > 1000:
                self.q_values_history.pop(0)
        
        # Epsilon-greedy selection (with noisy nets, epsilon can be smaller)
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in n-step buffer and replay buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # Calculate n-step return
            n_step_state, n_step_action = self.n_step_buffer[0][:2]
            n_step_reward = sum([self.gamma**i * exp[2] for i, exp in enumerate(self.n_step_buffer)])
            n_step_next_state, n_step_done = self.n_step_buffer[-1][3:]
            
            # Calculate initial priority (using TD error approximation)
            priority = abs(n_step_reward) + 1.0
            
            self.replay_buffer.add(
                n_step_state, n_step_action, n_step_reward, 
                n_step_next_state, n_step_done, priority
            )
    
    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """Update networks using prioritized experience replay."""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(batch_size)
        if not experiences:
            return {}
        
        # Prepare batch data
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-value distribution
        current_q_dist = self.q_network(states)
        current_q_dist = current_q_dist[range(batch_size), actions]
        
        # Next Q-value distribution (Double DQN)
        with torch.no_grad():
            # Use online network to select actions
            next_q_dist = self.q_network(next_states)
            next_q_values = (next_q_dist * self.q_network.supports).sum(dim=2)
            next_actions = next_q_values.argmax(dim=1)
            
            # Use target network to evaluate actions
            target_q_dist = self.target_network(next_states)
            target_q_dist = target_q_dist[range(batch_size), next_actions]
            
            # Distributional Bellman update
            target_supports = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * \
                            self.gamma**self.n_step * self.q_network.supports.unsqueeze(0)
            target_supports = target_supports.clamp(self.q_network.v_min, self.q_network.v_max)
            
            # Project onto support atoms
            b = (target_supports - self.q_network.v_min) / self.q_network.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Distribute probability
            target_q_dist_projected = torch.zeros_like(target_q_dist)
            target_q_dist_projected.view(-1).index_add_(
                0, (l + (torch.arange(batch_size).unsqueeze(1) * self.q_network.atoms).to(self.device)).view(-1),
                (target_q_dist * (u.float() - b)).view(-1)
            )
            target_q_dist_projected.view(-1).index_add_(
                0, (u + (torch.arange(batch_size).unsqueeze(1) * self.q_network.atoms).to(self.device)).view(-1),
                (target_q_dist * (b - l.float())).view(-1)
            )
        
        # Calculate loss (cross-entropy)
        loss = -(target_q_dist_projected * current_q_dist.log()).sum(dim=1)
        
        # Apply importance sampling weights
        loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            td_errors = (target_q_dist_projected * current_q_dist.log()).sum(dim=1).abs()
            self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
        
        # Soft update target network
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # Store metrics
        self.training_step += 1
        self.losses.append(loss.item())
        if len(self.losses) > 1000:
            self.losses.pop(0)
        
        return {
            'loss': loss.item(),
            'q_value_mean': torch.mean((current_q_dist * self.q_network.supports).sum(dim=1)).item(),
            'training_step': self.training_step
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'losses': self.losses,
            'q_values_history': self.q_values_history
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.losses = checkpoint.get('losses', [])
        self.q_values_history = checkpoint.get('q_values_history', [])

class AdaptiveCurriculum:
    """Adaptive curriculum learning for progressive difficulty."""
    
    def __init__(self, initial_difficulty: float = 0.1, max_difficulty: float = 1.0,
                 adaptation_rate: float = 0.1):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.adaptation_rate = adaptation_rate
        
        self.success_history = deque(maxlen=100)
        self.performance_threshold = 0.7
        
    def update_difficulty(self, success_rate: float):
        """Update curriculum difficulty based on performance."""
        self.success_history.append(success_rate)
        
        if len(self.success_history) >= 10:
            recent_performance = np.mean(list(self.success_history)[-10:])
            
            if recent_performance > self.performance_threshold:
                # Increase difficulty
                self.current_difficulty = min(
                    self.max_difficulty,
                    self.current_difficulty + self.adaptation_rate
                )
            elif recent_performance < 0.3:
                # Decrease difficulty
                self.current_difficulty = max(
                    0.1,
                    self.current_difficulty - self.adaptation_rate
                )
    
    def get_current_difficulty(self) -> float:
        """Get current curriculum difficulty."""
        return self.current_difficulty
    
    def get_scenario_config(self) -> Dict[str, Any]:
        """Get environment configuration based on current difficulty."""
        return {
            'vulnerability_density': self.current_difficulty,
            'network_complexity': int(self.current_difficulty * 10) + 1,
            'defensive_strength': self.current_difficulty * 0.8,
            'time_pressure': self.current_difficulty > 0.5
        }
