#!/usr/bin/env python3
"""
core/learning/neural_trainer.py — ARIASKA Neural Network Training Engine v3.0
🧠 Real Learning Implementation | 🎯 Gradient-Based Updates | 📊 Progressive GPT Reduction
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
from rich.console import Console

console = Console()

class NeuralTrainer:
    """
    Advanced neural network trainer that actually learns from experience.
    Implements proper Q-learning with gradual GPT dependency reduction.
    """
    
    def __init__(
        self,
        policy_network: nn.Module,
        target_network: nn.Module,
        value_network: Optional[nn.Module] = None,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        tau: float = 0.005,
        gradient_clip: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.policy_net = policy_network.to(device)
        self.target_net = target_network.to(device)
        self.value_net = value_network.to(device) if value_network else None
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.gradient_clip = gradient_clip
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate) if self.value_net else None
        
        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=10000, gamma=0.9)
        self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=10000, gamma=0.9) if self.value_optimizer else None
        
        # Training metrics
        self.training_steps = 0
        self.training_losses = deque(maxlen=1000)
        self.value_losses = deque(maxlen=1000)
        self.td_errors = deque(maxlen=1000)
        self.confidence_history = deque(maxlen=1000)
        self.q_value_history = deque(maxlen=1000)
        
        # GPT fallback strategy
        self.gpt_fallback_rate = 0.8  # Start with 80% GPT usage
        self.min_gpt_fallback = 0.1   # Minimum 10% GPT usage
        self.confidence_threshold = 0.7
        
        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        console.print(f"[green]✓ Enhanced Neural Trainer initialized on {device}[/green]")
        self.total_loss = 0.0
        self.q_values_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        
        # Neural network confidence tracking
        self.nn_confidence_threshold = 0.7
        self.confidence_history = deque(maxlen=100)
        self.gpt_fallback_rate = 1.0  # Start with 100% GPT dependency
        
        console.print(f"[green]✓ Neural trainer initialized on {device}[/green]")
        
    def train_on_batch(self, experiences: List[Dict[str, Any]], priorities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Perform actual neural network training with proper backpropagation.
        
        Args:
            experiences: List of experience dictionaries
            priorities: Optional importance sampling weights
            
        Returns:
            Dictionary with training metrics
        """
        if not experiences:
            return {"loss": 0.0, "q_mean": 0.0, "confidence": 0.0}
            
        batch_size = len(experiences)
        
        # Convert experiences to tensors
        states = self._prepare_states([exp['state'] for exp in experiences])
        actions = torch.tensor([exp.get('action', 0) for exp in experiences], 
                              dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in experiences], 
                              dtype=torch.float32, device=self.device)
        next_states = self._prepare_states([exp['next_state'] for exp in experiences])
        dones = torch.tensor([exp.get('done', False) for exp in experiences], 
                            dtype=torch.float32, device=self.device)
        
        # Importance sampling weights
        if priorities is not None:
            weights = torch.tensor(priorities, dtype=torch.float32, device=self.device)
        else:
            weights = torch.ones(batch_size, device=self.device)
            
        # Double DQN: Use policy network to select actions, target network to evaluate
        with torch.no_grad():
            # Get next actions from policy network
            next_q_values_policy = self.policy_net(next_states)
            next_actions = next_q_values_policy.argmax(dim=1)
            
            # Evaluate next actions with target network
            next_q_values_target = self.target_net(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate target Q-values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
        # Get current Q-values from policy network
        current_q_values = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate loss with importance sampling
        td_errors = current_q_values - target_q_values
        loss = (weights * (td_errors ** 2)).mean()
        
        # Backpropagation
        self.policy_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        
        # Update weights
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        
        # Calculate neural network confidence
        with torch.no_grad():
            q_variance = torch.var(current_q_values).item()
            q_mean = torch.mean(current_q_values).item()
            confidence = 1.0 / (1.0 + q_variance)  # Higher confidence with lower variance
            
        # Update confidence tracking
        self.confidence_history.append(confidence)
        avg_confidence = float(np.mean(list(self.confidence_history)))
        
        # Update GPT fallback rate based on confidence
        self._update_gpt_dependency(avg_confidence)
        
        # Soft update target network
        self._soft_update_target()
        
        # Track metrics
        self.training_steps += 1
        self.total_loss += loss.item()
        self.q_values_history.append(q_mean)
        self.loss_history.append(loss.item())
        
        # Value network training (if available)
        value_loss = 0.0
        if self.value_net and self.value_optimizer:
            value_loss = self._train_value_network(states, target_q_values)
            
        return {
            "loss": loss.item(),
            "value_loss": value_loss,
            "q_mean": q_mean,
            "q_variance": q_variance,
            "confidence": confidence,
            "avg_confidence": float(avg_confidence),
            "gpt_dependency": self.gpt_fallback_rate,
            "td_errors": td_errors.detach().cpu().numpy()
        }
    
    def _prepare_states(self, states: List[Any]) -> torch.Tensor:
        """Convert state data to proper tensor format."""
        if isinstance(states[0], torch.Tensor):
            return torch.stack(states).to(self.device)
        elif isinstance(states[0], np.ndarray):
            return torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        elif isinstance(states[0], dict):
            # Convert dictionary states to feature vectors
            features = []
            for state in states:
                feature_vec = self._dict_to_features(state)
                features.append(feature_vec)
            return torch.tensor(np.stack(features), dtype=torch.float32, device=self.device)
        else:
            # Fallback to simple tensor conversion
            return torch.tensor(states, dtype=torch.float32, device=self.device)
    
    def _dict_to_features(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Convert state dictionary to feature vector."""
        # Basic feature extraction - can be enhanced based on specific state format
        features = []
        
        # Extract numeric features
        for key, value in state_dict.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple string hashing for categorical features
                features.append(float(hash(value) % 1000) / 1000.0)
                
        # Ensure minimum feature size
        while len(features) < 32:
            features.append(0.0)
            
        return np.array(features[:512])  # Limit to 512 features
    
    def _train_value_network(self, states: torch.Tensor, targets: torch.Tensor) -> float:
        """Train value network for state value estimation."""
        if self.value_net is None or self.value_optimizer is None:
            return 0.0
            
        predicted_values = self.value_net(states).squeeze()
        value_loss = nn.MSELoss()(predicted_values, targets)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.gradient_clip)
        self.value_optimizer.step()
        if self.value_scheduler:
            self.value_scheduler.step()
        
        return value_loss.item()
    
    def _soft_update_target(self):
        """Soft update of target network parameters."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def _update_gpt_dependency(self, confidence: float):
        """Update GPT fallback rate based on neural network confidence."""
        # Gradually reduce GPT dependency as NN confidence increases
        if confidence > self.nn_confidence_threshold:
            # Reduce GPT dependency
            reduction_rate = 0.01 * (confidence - self.nn_confidence_threshold)
            self.gpt_fallback_rate = max(0.1, self.gpt_fallback_rate - reduction_rate)
        else:
            # Slightly increase GPT dependency if confidence is low
            increase_rate = 0.005 * (self.nn_confidence_threshold - confidence)
            self.gpt_fallback_rate = min(0.9, self.gpt_fallback_rate + increase_rate)
    
    def should_use_gpt(self) -> bool:
        """Determine whether to use GPT or neural network for action selection."""
        return np.random.random() < self.gpt_fallback_rate
    
    def get_neural_action(self, state: Any, epsilon: float = 0.1) -> Tuple[int, float]:
        """Get action from neural network with confidence score."""
        self.policy_net.eval()
        
        with torch.no_grad():
            state_tensor = self._prepare_states([state])
            q_values = self.policy_net(state_tensor)
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(0, q_values.size(1))
                confidence = 0.5  # Random actions have medium confidence
            else:
                action = q_values.argmax().item()
                # Calculate confidence based on Q-value distribution
                q_vals = q_values.squeeze()
                max_q = torch.max(q_vals)
                mean_q = torch.mean(q_vals)
                confidence = float(torch.sigmoid(max_q - mean_q))
                
        self.policy_net.train()
        return action, confidence
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.loss_history:
            return {"training_steps": 0}
            
        return {
            "training_steps": self.training_steps,
            "avg_loss": np.mean(list(self.loss_history)),
            "avg_q_value": np.mean(list(self.q_values_history)) if self.q_values_history else 0.0,
            "avg_confidence": np.mean(list(self.confidence_history)) if self.confidence_history else 0.0,
            "gpt_dependency": self.gpt_fallback_rate,
            "learning_rate": self.policy_optimizer.param_groups[0]['lr']
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'gpt_fallback_rate': self.gpt_fallback_rate,
            'confidence_threshold': self.nn_confidence_threshold
        }
        
        if self.value_net and self.value_optimizer:
            checkpoint['value_net_state_dict'] = self.value_net.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
            
        torch.save(checkpoint, path)
        console.print(f"[green]💾 Training checkpoint saved to {path}[/green]")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.gpt_fallback_rate = checkpoint['gpt_fallback_rate']
        self.nn_confidence_threshold = checkpoint['confidence_threshold']
        
        if self.value_net and 'value_net_state_dict' in checkpoint and self.value_optimizer:
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            
        console.print(f"[green]📂 Training checkpoint loaded from {path}[/green]")
