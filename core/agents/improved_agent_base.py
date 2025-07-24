"""
Improved base agent class for ARIASKA_RL.
Provides standardized interface and common functionality for all agents.
"""
import abc
import logging
import pickle
import time
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
from pathlib import Path

from core.utils.config_manager import AgentConfig
from core.memory.memory_router import MemoryRouter
from core.gpt_manager import GPTManager

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of an agent action"""
    action: Union[int, str, Dict[str, Any]]
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentMetrics:
    """Metrics tracking for agents"""
    episodes_trained: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    best_reward: float = float('-inf')
    worst_reward: float = float('inf')
    last_episode_reward: float = 0.0
    exploration_rate: float = 1.0
    learning_rate: float = 1e-4
    loss_history: List[float] = None
    action_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []
        if self.action_counts is None:
            self.action_counts = {}
    
    def update_episode(self, episode_reward: float):
        """Update metrics after episode completion"""
        self.episodes_trained += 1
        self.total_reward += episode_reward
        self.average_reward = self.total_reward / self.episodes_trained
        self.best_reward = max(self.best_reward, episode_reward)
        self.worst_reward = min(self.worst_reward, episode_reward)
        self.last_episode_reward = episode_reward
    
    def update_loss(self, loss: float):
        """Update loss history"""
        self.loss_history.append(loss)
        # Keep only last 1000 losses to prevent memory issues
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
    
    def get_recent_loss(self, n: int = 10) -> float:
        """Get average of recent losses"""
        if not self.loss_history:
            return 0.0
        recent_losses = self.loss_history[-n:]
        return sum(recent_losses) / len(recent_losses)


class BaseAgent(abc.ABC):
    """
    Improved base class for all ARIASKA_RL agents.
    
    Provides standardized interface and common functionality including:
    - Configuration management
    - Memory integration
    - GPT integration
    - Metrics tracking
    - Checkpoint saving/loading
    - Error handling
    """
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        memory_router: Optional[MemoryRouter] = None,
        gpt_manager: Optional[GPTManager] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration
            memory_router: Shared memory system
            gpt_manager: GPT integration manager
            device: Compute device (CPU/GPU)
        """
        self.agent_id = agent_id
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core systems
        self.memory_router = memory_router
        self.gpt_manager = gpt_manager
        
        # Metrics and state
        self.metrics = AgentMetrics()
        self.is_training = True
        self.episode_count = 0
        self.step_count = 0
        
        # Initialize components
        self._initialize_networks()
        self._initialize_optimizer()
        self._initialize_memory()
        
        logger.info(f"Initialized {self.agent_id} on device {self.device}")
    
    @abc.abstractmethod
    def _initialize_networks(self):
        """Initialize neural networks (implemented by subclasses)"""
        pass
    
    @abc.abstractmethod
    def _initialize_optimizer(self):
        """Initialize optimizer (implemented by subclasses)"""
        pass
    
    def _initialize_memory(self):
        """Initialize memory systems"""
        if self.memory_router is None:
            logger.warning(f"{self.agent_id}: No memory router provided")
        else:
            self.memory_router.register_agent(self.agent_id)
    
    @abc.abstractmethod
    def select_action(
        self,
        state: torch.Tensor,
        training: bool = True,
        **kwargs
    ) -> ActionResult:
        """
        Select action based on current state.
        
        Args:
            state: Current environment state
            training: Whether in training mode
            **kwargs: Additional parameters
            
        Returns:
            ActionResult containing action and metadata
        """
        pass
    
    @abc.abstractmethod
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, float]:
        """
        Update agent parameters using a batch of experiences.
        
        Args:
            batch: Batch of experiences
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def store_experience(
        self,
        state: torch.Tensor,
        action: Union[int, torch.Tensor],
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        **kwargs
    ):
        """Store experience in memory"""
        if self.memory_router is not None:
            experience = {
                'state': state.cpu().numpy() if isinstance(state, torch.Tensor) else state,
                'action': action.item() if isinstance(action, torch.Tensor) else action,
                'reward': reward,
                'next_state': next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state,
                'done': done,
                'agent_id': self.agent_id,
                'timestamp': time.time(),
                **kwargs
            }
            self.memory_router.store_experience(self.agent_id, experience)
    
    def get_gpt_advice(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Get advice from GPT manager"""
        if self.gpt_manager is None:
            logger.warning(f"{self.agent_id}: No GPT manager available")
            return ""
        
        try:
            # Add agent context to prompt
            full_prompt = f"[Agent: {self.agent_id}] {prompt}"
            if context:
                full_prompt += f"\nContext: {context}"
            
            response = self.gpt_manager.get_response(
                prompt=full_prompt,
                agent_id=self.agent_id,
                **kwargs
            )
            
            return response
        except Exception as e:
            logger.error(f"{self.agent_id}: Error getting GPT advice: {e}")
            return ""
    
    def set_training_mode(self, training: bool):
        """Set training mode"""
        self.is_training = training
        if hasattr(self, 'policy_net'):
            self.policy_net.train(training)
        if hasattr(self, 'value_net'):
            self.value_net.train(training)
    
    def update_exploration_rate(self, episode: int):
        """Update exploration rate (epsilon decay)"""
        if hasattr(self.config, 'epsilon_start') and hasattr(self.config, 'epsilon_end'):
            decay_rate = max(
                self.config.epsilon_end,
                self.config.epsilon_start * (self.config.epsilon_decay ** episode)
            )
            self.metrics.exploration_rate = decay_rate
    
    def save_checkpoint(self, filepath: str) -> bool:
        """
        Save agent checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_data = {
                'agent_id': self.agent_id,
                'config': self.config.__dict__,
                'metrics': self.metrics,
                'episode_count': self.episode_count,
                'step_count': self.step_count,
                'timestamp': time.time(),
            }
            
            # Add network states if they exist
            if hasattr(self, 'policy_net'):
                checkpoint_data['policy_net_state'] = self.policy_net.state_dict()
            if hasattr(self, 'value_net'):
                checkpoint_data['value_net_state'] = self.value_net.state_dict()
            if hasattr(self, 'optimizer'):
                checkpoint_data['optimizer_state'] = self.optimizer.state_dict()
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint
            torch.save(checkpoint_data, filepath)
            logger.info(f"{self.agent_id}: Checkpoint saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load agent checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"{self.agent_id}: Checkpoint file not found: {filepath}")
                return False
            
            checkpoint_data = torch.load(filepath, map_location=self.device)
            
            # Restore metrics and counters
            if 'metrics' in checkpoint_data:
                self.metrics = checkpoint_data['metrics']
            if 'episode_count' in checkpoint_data:
                self.episode_count = checkpoint_data['episode_count']
            if 'step_count' in checkpoint_data:
                self.step_count = checkpoint_data['step_count']
            
            # Restore network states
            if hasattr(self, 'policy_net') and 'policy_net_state' in checkpoint_data:
                self.policy_net.load_state_dict(checkpoint_data['policy_net_state'])
            if hasattr(self, 'value_net') and 'value_net_state' in checkpoint_data:
                self.value_net.load_state_dict(checkpoint_data['value_net_state'])
            if hasattr(self, 'optimizer') and 'optimizer_state' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
            
            logger.info(f"{self.agent_id}: Checkpoint loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Failed to load checkpoint: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics"""
        return {
            'agent_id': self.agent_id,
            'episodes_trained': self.metrics.episodes_trained,
            'total_steps': self.metrics.total_steps,
            'average_reward': self.metrics.average_reward,
            'best_reward': self.metrics.best_reward,
            'last_episode_reward': self.metrics.last_episode_reward,
            'exploration_rate': self.metrics.exploration_rate,
            'recent_loss': self.metrics.get_recent_loss(),
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'is_training': self.is_training,
        }
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_count += 1
        # Update exploration rate
        self.update_exploration_rate(self.episode_count)
    
    def step(self):
        """Increment step counter"""
        self.step_count += 1
        self.metrics.total_steps += 1
    
    def validate_state(self, state: torch.Tensor) -> bool:
        """Validate input state tensor"""
        if not isinstance(state, torch.Tensor):
            logger.error(f"{self.agent_id}: State must be a torch.Tensor")
            return False
        
        if torch.isnan(state).any() or torch.isinf(state).any():
            logger.error(f"{self.agent_id}: State contains NaN or Inf values")
            return False
        
        return True
    
    def safe_tensor(self, data: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """Safely convert data to tensor on correct device"""
        try:
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).float().to(self.device)
            elif isinstance(data, list):
                return torch.tensor(data, dtype=torch.float32).to(self.device)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            logger.error(f"{self.agent_id}: Error converting to tensor: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1, device=self.device)
    
    def log_action(self, action: Union[int, str], reasoning: str = ""):
        """Log action for debugging and analysis"""
        action_str = str(action)
        if action_str in self.metrics.action_counts:
            self.metrics.action_counts[action_str] += 1
        else:
            self.metrics.action_counts[action_str] = 1
        
        logger.debug(f"{self.agent_id}: Action {action} - {reasoning}")
    
    def __str__(self) -> str:
        """String representation of agent"""
        return f"{self.agent_id}(episodes={self.episode_count}, avg_reward={self.metrics.average_reward:.2f})"
    
    def __repr__(self) -> str:
        """Detailed representation of agent"""
        return (f"{self.__class__.__name__}("
                f"agent_id='{self.agent_id}', "
                f"episodes={self.episode_count}, "
                f"device='{self.device}', "
                f"training={self.is_training})")


class RLAgent(BaseAgent):
    """
    Base class for reinforcement learning agents.
    Extends BaseAgent with RL-specific functionality.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = None
        self.target_network_update_frequency = getattr(
            self.config, 'target_update_frequency', 100
        )
        self.update_counter = 0
    
    def soft_update_target_network(self, tau: float = 0.005):
        """Soft update of target network"""
        if hasattr(self, 'target_net') and hasattr(self, 'policy_net'):
            for target_param, local_param in zip(
                self.target_net.parameters(), 
                self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    tau * local_param.data + (1.0 - tau) * target_param.data
                )
    
    def hard_update_target_network(self):
        """Hard update of target network"""
        if hasattr(self, 'target_net') and hasattr(self, 'policy_net'):
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def should_update_target_network(self) -> bool:
        """Check if target network should be updated"""
        return self.update_counter % self.target_network_update_frequency == 0
    
    def post_update_hook(self):
        """Hook called after each update"""
        self.update_counter += 1
        if self.should_update_target_network():
            self.hard_update_target_network()


# Example concrete implementation
class DQNAgent(RLAgent):
    """Example DQN agent implementation using the improved base class"""
    
    def _initialize_networks(self):
        """Initialize DQN networks"""
        from core.algorithms.dqn import DQNNetwork
        
        self.policy_net = DQNNetwork(
            state_dim=getattr(self.config, 'state_dim', 128),
            action_dim=getattr(self.config, 'action_dim', 4),
            hidden_dims=getattr(self.config, 'hidden_dims', [256, 256]),
            dueling=getattr(self.config, 'dueling', True)
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            state_dim=getattr(self.config, 'state_dim', 128),
            action_dim=getattr(self.config, 'action_dim', 4),
            hidden_dims=getattr(self.config, 'hidden_dims', [256, 256]),
            dueling=getattr(self.config, 'dueling', True)
        ).to(self.device)
        
        # Initialize target network with same weights
        self.hard_update_target_network()
    
    def _initialize_optimizer(self):
        """Initialize optimizer"""
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate
        )
    
    def select_action(self, state: torch.Tensor, training: bool = True, **kwargs) -> ActionResult:
        """Select action using epsilon-greedy strategy"""
        if not self.validate_state(state):
            return ActionResult(action=0, confidence=0.0, reasoning="Invalid state")
        
        state = self.safe_tensor(state)
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.metrics.exploration_rate:
            action = np.random.randint(0, getattr(self.config, 'action_dim', 4))
            reasoning = f"Random exploration (ε={self.metrics.exploration_rate:.3f})"
            confidence = 0.0
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0) if state.dim() == 1 else state)
                action = q_values.argmax().item()
                confidence = torch.softmax(q_values, dim=-1).max().item()
                reasoning = f"Greedy selection (Q={q_values.max().item():.3f})"
        
        self.log_action(action, reasoning)
        
        return ActionResult(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={'q_values': q_values.cpu().numpy() if 'q_values' in locals() else None}
        )
    
    def update(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, float]:
        """Update DQN using batch of experiences"""
        if len(batch) == 0:
            return {'loss': 0.0}
        
        # Extract batch components
        states = self.safe_tensor(batch['states'])
        actions = self.safe_tensor(batch['actions']).long()
        rewards = self.safe_tensor(batch['rewards'])
        next_states = self.safe_tensor(batch['next_states'])
        dones = self.safe_tensor(batch['dones']).bool()
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss
        loss = torch.nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.config, 'gradient_clip_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.config.gradient_clip_norm
            )
        
        self.optimizer.step()
        
        # Update metrics
        self.metrics.update_loss(loss.item())
        self.post_update_hook()
        
        return {
            'loss': loss.item(),
            'avg_q_value': current_q_values.mean().item(),
            'target_q_value': target_q_values.mean().item(),
        }