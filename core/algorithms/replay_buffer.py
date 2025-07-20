# core/algorithms/replay_buffer.py — ARIASKA_RL Prioritized Memory System v2.1 APEX
# 🧠 Structured Experience Storage | 🎯 Prioritized Sampling | 🗄️ Memory Deduplication

import os
import random
import json
import time
import numpy as np
import torch
from collections import deque
from rich.console import Console
from typing import Dict, List, Any, Union, Tuple, Optional

console = Console()

class ReplayBuffer:
    """
    Advanced replay buffer with prioritized experience replay and memory deduplication.
    
    Features:
    - Prioritized sampling based on TD error or reward magnitude
    - Memory deduplication to prevent redundant experiences
    - Selective memory retention based on environmental phase
    - Integration with MemoryRouter for multi-agent experience sharing
    - Persistent storage and loading from disk
    - GPT token usage tracking for LLM-generated experiences
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        prioritized: bool = True,
        alpha: float = 0.6, 
        beta: float = 0.4,
        beta_increment: float = 0.001,
        agent_id: str = None,
        storage_path: str = "core/memories",
        memory_router = None,
        deduplicate: bool = True
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment  # Beta annealing
        self.epsilon = 1e-6  # Small value to ensure non-zero priorities
        self.agent_id = agent_id or "agent"
        self.memory_router = memory_router
        self.deduplicate = deduplicate
        
        # Main memory structures
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Track insertion timestamps for cleanup
        self.timestamps = np.zeros((capacity,), dtype=np.float32)
        
        # Phase-specific memory segments
        self.phase_buffers = {
            "recon": [], 
            "enumeration": [], 
            "exploit": [], 
            "privesc": [],
            "exfiltrate": []
        }
        self.phase_priorities = {
            "recon": [],
            "enumeration": [],
            "exploit": [],
            "privesc": [],
            "exfiltrate": []
        }
        
        # Fingerprint cache for deduplication
        self.fingerprints = set()
        
        # Set up storage path
        self.storage_path = storage_path
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
            
        # Recent experience tracking
        self.last_experiences = deque(maxlen=10)
        
        # Stats tracking
        self.total_adds = 0
        self.total_samples = 0
        self.dupes_filtered = 0
        self.gpt_tokens = {}  # Track token usage per experience
        
        console.print(f"[green]✓ ReplayBuffer initialized for {agent_id or 'agent'}[/green]")
    
    def _get_fingerprint(self, experience: Dict[str, Any]) -> str:
        """Generate a unique fingerprint for experience deduplication."""
        if not isinstance(experience, dict):
            return str(hash(time.time()))
            
        # Extract key components depending on type
        fp_components = []
        
        # Extract action fingerprint
        action = experience.get('action')
        if isinstance(action, str):
            # For string actions (commands), use the first word/command
            fp_components.append(action.split()[0] if ' ' in action else action)
        elif isinstance(action, (int, float)):
            fp_components.append(str(action))
        else:
            fp_components.append('unknown_action')
            
        # Get phase
        phase = experience.get('phase', 'unknown')
        fp_components.append(phase)
        
        # Get reward bucket (rounded to nearest integer)
        reward = experience.get('reward', 0)
        if isinstance(reward, (int, float)):
            reward_bucket = round(reward)
            fp_components.append(f"r{reward_bucket}")
            
        # Create fingerprint by joining components
        return '_'.join([str(c) for c in fp_components if c is not None])
    
    def add(self, experience: Dict[str, Any], priority: float = None) -> bool:
        """
        Add experience to replay buffer with deduplication and prioritization.
        
        Args:
            experience: Dictionary containing state, action, reward, next_state, done
            priority: Optional explicit priority, otherwise calculated from reward
            
        Returns:
            bool: True if added successfully, False if filtered as duplicate
        """
        # Track statistics
        self.total_adds += 1
        
        # Check for deduplication if enabled
        if self.deduplicate:
            fingerprint = self._get_fingerprint(experience)
            if fingerprint in self.fingerprints:
                self.dupes_filtered += 1
                return False
            self.fingerprints.add(fingerprint)
        
        # Calculate priority if not provided
        if priority is None:
            # Use absolute reward as priority basis with small offset to ensure non-zero
            reward = experience.get('reward', 0)
            priority = abs(reward) + self.epsilon
            
            # Boost priority for terminal states and high-reward experiences
            if experience.get('done', False):
                priority *= 1.5
            if abs(reward) > 10:
                priority *= 1.2
                
        # Apply priority exponent
        priority = priority ** self.alpha
        
        # Store timestamp for potential cleanup
        current_time = time.time()
        
        # Add to main buffer
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.priorities[self.size] = priority
            self.timestamps[self.size] = current_time
            self.size += 1
        else:
            # Replace oldest experience in the buffer
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            self.timestamps[self.position] = current_time
            self.position = (self.position + 1) % self.capacity
            
        # Add to phase-specific buffer if phase is provided
        phase = experience.get('phase')
        if phase and phase in self.phase_buffers:
            max_phase_entries = 1000  # Limit phase-specific memories
            phase_buffer = self.phase_buffers[phase]
            phase_priorities = self.phase_priorities[phase]
            
            if len(phase_buffer) < max_phase_entries:
                phase_buffer.append(experience)
                phase_priorities.append(priority)
            else:
                # Replace lowest priority experience in phase buffer
                if phase_priorities:
                    min_idx = np.argmin(np.array(phase_priorities))
                    phase_buffer[min_idx] = experience
                    phase_priorities[min_idx] = priority
                    
        # Track in recent experiences
        self.last_experiences.append(experience)
        
        # Track any GPT token usage if present
        if 'gpt_tokens' in experience:
            tokens = experience['gpt_tokens']
            model = experience.get('gpt_model', 'default')
            self.gpt_tokens[model] = self.gpt_tokens.get(model, 0) + tokens
            
        # Report to memory router if available
        if self.memory_router and hasattr(self.memory_router, 'record_experience'):
            self.memory_router.record_experience(self.agent_id, experience, priority)
            
        return True
    
    def sample(self, batch_size: int, prioritized: bool = None) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences with prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            prioritized: Override whether to use prioritization
            
        Returns:
            Tuple containing:
                - List of experience dictionaries
                - Array of indices sampled
                - Array of importance sampling weights
        """
        # Update stats
        self.total_samples += 1
        
        # Can't sample if buffer is empty
        if self.size == 0:
            return [], np.array([]), np.array([])
            
        # Determine whether to use prioritization
        use_prioritized = prioritized if prioritized is not None else self.prioritized
        
        # Prepare batch
        batch_size = min(batch_size, self.size)
        
        if use_prioritized:
            # Get sampling probabilities from priorities
            probs = self.priorities[:self.size] / np.sum(self.priorities[:self.size])
            
            # Sample indices according to probabilities
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
            
            # Calculate importance sampling weights
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights = weights / np.max(weights)  # Normalize
            
            # Increment beta for annealing
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # Uniform sampling
            indices = np.random.choice(self.size, batch_size, replace=False)
            weights = np.ones(batch_size)
            
        # Extract experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def sample_phase(self, phase: str, batch_size: int = 16) -> List[Dict[str, Any]]:
        """Sample experiences from a specific phase."""
        if phase not in self.phase_buffers or not self.phase_buffers[phase]:
            return []
            
        phase_buffer = self.phase_buffers[phase]
        count = min(batch_size, len(phase_buffer))
        
        # Simple random sampling from phase buffer
        return random.sample(phase_buffer, count)
    
    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray) -> None:
        """Update priorities for experiences based on TD error or other criterion."""
        if not isinstance(indices, (list, np.ndarray)) or not len(indices):
            return
            
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < self.size:
                # Ensure non-zero priority with epsilon
                priority = (abs(priority) + self.epsilon) ** self.alpha
                self.priorities[idx] = priority
    
    def sample_and_split(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample and convert to PyTorch tensors organized by key."""
        experiences, indices, weights = self.sample(batch_size)
        if not experiences:
            return {}
            
        # Extract tensors by key
        result = {}
        for key in experiences[0].keys():
            if key in ['state', 'next_state', 'action', 'reward', 'done']:
                # Convert to appropriate tensor type
                try:
                    if key in ['state', 'next_state']:
                        # States might be nested structures
                        if isinstance(experiences[0][key], dict):
                            # Handle dictionary states
                            result[key] = {
                                k: torch.tensor([exp[key][k] for exp in experiences if k in exp[key]])
                                for k in experiences[0][key]
                            }
                        elif isinstance(experiences[0][key], (list, np.ndarray)):
                            # Handle array states
                            result[key] = torch.tensor([exp[key] for exp in experiences])
                        else:
                            # Fallback
                            result[key] = [exp[key] for exp in experiences]
                    elif key == 'action':
                        # Actions might be strings or numbers
                        if isinstance(experiences[0][key], (int, float, np.integer, np.float)):
                            result[key] = torch.tensor([exp[key] for exp in experiences]).long()
                        else:
                            result[key] = [exp[key] for exp in experiences]
                    elif key == 'reward':
                        result[key] = torch.tensor([exp[key] for exp in experiences]).float()
                    elif key == 'done':
                        result[key] = torch.tensor([exp[key] for exp in experiences]).bool()
                    else:
                        # Other fields just collect as is
                        result[key] = [exp[key] for exp in experiences]
                except Exception as e:
                    console.print(f"[yellow]⚠ Error processing {key}: {e}[/yellow]")
                    result[key] = [exp.get(key) for exp in experiences]
                    
        # Add indices and weights
        result['indices'] = indices
        result['weights'] = weights
        
        return result
        
    def optimize_and_deduplicate(self) -> int:
        """Optimize buffer by removing duplicates and low-priority experiences."""
        if not self.deduplicate or self.size == 0:
            return 0
            
        # Track how many items removed
        removed = 0
        
        # Get all items, remove dupes, sort by priority
        all_items = []
        all_fingerprints = set()
        
        for i in range(self.size):
            fp = self._get_fingerprint(self.buffer[i])
            if fp not in all_fingerprints:
                all_fingerprints.add(fp)
                all_items.append((i, self.buffer[i], self.priorities[i], fp))
                
        # Sort by priority (higher first)
        all_items.sort(key=lambda x: x[2], reverse=True)
        
        # Rebuild buffer with prioritized unique items
        new_size = min(len(all_items), self.capacity)
        new_buffer = []
        new_priorities = np.zeros((self.capacity,), dtype=np.float32)
        new_timestamps = np.zeros((self.capacity,), dtype=np.float32)
        new_fingerprints = set()
        
        for i in range(new_size):
            idx, item, priority, fp = all_items[i]
            new_buffer.append(item)
            new_priorities[i] = priority
            new_timestamps[i] = self.timestamps[idx]
            new_fingerprints.add(fp)
            
        # Update buffer
        removed = self.size - new_size
        self.buffer = new_buffer
        self.priorities = new_priorities
        self.timestamps = new_timestamps
        self.fingerprints = new_fingerprints
        self.size = new_size
        self.position = new_size % self.capacity
        
        return removed
    
    def save_to_disk(self) -> str:
        """Save replay buffer to disk for persistence."""
        if not self.agent_id:
            return "No agent ID specified"
            
        # Create agent-specific directory
        agent_dir = os.path.join(self.storage_path, f"{self.agent_id}_memory")
        os.makedirs(agent_dir, exist_ok=True)
        
        # Save main memory buffer (serialize first)
        timestamp = int(time.time())
        filename = os.path.join(agent_dir, f"memory_{timestamp}.json")
        
        try:
            # Prepare data for serialization
            serializable_data = []
            
            for i in range(self.size):
                # Skip entries with numpy arrays or non-serializable items
                try:
                    item = self.buffer[i]
                    # Convert experience to JSON-serializable form
                    serializable_exp = {}
                    for k, v in item.items():
                        if isinstance(v, np.ndarray):
                            serializable_exp[k] = v.tolist()
                        elif isinstance(v, torch.Tensor):
                            serializable_exp[k] = v.cpu().tolist()
                        else:
                            serializable_exp[k] = v
                            
                    # Add metadata
                    serializable_exp['_priority'] = float(self.priorities[i])
                    serializable_exp['_timestamp'] = float(self.timestamps[i])
                    serializable_data.append(serializable_exp)
                except Exception as e:
                    # Skip this item
                    console.print(f"[yellow]⚠ Skipping non-serializable memory item: {e}[/yellow]")
                    continue
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump({
                    'agent_id': self.agent_id,
                    'timestamp': timestamp,
                    'size': len(serializable_data),
                    'stats': {
                        'total_adds': self.total_adds,
                        'total_samples': self.total_samples,
                        'dupes_filtered': self.dupes_filtered,
                        'gpt_tokens': self.gpt_tokens
                    },
                    'memories': serializable_data
                }, f, indent=2)
                
            console.print(f"[green]✓ Saved {len(serializable_data)} memories for {self.agent_id}[/green]")
            return filename
        except Exception as e:
            console.print(f"[red]❌ Error saving replay buffer: {e}[/red]")
            return None
    
    def load_from_disk(self, filename: str = None) -> bool:
        """Load replay buffer from disk."""
        if not self.agent_id and not filename:
            return False
            
        try:
            # Find the latest file if not specified
            if not filename:
                agent_dir = os.path.join(self.storage_path, f"{self.agent_id}_memory")
                if not os.path.exists(agent_dir):
                    return False
                    
                files = [f for f in os.listdir(agent_dir) if f.startswith('memory_') and f.endswith('.json')]
                if not files:
                    return False
                    
                # Get the latest file
                files.sort(reverse=True)
                filename = os.path.join(agent_dir, files[0])
                
            # Load from file
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Extract memories
            memories = data.get('memories', [])
            
            # Reset buffer
            self.buffer = []
            self.priorities = np.zeros((self.capacity,), dtype=np.float32)
            self.timestamps = np.zeros((self.capacity,), dtype=np.float32)
            self.fingerprints = set()
            
            # Add memories
            for item in memories:
                # Extract metadata
                priority = item.pop('_priority', 1.0) if '_priority' in item else 1.0
                timestamp = item.pop('_timestamp', time.time()) if '_timestamp' in item else time.time()
                
                # Add to buffer
                if len(self.buffer) < self.capacity:
                    self.buffer.append(item)
                    self.priorities[len(self.buffer) - 1] = priority
                    self.timestamps[len(self.buffer) - 1] = timestamp
                    self.fingerprints.add(self._get_fingerprint(item))
                    
            # Update size
            self.size = min(len(memories), self.capacity)
            self.position = self.size % self.capacity
            
            # Restore phase buffers
            self._rebuild_phase_buffers()
            
            # Restore stats
            stats = data.get('stats', {})
            self.total_adds = stats.get('total_adds', self.total_adds)
            self.total_samples = stats.get('total_samples', self.total_samples)
            self.dupes_filtered = stats.get('dupes_filtered', self.dupes_filtered)
            self.gpt_tokens = stats.get('gpt_tokens', self.gpt_tokens)
            
            console.print(f"[green]✓ Loaded {self.size} memories for {self.agent_id}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Error loading replay buffer: {e}[/red]")
            return False
    
    def _rebuild_phase_buffers(self) -> None:
        """Rebuild phase-specific buffers from main buffer."""
        # Clear phase buffers
        for phase in self.phase_buffers:
            self.phase_buffers[phase] = []
            self.phase_priorities[phase] = []
            
        # Add items to phase buffers
        for i in range(self.size):
            item = self.buffer[i]
            phase = item.get('phase')
            if phase and phase in self.phase_buffers:
                self.phase_buffers[phase].append(item)
                self.phase_priorities[phase].append(self.priorities[i])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring and debugging."""
        # Calculate phase distribution
        phase_distribution = {phase: len(buffer) for phase, buffer in self.phase_buffers.items()}
        
        # Calculate reward distribution
        rewards = [exp.get('reward', 0) for exp in self.buffer[:self.size]]
        reward_stats = {
            'min': min(rewards) if rewards else 0,
            'max': max(rewards) if rewards else 0,
            'mean': sum(rewards) / len(rewards) if rewards else 0,
            'stddev': np.std(rewards) if rewards else 0
        }
        
        # Calculate priority statistics
        priorities = self.priorities[:self.size]
        priority_stats = {
            'min': float(np.min(priorities)) if self.size > 0 else 0,
            'max': float(np.max(priorities)) if self.size > 0 else 0,
            'mean': float(np.mean(priorities)) if self.size > 0 else 0,
            'stddev': float(np.std(priorities)) if self.size > 0 else 0
        }
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'fill_percentage': (self.size / self.capacity) * 100 if self.capacity > 0 else 0,
            'beta': self.beta,
            'total_adds': self.total_adds,
            'total_samples': self.total_samples,
            'dupes_filtered': self.dupes_filtered,
            'phase_distribution': phase_distribution,
            'reward_stats': reward_stats,
            'priority_stats': priority_stats,
            'gpt_tokens': self.gpt_tokens
        }

    def __len__(self) -> int:
        return self.size
