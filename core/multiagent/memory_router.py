"""
memory_router.py — Enhanced Memory Router with Prioritized Experience Replay
Implements a prioritized experience memory system for multiple agents with:
- Deduplication
- Priority-based sampling
- Efficient storage of important transitions
- Sum tree for efficient sampling from prioritized distribution
- Support for episode summaries and contextual querying
- Strategic directive logging and retrieval
"""

import os
import json
import time
import uuid
import hashlib
import random
import threading
import sqlite3
import numpy as np
import logging
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from pathlib import Path
from dataclasses import dataclass, field

# Logger setup
logger = logging.getLogger(__name__)

@dataclass
class Transition:
    """Data structure for storing agent transitions with metadata"""
    agent_id: str
    state: Dict[str, Any]
    action: Any
    reward: float
    next_state: Dict[str, Any]
    done: bool = False
    gpt_tokens: int = 0
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    phase: str = "unknown"
    
    def get_significance(self) -> float:
        """
        Calculate significance score for prioritization.
        Higher scores = more important transitions.
        """
        # Base significance from reward (positive or negative)
        score = abs(self.reward)
        
        # Increase priority for terminal states
        if self.done:
            score *= 2.0
            
        # Increase for significant events
        if self.metadata.get('new_discovery', False):
            score *= 1.5
        if self.metadata.get('exploit_success', False):
            score *= 3.0
        if self.metadata.get('detection', False):
            score *= 2.0
        if self.metadata.get('critical_action', False):
            score *= 2.5
            
        # Ensure minimum priority
        return max(score, 0.01)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transition to dictionary for storage"""
        return {
            "agent_id": self.agent_id,
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "gpt_tokens": self.gpt_tokens,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "phase": self.phase
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transition':
        """Create transition from dictionary"""
        return cls(**data)

class SumTree:
    """
    Sum tree data structure for efficient sampling from prioritized distribution.
    Binary tree where leaf nodes contain priorities and internal nodes maintain sums.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Array to store the tree
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.next_idx = 0
    
    def add(self, priority: float, data: Any) -> None:
        """Add a new element with given priority"""
        tree_idx = self.next_idx + self.capacity - 1
        self.data[self.next_idx] = data
        self.update(tree_idx, priority)
        
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, tree_idx: int, priority: float) -> None:
        """Update the priority of a node"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, value: float) -> Tuple[int, float, Any]:
        """Get leaf node based on a value in [0, total_priority]"""
        parent_idx = 0
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            # Reach bottom of the tree
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            # Navigate based on value
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]
                parent_idx = right_idx
        
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def total_priority(self) -> float:
        """Return the total priority (sum at root)"""
        return self.tree[0]

class MemoryRouter:
    """
    Enhanced memory router with prioritized experience replay.
    Stores critical transitions with higher priority and implements efficient sampling.
    Features:
    - Deduplication of redundant experiences
    - Priority-based experience sampling
    - Episode and phase summarization
    - Persistence and snapshots
    - Token usage tracking
    - Thread-safe operations
    - Strategic directive logging
    """
    def __init__(
        self, 
        buffer_size: int = 10000, 
        alpha: float = 0.6, 
        beta_start: float = 0.4, 
        beta_frames: int = 10000,
        persistence_path: str = None,
        enable_sqlite: bool = True
    ):
        self.buffer_size = buffer_size
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Importance sampling exponent start
        self.beta_frames = beta_frames  # Frames over which beta increases
        self.beta = beta_start  # Current beta value
        self.frame = 0  # Current frame for beta annealing
        
        # Agent-specific buffers using sum trees
        self.buffers = {}
        self.hash_set = defaultdict(set)  # For deduplication
        self.gpt_tokens = defaultdict(int)  # Token tracking per agent
        self.episodes = defaultdict(list)  # Episode tracking
        self.episode_summaries = defaultdict(dict)  # Summaries by agent & episode
        self.phase_summaries = defaultdict(dict)  # Summaries by agent & phase
        
        # Strategic directives tracking
        self.directives = []
        self.directive_by_agent = defaultdict(list)
        self.directive_by_source = defaultdict(list)
        self.directive_by_type = defaultdict(list)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Set up persistence - Use the standardized path
        self.persistence_path = persistence_path or os.path.join("core", "memory", "memory_router.db")
        self.enable_sqlite = enable_sqlite
        if enable_sqlite:
            self._init_sqlite()
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database for persistence"""
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        
        # Use connection timeout and proper locking settings
        self.conn = sqlite3.connect(
            self.persistence_path, 
            check_same_thread=False,
            timeout=30.0  # Increase timeout for busy database
        )
        
        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        
        # Create all required tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transitions (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                data TEXT,
                priority REAL,
                timestamp REAL,
                episode_id TEXT,
                phase TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY, 
                agent_id TEXT,
                episode_id TEXT,
                phase TEXT,
                summary TEXT,
                timestamp REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                agent_id TEXT PRIMARY KEY,
                tokens INTEGER,
                timestamp REAL
            )
        """)
        
        # New table for strategic directives
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS directives (
                id TEXT PRIMARY KEY,
                source_agent TEXT,
                target_agent TEXT,
                directive_type TEXT,
                parameters TEXT,
                priority INTEGER,
                status TEXT,
                timestamp REAL,
                step INTEGER,
                episode INTEGER,
                ttl INTEGER
            )
        """)
        
        # New table for action chains
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS action_chains (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                phase TEXT,
                chain TEXT,
                episode INTEGER,
                step INTEGER,
                timestamp REAL
            )
        """)
        
        self.conn.commit()
    
    def _hash_transition(self, agent_id: str, state: Dict[str, Any], action: Any) -> str:
        """Create a hash for deduplication based on state and action"""
        key = f"{agent_id}:{str(state)}:{str(action)}"
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _get_agent_buffer(self, agent_id: str) -> SumTree:
        """Get or create a sum tree buffer for an agent"""
        if agent_id not in self.buffers:
            self.buffers[agent_id] = SumTree(self.buffer_size)
        return self.buffers[agent_id]
    
    def add_transition(
        self,
        agent_id: str,
        state: Dict[str, Any],
        action: Any,
        reward: float,
        next_state: Dict[str, Any],
        done: bool = False,
        gpt_tokens: int = 0,
        metadata: Dict[str, Any] = None,
        phase: str = "unknown",
        episode_id: str = None
    ) -> bool:
        """
        Add a transition to the memory router if it's significant enough.
        
        Args:
            agent_id: ID of the agent
            state: State representation
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether this is a terminal state
            gpt_tokens: Number of GPT tokens used
            metadata: Additional information about the transition
            phase: Current phase (recon, exploit, etc.)
            episode_id: Optional episode identifier
            
        Returns:
            bool: True if added, False if filtered out or duplicate
        """
        metadata = metadata or {}
        episode_id = episode_id or f"ep_{int(time.time())}"
        
        # Create transition object
        transition = Transition(
            agent_id=agent_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            gpt_tokens=gpt_tokens,
            metadata=metadata,
            phase=phase
        )
        
        # Calculate significance/priority
        priority = transition.get_significance()
        transition.priority = priority
        
        # Filter out insignificant transitions (near-zero reward and no metadata flags)
        is_significant = (
            abs(reward) > 0.1 or 
            done or
            any(metadata.get(k, False) for k in 
                ['new_discovery', 'exploit_success', 'detection', 'critical_action'])
        )
        
        if not is_significant:
            return False
        
        # Check for duplicates
        transition_hash = self._hash_transition(agent_id, state, action)
        
        with self.lock:
            # Skip if it's a duplicate
            if transition_hash in self.hash_set[agent_id]:
                return False
            
            # Track the hash to avoid future duplicates
            self.hash_set[agent_id].add(transition_hash)
            
            # Update token usage
            self.gpt_tokens[agent_id] += gpt_tokens
            
            # Add to agent's buffer with priority^alpha
            buffer = self._get_agent_buffer(agent_id)
            buffer.add(priority ** self.alpha, transition)
            
            # Track episode data
            if episode_id:
                self.episodes[agent_id].append({
                    "episode_id": episode_id,
                    "transition_hash": transition_hash,
                    "reward": reward,
                    "phase": phase,
                    "timestamp": transition.timestamp
                })
            
            # Persist to SQLite if enabled
            if self.enable_sqlite:
                transition_id = f"{agent_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                self.conn.execute(
                    "INSERT OR REPLACE INTO transitions (id, agent_id, data, priority, timestamp, episode_id, phase) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        transition_id,
                        agent_id,
                        json.dumps(transition.to_dict()),
                        priority,
                        transition.timestamp,
                        episode_id,
                        phase
                    )
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO token_usage (agent_id, tokens, timestamp) VALUES (?, ?, ?)",
                    (agent_id, self.gpt_tokens[agent_id], time.time())
                )
                self.conn.commit()
        
        # Update beta for importance sampling
        self.frame += 1
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        
        return True
    
    def sample_transitions(
        self, 
        agent_id: str, 
        batch_size: int = 32, 
        uniform: bool = False
    ) -> List[Transition]:
        """
        Sample transitions using prioritized experience replay.
        
        Args:
            agent_id: ID of the agent to sample from
            batch_size: Number of transitions to sample
            uniform: If True, use uniform sampling instead of prioritized
            
        Returns:
            List of sampled transitions
        """
        with self.lock:
            buffer = self._get_agent_buffer(agent_id)
            
            # If buffer is empty or too small
            if buffer.size == 0:
                return []
            
            # Adjust batch_size if buffer is smaller
            actual_batch_size = min(batch_size, buffer.size)
            
            if uniform:
                # Uniform sampling for baseline comparison
                indices = np.random.randint(0, buffer.size, size=actual_batch_size)
                samples = [buffer.data[idx] for idx in indices]
                
                # No weights needed for uniform sampling
                weights = np.ones(len(samples))
                return samples, weights, indices
            
            # Prioritized sampling
            samples = []
            indices = []
            weights = np.zeros(actual_batch_size)
            total_priority = buffer.total_priority()
            
            # Segment tree for prioritized sampling
            segment = total_priority / actual_batch_size
            
            for i in range(actual_batch_size):
                # Get uniform sample within each segment
                a = segment * i
                b = segment * (i + 1)
                value = random.uniform(a, b)
                
                # Retrieve sample from tree
                idx, priority, data = buffer.get_leaf(value)
                indices.append(idx)
                samples.append(data)
                
                # Calculate importance sampling weights
                # P(i) = p_i^α / sum(p_j^α)
                # Weight = (1/N * 1/P(i))^β = (N*P(i))^(-β)
                if priority > 0:
                    prob = priority / total_priority
                    weights[i] = (buffer.size * prob) ** (-self.beta)
                
            # Normalize weights to prevent large gradient updates
            weights = weights / np.max(weights)
            
            return samples, weights, indices
    
    def update_priorities(
        self, 
        agent_id: str, 
        indices: List[int], 
        priorities: List[float]
    ) -> None:
        """
        Update priorities for transitions based on TD errors.
        
        Args:
            agent_id: ID of the agent
            indices: Tree indices of the transitions
            priorities: New priority values
        """
        with self.lock:
            if agent_id not in self.buffers:
                return
                
            buffer = self.buffers[agent_id]
            
            for idx, priority in zip(indices, priorities):
                # Ensure minimum priority and apply alpha exponent
                priority = max(priority, 0.01) ** self.alpha
                buffer.update(idx, priority)
    
    def get_token_usage(self, agent_id: str = None) -> Dict[str, int]:
        """
        Get token usage statistics.
        
        Args:
            agent_id: Optional agent ID filter
            
        Returns:
            Dictionary of token usage by agent
        """
        with self.lock:
            if agent_id:
                return {agent_id: self.gpt_tokens.get(agent_id, 0)}
            return dict(self.gpt_tokens)
    
    def add_summary(
        self, 
        agent_id: str, 
        summary: Dict[str, Any], 
        episode_id: str = None, 
        phase: str = None
    ) -> str:
        """
        Add a summary for an episode or phase.
        
        Args:
            agent_id: ID of the agent
            summary: Summary data
            episode_id: Optional episode identifier
            phase: Optional phase identifier
            
        Returns:
            ID of the stored summary
        """
        summary_id = f"{agent_id}_{episode_id or phase}_{int(time.time())}"
        
        with self.lock:
            if episode_id:
                self.episode_summaries[(agent_id, episode_id)] = summary
            
            if phase:
                self.phase_summaries[(agent_id, phase)] = summary
            
            # Persist to SQLite if enabled
            if self.enable_sqlite:
                self.conn.execute(
                    "INSERT OR REPLACE INTO summaries (id, agent_id, episode_id, phase, summary, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        summary_id,
                        agent_id,
                        episode_id or "",
                        phase or "",
                        json.dumps(summary),
                        time.time()
                    )
                )
                self.conn.commit()
                
        return summary_id
    
    def get_summary(
        self, 
        agent_id: str, 
        episode_id: str = None, 
        phase: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a summary for an episode or phase.
        
        Args:
            agent_id: ID of the agent
            episode_id: Optional episode identifier
            phase: Optional phase identifier
            
        Returns:
            Summary data if found, None otherwise
        """
        with self.lock:
            if episode_id and (agent_id, episode_id) in self.episode_summaries:
                return self.episode_summaries[(agent_id, episode_id)]
                
            if phase and (agent_id, phase) in self.phase_summaries:
                return self.phase_summaries[(agent_id, phase)]
                
            # Try SQLite if enabled and not in memory
            if self.enable_sqlite:
                query_parts = ["agent_id = ?"]
                params = [agent_id]
                
                if episode_id:
                    query_parts.append("episode_id = ?")
                    params.append(episode_id)
                    
                if phase:
                    query_parts.append("phase = ?")
                    params.append(phase)
                    
                where_clause = " AND ".join(query_parts)
                query = f"SELECT summary FROM summaries WHERE {where_clause} ORDER BY timestamp DESC LIMIT 1"
                
                cursor = self.conn.execute(query, params)
                row = cursor.fetchone()
                
                if row:
                    summary = json.loads(row[0])
                    
                    # Cache in memory
                    if episode_id:
                        self.episode_summaries[(agent_id, episode_id)] = summary
                    if phase:
                        self.phase_summaries[(agent_id, phase)] = summary
                        
                    return summary
                    
        return None
    
    def get_recent_transitions(
        self, 
        agent_id: str, 
        limit: int = 10, 
        phase: str = None
    ) -> List[Transition]:
        """
        Get most recent transitions for an agent.
        
        Args:
            agent_id: ID of the agent
            limit: Maximum number of transitions
            phase: Optional phase filter
            
        Returns:
            List of recent transitions
        """
        if not self.enable_sqlite:
            return []
            
        with self.lock:
            query_parts = ["agent_id = ?"]
            params = [agent_id]
            
            if phase:
                query_parts.append("phase = ?")
                params.append(phase)
                
            where_clause = " AND ".join(query_parts)
            query = f"SELECT data FROM transitions WHERE {where_clause} ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            transitions = []
            for row in rows:
                data = json.loads(row[0])
                transitions.append(Transition.from_dict(data))
                
            return transitions
    
    def clear_agent_data(self, agent_id: str) -> None:
        """
        Clear all data for a specific agent.
        
        Args:
            agent_id: ID of the agent
        """
        with self.lock:
            # Clear in-memory data
            if agent_id in self.buffers:
                del self.buffers[agent_id]
            
            if agent_id in self.hash_set:
                del self.hash_set[agent_id]
                
            if agent_id in self.gpt_tokens:
                del self.gpt_tokens[agent_id]
            
            # Clear episode data
            if agent_id in self.episodes:
                del self.episodes[agent_id]
            
            # Clear summaries
            keys_to_remove = []
            for key in self.episode_summaries:
                if key[0] == agent_id:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.episode_summaries[key]
                
            keys_to_remove = []
            for key in self.phase_summaries:
                if key[0] == agent_id:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.phase_summaries[key]
            
            # Clear from SQLite if enabled
            if self.enable_sqlite:
                self.conn.execute("DELETE FROM transitions WHERE agent_id = ?", (agent_id,))
                self.conn.execute("DELETE FROM summaries WHERE agent_id = ?", (agent_id,))
                self.conn.execute("DELETE FROM token_usage WHERE agent_id = ?", (agent_id,))
                self.conn.commit()
    
    def get_stats(self, agent_id: str = None) -> Dict[str, Any]:
        """
        Get statistics about the memory router.
        
        Args:
            agent_id: Optional agent ID filter
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_transitions": 0,
            "agents": {},
            "token_usage": dict(self.gpt_tokens),
            "total_tokens": sum(self.gpt_tokens.values()),
            "buffer_size": self.buffer_size,
            "frame": self.frame,
            "beta": self.beta
        }
        
        with self.lock:
            for a_id, buffer in self.buffers.items():
                if agent_id and a_id != agent_id:
                    continue
                    
                agent_stats = {
                    "transitions": buffer.size,
                    "total_priority": buffer.total_priority(),
                    "tokens": self.gpt_tokens.get(a_id, 0),
                    "hash_keys": len(self.hash_set.get(a_id, set())),
                    "episode_count": len(set(e["episode_id"] for e in self.episodes.get(a_id, [])))
                }
                
                stats["agents"][a_id] = agent_stats
                stats["total_transitions"] += buffer.size
        
        # Add directive statistics
        directive_stats = self.get_directive_stats()
        stats["directives"] = directive_stats
                
        return stats
    
    def snapshot(self, path: str = None) -> str:
        """
        Create a snapshot of the memory state.
        
        Args:
            path: Optional path to save snapshot
            
        Returns:
            Path where snapshot is saved
        """
        if not path:
            os.makedirs("core/memories/snapshots", exist_ok=True)
            path = f"core/memories/snapshots/memory_snapshot_{int(time.time())}.db"
            
        if self.enable_sqlite:
            # Create a copy of the SQLite database
            with self.lock:
                try:
                    if os.path.exists(self.persistence_path):
                        import shutil
                        shutil.copy2(self.persistence_path, path)
                    return path
                except Exception as e:
                    logger.error(f"Failed to create memory snapshot: {e}")
                    return None
        
        return None
    
    def close(self) -> None:
        """Close database connections and clean up resources"""
        if self.enable_sqlite and hasattr(self, 'conn'):
            try:
                self.conn.close()
                logger.info("Successfully closed SQLite connection")
            except Exception as e:
                logger.error(f"Error closing SQLite connection: {e}")
    
    def log_directive(
        self,
        source_agent: str,
        target_agent: str,
        directive_type: str,
        parameters: Dict[str, Any],
        priority: int = 1,
        step: int = 0,
        episode: int = 0,
        ttl: int = 10
    ) -> str:
        """
        Log a strategic directive issued by one agent to another.
        
        Args:
            source_agent: Agent issuing the directive
            target_agent: Agent receiving the directive
            directive_type: Type of directive
            parameters: Additional parameters for the directive
            priority: Priority level (1-5)
            step: Current step number
            episode: Current episode number
            ttl: Time-to-live for the directive
            
        Returns:
            ID of the logged directive
        """
        directive_id = str(uuid.uuid4())
        timestamp = time.time()
        
        directive = {
            "id": directive_id,
            "source_agent": source_agent,
            "target_agent": target_agent,
            "directive_type": directive_type,
            "parameters": parameters,
            "priority": priority,
            "status": "issued",
            "timestamp": timestamp,
            "step": step,
            "episode": episode,
            "ttl": ttl
        }
        
        with self.lock:
            self.directives.append(directive)
            self.directive_by_agent[target_agent].append(directive_id)
            self.directive_by_source[source_agent].append(directive_id)
            self.directive_by_type[directive_type].append(directive_id)
            
            # Persist to SQLite if enabled
            if self.enable_sqlite:
                self.conn.execute(
                    "INSERT INTO directives VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        directive_id,
                        source_agent,
                        target_agent,
                        directive_type,
                        json.dumps(parameters),
                        priority,
                        "issued",
                        timestamp,
                        step,
                        episode,
                        ttl
                    )
                )
                self.conn.commit()
        
        return directive_id
    
    def update_directive_status(
        self,
        directive_id: str,
        status: str,
        result: Dict[str, Any] = None
    ) -> bool:
        """
        Update the status of a strategic directive.
        
        Args:
            directive_id: ID of the directive
            status: New status (processed, rejected, expired)
            result: Optional result from processing the directive
            
        Returns:
            True if directive was found and updated, False otherwise
        """
        with self.lock:
            # Find directive in memory
            for directive in self.directives:
                if directive["id"] == directive_id:
                    directive["status"] = status
                    if result:
                        directive["result"] = result
                    
                    # Update in SQLite if enabled
                    if self.enable_sqlite:
                        self.conn.execute(
                            "UPDATE directives SET status = ? WHERE id = ?",
                            (status, directive_id)
                        )
                        if result:
                            self.conn.execute(
                                "UPDATE directives SET parameters = ? WHERE id = ?",
                                (json.dumps({**directive["parameters"], "result": result}), directive_id)
                            )
                        self.conn.commit()
                    
                    return True
            
            return False
    
    def get_agent_directives(
        self,
        agent_id: str,
        active_only: bool = True,
        as_target: bool = True,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get strategic directives for a specific agent.
        
        Args:
            agent_id: ID of the agent
            active_only: Only return active (non-processed) directives
            as_target: If True, get directives where agent is target; 
                      If False, get directives issued by the agent
            limit: Maximum number of directives to return
            
        Returns:
            List of directives
        """
        with self.lock:
            if not self.enable_sqlite:
                # Use in-memory directives
                directives = []
                for d in self.directives:
                    if as_target and d["target_agent"] == agent_id:
                        if not active_only or d["status"] == "issued":
                            directives.append(d)
                    elif not as_target and d["source_agent"] == agent_id:
                        directives.append(d)
                
                # Sort by priority (desc) then timestamp (desc)
                directives.sort(key=lambda d: (-d["priority"], -d["timestamp"]))
                return directives[:limit]
            
            # Use SQLite
            query_parts = []
            params = []
            
            if as_target:
                query_parts.append("target_agent = ?")
                params.append(agent_id)
                if active_only:
                    query_parts.append("status = ?")
                    params.append("issued")
            else:
                query_parts.append("source_agent = ?")
                params.append(agent_id)
            
            where_clause = " AND ".join(query_parts)
            query = f"""
                SELECT id, source_agent, target_agent, directive_type, parameters,
                       priority, status, timestamp, step, episode, ttl
                FROM directives
                WHERE {where_clause}
                ORDER BY priority DESC, timestamp DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            directives = []
            for row in rows:
                directives.append({
                    "id": row[0],
                    "source_agent": row[1],
                    "target_agent": row[2],
                    "directive_type": row[3],
                    "parameters": json.loads(row[4]),
                    "priority": row[5],
                    "status": row[6],
                    "timestamp": row[7],
                    "step": row[8],
                    "episode": row[9],
                    "ttl": row[10]
                })
            
            return directives
    
    def log_action_chain(
        self,
        agent_id: str,
        phase: str,
        chain: List[str],
        episode: int = 0,
        step: int = 0
    ) -> str:
        """
        Log an action chain generated by an agent (typically OrionAgent).
        
        Args:
            agent_id: ID of the agent generating the chain
            phase: Current phase (recon, exploit, etc.)
            chain: List of ordered actions forming the chain
            episode: Current episode number
            step: Current step number
            
        Returns:
            ID of the logged action chain
        """
        chain_id = str(uuid.uuid4())
        timestamp = time.time()
        
        with self.lock:
            # Persist to SQLite if enabled
            if self.enable_sqlite:
                self.conn.execute(
                    "INSERT INTO action_chains VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        chain_id,
                        agent_id,
                        phase,
                        json.dumps(chain),
                        episode,
                        step,
                        timestamp
                    )
                )
                self.conn.commit()
        
        return chain_id
    
    def get_latest_action_chain(
        self,
        agent_id: str = None,
        phase: str = None
    ) -> Optional[List[str]]:
        """
        Get the most recent action chain.
        
        Args:
            agent_id: Optional agent ID filter
            phase: Optional phase filter
            
        Returns:
            List of actions in the chain, or None if not found
        """
        if not self.enable_sqlite:
            return None
        
        with self.lock:
            query_parts = []
            params = []
            
            if agent_id:
                query_parts.append("agent_id = ?")
                params.append(agent_id)
            
            if phase:
                query_parts.append("phase = ?")
                params.append(phase)
            
            where_clause = " AND ".join(query_parts) if query_parts else "1=1"
            query = f"""
                SELECT chain
                FROM action_chains
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            cursor = self.conn.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return json.loads(row[0])
            
            return None
    
    def get_directive_stats(self) -> Dict[str, Any]:
        """
        Get statistics about strategic directives.
        
        Returns:
            Dictionary with directive statistics
        """
        stats = {
            "total_directives": 0,
            "active_directives": 0,
            "by_status": defaultdict(int),
            "by_agent": defaultdict(int),
            "by_source": defaultdict(int),
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int)
        }
        
        with self.lock:
            if not self.enable_sqlite:
                # Use in-memory directives
                stats["total_directives"] = len(self.directives)
                
                for d in self.directives:
                    if d["status"] == "issued":
                        stats["active_directives"] += 1
                    stats["by_status"][d["status"]] += 1
                    stats["by_agent"][d["target_agent"]] += 1
                    stats["by_source"][d["source_agent"]] += 1
                    stats["by_type"][d["directive_type"]] += 1
                    stats["by_priority"][d["priority"]] += 1
                
                return stats
            
            # Use SQLite
            cursor = self.conn.execute("SELECT COUNT(*) FROM directives")
            stats["total_directives"] = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM directives WHERE status = 'issued'")
            stats["active_directives"] = cursor.fetchone()[0]
            
            for field, table_field in [
                ("status", "status"),
                ("target_agent", "by_agent"),
                ("source_agent", "by_source"),
                ("directive_type", "by_type"),
                ("priority", "by_priority")
            ]:
                cursor = self.conn.execute(f"SELECT {field}, COUNT(*) FROM directives GROUP BY {field}")
                for row in cursor.fetchall():
                    stats[f"by_{table_field}"][row[0]] = row[1]
            
            return stats
