#!/usr/bin/env python3
"""
Enhanced Memory Router for ARIASKA_RL Training System
🧠 Advanced Memory Management | 🔄 Multi-Agent Coordination | 📊 Persistent Storage
"""

import os
import json
import time
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Define our own Transition class to avoid import conflicts
@dataclass
class Transition:
    """Enhanced transition data structure for memory storage."""
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

logger = logging.getLogger(__name__)

class EnhancedMemoryRouter:
    """
    Enhanced Memory Router with persistent storage and advanced coordination features.
    Extends the base MemoryRouter with training-specific functionality.
    """
    
    def __init__(self, persistence_path: Optional[Union[str, Path]] = None, capacity: int = 10000):
        """
        Initialize enhanced memory router with persistent storage.
        
        Args:
            persistence_path: Path to SQLite database file for persistence
            capacity: Maximum number of transitions to keep in memory
        """
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.capacity = capacity
        
        # Initialize base memory router if available
        try:
            from core.multiagent.memory_router import MemoryRouter
            self.base_router = MemoryRouter()
        except ImportError:
            self.base_router = None
            logger.warning("Base MemoryRouter not available, using fallback implementation")
        
        # Memory structures
        self.transitions: deque = deque(maxlen=capacity)
        self.agent_memories: Dict[str, List[Dict]] = defaultdict(list)
        self.episode_data: List[Dict] = []
        
        # Setup persistent storage
        if self.persistence_path:
            self._setup_database()
    
    def _setup_database(self) -> None:
        """Setup SQLite database for persistent storage."""
        try:
            if self.persistence_path:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
                with sqlite3.connect(str(self.persistence_path)) as conn:
                    cursor = conn.cursor()
                
                # Create transitions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transitions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT NOT NULL,
                        state TEXT NOT NULL,
                        action TEXT NOT NULL,
                        reward REAL NOT NULL,
                        next_state TEXT NOT NULL,
                        done BOOLEAN NOT NULL,
                        gpt_tokens INTEGER DEFAULT 0,
                        priority REAL DEFAULT 1.0,
                        timestamp REAL NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        phase TEXT DEFAULT 'unknown'
                    )
                ''')
                
                # Create episodes table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS episodes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        episode_id TEXT NOT NULL,
                        data TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
    
    def store_transition(self, transition: Transition) -> None:
        """Store a transition in memory and persistent storage."""
        # Store in memory
        self.transitions.append(transition)
        
        # Store by agent
        agent_data = {
            'state': transition.state,
            'action': transition.action,
            'reward': transition.reward,
            'timestamp': transition.timestamp,
            'phase': transition.phase
        }
        self.agent_memories[transition.agent_id].append(agent_data)
        
        # Delegate to base router if available
        if self.base_router:
            try:
                # Try to call store_transition if method exists
                store_method = getattr(self.base_router, 'store_transition', None)
                if store_method:
                    store_method(transition)
            except Exception as e:
                logger.warning(f"Base router storage failed: {e}")
        
        # Store in database if available
        if self.persistence_path:
            self._store_transition_db(transition)
    
    def _store_transition_db(self, transition: Transition) -> None:
        """Store transition in SQLite database."""
        try:
            with sqlite3.connect(str(self.persistence_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO transitions 
                    (agent_id, state, action, reward, next_state, done, gpt_tokens, 
                     priority, timestamp, metadata, phase)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transition.agent_id,
                    json.dumps(transition.state),
                    str(transition.action),
                    transition.reward,
                    json.dumps(transition.next_state),
                    transition.done,
                    transition.gpt_tokens,
                    transition.priority,
                    transition.timestamp,
                    json.dumps(transition.metadata),
                    transition.phase
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store transition in database: {e}")
    
    def get_agent_memory(self, agent_id: str, limit: int = 100) -> List[Dict]:
        """Get recent memory for a specific agent."""
        if agent_id in self.agent_memories:
            return list(self.agent_memories[agent_id])[-limit:]
        return []
    
    def get_recent_transitions(self, limit: int = 50) -> List[Transition]:
        """Get recent transitions across all agents."""
        return list(self.transitions)[-limit:]
    
    def store_episode(self, episode_id: str, episode_data: Dict[str, Any]) -> None:
        """Store episode data."""
        episode_entry = {
            'episode_id': episode_id,
            'data': episode_data,
            'timestamp': time.time()
        }
        
        self.episode_data.append(episode_entry)
        
        # Store in database
        if self.persistence_path:
            try:
                with sqlite3.connect(str(self.persistence_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO episodes (episode_id, data, timestamp)
                        VALUES (?, ?, ?)
                    ''', (episode_id, json.dumps(episode_data), time.time()))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to store episode: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory router statistics."""
        stats = {
            'total_transitions': len(self.transitions),
            'agents': list(self.agent_memories.keys()),
            'episodes': len(self.episode_data),
            'database_enabled': self.persistence_path is not None
        }
        
        # Add agent-specific stats
        for agent_id in self.agent_memories:
            stats[f'{agent_id}_memories'] = len(self.agent_memories[agent_id])
        
        return stats
    
    def clear_memory(self, agent_id: Optional[str] = None) -> None:
        """Clear memory for specific agent or all agents."""
        if agent_id:
            if agent_id in self.agent_memories:
                self.agent_memories[agent_id].clear()
        else:
            self.transitions.clear()
            self.agent_memories.clear()
            self.episode_data.clear()
    
    def sample_batch(self, batch_size: int = 32) -> List[Transition]:
        """Sample a batch of transitions for training."""
        if len(self.transitions) < batch_size:
            return list(self.transitions)
        
        # Simple random sampling for now
        import random
        return random.sample(list(self.transitions), batch_size)
    
    def update_priority(self, transition_id: str, priority: float) -> None:
        """Update priority of a stored transition."""
        # This is a simplified implementation
        # In a full system, you'd use the transition_id to locate and update
        logger.debug(f"Priority update requested for {transition_id}: {priority}")
    
    def sync_with_agents(self, agents: Dict[str, Any]) -> None:
        """Synchronize memory with active agents."""
        for agent_id, agent in agents.items():
            if hasattr(agent, 'sync_memory'):
                try:
                    recent_memories = self.get_agent_memory(agent_id)
                    agent.sync_memory(recent_memories)
                except Exception as e:
                    logger.warning(f"Failed to sync memory with {agent_id}: {e}")
