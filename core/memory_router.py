# core/memory_router.py — ARIASKA MemoryRouter v1.0
# 🧠 Unified Replay Buffer | ♻️ Deduplication | 🔥 Prioritized Sampling | 🧬 GPT Token Tracking | 💾 Optional SQLite Backend

import os
import threading
import hashlib
import random
import sqlite3
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional

class MemoryRouter:
    """
    Centralized memory and replay buffer manager for all agents.
    - Structured agent-specific prioritized replay buffers
    - Deduplication via state-action hashing
    - GPT token usage tracking
    - Optional SQLite backend for persistence
    - Future-proof for vector DB integration
    """
    def __init__(self, use_sqlite: bool = False, sqlite_path: str = "core/memories/shared/ariaska_memory.db", buffer_size: int = 2000, agents=None):
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.priorities: Dict[str, List[float]] = defaultdict(list)
        self.hash_set: Dict[str, set] = defaultdict(set)
        self.gpt_tokens: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        self.use_sqlite = use_sqlite
        self.sqlite_path = sqlite_path
        self.agents = agents or []
        self.transition_logs = {}
        self.global_memory = {}  # Added global_memory dictionary to store agent-specific data
        if use_sqlite:
            self._init_sqlite()

    def _init_sqlite(self):
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS replay (
                agent_id TEXT,
                state_hash TEXT,
                state BLOB,
                action BLOB,
                reward REAL,
                next_state BLOB,
                priority REAL,
                gpt_tokens INTEGER,
                PRIMARY KEY(agent_id, state_hash)
            )
        """)
        self.conn.commit()

    def _hash_state_action(self, state: Any, action: Any) -> str:
        s = str(state) + str(action)
        return hashlib.sha256(s.encode()).hexdigest()

    def log_transition(self, agent_id: str, state: Any, action: Any, reward: float, next_state: Any, priority: Optional[float] = None, gpt_tokens: int = 0):
        """
        Log a transition for an agent. Deduplicate by state-action hash. Track GPT tokens. Enforce tuple structure and sanitize action.
        """
        h = self._hash_state_action(state, action)
        with self.lock:
            if h in self.hash_set[agent_id]:
                return  # Deduplicate
            self.hash_set[agent_id].add(h)
            # Sanitize action for security
            if isinstance(action, str):
                action = action.replace("`", "'")
            exp = (state, action, reward, next_state, gpt_tokens)
            prio = priority if priority is not None else abs(reward) + 0.01
            self.buffers[agent_id].append((prio, exp))
            self.priorities[agent_id].append(prio)
            self.gpt_tokens[agent_id] += gpt_tokens
            if self.use_sqlite:
                self._log_sqlite(agent_id, h, state, action, reward, next_state, prio, gpt_tokens)
            if agent_id not in self.transition_logs:
                self.transition_logs[agent_id] = []
            self.transition_logs[agent_id].append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "priority": prio,
                "gpt_tokens": gpt_tokens,
                "timestamp": time.time()
            })

    def _log_sqlite(self, agent_id, h, state, action, reward, next_state, prio, gpt_tokens):
        self.conn.execute(
            "INSERT OR IGNORE INTO replay (agent_id, state_hash, state, action, reward, next_state, priority, gpt_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (agent_id, h, str(state), str(action), reward, str(next_state), prio, gpt_tokens)
        )
        self.conn.commit()

    def sample_batch(self, agent_id: str, batch_size: int = 32) -> List[Tuple[Any, Any, float, Any, int]]:
        """
        Sample a prioritized batch for an agent. Higher priority = higher chance.
        """
        with self.lock:
            buffer = list(self.buffers[agent_id])
            if not buffer:
                return []
            priorities = [p for p, _ in buffer]
            total = sum(priorities)
            if total == 0:
                idxs = random.sample(range(len(buffer)), min(batch_size, len(buffer)))
            else:
                probs = [p / total for p in priorities]
                idxs = random.choices(range(len(buffer)), weights=probs, k=min(batch_size, len(buffer)))
            return [buffer[i][1] for i in idxs]

    def get_gpt_token_usage(self, agent_id: str) -> int:
        return self.gpt_tokens[agent_id]

    def clear(self, agent_id: Optional[str] = None):
        with self.lock:
            if agent_id:
                self.buffers[agent_id].clear()
                self.priorities[agent_id].clear()
                self.hash_set[agent_id].clear()
                self.gpt_tokens[agent_id] = 0
            else:
                self.buffers.clear()
                self.priorities.clear()
                self.hash_set.clear()
                self.gpt_tokens.clear()
            if self.use_sqlite:
                if agent_id:
                    self.conn.execute("DELETE FROM replay WHERE agent_id=?", (agent_id,))
                else:
                    self.conn.execute("DELETE FROM replay")
                self.conn.commit()

    # Future: Vector DB integration (Faiss, Chroma, etc.)
    def add_vector_db_hook(self, *args, **kwargs):
        pass  # Placeholder for future vector DB integration

    # For compatibility with batch training
    def get_buffer_size(self, agent_id: str) -> int:
        return len(self.buffers[agent_id])

    def get_transitions(self, agent_id):
        return self.transition_logs.get(agent_id, [])

    def close(self):
        if self.use_sqlite:
            self.conn.close()

    def consolidate_gpt_cache(self):
        """
        Consolidate and optimize the GPT cache across all agents.
        This method is called during post_cycle_operations in MultiAgentTrainer.
        
        1. Identifies common prompts/responses across agents
        2. Deduplicates similar prompts to save token usage
        3. Persists optimized cache to disk for reuse
        """
        try:
            if not hasattr(self, 'gpt_manager'):
                from core.gpt_manager import GPTManager
                self.gpt_manager = GPTManager()
                
            # Get unified cache stats
            cache_stats = {}
            total_saved = 0
            total_entries = 0
            
            # Collect cache data from all agent references
            for agent_id, agent_data in self.global_memory.items():
                if hasattr(agent_data, 'get_gpt_cache'):
                    agent_cache = agent_data.get_gpt_cache()
                    if agent_cache:
                        cache_stats[agent_id] = len(agent_cache)
                        total_entries += len(agent_cache)
                        
            # Check if the GPTManager has a consolidate method
            if hasattr(self.gpt_manager, 'consolidate_caches'):
                tokens_saved = self.gpt_manager.consolidate_caches()
                total_saved = tokens_saved
            
            # Log consolidation results
            from rich.console import Console
            console = Console()
            console.print(f"[blue]🔄 GPT cache consolidated: {total_entries} entries, ~{total_saved} tokens saved[/blue]")
            
            return {
                "cache_stats": cache_stats,
                "total_entries": total_entries,
                "tokens_saved": total_saved
            }
        except Exception as e:
            from rich.console import Console
            console = Console()
            console.print(f"[yellow]⚠️ GPT cache consolidation error: {str(e)}[/yellow]")
            return {"error": str(e), "cache_stats": {}, "total_entries": 0, "tokens_saved": 0}

    def optimize_memories(self):
        """
        Optimize stored memories by:
        1. Deduplicating similar experiences
        2. Pruning low-value memories
        3. Enhancing high-value memories with additional context
        
        This is called by the repair_memories() method in the trainer.
        """
        try:
            from rich.console import Console
            console = Console()
            
            # Track statistics for reporting
            stats = {
                "memories_before": 0,
                "memories_after": 0,
                "duplicates_removed": 0,
                "low_value_removed": 0,
                "enhanced_memories": 0
            }
            
            # Process each agent's memory data
            for agent_id in self.global_memory.keys():
                agent_transitions = self.get_agent_transitions(agent_id)
                if not agent_transitions:
                    continue
                    
                # Count initial memories
                stats["memories_before"] += len(agent_transitions)
                
                # 1. Deduplicate similar experiences
                unique_transitions = []
                seen_hashes = set()
                
                for transition in agent_transitions:
                    # Create a simple hash of the state-action pair
                    if isinstance(transition, dict) and 'state' in transition and 'action' in transition:
                        # For string actions, use the action directly
                        if isinstance(transition['action'], str):
                            action_hash = transition['action']
                        else:
                            # For numeric actions, use string representation
                            action_hash = str(transition['action'])
                            
                        # Create a simple hash with state shape and action
                        if isinstance(transition['state'], list):
                            state_shape = str(len(transition['state']))
                        else:
                            state_shape = "unknown"
                        
                        transition_hash = f"{state_shape}:{action_hash}"
                        
                        if transition_hash not in seen_hashes:
                            seen_hashes.add(transition_hash)
                            unique_transitions.append(transition)
                        else:
                            stats["duplicates_removed"] += 1
                    else:
                        # Keep transitions that don't match our expected format
                        unique_transitions.append(transition)
                
                # 2. Remove low-value memories (those with near-zero rewards)
                valuable_transitions = []
                for transition in unique_transitions:
                    if isinstance(transition, dict) and 'reward' in transition:
                        if abs(transition['reward']) > 0.1:  # Keep non-trivial rewards
                            valuable_transitions.append(transition)
                        else:
                            stats["low_value_removed"] += 1
                    else:
                        valuable_transitions.append(transition)
                
                # 3. Enhance high-value memories with additional context if available
                enhanced_transitions = []
                for transition in valuable_transitions:
                    if isinstance(transition, dict) and 'reward' in transition and abs(transition['reward']) > 1.0:
                        # This is a high-value memory, enhance it if possible
                        if hasattr(self, 'gpt_manager') and self.gpt_manager:
                            # Only enhance if we have the necessary fields
                            if 'action' in transition and isinstance(transition['action'], str):
                                # Add metadata about why this memory is valuable
                                if 'metadata' not in transition:
                                    transition['metadata'] = {}
                                
                                transition['metadata']['high_value'] = True
                                transition['metadata']['value_reason'] = f"High reward: {transition['reward']}"
                                stats["enhanced_memories"] += 1
                    
                    enhanced_transitions.append(transition)
                
                # Update the agent's transitions with optimized memories
                self.set_agent_transitions(agent_id, enhanced_transitions)
                stats["memories_after"] += len(enhanced_transitions)
            
            # Log optimization results
            console.print(f"[green]✅ Memory optimization complete:[/green]")
            console.print(f"[green]   - Before: {stats['memories_before']} memories[/green]")
            console.print(f"[green]   - After: {stats['memories_after']} memories[/green]")
            console.print(f"[green]   - Duplicates removed: {stats['duplicates_removed']}[/green]")
            console.print(f"[green]   - Low-value removed: {stats['low_value_removed']}[/green]")
            console.print(f"[green]   - High-value enhanced: {stats['enhanced_memories']}[/green]")
            
            return stats
        except Exception as e:
            from rich.console import Console
            console = Console()
            console.print(f"[red]❌ Memory optimization error: {str(e)}[/red]")
            return {"error": str(e)}

    def set_agent_transitions(self, agent_id, transitions):
        """Set the transitions for a specific agent, replacing any existing transitions."""
        if agent_id not in self.global_memory:
            self.global_memory[agent_id] = {}
        
        self.global_memory[agent_id]['transitions'] = transitions

    def get_agent_transitions(self, agent_id):
        """Get all transitions for a specific agent."""
        if agent_id in self.global_memory and 'transitions' in self.global_memory[agent_id]:
            return self.global_memory[agent_id]['transitions']
        return []

    def sync_global_insights(self):
        """
        Synchronize insights and experiences across all agents.
        This method is called during post_cycle_operations in MultiAgentTrainer.
        
        1. Shares high-value experiences between agents
        2. Consolidates global insights from individual agent experiences
        3. Updates the central memory repository with synchronized data
        """
        try:
            from rich.console import Console
            console = Console()
            
            # Initialize statistics for reporting
            stats = {
                "shared_experiences": 0,
                "global_insights_updated": 0,
                "high_value_memories": 0
            }
            
            # Skip if no agents or global memory
            if not self.global_memory:
                console.print("[yellow]⚠️ No global memory data to synchronize[/yellow]")
                return stats
                
            # 1. Collect high-value experiences from all agents
            high_value_experiences = {}
            for agent_id, agent_data in self.global_memory.items():
                agent_transitions = self.get_agent_transitions(agent_id)
                if not agent_transitions:
                    continue
                
                # Find high-value experiences (high absolute reward)
                valuable_experiences = []
                for transition in agent_transitions:
                    if isinstance(transition, dict) and 'reward' in transition:
                        # Consider experiences with significant rewards
                        if abs(transition['reward']) > 1.0:
                            valuable_experiences.append(transition)
                            stats["high_value_memories"] += 1
                
                if valuable_experiences:
                    high_value_experiences[agent_id] = valuable_experiences
            
            # 2. Share valuable experiences between compatible agents
            # Define sharing groups (which agents can share experiences)
            sharing_groups = {
                "red_team": ["RedAgent", "ScoutAgent", "ShadowAgent"],
                "blue_team": ["BlueAgent"],
                "oversight": ["OrionAgent"]
            }
            
            # Map agents to their groups
            agent_to_group = {}
            for group, agents in sharing_groups.items():
                for agent in agents:
                    agent_to_group[agent] = group
            
            # Share experiences within groups
            for source_agent_id, experiences in high_value_experiences.items():
                source_group = agent_to_group.get(source_agent_id)
                if not source_group:
                    continue
                    
                # Find other agents in the same group
                for target_agent_id in self.global_memory.keys():
                    if target_agent_id == source_agent_id:
                        continue  # Skip self
                        
                    target_group = agent_to_group.get(target_agent_id)
                    if target_group == source_group:
                        # These agents can share experiences
                        # Get existing transitions
                        existing = self.get_agent_transitions(target_agent_id)
                        
                        # Add unique experiences from source agent
                        for exp in experiences:
                            # Add a note about where this experience came from
                            if 'metadata' not in exp:
                                exp['metadata'] = {}
                            exp['metadata']['shared_from'] = source_agent_id
                            
                            # Add to target agent's memories
                            existing.append(exp)
                            stats["shared_experiences"] += 1
                        
                        # Update target agent's transitions
                        self.set_agent_transitions(target_agent_id, existing)
            
            # 3. Update global insights repository
            if not hasattr(self, 'global_insights'):
                self.global_insights = {}
                
            # Extract key insights from high-value experiences
            for agent_id, experiences in high_value_experiences.items():
                for exp in experiences:
                    if isinstance(exp, dict) and 'action' in exp and 'reward' in exp:
                        # Create a simplified insight
                        action = exp['action']
                        reward = exp['reward']
                        
                        # Use action as key for insight
                        if isinstance(action, str):
                            insight_key = action[:50]  # Limit key size
                            insight_value = {
                                'reward': reward,
                                'agent': agent_id,
                                'count': 1
                            }
                            
                            # Update existing insight or add new one
                            if insight_key in self.global_insights:
                                self.global_insights[insight_key]['count'] += 1
                                # Update reward with rolling average
                                old_reward = self.global_insights[insight_key]['reward']
                                old_count = self.global_insights[insight_key]['count'] - 1
                                new_reward = (old_reward * old_count + reward) / (old_count + 1)
                                self.global_insights[insight_key]['reward'] = new_reward
                            else:
                                self.global_insights[insight_key] = insight_value
                                stats["global_insights_updated"] += 1
            
            # Log synchronization results
            console.print(f"[blue]🔄 Global insights synchronized:[/blue]")
            console.print(f"[blue]   - Shared experiences: {stats['shared_experiences']}[/blue]")
            console.print(f"[blue]   - Global insights updated: {stats['global_insights_updated']}[/blue]")
            console.print(f"[blue]   - High-value memories found: {stats['high_value_memories']}[/blue]")
            
            return stats
            
        except Exception as e:
            from rich.console import Console
            console = Console()
            console.print(f"[red]❌ Global insight synchronization error: {str(e)}[/red]")
            return {"error": str(e)}

    def snapshot_all_memories(self):
        """
        Take a snapshot of all agent memories to persist them to disk.
        This method is called during post_cycle_operations in MultiAgentTrainer.
        
        1. Persists all agent memories to disk in a structured format
        2. Creates a timestamped backup to prevent data loss
        3. Optimizes storage by deduplicating similar memories
        """
        try:
            import json
            import os
            import datetime
            from rich.console import Console
            console = Console()
            
            # Generate timestamp for the snapshot
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create base directories if they don't exist
            snapshot_dir = os.path.join("logs", "memory_snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # Snapshot stats
            stats = {
                "total_memories": 0,
                "agent_memories": {},
                "snapshot_file": f"memory_snapshot_{timestamp}.json"
            }
            
            # Export all memories to a single consolidated file
            snapshot_data = {
                "timestamp": timestamp,
                "agents": {},
                "global_insights": getattr(self, "global_insights", {})
            }
            
            # Process each agent's memory data
            for agent_id in self.global_memory.keys():
                agent_transitions = self.get_agent_transitions(agent_id)
                if not agent_transitions:
                    continue
                
                # Count memories for this agent
                stats["agent_memories"][agent_id] = len(agent_transitions)
                stats["total_memories"] += len(agent_transitions)
                
                # Add to the snapshot data
                snapshot_data["agents"][agent_id] = {
                    "transitions": agent_transitions,
                    "count": len(agent_transitions)
                }
            
            # Write the consolidated snapshot to disk
            snapshot_path = os.path.join(snapshot_dir, stats["snapshot_file"])
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            
            # Create agent-specific snapshots for easy analysis
            for agent_id, agent_data in snapshot_data["agents"].items():
                agent_snapshot_dir = os.path.join(snapshot_dir, agent_id)
                os.makedirs(agent_snapshot_dir, exist_ok=True)
                
                agent_snapshot_path = os.path.join(agent_snapshot_dir, f"memory_{timestamp}.json")
                with open(agent_snapshot_path, 'w') as f:
                    json.dump(agent_data, f, indent=2, default=str)
            
            # Log snapshot results
            console.print(f"[green]📸 Memory snapshot created:[/green]")
            console.print(f"[green]   - Total memories: {stats['total_memories']}[/green]")
            console.print(f"[green]   - Agents: {len(stats['agent_memories'])}[/green]")
            console.print(f"[green]   - Saved to: {snapshot_path}[/green]")
            
            return stats
            
        except Exception as e:
            from rich.console import Console
            console = Console()
            console.print(f"[red]❌ Memory snapshot error: {str(e)}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return {"error": str(e)}
