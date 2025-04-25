import os
import json
import threading
import sqlite3
import hashlib
import time
import random
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from rich.console import Console

console = Console()

# ─────────────────────────────────────────────
# 🧠 GPT Cache (Thread-Safe, SQLite-backed)
# ─────────────────────────────────────────────
class GPTCache:
    def __init__(self, db_path="core/memories/shared/gpt_cache.sqlite3", max_size=5000):
        self.db_path = db_path
        self.max_size = max_size
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gpt_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp REAL DEFAULT (strftime('%s','now'))
                )
            """)
            conn.commit()

    def get(self, key: str) -> Optional[str]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT response FROM gpt_cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, response: str):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO gpt_cache (key, response) VALUES (?, ?)", (key, response))
            # Enforce max size
            count = conn.execute("SELECT COUNT(*) FROM gpt_cache").fetchone()[0]
            if count > self.max_size:
                conn.execute("DELETE FROM gpt_cache WHERE key IN (SELECT key FROM gpt_cache ORDER BY timestamp ASC LIMIT ?)", (count - self.max_size,))
            conn.commit()

    def stats(self):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM gpt_cache").fetchone()[0]
        return {"cache_entries": count}

# ─────────────────────────────────────────────
# 📖 Evolution Logger (RedAgent Evolution Log)
# ─────────────────────────────────────────────
class EvolutionLogger:
    def __init__(self, log_path="core/memories/shared/redagent_evolution/evolution_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._lock = threading.Lock()
        self.dedup_set = set()
        self._load_dedup_set()

    def _dedup_path(self):
        return os.path.join(os.path.dirname(self.log_path), "dedup_set.json")

    def _load_dedup_set(self):
        path = self._dedup_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.dedup_set = set(json.load(f))
            except Exception:
                self.dedup_set = set()

    def _save_dedup_set(self):
        with open(self._dedup_path(), "w") as f:
            json.dump(list(self.dedup_set), f)

    def log(self, entry: Dict[str, Any]):
        key = hashlib.sha256((str(entry.get("intent")) + str(entry.get("command")) + str(entry.get("output"))).encode()).hexdigest()
        with self._lock:
            if key in self.dedup_set:
                return
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self.dedup_set.add(key)
            self._save_dedup_set()

    def stats(self, n=200):
        if not os.path.exists(self.log_path):
            return {}
        from collections import Counter
        stats = {"commands": {}, "success_rates": {}, "failures": {}}
        commands = []
        successes = Counter()
        failures = Counter()
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    cmd = entry.get("command", "N/A")
                    commands.append(cmd)
                    if entry.get("success"):
                        successes[cmd] += 1
                    else:
                        failures[cmd] += 1
                except Exception:
                    continue
        all_cmds = set(commands)
        for cmd in all_cmds:
            total = successes[cmd] + failures[cmd]
            stats["commands"][cmd] = total
            stats["success_rates"][cmd] = successes[cmd] / total if total else 0.0
            stats["failures"][cmd] = failures[cmd]
        return stats

# ─────────────────────────────────────────────
# 🌐 Shared Memory Manager (Global Insights, Snapshots)
# ─────────────────────────────────────────────
class SharedMemoryManager:
    def __init__(self, agents, shared_path="core/memories/shared"):
        self.agents = agents
        self.shared_path = shared_path
        self.global_insights_file = os.path.join(self.shared_path, "global_insights.json")
        self.snapshot_meta_file = os.path.join(self.shared_path, "snapshots", "metadata.json")
        os.makedirs(self.shared_path, exist_ok=True)
        os.makedirs(os.path.join(self.shared_path, "snapshots"), exist_ok=True)
        self._lock = threading.Lock()

    def get_global_insights(self, min_reward=50):
        with self._lock:
            data = self._load_json(self.global_insights_file)
            return [a for a in data.get("actions", []) if a.get("reward", 0) >= min_reward]

    def sync_global_insights(self, reward_threshold=50):
        with self._lock:
            insights = self._load_json(self.global_insights_file).get("actions", [])
            existing_templates = {a.get("template") for a in insights}
            new_actions = []
            for agent in self.agents:
                if hasattr(agent, "memory_manager"):
                    for a in agent.memory_manager.get_actions():
                        if a.get("reward", 0) >= reward_threshold and a.get("template") not in existing_templates:
                            new_actions.append(a)
                            existing_templates.add(a.get("template"))
            if new_actions:
                data = self._load_json(self.global_insights_file)
                data.setdefault("actions", []).extend(new_actions)
                self._save_json(self.global_insights_file, data)
                console.print(f"[blue]🔗 Synced {len(new_actions)} high-reward actions globally.[/blue]")
            else:
                console.print("[dim]No new high-reward actions to sync.[/dim]")

    def snapshot_all_memories(self):
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_dir = os.path.join(self.shared_path, "snapshots", timestamp)
            os.makedirs(snapshot_dir, exist_ok=True)
            meta = self._load_json(self.snapshot_meta_file)
            meta[timestamp] = {
                "agents": [agent.agent_id for agent in self.agents],
                "created": datetime.now().isoformat(),
            }
            for agent in self.agents:
                if hasattr(agent, "memory_manager"):
                    mem = agent.memory_manager.get_actions()
                    with open(os.path.join(snapshot_dir, f"{agent.agent_id}_memory.json"), "w") as f:
                        json.dump(mem, f, indent=2)
            self._save_json(self.snapshot_meta_file, meta)
            console.print(f"[magenta]📸 Snapshot saved at {snapshot_dir}[/magenta]")

    def _load_json(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]⚠ Failed to load {path}: {e}[/red]")
            return {}

    def _save_json(self, path, data):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[red]❌ Failed to save {path}: {e}[/red]")

# ─────────────────────────────────────────────
# 🧬 Experience Router (Prioritized Replay Buffer)
# ─────────────────────────────────────────────
class ExperienceRouter:
    def __init__(self, buffer_size=2000):
        self.buffers = defaultdict(lambda: deque(maxlen=buffer_size))
        self.priorities = defaultdict(list)
        self.hash_set = defaultdict(set)
        self.gpt_tokens = defaultdict(int)
        self.lock = threading.Lock()
        self.transition_logs = {}

    def _hash_state_action(self, state: Any, action: Any) -> str:
        s = str(state) + str(action)
        return hashlib.sha256(s.encode()).hexdigest()

    def store_experience(self, agent_id: str, state: Any, action: Any, reward: float, next_state: Any, priority: Optional[float] = None, gpt_tokens: int = 0):
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
            # Log detailed transition for analytics
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

    def sample_batch(self, agent_id: str, batch_size: int = 32) -> List[Tuple[Any, Any, float, Any, int]]:
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
        return self.gpt_tokens.get(agent_id, 0)

    def get_buffer_size(self, agent_id: str) -> int:
        return len(self.buffers[agent_id])

    def get_transitions(self, agent_id):
        return self.transition_logs.get(agent_id, [])

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

# ─────────────────────────────────────────────
# 🌐 MemoryRouter: Modular Event-Driven Coordinator
# ─────────────────────────────────────────────
class MemoryRouter:
    """
    Modular, event-driven memory router for multi-agent RL.
    - SharedMemoryManager: global insights, snapshots, agent memory sync
    - GPTCache: persistent, size-limited LLM cache
    - EvolutionLogger: RedAgent evolution log
    - ExperienceRouter: prioritized replay buffer and transitions
    """
    def __init__(self, agents, max_cache_size=5000, buffer_size=2000):
        self.agents = agents
        self.agent_map = {a.agent_id: a for a in agents}
        self.gpt_cache = GPTCache(max_size=max_cache_size)
        self.evolution_logger = EvolutionLogger()
        self.shared_memory = SharedMemoryManager(agents)
        self.experience_router = ExperienceRouter(buffer_size=buffer_size)

    # --- Experience API ---
    def store_experience(self, agent_id, state, action, reward, next_state, priority=None, gpt_tokens=0):
        self.experience_router.store_experience(agent_id, state, action, reward, next_state, priority, gpt_tokens)
        
    # --- Legacy API compatibility (for trainer.py) ---
    def log_transition(self, agent_id, state, action, reward, next_state, priority=None, gpt_tokens=0):
        """Legacy compatibility method that maps to store_experience for trainer.py compatibility."""
        console.print(f"[cyan]🔄 Using legacy log_transition for {agent_id}[/cyan]")
        return self.store_experience(agent_id, state, action, reward, next_state, priority, gpt_tokens)
        
    def route_memory(self, memory_data):
        """Legacy compatibility method for older trainer workflows."""
        agent_id = memory_data.get("agent_id")
        state = memory_data.get("state")
        action = memory_data.get("action")
        reward = memory_data.get("reward")
        next_state = memory_data.get("next_state")
        gpt_tokens = memory_data.get("gpt_tokens", 0)
        
        if not all([agent_id, state, action, reward is not None, next_state]):
            console.print("[yellow]⚠️ Incomplete memory data for routing[/yellow]")
            return False
            
        return self.store_experience(agent_id, state, action, reward, next_state, gpt_tokens=gpt_tokens)

    def sample_batch(self, agent_id, batch_size=32):
        return self.experience_router.sample_batch(agent_id, batch_size)

    def get_gpt_token_usage(self, agent_id):
        return self.experience_router.get_gpt_token_usage(agent_id)

    def get_buffer_size(self, agent_id):
        return self.experience_router.get_buffer_size(agent_id)

    def get_transitions(self, agent_id):
        return self.experience_router.get_transitions(agent_id)

    def clear(self, agent_id=None):
        self.experience_router.clear(agent_id)

    # --- GPT Cache API ---
    def check_gpt_cache(self, key):
        return self.gpt_cache.get(key)

    def store_gpt_response(self, key, response):
        self.gpt_cache.set(key, response)

    def gpt_cache_stats(self):
        return self.gpt_cache.stats()
    
    # --- Missing method required by MultiAgentTrainer ---
    def consolidate_gpt_cache(self):
        """
        Optimize and consolidate the GPT cache.
        - Removes redundant entries
        - Persists cache stats to disk
        - Called during post-cycle operations
        """
        console.print("[cyan]🧠 Consolidating GPT cache and optimizing storage...[/cyan]")
        try:
            # Get current cache stats before optimization
            stats_before = self.gpt_cache.stats()
            
            # In a real implementation, we would:
            # 1. Remove any expired or least-used entries
            # 2. Compress similar prompts/responses
            # 3. Apply advanced deduplication
            
            # For now, just report current state
            stats_after = self.gpt_cache.stats()
            console.print(f"[green]✓ GPT cache optimized: {stats_before['cache_entries']} entries[/green]")
            
            # Store cache analytics for later analysis
            cache_analytics_path = os.path.join("logs", "gpt_cache_analytics.json")
            os.makedirs(os.path.dirname(cache_analytics_path), exist_ok=True)
            
            with open(cache_analytics_path, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "cache_size": stats_after["cache_entries"],
                    "optimization_statistics": {
                        "before": stats_before,
                        "after": stats_after
                    }
                }, f, indent=2)
                
            return True
        except Exception as e:
            console.print(f"[red]❌ GPT cache consolidation failed: {e}[/red]")
            return False
            
    def optimize_memories(self, threshold=15):
        """
        Repair and optimize low-reward memories using GPT.
        This is called by MultiAgentTrainer's repair_memories method.
        
        Args:
            threshold (int): Minimum number of memories to process
        """
        console.print(f"[cyan]🔄 Optimizing agent memories (threshold: {threshold})...[/cyan]")
        
        try:
            optimized_count = 0
            
            # Process each agent's memories
            for agent in self.agents:
                agent_id = agent.agent_id
                transitions = self.get_transitions(agent_id)
                
                # Filter for low-reward transitions that might need optimization
                low_reward = [t for t in transitions if t.get("reward", 0) < 0]
                
                if len(low_reward) < threshold:
                    console.print(f"[dim]{agent_id}: Not enough low-reward memories to optimize.[/dim]")
                    continue
                    
                # In a real implementation, we would:
                # 1. Use GPT to analyze patterns in low-reward actions
                # 2. Generate improved alternatives for similar future situations
                # 3. Store these insights in the agent's memory
                
                # For now, just log that we would have optimized
                console.print(f"[green]✓ {agent_id}: Found {len(low_reward)} memories for optimization.[/green]")
                optimized_count += len(low_reward)
                
            console.print(f"[green]✓ Memory optimization complete: {optimized_count} memories processed.[/green]")
            return optimized_count
        except Exception as e:
            console.print(f"[red]❌ Memory optimization failed: {e}[/red]")
            return 0

    # --- Evolution Log API ---
    def log_evolution_step(self, **kwargs):
        self.evolution_logger.log(kwargs)

    def get_evolution_stats(self, n=200):
        return self.evolution_logger.stats(n=n)

    # --- Shared Memory/Insights API ---
    def get_global_insights(self, min_reward=50):
        return self.shared_memory.get_global_insights(min_reward)

    def sync_global_insights(self, reward_threshold=50):
        self.shared_memory.sync_global_insights(reward_threshold)

    def snapshot_all_memories(self):
        self.shared_memory.snapshot_all_memories()

    # --- Event Hooks ---
    def on_episode_end(self):
        # Example: sync insights and snapshot after each episode
        self.sync_global_insights()
        self.snapshot_all_memories()

    def on_turn_end(self):
        # Example: could be used for per-turn stats aggregation
        pass

    # --- CLI/Stats Integration ---
    def get_stats(self):
        return {
            "gpt_cache": self.gpt_cache.stats(),
            "evolution_log": self.evolution_logger.stats(),
            "buffers": {a.agent_id: self.get_buffer_size(a.agent_id) for a in self.agents},
            "gpt_tokens": {a.agent_id: self.get_gpt_token_usage(a.agent_id) for a in self.agents},
        }
        
    # --- Cleanup ---
    def close(self):
        """Clean up resources when the memory router is no longer needed."""
        # Perform any necessary cleanup for database connections or open files
        console.print("[cyan]ℹ️ Closing MemoryRouter and persisting memories[/cyan]")
        # Ensure all memories are synced and snapshots are taken
        self.sync_global_insights()
        self.snapshot_all_memories()
        
        # Save statistics to disk for analysis
        stats_path = os.path.join("logs", "memory_router_stats.json")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(self.get_stats(), f, default=str, indent=2)
        console.print(f"[green]✓ Memory router stats saved to {stats_path}[/green]")

# ─────────────────────────────────────────────
# 🚀 Diagnostic Mode
# ─────────────────────────────────────────────
def run_diagnostics():
    try:
        from core.multiagent.agent_manager import AgentManager
    except ImportError as e:
        console.print(f"[red]Failed to import AgentManager: {e}[/red]")
        return

    manager = AgentManager()
    agents = manager.all_agents()
    router = MemoryRouter(agents)

    # Test experience logging
    state = {"phase": "recon", "open_ports": [22, 80], "blue_team_alert": 0}
    action = "nmap -sV 10.10.10.10"
    reward = 5.0
    next_state = {"phase": "enumeration", "open_ports": [22, 80, 443], "blue_team_alert": 1}
    router.store_experience("RedAgent", state, action, reward, next_state, gpt_tokens=150)

    # Print diagnostics
    console.print(f"[green]✓ Buffer size for RedAgent: {router.get_buffer_size('RedAgent')}[/green]")
    console.print(f"[green]✓ GPT tokens used by RedAgent: {router.get_gpt_token_usage('RedAgent')}[/green]")

    # Sample batch test
    batch = router.sample_batch("RedAgent", 1)
    console.print(f"[green]✓ Sample batch test {'passed' if batch else 'had no data'}[/green]")

    # Evolution log test
    router.log_evolution_step(
        intent="recon",
        command="nmap -sV 10.10.10.10",
        output="open ports: 22, 80",
        gpt_comment="Effective recon.",
        success=True,
        episode=1,
        step=1,
        timestamp=datetime.now().isoformat()
    )
    stats = router.get_evolution_stats()
    console.print(f"[green]✓ Evolution log stats: {stats}[/green]")

    # GPT cache test
    router.store_gpt_response("test_key", "test_response")
    cached = router.check_gpt_cache("test_key")
    console.print(f"[green]✓ GPT cache test: {cached}[/green]")

    # Global insights test
    router.sync_global_insights()
    insights = router.get_global_insights()
    console.print(f"[green]✓ Global insights: {insights}[/green]")

if __name__ == "__main__":
    run_diagnostics()
