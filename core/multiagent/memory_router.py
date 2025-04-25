# core/utils/memory_router.py — ARIASKA MemoryRouter v11.5 APEX PRIME
# 🧠 Adaptive GPT Cache | 🌐 Global Insight Engine | 📸 Smart Snapshots | ⚡ Real-Time Optimization

"""
MemoryRouter — Global Memory, GPT Cache, and RedAgent Evolution Log
------------------------------------------------------------------
• Centralizes agent memory, GPT cache, and global insights
• Logs RedAgent evolution steps (intent, command, output, GPT comment, success, etc.)
• Provides deduplication and statistics for RedAgent evolution log
• Supports prioritized replay, memory routing, and meta-learning
"""

import os
import json
import re
import hashlib
import threading
import random
import time
from collections import defaultdict, deque
from datetime import datetime
from rich.console import Console
from typing import Any, Dict, List, Tuple, Optional

console = Console()


class MemoryRouter:
    """
    MemoryRouter: Centralized memory, cache, and evolution log for all agents.
    - Manages agent memories, GPT cache, and global insights
    - Logs RedAgent evolution steps for meta-learning and dashboard
    - Provides deduplication, stats aggregation, and prioritized replay
    - Supports memory routing and context summarization for multi-agent learning
    """
    def __init__(self, agents, max_cache_size=5000):
        self.agents = agents
        self.agent_map = {a.agent_id: a for a in agents}
        self.shared_path = os.path.join("core", "memories", "shared")
        self.gpt_cache_file = os.path.join(self.shared_path, "gpt_cache.json")
        self.global_insights_file = os.path.join(
            self.shared_path, "global_insights.json"
        )
        self.snapshot_meta_file = os.path.join(
            self.shared_path, "snapshots", "metadata.json"
        )
        self.max_cache_size = max_cache_size
        self.caches = {}  # Ensure caches dict is always initialized
        self.memories = {}  # For agent memories
        self.evolution_log_path = os.path.join(self.shared_path, "redagent_evolution", "evolution_log.jsonl")
        os.makedirs(os.path.dirname(self.evolution_log_path), exist_ok=True)
        self.evolution_index = {}  # intent_hash → stats
        
        # Initialize buffers for transition storage and replay
        self.buffers = defaultdict(lambda: deque(maxlen=2000))
        self.priorities = defaultdict(list) 
        self.hash_set = defaultdict(set)
        self.gpt_tokens = defaultdict(int)
        self.lock = threading.Lock()
        self.transition_logs = {}
        
        self._initialize_directories()
        self._initialize_files()

    def get_memory(self, agent_id):
        # Defensive: ensure self.memories exists
        if not hasattr(self, "memories"):
            self.memories = {}
        return self.memories.get(agent_id, {"actions": [], "rewards": {}, "scenarios": []})

    def save_memory(self, agent_id, memory):
        # Save memory for a given agent.
        # Ensure all required fields are present
        for action in memory.get("actions", []):
            if "full_command" not in action or not action.get("full_command"):
                action["full_command"] = action.get("command", "")
            if "template" not in action:
                action["template"] = action.get("command", "")
        self.memories[agent_id] = memory

    def _initialize_directories(self):
        os.makedirs(self.shared_path, exist_ok=True)
        os.makedirs(os.path.join(self.shared_path, "snapshots"), exist_ok=True)

    def _initialize_files(self):
        for path, default in [
            (self.gpt_cache_file, {}),
            (self.global_insights_file, {"actions": []}),
            (self.snapshot_meta_file, {}),
        ]:
            if not os.path.exists(path):
                with open(path, "w") as f:
                    json.dump(default, f, indent=2)

    # ─────────────────────────────────────────────
    # ⚡ Adaptive GPT Cache Management
    # ─────────────────────────────────────────────
    def check_gpt_cache(self, key):
        # Defensive: ensure self.caches exists
        if not hasattr(self, "caches"):
            self.caches = {}
        """
        Check if a GPT response is in cache.
        Return the response if found, None otherwise.
        """
        for agent_id in self.caches:
            cache = self.caches[agent_id]
            if key in cache:
                return cache[key]
        return None

    def store_gpt_response(self, agent_id, key, response):
        # Defensive: ensure self.caches exists
        if not hasattr(self, "caches"):
            self.caches = {}
        if agent_id not in self.caches:
            self.caches[agent_id] = {}
        self.caches[agent_id][key] = response

    def _enforce_cache_limit(self, cache):
        if len(cache) > self.max_cache_size:
            # Remove oldest entries
            for k in list(cache.keys())[:len(cache) - self.max_cache_size]:
                del cache[k]
        return cache

    # ─────────────────────────────────────────────
    # 🌐 Global Knowledge Synchronization
    # ─────────────────────────────────────────────
    def sync_global_insights(self, agents=None, reward_threshold=50):
        # Defensive: ensure self.caches exists
        if not hasattr(self, "caches"):
            self.caches = {}
        if agents is None:
            agents = self.agents
        insights = self._load_json(self.global_insights_file).get("actions", [])
        existing_templates = {a.get("template") for a in insights}

        new_actions = []
        for agent in agents:
            if hasattr(agent, "memory_manager"):
                for a in agent.memory_manager.memory.get("actions", []):
                    if a.get("reward", 0) >= reward_threshold and a.get("template") not in existing_templates:
                        new_actions.append(a)
                        existing_templates.add(a.get("template"))

        if new_actions:
            data = self._load_json(self.global_insights_file)
            data["actions"].extend(new_actions)
            self._save_json(self.global_insights_file, data)
            console.print(f"[blue]🔗 Synced {len(new_actions)} high-reward actions globally.[/blue]")
        else:
            console.print("[dim]No new high-reward actions to sync.[/dim]")

    def load_global_insights(self):
        data = self._load_json(self.global_insights_file)
        return data.get("actions", [])

    # ─────────────────────────────────────────────
    # 📸 Smart Snapshot & Restore System
    # ─────────────────────────────────────────────
    def snapshot_all_memories(self, agents=None):
        # Defensive: ensure self.caches exists
        if not hasattr(self, "caches"):
            self.caches = {}
        if agents is None:
            agents = self.agents
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = os.path.join(self.shared_path, "snapshots", timestamp)
        os.makedirs(snapshot_dir, exist_ok=True)

        meta = self._load_json(self.snapshot_meta_file)
        meta[timestamp] = {
            "agents": [agent.agent_id for agent in agents],
            "created": datetime.now().isoformat(),
        }

        for agent in agents:
            if hasattr(agent, "memory_manager"):
                mem = agent.memory_manager.get_all_memory()
                with open(os.path.join(snapshot_dir, f"{agent.agent_id}_memory.json"), "w") as f:
                    json.dump(mem, f, indent=2)

        self._save_json(self.snapshot_meta_file, meta)
        console.print(f"[magenta]📸 Snapshot saved at {snapshot_dir}[/magenta]")

        # After snapshot, flush in-memory logs to avoid indefinite growth
        for agent in agents:
            if hasattr(agent, "memory_manager"):
                agent.memory_manager.memory["actions"] = []
                agent.memory_manager.memory["rewards"] = {}
                agent.memory_manager.memory["scenarios"] = []

    def restore_snapshot(self, snapshot_name, agents=None):
        if agents is None:
            agents = self.agents
        snapshot_dir = os.path.join(self.shared_path, "snapshots", snapshot_name)
        if not os.path.exists(snapshot_dir):
            console.print(f"[red]❌ Snapshot {snapshot_name} not found.[/red]")
            return
        for agent in agents:
            mem_file = os.path.join(snapshot_dir, f"{agent.agent_id}_memory.json")
            if os.path.exists(mem_file) and hasattr(agent, "memory_manager"):
                with open(mem_file, "r") as f:
                    mem = json.load(f)
                agent.memory_manager.memory = mem.get("memory", {})
                agent.memory_manager.history = mem.get("history", [])
                agent.memory_manager.gpt_cache = mem.get("gpt_cache", {})
        console.print(f"[cyan]🔄 Restored snapshot {snapshot_name} for all agents.[/cyan]")

    # ─────────────────────────────────────────────
    # 🧠 Memory Optimization & GPT-Aware Patching
    # ─────────────────────────────────────────────
    def optimize_memories(self, agents=None, threshold=10):
        """
        Deduplicate and optimize all agent memories using prioritized replay and redundancy detection.
        All replay buffer pruning uses core.logic.redundancy_detector and prioritized sampling.
        """
        if agents is None:
            agents = self.agents
        from core.logic.redundancy_detector import detect_redundancy_batch
        for agent in agents:
            if hasattr(agent, "replay_buffer"):
                # Prune redundancy in replay buffer
                agent.replay_buffer.prune_redundancy(detect_redundancy_batch)
            if hasattr(agent, "memory_manager") and hasattr(agent.memory_manager, "optimize_memory"):
                import asyncio
                asyncio.run(agent.memory_manager.optimize_memory(reward_floor=threshold))
        console.print("[green]♻️ All agent memories deduplicated and prioritized.[/green]")

    def consolidate_gpt_cache(self):
        # Defensive: ensure self.caches exists
        if not hasattr(self, "caches"):
            self.caches = {}
        # Merge all agent GPT caches into the shared cache
        cache = self._load_json(self.gpt_cache_file)
        for agent in self.agents:
            if hasattr(agent, "memory_manager"):
                agent_cache = getattr(agent.memory_manager, "gpt_cache", {})
                for k, v in agent_cache.items():
                    if k not in cache:
                        cache[k] = v
        cache = self._enforce_cache_limit(cache)
        self._save_json(self.gpt_cache_file, cache)
        console.print("[cyan]🧠 Consolidated GPT cache across all agents.[/cyan]")

    # ─────────────────────────────────────────────
    # 🔧 Robust JSON Handlers
    # ─────────────────────────────────────────────
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

    # Optional: inject_action for RedAgent/BlueAgent
    def inject_action(self, agent_id, command, reward, context, parsed):
        # Defensive: ensure self.memories exists
        if not hasattr(self, "memories"):
            self.memories = {}
        """
        Record an agent's action in memory for reinforcement learning.
        """
        if agent_id not in self.memories:
            self.memories[agent_id] = {"actions": [], "rewards": {}, "scenarios": []}

        # Create a standardized action record
        action = {
            "command": command,
            "reward": float(reward),
            "phase": context.get("phase", "unknown"),
            "timestamp": self._get_timestamp(),
            "success": parsed.get("success", False),
            "artifacts": parsed.get("artifacts", []),
            "full_command": command,  # Store the full command
            "template": self._templatize_command(command)  # For pattern learning
        }

        self.memories[agent_id]["actions"].append(action)

        # Update reward tracking
        phase = context.get("phase", "unknown")
        if phase not in self.memories[agent_id]["rewards"]:
            self.memories[agent_id]["rewards"][phase] = []
        self.memories[agent_id]["rewards"][phase].append(float(reward))

        # Keep memory size under control
        memory_limit = getattr(self, "memory_limit", 200)
        if len(self.memories[agent_id]["actions"]) > memory_limit:
            self.memories[agent_id]["actions"] = self.memories[agent_id]["actions"][-memory_limit:]

        # Save periodically
        if len(self.memories[agent_id]["actions"]) % 10 == 0:
            self.save_memory(agent_id)

    def _templatize_command(self, command):
        """Convert command to template for pattern learning"""
        if not command or not isinstance(command, str):
            return "unknown_command_template"
        # Replace specific values with placeholders
        template = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_ADDR', command)
        template = re.sub(r'\b\d+\b', 'NUM', template)
        # Extract the primary command and common flags
        parts = template.split()
        if not parts:
            return "empty_command"
        base_cmd = parts[0]
        flags = [p for p in parts if p.startswith('-')]
        return f"{base_cmd}_{':'.join(flags)}" if flags else base_cmd

    def _get_timestamp(self):
        return datetime.now().isoformat()

    def route_memory(self, source_agent_id, target_agent_id, filter_critical=True):
        """
        Route memory from source to target, prioritizing critical/novel experiences.
        Optionally filter for only high-value or recent actions.
        """
        source_mem = self.get_memory(source_agent_id)
        if not source_mem or "actions" not in source_mem:
            return

        # Smarter filtering: prioritize high-reward, novel, or recent actions
        actions = source_mem["actions"]
        if filter_critical:
            filtered = [
                a for a in actions
                if a.get("reward", 0) > 10 or a.get("is_novel", False)
            ]
            # Limit to last 20 for efficiency
            filtered = filtered[-20:]
        else:
            filtered = actions[-50:]

        # Summarize context using GPT-4o-mini for lightweight context
        context_summary = self._summarize_context(filtered)
        routed = {
            "actions": filtered,
            "context_summary": context_summary,
        }
        self._store_routed_memory(target_agent_id, routed)
        return routed

    def _summarize_context(self, actions):
        """
        Use GPT-4o-mini to summarize the context of routed actions for token efficiency.
        """
        if not actions:
            return ""
        try:
            # Use GPTManager instead of direct subprocess call
            from core.gpt_manager import GPTManager
            gpt_manager = GPTManager()
            
            prompt = (
                "Summarize the following actions for context-aware transfer:\n"
                + json.dumps(actions[-5:], indent=2)
                + "\nRespond in 1-2 sentences."
            )
            
            response = gpt_manager.gpt_request(prompt, 
                                               task_type="analysis", 
                                               agent_id="memory_router",
                                               model="gpt-4o-mini")
            
            return response
        except Exception as e:
            console.print(f"[yellow]⚠ Context summarization error: {e}[/yellow]")
            return "Context summary unavailable."

    def _store_routed_memory(self, target_agent_id, routed):
        """
        Store routed memory for the target agent, merging with existing if needed.
        """
        target_mem = self.get_memory(target_agent_id)
        if not target_mem:
            target_mem = {"actions": []}
        # Merge actions, avoid duplicates by template
        existing_templates = {a.get("template") for a in target_mem.get("actions", [])}
        new_actions = [
            a for a in routed["actions"] if a.get("template") not in existing_templates
        ]
        target_mem["actions"].extend(new_actions)
        # Optionally store context summary
        target_mem["context_summary"] = routed.get("context_summary", "")
        self.save_memory(target_agent_id, target_mem)

    # ─────────────────────────────────────────────
    # 🦾 RedAgent Evolution Log & Stats
    # ─────────────────────────────────────────────
    def log_redagent_evolution(self, intent, command, output, gpt_comment, success, episode, step):
        """
        Log RedAgent's intent, command, output, GPT comment, and success to a shared evolution log (JSONL).
        Deduplicate by (intent, command, output) hash. Aggregate stats for command success rates.
        Args:
            intent (str): The agent's intent or phase
            command (str): The command executed
            output (str): The output/result of the command
            gpt_comment (str): GPT feedback or reasoning
            success (bool): Whether the command succeeded
            episode (int): Episode number
            step (int): Step number within the episode
        """
        log_dir = os.path.join(self.shared_path, "redagent_evolution")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "evolution_log.jsonl")
        import hashlib
        entry = {
            "timestamp": datetime.now().isoformat(),
            "episode": episode,
            "step": step,
            "intent": intent,
            "command": command,
            "output": output,
            "gpt_comment": gpt_comment,
            "success": success,
        }
        # Deduplication by hash
        key = hashlib.sha256((str(intent) + str(command) + str(output)).encode()).hexdigest()
        dedup_set_path = os.path.join(log_dir, "dedup_set.json")
        if os.path.exists(dedup_set_path):
            with open(dedup_set_path, "r") as f:
                dedup_set = set(json.load(f))
        else:
            dedup_set = set()
        if key not in dedup_set:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            dedup_set.add(key)
            with open(dedup_set_path, "w") as f:
                json.dump(list(dedup_set), f)

    def get_redagent_evolution_stats(self, n=200):
        """
        Aggregate stats from the RedAgent evolution log: command success rates, most/least effective commands.
        Returns:
            dict: {"commands": ..., "success_rates": ..., "failures": ...}
        """
        log_dir = os.path.join(self.shared_path, "redagent_evolution")
        log_path = os.path.join(log_dir, "evolution_log.jsonl")
        if not os.path.exists(log_path):
            return {}
        from collections import Counter
        stats = {"commands": {}, "success_rates": {}, "failures": {}}
        commands = []
        successes = Counter()
        failures = Counter()
        with open(log_path, "r") as f:
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
    # 🧬 Transition Logging for RL (NEW IMPLEMENTATION)
    # ─────────────────────────────────────────────
    def _hash_state_action(self, state: Any, action: Any) -> str:
        """Generate a unique hash for state-action pair to prevent duplicates."""
        s = str(state) + str(action)
        return hashlib.sha256(s.encode()).hexdigest()

    def log_transition(self, agent_id: str, state: Any, action: Any, reward: float, next_state: Any, 
                      priority: Optional[float] = None, gpt_tokens: int = 0):
        """
        Log a transition for an agent. Deduplicate by state-action hash.
        Track GPT tokens. Enforce tuple structure and sanitize action.
        
        Args:
            agent_id: The unique identifier for the agent
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The resulting next state
            priority: Optional priority value for prioritized replay (defaults to |reward| + 0.01)
            gpt_tokens: Number of GPT tokens used in this transition
        """
        # Deduplicate by both template and full_command
        h = self._hash_state_action(state, action)
        with self.lock:
            # Deduplicate by hash
            if h in self.hash_set[agent_id]:
                return
                
            # Add to hash set to prevent future duplicates
            self.hash_set[agent_id].add(h)
            
            # Sanitize action for security
            if isinstance(action, str):
                action = action.replace("`", "'")
                
            # Create experience tuple
            exp = (state, action, reward, next_state, gpt_tokens)
            
            # Set priority (defaults to |reward| + small constant)
            prio = priority if priority is not None else abs(reward) + 0.01
            
            # Add to agent's buffer with priority
            self.buffers[agent_id].append((prio, exp))
            self.priorities[agent_id].append(prio)
            
            # Track GPT token usage
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
        """
        Sample a prioritized batch for an agent. Higher priority = higher chance.
        """
        with self.lock:
            buffer = list(self.buffers[agent_id])
            if not buffer:
                return []
                
            # Get priorities for sampling
            priorities = [p for p, _ in buffer]
            total = sum(priorities)
            
            # Handle edge case of empty buffer
            if total == 0:
                idxs = random.sample(range(len(buffer)), min(batch_size, len(buffer)))
            else:
                # Prioritized sampling based on rewards
                probs = [p / total for p in priorities]
                idxs = random.choices(range(len(buffer)), weights=probs, k=min(batch_size, len(buffer)))
                
            # Return sampled experiences
            return [buffer[i][1] for i in idxs]

    def get_gpt_token_usage(self, agent_id: str) -> int:
        """Get total GPT token usage for an agent."""
        return self.gpt_tokens.get(agent_id, 0)

    def clear(self, agent_id: Optional[str] = None):
        """Clear memory for one agent or all agents."""
        with self.lock:
            if agent_id:
                # Clear just one agent's memory
                self.buffers[agent_id].clear()
                self.priorities[agent_id].clear()
                self.hash_set[agent_id].clear()
                self.gpt_tokens[agent_id] = 0
            else:
                # Clear all memory
                self.buffers.clear()
                self.priorities.clear()
                self.hash_set.clear()
                self.gpt_tokens.clear()

    def get_buffer_size(self, agent_id: str) -> int:
        """Get current size of an agent's replay buffer."""
        return len(self.buffers[agent_id])

    def get_transitions(self, agent_id):
        """Get transition logs for an agent."""
        return self.transition_logs.get(agent_id, [])

    def close(self):
        """Close any open resources."""
        pass
        
    # ─────────────────────────────────────────────
    # 🔄 Vector DB Integration (Coming Soon)
    # ─────────────────────────────────────────────
    def add_vector_db_hook(self, *args, **kwargs):
        """Placeholder for future vector DB integration."""
        pass

# ─────────────────────────────────────────────
# 🚀 Diagnostic Mode
# ─────────────────────────────────────────────
def run_diagnostics():
    # Import AgentManager only inside this function to avoid circular imports
    try:
        from core.multiagent.agent_manager import AgentManager
    except ImportError as e:
        console.print(f"[red]Failed to import AgentManager: {e}[/red]")
        return

    manager = AgentManager()
    agents = manager.all_agents()
    router = MemoryRouter(agents)

    # Only call methods that exist in MemoryRouter
    router.sync_global_insights(agents=agents)
    router.snapshot_all_memories(agents=agents)
    router.optimize_memories(agents=agents)
    
    # Test log_transition
    state = {"phase": "recon", "open_ports": [22, 80], "blue_team_alert": 0}
    action = "nmap -sV 10.10.10.10"
    reward = 5.0
    next_state = {"phase": "enumeration", "open_ports": [22, 80, 443], "blue_team_alert": 1}
    router.log_transition("RedAgent", state, action, reward, next_state, gpt_tokens=150)
    
    # Print diagnostics
    console.print(f"[green]✓ Buffer size for RedAgent: {router.get_buffer_size('RedAgent')}[/green]")
    console.print(f"[green]✓ GPT tokens used by RedAgent: {router.get_gpt_token_usage('RedAgent')}[/green]")
    
    # Sample batch test
    batch = router.sample_batch("RedAgent", 1)
    console.print(f"[green]✓ Sample batch test {'passed' if batch else 'had no data'}[/green]")

if __name__ == "__main__":
    run_diagnostics()
