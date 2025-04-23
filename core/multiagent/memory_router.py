# core/utils/memory_router.py — ARIASKA MemoryRouter v11.5 APEX PRIME
# 🧠 Adaptive GPT Cache | 🌐 Global Insight Engine | 📸 Smart Snapshots | ⚡ Real-Time Optimization

import os
import json
import re
from datetime import datetime
from rich.console import Console

console = Console()


class MemoryRouter:
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

        self._initialize_directories()
        self._initialize_files()

    def get_memory(self, agent_id):
        # Defensive: ensure self.memories exists
        if not hasattr(self, "memories"):
            self.memories = {}
        return self.memories.get(agent_id, {"actions": [], "rewards": {}, "scenarios": []})

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
        if agents is None:
            agents = self.agents
        for agent in agents:
            if hasattr(agent, "memory_manager") and hasattr(agent.memory_manager, "optimize_memory"):
                import asyncio
                asyncio.run(agent.memory_manager.optimize_memory(reward_floor=threshold))

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
            return "unknown_command"

        parts = command.split()
        if not parts:
            return "empty_command"

        base = parts[0]
        # Keep structure but replace specific values
        template = base
        for part in parts[1:]:
            if re.match(r'\d{1,3}(\.\d{1,3}){3}', part):
                template += " <IP>"
            elif re.match(r'(-|--)\w+', part):
                template += f" {part}"  # Keep flags
            elif re.match(r'/\w+', part):
                template += " <PATH>"
            else:
                template += " <PARAM>"
        return template

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

if __name__ == "__main__":
    run_diagnostics()
