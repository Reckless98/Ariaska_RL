# core/utils/memory_manager.py — ARIASKA MemoryManager v12.0 Nexus Prime
# 🧠 GPT-First Intelligence | 🌐 Insight Scoring | 📸 Smart Snapshots | ⚡ Async-Optimized

import os
import json
from datetime import datetime
from rich.console import Console
import asyncio
import threading

console = Console()


class MemoryManager:
    def __init__(self, agent_name):
        self.agent_name = agent_name.lower()
        self.memory_dir = os.path.join("core", "memories", f"{self.agent_name}_memory")
        self.shared_dir = os.path.join("core", "memories", "shared")
        self.snapshots_dir = os.path.join(self.memory_dir, "snapshots")
        self.meta_file = os.path.join(self.snapshots_dir, "snapshot_meta.json")

        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.shared_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self.memory_file = os.path.join(self.memory_dir, "memory.json")
        self.history_file = os.path.join(self.memory_dir, "history.json")
        self.gpt_cache_file = os.path.join(self.memory_dir, "gpt_cache.json")

        self.memory = self._load_json(
            self.memory_file, {"actions": [], "rewards": {}, "scenarios": []}
        )
        self.history = self._load_json(self.history_file, [])
        self.gpt_cache = self._load_json(self.gpt_cache_file, {})

        self._initialize_meta()

        self._lock = threading.Lock()

        console.print(
            f"[green]✔ {self.agent_name.capitalize()} MemoryManager v12.0 initialized[/green]"
        )

    def _initialize_meta(self):
        if not os.path.exists(self.meta_file):
            self._save_json(self.meta_file, {"snapshots": []})

    def _load_json(self, path, default):
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[red]❌ Failed to load {path}: {e}[/red]")
        return default

    def _save_json(self, path, data):
        with self._lock:
            try:
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                console.print(f"[red]❌ Failed to save {path}: {e}[/red]")

    def append_jsonl(self, path, entry):
        with self._lock:
            try:
                with open(path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                console.print(f"[red]❌ Failed to append to {path}: {e}[/red]")

    def save_all(self):
        self._save_json(self.memory_file, self.memory)
        self._save_json(self.history_file, self.history)
        self._save_json(self.gpt_cache_file, self.gpt_cache)
        console.print(
            f"[cyan]💾 {self.agent_name}: All memory components saved.[/cyan]"
        )

    def save_memory(self):
        """Save only the main memory file (for compatibility with agent code)."""
        self._save_json(self.memory_file, self.memory)

    def load_gpt_cache(self):
        """
        Return the GPT cache dict (for compatibility with agent code).
        """
        return self.gpt_cache

    def get_all_memory(self):
        """
        Return a dict with all memory components for export/logging.
        """
        return {
            "memory": self.memory,
            "history": self.history,
            "gpt_cache": self.gpt_cache,
        }

    def load_shared_knowledge(self, filename=None):
        """
        Load shared knowledge file (used by ShadowAgent, etc).
        """
        if filename is None:
            filename = "shared_knowledge.json"
        path = os.path.join(self.shared_dir, filename)
        return self._load_json(path, {"insights": []})

    def save_shared_knowledge(self, data, filename=None):
        """
        Save shared knowledge file.
        """
        if filename is None:
            filename = "shared_knowledge.json"
        path = os.path.join(self.shared_dir, filename)
        self._save_json(path, data)

    # ─────────────────────────────────────────────
    # 🧠 Advanced GPT Cache Operations
    # ─────────────────────────────────────────────
    def cache_gpt_response(self, key, response, ttl_days=7):
        now = datetime.now().isoformat()
        self.gpt_cache[key] = {
            "response": response,
            "cached_at": now,
            "ttl_days": ttl_days,
        }
        self._save_json(self.gpt_cache_file, self.gpt_cache)
        console.print(f"[cyan]🧠 Cached GPT response:[/cyan] {key}")

    def prune_expired_cache(self):
        before = len(self.gpt_cache)
        self.gpt_cache = {
            k: v
            for k, v in self.gpt_cache.items()
            if self._is_cache_valid(v.get("cached_at"), v.get("ttl_days", 7))
        }
        after = len(self.gpt_cache)
        self._save_json(self.gpt_cache_file, self.gpt_cache)
        console.print(
            f"[blue]⚡ Pruned GPT cache: {before - after} expired entries removed.[/blue]"
        )

    def _is_cache_valid(self, cached_at, ttl_days):
        try:
            cached_time = datetime.fromisoformat(cached_at)
            return (datetime.now() - cached_time).days < ttl_days
        except:
            return False

    # ─────────────────────────────────────────────
    # 🌐 Global Insight Sync with Scoring
    # ─────────────────────────────────────────────
    def sync_to_global_insights(self, threshold=50):
        global_file = os.path.join(self.shared_dir, "global_insights.json")
        global_data = self._load_json(global_file, {"actions": []})
        existing_templates = {a.get("template") for a in global_data["actions"]}

        new_insights = [
            {**a, "synced_at": datetime.now().isoformat()}
            for a in self.memory.get("actions", [])
            if a.get("reward", 0) >= threshold
            and a.get("template") not in existing_templates
        ]

        global_data["actions"].extend(new_insights)
        self._save_json(global_file, global_data)
        console.print(
            f"[blue]🔗 Synced {len(new_insights)} high-reward actions globally.[/blue]"
        )

    # ─────────────────────────────────────────────
    # 📸 Smart Snapshot System (Rotating + Meta)
    # ─────────────────────────────────────────────
    def snapshot_memory(self, max_snapshots=10, force=False, episode_num=None, critical_event=False):
        # Only snapshot every 2 episodes or on critical event
        if not force and not critical_event and (episode_num is not None and episode_num % 2 != 0):
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(
            self.snapshots_dir, f"{self.agent_name}_snapshot_{timestamp}.json"
        )
        self._save_json(snapshot_path, self.memory)

        meta = self._load_json(self.meta_file, {"snapshots": []})
        meta["snapshots"].append({"file": snapshot_path, "created": timestamp})
        if len(meta["snapshots"]) > max_snapshots:
            oldest = meta["snapshots"].pop(0)
            if os.path.exists(oldest["file"]):
                try:
                    os.remove(oldest["file"])
                except Exception as e:
                    console.print(f"[yellow]⚠ Failed to remove old snapshot {oldest['file']}: {e}[/yellow]")
        self._save_json(self.meta_file, meta)
        console.print(f"[magenta][Snapshot] Saved: {snapshot_path}[/magenta]")

    def restore_latest_snapshot(self):
        meta = self._load_json(self.meta_file, {"snapshots": []})
        if not meta["snapshots"]:
            console.print("[red]❌ No snapshots available to restore.[/red]")
            return
        latest = meta["snapshots"][-1]["file"]
        if os.path.exists(latest):
            self.memory = self._load_json(latest, self.memory)
            console.print(f"[green]✔ Restored from: {latest}[/green]")
        else:
            console.print(f"[red]❌ Snapshot file not found: {latest}[/red]")

    # ─────────────────────────────────────────────
    # 🧹 Async Memory Optimization
    # ─────────────────────────────────────────────
    async def optimize_memory(self, reward_floor=15):
        patched = 0
        for action in self.memory.get("actions", []):
            if action.get("reward", 0) < reward_floor:
                action["reward"] = max(action.get("reward", 0), reward_floor)
                patched += 1
        self._save_json(self.memory_file, self.memory)
        console.print(f"[green]✔ Async optimized {patched} low-reward actions[/green]")

    # ─────────────────────────────────────────────
    # 🚀 Diagnostic & Maintenance Mode
    # ─────────────────────────────────────────────


if __name__ == "__main__":
    mm = MemoryManager(agent_name="red_agent")
    mm.snapshot_memory()
    mm.sync_to_global_insights()
    asyncio.run(mm.optimize_memory())
    mm.prune_expired_cache()
    mm.restore_latest_snapshot()
