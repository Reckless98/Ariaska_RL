# core/utils/memory_manager.py — ARIASKA MemoryManager v12.0 Nexus Prime
# 🧠 GPT-First Intelligence | 🌐 Insight Scoring | 📸 Smart Snapshots | ⚡ Async-Optimized

import os
import sqlite3
import threading
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console

console = Console()

DEFAULT_MEMORY_DIR = "core/memories"
DEFAULT_DB_PATH = os.path.join(DEFAULT_MEMORY_DIR, "memory.sqlite3")
DEFAULT_CACHE_SIZE = 2000

class ExperienceMemory:
    """
    Experience replay buffer using SQLite for fast, transactional storage.
    Stores (state, action, reward, next_state, gpt_tokens) tuples.
    """
    def __init__(self, db_path=DEFAULT_DB_PATH, table="experiences", capacity=DEFAULT_CACHE_SIZE):
        self.db_path = db_path
        self.table = table
        self.capacity = capacity
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT,
                    action TEXT,
                    reward REAL,
                    next_state TEXT,
                    gpt_tokens INTEGER,
                    timestamp REAL DEFAULT (strftime('%s','now'))
                )
            """)
            conn.commit()

    def append(self, state, action, reward, next_state, gpt_tokens=0):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO {self.table} (state, action, reward, next_state, gpt_tokens) VALUES (?, ?, ?, ?, ?)",
                (json.dumps(state), str(action), float(reward), json.dumps(next_state), int(gpt_tokens))
            )
            # Prune oldest if over capacity
            count = conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
            if count > self.capacity:
                to_remove = count - self.capacity
                conn.execute(f"DELETE FROM {self.table} WHERE id IN (SELECT id FROM {self.table} ORDER BY id ASC LIMIT ?)", (to_remove,))
            conn.commit()

    def sample_batch(self, batch_size) -> List[Dict[str, Any]]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT state, action, reward, next_state, gpt_tokens FROM {self.table} ORDER BY RANDOM() LIMIT ?",
                (batch_size,)
            ).fetchall()
        batch = []
        for state, action, reward, next_state, gpt_tokens in rows:
            batch.append({
                "state": json.loads(state),
                "action": action,
                "reward": reward,
                "next_state": json.loads(next_state),
                "gpt_tokens": gpt_tokens
            })
        return batch

    def size(self) -> int:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]

    def stats(self) -> Dict[str, Any]:
        return {"size": self.size(), "capacity": self.capacity}

class ActionKnowledgeBase:
    """
    Stores static action metadata (command, phase, reward, etc.) in SQLite.
    """
    def __init__(self, db_path=DEFAULT_DB_PATH, table="actions"):
        self.db_path = db_path
        self.table = table
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT,
                    phase TEXT,
                    reward REAL,
                    meta TEXT,
                    timestamp REAL DEFAULT (strftime('%s','now'))
                )
            """)
            conn.commit()

    def add_action(self, command, phase, reward, meta=None):
        meta_json = json.dumps(meta or {})
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO {self.table} (command, phase, reward, meta) VALUES (?, ?, ?, ?)",
                (command, phase, float(reward), meta_json)
            )
            conn.commit()

    def get_actions(self, limit=100) -> List[Dict[str, Any]]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT command, phase, reward, meta FROM {self.table} ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [
            {
                "command": cmd,
                "phase": phase,
                "reward": reward,
                "meta": json.loads(meta) if meta else {}
            }
            for cmd, phase, reward, meta in rows
        ]

    def stats(self) -> Dict[str, Any]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            count = conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
        return {"actions": count}

class LLMCache:
    """
    Persistent cache for GPT/LLM responses, keyed by prompt hash.
    """
    def __init__(self, db_path=DEFAULT_DB_PATH, table="llm_cache"):
        self.db_path = db_path
        self.table = table
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp REAL DEFAULT (strftime('%s','now'))
                )
            """)
            conn.commit()

    def save_response(self, key: str, response: str):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {self.table} (key, response) VALUES (?, ?)",
                (key, response)
            )
            conn.commit()

    def get_response(self, key: str) -> Optional[str]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"SELECT response FROM {self.table} WHERE key=?",
                (key,)
            ).fetchone()
        return row[0] if row else None

    def stats(self) -> Dict[str, Any]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            count = conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
        return {"cache_entries": count}

class MemoryManager:
    """
    Unified per-agent memory manager using modular, database-backed storage.
    - ExperienceMemory: RL transitions for DQN/replay
    - ActionKnowledgeBase: static action info
    - LLMCache: GPT/LLM response cache
    All paths and sizes are configurable.
    """
    def __init__(self, agent_id="RedAgent", memory_dir=DEFAULT_MEMORY_DIR, capacity=DEFAULT_CACHE_SIZE):
        self.agent_id = agent_id
        self.memory_dir = os.path.join(memory_dir, agent_id.lower())
        os.makedirs(self.memory_dir, exist_ok=True)
        self.db_path = os.path.join(self.memory_dir, "memory.sqlite3")
        self.experience = ExperienceMemory(self.db_path, capacity=capacity)
        self.actions = ActionKnowledgeBase(self.db_path)
        self.llm_cache = LLMCache(self.db_path)
        self._stats = {"last_snapshot": None}

    def save_experience(self, state, action, reward, next_state, gpt_tokens=0):
        self.experience.append(state, action, reward, next_state, gpt_tokens)

    def sample_batch(self, batch_size) -> List[Dict[str, Any]]:
        return self.experience.sample_batch(batch_size)

    def add_action(self, action: Dict[str, Any]):
        self.actions.add_action(
            action.get("command"),
            action.get("phase"),
            action.get("reward"),
            meta=action.get("meta", {})
        )

    def get_actions(self, limit=100) -> List[Dict[str, Any]]:
        return self.actions.get_actions(limit=limit)

    def save_gpt_response(self, key: str, response: str):
        self.llm_cache.save_response(key, response)

    def get_gpt_response(self, key: str) -> Optional[str]:
        return self.llm_cache.get_response(key)

    def snapshot(self):
        """
        Save a snapshot of current memory stats (non-blocking).
        """
        stats = {
            "experience": self.experience.stats(),
            "actions": self.actions.stats(),
            "llm_cache": self.llm_cache.stats(),
            "timestamp": time.time()
        }
        path = os.path.join(self.memory_dir, f"snapshot_{int(time.time())}.json")
        threading.Thread(target=self._write_snapshot, args=(path, stats), daemon=True).start()
        self._stats["last_snapshot"] = path

    def _write_snapshot(self, path, stats):
        try:
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        return {
            "experience": self.experience.stats(),
            "actions": self.actions.stats(),
            "llm_cache": self.llm_cache.stats(),
            "last_snapshot": self._stats.get("last_snapshot")
        }

if __name__ == "__main__":
    mm = MemoryManager(agent_id="red_agent")
    mm.save_experience({"state": 1}, "action", 1.0, {"state": 2})
    mm.add_action({"command": "cmd", "phase": "test", "reward": 10})
    mm.save_gpt_response("key", "response")
    print(mm.sample_batch(1))
    print(mm.get_actions())
    print(mm.get_gpt_response("key"))
    mm.snapshot()
    print(mm.stats())
