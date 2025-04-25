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
