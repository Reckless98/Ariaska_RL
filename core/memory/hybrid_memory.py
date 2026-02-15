#!/usr/bin/env python3
"""
core/memory/hybrid_memory.py — ARIASKA Hybrid Memory System v1.0

Three-tier memory architecture optimized for RL training throughput:

    L0: RAM Working Memory     — Current episode transitions, deque-based, O(1) append
    L1: Memory-Mapped Bulk     — Historical transitions, numpy memmap, sequential I/O
    L2: LMDB Persistent Store  — Metadata, indices, priorities, SIL traces, episode summaries

Design Goals:
    - Zero step-loop overhead: writes are async + batched
    - Multi-view access: PPO, SAC, DDQN, CognitionNode, SIL all read same data
    - Prioritized replay: priorities stored in LMDB, O(log n) sampling via SumTree in RAM
    - Episode-aware: can replay full episodes (for SIL / hindsight)
    - Crash-resilient: LMDB is ACID, memmap survives crashes

Architecture:
    ┌─────────────────────────────────────────────────┐
    │               HybridMemory                      │
    │  ┌───────────┐  ┌──────────────┐  ┌───────────┐│
    │  │ L0: RAM   │  │ L1: MemMap   │  │ L2: LMDB  ││
    │  │ (deque)   │→ │ (numpy)      │→ │ (persist) ││
    │  │ ~2K trans │  │ ~500K trans  │  │ metadata  ││
    │  │ O(1) R/W  │  │ seq append   │  │ priorities││
    │  │ cur_ep    │  │ bulk history │  │ episodes  ││
    │  └───────────┘  └──────────────┘  │ SIL trace ││
    │                                   │ indexes   ││
    │  ┌───────────────────────────┐    └───────────┘│
    │  │ AsyncWriter (daemon)     │                  │
    │  │ batches L0 → L1/L2      │                  │
    │  │ flush every 0.5s or 64  │                  │
    │  └───────────────────────────┘                  │
    │  ┌───────────────────────────┐                  │
    │  │ SumTree (priorities)     │                  │
    │  │ O(log n) PER sampling   │                  │
    │  │ synced from L2          │                  │
    │  └───────────────────────────┘                  │
    └─────────────────────────────────────────────────┘

Usage:
    from core.memory.hybrid_memory import get_hybrid_memory

    mem = get_hybrid_memory()
    mem.store_transition(state, action, reward, next_state, done, metadata)
    batch = mem.sample(batch_size=64, priority_mode=True)
    mem.end_episode(episode_id, summary)
    mem.get_episode(episode_id)  # replay full episode

Author: Filip Volf / Ariaska System
"""

import atexit
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("ariaska.hybrid_memory")

# ─── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MEMMAP_DIR = DATA_DIR / "memmap_store"
LMDB_DIR = DATA_DIR / "lmdb_store"

STATE_DIM = 512  # Must match state_encoder.py

# Try to import lmdb, fall back gracefully
try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False
    logger.info("lmdb not installed — L2 persistence will use JSON fallback. "
                "Install with: pip install lmdb")


@dataclass
class HybridMemoryConfig:
    """Configuration for the hybrid memory system."""
    # L0: RAM
    l0_capacity: int = 2048           # Working memory capacity (current episode)
    l0_max_episodes: int = 5          # Keep last N episodes in L0

    # L1: MemMap
    l1_capacity: int = 500_000        # Total transitions in memmap
    l1_state_dim: int = STATE_DIM     # State vector dimension
    l1_dir: Path = MEMMAP_DIR

    # L2: LMDB
    l2_dir: Path = LMDB_DIR
    l2_map_size: int = 1 * 1024**3    # 1 GB LMDB map size

    # Async writer
    batch_size: int = 64              # Flush after N transitions
    flush_interval: float = 0.5       # Flush every 0.5 seconds
    enable_async: bool = True         # Use async writer daemon

    # PER
    per_alpha: float = 0.6            # PER exponent
    per_beta_start: float = 0.4       # IS weight annealing start
    per_beta_end: float = 1.0         # IS weight annealing end
    per_epsilon: float = 1e-6         # Minimum priority

    # Multi-view
    enable_sil: bool = True           # Store SIL traces
    enable_episode_replay: bool = True  # Store full episodes for hindsight


# ─── SumTree for O(log n) Prioritized Sampling ─────────────────────────────

class SumTree:
    """
    Binary sum tree for prioritized experience replay.
    O(log n) sampling, O(log n) priority update.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_idx = 0
        self.count = 0

    @property
    def total_priority(self) -> float:
        return self.tree[0]

    def update(self, data_idx: int, priority: float):
        """Update priority for a data index."""
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority: float) -> int:
        """Add a new entry with given priority. Returns data index."""
        data_idx = self.write_idx
        self.update(data_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        return data_idx

    def sample(self, value: float) -> int:
        """Sample a data index proportional to priorities."""
        tree_idx = 0
        while True:
            left = 2 * tree_idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left] or right >= len(self.tree):
                tree_idx = left
            else:
                value -= self.tree[left]
                tree_idx = right
        return tree_idx - (self.capacity - 1)

    def batch_sample(self, batch_size: int) -> List[int]:
        """Sample a batch of indices proportionally."""
        total = self.total_priority
        if total <= 0 or self.count == 0:
            return list(range(min(batch_size, self.count)))

        segment = total / batch_size
        indices = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            idx = self.sample(value)
            idx = max(0, min(idx, self.count - 1))
            indices.append(idx)
        return indices


# ─── L1: Memory-Mapped Transition Store ─────────────────────────────────────

class MemmapStore:
    """
    Memory-mapped numpy arrays for bulk transition storage.
    Sequential append, random read. Survives process crashes.
    """

    def __init__(self, config: HybridMemoryConfig):
        self.config = config
        self.dir = config.l1_dir
        os.makedirs(self.dir, exist_ok=True)

        self.capacity = config.l1_capacity
        self.state_dim = config.l1_state_dim
        self.write_idx = 0
        self.count = 0

        # Create or load memmaps
        self._states = self._open_memmap("states", (self.capacity, self.state_dim))
        self._actions = self._open_memmap("actions", (self.capacity,), dtype=np.int32)
        self._rewards = self._open_memmap("rewards", (self.capacity,))
        self._next_states = self._open_memmap("next_states", (self.capacity, self.state_dim))
        self._dones = self._open_memmap("dones", (self.capacity,), dtype=np.bool_)
        self._episodes = self._open_memmap("episodes", (self.capacity,), dtype=np.int32)

        # Load write position
        self._load_position()

    def _open_memmap(self, name: str, shape: tuple,
                     dtype=np.float32) -> np.memmap:
        """Open or create a memory-mapped numpy array."""
        path = self.dir / f"{name}.dat"
        mode = "r+" if path.exists() else "w+"
        return np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)

    def _load_position(self):
        """Load write position from disk."""
        pos_file = self.dir / "position.json"
        if pos_file.exists():
            try:
                data = json.loads(pos_file.read_text())
                self.write_idx = data.get("write_idx", 0)
                self.count = data.get("count", 0)
            except Exception:
                pass

    def _save_position(self):
        """Save write position to disk."""
        pos_file = self.dir / "position.json"
        pos_file.write_text(json.dumps({
            "write_idx": self.write_idx,
            "count": self.count,
        }))

    def append(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool, episode_id: int = 0):
        """Append a single transition."""
        idx = self.write_idx

        self._states[idx] = state[:self.state_dim]
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state[:self.state_dim]
        self._dones[idx] = done
        self._episodes[idx] = episode_id

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def append_batch(self, states: np.ndarray, actions: np.ndarray,
                     rewards: np.ndarray, next_states: np.ndarray,
                     dones: np.ndarray, episode_ids: np.ndarray):
        """Append a batch of transitions (from async writer flush)."""
        n = len(states)
        for i in range(n):
            self.append(states[i], int(actions[i]), float(rewards[i]),
                        next_states[i], bool(dones[i]),
                        int(episode_ids[i]) if len(episode_ids) > i else 0)

    def get(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """Get transitions by indices."""
        valid = [i for i in indices if 0 <= i < self.count]
        if not valid:
            return {
                "states": np.zeros((0, self.state_dim)),
                "actions": np.zeros(0, dtype=np.int32),
                "rewards": np.zeros(0),
                "next_states": np.zeros((0, self.state_dim)),
                "dones": np.zeros(0, dtype=np.bool_),
            }
        idx = np.array(valid)
        return {
            "states": np.array(self._states[idx]),
            "actions": np.array(self._actions[idx]),
            "rewards": np.array(self._rewards[idx]),
            "next_states": np.array(self._next_states[idx]),
            "dones": np.array(self._dones[idx]),
        }

    def flush(self):
        """Flush memmaps to disk."""
        self._states.flush()
        self._actions.flush()
        self._rewards.flush()
        self._next_states.flush()
        self._dones.flush()
        self._episodes.flush()
        self._save_position()

    def __len__(self) -> int:
        return self.count


# ─── L2: LMDB Persistent Store ─────────────────────────────────────────────

class LMDBStore:
    """
    LMDB-backed persistent store for metadata, priorities, episode summaries,
    and SIL traces. ACID-compliant, crash-safe.

    Falls back to JSON files if lmdb is not installed.
    """

    def __init__(self, config: HybridMemoryConfig):
        self.config = config
        self.dir = config.l2_dir
        os.makedirs(self.dir, exist_ok=True)

        self._env = None
        self._json_fallback: Dict[str, Dict] = {}
        self._json_file = self.dir / "fallback_store.json"

        if HAS_LMDB:
            try:
                self._env = lmdb.open(
                    str(self.dir / "main.lmdb"),
                    map_size=config.l2_map_size,
                    max_dbs=8,
                    sync=False,  # Async for speed
                    writemap=True,
                )
                # Named databases
                with self._env.begin(write=True) as txn:
                    self._db_priorities = self._env.open_db(b"priorities", txn=txn)
                    self._db_episodes = self._env.open_db(b"episodes", txn=txn)
                    self._db_sil = self._env.open_db(b"sil_traces", txn=txn)
                    self._db_meta = self._env.open_db(b"metadata", txn=txn)
                    self._db_agent = self._env.open_db(b"agent_data", txn=txn)
                logger.info("L2: LMDB initialized")
            except Exception as e:
                logger.error(f"L2: LMDB init failed, using JSON fallback: {e}")
                self._env = None
                self._load_json_fallback()
        else:
            self._load_json_fallback()

    def _load_json_fallback(self):
        """Load JSON fallback store."""
        if self._json_file.exists():
            try:
                self._json_fallback = json.loads(self._json_file.read_text())
            except Exception:
                self._json_fallback = {}
        for key in ["priorities", "episodes", "sil_traces", "metadata", "agent_data"]:
            if key not in self._json_fallback:
                self._json_fallback[key] = {}

    def _save_json_fallback(self):
        """Save JSON fallback store."""
        try:
            self._json_file.write_text(json.dumps(self._json_fallback, indent=1, default=str))
        except Exception as e:
            logger.error(f"L2 JSON save failed: {e}")

    def put(self, db_name: str, key: str, value: Any):
        """Store a key-value pair."""
        if self._env:
            db = getattr(self, f"_db_{db_name}", None)
            if db is not None:
                with self._env.begin(write=True) as txn:
                    txn.put(
                        key.encode(),
                        json.dumps(value, default=str).encode(),
                        db=db,
                    )
        else:
            self._json_fallback.setdefault(db_name, {})[key] = value

    def get(self, db_name: str, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        if self._env:
            db = getattr(self, f"_db_{db_name}", None)
            if db is not None:
                with self._env.begin() as txn:
                    data = txn.get(key.encode(), db=db)
                    if data:
                        return json.loads(data.decode())
        else:
            return self._json_fallback.get(db_name, {}).get(key)
        return None

    def put_batch(self, db_name: str, items: Dict[str, Any]):
        """Store multiple key-value pairs in a single transaction."""
        if self._env:
            db = getattr(self, f"_db_{db_name}", None)
            if db is not None:
                with self._env.begin(write=True) as txn:
                    for key, value in items.items():
                        txn.put(
                            key.encode(),
                            json.dumps(value, default=str).encode(),
                            db=db,
                        )
        else:
            for key, value in items.items():
                self._json_fallback.setdefault(db_name, {})[key] = value

    def get_all(self, db_name: str) -> Dict[str, Any]:
        """Get all entries from a database."""
        if self._env:
            db = getattr(self, f"_db_{db_name}", None)
            if db is not None:
                result = {}
                with self._env.begin() as txn:
                    cursor = txn.cursor(db=db)
                    for key, value in cursor:
                        result[key.decode()] = json.loads(value.decode())
                return result
        return dict(self._json_fallback.get(db_name, {}))

    def count(self, db_name: str) -> int:
        """Count entries in a database."""
        if self._env:
            db = getattr(self, f"_db_{db_name}", None)
            if db is not None:
                with self._env.begin() as txn:
                    return txn.stat(db=db)["entries"]
        return len(self._json_fallback.get(db_name, {}))

    def flush(self):
        """Force sync to disk."""
        if self._env:
            self._env.sync()
        else:
            self._save_json_fallback()

    def close(self):
        """Close LMDB environment."""
        if self._env:
            self._env.close()
        else:
            self._save_json_fallback()


# ─── Async Writer Daemon ───────────────────────────────────────────────────

class AsyncWriter:
    """
    Background daemon that batches L0 → L1/L2 writes.
    Runs in a separate thread, flushes every batch_size transitions or
    flush_interval seconds (whichever comes first).
    """

    def __init__(self, l1: MemmapStore, l2: LMDBStore,
                 sum_tree: SumTree, config: HybridMemoryConfig):
        self.l1 = l1
        self.l2 = l2
        self.sum_tree = sum_tree
        self.config = config

        self._queue: Deque[Dict] = deque(maxlen=config.l1_capacity)
        self._priority_queue: Deque[Tuple[int, float]] = deque(maxlen=config.l1_capacity)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._flush_event = threading.Event()

    def start(self):
        """Start the async writer daemon."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="HybridMemory-AsyncWriter"
        )
        self._thread.start()
        logger.debug("AsyncWriter daemon started")

    def stop(self):
        """Stop the async writer daemon and flush remaining."""
        self._running = False
        self._flush_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._flush_all()
        logger.debug("AsyncWriter daemon stopped")

    def enqueue(self, transition: Dict):
        """Enqueue a transition for async write."""
        with self._lock:
            self._queue.append(transition)
        if len(self._queue) >= self.config.batch_size:
            self._flush_event.set()

    def enqueue_priority(self, data_idx: int, priority: float):
        """Enqueue a priority update."""
        with self._lock:
            self._priority_queue.append((data_idx, priority))

    def _run(self):
        """Main daemon loop."""
        while self._running:
            self._flush_event.wait(timeout=self.config.flush_interval)
            self._flush_event.clear()
            self._flush_all()

    def _flush_all(self):
        """Flush all queued transitions and priorities."""
        # Flush transitions
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()
            priorities = list(self._priority_queue)
            self._priority_queue.clear()

        if batch:
            for t in batch:
                state = t.get("state", np.zeros(self.config.l1_state_dim))
                next_state = t.get("next_state", np.zeros(self.config.l1_state_dim))
                if isinstance(state, (list, tuple)):
                    state = np.array(state, dtype=np.float32)
                if isinstance(next_state, (list, tuple)):
                    next_state = np.array(next_state, dtype=np.float32)

                # Ensure correct dimension
                if len(state) < self.config.l1_state_dim:
                    padded = np.zeros(self.config.l1_state_dim, dtype=np.float32)
                    padded[:len(state)] = state
                    state = padded
                if len(next_state) < self.config.l1_state_dim:
                    padded = np.zeros(self.config.l1_state_dim, dtype=np.float32)
                    padded[:len(next_state)] = next_state
                    next_state = padded

                self.l1.append(
                    state=state,
                    action=t.get("action", 0),
                    reward=t.get("reward", 0.0),
                    next_state=next_state,
                    done=t.get("done", False),
                    episode_id=t.get("episode_id", 0),
                )

                # Add to SumTree with default priority
                priority = t.get("priority", 1.0)
                data_idx = self.sum_tree.add(priority)

                # Store metadata in L2
                meta = {k: v for k, v in t.items()
                        if k not in ("state", "next_state") and not isinstance(v, np.ndarray)}
                if meta:
                    self.l2.put("metadata", str(data_idx), meta)

            self.l1.flush()
            logger.debug(f"AsyncWriter flushed {len(batch)} transitions")

        # Flush priority updates
        if priorities:
            for data_idx, priority in priorities:
                self.sum_tree.update(data_idx, priority)

    @property
    def pending(self) -> int:
        return len(self._queue)


# ─── Main HybridMemory Class ───────────────────────────────────────────────

class HybridMemory:
    """
    Three-tier hybrid memory system for multi-algorithm RL training.

    Provides unified storage and multi-view access for PPO, SAC, DDQN, RND, and SIL.
    """

    _instance: Optional["HybridMemory"] = None

    def __new__(cls, *args, **kwargs):
        """Singleton — one memory for all algorithms."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[HybridMemoryConfig] = None):
        if self._initialized:
            return

        self.config = config or HybridMemoryConfig()

        # L0: RAM working memory
        self._l0: Deque[Dict] = deque(maxlen=self.config.l0_capacity)
        self._l0_episodes: Dict[int, List[Dict]] = {}  # episode_id → transitions
        self._current_episode: int = 0

        # L1: Memory-mapped bulk store
        self._l1 = MemmapStore(self.config)

        # L2: LMDB persistent store
        self._l2 = LMDBStore(self.config)

        # PER: SumTree
        self._sum_tree = SumTree(self.config.l1_capacity)

        # Async writer
        self._writer: Optional[AsyncWriter] = None
        if self.config.enable_async:
            self._writer = AsyncWriter(self._l1, self._l2, self._sum_tree, self.config)
            self._writer.start()

        # Episode tracking
        self._episode_step = 0
        self._total_transitions = 0

        # SIL buffer (Self-Imitation Learning — high-reward episodes)
        self._sil_buffer: Deque[Dict] = deque(maxlen=1000)

        # Register shutdown handler
        atexit.register(self._shutdown)

        self._initialized = True
        logger.info(f"HybridMemory initialized: L0={self.config.l0_capacity}, "
                     f"L1={self.config.l1_capacity}, L2={'LMDB' if HAS_LMDB else 'JSON'}, "
                     f"async={'on' if self._writer else 'off'}")

    # ─── Core Operations ────────────────────────────────────────────────

    def store_transition(self, state, action: int, reward: float,
                         next_state, done: bool,
                         metadata: Optional[Dict] = None,
                         priority: Optional[float] = None):
        """
        Store a single transition. Goes to L0 immediately,
        async-batched to L1/L2.

        Args:
            state: State vector (numpy array or list)
            action: Action index
            reward: Scalar reward
            next_state: Next state vector
            done: Episode termination flag
            metadata: Optional dict with extra info (agent_id, source, etc.)
            priority: Optional PER priority (default: |reward| + epsilon)
        """
        # Convert to numpy if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32) if state is not None else np.zeros(STATE_DIM)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32) if next_state is not None else np.zeros(STATE_DIM)

        # Compute priority
        if priority is None:
            priority = abs(reward) + self.config.per_epsilon

        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "episode_id": self._current_episode,
            "step": self._episode_step,
            "priority": priority,
            "timestamp": time.time(),
        }
        if metadata:
            transition.update(metadata)

        # L0: Immediate RAM store
        self._l0.append(transition)

        # Track per-episode
        if self._current_episode not in self._l0_episodes:
            self._l0_episodes[self._current_episode] = []
        self._l0_episodes[self._current_episode].append(transition)

        # Async write to L1/L2
        if self._writer:
            self._writer.enqueue(transition)
        else:
            # Synchronous fallback
            self._sync_write(transition)

        self._episode_step += 1
        self._total_transitions += 1

        # SIL: store high-reward transitions
        if self.config.enable_sil and reward > 5.0:
            self._sil_buffer.append(transition)

    def _sync_write(self, transition: Dict):
        """Synchronous write to L1 (when async is disabled)."""
        state = transition["state"]
        next_state = transition["next_state"]
        if len(state) < STATE_DIM:
            padded = np.zeros(STATE_DIM, dtype=np.float32)
            padded[:len(state)] = state
            state = padded
        if len(next_state) < STATE_DIM:
            padded = np.zeros(STATE_DIM, dtype=np.float32)
            padded[:len(next_state)] = next_state
            next_state = padded

        self._l1.append(
            state=state,
            action=transition["action"],
            reward=transition["reward"],
            next_state=next_state,
            done=transition["done"],
            episode_id=transition.get("episode_id", 0),
        )
        priority = transition.get("priority", 1.0)
        self._sum_tree.add(priority)

    # ─── Sampling ───────────────────────────────────────────────────────

    def sample(self, batch_size: int = 64,
               priority_mode: bool = True,
               source: str = "any") -> Dict[str, Any]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            priority_mode: Use prioritized replay (PER) if True
            source: "l0" (current episode), "l1" (historical), "any" (both)

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones,
            indices, weights (IS weights for PER)
        """
        if source == "l0" or (source == "any" and len(self._l0) >= batch_size):
            return self._sample_l0(batch_size)

        # L1 sampling
        if len(self._l1) == 0:
            return self._sample_l0(batch_size)

        if priority_mode and self._sum_tree.count > 0:
            return self._sample_per(batch_size)
        else:
            return self._sample_uniform(batch_size)

    def _sample_l0(self, batch_size: int) -> Dict[str, Any]:
        """Sample from L0 RAM (current/recent episodes)."""
        n = min(batch_size, len(self._l0))
        if n == 0:
            return self._empty_batch()

        indices = np.random.choice(len(self._l0), size=n, replace=False)
        transitions = [self._l0[i] for i in indices]

        return self._transitions_to_batch(transitions, list(indices))

    def _sample_per(self, batch_size: int) -> Dict[str, Any]:
        """Sample from L1 using prioritized experience replay."""
        indices = self._sum_tree.batch_sample(batch_size)
        batch = self._l1.get(indices)

        # Compute IS weights
        total_p = self._sum_tree.total_priority
        min_p = self.config.per_epsilon
        weights = np.ones(len(indices))

        if total_p > 0:
            for i, idx in enumerate(indices):
                tree_idx = idx + self._sum_tree.capacity - 1
                if 0 <= tree_idx < len(self._sum_tree.tree):
                    p = self._sum_tree.tree[tree_idx] / total_p
                    p = max(p, min_p)
                    weights[i] = (1.0 / (len(self._l1) * p)) ** self.config.per_beta_start
            weights /= weights.max() + 1e-8

        batch["indices"] = np.array(indices)
        batch["weights"] = weights
        return batch

    def _sample_uniform(self, batch_size: int) -> Dict[str, Any]:
        """Uniform random sample from L1."""
        n = min(batch_size, len(self._l1))
        indices = np.random.choice(len(self._l1), size=n, replace=False)
        batch = self._l1.get(list(indices))
        batch["indices"] = indices
        batch["weights"] = np.ones(n)
        return batch

    def _transitions_to_batch(self, transitions: List[Dict],
                               indices: List[int]) -> Dict[str, Any]:
        """Convert list of transition dicts to batch arrays."""
        n = len(transitions)
        if n == 0:
            return self._empty_batch()

        states = np.zeros((n, STATE_DIM), dtype=np.float32)
        actions = np.zeros(n, dtype=np.int32)
        rewards = np.zeros(n, dtype=np.float32)
        next_states = np.zeros((n, STATE_DIM), dtype=np.float32)
        dones = np.zeros(n, dtype=np.bool_)

        for i, t in enumerate(transitions):
            s = t.get("state", np.zeros(STATE_DIM))
            ns = t.get("next_state", np.zeros(STATE_DIM))
            if isinstance(s, np.ndarray):
                states[i, :len(s)] = s[:STATE_DIM]
            if isinstance(ns, np.ndarray):
                next_states[i, :len(ns)] = ns[:STATE_DIM]
            actions[i] = t.get("action", 0)
            rewards[i] = t.get("reward", 0.0)
            dones[i] = t.get("done", False)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "indices": np.array(indices),
            "weights": np.ones(n),
        }

    def _empty_batch(self) -> Dict[str, Any]:
        return {
            "states": np.zeros((0, STATE_DIM)),
            "actions": np.zeros(0, dtype=np.int32),
            "rewards": np.zeros(0),
            "next_states": np.zeros((0, STATE_DIM)),
            "dones": np.zeros(0, dtype=np.bool_),
            "indices": np.zeros(0, dtype=np.int32),
            "weights": np.zeros(0),
        }

    # ─── Priority Updates ───────────────────────────────────────────────

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update PER priorities from TD errors (called by PPO/DDQN/SAC)."""
        priorities = (np.abs(td_errors) + self.config.per_epsilon) ** self.config.per_alpha
        for idx, p in zip(indices, priorities):
            if self._writer:
                self._writer.enqueue_priority(idx, float(p))
            else:
                self._sum_tree.update(idx, float(p))

    # ─── Episode Management ────────────────────────────────────────────

    def start_episode(self, episode_id: int):
        """Signal start of a new episode."""
        self._current_episode = episode_id
        self._episode_step = 0

    def end_episode(self, episode_id: int, summary: Optional[Dict] = None):
        """
        Signal end of episode. Stores episode summary in L2,
        optionally marks high-reward episodes for SIL.
        """
        # Store episode summary
        if summary:
            self._l2.put("episodes", str(episode_id), summary)

        # Compute episode total reward
        ep_transitions = self._l0_episodes.get(episode_id, [])
        if ep_transitions:
            total_reward = sum(t.get("reward", 0.0) for t in ep_transitions)

            # SIL: if this was a great episode, mark it
            if self.config.enable_sil and total_reward > 50.0:
                sil_trace = {
                    "episode_id": episode_id,
                    "total_reward": total_reward,
                    "steps": len(ep_transitions),
                    "timestamp": time.time(),
                }
                self._l2.put("sil_traces", str(episode_id), sil_trace)

        # Clean old episodes from L0 (keep last N)
        if len(self._l0_episodes) > self.config.l0_max_episodes:
            oldest = sorted(self._l0_episodes.keys())
            for old_ep in oldest[:len(self._l0_episodes) - self.config.l0_max_episodes]:
                del self._l0_episodes[old_ep]

    def get_episode(self, episode_id: int) -> List[Dict]:
        """Get all transitions from a specific episode (for replay/SIL)."""
        # Check L0 first
        if episode_id in self._l0_episodes:
            return self._l0_episodes[episode_id]

        # Check L2 for episode metadata
        summary = self._l2.get("episodes", str(episode_id))
        if summary:
            return summary.get("transitions", [])

        return []

    # ─── SIL Access ─────────────────────────────────────────────────────

    def sample_sil(self, batch_size: int = 32) -> Dict[str, Any]:
        """Sample from SIL buffer (high-reward transitions only)."""
        n = min(batch_size, len(self._sil_buffer))
        if n == 0:
            return self._empty_batch()
        indices = np.random.choice(len(self._sil_buffer), size=n, replace=False)
        transitions = [self._sil_buffer[i] for i in indices]
        return self._transitions_to_batch(transitions, list(indices))

    def get_sil_traces(self) -> List[Dict]:
        """Get all SIL traces (high-reward episode summaries)."""
        return list(self._l2.get_all("sil_traces").values())

    # ─── Agent-Specific Views ──────────────────────────────────────────

    def store_agent_data(self, agent_id: str, key: str, data: Any):
        """Store agent-specific data in L2."""
        composite_key = f"{agent_id}:{key}"
        self._l2.put("agent_data", composite_key, data)

    def get_agent_data(self, agent_id: str, key: str) -> Optional[Any]:
        """Retrieve agent-specific data from L2."""
        composite_key = f"{agent_id}:{key}"
        return self._l2.get("agent_data", composite_key)

    # ─── Statistics ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "l0_count": len(self._l0),
            "l0_episodes": len(self._l0_episodes),
            "l1_count": len(self._l1),
            "l1_capacity": self.config.l1_capacity,
            "l2_episodes": self._l2.count("episodes"),
            "l2_sil_traces": self._l2.count("sil_traces"),
            "sumtree_total": self._sum_tree.total_priority,
            "sumtree_count": self._sum_tree.count,
            "sil_buffer": len(self._sil_buffer),
            "total_transitions": self._total_transitions,
            "async_pending": self._writer.pending if self._writer else 0,
        }

    # ─── Lifecycle ──────────────────────────────────────────────────────

    def flush(self):
        """Force flush all pending writes."""
        if self._writer:
            self._writer._flush_all()
        self._l1.flush()
        self._l2.flush()

    def _shutdown(self):
        """Graceful shutdown."""
        logger.debug("HybridMemory shutting down...")
        if self._writer:
            self._writer.stop()
        self._l1.flush()
        self._l2.close()

    @classmethod
    def reset_singleton(cls):
        """Reset singleton (for testing)."""
        if cls._instance and cls._instance._initialized:
            try:
                cls._instance._shutdown()
            except Exception:
                pass
        cls._instance = None

    def __len__(self) -> int:
        return len(self._l0) + len(self._l1)

    def __repr__(self) -> str:
        return (f"HybridMemory(L0={len(self._l0)}, L1={len(self._l1)}, "
                f"total={self._total_transitions})")


# ─── Factory ────────────────────────────────────────────────────────────────

_hybrid_memory: Optional[HybridMemory] = None


def get_hybrid_memory(config: Optional[HybridMemoryConfig] = None) -> HybridMemory:
    """Get or create the singleton HybridMemory."""
    return HybridMemory(config=config)


if __name__ == "__main__":
    """Quick test."""
    logging.basicConfig(level=logging.DEBUG)

    config = HybridMemoryConfig(enable_async=False)
    mem = HybridMemory(config=config)

    # Store some transitions
    for i in range(100):
        state = np.random.randn(512).astype(np.float32)
        next_state = np.random.randn(512).astype(np.float32)
        mem.store_transition(state, i % 5, float(i) * 0.1, next_state, i % 20 == 19)

    print(f"Stats: {json.dumps(mem.stats(), indent=2, default=str)}")

    # Sample
    batch = mem.sample(32)
    print(f"Batch states shape: {batch['states'].shape}")
    print(f"Batch rewards: {batch['rewards'][:5]}")

    HybridMemory.reset_singleton()
