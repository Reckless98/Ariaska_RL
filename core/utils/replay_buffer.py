import random
import os
import json
import sqlite3
from typing import Optional
import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer for RL agents. Supports prioritized replay, deduplication, and optional SQLite storage.
    Stores (state, action, reward, next_state, gpt_tokens) tuples for efficient sampling and memory management.
    """
    def __init__(self, capacity=2000, alpha=0.6, use_sqlite=False, db_path=None):
        """
        Initialize the replay buffer.
        Args:
            capacity (int): Maximum number of experiences to store.
            alpha (float): Prioritization factor for sampling.
            use_sqlite (bool): If True, use SQLite backend for scalable storage.
            db_path (str): Path to SQLite database file.
        """
        self.buffer = []
        self.capacity = capacity
        self.alpha = alpha  # Prioritization factor
        self.use_sqlite = use_sqlite
        self.db_path = db_path or "./replay_buffer.sqlite3"
        if self.use_sqlite:
            self._init_sqlite()

    def _init_sqlite(self):
        """
        Initialize SQLite database for experience storage.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            priority REAL,
            experience TEXT
        )''')
        self.conn.commit()

    def add(self, experience):
        """
        Add an experience to the buffer, deduplicating by (command, state, action).
        Args:
            experience (dict): Experience tuple to store.
        """
        # Deduplicate by command/state/action tuple
        key = self._experience_key(experience)
        if self.use_sqlite:
            if self._sqlite_exists(key):
                return
            priority = abs(experience.get('reward', 0)) + 0.01
            exp_json = json.dumps(experience)
            self.conn.execute("INSERT INTO experiences (priority, experience) VALUES (?, ?)", (priority, exp_json))
            self.conn.commit()
            self._sqlite_prune()
        else:
            if any(self._experience_key(e) == key for _, e in self.buffer):
                return
            priority = abs(experience.get('reward', 0)) + 0.01
            self.buffer.append((priority, experience))
            self.buffer = sorted(self.buffer, key=lambda x: x[0], reverse=True)
            if len(self.buffer) > self.capacity:
                self.buffer.pop()

    def sample(self, batch_size, prioritized=False):
        """
        Sample a batch of experiences.
        By default, uses uniform random sampling for stability.
        If prioritized=True, uses proportional sampling by priority^alpha.
        Args:
            batch_size (int): Number of experiences to sample.
            prioritized (bool): Whether to use prioritized sampling.
        Returns:
            list: Sampled experiences.
        """
        if self.use_sqlite:
            cursor = self.conn.execute("SELECT experience FROM experiences ORDER BY priority DESC LIMIT ?", (batch_size,))
            return [json.loads(row[0]) for row in cursor.fetchall()]
        if not self.buffer:
            return []
        if prioritized:
            # --- Proportional sampling by priority^alpha (importance sampling) ---
            import numpy as np
            priorities = np.array([priority for priority, _ in self.buffer])
            probs = priorities ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probs, replace=False)
            batch = [self.buffer[i][1] for i in indices]
            return batch
        else:
            # --- Uniform random sampling (default, recommended for DQN stability) ---
            indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
            batch = [self.buffer[i][1] for i in indices]
            return batch

    def prune_redundancy(self, redundancy_detector):
        """
        Remove redundant experiences using a provided redundancy detector function.
        Args:
            redundancy_detector (callable): Function that returns indices of redundant commands.
        """
        if self.use_sqlite:
            # Load all, prune, and rewrite
            cursor = self.conn.execute("SELECT id, experience FROM experiences")
            exps = [(row[0], json.loads(row[1])) for row in cursor.fetchall()]
            commands = [e['command'] for _, e in exps if 'command' in e]
            redundant_idxs = set(redundancy_detector(commands))
            for idx, (rowid, _) in enumerate(exps):
                if idx in redundant_idxs:
                    self.conn.execute("DELETE FROM experiences WHERE id=?", (rowid,))
            self.conn.commit()
        else:
            commands = [exp['command'] for _, exp in self.buffer if 'command' in exp]
            redundant_idxs = redundancy_detector(commands)
            self.buffer = [item for idx, item in enumerate(self.buffer) if idx not in redundant_idxs]

    def _experience_key(self, exp):
        """
        Generate a deduplication key for an experience.
        Args:
            exp (dict): Experience tuple.
        Returns:
            tuple: Key for deduplication.
        """
        # Use a tuple of (command, state, action) for deduplication
        return (
            exp.get('command'),
            str(exp.get('state', '')),
            str(exp.get('action', '')),
        )

    def _sqlite_exists(self, key):
        """
        Check if an experience already exists in the SQLite database.
        Args:
            key (tuple): Deduplication key.
        Returns:
            bool: True if exists, False otherwise.
        """
        # Check for duplicate by command/state/action
        cursor = self.conn.execute("SELECT experience FROM experiences")
        for row in cursor.fetchall():
            e = json.loads(row[0])
            if self._experience_key(e) == key:
                return True
        return False

    def _sqlite_prune(self):
        """
        Prune the SQLite database to maintain capacity.
        """
        # Keep only top N by priority
        cursor = self.conn.execute("SELECT id FROM experiences ORDER BY priority DESC")
        ids = [row[0] for row in cursor.fetchall()]
        if len(ids) > self.capacity:
            for id_to_remove in ids[self.capacity:]:
                self.conn.execute("DELETE FROM experiences WHERE id=?", (id_to_remove,))
            self.conn.commit()
