# core/agents/redagent_brain.py — RedAgent Brain/Episodic Memory
# 🧠 Episodic Memory | GPT Feedback Loop | Self-Reflection | Strategy Evolution
import os
import json
import threading
from collections import deque
from datetime import datetime
from rich.console import Console

console = Console()

"""
RedAgentBrain — Episodic Memory & Feedback Logger for RedAgent
------------------------------------------------------------
• Logs every RedAgent step (state, command, GPT response, output, reward, success, model, tokens, etc.)
• Stores GPT feedback for learning and dashboard
• Provides deduplication, disk flush, and stats aggregation
• Used for continual self-improvement and meta-learning
"""

class RedAgentBrain:
    """
    Episodic memory and feedback logger for RedAgent.
    Stores step-by-step logs and GPT feedback for learning and dashboard.

    Methods:
        log_step: Log a single agent step (state, command, output, etc.)
        log_gpt_feedback: Log GPT meta-analysis/feedback
        load_recent_steps: Retrieve recent steps for dashboard/learning
        load_recent_gpt_feedback: Retrieve recent GPT feedback
        flush_to_disk: (No-op, append-only by default)
    """
    def __init__(self, log_dir="logs/redagent_evolution", max_steps=200):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.step_log_path = os.path.join(self.log_dir, "steps.jsonl")
        self.gpt_log_path = os.path.join(self.log_dir, "gpt_feedback.jsonl")
        self.steps = deque(maxlen=max_steps)
        self.gpt_feedback = deque(maxlen=20)
        self._load_from_disk()

    def log_step(self, **kwargs):
        # Log a single step (state, command, gpt_response, output, reward, etc.)
        entry = {
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.steps.append(entry)
        try:
            with open(self.step_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            console.print(f"[yellow]⚠ RedAgentBrain: Failed to log step: {e}[/yellow]")

    def log_gpt_feedback(self, prompt, gpt_feedback, summary, episode):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "episode": episode,
            "prompt": prompt,
            "gpt_feedback": gpt_feedback,
            "summary": summary
        }
        self.gpt_feedback.append(entry)
        try:
            with open(self.gpt_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            console.print(f"[yellow]⚠ RedAgentBrain: Failed to log GPT feedback: {e}[/yellow]")

    def load_recent_steps(self, n=20):
        # Return last n steps from memory (for dashboard/learning)
        return list(self.steps)[-n:]

    def load_recent_gpt_feedback(self, n=5):
        return list(self.gpt_feedback)[-n:]

    def flush_to_disk(self):
        # No-op: logs are append-only, but could flush in-memory buffer if needed
        pass

    def _load_from_disk(self):
        # Load recent steps and feedback from disk (for dashboard reload)
        try:
            if os.path.exists(self.step_log_path):
                with open(self.step_log_path, "r") as f:
                    lines = f.readlines()[-self.steps.maxlen :]
                    for line in lines:
                        self.steps.append(json.loads(line))
            if os.path.exists(self.gpt_log_path):
                with open(self.gpt_log_path, "r") as f:
                    lines = f.readlines()[-self.gpt_feedback.maxlen :]
                    for line in lines:
                        self.gpt_feedback.append(json.loads(line))
        except Exception as e:
            console.print(f"[yellow]⚠ RedAgentBrain: Failed to load logs: {e}[/yellow]")
