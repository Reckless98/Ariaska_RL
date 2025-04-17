import time
from collections import deque
from rich.console import Console
from rich.table import Table

console = Console()

class StatsMonitor:
    def __init__(self, max_history=100):
        # Rolling logs
        self.rewards = deque(maxlen=max_history)
        self.steps = deque(maxlen=max_history)
        self.detections = deque(maxlen=max_history)
        self.phases = {}

        # Track bests
        self.best_reward = float('-inf')
        self.worst_reward = float('inf')
        self.total_episodes = 0
        self.total_steps = 0

        # Timing
        self.start_time = time.time()

    def log_step(self, reward, detection_risk, phase):
        self.rewards.append(reward)
        self.detections.append(detection_risk)
        self.steps.append(1)

        self.total_steps += 1
        self.best_reward = max(self.best_reward, reward)
        self.worst_reward = min(self.worst_reward, reward)

        if phase not in self.phases:
            self.phases[phase] = 0
        self.phases[phase] += 1

    def log_episode(self):
        self.total_episodes += 1

    def show(self):
        elapsed = time.time() - self.start_time
        avg_reward = sum(self.rewards) / len(self.rewards) if self.rewards else 0
        avg_detection = sum(self.detections) / len(self.detections) if self.detections else 0

        table = Table(title=f"🧠 Agent Stats | Episodes: {self.total_episodes} | Steps: {self.total_steps} | Time: {elapsed:.1f}s", style="cyan")

        table.add_column("Avg Reward", justify="center")
        table.add_column("Best Reward", justify="center")
        table.add_column("Worst Reward", justify="center")
        table.add_column("Detection Risk", justify="center")
        table.add_column("Phase Counts", justify="center")

        table.add_row(
            f"{avg_reward:.2f}",
            f"{self.best_reward:.2f}",
            f"{self.worst_reward:.2f}",
            f"{avg_detection:.2f}",
            str(self.phases)
        )

        console.clear()  # optional: refreshes the console clean
        console.print(table)

    def reset(self):
        self.rewards.clear()
        self.steps.clear()
        self.detections.clear()
        self.phases.clear()
        self.start_time = time.time()

