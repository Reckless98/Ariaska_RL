# core/monitor.py

from rich.console import Console
from rich.table import Table
from collections import deque

console = Console()

class StatsMonitor:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.episode_counter = 0
        self.rewards = deque(maxlen=window_size)
        self.policy_losses = deque(maxlen=window_size)
        self.value_losses = deque(maxlen=window_size)
        self.entropies = deque(maxlen=window_size)

    # Log reward after each episode
    def log_episode_reward(self, reward):
        self.rewards.append(reward)
        self.episode_counter += 1

    # Log training metrics after each batch train
    def log_training_metrics(self, policy_loss, value_loss, entropy):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)

    # Calculate moving averages (safe even if empty)
    def get_avg_reward(self):
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def get_avg_policy_loss(self):
        return sum(self.policy_losses) / len(self.policy_losses) if self.policy_losses else 0.0

    def get_avg_value_loss(self):
        return sum(self.value_losses) / len(self.value_losses) if self.value_losses else 0.0

    def get_avg_entropy(self):
        return sum(self.entropies) / len(self.entropies) if self.entropies else 0.0

    def get_reward_momentum(self):
        if len(self.rewards) >= 2:
            return self.rewards[-1] - self.rewards[-2]
        return 0.0

    def reset(self):
        """
        Resets all tracked stats.
        """
        self.episode_counter = 0
        self.rewards.clear()
        self.policy_losses.clear()
        self.value_losses.clear()
        self.entropies.clear()

    # Display live dashboard
    def display_stats(self):
        table = Table(title="Ariaska RL - Live Stats Dashboard")

        table.add_column("Metric", style="cyan", justify="center")
        table.add_column("Latest", justify="center")
        table.add_column(f"Avg Last {self.window_size}", justify="center")

        table.add_row("Episode Count", str(self.episode_counter), "-")

        table.add_row("Reward", 
                      f"{self.rewards[-1]:.2f}" if self.rewards else "-", 
                      f"{self.get_avg_reward():.2f}")

        table.add_row("Reward Δ", 
                      f"{self.get_reward_momentum():+.2f}" if len(self.rewards) >= 2 else "-", 
                      "-")

        table.add_row("Policy Loss", 
                      f"{self.policy_losses[-1]:.4f}" if self.policy_losses else "-", 
                      f"{self.get_avg_policy_loss():.4f}")

        table.add_row("Value Loss", 
                      f"{self.value_losses[-1]:.4f}" if self.value_losses else "-", 
                      f"{self.get_avg_value_loss():.4f}")

        table.add_row("Entropy", 
                      f"{self.entropies[-1]:.4f}" if self.entropies else "-", 
                      f"{self.get_avg_entropy():.4f}")

        console.clear()  # Optional: Clears terminal screen for updated stats
        console.print(table) 
