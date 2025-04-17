# rl_agent.py — ARIASKA v6.3 Tactical Intelligence Core (Finalized History Logic + Replay Patch)

import os
import json
import random
import subprocess
import re
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.models.policy_net import PolicyNet
from core.models.value_net import ValueNet
from core.cyber_environment import CyberEnvironment
from core.rule_engine import rule_based_selection
from core.output_interpreter import analyze_output
from core.ui_helpers import display_status_bar
from core.monitor import StatsMonitor

console = Console()


class RLAgent:
    def __init__(self, agent_id="RedAgent", role="Red Team", device="cuda"):
        self.agent_id = agent_id
        self.role = role
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.memory_path = "data/memory.json"
        self.history_path = "data/history.json"
        self.memory = self.load_memory()
        self.history = self.load_history()

        self.policy_net = PolicyNet(
            input_size=512, output_size=5, device=self.device
        ).to(self.device)
        self.value_net = ValueNet(input_size=512, device=self.device).to(self.device)

        self.env = CyberEnvironment(blue_team=True)
        self.stats_monitor = StatsMonitor(window_size=50)

        self.phase_tracker = {
            p: {"episodes": 0, "avg_reward": 0.0}
            for p in ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
        }

        self.prioritized_memory = []
        self.replay_memory_size = 1200
        self.batch_size = 32
        self.gamma = 0.985
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.996
        self.entropy_beta = 0.012

        self.total_steps = 0
        self.total_episodes = 0
        self.repeated_command_penalty = 0.85
        self.honeypot_trigger_count = 0

        console.print(f"[green]✔ {self.agent_id} initialized on {self.device}[/green]")

    def load_memory(self):
        if not os.path.exists(self.memory_path):
            return {"actions": [], "chains": [], "scenarios": []}
        try:
            with open(self.memory_path, "r") as file:
                return json.load(file)
        except Exception as e:
            console.print(f"[red]⚠ Error loading memory: {e}[/red]")
            return {"actions": [], "chains": [], "scenarios": []}

    def load_history(self):
        if not os.path.exists(self.history_path):
            return {"history": []}
        try:
            with open(self.history_path, "r") as file:
                return json.load(file)
        except Exception as e:
            console.print(f"[red]⚠ Error loading history: {e}[/red]")
            return {"history": []}

    def extract_output(self, command):
        try:
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )
            return result.stdout.decode("utf-8", errors="ignore")[:4000]
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except Exception as e:
            return f"[ERROR] {str(e)}"

    def execute_command(self, command):
        output = self.extract_output(command)
        parsed = analyze_output(command, output)
        return {
            "output": output,
            "recommendations": [
                {
                    "command": command,
                    "params": "manual",
                    "why": "User issued",
                    "Full Command": command,
                }
            ],
            "analysis": parsed,
        }

    def add_to_prioritized_memory(self, experience, td_error):
        if len(self.prioritized_memory) >= self.replay_memory_size:
            self.prioritized_memory.pop(0)
        experience["priority"] = td_error
        self.prioritized_memory.append(experience)

    def sample_prioritized_batch(self):
        if len(self.prioritized_memory) < self.batch_size:
            return []
        priorities = [e.get("priority", 1.0) for e in self.prioritized_memory]
        total = sum(priorities)
        probs = [p / total for p in priorities]
        indices = random.choices(
            range(len(self.prioritized_memory)), weights=probs, k=self.batch_size
        )
        return [self.prioritized_memory[i] for i in indices]

    def update_priorities(self, batch, td_errors):
        for i, sample in enumerate(batch):
            sample["priority"] = td_errors[i]

    def simulate_train(self, episodes=10, max_steps=60):
        console.print(f"[cyan]🔬 Simulating {episodes} episodes[/cyan]")
        for ep in range(episodes):
            state = self.env.reset()
            done, steps, total_reward = False, 0, 0.0
            recent_commands = set()
            while not done and steps < max_steps:
                state_vec = self.encode_env_state(state)
                action = self.select_action(state_vec, state.get("phase"))
                command = self.generate_command(state)

                reward_penalty = (
                    self.repeated_command_penalty if command in recent_commands else 1.0
                )
                recent_commands.add(command)

                output = self.extract_output(command)
                parsed = analyze_output(command, output)

                interpreted_context = {
                    **state,
                    "phase": parsed.get("phase", state.get("phase", "unknown")),
                    "artifacts": parsed.get("artifacts", []),
                    "stealth": parsed.get("success", False),
                    "honeypot_triggered": "fake_" in output,
                    "port_lockdown": len(state.get("open_ports", [])) <= 2,
                }

                next_state, _, done, info = self.env.step(action)
                reward = self.evaluate_reward(state, next_state, info) * reward_penalty
                total_reward += reward

                self.append_to_history(
                    command, reward, interpreted_context, output, parsed
                )
                self.inject_into_memory(command, reward, interpreted_context)

                experience = {
                    "state": state_vec,
                    "action": action,
                    "reward": reward,
                    "next_state": self.encode_env_state(next_state),
                    "done": done,
                }
                self.add_to_prioritized_memory(experience, abs(reward) + 1e-5)

                console.print(
                    f"[bold blue]Phase:[/bold blue] {state['phase']} | [green]Command:[/green] {command}"
                )
                console.print(
                    f"[magenta]Reward:[/magenta] {reward:.1f} | Alert: {state.get('blue_team_alert', 0):.2f}"
                )
                console.print(f"[dim white]Output:[/dim white] {output[:180].strip()}")

                state, steps = next_state, steps + 1
                self.phase_tracker[state.get("phase", "unknown")]["episodes"] += 1
                display_status_bar(agent=self.agent_id, episode=ep + 1, step=steps)

            self.total_episodes += 1
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.stats_monitor.log_episode_reward(total_reward)
            self.stats_monitor.display_stats()
            self.train_on_batch()

    def train_on_batch(self):
        batch = self.sample_prioritized_batch()
        if not batch:
            console.print("[yellow]⚠ Not enough data for training.[/yellow]")
            return

        states = torch.tensor([b["state"] for b in batch]).float().to(self.device)
        actions = torch.tensor([b["action"] for b in batch]).long().to(self.device)
        rewards = torch.tensor([b["reward"] for b in batch]).float().to(self.device)
        next_states = (
            torch.tensor([b["next_state"] for b in batch]).float().to(self.device)
        )
        dones = torch.tensor([b["done"] for b in batch]).float().to(self.device)

        next_values, _ = self.value_net.forward(next_states)
        next_values = next_values.squeeze()
        targets = rewards + self.gamma * next_values * (1 - dones)

        values, _ = self.value_net.forward(states)
        values = values.squeeze()
        advantages = targets.detach() - values

        policy_loss, entropy = self.policy_net.train_step(
            states, actions, advantages, entropy_beta=self.entropy_beta
        )
        value_loss = self.value_net.train_step(states, targets.detach())

        td_errors = advantages.abs().detach().cpu().numpy()
        self.update_priorities(batch, td_errors)

        console.print(
            f"[cyan]Trained: Policy={policy_loss:.3f} | Value={value_loss:.3f} | Entropy={entropy:.3f}[/cyan]"
        )
        console.print(
            f"[dim]Buffer size: {len(self.prioritized_memory)} | Batch size: {self.batch_size}[/dim]"
        )

    def evaluate_reward(self, state, next_state, info=None):
        base_rewards = {
            "recon": 5.0,
            "enumeration": 6.0,
            "exploit": 24.0,
            "privesc": 40.0,
            "exfiltrate": 140.0,
        }
        base = base_rewards.get(next_state.get("phase", ""), 0)
        detection_penalty = -7 * next_state.get("detection_risk", 0.0)
        alert_penalty = min(-3 * next_state.get("blue_team_alert", 0.0), -100)
        exfil_bonus = 150 if next_state.get("data_exfiltrated") else 0
        stealth_bonus = (
            40
            if next_state.get("detection_risk", 1.0) < 0.15
            and next_state.get("phase") == "exfiltrate"
            else 0
        )
        honeypot_penalty = -80 if next_state.get("honeypot_triggered") else 0
        return (
            base
            + detection_penalty
            + alert_penalty
            + exfil_bonus
            + stealth_bonus
            + honeypot_penalty
        )

    def encode_env_state(self, obs):
        mapping = {
            "recon": 0,
            "enumeration": 1,
            "exploit": 2,
            "privesc": 3,
            "exfiltrate": 4,
        }
        encoded = [
            mapping.get(obs.get("phase"), -1),
            len(obs.get("open_ports", [])),
            len(obs.get("services", [])),
            int(obs.get("credentials_found", False)),
            {"none": 0, "user": 1, "root": 2}.get(obs.get("privilege_level"), -1),
            obs.get("detection_risk", 0.0),
            obs.get("blue_team_alert", 0.0),
            int(obs.get("honeypot_triggered", 0)),
            int(obs.get("port_lockdown", False)),
        ]
        return encoded + [0.0] * (512 - len(encoded))

    def encode_phase_vector(self, phase):
        mapping = {
            "recon": 0,
            "enumeration": 1,
            "exploit": 2,
            "privesc": 3,
            "exfiltrate": 4,
        }
        vec = [0.0] * 5
        idx = mapping.get(phase, -1)
        if idx != -1:
            vec[idx] = 1.0
        return torch.tensor(vec, dtype=torch.float32, device=self.device)

    def select_action(self, state_vector, phase_name):
        phase_vec = self.encode_phase_vector(phase_name)
        tensor = torch.tensor(
            state_vector, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits = self.policy_net.forward(tensor, phase_vector=phase_vec)
        probs = torch.softmax(logits, dim=-1).squeeze()
        if random.random() < self.epsilon:
            return random.randint(0, probs.shape[0] - 1)
        return torch.multinomial(probs, 1).item()

    def inject_into_memory(self, command, reward, context):
        template = self.template_from_command(command)
        exists = any(
            self.template_from_command(e["command"]) == template
            for e in self.memory["actions"]
        )
        if not exists:
            self.memory["actions"].append(
                {"command": template, "reward": reward, "context": context}
            )
            self.save_memory()

    def append_to_history(
        self, command, reward, context=None, output=None, parsed=None
    ):
        context = context or {}
        parsed = parsed or {}
        command_template = self.template_from_command(command)
        output_cleaned = output or ""
        output_cleaned = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "{IP}", output_cleaned)
        output_cleaned = re.sub(r"\b\d{2,5}\b", "{PORT}", output_cleaned)
        output_cleaned = re.sub(r"\b[a-f0-9]{32,64}\b", "{HASH}", output_cleaned)

        entry = {
            "command": command_template,
            "reward": reward,
            "context": context,
            "output": output_cleaned,
            "analysis": parsed,
        }

        self.history["history"].append(entry)
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            console.print(f"[red]⚠ Error saving history: {e}[/red]")

    def save_memory(self):
        try:
            with open(self.memory_path, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            console.print(f"[red]⚠ Error saving memory: {e}[/red]")

    def template_from_command(self, command):
        ip_pattern = re.compile(r"(\b(?:\d{1,3}\.){3}\d{1,3}\b)")
        port_pattern = re.compile(r"\b\d{2,5}\b")
        hash_pattern = re.compile(r"\b[a-f0-9]{32,64}\b")

        tokens = command.strip().split()
        rebuilt = []

        for tok in tokens:
            if ip_pattern.search(tok):
                rebuilt.append(re.sub(ip_pattern, "{IP}", tok))
            elif port_pattern.fullmatch(tok):
                rebuilt.append("{PORT}")
            elif hash_pattern.match(tok):
                rebuilt.append("{HASH}")
            else:
                rebuilt.append(tok)

        return " ".join(rebuilt)

    def generate_command(self, state):
        phase = state.get("phase", "unknown")
        filtered = [
            (self.template_from_command(a["command"]), a["reward"])
            for a in self.memory["actions"]
            if a["context"].get("phase") == phase
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)

        if filtered:
            template = filtered[0][0]
        else:
            template = rule_based_selection(self.history["history"], state)["command"]

        filled = template.replace("{IP}", state.get("target_ip", "10.10.10.10"))
        return filled.replace("{PORT}", str(state.get("open_ports", [80])[0]))

    def display_status(self):
        table = Table(title=f"{self.agent_id} Status", style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Agent ID", self.agent_id)
        table.add_row("Role", self.role)
        table.add_row("Device", str(self.device))
        table.add_row("Total Steps", str(self.total_steps))
        table.add_row("Total Episodes", str(self.total_episodes))
        table.add_row("Epsilon", f"{self.epsilon:.4f}")
        table.add_row("Difficulty Level", str(self.env.difficulty_level))
        table.add_row("Avg Reward", f"{self.stats_monitor.get_avg_reward():.2f}")
        console.print(Panel(table, title="Agent Overview", border_style="bright_blue"))

    def generate_hint(self):
        try:
            state = self.env.get_state()
            phase = state.get("phase", "unknown")

            matching = [
                (self.template_from_command(entry["command"]), entry["reward"])
                for entry in self.memory.get("actions", [])
                if entry.get("context", {}).get("phase") == phase
            ]
            matching.sort(key=lambda x: x[1], reverse=True)

            if matching:
                best_template = matching[0][0]
            else:
                rule_based = rule_based_selection(
                    self.history.get("history", []), state
                )
                best_template = rule_based.get("command", "nmap -p- -sC -sV {IP}")

            return best_template.replace(
                "{IP}", state.get("target_ip", "10.10.10.10")
            ).replace("{PORT}", str(state.get("open_ports", [80])[0]))

        except Exception as e:
            console.print(f"[red]⚠ Hint generation failed: {e}[/red]")
            return "nmap -p- -sC -sV 10.10.10.10"


# === CLI Hook ===
def handle_command(command, agent):
    args = command.strip().split()
    if not args:
        return
    cmd = args[0]
    if cmd == "simulate-train":
        episodes = int(args[1]) if len(args) > 1 else 50
        agent.simulate_train(episodes=episodes)
    elif cmd == "train-batch":
        agent.train_on_batch()
    elif cmd == "status":
        agent.display_status()
    elif cmd == "chains":
        for chain in agent.memory.get("chains", []):
            console.print(chain)
    elif cmd in ("exit", "quit"):
        console.print("[bold red]Exiting...[/bold red]")
        exit()
    else:
        console.print(f"[yellow]Unknown command: {command}[/yellow]")


if __name__ == "__main__":
    try:
        agent = RLAgent()
        agent.simulate_train(episodes=1)

        while True:
            try:
                command = input("zer0@ARIASKA > ").strip()
                if command:
                    handle_command(command, agent)
            except KeyboardInterrupt:
                console.print(
                    "\n[bold red]⚠ Interrupted. Shutting down Ariaska RL...[/bold red]"
                )
                break
            except Exception as inner_e:
                console.print(f"[red]❌ Command error: {inner_e}[/red]")

    except Exception as e:
        console.print(f"[red]❌ Critical failure: {e}[/red]")
