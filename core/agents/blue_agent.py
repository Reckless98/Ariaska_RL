# core/agents/blue_agent.py — ARIASKA BlueAgent v11.1 Sentinel Prime
# 🛡️ Defensive AI | Anomaly Detection | Intrusion Response | GPT-Enhanced Threat Defense

import os
import random
import subprocess
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.models.policy_net import PolicyNet
from core.models.value_net import ValueNet
from core.models.layers import get_phase_vector
from core.monitor.stats_monitor import StatsMonitor
from core.environment.cyber_environment import CyberEnvironment
from core.teach.teach import TeachModule
from core.ui_helpers import display_status_bar
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.utils.replay_buffer import ReplayBuffer
from core.utils.gpt_cache_handler import GPTCacheHandler
from core.interfaces.agent_interface import AgentInterface

console = Console()

def safe_tensor(data, device):
    import numpy as np
    if isinstance(data, torch.Tensor):
        return data.clone().detach().to(device)
    elif isinstance(data, (list, tuple, np.ndarray)):
        return torch.as_tensor(data, dtype=torch.float32, device=device).clone().detach()
    elif isinstance(data, dict):
        return BlueAgent.encode_env_state_static(data, device)
    else:
        raise TypeError(f"Cannot convert type {type(data)} to tensor.")

class BlueAgent(AgentInterface, MemorySyncInterface):
    def __init__(
        self,
        agent_id="BlueAgent",
        role="CyberDefense",
        device="cuda",
        agent_manager=None,
        memory_router=None,
        memory_manager=None,
        verbosity="standard"
    ):
        self.agent_id = agent_id
        self.role = role
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(input_size=512, output_size=5, device=self.device).to(self.device)
        self.value_net = ValueNet(input_size=512, device=self.device).to(self.device)
        self.env = CyberEnvironment(agent_manager=agent_manager, defer_reset=True)
        self.stats_monitor = StatsMonitor()
        self.teacher = TeachModule(agent_name=self.agent_id)
        self.memory_router = memory_router
        self.agent_manager = agent_manager
        self.memory_manager = memory_manager
        self.red = None
        self.orion = None
        self.replay_memory_size = 1500
        self.batch_size = 40
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.994
        self.entropy_beta = 0.012
        self.total_steps = 0
        self.total_episodes = 0
        self.command_history = []
        self.last_reasoning = ""
        self.current_mode = "Standard"
        self.prioritized_experiences = []
        self.prioritized_priorities = []
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs("logs", exist_ok=True)
        console.print(f"[green]✔ {self.agent_id} initialized — Sentinel Prime Mode on {self.device}[/green]")
        self.gpt_reasoning_cache = {}
        self.replay_buffer = ReplayBuffer(capacity=self.replay_memory_size)
        self.gpt_handler = GPTCacheHandler()
        self.gpt_cache = {}
        self.verbosity = verbosity
        self.last_action = None
        self.repeated_action_count = 0

    def _init_multiagent_links(self):
        self.red = self.agent_manager.get_agent("RedAgent")
        self.orion = self.agent_manager.get_agent("OrionAgent")

    def query_tactical_gpt(self, prompt, complexity="standard"):
        model = "gpt-4.1-nano" if complexity == "standard" else "gpt-4o-mini"
        try:
            result = subprocess.run(
                ["sgpt", "--model", model, "--role", "blue_agent", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                text=True,
            )
            response = result.stdout.strip()
            console.print(f"[magenta]🧠 GPT Tactical Insight:[/magenta] {response}")
            return response
        except Exception as e:
            console.print(f"[red]⚠ GPT query failed: {e}[/red]")
            return "Maintain current strategy."

    def select_action(self, state_tensor, phase=None):
        # Fix: Ensure phase is a string
        if isinstance(phase, dict) and "response" in phase:
            phase = phase["response"]
        if random.random() < self.epsilon:
            action = random.randint(0, self.policy_net.output_size - 1)
            console.print(f"[yellow]🎲 Random action selected due to exploration: {action}[/yellow]")
            return action
        action = self.policy_net.predict(
            state_tensor, get_phase_vector(phase, self.device)
        )
        console.print(f"[cyan]🎯 PolicyNet action selected: {action}[/cyan]")
        if self.current_mode != "Defensive":
            mode_switch_prompt = f"Current mode: {self.current_mode}. Should BlueAgent switch to a more defensive posture based on the current threat level?"
            mode_switch_decision = self.query_tactical_gpt(mode_switch_prompt)
            if "yes" in mode_switch_decision.lower():
                self.current_mode = "Defensive"
                console.print(f"[magenta]🚨 Mode switched to Defensive.[/magenta]")
        return action

    def simulate_step(self, episode=1, step=1, shared_context=None):
        try:
            state = self.env.reset() if step == 1 else getattr(self, "_last_state", self.env.get_global_state())
            # Use shared context for phase coordination if available
            if shared_context and "ScoutAgent_phase" in shared_context:
                state["phase"] = shared_context["ScoutAgent_phase"]
            if hasattr(self, 'agent_manager') and self.agent_manager:
                scout = getattr(self.agent_manager, 'scout_agent', None)
                if (scout and hasattr(scout, 'advise_phase')):
                    state["phase"] = scout.advise_phase(state, self.agent_manager.all_agents())
            try:
                state_tensor = self.encode_env_state(state)
            except Exception:
                # fallback for dict state
                state_tensor = torch.zeros(512, device=self.device)
            action = self.select_action(state_tensor, state.get("phase", None))
            # Defensive: ensure action is a string
            if isinstance(action, dict):
                action = action.get("response", str(action))
            if not isinstance(action, str):
                action = str(action)
            if not action or action == "N/A":
                action = "nmap -p- -sC -sV TARGET"
                console.print(f"[yellow]⚠ BlueAgent fallback to default action: {action}[/yellow]")
            next_state, reward, done, _ = self.env.step(action)
            # Defensive: ensure reward is a float
            try:
                if isinstance(reward, dict):
                    reward = reward.get("reward", 0.0)
                reward = float(reward)
            except Exception:
                reward = 0.0
            self.prioritized_experiences.append(
                {
                    "state": state_tensor,
                    "action": action,
                    "reward": reward,
                    "next_state": self.encode_env_state(next_state),
                    "done": done,
                }
            )
            self.prioritized_priorities.append(reward)
            self.command_history.append(str(action))
            self.stats_monitor.log_step(self.agent_id, reward, command=str(action))
            self._last_state = next_state
            self.last_output = f"Action: {action}, Reward: {reward}, Phase: {state.get('phase', 'N/A')}"
            reasoning_key = f"{action}|{state.get('phase')}"
            if reasoning_key in self.gpt_reasoning_cache:
                self.last_reasoning = self.gpt_reasoning_cache[reasoning_key]
            else:
                self.last_reasoning = self.gpt_handler.query(
                    f"Explain why action {action} is optimal for phase {state.get('phase')}.", model="gpt-4.1-nano"
                )
                self.gpt_reasoning_cache[reasoning_key] = self.last_reasoning
            display_status_bar(self.agent_id, episode, step)
            self._log_training_step(episode, step, str(action), state, self.last_output)
            # Print more info about agent state
            console.print(f"[dim]Replay buffer: {len(self.prioritized_experiences)} | Epsilon: {self.epsilon:.3f} | Entropy: {self.entropy_beta:.3f}[/dim]")
            # After decision, broadcast own phase/intent
            if self.agent_manager and hasattr(self.agent_manager, "broadcast"):
                self.agent_manager.broadcast(f"{self.agent_id}_phase", state.get("phase", "N/A"), sender=self.agent_id)
            # Activate countermeasures if alert level is high
            if state.get("blue_team_alert", 0) > 60:
                self.current_mode = "Defensive"
                console.print("[red]🚨 BlueAgent: Alert level high, activating countermeasures![/red]")
                # Example: deploy honeypots or reset credentials
                if hasattr(self.env, "honeypots"):
                    self.env.honeypots.append("fake_ssh")
            self.replay_buffer.add({
                "state": state_tensor,
                "action": action,
                "reward": reward,
                "next_state": self.encode_env_state(next_state),
                "done": done,
                "command": str(action),
            })
            # Track repeated actions
            if self.command_history:
                if self.command_history[-1] == self.last_action:
                    self.repeated_action_count += 1
                else:
                    self.repeated_action_count = 1
                self.last_action = self.command_history[-1]
            else:
                self.repeated_action_count = 1
                self.last_action = str(action)
            # Condensed logging
            if self.verbosity == "detailed":
                from rich.panel import Panel
                from rich.table import Table
                table = Table.grid()
                table.add_row("Action", str(action))
                table.add_row("Phase", str(state.get("phase", "N/A")))
                table.add_row("Reward", f"{reward:+.2f}")
                table.add_row("GPT", "4.1-nano" if "4.1" in self.last_reasoning else "4o-mini")
                console.print(Panel(table, title=f"{self.agent_id} Step {step}", border_style="blue"))
            elif self.verbosity == "standard":
                if self.repeated_action_count > 1:
                    console.print(f"[{self.agent_id}] Step {step} | Action: {action} (repeated x{self.repeated_action_count}) | Reward: {reward:.2f}")
                elif step == 1 or step % 5 == 0:
                    console.print(f"[{self.agent_id}] Step {step} | Action: {action} | Phase: {state.get('phase', 'N/A')} | Reward: {reward:.2f}")
            elif self.verbosity == "quiet":
                if reward < 0 or step == 1 or step % 10 == 0:
                    console.print(f"[{self.agent_id}] 🚨 Step {step} | Reward: {reward:.2f}")
            # Only display replay buffer status when thresholds are crossed
            if len(self.prioritized_experiences) % 100 == 0 and self.verbosity != "quiet":
                console.print(f"[dim]Replay buffer: {len(self.prioritized_experiences)}[/dim]")
            # Summarize every 5 steps
            if step % 5 == 0 and self.verbosity != "quiet":
                avg_reward = sum(self.stats_monitor.agent_stats[self.agent_id]["rewards"][-5:]) / 5
                alert = state.get("blue_team_alert", 0)
                console.print(f"[Summary] {self.agent_id} Steps {step-4}-{step}: Avg Reward {avg_reward:+.1f} | Alert {alert:.1f}")
            # --- Add: Print stats after each step ---
            if hasattr(self.stats_monitor, "show"):
                if step % 10 == 0 or step == 1:
                    self.stats_monitor.show()
            # Always return floats for reward, epsilon, entropy
            return {
                "command": str(action),
                "phase": state.get("phase", "N/A"),
                "reward": float(reward),
                "gpt_calls": self.stats_monitor.agent_stats[self.agent_id]["gpt_calls"],
                "output": self.last_output,
                "reasoning": self.last_reasoning,
                "step": step,
                "episode": episode,
                "agent_id": self.agent_id,
                "replay_buffer": len(self.prioritized_experiences),
                "epsilon": float(self.epsilon),
                "entropy_beta": float(self.entropy_beta),
            }
        except Exception as e:
            console.print(f"[red]❌ Error in BlueAgent simulate_step: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return {
                "command": "N/A",
                "phase": "N/A",
                "reward": 0.0,
                "gpt_calls": 0,
                "output": f"Error: {e}",
                "reasoning": "Error occurred",
                "step": step,
                "episode": episode,
                "agent_id": self.agent_id,
                "replay_buffer": 0,
                "epsilon": float(self.epsilon),
                "entropy_beta": float(self.entropy_beta),
            }

    def _log_training_step(self, episode, step, action, state, output):
        with open(self.training_log_path, "a") as f:
            f.write(f"Episode {episode}, Step {step}, Action: {action}, Phase: {state.get('phase')}, Output: {output}\n")

    def react_to_action(self, command, parsed_output=None):
        if not isinstance(command, str):
            console.print(f"[yellow]⚠ BlueAgent received non-string command: {command}[/yellow]")
            return {}
        if not parsed_output:
            parsed_output = {"phase": "unknown", "success": False, "risk_score": 0.0}
        threat_level = 0.0
        if "exploit" in command.lower() or "privesc" in command.lower():
            threat_level = 0.7
        elif "recon" in command.lower() or "scan" in command.lower():
            threat_level = 0.3
        alert_increase = threat_level * 10 * (1.0 if parsed_output.get("success", False) else 0.5)
        risk_increase = threat_level * 0.5
        honeypots_deployed = []
        credentials_reset = False
        if threat_level > 0.5 and random.random() < 0.3:
            honeypots_deployed = ["fake_ssh", "fake_http"]
            console.print(f"[blue]🛡️ {self.agent_id}: Deploying honeypot in response to threat[/blue]")
        if "exploit" in command.lower() and parsed_output.get("success", False) and random.random() < 0.2:
            credentials_reset = True
            console.print(f"[blue]🔐 {self.agent_id}: Resetting credentials after suspected breach[/blue]")
        # Dynamic risk adaptation
        if self.env.detection_risk > 5.0:
            self.current_mode = "Defensive"
            if self.verbosity != "silent":
                console.print("[red]🔴 High Risk Detected! Switching to Defensive Mode.[/red]")
            # Deploy honeypot if not already present
            if "fake_ssh" not in self.env.honeypots:
                self.env.honeypots.append("fake_ssh")
                if self.verbosity == "verbose":
                    console.print("[yellow]🛡️ BlueAgent deployed honeypot: fake_ssh[/yellow]")
        return {
            "alert_increase": alert_increase,
            "risk_increase": risk_increase,
            "honeypots": honeypots_deployed,
            "credentials_reset": credentials_reset
        }

    def train_on_batch(self):
        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            console.print("[yellow]⚠ Not enough experiences for batch training.[/yellow]")
            return
        states = torch.stack([exp["state"] for exp in batch]).to(self.device)
        actions = torch.tensor([exp["action"] for exp in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp["reward"] for exp in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp["next_state"] for exp in batch]).to(self.device)
        dones = torch.tensor([exp["done"] for exp in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            target_values = self.value_net(next_states)[0].squeeze()
            targets = rewards + self.gamma * target_values * (1 - dones)
        self.policy_net.train_step(states, actions, targets, entropy_beta=self.entropy_beta)
        self.value_net.train_step(states, targets)
        console.print(f"[cyan]🔧 {self.agent_id}: Batch training complete.[/cyan]")
        console.print(f"[bold magenta]🧠 {self.agent_id}: Policy/Value networks updated.[/bold magenta]")
        self._log_training_event("Batch training complete.")

    def _log_training_event(self, msg):
        with open(self.training_log_path, "a") as f:
            f.write(f"{msg}\n")

    def display_advanced_status(self):
        table = Table(title=f"🛡️ {self.agent_id} Defensive Status", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        avg_reward = self.stats_monitor.get_average_reward()
        detection_rate = self.stats_monitor.get_detection_rate()
        table.add_row("Role", self.role)
        table.add_row("Episodes Completed", str(self.total_episodes))
        table.add_row("Total Steps", str(self.total_steps))
        table.add_row("Avg Reward", f"{avg_reward:.2f}")
        table.add_row("Detection Rate", f"{detection_rate:.2%}")
        table.add_row("Epsilon (Exploration)", f"{self.epsilon:.4f}")
        table.add_row("Entropy Beta", f"{self.entropy_beta:.4f}")
        table.add_row("Replay Buffer", f"{len(self.prioritized_experiences)} / {self.replay_memory_size}")
        table.add_row("Last Reasoning", self.last_reasoning[:60] + "..." if self.last_reasoning else "N/A")
        console.print(Panel(table, title="🧠 Defensive Overview", border_style="bright_blue"))
        self.stats_monitor.visualize_phase_distribution()
        console.print(f"[dim]Training log: {self.training_log_path}[/dim]")

    def get_visualization_panel(self):
        from rich.panel import Panel
        from rich.table import Table
        table = Table(title=f"{self.agent_id} — Advanced Status", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Role", self.role)
        table.add_row("Episodes", str(self.total_episodes))
        table.add_row("Steps", str(self.total_steps))
        table.add_row("Avg Reward", f"{self.stats_monitor.get_average_reward():.2f}")
        table.add_row("Epsilon", f"{self.epsilon:.4f}")
        table.add_row("Entropy Beta", f"{self.entropy_beta:.4f}")
        table.add_row("Last Reasoning", self.last_reasoning[:80] + "..." if self.last_reasoning else "N/A")
        return Panel(table, title=f"🛡️ {self.agent_id} Defensive Panel", border_style="blue")

    def save_models(self, prefix="models/blue_agent"):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        try:
            self.policy_net.save(f"{prefix}_policy.pt")
            self.value_net.save(f"{prefix}_value.pt")
            console.print(f"[green]💾 {self.agent_id}: Models saved to {prefix}_*.pt[/green]")
        except Exception as e:
            console.print(f"[red]❌ {self.agent_id}: Model save failed: {e}[/red]")

    def load_models(self, prefix="models/blue_agent"):
        try:
            self.policy_net.load(f"{prefix}_policy.pt")
            self.value_net.load(f"{prefix}_value.pt")
            console.print(f"[cyan]✔ {self.agent_id}: Models loaded from {prefix}_*.pt[/cyan]")
        except Exception as e:
            console.print(f"[red]⚠ {self.agent_id}: Failed to load models: {e}[/red]")

    def safe_shutdown(self):
        self.save_models()
        console.print(f"[blue]🛡️ {self.agent_id}: Safe shutdown complete.[/blue]")
        self._log_training_event("Safe shutdown complete.")

    @staticmethod
    def encode_env_state_static(state, device):
        import numpy as np
        vec = []
        vec.append(float(state.get("phase", 0) in ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]))
        vec.append(float(state.get("privilege_level", "none") == "root"))
        vec.append(float(state.get("detection_risk", 0)))
        vec.append(float(state.get("blue_team_alert", 0)))
        vec.append(float(state.get("difficulty", 1)))
        vec.append(float(state.get("data_exfiltrated", False)))
        vec.append(float(state.get("credentials_found", False)))
        vec.append(float(state.get("done", False)))
        vec.append(float(len(state.get("open_ports", []))))
        vec.append(float(len(state.get("services", []))))
        while len(vec) < 512:
            vec.append(0.0)
        return torch.tensor(vec, dtype=torch.float32, device=device)

    def encode_env_state(self, state):
        return self.encode_env_state_static(state, self.device)

    def generate_hint(self):
        # New: Provide a tactical hint using memory or GPT
        if self.memory_router:
            mem = self.memory_router.get_memory(self.agent_id)
            if mem and mem.get("actions"):
                return mem["actions"][-1].get("full_command", "nmap -p- -sC -sV TARGET")
        return self.query_tactical_gpt("Suggest a defensive command for the current phase.")

    def execute_command(self, command):
        try:
            output = f"Executed: {command}"
            reward = random.uniform(0, 10)
            self.stats_monitor.log_step(self.agent_id, reward, command=command)
            return {
                "output": output,
                "recommendations": [{"command": command, "params": "Auto", "why": "Manual execution"}],
                "phase": "unknown",
                "reward": reward,
                "alert": 0.0,
                "entropy": None,
            }
        except Exception as e:
            console.print(f"[red]❌ Error executing command: {e}[/red]")
            return {
                "output": f"Error executing command: {e}",
                "recommendations": [],
                "phase": "unknown",
                "reward": 0,
                "alert": 0.0,
                "entropy": None,
            }

    def get_base_commands(self):
        # For CLI completion/autosuggest
        return [
            "nmap", "hydra", "msfconsole", "sqlmap", "ffuf", "gobuster",
            "linpeas", "winpeas", "evil-winrm", "masscan", "amass", "crackmapexec", "enum4linux", "pspy"
        ]
    
    def sync_memory(self):
        if self.memory_router:
            self.memory_router.sync_global_insights()
        # Import or define detect_redundancy_batch here
        from core.logic.redundancy_detector import detect_redundancy
        # Fallback: define a dummy function if not available
        def detect_redundancy_batch(cmds):
            return cmds
        self.replay_buffer.prune_redundancy(lambda cmds: detect_redundancy_batch(cmds))


# ─────────────────────────────────────────────
# 🎬 Execution Test Hook
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Import AgentManager and MemoryRouter here to avoid circular imports
    from core.multiagent.agent_manager import AgentManager
    from core.multiagent.memory_router import MemoryRouter

    agent_manager = AgentManager()
    memory_router = MemoryRouter()
    agent = BlueAgent(agent_manager=agent_manager, memory_router=memory_router)

    if hasattr(agent, "simulate_train"):
        agent.simulate_train(episodes=3)
    if hasattr(agent, "display_advanced_status"):
        agent.display_advanced_status()
    if hasattr(agent, "generate_chain_snapshot"):
        from core.logic.chainbuilder import build_and_store_chain
        agent.generate_chain_snapshot()

    agent.safe_shutdown()
