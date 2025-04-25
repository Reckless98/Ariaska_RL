# core/agents/blue_agent.py — ARIASKA BlueAgent v11.1 Sentinel Prime
# 🛡️ Defensive AI | Anomaly Detection | Intrusion Response | GPT-Enhanced Threat Defense

import os
import random
import subprocess
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    from core.models.policy_net import PolicyNet
    from core.models.value_net import ValueNet
except ModuleNotFoundError:
    from ..models.policy_net import PolicyNet
    from ..models.value_net import ValueNet

from core.models.layers import get_phase_vector
from core.monitor.stats_monitor import StatsMonitor
from core.environment.cyber_environment import CyberEnvironment
from core.teach.teach import TeachModule
from core.ui_helpers import display_status_bar
from core.ui_helpers import get_action_description
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.utils.replay_buffer import ReplayBuffer
from core.utils.gpt_cache_handler import GPTCacheHandler
from core.interfaces.agent_interface import AgentInterface
from core.gpt_manager import GPTManager
from core.memory_router import MemoryRouter
from core.utils.local_llm_manager import LocalLLMManager

console = Console()


def safe_tensor(data, device):
    import numpy as np

    if isinstance(data, torch.Tensor):
        return data.clone().detach().to(device)
    elif isinstance(data, (list, tuple, np.ndarray)):
        return (
            torch.as_tensor(data, dtype=torch.float32, device=device).clone().detach()
        )
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
        verbosity="standard",
    ):
        self.agent_id = agent_id
        self.role = role
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(
            input_size=512, output_size=5, device=self.device
        ).to(self.device)
        # --- DQN Target Network ---
        self.target_net = PolicyNet(
            input_size=512, output_size=5, device=self.device
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.env = CyberEnvironment(agent_manager=agent_manager, defer_reset=True)
        self.stats_monitor = StatsMonitor()
        self.teacher = TeachModule(agent_name=self.agent_id)
        self.memory_router = memory_router or MemoryRouter()
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
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs("logs", exist_ok=True)
        console.print(
            f"[green]✔ {self.agent_id} initialized — Sentinel Prime Mode on {self.device}[/green]"
        )
        self.gpt_reasoning_cache = {}
        self.gpt_handler = GPTCacheHandler()
        self.gpt_cache = {}
        self.verbosity = verbosity
        self.last_action = None
        self.repeated_action_count = 0
        self.repetition_count = {}
        self.last_action = None
        self.repeat_steps = 0
        self.gpt_calls_this_episode = 0
        self.gpt_call_limit = 10
        self.gpt_manager = GPTManager()
        self.replay_buffer = ReplayBuffer(capacity=self.replay_memory_size, alpha=0.6, use_sqlite=True, db_path=f"core/memories/blueagent_memory/replay_buffer.sqlite3")
        self.prioritized_experiences = []
        self.prioritized_priorities = []
        self.mode_switch_cooldown = 0  # Add cooldown for GPT mode switch
        self.target_update_freq = 50  # Update target net every N steps
        self.step_count = 0
        self.local_llm = LocalLLMManager(model_name="wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF")

    def _init_multiagent_links(self):
        self.red = self.agent_manager.get_agent("RedAgent")
        self.orion = self.agent_manager.get_agent("OrionAgent")

    def query_tactical_gpt(self, prompt, complexity="standard"):
        """
        Use both SenecaLLM and GPTManager for defensive/tactical suggestions.
        """
        try:
            seneca_suggestion = self.local_llm.query(prompt)
            review_prompt = (
                f"As a blue team strategist, review the AI's suggested command:\n\n"
                f"Task: {prompt}\n"
                f"Suggestion: {seneca_suggestion}\n\n"
                f"Do you approve this command? If not, refine it. Respond ONLY with the final Linux command."
            )
            final_command = self.gpt_manager.gpt_request(
                review_prompt, task_type="reasoning", model="gpt-4o-mini"
            )
            return self.gpt_manager._sanitize_output(final_command)
        except Exception as e:
            console.print(f"[red]❌ query_tactical_gpt error: {e}[/red]")
            return self.gpt_manager.smart_decision(task_type="defense", task_description=prompt)

    def select_action(self, state_tensor, phase=None):
        # DQN: argmax Q-value from policy_net
        if isinstance(phase, dict) and "response" in phase:
            phase = phase["response"]
        if random.random() < self.epsilon:
            action = random.randint(0, self.policy_net.output_size - 1)
            console.print(f"[yellow]🎲 Random action selected due to exploration: {action}[/yellow]")
            return action
        phase_vec = get_phase_vector(phase, self.device)
        q_values = self.policy_net(state_tensor.unsqueeze(0), phase_vector=phase_vec)
        action = torch.argmax(q_values, dim=-1).item()
        console.print(f"[cyan]🎯 PolicyNet action selected: {action}[/cyan]")
        if self.current_mode != "Defensive" and self.mode_switch_cooldown == 0:
            mode_switch_prompt = f"Current mode: {self.current_mode}. Should BlueAgent switch to a more defensive posture based on the current threat level?"
            mode_switch_decision = self.query_tactical_gpt(mode_switch_prompt)
            if "yes" in mode_switch_decision.lower():
                self.current_mode = "Defensive"
                console.print(f"[magenta]🚨 Mode switched to Defensive.[/magenta]")
            self.mode_switch_cooldown = 5  # Debounce: only ask every 5 steps
        elif self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= 1
        return action

    def simulate_step(self, episode=1, step=1, shared_context=None):
        try:
            state = (
                self.env.reset()
                if step == 1
                else getattr(self, "_last_state", self.env.get_global_state())
            )
            # Use shared context for phase coordination if available
            if shared_context and "ScoutAgent_phase" in shared_context:
                state["phase"] = shared_context["ScoutAgent_phase"]
            if hasattr(self, "agent_manager") and self.agent_manager:
                scout = getattr(self.agent_manager, "scout_agent", None)
                if scout and hasattr(scout, "advise_phase"):
                    state["phase"] = scout.advise_phase(
                        state, self.agent_manager.all_agents()
                    )
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
            # --- Repetition Counter for Stagnation Prevention ---
            self.repetition_count.setdefault(action, 0)
            if self.last_action == action:
                self.repetition_count[action] += 1
                self.repeat_steps += 1
            else:
                self.repetition_count[action] = 1
                self.repeat_steps = 1
            self.last_action = action
            if self.repetition_count[action] >= 3:
                console.print(
                    f"[yellow][Summary] BlueAgent repeated action '{action}' x{self.repetition_count[action]} times — triggering forced exploration.[/yellow]"
                )
                action = str(random.randint(0, self.policy_net.output_size - 1))
                self.repetition_count[action] = 0
                self.repeat_steps = 0
            next_state, reward, done, _ = self.env.step(action)
            # Defensive: ensure reward is a float
            try:
                if isinstance(reward, dict):
                    reward = float(reward.get("reward", 0.0))
                else:
                    reward = float(reward)
            except Exception:
                reward = 0.0
            self.command_history.append(str(action))
            self.stats_monitor.log_step(self.agent_id, reward, command=str(action))
            self._last_state = next_state
            self.last_output = f"Action: {action}, Reward: {reward}, Phase: {state.get('phase', 'N/A')}"
            reasoning_key = f"{action}|{state.get('phase')}"
            if reasoning_key in self.gpt_reasoning_cache:
                self.last_reasoning = self.gpt_reasoning_cache[reasoning_key]
            else:
                self.last_reasoning = self.gpt_handler.query(
                    f"Explain why action {action} is optimal for phase {state.get('phase')}.",
                    model="gpt-4o-mini",
                )
                self.gpt_reasoning_cache[reasoning_key] = self.last_reasoning
            display_status_bar(self.agent_id, episode, step)
            self._log_training_step(episode, step, str(action), state, self.last_output)
            # Print more info about agent state
            console.print(
                f"[dim]Replay buffer: {len(self.prioritized_experiences)} | Epsilon: {self.epsilon:.3f} | Entropy: {self.entropy_beta:.3f}[/dim]"
            )
            # After decision, broadcast own phase/intent
            if self.agent_manager and hasattr(self.agent_manager, "broadcast"):
                self.agent_manager.broadcast(
                    f"{self.agent_id}_phase",
                    state.get("phase", "N/A"),
                    sender=self.agent_id,
                )
            # Activate countermeasures if alert level is high
            if state.get("blue_team_alert", 0) > 60:
                self.current_mode = "Defensive"
                console.print(
                    "[red]🚨 BlueAgent: Alert level high, activating countermeasures![/red]"
                )
                # Example: deploy honeypots or reset credentials
                if hasattr(self.env, "honeypots"):
                    self.env.honeypots.append("fake_ssh")
            # --- Enhanced Threat Detection ---
            # Analyze RedAgent's last action for threat patterns
            red_agent = getattr(self.agent_manager, "red_agent", None)
            red_command = None
            if red_agent and hasattr(red_agent, "command_history") and red_agent.command_history:
                red_command = red_agent.command_history[-1]
                # Use GPT-4o-mini for anomaly detection if high risk
                if "exploit" in red_command or "privesc" in red_command:
                    anomaly_prompt = (
                        f"RedAgent issued: {red_command}. "
                        f"Blue team alert: {state.get('blue_team_alert', 0)}. "
                        "Is this a high-risk offensive action? Suggest a defensive countermeasure."
                    )
                    gpt_defense = self.query_tactical_gpt(anomaly_prompt)
                    if "honeypot" in gpt_defense.lower():
                        if hasattr(self.env, "honeypots"):
                            self.env.honeypots.append("auto_gpt_honeypot")
                        console.print(f"[blue]🛡️ BlueAgent: Deployed GPT-suggested honeypot.[/blue]")
                    if "reset" in gpt_defense.lower():
                        console.print(f"[blue]🔐 BlueAgent: Resetting credentials per GPT defense.[/blue]")
                    console.print(f"[yellow]🧠 GPT-4o-mini Defense: {gpt_defense}[/yellow]")
            # --- Dynamic Defensive Response ---
            if state.get("blue_team_alert", 0) > 60 or state.get("detection_risk", 0) > 5.0:
                # Use GPT-4o-mini for adaptive defense
                defense_prompt = (
                    f"Alert: {state.get('blue_team_alert', 0)}, Risk: {state.get('detection_risk', 0)}. "
                    "Suggest immediate defensive action."
                )
                gpt_action = self.query_tactical_gpt(defense_prompt)
                console.print(f"[cyan]🛡️ Adaptive Defense Triggered: {gpt_action}[/cyan]")
                if "honeypot" in gpt_action.lower():
                    if hasattr(self.env, "honeypots"):
                        self.env.honeypots.append("adaptive_gpt_honeypot")
                if "lockdown" in gpt_action.lower():
                    console.print("[red]🔒 BlueAgent: Initiating system lockdown![/red]")
                self.current_mode = "Defensive"
            # --- Detailed Logging ---
            console.print(
                f"[dim][BlueAgent] Step {step} | Action: {action} | Phase: {state.get('phase', 'N/A')} | Reward: {reward:.2f} | Mode: {self.current_mode}[/dim]"
            )
            gpt_tokens = self.gpt_manager.get_token_usage() if hasattr(self.gpt_manager, 'get_token_usage') else 0
            experience = {
                'state': state_tensor.cpu().tolist(),
                'action': action,
                'reward': reward,
                'next_state': self.encode_env_state(next_state).cpu().tolist(),
                'gpt_tokens': gpt_tokens
            }
            self.replay_buffer.add(experience)
            # At the end of episode, decay epsilon
            if step == 1:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            # --- DQN Target Network Update ---
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
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
            f.write(
                f"Episode {episode}, Step {step}, Action: {action}, Phase: {state.get('phase')}, Output: {output}\n"
            )

    def react_to_action(self, command, parsed_output=None):
        if not isinstance(command, str):
            console.print(
                f"[yellow]⚠ BlueAgent received non-string command: {command}[/yellow]"
            )
            return {}
        if not parsed_output:
            parsed_output = {"phase": "unknown", "success": False, "risk_score": 0.0}
        threat_level = 0.0
        if "exploit" in command.lower() or "privesc" in command.lower():
            threat_level = 0.7
        elif "recon" in command.lower() or "scan" in command.lower():
            threat_level = 0.3
        alert_increase = (
            threat_level * 10 * (1.0 if parsed_output.get("success", False) else 0.5)
        )
        risk_increase = threat_level * 0.5
        honeypots_deployed = []
        credentials_reset = False
        if threat_level > 0.5 and random.random() < 0.3:
            honeypots_deployed = ["fake_ssh", "fake_http"]
            console.print(
                f"[blue]🛡️ {self.agent_id}: Deploying honeypot in response to threat[/blue]"
            )
        if (
            "exploit" in command.lower()
            and parsed_output.get("success", False)
            and random.random() < 0.2
        ):
            credentials_reset = True
            console.print(
                f"[blue]🔐 {self.agent_id}: Resetting credentials after suspected breach[/blue]"
            )
        # Dynamic risk adaptation
        if self.env.detection_risk > 5.0:
            self.current_mode = "Defensive"
            if self.verbosity != "silent":
                console.print(
                    "[red]🔴 High Risk Detected! Switching to Defensive Mode.[/red]"
                )
            # Deploy honeypot if not already present
            if "fake_ssh" not in self.env.honeypots:
                self.env.honeypots.append("fake_ssh")
                if self.verbosity == "verbose":
                    console.print(
                        "[yellow]🛡️ BlueAgent deployed honeypot: fake_ssh[/yellow]"
                    )
        # --- Human-Readable Logging ---
        console.print(
            f"[dim][BlueAgent] Detected threat: {command} | Alert+{alert_increase:.1f} | Risk+{risk_increase:.2f} | Honeypots: {honeypots_deployed} | Credentials Reset: {credentials_reset}[/dim]"
        )
        return {
            "alert_increase": alert_increase,
            "risk_increase": risk_increase,
            "honeypots": honeypots_deployed,
            "credentials_reset": credentials_reset,
        }

    def train_on_batch(self):
        # DQN: Use target_net for stable Q-targets, prioritized replay, ε-decay
        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            console.print(
                "[yellow]⚠ Not enough experiences for batch training.[/yellow]"
            )
            return
        # Prepare tensors
        states = torch.stack([torch.tensor(exp['state'], dtype=torch.float32, device=self.device) for exp in batch])
        actions = torch.tensor(
            [self.select_action(torch.tensor(exp['state'], dtype=torch.float32, device=self.device), None) for exp in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [exp['reward'] for exp in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([torch.tensor(exp['next_state'], dtype=torch.float32, device=self.device) for exp in batch])
        dones = torch.zeros(len(batch), dtype=torch.float32, device=self.device)  # Placeholder

        # --- DQN Q-value computation ---
        # Q(s,a) from policy_net
        q_values = self.policy_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        # Q(s',a') from target_net
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)
        # Loss: MSE or Huber
        loss = torch.nn.functional.smooth_l1_loss(q_selected, targets)
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_net.optimizer.step()
        self.policy_net.scheduler.step()
        self.target_net.eval()
        # --- Epsilon decay after each batch ---
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # --- Target network update ---
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        console.print(f"[cyan]🔧 {self.agent_id}: DQN batch training complete. Loss: {loss.item():.4f}[/cyan]")
        self._log_training_event("Batch training complete.")
        # Prune buffer if needed
        if hasattr(self.replay_buffer, "buffer") and len(self.replay_buffer.buffer) > self.replay_memory_size:
            self.replay_buffer.buffer = self.replay_buffer.buffer[-self.replay_memory_size:]

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
        table.add_row(
            "Replay Buffer",
            f"{len(self.prioritized_experiences)} / {self.replay_memory_size}",
        )
        table.add_row(
            "Last Reasoning",
            self.last_reasoning[:60] + "..." if self.last_reasoning else "N/A",
        )
        console.print(
            Panel(table, title="🧠 Defensive Overview", border_style="bright_blue")
        )
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
        table.add_row(
            "Last Reasoning",
            self.last_reasoning[:80] + "..." if self.last_reasoning else "N/A",
        )
        return Panel(
            table, title=f"🛡️ {self.agent_id} Defensive Panel", border_style="blue"
        )

    def save_models(self, prefix="models/blue_agent"):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        try:
            self.policy_net.save(f"{prefix}_policy.pt")
            self.value_net.save(f"{prefix}_value.pt")
            console.print(
                f"[green]💾 {self.agent_id}: Models saved to {prefix}_*.pt[/green]"
            )
        except Exception as e:
            console.print(f"[red]❌ {self.agent_id}: Model save failed: {e}[/red]")

    def load_models(self, prefix="models/blue_agent"):
        try:
            self.policy_net.load(f"{prefix}_policy.pt")
            self.value_net.load(f"{prefix}_value.pt")
            console.print(
                f"[cyan]✔ {self.agent_id}: Models loaded from {prefix}_*.pt[/cyan]"
            )
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
        vec.append(
            float(
                state.get("phase", 0)
                in ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
            )
        )
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
        return self.query_tactical_gpt(
            "Suggest a defensive command for the current phase."
        )

    def execute_command(self, command):
        try:
            output = f"Executed: {command}"
            reward = random.uniform(0, 10)
            self.stats_monitor.log_step(self.agent_id, reward, command=command)
            return {
                "output": output,
                "recommendations": [
                    {"command": command, "params": "Auto", "why": "Manual execution"}
                ],
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
            "nmap",
            "hydra",
            "msfconsole",
            "sqlmap",
            "ffuf",
            "gobuster",
            "linpeas",
            "winpeas",
            "evil-winrm",
            "masscan",
            "amass",
            "crackmapexec",
            "enum4linux",
            "pspy",
        ]

    def sync_memory(self):
        if self.memory_router:
            self.memory_router.sync_global_insights()
        from core.logic.redundancy_detector import detect_redundancy_batch
        self.replay_buffer.prune_redundancy(lambda cmds: detect_redundancy_batch(cmds))

    def reset(self):
        """Reset stats and replay buffer for new episode."""
        self.total_steps = 0
        self.command_history.clear()
        self.prioritized_experiences.clear()
        self.prioritized_priorities.clear()
        if hasattr(self, "stats_monitor"):
            self.stats_monitor.reset()

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
