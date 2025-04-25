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
        self.policy_net = PolicyNet(input_size=512, output_size=5, device=self.device).to(self.device)
        self.value_net = ValueNet(input_size=512, device=self.device).to(self.device)
        self.target_value_net = ValueNet(input_size=512, device=self.device).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_value_net.eval()
        self.target_net = PolicyNet(input_size=512, output_size=5, device=self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.env = CyberEnvironment(agent_manager=agent_manager, defer_reset=True)
        self.stats_monitor = StatsMonitor()
        self.teacher = TeachModule(agent_name=self.agent_id)
        self.memory_router = memory_router or MemoryRouter()
        self.agent_manager = agent_manager
        self.memory_manager = memory_manager
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
        self.gpt_reasoning_cache = {}
        self.gpt_handler = GPTCacheHandler()
        self.verbosity = verbosity
        self.last_action = None
        self.repetition_count = {}
        self.repeat_steps = 0
        self.gpt_calls_this_episode = 0
        self.gpt_call_limit = 10
        self.gpt_manager = GPTManager()
        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_memory_size,
            alpha=0.6,
            use_sqlite=True,
            db_path=f"core/memories/blueagent_memory/replay_buffer.sqlite3"
        )
        self.prioritized_experiences = []
        self.prioritized_priorities = []
        self.mode_switch_cooldown = 0
        self.target_update_freq = 1000  # More stable: update every 1000 steps
        self.step_count = 0
        self.local_llm = LocalLLMManager(model_name="wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF")
        self.lily_llm = LocalLLMManager(model_name="QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0")

    def _init_multiagent_links(self):
        self.red = self.agent_manager.get_agent("RedAgent")
        self.orion = self.agent_manager.get_agent("OrionAgent")
        self.local_llm = LocalLLMManager(model_name="wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF")

    def query_tactical_gpt(self, prompt, complexity="standard"):
        """
        Use LilyLLM for concise tactical advice, fallback to SenecaLLM+GPT review.
        """
        try:
            # LilyLLM prompt template
            lily_prompt = (
                "Provide a concise tactical recommendation in one sentence. "
                "Avoid any self-referential commentary.\n"
                f"{prompt.strip()}"
            )
            lily_suggestion = self.lily_llm.query(lily_prompt)
            lily_suggestion = self._postprocess_lily_output(lily_suggestion)
            if self.gpt_manager._is_simple_command(lily_suggestion):
                return lily_suggestion
            # Fallback: SenecaLLM + GPT review
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

    def _postprocess_lily_output(self, output: str) -> str:
        """
        Remove verbose/AI disclaimers from LilyLLM output.
        """
        import re
        if not output:
            return ""
        patterns = [
            r"(?i)^as an ai( language)? model[,. ]*",
            r"(?i)^as a (cybersecurity )?ai( assistant)?[,. ]*",
            r"(?i)^i am (an|a) (ai|language model)[,. ]*",
            r"(?i)^note:.*",
            r"(?i)^please note.*",
        ]
        for pat in patterns:
            output = re.sub(pat, "", output).strip()
        output = re.sub(r"(?i)for more information.*$", "", output).strip()
        return output

    def select_action(self, state_tensor, phase=None):
        if isinstance(phase, dict) and "response" in phase:
            phase = phase["response"]
        # Use get_action_description for readable logs
        action_idx = None
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.policy_net.output_size - 1)
            if self.verbosity in ("debug", "verbose"):
                console.print(f"[yellow]🎲 Random action selected: {get_action_description(action_idx)}[/yellow]")
            return action_idx
        phase_vec = get_phase_vector(phase, self.device)
        q_values = self.policy_net(state_tensor.unsqueeze(0), phase_vector=phase_vec)
        action_idx = torch.argmax(q_values, dim=-1).item()
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[cyan]🎯 PolicyNet action selected: {get_action_description(action_idx)}[/cyan]")
        if self.current_mode != "Defensive" and self.mode_switch_cooldown == 0:
            mode_switch_prompt = f"Current mode: {self.current_mode}. Should BlueAgent switch to a more defensive posture based on the current threat level?"
            mode_switch_decision = self.query_tactical_gpt(mode_switch_prompt)
            if "yes" in mode_switch_decision.lower():
                self.current_mode = "Defensive"
                console.print(f"[magenta]🚨 Mode switched to Defensive.[/magenta]")
            self.mode_switch_cooldown = 5
        elif self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= 1
        return action_idx

    def simulate_step(self, episode=1, step=1, shared_context=None):
        try:
            state = (
                self.env.reset()
                if step == 1
                else getattr(self, "_last_state", self.env.get_global_state())
            )
            # Inject ScoutAgent_phase if available
            if shared_context and "ScoutAgent_phase" in shared_context:
                state["phase"] = shared_context["ScoutAgent_phase"]
            if hasattr(self, "agent_manager") and self.agent_manager:
                scout = getattr(self.agent_manager, "scout_agent", None)
                if scout and hasattr(scout, "advise_phase"):
                    state["phase"] = scout.advise_phase(state, self.agent_manager.all_agents())
            try:
                state_tensor = self.encode_env_state(state)
            except Exception:
                state_tensor = torch.zeros(512, device=self.device)
            action_idx = self.select_action(state_tensor, state.get("phase", None))
            action = get_action_description(action_idx)
            if isinstance(action, dict):
                action = action.get("response", str(action))
            if not isinstance(action, str):
                action = str(action)
            if not action or action == "N/A":
                action = "nmap -p- -sC -sV TARGET"
                console.print(f"[yellow]⚠ BlueAgent fallback to default action: {action}[/yellow]")
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
            console.print(
                f"[dim]Replay buffer: {len(self.prioritized_experiences)} | Epsilon: {self.epsilon:.3f} | Entropy: {self.entropy_beta:.3f}[/dim]"
            )
            if self.agent_manager and hasattr(self.agent_manager, "broadcast"):
                self.agent_manager.broadcast(
                    f"{self.agent_id}_phase",
                    state.get("phase", "N/A"),
                    sender=self.agent_id,
                )
            if state.get("blue_team_alert", 0) > 60:
                self.current_mode = "Defensive"
                console.print(
                    "[red]🚨 BlueAgent: Alert level high, activating countermeasures![/red]"
                )
                if hasattr(self.env, "honeypots"):
                    self.env.honeypots.append("fake_ssh")
            red_agent = getattr(self.agent_manager, "red_agent", None)
            red_command = None
            if red_agent and hasattr(red_agent, "command_history") and red_agent.command_history:
                red_command = red_agent.command_history[-1]
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
            if state.get("blue_team_alert", 0) > 60 or state.get("detection_risk", 0) > 5.0:
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
            if step == 1:
                old_epsilon = self.epsilon
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                if self.verbosity in ("debug", "verbose"):
                    console.print(f"[cyan]🔧 Epsilon decayed: {old_epsilon:.4f} → {self.epsilon:.4f}[/cyan]")
            self.step_count += 1
            # --- Target network update every target_update_freq steps ---
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            # Log epsilon for monitoring
            if (step == 1 or step % 10 == 0) and self.verbosity in ("debug", "verbose"):
                console.print(f"[cyan]Epsilon: {self.epsilon:.4f}[/cyan]")
            # LLM usage summary (stub, see below)
            if step == 1 or step % 10 == 0:
                gpt_calls = self.gpt_manager.get_token_usage(self.agent_id)
                console.print(f"[blue]LLM usage: Seneca calls: N/A, Lily calls: N/A, GPT calls: {gpt_calls}[/blue]")
            # --- Stuck-Agent Detection ---
            stuck_window = 3
            stuck = False
            if len(self.command_history) >= stuck_window:
                recent_cmds = self.command_history[-stuck_window:]
                if len(set(recent_cmds)) < stuck_window:
                    stuck = True
            # No-progress detection: if reward hasn't increased for K steps
            no_progress_steps = 4
            if hasattr(self, "last_rewards"):
                self.last_rewards.append(self.stats_monitor.agent_stats[self.agent_id]["rewards"][-1] if self.stats_monitor.agent_stats[self.agent_id]["rewards"] else 0)
                if len(self.last_rewards) > no_progress_steps:
                    self.last_rewards.pop(0)
                if (
                    len(self.last_rewards) == no_progress_steps
                    and max(self.last_rewards) - min(self.last_rewards) < 1e-3
                ):
                    stuck = True
            else:
                self.last_rewards = [self.stats_monitor.agent_stats[self.agent_id]["rewards"][-1] if self.stats_monitor.agent_stats[self.agent_id]["rewards"] else 0]
            if stuck:
                console.print(f"[yellow]⚠ {self.agent_id} appears stuck. Triggering GPT-based recovery.[/yellow]")
                recovery_prompt = (
                    f"Agent {self.agent_id} has repeated action '{self.last_action}' {stuck_window} times with no success. "
                    f"Recent rewards: {getattr(self, 'last_rewards', [])}. "
                    "Suggest an alternative defensive strategy or command to escape this local minimum. Respond ONLY with the command."
                )
                recovery_cmd = self.gpt_manager.gpt_request(
                    recovery_prompt, task_type="reasoning", agent_id=self.agent_id, use_gpt=True
                )
                if recovery_cmd and isinstance(recovery_cmd, str) and len(recovery_cmd.split()) > 1:
                    action = recovery_cmd
                    console.print(f"[green]🧠 GPT Recovery Command Applied: {action}[/green]")
                    self.repetition_count[action] = 0
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

    def train_on_batch(self):
        # DQN: Use target_value_net for stable Q-targets, prioritized replay, ε-decay
        batch = self.replay_buffer.sample(self.batch_size, prioritized=False)  # Use uniform random sampling
        if not batch:
            console.print(
                "[yellow]⚠ Not enough experiences for batch training.[/yellow]"
            )
            return
        # --- Use stored actions, not select_action (fixes DQN bug) ---
        states = torch.stack([torch.tensor(exp['state'], dtype=torch.float32, device=self.device) for exp in batch])
        actions = torch.tensor(
            [exp['action'] if isinstance(exp['action'], int) else 0 for exp in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [exp['reward'] for exp in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([torch.tensor(exp['next_state'], dtype=torch.float32, device=self.device) for exp in batch])
        dones = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        # Q(s,a) from policy_net
        q_values = self.policy_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        # --- Double DQN (optional, scaffolded) ---
        # next_actions = self.policy_net(next_states).argmax(dim=1)
        # next_q_values = self.target_value_net(next_states)
        # max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
        # --- Standard DQN target ---
        with torch.no_grad():
            next_q_values = self.target_value_net(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)
        loss = torch.nn.functional.smooth_l1_loss(q_selected, targets)
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_net.optimizer.step()
        self.policy_net.scheduler.step()
        self.target_value_net.eval()
        # --- Target value net update ---
        # Only update every target_update_freq steps (handled in simulate_step)
        # --- Epsilon decay after each batch ---
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[cyan]🔧 {self.agent_id}: DQN batch training complete. Loss: {loss.item():.4f} | Epsilon: {old_epsilon:.4f}→{self.epsilon:.4f}[/cyan]")
        self._log_training_event(f"Batch training complete. Loss: {loss.item():.4f}")
        # --- Moving average reward/loss logging ---
        if not hasattr(self, "loss_history"):
            self.loss_history = []
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        avg_loss = sum(self.loss_history) / len(self.loss_history)
        if self.verbosity in ("debug", "verbose"):
            console.print(f"[magenta]Moving Avg Loss (last 100): {avg_loss:.4f}[/magenta]")
        # Prune buffer if needed
        if hasattr(self.replay_buffer, "buffer") and len(self.replay_buffer.buffer) > self.replay_memory_size:
            self.replay_buffer.buffer = self.replay_buffer.buffer[-self.replay_memory_size:]
        # Prune redundancy after each batch
        from core.logic.redundancy_detector import detect_redundancy_batch
        self.replay_buffer.prune_redundancy(lambda cmds: detect_redundancy_batch(cmds))

    def _log_training_event(self, msg):
        with open(self.training_log_path, "a") as f:
            f.write(f"{msg}\n")

    @staticmethod
    def encode_env_state_static(state, device):
        import numpy as np
        vec = []
        # Add phase one-hot vector (5 dims)
        phases = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
        phase_vec = [1.0 if state.get("phase") == p else 0.0 for p in phases]
        vec.extend(phase_vec)
        # Add last action index (if available)
        last_action_idx = 0
        if "last_action" in state:
            try:
                last_action_idx = int(state["last_action"])
            except Exception:
                last_action_idx = 0
        vec.append(float(last_action_idx))
        # LLM context features: last reasoning reward, chain updated flag
        llm_reward = state.get("llm_last_reward", 0.0)
        chain_updated = 1.0 if state.get("chain_updated", False) else 0.0
        vec.append(float(llm_reward))
        vec.append(float(chain_updated))
        # Pad with zeros for future LLM/context features
        vec.extend([0.0, 0.0])
        # Add other environment features
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
        # Ensure phase and last action are included
        state = dict(state)
        if hasattr(self, "last_action") and self.last_action is not None:
            state["last_action"] = self.last_action
        return self.encode_env_state_static(state, self.device)

    def sync_memory(self):
        if self.memory_router:
            self.memory_router.sync_global_insights()
        from core.logic.redundancy_detector import detect_redundancy_batch
        self.replay_buffer.prune_redundancy(lambda cmds: detect_redundancy_batch(cmds))

    def get_smart_command(self, state, phase, cache_key=None):
        """
        Use both SenecaLLM and GPTManager for command generation.
        Integrate dual-LLM critique loop for defensive strategy.
        """
        task_desc = (
            f"Phase: {phase}, Privilege: {state.get('privilege_level')}, "
            f"Alert: {state.get('blue_team_alert')}, Risk: {state.get('detection_risk')}."
            " Suggest the most effective, non-redundant, phase-appropriate defensive command."
        )
        dual_feedback = self.gpt_manager.dual_llm_feedback(task_desc, agent_id=self.agent_id)
        command = self.gpt_manager._sanitize_output(dual_feedback)
        # TODO: integrate dual-LLM critique loop with agent feedback/memory
        if hasattr(self, "gpt_reasoning_cache"):
            self.gpt_reasoning_cache[f"dual_llm_{phase}"] = dual_feedback
        return command

    def reset(self):
        self.total_steps = 0
        self.command_history.clear()
        self.prioritized_experiences.clear()
        self.prioritized_priorities.clear()
        if hasattr(self, "stats_monitor"):
            self.stats_monitor.reset()
        if hasattr(self, "gpt_reasoning_cache") and "dual_llm_reset" in self.gpt_reasoning_cache:
            feedback = self.gpt_reasoning_cache["dual_llm_reset"]
            # Log or use as needed for strategy adaptation

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
