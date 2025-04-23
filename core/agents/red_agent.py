# core/agents/red_agent.py — ARIASKA RedAgent v11.2 Apex++
# 🦂 Offensive Commander | GPT-4o-mini Tactical Core | Multi-Agent Synergy | Visual Intelligence | Entropy-Aware Decision AI

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
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.utils.replay_buffer import ReplayBuffer
from core.utils.gpt_cache_handler import GPTCacheHandler
from core.interfaces.agent_interface import AgentInterface


def safe_tensor(data, device):
    # Only convert if data is already a tensor, list, or np array
    import numpy as np

    if isinstance(data, torch.Tensor):
        return data.clone().detach().to(device)
    elif isinstance(data, (list, tuple, np.ndarray)):
        return (
            torch.as_tensor(data, dtype=torch.float32, device=device).clone().detach()
        )
    elif isinstance(data, dict):
        # Use encode_env_state to flatten dicts
        return RedAgent.encode_env_state_static(data, device)
    else:
        raise TypeError(f"Cannot convert type {type(data)} to tensor.")


console = Console()


class RedAgent(AgentInterface, MemorySyncInterface):
    def __init__(
        self,
        agent_id="RedAgent",
        role="CyberOffense",
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
        self.value_net = ValueNet(input_size=512, device=self.device).to(self.device)
        self.env = CyberEnvironment(agent_manager=agent_manager, defer_reset=True)
        self.env.blue_team = True
        self.stats_monitor = StatsMonitor()
        self.teacher = TeachModule(agent_name=self.agent_id)
        self.memory_router = memory_router
        self.agent_manager = agent_manager
        self.memory_manager = memory_manager
        self.scout = None
        self.shadow = None
        self.blue = None
        self.orion = None
        self.replay_memory_size = 1500
        self.batch_size = 40
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.994
        self.entropy_beta = 0.012
        self.redundancy_penalty = 0.75
        self.total_steps = 0
        self.total_episodes = 0
        self.prioritized_experiences = []
        self.prioritized_priorities = []
        self.command_history = []
        self.last_reasoning = ""
        self.current_mode = "Balanced"
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs("logs", exist_ok=True)
        self.gpt_reasoning_cache = {}
        self.replay_buffer = ReplayBuffer(capacity=self.replay_memory_size)
        self.gpt_handler = GPTCacheHandler()
        self.verbosity = verbosity
        self.gpt_calls = {"4o-mini": 0, "4.1-full": 0}
        self.redundancy_counter = 0
        self.no_reward_steps = 0
        self.last_reward = 0
        self.repeated_action_count = 0
        self.last_action = None
        self.repetition_count = {}
        self.last_action = None
        self.repeat_steps = 0
        self.gpt_calls_this_episode = 0
        self.gpt_call_limit = 10
        self.novelty_score = 0
        self.stagnation_window = []
        self.stagnation_threshold = 0.5
        console.print(
            f"[green]✔ {self.agent_id} initialized — Apex++ Mode on {self.device}[/green]"
        )

    def _init_multiagent_links(self):
        if self.agent_manager:
            self.scout = self.agent_manager.get_agent("ScoutAgent")
            self.shadow = self.agent_manager.get_agent("ShadowAgent")
            self.blue = self.agent_manager.get_agent("BlueAgent")
            self.orion = self.agent_manager.get_agent("OrionAgent")

    def query_tactical_gpt(self, prompt, complexity="standard"):
        # Use GPT-4.1 for 'full', else GPT-4o-mini
        if self.gpt_calls_this_episode >= self.gpt_call_limit:
            return "Maintain current strategy."
        model = "gpt-4.1" if complexity == "full" else "gpt-4o-mini"
        try:
            result = subprocess.run(
                ["sgpt", "--model", model, "--role", "aria", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20 if model == "gpt-4.1" else 10,
                text=True,
            )
            response = result.stdout.strip()
            if model == "gpt-4.1":
                self.gpt_calls["4.1-full"] += 1
            else:
                self.gpt_calls["4o-mini"] += 1
            self.gpt_calls_this_episode += 1
            console.print(f"[magenta]🧠 GPT Tactical Insight:[/magenta] {response}")
            # Track token usage for visualization
            if hasattr(self.stats_monitor, "log_gpt_call"):
                self.stats_monitor.log_gpt_call(self.agent_id)
            return response
        except Exception as e:
            console.print(f"[red]⚠ GPT query failed: {e}[/red]")
            return "Maintain current strategy."

    def get_smart_command(self, state, phase, cache_key=None):
        """
        Modular smart command generator using GPT (with cache).
        Returns (command, reasoning).
        """
        # 1. Try cache first
        if cache_key and hasattr(self.memory_router, "check_gpt_cache"):
            cached_cmd = self.memory_router.check_gpt_cache(cache_key)
            if isinstance(cached_cmd, dict) and "response" in cached_cmd:
                command = cached_cmd["response"]
            else:
                command = cached_cmd
            if command and isinstance(command, str) and len(command.split()) > 1:
                gpt_reason = self.memory_router.check_gpt_cache(
                    f"{self.agent_id}_reason_{command}"
                )
                if isinstance(gpt_reason, dict) and "response" in gpt_reason:
                    gpt_reason = gpt_reason["response"]
                return command, gpt_reason

        # 2. Use GPT for smart suggestion (with robust fallbacks)
        try:
            prompt = (
                f"You are ARIASKA's offensive strategist (role: aria). "
                f"Phase: {phase}, Privilege: {state.get('privilege_level')}, "
                f"Ports: {state.get('open_ports')}, Alert: {state.get('blue_team_alert')}, "
                f"Risk: {state.get('detection_risk')}. "
                "Suggest the most effective, non-redundant, phase-appropriate offensive command for this phase. "
                "Avoid trivial or repeated commands. Respond with command only."
            )
            command = self.query_tactical_gpt(prompt)

            # Verify command is valid
            if not command or not isinstance(command, str) or len(command.split()) < 2:
                # Use phase-based fallback commands rather than generic echo
                phase_commands = {
                    "recon": "nmap -sS -sV -p- -T4 --min-rate=1000 10.10.10.10",
                    "enumeration": "gobuster dir -u http://10.10.10.10 -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt",
                    "exploit": "searchsploit apache 2.4.49",
                    "privesc": "sudo -l",
                    "exfiltrate": "tar -czf /tmp/data.tar.gz /home/user/Documents",
                }
                command = phase_commands.get(phase, f"nmap -A 10.10.10.10")
                console.print(
                    f"[yellow]⚠ GPT failed to generate command. Using fallback for {phase}.[/yellow]"
                )

            if hasattr(self.memory_router, "store_gpt_response") and cache_key:
                self.memory_router.store_gpt_response(self.agent_id, cache_key, command)

            # 3. Generate reasoning via GPT (with fallback)
            reason_prompt = f"Explain in one sentence why '{command}' is optimal for phase '{phase}'."
            gpt_reason = self.query_tactical_gpt(reason_prompt)

            if not gpt_reason or not isinstance(gpt_reason, str) or len(gpt_reason) < 5:
                gpt_reason = f"This {command.split()[0]} command is appropriate for the {phase} phase."

            if hasattr(self.memory_router, "store_gpt_response") and command:
                self.memory_router.store_gpt_response(
                    self.agent_id, f"{self.agent_id}_reason_{command}", gpt_reason
                )

            return command, gpt_reason

        except Exception as e:
            # 4. Exception handler with robust fallbacks
            console.print(
                f"[red]❌ Error in get_smart_command: {e}. Using phase-based fallback.[/red]"
            )
            fallback_commands = {
                "recon": "nmap -sS -sV 10.10.10.10",
                "enumeration": "gobuster dir -u http://10.10.10.10 -w /usr/share/wordlists/common.txt",
                "exploit": "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://10.10.10.10",
                "privesc": "find / -perm -u=s -type f 2>/dev/null",
                "exfiltrate": "zip -r /tmp/data.zip /etc/passwd",
            }
            command = fallback_commands.get(phase, "echo 'Fallback command executed'")
            gpt_reason = f"Fallback command for {phase} phase after GPT error."
            return command, gpt_reason

    @staticmethod
    def encode_env_state_static(state, device):
        # Flatten environment dict into a numeric vector for RL input
        # This is a simple, robust encoding for RL state
        import numpy as np

        vec = []
        # Example: encode some common fields, pad to 512
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
        # Encode open_ports and services as counts
        vec.append(float(len(state.get("open_ports", []))))
        vec.append(float(len(state.get("services", []))))
        # Pad to 512
        while len(vec) < 512:
            vec.append(0.0)
        return torch.tensor(vec, dtype=torch.float32, device=device)

    def encode_env_state(self, state):
        return self.encode_env_state_static(state, self.device)

    def simulate_step(self, episode=1, step=1, shared_context=None):
        from core.logic.redundancy_detector import (
            detect_redundancy,
            suggest_alternative,
        )
        from core.logic.rule_engine import rule_based_selection
        from core.logic.output_interpreter import analyze_output

        try:
            state = (
                self.env.reset()
                if step == 1
                else getattr(self, "_last_state", self.env.get_global_state())
            )
            try:
                if self.scout and self.agent_manager:
                    state["phase"] = self.scout.advise_phase(
                        state, self.agent_manager.all_agents()
                    ) or state.get("phase", "recon")
                    if (
                        isinstance(state["phase"], dict)
                        and "response" in state["phase"]
                    ):
                        state["phase"] = state["phase"]["response"]
                else:
                    state["phase"] = state.get("phase", "recon")
            except Exception as e:
                console.print(
                    f"[yellow]⚠ Scout phase advice failed: {e}, using current phase[/yellow]"
                )
                state["phase"] = state.get("phase", "recon")

            # Use tensor-safe code
            state_tensor = self.encode_env_state(state)

            # Use shared context for phase coordination if available
            if shared_context and "ScoutAgent_phase" in shared_context:
                state["phase"] = shared_context["ScoutAgent_phase"]

            # Curriculum-driven exploration: occasionally randomize phase
            if random.random() < 0.05:
                state["phase"] = random.choice(
                    ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
                )

            # Validate GPT phase response
            if self.scout and self.agent_manager:
                phase_suggestion = self.scout.advise_phase(
                    state, self.agent_manager.all_agents()
                )
                if phase_suggestion not in [
                    "recon",
                    "enumeration",
                    "exploit",
                    "privesc",
                    "exfiltrate",
                ]:
                    phase_suggestion = state.get("phase", "recon")
                state["phase"] = phase_suggestion

            # Accept OrionAgent's strategic chain at episode start
            if step == 1 and hasattr(self.agent_manager, "orion_agent"):
                chain = getattr(self.agent_manager.orion_agent, "current_chain", None)
                if chain:
                    self.priority_queue = chain  # Inject chain for this episode
                    if self.verbosity != "quiet":
                        console.print(
                            f"[blue][Orion] Injected strategic chain for episode {episode}[/blue]"
                        )

            # --- Decision Pipeline ---
            phase = state["phase"]
            cache_key = f"{self.agent_id}_decision_{phase}"

            # Track if fallback is triggered
            fallback_triggered = False

            # --- Always define prompt before use ---
            prompt = (
                f"You are ARIASKA's offensive strategist (role: aria). "
                f"Phase: {phase}, Privilege: {state.get('privilege_level')}, "
                f"Ports: {state.get('open_ports')}, Alert: {state.get('blue_team_alert')}, "
                f"Risk: {state.get('detection_risk')}. "
                "Suggest the most effective, non-redundant, phase-appropriate offensive command for this phase. "
                "Avoid trivial or repeated commands. Respond with command only."
            )

            command, gpt_reason = self.get_smart_command(
                state, phase, cache_key=cache_key
            )
            if not command or not isinstance(command, str) or command.strip() == "":
                command = f"echo 'Fallback_Command_{phase}'"
                gpt_reason = "Fallback: GPT unavailable."
                fallback_triggered = True

            # --- Repetition Counter for Stagnation Prevention ---
            self.repetition_count.setdefault(command, 0)
            if self.last_action == command:
                self.repetition_count[command] += 1
                self.repeat_steps += 1
            else:
                self.repetition_count[command] = 1
                self.repeat_steps = 1
            self.last_action = command

            # Anti-redundancy: after 2 repeats, force GPT phase shift or alternative
            if self.repetition_count[command] >= 3:
                if self.gpt_calls_this_episode < self.gpt_call_limit:
                    alt_prompt = (
                        "You are ARIASKA's offensive strategist. "
                        f"Detected repeated command '{command}' x{self.repetition_count[command]}. "
                        "Suggest a novel, phase-shifting tactic (e.g., pivot to exploitation or lateral movement)."
                    )
                    command = self.query_tactical_gpt(alt_prompt, complexity="full")
                    self.gpt_calls_this_episode += 1
                else:
                    # Fallback to cached or heuristic
                    command = self._heuristic_phase_shift(command)
                self.repetition_count[command] = 0
                self.repeat_steps = 0

            # Dynamic risk-aware strategy
            if (
                state.get("blue_team_alert", 0) > 10
                or state.get("detection_risk", 0) > 2.0
            ):
                stealth_prompt = "Alert or risk is high. Suggest a stealthy, low-noise command for this phase."
                if self.gpt_calls_this_episode < self.gpt_call_limit:
                    command = self.query_tactical_gpt(
                        stealth_prompt, complexity="standard"
                    )
                    self.gpt_calls_this_episode += 1
                else:
                    command = "sleep 2 && nmap -T1 --top-ports 10 10.10.10.10"

            # GPT call optimization
            if self.gpt_calls_this_episode >= self.gpt_call_limit:
                # Use cache or fallback
                command = self.memory_router.check_gpt_cache(cache_key) or command

            # Command novelty score
            if command not in self.command_history[-10:]:
                self.novelty_score += 1

            # Stagnation detector
            self.stagnation_window.append(shaped_reward)
            if len(self.stagnation_window) > 5:
                self.stagnation_window.pop(0)
            if (
                len(self.stagnation_window) == 5
                and max(self.stagnation_window) - min(self.stagnation_window)
                < self.stagnation_threshold
            ):
                if self.gpt_calls_this_episode < self.gpt_call_limit:
                    command = self.query_tactical_gpt(
                        "Average reward plateau detected. Suggest a novel, high-reward command.",
                        complexity="full",
                    )
                    self.gpt_calls_this_episode += 1

            self.command_history.append(command)
            output = self.extract_output(command)
            if not isinstance(output, str):
                output = str(output)
            parsed = analyze_output(command, output)
            interpreted_context = {
                **state,
                "phase": parsed.get("phase", state["phase"]),
                "artifacts": parsed.get("artifacts", []),
                "stealth": parsed.get("success", False),
                "honeypot_triggered": "fake_" in output,
                "port_lockdown": len(state.get("open_ports", [])) <= 2,
            }
            if self.blue:
                self.blue.react_to_action(command, parsed)
            next_state, env_reward, done, info = self.env.step(command)
            # Defensive: ensure env_reward is a float
            try:
                if isinstance(env_reward, dict):
                    env_reward = env_reward.get("reward", 0.0)
                env_reward = float(env_reward)
            except Exception:
                env_reward = 0.0
            shaped_reward = self.calculate_reward(
                env_reward, parsed, command, None, detect_redundancy
            )
            try:
                shaped_reward = float(shaped_reward)
            except Exception:
                shaped_reward = 0.0
            if detect_redundancy(self.command_history[:-1], command):
                shaped_reward *= self.redundancy_penalty

            # Redundancy detection
            if self.command_history:
                if self.command_history[-1] == self.last_action:
                    self.repeated_action_count += 1
                else:
                    self.repeated_action_count = 1
                self.last_action = self.command_history[-1]
            else:
                self.repeated_action_count = 1
                self.last_action = command

            # Escalate to GPT-4.1 if same action repeats ≥ 3 times
            if self.repeated_action_count >= 3:
                if self.verbosity in ("standard", "detailed"):
                    console.print(
                        f"[red][RedAgent] 🔺 Escalating to GPT-4.1 due to repetitive actions: '{command}' x{self.repeated_action_count}[/red]"
                    )
                prompt = (
                    f"You are ARIASKA's offensive strategist (role: aria). "
                    f"Stuck in repeated actions: {command}. "
                    f"Suggest a novel, phase-appropriate offensive command for state: {state}. "
                    "Avoid any command similar to the last 5. Respond with command only."
                )
                command = self.query_tactical_gpt(prompt, complexity="full")
                self.repeated_action_count = 0  # Reset after escalation

            self.last_reasoning = gpt_reason
            self.update_memory_and_teach(
                command,
                shaped_reward,
                interpreted_context,
                parsed,
                state_tensor,
                next_state,
                done,
            )
            self.stats_monitor.log_step(self.agent_id, shaped_reward, command=command)
            self._last_state = next_state
            self.last_output = output
            display_status_bar(self.agent_id, episode, step)
            self._log_training_step(episode, step, command, state, output)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            # Print concise, informative summary for this step
            if self.verbosity == "silent":
                pass
            elif self.verbosity == "standard":
                console.log(
                    f"[{self.agent_id}] Step {step} | Reward: {shaped_reward:.2f} | Risk: {state.get('detection_risk', 0):.2f}"
                )
            elif self.verbosity == "verbose":
                console.print(
                    f"[bold magenta]🎯 RedAgent: Step {step} | Phase: {phase} | Command: {command}[/bold magenta]"
                )
                console.print(f"[cyan]🧠 Reasoning:[/cyan] {self.last_reasoning}")
            # Per-step compact panel logging
            if self.verbosity == "detailed":
                from rich.panel import Panel
                from rich.table import Table

                table = Table.grid()
                table.add_row("Action", str(command))
                table.add_row("Phase", str(phase))
                table.add_row("Reward", f"{shaped_reward:+.2f}")
                table.add_row("GPT", "4.1" if fallback_triggered else "4o-mini")
                console.print(
                    Panel(
                        table, title=f"{self.agent_id} Step {step}", border_style="red"
                    )
                )
            elif self.verbosity == "standard":
                if self.repeated_action_count > 1:
                    console.print(
                        f"[{self.agent_id}] Step {step} | Action: {command} (repeated x{self.repeated_action_count}) | Reward: {shaped_reward:.2f}"
                    )
                elif step == 1 or step % 5 == 0:
                    console.print(
                        f"[{self.agent_id}] Step {step} | Action: {command} | Phase: {phase} | Reward: {shaped_reward:.2f}"
                    )
            elif self.verbosity == "quiet":
                if (
                    fallback_triggered
                    or shaped_reward < 0
                    or step == 1
                    or step % 10 == 0
                ):
                    console.print(
                        f"[{self.agent_id}] 🚨 Step {step} | Reward: {shaped_reward:.2f}"
                    )

            # Summarize every 5 steps
            if step % 5 == 0 and self.verbosity != "quiet":
                avg_reward = (
                    sum(self.stats_monitor.agent_stats[self.agent_id]["rewards"][-5:])
                    / 5
                )
                alert = state.get("blue_team_alert", 0)
                console.print(
                    f"[Summary] {self.agent_id} Steps {step-4}-{step}: Avg Reward {avg_reward:+.1f} | Alert {alert:.1f}"
                )
            console.print(
                f"[dim]Replay buffer: {len(self.prioritized_experiences)} | Epsilon: {self.epsilon:.3f} | Entropy: {self.entropy_beta:.3f}[/dim]"
            )

            # Track token usage for visualization
            if hasattr(self.stats_monitor, "log_gpt_call"):
                self.stats_monitor.log_gpt_call(self.agent_id)

            # After decision, broadcast own phase/intent
            if self.agent_manager and hasattr(self.agent_manager, "broadcast"):
                self.agent_manager.broadcast(
                    f"{self.agent_id}_phase", state["phase"], sender=self.agent_id
                )

            # Add to replay buffer
            action = self.select_action(state_tensor, state["phase"])
            next_state_tensor = self.encode_env_state(next_state)
            self.replay_buffer.add(
                {
                    "state": state_tensor,
                    "action": action,
                    "reward": shaped_reward,
                    "next_state": next_state_tensor,
                    "done": done,
                    "command": command,
                }
            )

            self.last_reward = shaped_reward

            # --- Add: Print stats after each step ---
            if hasattr(self.stats_monitor, "show"):
                if step % 10 == 0 or step == 1:
                    self.stats_monitor.show()

            # Always return floats for reward, epsilon, entropy
            return {
                "command": command,
                "phase": state.get("phase", "N/A"),
                "reward": float(shaped_reward),
                "gpt_calls": (
                    self.stats_monitor.agent_stats[self.agent_id]["gpt_calls"]
                    if self.agent_id in self.stats_monitor.agent_stats
                    else 0
                ),
                "output": output,
                "reasoning": self.last_reasoning,
                "step": step,
                "episode": episode,
                "agent_id": self.agent_id,
                "replay_buffer": len(self.prioritized_experiences),
                "epsilon": float(self.epsilon),
                "entropy_beta": float(self.entropy_beta),
            }
        except Exception as e:
            console.print(f"[red]❌ Error in RedAgent simulate_step: {e}[/red]")
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

    def _log_training_step(self, episode, step, command, state, output):
        with open(self.training_log_path, "a") as f:
            f.write(
                f"Episode {episode}, Step {step}, Command: {command}, Phase: {state.get('phase')}, Output: {output}\n"
            )

    def calculate_reward(
        self, env_reward, parsed, command, cmd_data, detect_redundancy=None
    ):
        stealth_score = parsed.get("stealth_score", 0.0)
        risk_score = parsed.get("risk_score", 0.0)
        honeypot_penalty = -20.0 if parsed.get("honeypot_triggered") else 0.0
        base_shaping = (stealth_score * 2.5) - (risk_score * 0.6) + honeypot_penalty
        reward = env_reward + base_shaping
        if (
            cmd_data
            and cmd_data.get("source") == "memory"
            and detect_redundancy is not None
            and detect_redundancy(self.command_history, command)
        ):
            reward *= self.redundancy_penalty
        self.stats_monitor.log_step(self.agent_id, reward, command=command)
        return reward

    def update_memory_and_teach(
        self, command, reward, context, parsed, state_tensor, next_state, done
    ):
        if self.memory_router and hasattr(self.memory_router, "inject_action"):
            self.memory_router.inject_action(
                self.agent_id, command, reward, context, parsed
            )
        self.teacher.add_action(
            command=command,
            description=parsed.get("description", "Auto-parsed action."),
            phase=parsed.get("phase", context.get("phase")),
            reward=int(reward),
            when=f"Ep{self.total_episodes + 1}-Step{self.total_steps}",
            why=self.last_reasoning,
        )
        experience = {
            "state": state_tensor,
            "action": self.select_action(state_tensor, context.get("phase")),
            "reward": reward,
            "next_state": safe_tensor(next_state, self.device),
            "done": done,
        }
        self.add_to_prioritized_memory(experience, abs(reward) + 1e-4)

    def add_to_prioritized_memory(self, experience, priority):
        if len(self.prioritized_experiences) >= self.replay_memory_size:
            removed = self.prioritized_experiences.pop(0)
            self.prioritized_priorities.pop(0)
            console.print(f"[dim]♻ Removed oldest experience: {removed}[/dim]")
        self.prioritized_experiences.append(experience)
        self.prioritized_priorities.append(priority)

    def train_on_batch(self):
        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            console.print(
                "[yellow]⚠ Not enough experiences for batch training.[/yellow]"
            )
            return
        states = torch.stack([exp["state"] for exp in batch]).to(self.device)
        actions = torch.tensor(
            [exp["action"] for exp in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [exp["reward"] for exp in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([exp["next_state"] for exp in batch]).to(self.device)
        dones = torch.tensor(
            [exp["done"] for exp in batch], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            target_values = self.value_net(next_states)[0].squeeze()
            targets = rewards + self.gamma * target_values * (1 - dones)
        self.policy_net.train_step(
            states, actions, targets, entropy_beta=self.entropy_beta
        )
        self.value_net.train_step(states, targets)
        console.print(f"[cyan]🔧 {self.agent_id}: Batch training complete.[/cyan]")
        console.print(
            f"[bold magenta]🧠 {self.agent_id}: Policy/Value networks updated.[/bold magenta]"
        )
        self._log_training_event("Batch training complete.")

    def _log_training_event(self, msg):
        with open(self.training_log_path, "a") as f:
            f.write(f"{msg}\n")

    def display_advanced_status(self):
        table = Table(title=f"🦂 {self.agent_id} Strategic Overview", show_lines=True)
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
            Panel(table, title="🧠 Strategic Overview", border_style="bright_blue")
        )
        self.stats_monitor.visualize_phase_distribution()
        console.print(f"[dim]Training log: {self.training_log_path}[/dim]")

    def log_episode_summary(self, total_reward, state):
        episode_summary_table = Table(
            title=f"Episode Summary - {self.agent_id}", show_lines=True
        )
        episode_summary_table.add_column("Metric", style="cyan")
        episode_summary_table.add_column("Value", style="magenta")
        episode_summary_table.add_row("Episode Reward", f"{total_reward:.2f}")
        episode_summary_table.add_row("Total Steps", str(self.total_steps))
        episode_summary_table.add_row("Mode Used", self.current_mode)
        episode_summary_table.add_row(
            "Blue Team Alert", f"{state.get('blue_team_alert', 0.0)::.2f}"
        )
        episode_summary_table.add_row(
            "Detection Rate", f"{self.stats_monitor.get_detection_rate():.2%}"
        )
        episode_summary_table.add_row(
            "Average Reward", f"{self.stats_monitor.get_average_reward():.2f}"
        )
        episode_summary_table.add_row(
            "Redundancy Rate", f"{self.stats_monitor.get_redundancy_rate():.2f}"
        )
        console.print(
            Panel(
                episode_summary_table,
                title="🎯 Mega Episode Summary",
                border_style="bright_green",
            )
        )
        self._log_training_event(
            f"Episode summary: reward={total_reward}, phase={state.get('phase')}"
        )

    def save_models(self, prefix="models/red_agent"):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        try:
            self.policy_net.save(f"{prefix}_policy.pt")
            self.value_net.save(f"{prefix}_value.pt")
            console.print(
                f"[green]💾 {self.agent_id}: Models saved successfully.[/green]"
            )
        except Exception as e:
            console.print(f"[red]❌ {self.agent_id}: Model save failed: {e}[/red]")

    def load_models(self, prefix="models/red_agent"):
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

    def encode_env_state(self, state):
        # Dummy implementation for demonstration
        return torch.zeros(512, device=self.device)

    def select_action(self, state_tensor, phase):
        # Use policy_net for action selection
        try:
            phase_vec = get_phase_vector(phase, self.device)
            action = self.policy_net.predict(state_tensor, phase_vector=phase_vec)
            return action
        except Exception as e:
            console.print(
                f"[yellow]⚠ PolicyNet failed: {e}, using random action[/yellow]"
            )
            return random.randint(0, self.policy_net.output_size - 1)

    def extract_output(self, command):
        # Dummy implementation for demonstration
        return "output"

    def end_episode(self, cumulative_reward, state):
        self.total_episodes += 1
        self.log_episode_summary(cumulative_reward, state)
        self.gpt_calls_this_episode = 0
        self.stagnation_window.clear()
        # --- Add: Print stats after each episode ---
        if hasattr(self.stats_monitor, "display_episode_summary"):
            self.stats_monitor.display_episode_summary()

    def generate_chain_snapshot(self):
        try:
            from core.logic.chainbuilder import build_and_store_chain

            build_and_store_chain(self.agent_id)
        except ImportError:
            pass

    def generate_hint(self):
        # New: Provide a tactical hint using memory or GPT
        if self.memory_router:
            mem = self.memory_router.get_memory(self.agent_id)
            if mem and mem.get("actions"):
                return mem["actions"][-1].get("full_command", "nmap -p- -sC -sV TARGET")
        return self.query_tactical_gpt(
            "Suggest a tactical command for the current phase."
        )

    def execute_command(self, command):
        try:
            output = self.extract_output(command)
            if not output or output == "output":
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

    def get_visualization_panel(self):
        """Return a rich Panel with detailed agent info for visualization."""
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
            table, title=f"🦂 {self.agent_id} Strategic Panel", border_style="red"
        )

    def reset(self):
        """Reset stats and replay buffer for new episode."""
        self.total_steps = 0
        self.command_history.clear()
        self.prioritized_experiences.clear()
        self.prioritized_priorities.clear()
        if hasattr(self, "stats_monitor"):
            self.stats_monitor.reset()

    def sync_memory(self):
        # Implementation for MemorySyncInterface
        if self.memory_router:
            self.memory_router.sync_global_insights()
        from core.logic.redundancy_detector import detect_redundancy_batch

        self.replay_buffer.prune_redundancy(lambda cmds: detect_redundancy_batch(cmds))


if __name__ == "__main__":
    # Import AgentManager here to avoid circular import at top-level
    from core.multiagent.agent_manager import AgentManager

    agent_manager = AgentManager()
    memory_router = (
        agent_manager.get_memory_router()
        if hasattr(agent_manager, "get_memory_router")
        else None
    )
    memory_manager = (
        agent_manager.get_memory_manager()
        if hasattr(agent_manager, "get_memory_manager")
        else None
    )
    agent = RedAgent(
        agent_manager=agent_manager,
        memory_router=memory_router,
        memory_manager=memory_manager,
    )
    agent.simulate_train(episodes=3)  # 3 episodes for test
    agent.display_advanced_status()  # Show live metrics
    agent.generate_chain_snapshot()  # Optional chain snapshot for strategy review
    agent.safe_shutdown()  # Ensures all models and memory are saved at the end of execution
